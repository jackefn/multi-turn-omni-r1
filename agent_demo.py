import os
import re
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor


DEFAULT_SYSTEM_PROMPT = """You are an agent for Audio/Video QA with a retrieval tool over video frames and audio clips.
You must act in iterative steps and choose exactly ONE action per turn.

Output format rules:
- Every turn, you MUST output exactly TWO XML-like tags in this order:
  1) <think>...</think>
  2) Either <search>...</search> OR <answer>...</answer>
- The <think> tag is for reasoning.
- The <search> tag must contain a natural language search query ONLY.
- The <answer> tag must contain EXACTLY ONE option letter/word from the provided options ONLY.
- Do NOT output any JSON, markdown, code fences, or extra text outside the tags.

Allowed actions:
1) <search>your query</search>
2) <answer>ONE_OPTION</answer>

Rules:
- Max 20 turns total.
- Prefer <search> first, then decide <answer> after seeing retrieved segments (image+audio).
- The user will provide: question, options, and videos.
- When you answer, you must choose exactly one of the provided options (one word/letter).
"""


def parse_action(text: str) -> Dict[str, Any]:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
        if t.lower().startswith("json"):
            t = t[4:].strip()

    think_tags = re.findall(r"<think>(.*?)</think>", t, flags=re.DOTALL | re.IGNORECASE)
    search_tags = re.findall(r"<search>(.*?)</search>", t, flags=re.DOTALL | re.IGNORECASE)
    answer_tags = re.findall(r"<answer>(.*?)</answer>", t, flags=re.DOTALL | re.IGNORECASE)

    if think_tags or search_tags or answer_tags:
        if len(think_tags) != 1:
            return {"action": "invalid", "format_ok": False, "error": "missing_or_multiple_think"}
        if answer_tags:
            return {
                "action": "answer",
                "answer": answer_tags[0].strip(),
                "think": think_tags[0].strip(),
                "format_ok": True,
            }
        if len(search_tags) != 1:
            return {"action": "invalid", "format_ok": False, "error": "missing_or_multiple_action_tags"}
        if search_tags:
            return {
                "action": "search",
                "query": search_tags[0].strip(),
                "think": think_tags[0].strip(),
                "format_ok": True,
            }
        return {"action": "invalid", "format_ok": False, "error": "missing_action_tags"}

    try:
        obj = json.loads(t)
    except Exception:
        return {"action": "invalid", "format_ok": False, "error": "unparseable_output"}

    if isinstance(obj, dict):
        obj["format_ok"] = False
        obj.setdefault("error", "json_fallback")
        return obj
    return {"action": "invalid", "format_ok": False, "error": "json_not_object"}


def rag_search(rag_url: str, query: str, top_k: int) -> List[Dict[str, Any]]:
    r = requests.post(
        rag_url.rstrip("/") + "/query",
        headers={"Content-Type": "application/json"},
        json={"query": query, "top_k": top_k},
        timeout=60,
    )
    if r.status_code != 200:
        raise RuntimeError(f"RAG error: {r.status_code} - {r.text}")
    return r.json()


def load_json_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        try:
            obj = json.load(f)
            return obj if isinstance(obj, list) else [obj]
        except json.JSONDecodeError:
            f.seek(0)
            rows = []
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
            return rows


def build_user_text(sample: Dict[str, Any]) -> str:
    q = sample["extra_info"]["question"]
    choices = sample["reward_model_ground_truth"]["multi_choice"]
    opt_text = "\n".join([f"{i}. {c}" for i, c in enumerate(choices)])
    return (
        "You are doing Audio/Video QA.\n"
        f"Question: {q}\n"
        f"Options:\n{opt_text}\n\n"
        "Output format:\n"
        "- First turn MUST use <search> (do NOT answer).\n"
        "- Every turn must output exactly:\n"
        "  <think>...</think>\n"
        "  and either <search>...</search> or <answer>...</answer>\n"
        "- Do not output JSON or any text outside the tags.\n"
    )


def make_qwen_mm_user_message(
    text: str,
    image_paths: Optional[List[str]] = None,
    audio_paths: Optional[List[str]] = None,
    video_paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Qwen2.5-Omni Transformers expects:
      {"type":"image","path":...}, {"type":"audio","path":...}, {"type":"video","path":...}, {"type":"text","text":...}
    """
    content: List[Dict[str, Any]] = []
    for p in (image_paths or []):
        content.append({"type": "image", "path": p})
    for p in (video_paths or []):
        content.append({"type": "video", "path": p})
    for p in (audio_paths or []):
        content.append({"type": "audio", "path": p})

    content.append({"type": "text", "text": text})
    return {"role": "user", "content": content}


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"...[truncated {len(text) - max_chars} chars]"


def _inputs_meta(inputs: Dict[str, Any]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            meta[k] = {
                "shape": list(v.shape),
                "dtype": str(v.dtype),
                "device": str(v.device),
            }
        else:
            meta[k] = {"type": type(v).__name__}
    return meta


def _print_inputs_overview(inputs: Dict[str, Any]) -> None:
    for k, v in inputs.items():
        if torch.is_tensor(v):
            print(
                f"- {k}: shape={list(v.shape)} dtype={v.dtype} device={v.device}"
            )
        else:
            print(f"- {k}: type={type(v).__name__}")


def _messages_have_video(messages: List[Dict[str, Any]]) -> bool:
    for msg in messages:
        for block in msg.get("content", []) or []:
            if block.get("type") == "video":
                return True
    return False


def _tags_only_reminder(reason: str, *, force_search: bool = False) -> str:
    base = (
        "Format error: reply with EXACTLY two tags only:\n"
        "<think>...</think>\n"
        "and either <search>...</search> or <answer>...</answer>.\n"
        "No JSON, markdown, or extra text."
    )
    if force_search:
        return base + f"\nReason: {reason}\nYou MUST output <search> on this turn."
    return base + f"\nReason: {reason}"


def _extract_answer_tag(text: str) -> Optional[str]:
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"<answer>(.*?)</anawer>", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


@torch.inference_mode()
def qwen_local_chat(
    model: Qwen2_5OmniForConditionalGeneration,
    processor: Qwen2_5OmniProcessor,
    messages: List[Dict[str, Any]],
    *,
    temperature: float = 0.2,
    max_new_tokens: int = 512,
    fps: int = 1,
    use_audio_in_video: bool = True,
    load_audio_from_video: bool = True,
    debug_print: bool = False,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Use processor.apply_chat_template to build tensors, then model.generate.
    """
    def _build_prompt_and_inputs(
        *, use_audio: bool, load_audio: bool
    ) -> Tuple[str, Dict[str, Any]]:
        prompt = processor.apply_chat_template(
            [messages],
            add_generation_prompt=True,
            tokenize=False,
            fps=fps,
            use_audio_in_video=use_audio,
            load_audio_from_video=load_audio,
        )
        if isinstance(prompt, list):
            prompt = prompt[0]

        inputs_local = processor.apply_chat_template(
            [messages],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            fps=fps,
            use_audio_in_video=use_audio,
            load_audio_from_video=load_audio,
        )
        return prompt, inputs_local

    has_video = _messages_have_video(messages)
    fallback = False
    try:
        prompt_text, inputs = _build_prompt_and_inputs(
            use_audio=use_audio_in_video,
            load_audio=load_audio_from_video,
        )
    except StopIteration:
        if has_video and (use_audio_in_video or load_audio_from_video):
            fallback = True
            use_audio_in_video = False
            load_audio_from_video = False
            prompt_text, inputs = _build_prompt_and_inputs(
                use_audio=use_audio_in_video,
                load_audio=load_audio_from_video,
            )
        else:
            raise

    if has_video and (use_audio_in_video or load_audio_from_video):
        audio_lengths = inputs.get("audio_lengths")
        if (
            (torch.is_tensor(audio_lengths) and audio_lengths.numel() == 0)
            or (isinstance(audio_lengths, (list, tuple)) and len(audio_lengths) == 0)
        ):
            fallback = True
            use_audio_in_video = False
            load_audio_from_video = False
            prompt_text, inputs = _build_prompt_and_inputs(
                use_audio=use_audio_in_video,
                load_audio=load_audio_from_video,
            )

    if fallback and debug_print:
        print("audio_from_video fallback to False")

    # Qwen2.5-Omni 的 model 里通常 thinker 扛 LLM device
    device = getattr(getattr(model, "thinker", None), "device", None) or next(model.parameters()).device
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature,
        use_audio_in_video=use_audio_in_video,
    )

    out = model.generate(**inputs, **gen_kwargs)

    # 通常 generate 会把 prompt 拼在前面，切一下更稳
    in_len = inputs["input_ids"].shape[1]
    new_tokens = out[:, in_len:] if out.shape[1] >= in_len else out
    text = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return text.strip(), prompt_text, inputs


def format_retrieved_segments(
    results: List[Dict[str, Any]],
    prefix: str,
    attach_k: int,
) -> Tuple[str, List[str], List[str]]:
    """
    results: [{"rank":..,"score":..,"segment":{"t0":..,"t1":..,"frame":"frames/...jpg","audio":"audio/...wav"}}]
    """
    lines = []
    imgs: List[str] = []
    audios: List[str] = []

    for item in results[:attach_k]:
        seg = item.get("segment") or {}
        frame_rel = seg.get("frame", "")
        audio_rel = seg.get("audio", "")
        t0, t1 = seg.get("t0"), seg.get("t1")

        frame_abs = os.path.join(prefix, frame_rel) if frame_rel else ""
        audio_abs = os.path.join(prefix, audio_rel) if audio_rel else ""

        lines.append(
            f"- rank={item.get('rank')} score={item.get('score'):.6f} "
            f"t0={t0} t1={t1} frame={frame_rel} audio={audio_rel}"
        )

        if frame_abs and os.path.exists(frame_abs):
            imgs.append(frame_abs)
        if audio_abs and os.path.exists(audio_abs):
            audios.append(audio_abs)

    return "RAG results:\n" + "\n".join(lines), imgs, audios


def run_agent_for_sample(
    sample: Dict[str, Any],
    model,
    processor,
    rag_url: str,
    media_prefix: str,
    *,
    top_k: int = 5,
    max_turns: int = 20,
    attach_segments: int = 3,
    include_video_in_first_turn: bool = True,
    verbose: bool = False,
    debug_print: bool = False,
    debug_dump_dir: str = "",
    debug_max_chars: int = 0,
) -> Dict[str, Any]:
    options = sample["reward_model_ground_truth"]["multi_choice"]

    # 可选：第一轮是否把原视频也塞进去（通常不建议，太吃显存/时间）
    video_paths = []
    if include_video_in_first_turn:
        video_list = sample.get("videos") or []
        if video_list:
            video_path = video_list[0]
            if os.path.exists(video_path):
                video_paths = [video_path]

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": DEFAULT_SYSTEM_PROMPT}]},
        make_qwen_mm_user_message(
            text=build_user_text(sample),
            video_paths=video_paths,
        ),
    ]

    last_search_query = ""
    all_searches: List[str] = []

    for turn in range(1, max_turns + 1):
        raw, prompt_text, inputs = qwen_local_chat(
            model=model,
            processor=processor,
            messages=messages,
            temperature=0.2,
            max_new_tokens=512,
            fps=1,
            use_audio_in_video=True,
            load_audio_from_video=True,
            debug_print=debug_print,
        )

        if debug_print:
            print(f"\n[TURN {turn}] messages:")
            print(json.dumps(messages, ensure_ascii=False, indent=2))
            print(f"\n[TURN {turn}] rendered_prompt_text:")
            print(_truncate_text(prompt_text, debug_max_chars))
            print(f"\n[TURN {turn}] inputs:")
            _print_inputs_overview(inputs)
            print(f"\n[TURN {turn}] model_raw:")
            print(_truncate_text(raw, debug_max_chars))

        if debug_dump_dir:
            os.makedirs(debug_dump_dir, exist_ok=True)
            dump_path = os.path.join(debug_dump_dir, f"turn_{turn:03d}.json")
            with open(dump_path, "w", encoding="utf-8") as df:
                json.dump(
                    {
                        "turn_id": turn,
                        "messages": messages,
                        "rendered_prompt_text": prompt_text,
                        "model_raw": raw,
                        "inputs_meta": _inputs_meta(inputs),
                    },
                    df,
                    ensure_ascii=False,
                    indent=2,
                )
            torch.save(inputs, os.path.join(debug_dump_dir, f"turn_{turn:03d}_inputs.pt"))

        messages.append({"role": "assistant", "content": [{"type": "text", "text": raw}]})

        if verbose:
            print(f"\n[TURN {turn}] model_raw:\n{raw}\n")

        answer_from_tag = _extract_answer_tag(raw)
        if answer_from_tag is not None:
            return {
                "id": sample.get("id"),
                "question": sample["extra_info"]["question"],
                "options": options,
                "agent_turns": turn,
                "searches": all_searches,
                "prediction": answer_from_tag,
                "final_output": answer_from_tag,
                "last_search": last_search_query,
            }

        action = parse_action(raw)
        if action.get("action") == "answer":
            ans = (action.get("answer") or "").strip()
            return {
                "id": sample.get("id"),
                "question": sample["extra_info"]["question"],
                "options": options,
                "agent_turns": turn,
                "searches": all_searches,
                "prediction": ans,
                "final_output": ans,
                "last_search": last_search_query,
            }
        if not action.get("format_ok", False):
            reason = action.get("error", "invalid_output")
            if action.get("action") in ("search", "answer"):
                reason = f"{reason}; returned {action.get('action')} without required tags"
            messages.append(
                make_qwen_mm_user_message(
                    text=_tags_only_reminder(reason)
                )
            )
            continue
        if action.get("action") == "invalid":
            messages.append(
                make_qwen_mm_user_message(
                    text=_tags_only_reminder(action.get("error", "invalid_output"))
                )
            )
            continue

        act = (action.get("action") or "").strip().lower()

        if act == "search":
            query = (action.get("query") or "").strip()
            if not query:
                messages.append(
                    make_qwen_mm_user_message(
                        text=_tags_only_reminder("empty_search_query")
                    )
                )
                continue

            last_search_query = query
            all_searches.append(query)

            results = rag_search(rag_url, query=query, top_k=top_k)
            tool_text, img_paths, audio_paths = format_retrieved_segments(
                results=results,
                prefix=media_prefix,
                attach_k=attach_segments,
            )

            msg_text = (
                "TOOL(search) executed.\n"
                f"Query: {query}\n\n"
                f"{tool_text}\n\n"
                "Retrieved segments (images+audio) are attached.\n"
                "Now decide next action using the required tags only."
            )

            # 把图片+音频一起塞回上下文
            messages.append(
                make_qwen_mm_user_message(
                    text=msg_text,
                    image_paths=img_paths,
                    audio_paths=audio_paths,
                )
            )
            continue

        messages.append(
            make_qwen_mm_user_message(
                text=_tags_only_reminder(f"unknown_action '{act}'")
            )
        )

    return {"id": sample.get("id"), "error": "max_turns_reached", "searches": all_searches}


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--json", default="demo.json", help="input demo.json or demo.jsonl")
    ap.add_argument("--out", default="outputs.jsonl", help="output jsonl")

    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=5)

    ap.add_argument("--rag-url", default="http://127.0.0.1:8000", help="RAG service base url")
    ap.add_argument("--media-prefix", required=True, help="prefix for frame/audio relative paths in retriever output")

    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--attach-segments", type=int, default=3)

    ap.add_argument("--model-path", required=True, help="local path to Qwen2.5-Omni model")
    ap.add_argument("--flash-attn2", action="store_true", help="use flash_attention_2 if available")
    ap.add_argument("--enable-audio-output", action="store_true", help="enable audio output (costs more VRAM)")

    ap.add_argument("--include-video", action="store_true", help="include the original video in the first turn")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--debug-print", action="store_true", help="print full messages/prompt/inputs/model output each turn")
    ap.add_argument("--debug-dump-dir", default="", help="if set, dump turn JSON and inputs tensors to this dir")
    ap.add_argument("--debug-max-chars", type=int, default=0, help="truncate printed prompt/model text if >0")
    args = ap.parse_args()

    attn_impl = "flash_attention_2" if args.flash_attn2 else None
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation=attn_impl,
        enable_audio_output=args.enable_audio_output,
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)

    rows = load_json_rows(args.json)
    sub = rows[args.start : args.start + args.limit]

    with open(args.out, "w", encoding="utf-8") as f:
        for i, sample in enumerate(sub):
            print(f"\n========== SAMPLE {args.start + i} | id={sample.get('id')} ==========")
            result = run_agent_for_sample(
                sample=sample,
                model=model,
                processor=processor,
                rag_url=args.rag_url,
                media_prefix=args.media_prefix,
                top_k=args.top_k,
                max_turns=args.max_turns,
                attach_segments=args.attach_segments,
                include_video_in_first_turn=args.include_video,
                verbose=args.verbose,
                debug_print=args.debug_print,
                debug_dump_dir=args.debug_dump_dir,
                debug_max_chars=args.debug_max_chars,
            )
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

    print(f"\nDone. Wrote: {args.out}")


if __name__ == "__main__":
    main()
