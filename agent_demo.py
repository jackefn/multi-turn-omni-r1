import os
import json
import base64
import argparse
from typing import Any, Dict, List, Optional, Tuple

import requests


DEFAULT_SYSTEM_PROMPT = """You are an agent for Audio/Video QA with a retrieval tool over video frames.
You must act in iterative steps and choose exactly ONE action per turn.

Allowed actions (return STRICT JSON only, no markdown):
1) search: {"action":"search","query":"<natural language query>"}
2) answer: {"action":"answer","answer":"<final answer as ONE English word from the options>"}

Rules:
- Max 20 turns total.
- Prefer search first, then decide answer after seeing retrieved frames.
- The user will provide: question, options, and retrieved frame images (as tool results).
- When you answer, you must choose exactly one of the provided options (one word).
- Output MUST be parseable JSON only.
"""


def file_to_data_url(path: str, mime_type: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def qwen_chat(
    api_base: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.2,
    max_tokens: int = 4096,
    timeout: int = 180,
) -> str:
    url = api_base.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"Qwen API error: {r.status_code} - {r.text}")
    data = r.json()
    return data["choices"][0]["message"]["content"]


def parse_action(text: str) -> Dict[str, Any]:
    t = text.strip()
    # 防御：有些模型会套 ```json ... ```
    if t.startswith("```"):
        t = t.strip("`").strip()
        if t.lower().startswith("json"):
            t = t[4:].strip()
    return json.loads(t)


def _mask_data_url(url: str) -> str:
    if not isinstance(url, str):
        return url
    if not url.startswith("data:") or "base64," not in url:
        return url
    meta, b64 = url.split("base64,", 1)
    return f"{meta}base64,<BASE64_OMITTED,len={len(b64)}>"


def _sanitize_obj(obj: Any) -> Any:
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            if k == "url" and isinstance(v, str):
                new_obj[k] = _mask_data_url(v)
            else:
                new_obj[k] = _sanitize_obj(v)
        return new_obj
    if isinstance(obj, list):
        return [_sanitize_obj(x) for x in obj]
    if isinstance(obj, str):
        return _mask_data_url(obj)
    return obj


def _pretty(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _print_trace(trace: Dict[str, Any]) -> None:
    print(_pretty(trace))
    print("-" * 80)


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
    """
    Supports:
    - .json: single object or list of objects
    - .jsonl: one json object per line
    """
    with open(path, "r", encoding="utf-8") as f:
        try:
            # JSON：可能是 dict 或 list
            obj = json.load(f)
            if isinstance(obj, list):
                return obj
            return [obj]
        except json.JSONDecodeError:
            # JSONL：多行，每行一个 dict
            f.seek(0)
            rows = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
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
        "Decide next action.\n"
        "Return JSON only.\n"
    )


def format_tool_result_text(results: List[Dict[str, Any]], prefix: str) -> Tuple[str, List[str]]:
    """
    Convert tool results into:
    - a readable text summary for the model
    - a list of existing frame image absolute paths to attach
    """
    lines = []
    img_paths: List[str] = []
    for item in results:
        md = item.get("metadata", {}) or {}
        src = md.get("source", "")
        typ = md.get("type", "")
        dist = item.get("distance", None)
        full = os.path.join(prefix, src) if src else ""
        lines.append(f"- rank={item.get('rank')} type={typ} dist={dist} source={src} full_path={full}")
        if typ == "frame" and full and os.path.exists(full):
            img_paths.append(full)

    return "RAG results:\n" + "\n".join(lines), img_paths


def make_multimodal_user_message(
    text: str, media: List[Tuple[str, str, Optional[str]]]
) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
    for item in media:
        if len(item) == 2:
            data_url, mime = item
            src_path = None
        else:
            data_url, mime, src_path = item
        block = {
            "type": "image_url",
            "image_url": {"url": data_url},
            "mime_type": mime,
        }
        if src_path:
            block["source_path"] = src_path
        content.append(block)
    return {"role": "user", "content": content}


def run_agent_for_sample(
    sample: Dict[str, Any],
    api_base: str,
    api_key: str,
    model: str,
    rag_url: str,
    img_prefix: str,
    top_k: int = 5,
    max_turns: int = 20,
    attach_images: int = 3,
    verbose: bool = False,
    trace_out: Optional[Any] = None,
) -> Dict[str, Any]:
    # -------- validate sample --------
    video_list = sample.get("videos") or []
    if not isinstance(video_list, list) or len(video_list) == 0:
        return {"id": sample.get("id"), "error": "no_video_path_in_sample"}

    video_path = video_list[0]
    if not os.path.exists(video_path):
        return {"id": sample.get("id"), "error": f"video_not_found: {video_path}"}

    options = sample["reward_model_ground_truth"]["multi_choice"]

    # -------- initial messages --------
    video_data_url = file_to_data_url(video_path, "video/mp4")

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        make_multimodal_user_message(
            text=build_user_text(sample),
            media=[(video_data_url, "video/mp4", video_path)],
        ),
    ]

    last_search_query: str = ""
    all_searches: List[str] = []

    for turn in range(1, max_turns + 1):
        request_messages = _sanitize_obj(messages)
        raw = qwen_chat(
            api_base=api_base,
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=4096,
        )
        messages.append({"role": "assistant", "content": raw})
        if verbose:
            print(f"\n[TURN {turn}] model_raw:\n{raw}\n")

        # -------- parse action --------
        try:
            action = parse_action(raw)
            parsed_action = action
        except Exception as e:
            parsed_action = {"error": str(e)}
            messages.append(
                {"role": "user", "content": f"Invalid JSON. Error={e}. Return STRICT JSON only."}
            )
            trace = {
                "turn_id": turn,
                "request_messages": request_messages,
                "model_raw": raw,
                "parsed_action": parsed_action,
                "tool_call": None,
                "attached_media": [],
                "messages_after": _sanitize_obj(messages),
            }
            _print_trace(trace)
            if trace_out:
                trace_out.write(json.dumps(trace, ensure_ascii=False) + "\n")
                trace_out.flush()
            continue

        act = (action.get("action") or "").strip().lower()

        if act == "search":
            query = (action.get("query") or "").strip()
            if not query:
                messages.append({"role": "user", "content": "Empty query. Return JSON with non-empty query."})
                trace = {
                    "turn_id": turn,
                    "request_messages": request_messages,
                    "model_raw": raw,
                    "parsed_action": parsed_action,
                    "tool_call": None,
                    "attached_media": [],
                    "messages_after": _sanitize_obj(messages),
                }
                _print_trace(trace)
                if trace_out:
                    trace_out.write(json.dumps(trace, ensure_ascii=False) + "\n")
                    trace_out.flush()
                continue

            last_search_query = query
            all_searches.append(query)

            # -------- call tool --------
            results = rag_search(rag_url, query=query, top_k=top_k)
            tool_text, img_paths = format_tool_result_text(results, prefix=img_prefix)

            # -------- attach top frames (as images) --------
            media: List[Tuple[str, str, Optional[str]]] = []
            attached_media: List[Dict[str, str]] = []
            for p in img_paths[:attach_images]:
                ext = os.path.splitext(p)[1].lower()
                mime = "image/jpeg"
                if ext == ".png":
                    mime = "image/png"
                elif ext == ".webp":
                    mime = "image/webp"
                try:
                    media.append((file_to_data_url(p, mime), mime, p))
                    attached_media.append({"path": p, "mime": mime})
                except Exception:
                    pass

            msg_text = (
                "TOOL(search) executed.\n"
                f"Query: {query}\n\n"
                f"{tool_text}\n\n"
                "Retrieved frames are attached. Decide next action.\n"
                "Return JSON only."
            )
            messages.append(make_multimodal_user_message(text=msg_text, media=media))
            trace = {
                "turn_id": turn,
                "request_messages": request_messages,
                "model_raw": raw,
                "parsed_action": parsed_action,
                "tool_call": {
                    "query": query,
                    "top_k": top_k,
                    "rag_url": rag_url,
                    "results": results,
                },
                "attached_media": attached_media,
                "messages_after": _sanitize_obj(messages),
            }
            _print_trace(trace)
            if trace_out:
                trace_out.write(json.dumps(trace, ensure_ascii=False) + "\n")
                trace_out.flush()
            continue

        if act == "answer":
            ans = (action.get("answer") or "").strip()
            if ans not in options:
                messages.append(
                    {"role": "user", "content": f"Invalid answer '{ans}'. Choose exactly one of: {options}. Return JSON only."}
                )
                trace = {
                    "turn_id": turn,
                    "request_messages": request_messages,
                    "model_raw": raw,
                    "parsed_action": parsed_action,
                    "tool_call": None,
                    "attached_media": [],
                    "messages_after": _sanitize_obj(messages),
                }
                _print_trace(trace)
                if trace_out:
                    trace_out.write(json.dumps(trace, ensure_ascii=False) + "\n")
                    trace_out.flush()
                continue

            # -------- produce final format you want --------
            think = "I used retrieval over video frames to identify the relevant content and selected the matching option."
            final = f"<think>{think}</think>\n<search>{last_search_query}</search>\n<answer>{ans}</answer>"

            trace = {
                "turn_id": turn,
                "request_messages": request_messages,
                "model_raw": raw,
                "parsed_action": parsed_action,
                "tool_call": None,
                "attached_media": [],
                "messages_after": _sanitize_obj(messages),
            }
            _print_trace(trace)
            if trace_out:
                trace_out.write(json.dumps(trace, ensure_ascii=False) + "\n")
                trace_out.flush()

            return {
                "id": sample.get("id"),
                "video": sample.get("videos"),
                "question": sample["extra_info"]["question"],
                "options": options,
                "agent_turns": turn,
                "searches": all_searches,
                "prediction": ans,
                "final_output": final,
            }

        # unknown action
        messages.append({"role": "user", "content": "Unknown action. Use only 'search' or 'answer' in JSON."})
        trace = {
            "turn_id": turn,
            "request_messages": request_messages,
            "model_raw": raw,
            "parsed_action": parsed_action,
            "tool_call": None,
            "attached_media": [],
            "messages_after": _sanitize_obj(messages),
        }
        _print_trace(trace)
        if trace_out:
            trace_out.write(json.dumps(trace, ensure_ascii=False) + "\n")
            trace_out.flush()

    return {"id": sample.get("id"), "error": "max_turns_reached", "searches": all_searches}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="demo.json", help="input demo.json or demo.jsonl")
    ap.add_argument("--out", default="outputs.jsonl", help="output jsonl")

    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=5)

    ap.add_argument("--rag-url", default="http://127.0.0.1:8000", help="RAG service base url")
    ap.add_argument(
        "--img-prefix",
        default="/mnt/hpfs/xiangc/mxy/multi-turn-omni-r1/build_rag_database",
        help="prefix for metadata.source",
    )
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--attach-images", type=int, default=3)

    ap.add_argument("--api-base", default="https://api.apiyi.com/v1")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--api-key-env", default="QWEN_API_KEY")

    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--trace-out", default="", help="optional jsonl file to write per-turn traces")
    args = ap.parse_args()

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"Missing API key in env var {args.api_key_env}")

    rows = load_json_rows(args.json)
    print(f"[INFO] Loaded {len(rows)} rows from: {args.json}")

    sub = rows[args.start : args.start + args.limit]

    trace_f = open(args.trace_out, "w", encoding="utf-8") if args.trace_out else None
    try:
        with open(args.out, "w", encoding="utf-8") as f:
            for i, sample in enumerate(sub):
                print(f"\n========== SAMPLE {args.start + i} | id={sample.get('id')} ==========")
                result = run_agent_for_sample(
                    sample=sample,
                    api_base=args.api_base,
                    api_key=api_key,
                    model=args.model,
                    rag_url=args.rag_url,
                    img_prefix=args.img_prefix,
                    top_k=args.top_k,
                    max_turns=args.max_turns,
                    attach_images=args.attach_images,
                    verbose=args.verbose,
                    trace_out=trace_f,
                )
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
    finally:
        if trace_f:
            trace_f.close()

    print(f"\nDone. Wrote: {args.out}")


if __name__ == "__main__":
    main()
