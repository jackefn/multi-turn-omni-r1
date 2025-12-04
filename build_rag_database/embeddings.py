import clip
import torch
from PIL import Image
import faiss
import numpy as np
import json
from build_rag_database.video_ingest import audio_to_text, extract_frames, video_to_audio
import os
import re

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def embed_text(text: str):
    tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        vec = model.encode_text(tokens)
    return vec[0].cpu().numpy()

def embed_image(image_path: str):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = model.encode_image(image)
    return vec[0].cpu().numpy()

# emb_text = embed_text("A photo of a cat")
# print(emb_text)
# emb_image = embed_image("frames/frame_000005.jpg")
# print(emb_image)

def create_index(dim, index_path):
    index = faiss.IndexFlatL2(dim)
    faiss.write_index(index, index_path)

def load_index(index_path):
    return faiss.read_index(index_path)

def add_to_index(index_path, vector):
    index = load_index(index_path)
    index.add(np.array([vector]).astype("float32"))
    faiss.write_index(index, index_path)

def add_metadata(meta_path, item):
    with open(meta_path, "a") as f:
        f.write(json.dumps(item) + "\n")

def load_metadata(meta_path):
    metas = []
    with open(meta_path) as f:
        for line in f:
            metas.append(json.loads(line))
    return metas

def split_asr_text(text, max_tokens=70):
    sentences = re.split(r"[.!?。！？]", text)

    clean_sentences = []
    for s in sentences:
        s = s.strip()
        if len(s) == 0:
            continue
        clean_sentences.append(s)

    return clean_sentences

def ingest_video(video_path, frame_dir, audio_path, index_path, meta_path):
    extract_frames(video_path, frame_dir, fps=1)
    video_to_audio(video_path, audio_path)
    asr_text = audio_to_text(audio_path)
    sentences = split_asr_text(asr_text)
    for idx, sent in enumerate(sentences):
        text_vec = embed_text(sent)
        add_to_index(index_path, text_vec)

        add_metadata(meta_path, {
            "type": "asr_sentence",
            "content": sent,
            "source": audio_path,
            "sentence_id": idx
        })

    for f in sorted(os.listdir(frame_dir)):
        img_path = os.path.join(frame_dir, f)
        img_vec = embed_image(img_path)
        add_to_index(index_path, img_vec)

        add_metadata(meta_path, {
            "type": "frame",
            "content": "",
            "source": img_path,
        })

    print(f"Ingest finished: {len(sentences)} ASR sentences + {len(os.listdir(frame_dir))} frames processed.")

if not os.path.exists("rag_db"):
    os.makedirs("rag_db")

# create_index(512, "rag_db/video_index.faiss")
# ingest_video(video_path="ZXoaMa6jlO4.mp4",
#              frame_dir="frames",
#              audio_path="output_audio.wav",
#              index_path="rag_db/video_index.faiss",
#              meta_path="rag_db/video_metadata.jsonl")

