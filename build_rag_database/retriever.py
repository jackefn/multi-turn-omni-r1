from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Any
import numpy as np

from embeddings import embed_text, load_index, load_metadata

app = FastAPI(title="Video RAG API", description="RAG over video frames + ASR", version="1.0")

INDEX_PATH = "rag_db/video_index.faiss"
META_PATH = "rag_db/video_metadata.jsonl"

def search(index_path, query_vector, top_k=5):
    index = load_index(index_path)
    query = np.array([query_vector]).astype("float32")
    D, I = index.search(query, top_k)
    return I[0], D[0]

def retrieve_results(meta_path):
    return load_metadata(meta_path)

def aggregate_by_segment(metas, ids, dists, want_k):
    best = {}
    for i, meta_id in enumerate(ids):
        if meta_id < 0 or meta_id >= len(metas):
            continue
        meta = metas[meta_id]
        seg_id = meta.get("segment_id")
        if not seg_id:
            continue
        dist = float(dists[i])
        prev = best.get(seg_id)
        if prev is None or dist < prev["score"]:
            best[seg_id] = {
                "score": dist,
                "segment": {
                    "segment_id": seg_id,
                    "t0": meta.get("t0"),
                    "t1": meta.get("t1"),
                    "frame": meta.get("frame"),
                    "audio": meta.get("audio"),
                },
            }

    ranked = sorted(best.values(), key=lambda x: x["score"])
    return ranked[:want_k]

def query_rag(query: str, top_k: int = 5):
    v = embed_text(query)  # CPU embedding
    search_k = max(50, top_k * 10)
    ids, dist = search(INDEX_PATH, v, top_k=search_k)
    metas = retrieve_results(META_PATH)
    results = aggregate_by_segment(metas, ids, dist, want_k=top_k)
    print(f"Query: {query}")
    print(f"Results: {results}")
    return [
        {
            "rank": idx,
            "score": results[idx]["score"],
            "segment": results[idx]["segment"],
        }
        for idx in range(len(results))
    ]

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class QueryResponse(BaseModel):
    rank: int
    score: float
    segment: Any

@app.get("/")
def root():
    return {"message": "Video RAG API is running!"}


@app.post("/query", response_model=List[QueryResponse])
def rag_query(req: QueryRequest):
    return query_rag(req.query, top_k=req.top_k)


@app.get("/query", response_model=List[QueryResponse])
def rag_query_get(query: str = Query(...), top_k: int = 5):
    return query_rag(query, top_k=top_k)
