from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Any
import numpy as np

from build_rag_database.embeddings import embed_text, load_index, load_metadata

app = FastAPI(title="Video RAG API", description="RAG over video frames + ASR", version="1.0")

INDEX_PATH = "rag_db/video_index.faiss"
META_PATH = "rag_db/video_metadata.jsonl"

def search(index_path, query_vector, top_k=5):
    index = load_index(index_path)
    query = np.array([query_vector]).astype("float32")
    D, I = index.search(query, top_k)
    return I[0], D[0]

def retrieve_results(meta_path, ids):
    metas = load_metadata(meta_path)
    return [metas[i] for i in ids]

def query_rag(query: str, top_k: int = 5):
    v = embed_text(query)  # CPU embedding
    ids, dist = search(INDEX_PATH, v, top_k=top_k)
    results = retrieve_results(META_PATH, ids)
    return [
        {
            "rank": idx,
            "distance": float(dist[idx]),
            "metadata": results[idx]
        }
        for idx in range(len(results))
    ]

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class QueryResponse(BaseModel):
    rank: int
    distance: float
    metadata: Any

@app.get("/")
def root():
    return {"message": "Video RAG API is running!"}


@app.post("/query", response_model=List[QueryResponse])
def rag_query(req: QueryRequest):
    return query_rag(req.query, top_k=req.top_k)


@app.get("/query", response_model=List[QueryResponse])
def rag_query_get(query: str = Query(...), top_k: int = 5):
    return query_rag(query, top_k=top_k)
