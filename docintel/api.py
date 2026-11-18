"""FastAPI service. Also serves the dashboard at / when running in Docker."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agents import Pipeline

app = FastAPI(title="DocIntel Agents", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

PIPELINE: Pipeline | None = None


def get_pipeline() -> Pipeline:
    global PIPELINE
    if PIPELINE is None:
        from .embeddings import get_embedder
        from .index import VectorIndex

        d = Path(os.getenv("INDEX_DIR", "data/index"))
        if not (d / "faiss.index").exists():
            raise HTTPException(503, "Index not built. Run: python -m docintel.build")
        PIPELINE = Pipeline(VectorIndex.load(d, get_embedder()))
    return PIPELINE


class Query(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


@app.get("/health")
def health():
    p = get_pipeline()
    return {"status": "ok", "chunks": len(p.index.chunks),
            "mode": "llm" if p.llm.available else "extractive"}


@app.get("/documents")
def documents():
    return sorted({c.doc for c in get_pipeline().index.chunks})


@app.post("/query")
def query(q: Query):
    return get_pipeline().ask(q.question)


if Path("dashboard").exists():
    app.mount("/", StaticFiles(directory="dashboard", html=True), name="dashboard")
