"""FAISS vector index (inner product over L2-normalised vectors = cosine)."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from .ingest import Chunk


class VectorIndex:
    def __init__(self, embedder, chunks: list[Chunk] | None = None, index=None):
        self.embedder = embedder
        self.chunks = chunks or []
        self.index = index

    def build(self, chunks: list[Chunk]) -> "VectorIndex":
        self.chunks = chunks
        vecs = self.embedder.encode([c.text for c in chunks])
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)
        return self

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        q = self.embedder.encode([query])
        scores, ids = self.index.search(q, min(k, len(self.chunks)))
        return [(self.chunks[i], float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0]

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(d / "faiss.index"))
        (d / "chunks.json").write_text(json.dumps([asdict(c) for c in self.chunks]))
        (d / "meta.json").write_text(json.dumps({"embedder": self.embedder.name}))

    @classmethod
    def load(cls, directory: str | Path, embedder) -> "VectorIndex":
        d = Path(directory)
        chunks = [Chunk(**c) for c in json.loads((d / "chunks.json").read_text())]
        meta = json.loads((d / "meta.json").read_text())
        if meta["embedder"] != embedder.name:
            raise ValueError(f"Index built with {meta['embedder']}, loaded with {embedder.name}")
        return cls(embedder, chunks, faiss.read_index(str(d / "faiss.index")))
