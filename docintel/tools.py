"""Tool layer shared by the agents (in-process) and the MCP server (over stdio)."""
from __future__ import annotations

from dataclasses import asdict

from .index import VectorIndex


class Toolbox:
    def __init__(self, index: VectorIndex):
        self.index = index

    def search_documents(self, query: str, k: int = 5) -> list[dict]:
        return [{"chunk_id": c.id, "doc": c.doc, "page": c.page, "text": c.text,
                 "score": round(s, 4)} for c, s in self.index.search(query, k)]

    def get_chunk(self, chunk_id: int) -> dict:
        return asdict(self.index.chunks[chunk_id])

    def list_documents(self) -> list[str]:
        return sorted({c.doc for c in self.index.chunks})
