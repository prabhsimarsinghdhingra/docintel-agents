"""MCP server exposing the retrieval tools. Run: python -m docintel.mcp_server
Connect from Claude Desktop / any MCP client over stdio."""
from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

from .embeddings import get_embedder
from .index import VectorIndex
from .tools import Toolbox

mcp = FastMCP("docintel")
_tb: Toolbox | None = None


def tb() -> Toolbox:
    global _tb
    if _tb is None:
        _tb = Toolbox(VectorIndex.load(os.getenv("INDEX_DIR", "data/index"), get_embedder()))
    return _tb


@mcp.tool()
def search_documents(query: str, k: int = 5) -> list[dict]:
    """Semantic search over the indexed PDFs. Returns chunks with doc, page, score."""
    return tb().search_documents(query, k)


@mcp.tool()
def get_chunk(chunk_id: int) -> dict:
    """Fetch one chunk by id."""
    return tb().get_chunk(chunk_id)


@mcp.tool()
def list_documents() -> list[str]:
    """List indexed document names."""
    return tb().list_documents()


if __name__ == "__main__":
    mcp.run()
