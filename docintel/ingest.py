"""PDF ingestion: text extraction, cleaning and sentence-aware chunking."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    id: int
    doc: str
    page: int
    text: str


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\[])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def clean(text: str) -> str:
    text = re.sub(r"-\n(\w)", r"\1", text)  # de-hyphenate line breaks
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    """Greedy sentence packing with character overlap between chunks."""
    chunks: list[str] = []
    cur = ""
    for s in split_sentences(text):
        pieces = [s[i:i + size] for i in range(0, len(s), size)] if len(s) > size else [s]
        for p in pieces:
            if cur and len(cur) + len(p) + 1 > size:
                chunks.append(cur.strip())
                cur = cur[-overlap:] + " " + p
            else:
                cur = (cur + " " + p).strip()
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def make_chunks(doc: str, pages: list[tuple[int, str]], start_id: int = 0) -> list[Chunk]:
    out: list[Chunk] = []
    nid = start_id
    for page, text in pages:
        for c in chunk_text(text):
            if len(c) >= 60:
                out.append(Chunk(nid, doc, page, c))
                nid += 1
    return out


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    from pypdf import PdfReader

    pages = []
    for i, p in enumerate(PdfReader(str(pdf_path)).pages, 1):
        t = clean(p.extract_text() or "")
        if t:
            pages.append((i, t))
    return pages


def ingest_dir(directory: str | Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for pdf in sorted(Path(directory).glob("*.pdf")):
        chunks += make_chunks(pdf.name, extract_pages(pdf), start_id=len(chunks))
    return chunks
