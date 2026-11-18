"""Build the FAISS index from PDFs: python -m docintel.build"""
from __future__ import annotations

import argparse

from .embeddings import get_embedder
from .index import VectorIndex
from .ingest import ingest_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="data/docs")
    ap.add_argument("--out", default="data/index")
    a = ap.parse_args()
    chunks = ingest_dir(a.docs)
    if not chunks:
        raise SystemExit(f"No PDFs found in {a.docs}. Run: python -m docintel.download")
    idx = VectorIndex(get_embedder()).build(chunks)
    idx.save(a.out)
    print(f"Indexed {len(chunks)} chunks from {len({c.doc for c in chunks})} documents -> {a.out}")


if __name__ == "__main__":
    main()
