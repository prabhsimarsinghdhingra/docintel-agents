"""Embedding backends. SentenceEmbedder is the real one; HashingEmbedder is a
dependency-free fallback used by unit tests and offline smoke runs."""
from __future__ import annotations

import os
import re
import zlib

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceEmbedder:
    def __init__(self, name: str = DEFAULT_MODEL):
        from sentence_transformers import SentenceTransformer  # lazy: heavy import

        self.name = name
        self.model = SentenceTransformer(name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        v = self.model.encode(texts, normalize_embeddings=True, batch_size=64,
                              show_progress_bar=False)
        return np.asarray(v, dtype="float32")


class HashingEmbedder:
    name = "hashing-384"

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", t.lower()):
                h = zlib.crc32(tok.encode())
                out[i, h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


def get_embedder():
    if os.getenv("EMBEDDER", "").lower() == "hashing":
        return HashingEmbedder()
    return SentenceEmbedder()
