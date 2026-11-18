import pytest

from docintel.embeddings import HashingEmbedder
from docintel.index import VectorIndex
from docintel.ingest import make_chunks

DOCS = {
    "transformer.pdf": [(1, "The Transformer relies entirely on self-attention to compute representations of its input and output without recurrence. "
                            "It uses multi-head attention with eight parallel attention heads in the base model. Positional encodings inject order information.")],
    "faiss.pdf": [(3, "FAISS performs efficient similarity search over dense vectors on GPU. "
                      "It implements k-selection with warp-level registers to find the nearest neighbours quickly in billion-scale collections.")],
    "bert.pdf": [(2, "BERT is pre-trained with a masked language model objective and a next sentence prediction task. "
                     "Fine-tuning adds a single output layer for downstream tasks.")],
}


@pytest.fixture(scope="session")
def index():
    chunks, nid = [], 0
    for doc, pages in DOCS.items():
        cs = make_chunks(doc, pages, nid)
        chunks += cs
        nid += len(cs)
    return VectorIndex(HashingEmbedder()).build(chunks)
