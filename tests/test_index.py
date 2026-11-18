import pytest

from docintel.embeddings import HashingEmbedder
from docintel.index import VectorIndex


def test_search_returns_relevant_doc(index):
    hits = index.search("k-selection on GPU", k=2)
    assert hits[0][0].doc == "faiss.pdf"
    assert hits[0][1] >= hits[1][1]


def test_save_load_roundtrip(index, tmp_path):
    index.save(tmp_path)
    loaded = VectorIndex.load(tmp_path, HashingEmbedder())
    assert loaded.search("masked language model", 1)[0][0].doc == "bert.pdf"


def test_embedder_mismatch_rejected(index, tmp_path):
    index.save(tmp_path)

    class Other(HashingEmbedder):
        name = "other"

    with pytest.raises(ValueError):
        VectorIndex.load(tmp_path, Other())
