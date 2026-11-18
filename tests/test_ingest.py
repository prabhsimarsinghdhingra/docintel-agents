from docintel.ingest import chunk_text, clean, make_chunks, split_sentences


def test_clean_dehyphenates():
    assert clean("retrie-\nval  works") == "retrieval works"


def test_chunk_size_and_overlap():
    text = " ".join(f"Sentence number {i} talks about retrieval." for i in range(80))
    chunks = chunk_text(text, size=300, overlap=50)
    assert len(chunks) > 3
    assert all(len(c) <= 360 for c in chunks)
    assert chunks[0][-20:] in chunks[1]  # overlap carried forward


def test_long_sentence_is_split():
    assert len(chunk_text("word " * 1000, size=300)) > 3


def test_make_chunks_ids_and_pages():
    cs = make_chunks("d.pdf", [(1, "A" * 100 + ". " + "B" * 100 + "."), (2, "C" * 100 + ".")], start_id=10)
    assert [c.id for c in cs] == list(range(10, 10 + len(cs)))
    assert {c.page for c in cs} == {1, 2}


def test_split_sentences():
    assert len(split_sentences("One is here. Two is here! Three is here?")) == 3
