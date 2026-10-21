import pytest
from pydantic import ValidationError

from atlas.chunking.chunker import Chunker, detect_sections, pack_sentences, split_sentences
from atlas.common.config import ChunkingConfig
from atlas.common.types import Document, PageContent


def test_split_sentences_joins_pdf_line_wraps():
    assert split_sentences("This is one\nsentence. And another one.") == ["This is one sentence.", "And another one."]


def test_pack_sentences_respects_size_and_overlap():
    sents = [f"Sentence number {i} has some words." for i in range(20)]
    chunks = pack_sentences(sents, size=120, overlap=40)
    assert all(len(c) <= 120 for c in chunks)
    # consecutive chunks share at least one sentence (overlap)
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert a.split(". ")[-1].rstrip(".") in b


def test_pack_sentences_hard_splits_oversized_sentence():
    chunks = pack_sentences(["x" * 250], size=100, overlap=10)
    assert len(chunks) >= 3 and all(len(c) <= 100 for c in chunks)


def test_chunks_keep_page_and_never_cross_pages():
    doc = Document(doc_id="d1", filename="f.pdf", source_path="f.pdf")
    pages = [PageContent(page=1, text="Alpha beta gamma. " * 30), PageContent(page=2, text="Delta epsilon. " * 30)]
    chunks = Chunker(ChunkingConfig(chunk_size=200, chunk_overlap=40, min_chunk_chars=10)).chunk_document(doc, pages)
    assert {c.page for c in chunks} == {1, 2}
    assert all(("Alpha" in c.text) != ("Delta" in c.text) for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_chunk_ids_are_deterministic():
    doc = Document(doc_id="d1", filename="f.txt", source_path="f.txt")
    pages = [PageContent(page=None, text="Stable text here. " * 20)]
    cfg = ChunkingConfig(chunk_size=100, chunk_overlap=20, min_chunk_chars=10)
    assert [c.chunk_id for c in Chunker(cfg).chunk_document(doc, pages)] == \
           [c.chunk_id for c in Chunker(cfg).chunk_document(doc, pages)]


def test_detect_sections_numbered_headings():
    text = "3.2\nAttention\nSome text.\n4 Why Self-Attention\nMore."
    titles = [t for _, t in detect_sections(text)]
    assert "3.2 Attention" in titles and "4 Why Self-Attention" in titles


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValidationError):
        ChunkingConfig(chunk_size=100, chunk_overlap=100)
