"""Tests for chunking algorithms and validation."""

import itertools

import pytest

from raglens.chunking.splitter import (
    RecursiveCharacterChunker,
    SentenceAwareChunker,
)
from raglens.ingest.models import Document, DocumentPage


def _doc(text: str, doc_id: str = "doc", source: str = "doc.txt") -> Document:
    return Document(
        doc_id=doc_id,
        source_path=source,
        text=text,
        pages=[DocumentPage(page_number=1, text=text, start_char=0, end_char=len(text))],
    )


def test_chunker_overlap_greater_than_or_equal_to_size_rejected() -> None:
    """Chunker must reject overlap >= size with a descriptive ValueError."""
    with pytest.raises(ValueError, match=r"chunk_overlap .* must be strictly less than"):
        RecursiveCharacterChunker(chunk_size=100, chunk_overlap=100)

    with pytest.raises(ValueError, match=r"chunk_overlap .* must be strictly less than"):
        RecursiveCharacterChunker(chunk_size=100, chunk_overlap=120)

    with pytest.raises(ValueError, match=r"chunk_size must be positive"):
        RecursiveCharacterChunker(chunk_size=0, chunk_overlap=0)

    with pytest.raises(ValueError, match=r"chunk_overlap cannot be negative"):
        RecursiveCharacterChunker(chunk_size=100, chunk_overlap=-5)

    with pytest.raises(ValueError, match=r"chunk_overlap .* must be strictly less than"):
        SentenceAwareChunker(chunk_size=50, chunk_overlap=50)


def test_chunk_empty_document() -> None:
    """Empty or whitespace-only documents produce zero chunks without error."""
    doc = Document(
        doc_id="empty_doc",
        source_path="empty.txt",
        text="   \n\t  ",
        pages=[DocumentPage(page_number=1, text="", start_char=0, end_char=0)],
    )
    chunker = RecursiveCharacterChunker(chunk_size=200, chunk_overlap=20)
    assert chunker.chunk_document(doc) == []


def test_chunk_unicode_characters() -> None:
    """Chunking preserves multi-byte UTF-8 and emoji characters without corruption."""
    doc = _doc("✨ Artificial Intelligence 🤖! " * 20, "unicode_doc", "unicode.txt")
    chunks = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=20).chunk_document(doc)

    assert len(chunks) > 1
    for chunk in chunks:
        assert doc.text[chunk.start_char : chunk.end_char] == chunk.text
        assert "✨" in chunk.text or "🤖" in chunk.text


def test_one_huge_paragraph() -> None:
    """Single gigantic paragraph without newlines is partitioned properly into bounded chunks."""
    doc = _doc("word " * 1000, "huge_p", "huge.txt")
    chunk_size = 250
    chunks = RecursiveCharacterChunker(chunk_size=chunk_size, chunk_overlap=50).chunk_document(doc)

    assert len(chunks) > 10
    for chunk in chunks:
        assert len(chunk.text) <= chunk_size
        assert doc.text[chunk.start_char : chunk.end_char] == chunk.text


def test_overlap_is_bounded_and_text_fully_covered() -> None:
    """Consecutive chunks overlap by at most chunk_overlap and together cover the whole text."""
    doc = _doc("alpha beta gamma delta epsilon. " * 60)
    size, overlap = 120, 30
    chunks = RecursiveCharacterChunker(chunk_size=size, chunk_overlap=overlap).chunk_document(doc)

    assert chunks[0].start_char == 0
    assert chunks[-1].end_char == len(doc.text.rstrip())  # nothing at the end is dropped
    for prev, nxt in itertools.pairwise(chunks):
        assert nxt.start_char > prev.start_char  # always makes progress
        assert nxt.start_char <= prev.end_char + 1  # no gaps beyond a separator
        assert prev.end_char - nxt.start_char <= overlap


def test_chunk_metadata_carries_title_and_slug() -> None:
    """Chunks inherit title, source path and a file-stem slug for evaluation."""
    doc = _doc("Some text about retrieval. " * 10, source="data/sample_corpus/09_rag.md")
    doc.title = "RAG"
    chunk = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=10).chunk_document(doc)[0]
    assert chunk.metadata["title"] == "RAG"
    assert chunk.metadata["doc_slug"] == "09_rag"
    assert chunk.chunk_id == "doc_chunk_0000"


def test_pdf_page_tracking_in_chunks() -> None:
    """Chunks spanning across multi-page documents attribute the proper page numbers."""
    page1_text = "This is content from page 1. It describes fundamental machine learning topics.\n"
    page2_text = "This is content from page 2. Deep neural networks and attention mechanisms.\n"
    full_text = page1_text + page2_text

    doc = Document(
        doc_id="pdf_doc",
        source_path="doc.pdf",
        text=full_text,
        pages=[
            DocumentPage(page_number=1, text=page1_text, start_char=0, end_char=len(page1_text)),
            DocumentPage(
                page_number=2,
                text=page2_text,
                start_char=len(page1_text),
                end_char=len(full_text),
            ),
        ],
    )

    chunks = RecursiveCharacterChunker(chunk_size=60, chunk_overlap=10).chunk_document(doc)

    assert len(chunks) >= 2
    assert chunks[0].page_number == 1
    page2_chunks = [c for c in chunks if c.start_char >= len(page1_text)]
    assert len(page2_chunks) > 0
    for c in page2_chunks:
        assert c.page_number == 2


def test_sentence_aware_chunker() -> None:
    """SentenceAwareChunker segments on sentence boundaries."""
    text = (
        "Gradient descent optimizes loss functions. "
        "Overfitting occurs when model variance is too high. "
        "Transformers utilize self-attention mechanisms. "
        "RAG combines retrieval with language generation."
    )
    doc = _doc(text, "sentences_doc")
    chunks = SentenceAwareChunker(chunk_size=110, chunk_overlap=20).chunk_document(doc)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert doc.text[chunk.start_char : chunk.end_char] == chunk.text
        assert chunk.text[0].isupper()  # every chunk starts at a sentence boundary


def test_sentence_aware_chunker_splits_overlong_sentence() -> None:
    """A single sentence longer than chunk_size is hard-split into bounded pieces."""
    doc = _doc("x" * 250 + ". Short tail sentence.")
    chunks = SentenceAwareChunker(chunk_size=100, chunk_overlap=10).chunk_document(doc)
    assert all(len(c.text) <= 100 for c in chunks)
    assert "".join(c.text for c in chunks).startswith("x" * 100)


def test_custom_separators_and_hard_split() -> None:
    """With no separators left the splitter falls back to fixed-size slices."""
    chunker = RecursiveCharacterChunker(chunk_size=10, chunk_overlap=2, separators=["|"])
    pieces = chunker.chunk_text("a" * 35)
    assert [len(p[0]) for p in pieces] == [10, 10, 10, 5]
    assert chunker.chunk_text("") == []
