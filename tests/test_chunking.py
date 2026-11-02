"""Tests for chunking algorithms and validation."""

import pytest

from raglens.chunking.splitter import (
    RecursiveCharacterChunker,
    SentenceAwareChunker,
)
from raglens.ingest.models import Document, DocumentPage


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
    chunks = chunker.chunk_document(doc)
    assert chunks == []


def test_chunk_unicode_characters() -> None:
    """Chunking preserves multi-byte UTF-8 and emoji characters without corruption."""
    unicode_text = "✨ Artificial Intelligence 🤖! " * 20
    doc = Document(
        doc_id="unicode_doc",
        source_path="unicode.txt",
        text=unicode_text,
        pages=[
            DocumentPage(page_number=1, text=unicode_text, start_char=0, end_char=len(unicode_text))
        ],
    )
    chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 1
    for chunk in chunks:
        assert doc.text[chunk.start_char : chunk.end_char] == chunk.text
        assert "✨" in chunk.text or "🤖" in chunk.text


def test_one_huge_paragraph() -> None:
    """Single gigantic paragraph without newlines is partitioned properly into bounded chunks."""
    huge_paragraph = "word " * 1000
    doc = Document(
        doc_id="huge_p",
        source_path="huge.txt",
        text=huge_paragraph,
        pages=[
            DocumentPage(
                page_number=1, text=huge_paragraph, start_char=0, end_char=len(huge_paragraph)
            )
        ],
    )
    chunk_size = 250
    overlap = 50
    chunker = RecursiveCharacterChunker(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 10
    for chunk in chunks:
        assert len(chunk.text) <= chunk_size
        assert doc.text[chunk.start_char : chunk.end_char] == chunk.text


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

    chunker = RecursiveCharacterChunker(chunk_size=60, chunk_overlap=10)
    chunks = chunker.chunk_document(doc)

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
    doc = Document(
        doc_id="sentences_doc",
        source_path="doc.txt",
        text=text,
        pages=[DocumentPage(page_number=1, text=text, start_char=0, end_char=len(text))],
    )
    chunker = SentenceAwareChunker(chunk_size=110, chunk_overlap=20)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert doc.text[chunk.start_char : chunk.end_char] == chunk.text
