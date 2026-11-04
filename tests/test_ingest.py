"""Tests for document ingestion modules."""

from pathlib import Path

import pytest

from raglens.ingest.models import Document
from raglens.ingest.parser import (
    compute_doc_id,
    ingest_directory,
    ingest_file,
    parse_markdown_file,
    parse_text_file,
)


def create_sample_pdf(tmp_path: Path) -> Path:
    """Helper to generate a multi-page PDF dynamically for testing."""
    pdf_content = (
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] /Contents 5 0 R "
        b"/Resources << /Font << /F1 7 0 R >> >> >> endobj\n"
        b"4 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] /Contents 6 0 R "
        b"/Resources << /Font << /F1 7 0 R >> >> >> endobj\n"
        b"5 0 obj << /Length 44 >> stream\n"
        b"BT /F1 12 Tf 50 250 Td (Page 1 content here) Tj ET\n"
        b"endstream\nendobj\n"
        b"6 0 obj << /Length 44 >> stream\n"
        b"BT /F1 12 Tf 50 250 Td (Page 2 content here) Tj ET\n"
        b"endstream\nendobj\n"
        b"7 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
        b"xref\n0 8\n0000000000 65535 f \n"
        b"0000000009 00000 n \n0000000056 00000 n \n0000000122 00000 n \n"
        b"0000000253 00000 n \n0000000384 00000 n \n0000000478 00000 n \n"
        b"0000000572 00000 n \ntrailer << /Size 8 /Root 1 0 R >>\n"
        b"startxref\n650\n%%EOF\n"
    )
    pdf_file = tmp_path / "test_doc.pdf"
    pdf_file.write_bytes(pdf_content)
    return pdf_file


def test_empty_text_file(tmp_path: Path) -> None:
    """Empty files should ingest cleanly with 0 length text."""
    p = tmp_path / "empty.txt"
    p.write_text("", encoding="utf-8")

    doc = parse_text_file(p)
    assert doc.text == ""
    assert len(doc.pages) == 1
    assert doc.pages[0].start_char == 0
    assert doc.pages[0].end_char == 0
    assert doc.doc_id != ""


def test_empty_markdown_file(tmp_path: Path) -> None:
    """Empty markdown file ingestion."""
    p = tmp_path / "empty.md"
    p.write_text("", encoding="utf-8")

    doc = parse_markdown_file(p)
    assert doc.text == ""
    assert doc.title == "Empty"


def test_unicode_handling(tmp_path: Path) -> None:
    """Ensure complex unicode, emojis, accents, and non-Latin scripts are preserved."""
    content = "# Über den Dächern von Zürich 🌟\n\n机器学习 és una meravella! 日本語と한국어."
    p = tmp_path / "unicode.md"
    p.write_text(content, encoding="utf-8")

    doc = ingest_file(p)
    assert doc.title == "Über den Dächern von Zürich 🌟"
    assert "机器学习" in doc.text
    assert "🌟" in doc.text
    assert doc.pages[0].end_char == len(content)


def test_pdf_page_tracking(tmp_path: Path) -> None:
    """PDF parsing should accurately track individual pages and character offsets."""
    doc = ingest_file(create_sample_pdf(tmp_path))

    assert isinstance(doc, Document)
    assert len(doc.pages) == 2
    assert doc.pages[0].page_number == 1
    assert "Page 1" in doc.pages[0].text
    assert doc.pages[1].page_number == 2
    assert "Page 2" in doc.pages[1].text
    assert doc.pages[1].start_char == doc.pages[0].end_char
    assert doc.text == doc.pages[0].text + doc.pages[1].text
    assert doc.metadata["total_pages"] == 2


def test_unsupported_file_format(tmp_path: Path) -> None:
    """Unsupported extension raises ValueError."""
    p = tmp_path / "test.unknown"
    p.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file format"):
        ingest_file(p)


def test_ingest_directory(tmp_path: Path) -> None:
    """Directory ingestion parses all supported formats recursively."""
    (tmp_path / "f1.txt").write_text("Hello text", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "f2.md").write_text("# Title 2\nBody text", encoding="utf-8")
    (tmp_path / "ignored.bin").write_bytes(b"\x00\x01\x02")

    docs = ingest_directory(tmp_path)
    assert len(docs) == 2
    assert any(d.title == "Title 2" for d in docs)
    assert any(d.text == "Hello text" for d in docs)


def test_ingest_missing_directory(tmp_path: Path) -> None:
    """A missing corpus directory is a clear error, not an empty result."""
    with pytest.raises(FileNotFoundError, match="Corpus directory not found"):
        ingest_directory(tmp_path / "nope")


def test_deterministic_doc_id(tmp_path: Path) -> None:
    """Same path and content produces identical doc_id."""
    p = tmp_path / "stable.txt"
    p.write_text("deterministic content", encoding="utf-8")
    assert compute_doc_id(p, p.read_bytes()) == compute_doc_id(p, p.read_bytes())


def test_doc_id_is_location_independent(tmp_path: Path) -> None:
    """The same file in two folders gets the same id; different content gets a different one."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    assert compute_doc_id(tmp_path / "a" / "x.md", b"same") == compute_doc_id(
        tmp_path / "b" / "x.md", b"same"
    )
    assert compute_doc_id("x.md", b"one") != compute_doc_id("x.md", b"two")
