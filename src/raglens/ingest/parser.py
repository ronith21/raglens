"""Document parsers for text, markdown, and PDF files."""

import hashlib
from pathlib import Path

from pypdf import PdfReader

from raglens.ingest.models import Document, DocumentPage

SUPPORTED_EXTENSIONS = {".txt", ".text", ".md", ".markdown", ".pdf"}


def compute_doc_id(source_path: str | Path, content: bytes) -> str:
    """Compute a deterministic doc_id from the file name and content hash.

    Only the file *name* is hashed (not the absolute path) so that the same corpus produces
    identical ids on every machine, which keeps the evaluation dataset portable.
    """
    hasher = hashlib.sha256()
    hasher.update(Path(source_path).name.encode("utf-8"))
    hasher.update(content)
    return hasher.hexdigest()[:16]


def _single_page_document(path: Path, raw_bytes: bytes, title: str, text: str) -> Document:
    """Build a one-page Document for plain text formats."""
    return Document(
        doc_id=compute_doc_id(path, raw_bytes),
        source_path=str(path),
        title=title,
        text=text,
        pages=[DocumentPage(page_number=1, text=text, start_char=0, end_char=len(text))],
        metadata={"extension": path.suffix.lower(), "size_bytes": len(raw_bytes)},
    )


def parse_text_file(path: str | Path) -> Document:
    """Parse a plain text file."""
    path = Path(path)
    raw_bytes = path.read_bytes()
    text = raw_bytes.decode("utf-8", errors="replace")
    return _single_page_document(path, raw_bytes, path.stem.replace("_", " ").title(), text)


def parse_markdown_file(path: str | Path) -> Document:
    """Parse a markdown file, extracting title from the first heading if available."""
    path = Path(path)
    raw_bytes = path.read_bytes()
    text = raw_bytes.decode("utf-8", errors="replace")

    title = path.stem.replace("_", " ").title()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            break
    return _single_page_document(path, raw_bytes, title, text)


def parse_pdf_file(path: str | Path) -> Document:
    """Parse a PDF file using pypdf, tracking per-page offsets and text."""
    path = Path(path)
    raw_bytes = path.read_bytes()
    reader = PdfReader(path)

    pages: list[DocumentPage] = []
    current_offset = 0
    for idx, pdf_page in enumerate(reader.pages):
        page_text = pdf_page.extract_text() or ""
        end_char = current_offset + len(page_text)
        pages.append(
            DocumentPage(
                page_number=idx + 1,
                text=page_text,
                start_char=current_offset,
                end_char=end_char,
            )
        )
        current_offset = end_char

    return Document(
        doc_id=compute_doc_id(path, raw_bytes),
        source_path=str(path),
        title=path.stem.replace("_", " ").title(),
        text="".join(p.text for p in pages),
        pages=pages,
        metadata={"extension": path.suffix.lower(), "total_pages": len(pages)},
    )


def ingest_file(path: str | Path) -> Document:
    """Ingest a single file based on its file extension."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".txt", ".text"):
        return parse_text_file(path)
    if suffix in (".md", ".markdown"):
        return parse_markdown_file(path)
    if suffix == ".pdf":
        return parse_pdf_file(path)
    raise ValueError(f"Unsupported file format: {suffix}")


def ingest_directory(directory_path: str | Path) -> list[Document]:
    """Ingest all supported documents from a directory, sorted by path."""
    dir_path = Path(directory_path)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {dir_path}")
    return [
        ingest_file(file_path)
        for file_path in sorted(dir_path.rglob("*"))
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
