"""Document parsers for text, markdown, and PDF files."""

import hashlib
from pathlib import Path

from pypdf import PdfReader

from raglens.ingest.models import Document, DocumentPage


def compute_doc_id(source_path: str | Path, content: bytes) -> str:
    """Compute a deterministic doc_id from source path and content hash."""
    hasher = hashlib.sha256()
    hasher.update(str(source_path).encode("utf-8"))
    hasher.update(content)
    return hasher.hexdigest()[:16]


def parse_text_file(path: str | Path) -> Document:
    """Parse a plain text file."""
    path = Path(path)
    raw_bytes = path.read_bytes()
    text = raw_bytes.decode("utf-8", errors="replace")
    doc_id = compute_doc_id(path, raw_bytes)
    page = DocumentPage(
        page_number=1,
        text=text,
        start_char=0,
        end_char=len(text),
    )
    return Document(
        doc_id=doc_id,
        source_path=str(path),
        title=path.stem.replace("_", " ").title(),
        text=text,
        pages=[page],
        metadata={"extension": path.suffix.lower(), "size_bytes": len(raw_bytes)},
    )


def parse_markdown_file(path: str | Path) -> Document:
    """Parse a markdown file, extracting title from the first heading if available."""
    path = Path(path)
    raw_bytes = path.read_bytes()
    text = raw_bytes.decode("utf-8", errors="replace")
    doc_id = compute_doc_id(path, raw_bytes)

    title = path.stem.replace("_", " ").title()
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            title = line[2:].strip()
            break

    page = DocumentPage(
        page_number=1,
        text=text,
        start_char=0,
        end_char=len(text),
    )
    return Document(
        doc_id=doc_id,
        source_path=str(path),
        title=title,
        text=text,
        pages=[page],
        metadata={"extension": path.suffix.lower(), "size_bytes": len(raw_bytes)},
    )


def parse_pdf_file(path: str | Path) -> Document:
    """Parse a PDF file using pypdf, tracking per-page offsets and text."""
    path = Path(path)
    raw_bytes = path.read_bytes()
    doc_id = compute_doc_id(path, raw_bytes)

    reader = PdfReader(path)
    pages: list[DocumentPage] = []
    full_text_parts: list[str] = []
    current_offset = 0

    for idx, pdf_page in enumerate(reader.pages):
        page_num = idx + 1
        page_text = pdf_page.extract_text() or ""
        start_char = current_offset
        end_char = start_char + len(page_text)

        pages.append(
            DocumentPage(
                page_number=page_num,
                text=page_text,
                start_char=start_char,
                end_char=end_char,
            )
        )
        full_text_parts.append(page_text)
        current_offset = end_char

    full_text = "".join(full_text_parts)

    return Document(
        doc_id=doc_id,
        source_path=str(path),
        title=path.stem.replace("_", " ").title(),
        text=full_text,
        pages=pages,
        metadata={"extension": path.suffix.lower(), "total_pages": len(pages)},
    )


def ingest_file(path: str | Path) -> Document:
    """Ingest a single file based on its file extension."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".txt", ".text"):
        return parse_text_file(path)
    elif suffix in (".md", ".markdown"):
        return parse_markdown_file(path)
    elif suffix == ".pdf":
        return parse_pdf_file(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def ingest_directory(directory_path: str | Path) -> list[Document]:
    """Ingest all supported documents from a directory, sorted by path."""
    dir_path = Path(directory_path)
    supported_extensions = {".txt", ".text", ".md", ".markdown", ".pdf"}
    documents: list[Document] = []
    for file_path in sorted(dir_path.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            documents.append(ingest_file(file_path))
    return documents
