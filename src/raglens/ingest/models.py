"""Document data models for ingestion and chunking."""

from typing import Any

from pydantic import BaseModel, Field


class DocumentPage(BaseModel):
    """Represents a single page or section in an ingested document."""

    page_number: int = Field(default=1, description="1-indexed page number or 1 for text/markdown")
    text: str = Field(default="", description="Text content of the page")
    start_char: int = Field(default=0, description="Starting character offset in document text")
    end_char: int = Field(default=0, description="Ending character offset in document text")


class Document(BaseModel):
    """Ingested document model."""

    doc_id: str = Field(..., description="Unique document identifier (e.g. SHA256 or slug)")
    source_path: str = Field(..., description="Path to original file")
    title: str = Field(default="", description="Document title extracted from filename or header")
    text: str = Field(default="", description="Full text content of document")
    pages: list[DocumentPage] = Field(
        default_factory=list, description="Page-level text breakdowns"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")


class Chunk(BaseModel):
    """Text chunk produced by a chunker."""

    chunk_id: str = Field(..., description="Unique chunk identifier (e.g. doc_id_chunk_idx)")
    doc_id: str = Field(..., description="Identifier of the source document")
    text: str = Field(default="", description="Chunk text content")
    page_number: int = Field(default=1, description="Source page number where chunk starts")
    start_char: int = Field(default=0, description="Start character offset in full document text")
    end_char: int = Field(default=0, description="End character offset in full document text")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Metadata passed from document"
    )
