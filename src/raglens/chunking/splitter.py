"""Text chunking strategies: recursive character and sentence-aware splitters."""

import re
from abc import ABC, abstractmethod
from pathlib import Path

from raglens.ingest.models import Chunk, Document


def _find_page_number_for_offset(doc: Document, offset: int) -> int:
    """Find the page number that contains the given character offset."""
    if not doc.pages:
        return 1
    for page in doc.pages:
        if page.start_char <= offset < page.end_char:
            return page.page_number
    return doc.pages[-1].page_number


def _advance_with_overlap(
    spans: list[tuple[int, int]], start_idx: int, end_idx: int, end_char: int, overlap: int
) -> int:
    """Pick the next window start so it begins at or after ``end_char - overlap``.

    ``spans`` are (start, end) character offsets; the window just emitted covered
    ``spans[start_idx:end_idx]``. The returned index always moves forward.
    """
    target_start = end_char - overlap
    next_i = end_idx
    for k in range(start_idx + 1, end_idx):
        if spans[k][0] >= target_start:
            next_i = k
            break
    return max(next_i, start_idx + 1)


def _merge_spans(
    text: str, spans: list[tuple[int, int]], size: int, overlap: int
) -> list[tuple[str, int, int]]:
    """Greedily merge adjacent spans into windows of at most ``size`` characters."""
    chunks: list[tuple[str, int, int]] = []
    i = 0
    n = len(spans)
    while i < n:
        start_char, end_char = spans[i]
        j = i + 1
        while j < n and spans[j][1] - start_char <= size:
            end_char = spans[j][1]
            j += 1
        # Trim whitespace at the window edges (separators are kept on the left piece).
        lo, hi = start_char, end_char
        while lo < hi and text[lo].isspace():
            lo += 1
        while hi > lo and text[hi - 1].isspace():
            hi -= 1
        if hi > lo:
            chunks.append((text[lo:hi], lo, hi))
        if j >= n:
            break
        i = _advance_with_overlap(spans, i, j, end_char, overlap)
    return chunks


class BaseChunker(ABC):
    """Abstract base class for document chunkers."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than "
                f"chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk_text(self, text: str) -> list[tuple[str, int, int]]:
        """Split text into (chunk_text, start_offset, end_offset) tuples."""

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split a document into Chunk objects with page and character tracking."""
        if not document.text.strip():
            return []

        chunks: list[Chunk] = []
        for idx, (piece_text, start_char, end_char) in enumerate(self.chunk_text(document.text)):
            metadata = dict(document.metadata)
            metadata.setdefault("title", document.title)
            metadata.setdefault("source_path", document.source_path)
            metadata.setdefault("doc_slug", Path(document.source_path).stem)
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}_chunk_{idx:04d}",
                    doc_id=document.doc_id,
                    text=piece_text,
                    page_number=_find_page_number_for_offset(document, start_char),
                    start_char=start_char,
                    end_char=end_char,
                    metadata=metadata,
                )
            )
        return chunks

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """Chunk multiple documents."""
        all_chunks: list[Chunk] = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks


class RecursiveCharacterChunker(BaseChunker):
    """Recursive character splitter splitting hierarchically on separators."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: list[str] | None = None,
    ) -> None:
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def chunk_text(self, text: str) -> list[tuple[str, int, int]]:
        """Split text recursively by candidate separators while maintaining exact offsets."""
        if not text:
            return []
        slices = self._split_recursive(text, 0, self.separators)
        spans = [(start, end) for _, start, end in slices]
        return _merge_spans(text, spans, self.chunk_size, self.chunk_overlap)

    def _split_recursive(
        self, text: str, base_offset: int, separators: list[str]
    ) -> list[tuple[str, int, int]]:
        """Recursively split text using the first separator that appears."""
        if len(text) <= self.chunk_size:
            return [(text, base_offset, base_offset + len(text))]

        if not separators:
            result: list[tuple[str, int, int]] = []
            for curr in range(0, len(text), self.chunk_size):
                end = min(curr + self.chunk_size, len(text))
                result.append((text[curr:end], base_offset + curr, base_offset + end))
            return result

        separator, remaining = separators[0], separators[1:]
        if separator == "":
            return [(char, base_offset + i, base_offset + i + 1) for i, char in enumerate(text)]
        if separator not in text:
            return self._split_recursive(text, base_offset, remaining)

        # Keep each separator attached to the piece before it, so pieces tile the text exactly
        # and nothing (such as a document's final ". ") is silently dropped.
        parts = text.split(separator)
        pieces: list[tuple[str, int, int]] = []
        curr_offset = 0
        for i, part in enumerate(parts):
            piece = part + separator if i < len(parts) - 1 else part
            if piece:
                start = base_offset + curr_offset
                if len(piece) > self.chunk_size:
                    pieces.extend(self._split_recursive(piece, start, remaining))
                else:
                    pieces.append((piece, start, start + len(piece)))
            curr_offset += len(piece)
        return pieces


class SentenceAwareChunker(BaseChunker):
    """Sentence-aware chunker that uses regex sentence boundary detection."""

    _SENTENCE_REGEX = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")

    def chunk_text(self, text: str) -> list[tuple[str, int, int]]:
        """Split text by sentences, grouping them into chunks of size <= chunk_size."""
        if not text:
            return []

        spans: list[tuple[int, int]] = []
        curr_pos = 0
        for match in self._SENTENCE_REGEX.finditer(text):
            split_start, split_end = match.span()
            if split_start > curr_pos:
                spans.append((curr_pos, split_start))
            curr_pos = split_end
        if curr_pos < len(text):
            spans.append((curr_pos, len(text)))

        final_spans: list[tuple[int, int]] = []
        for start, end in spans:
            if end - start <= self.chunk_size:
                final_spans.append((start, end))
                continue
            for sub in range(start, end, self.chunk_size):
                final_spans.append((sub, min(sub + self.chunk_size, end)))

        return _merge_spans(text, final_spans, self.chunk_size, self.chunk_overlap)
