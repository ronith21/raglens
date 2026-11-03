"""Text chunking strategies: recursive character and sentence-aware splitters."""

import re
from abc import ABC, abstractmethod

from raglens.ingest.models import Chunk, Document


def _find_page_number_for_offset(doc: Document, offset: int) -> int:
    """Find the page number that contains the given character offset."""
    if not doc.pages:
        return 1
    for page in doc.pages:
        if page.start_char <= offset < page.end_char:
            return page.page_number
    return doc.pages[-1].page_number


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
        pass

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split a document into Chunk objects with page and character tracking."""
        if not document.text.strip():
            return []

        raw_chunks = self.chunk_text(document.text)
        chunks: list[Chunk] = []

        for idx, (piece_text, start_char, end_char) in enumerate(raw_chunks):
            page_num = _find_page_number_for_offset(document, start_char)
            chunk_id = f"{document.doc_id}_chunk_{idx:04d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    text=piece_text,
                    page_number=page_num,
                    start_char=start_char,
                    end_char=end_char,
                    metadata=dict(document.metadata),
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
        # Merge small slices up to chunk_size with chunk_overlap
        return self._merge_slices(text, slices)

    def _split_recursive(
        self, text: str, base_offset: int, separators: list[str]
    ) -> list[tuple[str, int, int]]:
        """Recursively split text using the first separator that appears."""
        if len(text) <= self.chunk_size:
            return [(text, base_offset, base_offset + len(text))]

        if not separators:
            # Force hard split into chunk_size pieces
            result: list[tuple[str, int, int]] = []
            curr = 0
            while curr < len(text):
                end = min(curr + self.chunk_size, len(text))
                result.append((text[curr:end], base_offset + curr, base_offset + end))
                curr = end
            return result

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            return [(char, base_offset + i, base_offset + i + 1) for i, char in enumerate(text)]

        if separator not in text:
            return self._split_recursive(text, base_offset, remaining_separators)

        parts = text.split(separator)
        pieces: list[tuple[str, int, int]] = []
        curr_offset = 0

        for i, part in enumerate(parts):
            if i > 0:
                curr_offset += len(separator)

            part_len = len(part)
            if part_len > 0:
                start = base_offset + curr_offset
                if part_len > self.chunk_size:
                    sub_pieces = self._split_recursive(part, start, remaining_separators)
                    pieces.extend(sub_pieces)
                else:
                    pieces.append((part, start, start + part_len))
            curr_offset += part_len

        return pieces

    def _merge_slices(
        self, full_text: str, slices: list[tuple[str, int, int]]
    ) -> list[tuple[str, int, int]]:
        """Merge adjacent pieces up to chunk_size and advance with overlap."""
        if not slices:
            return []

        chunks: list[tuple[str, int, int]] = []
        i = 0
        n = len(slices)

        while i < n:
            start_slice_idx = i
            start_char = slices[start_slice_idx][1]
            end_char = slices[start_slice_idx][2]
            j = i + 1

            while j < n:
                next_end_char = slices[j][2]
                if next_end_char - start_char <= self.chunk_size:
                    end_char = next_end_char
                    j += 1
                else:
                    break

            chunk_content = full_text[start_char:end_char]
            chunks.append((chunk_content, start_char, end_char))

            if j >= n:
                break

            target_start = end_char - self.chunk_overlap
            next_i = j
            for k in range(start_slice_idx + 1, j):
                if slices[k][1] <= target_start:
                    next_i = k
                else:
                    if next_i == j:
                        next_i = k
                    break

            if next_i <= i:
                next_i = i + 1
            i = next_i

        return chunks


class SentenceAwareChunker(BaseChunker):
    """Sentence-aware chunker that uses regex sentence boundary detection."""

    _SENTENCE_REGEX = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def chunk_text(self, text: str) -> list[tuple[str, int, int]]:
        """Split text by sentences, grouping them into chunks of size <= chunk_size."""
        if not text:
            return []

        spans: list[tuple[int, int]] = []
        matches = list(self._SENTENCE_REGEX.finditer(text))
        curr_pos = 0

        for match in matches:
            split_start, split_end = match.span()
            if split_start > curr_pos:
                spans.append((curr_pos, split_start))
            curr_pos = split_end

        if curr_pos < len(text):
            spans.append((curr_pos, len(text)))

        final_spans: list[tuple[int, int]] = []
        for start, end in spans:
            span_text = text[start:end]
            if len(span_text) <= self.chunk_size:
                final_spans.append((start, end))
            else:
                sub_curr = start
                while sub_curr < end:
                    sub_end = min(sub_curr + self.chunk_size, end)
                    final_spans.append((sub_curr, sub_end))
                    sub_curr = sub_end

        chunks: list[tuple[str, int, int]] = []
        i = 0
        n = len(final_spans)

        while i < n:
            start_idx = i
            start_char = final_spans[start_idx][0]
            end_char = final_spans[start_idx][1]
            j = i + 1

            while j < n:
                cand_end_char = final_spans[j][1]
                if cand_end_char - start_char <= self.chunk_size:
                    end_char = cand_end_char
                    j += 1
                else:
                    break

            chunks.append((text[start_char:end_char], start_char, end_char))
            if j >= n:
                break

            target_start = end_char - self.chunk_overlap
            next_i = j
            for k in range(start_idx + 1, j):
                if final_spans[k][0] <= target_start:
                    next_i = k
                else:
                    if next_i == j:
                        next_i = k
                    break

            if next_i <= i:
                next_i = i + 1
            i = next_i

        return chunks
