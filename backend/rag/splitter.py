"""
Text chunking module.
Splits extracted document pages into smaller, overlapping chunks
suitable for embedding and retrieval.
"""

import uuid
from dataclasses import dataclass, field
from typing import Optional

from backend.rag.loader import ExtractedPage


@dataclass
class TextChunk:
    """A chunk of text ready for embedding."""
    chunk_id: str
    text: str
    page_number: int
    document_name: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


class RecursiveTextSplitter:
    """
    Recursively splits text into chunks using a hierarchy of separators.
    Tries paragraph breaks first, then sentences, then words.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        separators: Optional[list[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]

    def split_pages(self, pages: list[ExtractedPage]) -> list[TextChunk]:
        """
        Split a list of ExtractedPages into TextChunks.
        Preserves page number and document metadata for each chunk.
        """
        all_chunks = []
        global_index = 0

        for page in pages:
            if not page.text.strip():
                continue

            text_splits = self._split_text(page.text)

            for split_text in text_splits:
                chunk = TextChunk(
                    chunk_id=str(uuid.uuid4()),
                    text=split_text,
                    page_number=page.page_number,
                    document_name=page.document_name,
                    chunk_index=global_index,
                    metadata={
                        **page.metadata,
                        "char_count": len(split_text),
                        "word_count": len(split_text.split()),
                    }
                )
                all_chunks.append(chunk)
                global_index += 1

        return all_chunks

    def _split_text(self, text: str) -> list[str]:
        """Split text into chunks using recursive separator strategy."""
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        return self._recursive_split(text, 0)

    def _recursive_split(self, text: str, separator_index: int) -> list[str]:
        """Recursively split text, trying separators from most to least preferred."""
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        if separator_index >= len(self.separators):
            # Last resort: hard split by character count
            return self._hard_split(text)

        separator = self.separators[separator_index]
        parts = text.split(separator)

        if len(parts) == 1:
            # This separator didn't split anything, try the next one
            return self._recursive_split(text, separator_index + 1)

        # Merge parts into chunks respecting chunk_size
        chunks: list[str] = []
        current_chunk: list[str] = []
        current_length = 0

        for part in parts:
            part_length = len(part) + len(separator)

            if current_length + part_length > self.chunk_size and current_chunk:
                # Finalize current chunk
                chunk_text = separator.join(current_chunk).strip()
                if chunk_text:
                    chunks.append(chunk_text)

                # Start new chunk with overlap
                overlap_chunks = self._get_overlap_parts(
                    current_chunk, separator, self.chunk_overlap
                )
                current_chunk = overlap_chunks
                current_length = sum(len(c) + len(separator) for c in current_chunk)

            current_chunk.append(part)
            current_length += part_length

        # Finalize the last chunk
        if current_chunk:
            chunk_text = separator.join(current_chunk).strip()
            if chunk_text:
                # If still too large, recurse with next separator
                if len(chunk_text) > self.chunk_size:
                    chunks.extend(
                        self._recursive_split(chunk_text, separator_index + 1)
                    )
                else:
                    chunks.append(chunk_text)

        return chunks

    def _get_overlap_parts(
        self, parts: list[str], separator: str, max_overlap: int
    ) -> list[str]:
        """Get parts from the end of the list that fit within the overlap budget."""
        overlap_parts: list[str] = []
        overlap_length = 0

        for part in reversed(parts):
            part_len = len(part) + len(separator)
            if overlap_length + part_len > max_overlap:
                break
            overlap_parts.insert(0, part)
            overlap_length += part_len

        return overlap_parts

    def _hard_split(self, text: str) -> list[str]:
        """Last resort: split text by character count with overlap."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            # Move forward, minus overlap
            start = end - self.chunk_overlap

        return chunks


def split_document(
    pages: list[ExtractedPage],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[TextChunk]:
    """
    Convenience function to split document pages into chunks.

    Args:
        pages: List of extracted pages from the document loader.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        List of TextChunk objects ready for embedding.
    """
    splitter = RecursiveTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_pages(pages)
