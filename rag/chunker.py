"""Overlap-aware chunking tuned for textbook prose and worked examples.

``RecursiveCharacterTextSplitter`` is used with separators that respect
paragraphs, sentences and equations.  Each chunk keeps the page metadata and
gains ``chunk_index`` / ``chunk_id`` so answers can cite an exact location and
so re-ingestion can detect duplicates.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Settings
from utils.logging_utils import get_logger
from utils.text_utils import clean_text, hash_text

__all__ = ["DocumentChunker", "DEFAULT_SEPARATORS"]

logger = get_logger(__name__)

DEFAULT_SEPARATORS: Sequence[str] = (
    "\n\n",  # paragraphs
    "\n",  # bullet points / table rows
    ". ",
    "? ",
    "! ",
    "; ",
    ") ",
    " ",  # last resort: word boundary
)


class DocumentChunker:
    """Split page documents into retrieval-friendly, metadata-rich chunks."""

    def __init__(
        self,
        settings: Settings,
        *,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.settings = settings
        size = chunk_size or settings.chunk_size
        overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        if overlap >= size:
            raise ValueError(
                f"chunk_overlap ({overlap}) must be smaller than chunk_size ({size})"
            )
        self.chunk_size = size
        self.chunk_overlap = overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            separators=list(DEFAULT_SEPARATORS),
            keep_separator=True,
            length_function=len,
            add_start_index=True,
        )

    def split(self, documents: Iterable[Document]) -> list[Document]:
        """Split documents, dropping fragments that are too short to be useful."""
        chunks: list[Document] = []
        for document in documents:
            content = clean_text(document.page_content)
            if not content:
                continue

            pieces = self._splitter.split_text(content)
            page_id = str(document.metadata.get("doc_id") or hash_text(content))
            kept = 0
            for index, piece in enumerate(pieces):
                piece = piece.strip()
                if len(piece) < self.settings.min_chunk_chars and len(pieces) > 1:
                    logger.debug("Dropping %d-char fragment from %s", len(piece), page_id)
                    continue
                chunks.append(
                    Document(
                        page_content=piece,
                        metadata={
                            **document.metadata,
                            "chunk_index": index,
                            "chunk_id": f"{page_id}-c{index:03d}",
                            "chunk_chars": len(piece),
                        },
                    )
                )
                kept += 1
            logger.debug("Page %s -> %d/%d chunks kept", page_id, kept, len(pieces))

        page_count = len({str(chunk.metadata.get("doc_id")) for chunk in chunks})
        logger.info(
            "Chunked %d page(s) into %d chunk(s) (size=%d, overlap=%d)",
            page_count,
            len(chunks),
            self.chunk_size,
            self.chunk_overlap,
        )
        return chunks
