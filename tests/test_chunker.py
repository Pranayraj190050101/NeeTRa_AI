"""Tests for the page -> chunk transformation."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from config import Settings
from rag.chunker import DocumentChunker

LONG_PAGE = "\n\n".join(
    f"Paragraph {index}. " + ("Photosynthesis converts light energy into chemical energy. " * 6)
    for index in range(1, 6)
)


def _page(text: str = LONG_PAGE) -> Document:
    return Document(
        page_content=text,
        metadata={"source": "biology/plant_physiology.pdf", "subject": "biology", "page": 45, "doc_id": "doc-bio-1"},
    )


def test_split_respects_chunk_size_and_overlap(settings: Settings) -> None:
    chunker = DocumentChunker(settings, chunk_size=300, chunk_overlap=50)
    chunks = chunker.split([_page()])

    assert len(chunks) > 1
    for chunk in chunks:
        # RecursiveCharacterTextSplitter may overshoot slightly on atomic
        # separators, but never beyond chunk_size + overlap.
        assert len(chunk.page_content) <= 300 + 50


def test_split_preserves_and_extends_metadata(settings: Settings) -> None:
    chunker = DocumentChunker(settings, chunk_size=300, chunk_overlap=50)
    chunks = chunker.split([_page()])

    first = chunks[0]
    assert first.metadata["subject"] == "biology"
    assert first.metadata["page"] == 45
    assert first.metadata["chunk_index"] == 0
    assert first.metadata["chunk_id"] == "doc-bio-1-c000"
    assert first.metadata["chunk_chars"] == len(first.page_content)
    assert chunks[1].metadata["chunk_index"] == 1


def test_split_drops_fragments_below_min_chars(settings: Settings) -> None:
    settings = settings.model_copy(update={"min_chunk_chars": 100})
    chunker = DocumentChunker(settings, chunk_size=300, chunk_overlap=0)
    # The trailing sentence is deliberately shorter than min_chunk_chars.
    page = _page(("A" * 280) + "\n\nshort tail fragment.")

    chunks = chunker.split([page])

    assert all(len(chunk.page_content) >= 100 for chunk in chunks)


def test_split_skips_empty_documents(settings: Settings) -> None:
    chunker = DocumentChunker(settings)

    assert chunker.split([_page("   \n\n   ")]) == []


def test_invalid_overlap_is_rejected(settings: Settings) -> None:
    with pytest.raises(ValueError, match="must be smaller"):
        DocumentChunker(settings, chunk_size=400, chunk_overlap=400)
