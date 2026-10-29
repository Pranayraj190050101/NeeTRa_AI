"""Tests for the FAISS vector store (build, persist, reload, filter, stats)."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from config import Settings
from rag.vector_store import (
    VectorStoreManager,
    VectorStoreNotReadyError,
    read_index_metadata,
)


def test_build_persists_and_reloads(
    settings: Settings,
    vector_store: VectorStoreManager,
    fake_embeddings: DeterministicFakeEmbedding,
) -> None:
    assert vector_store.is_ready()
    assert (settings.index_dir / "index.faiss").exists()
    assert (settings.index_dir / "index.pkl").exists()

    # A brand new manager must read the same index back from disk.
    reloaded = VectorStoreManager(settings, embeddings_factory=lambda: fake_embeddings)
    stats = reloaded.stats()
    assert stats.chunks == 3
    assert stats.files == 3
    assert stats.subjects == {"biology": 1, "chemistry": 1, "physics": 1}


def test_search_returns_scored_chunks(vector_store: VectorStoreManager) -> None:
    chunks = vector_store.search("Newton's second law", k=2)
    assert 1 <= len(chunks) <= 2
    assert all(0.0 <= chunk.score <= 1.0 for chunk in chunks)
    assert all(chunk.file_name.endswith(".pdf") for chunk in chunks)
    # Scores are sorted descending.
    assert chunks == sorted(chunks, key=lambda item: item.score, reverse=True)


def test_search_respects_subject_filter(vector_store: VectorStoreManager) -> None:
    chunks = vector_store.search("mole concept", k=3, subject="biology")
    assert chunks, "expected a fallback hit inside the biology filter"
    assert {chunk.subject for chunk in chunks} == {"biology"}


def test_citation_label_contains_page(vector_store: VectorStoreManager) -> None:
    chunk = vector_store.search("photosynthesis", k=1)[0]
    label = chunk.citation_label(1)
    assert label.startswith("[1] ")
    assert ".pdf" in label


def test_stats_and_document_ids(vector_store: VectorStoreManager) -> None:
    assert vector_store.document_ids() == {
        "doc-physics-1",
        "doc-biology-1",
        "doc-chemistry-1",
    }


def test_metadata_sidecar(vector_store: VectorStoreManager, settings: Settings) -> None:
    metadata = read_index_metadata(settings)
    assert metadata is not None
    assert metadata["chunks"] == 3
    assert metadata["subjects"]["physics"] == 1
    assert metadata["embedding_model"] == settings.embedding_model


def test_missing_index_raises(settings: Settings, fake_embeddings: DeterministicFakeEmbedding) -> None:
    empty = VectorStoreManager(settings, embeddings_factory=lambda: fake_embeddings)
    assert empty.is_ready() is False
    assert empty.stats().chunks == 0
    try:
        empty.search("anything")
    except VectorStoreNotReadyError:
        pass
    else:  # pragma: no cover - the assertion documents the contract
        raise AssertionError("search() must raise VectorStoreNotReadyError without an index")


def test_add_documents_is_incremental(
    vector_store: VectorStoreManager,
) -> None:
    extra = Document(
        page_content="Ohm's law states that V = I R at constant temperature.",
        metadata={
            "source": "physics/current_electricity.pdf",
            "file_name": "current_electricity.pdf",
            "subject": "physics",
            "page": 3,
            "doc_id": "doc-physics-2",
            "chunk_id": "doc-physics-2-c000",
        },
    )
    added = vector_store.add([extra], persist=True)
    assert added == 1
    assert vector_store.stats().chunks == 4
    assert "doc-physics-2" in vector_store.document_ids()
