"""Shared pytest fixtures.

The fixtures deliberately avoid downloading models: FAISS is exercised with
``DeterministicFakeEmbedding`` and the LLM with ``MockLLM``, so the whole suite
runs offline in a couple of seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Settings  # noqa: E402
from rag.vector_store import VectorStoreManager, write_index_metadata  # noqa: E402

EMBEDDING_SIZE = 384


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Isolated settings pointing at temporary directories."""
    return Settings(
        data_dir=tmp_path / "data",
        vector_store_dir=tmp_path / "vectordb",
        log_dir=tmp_path / "logs",
        progress_dir=tmp_path / "progress",
        prompt_dir=PROJECT_ROOT / "prompts",
        log_to_file=False,
        llm_backend="mock",
        # Pin the provider layer so an ambient LLM_PROVIDER/GROQ_* in the
        # developer's shell can never leak into the offline suite.
        llm_provider="mock",
        groq_api_key=None,
        top_k=3,
        score_threshold=0.0,
        stream_tokens=False,
    )


@pytest.fixture
def fake_embeddings() -> DeterministicFakeEmbedding:
    return DeterministicFakeEmbedding(size=EMBEDDING_SIZE)


@pytest.fixture
def sample_documents() -> list[Document]:
    """Three passages across two subjects with full metadata."""
    return [
        Document(
            page_content=(
                "Newton's second law states that the rate of change of momentum of a body is "
                "directly proportional to the applied force and takes place in the direction of "
                "the force. F = m * a."
            ),
            metadata={
                "source": "physics/mechanics.pdf",
                "file_name": "mechanics.pdf",
                "subject": "physics",
                "page": 12,
                "doc_id": "doc-physics-1",
                "chunk_id": "doc-physics-1-c000",
            },
        ),
        Document(
            page_content=(
                "Photosynthesis occurs in the chloroplast. Light reactions take place in the "
                "thylakoid membranes and produce ATP and NADPH, which drive the Calvin cycle in "
                "the stroma."
            ),
            metadata={
                "source": "biology/plant_physiology.pdf",
                "file_name": "plant_physiology.pdf",
                "subject": "biology",
                "page": 45,
                "doc_id": "doc-biology-1",
                "chunk_id": "doc-biology-1-c000",
            },
        ),
        Document(
            page_content=(
                "The mole is the amount of substance containing 6.022 x 10^23 entities. Molar mass "
                "is the mass of one mole of a substance expressed in grams per mole."
            ),
            metadata={
                "source": "chemistry/mole_concept.pdf",
                "file_name": "mole_concept.pdf",
                "subject": "chemistry",
                "page": 7,
                "doc_id": "doc-chemistry-1",
                "chunk_id": "doc-chemistry-1-c000",
            },
        ),
    ]


@pytest.fixture
def vector_store(
    settings: Settings,
    fake_embeddings: DeterministicFakeEmbedding,
    sample_documents: list[Document],
) -> Iterator[VectorStoreManager]:
    """A persisted FAISS index backed by fake embeddings."""
    store = VectorStoreManager(settings, embeddings_factory=lambda: fake_embeddings)
    store.build(sample_documents, persist=True)
    write_index_metadata(
        settings,
        chunks=len(sample_documents),
        files=3,
        subjects={"physics": 1, "biology": 1, "chemistry": 1},
        extra={
            # What ingestion records in production: the chapter catalog is read
            # from this sidecar instead of loading the FAISS index.
            "sources": [str(document.metadata["source"]) for document in sample_documents],
        },
    )
    yield store
