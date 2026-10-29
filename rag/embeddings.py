"""Embedding model factory (HuggingFace Sentence Transformers).

``all-MiniLM-L6-v2`` produces 384-dimensional vectors.  It is small, fast and
good enough for semantic search over a single exam syllabus.  Vectors are L2
normalised so FAISS' inner-product/L2 distance maps directly to cosine
similarity.
"""

from __future__ import annotations

from typing import Any

from config import Settings
from utils.device import resolve_device
from utils.logging_utils import get_logger

__all__ = ["build_embeddings", "EMBEDDING_DIMENSION"]

logger = get_logger(__name__)

# Dimensionality of ``sentence-transformers/all-MiniLM-L6-v2``.
EMBEDDING_DIMENSION: int = 384


def _import_embeddings_class() -> type:
    """Import ``HuggingFaceEmbeddings`` from the modern or legacy package."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings
    except ImportError:  # pragma: no cover - legacy langchain installs
        from langchain_community.embeddings import HuggingFaceEmbeddings

        logger.warning(
            "langchain-huggingface not installed; falling back to the deprecated "
            "langchain_community embeddings. Install 'langchain-huggingface'."
        )
        return HuggingFaceEmbeddings


def build_embeddings(settings: Settings) -> Any:
    """Build the embedding client; the model is loaded on first use."""
    embeddings_cls = _import_embeddings_class()
    device = resolve_device(settings.embedding_device)
    model_kwargs: dict[str, Any] = {"device": device}
    if settings.hf_token:
        model_kwargs["token"] = settings.hf_token

    logger.info(
        "Configuring embeddings model=%s device=%s normalize=%s",
        settings.embedding_model,
        device,
        settings.normalize_embeddings,
    )
    return embeddings_cls(
        model_name=settings.embedding_model,
        model_kwargs=model_kwargs,
        encode_kwargs={
            "normalize_embeddings": settings.normalize_embeddings,
            "batch_size": settings.embedding_batch_size,
        },
    )
