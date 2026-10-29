"""FAISS vector store management: build, persist, reload and search.

The store lives on disk at ``vectordb/<vector_store_name>/`` (``index.faiss`` +
``index.pkl``) and is loaded lazily, so importing this module never downloads
an embedding model.  A JSON sidecar (``vectordb/<name>.meta.json``) records
index statistics that the UI can display without touching the heavy model.
"""

from __future__ import annotations

import inspect
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config import Settings
from utils.logging_utils import get_logger

__all__ = [
    "VectorStoreManager",
    "RetrievedChunk",
    "IndexStats",
    "VectorStoreError",
    "VectorStoreNotReadyError",
    "read_index_metadata",
    "write_index_metadata",
]

logger = get_logger(__name__)

INDEX_NAME = "index"
EmbeddingsFactory = Callable[[], Embeddings]


def _deserialize_flag_kwarg(load_local: Callable[..., Any], flag: bool) -> dict[str, bool]:
    """Return the pickle-trust keyword used by the installed LangChain build.

    ``langchain-community`` <= 0.3 spells it ``allow_dangerous_deserialize``;
    >= 0.4 renamed it to ``allow_dangerous_deserialization``.  Detecting the
    parameter keeps the project working across both releases.
    """
    try:
        parameters = inspect.signature(load_local).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        parameters = {}
    name = (
        "allow_dangerous_deserialization"
        if "allow_dangerous_deserialization" in parameters
        else "allow_dangerous_deserialize"
    )
    return {name: bool(flag)}


class VectorStoreError(RuntimeError):
    """Raised when the vector store cannot be built, loaded or queried."""


class VectorStoreNotReadyError(VectorStoreError):
    """Raised when a search is attempted before ingestion has run."""


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A single retrieved passage with a calibrated similarity score."""

    content: str
    metadata: dict[str, Any]
    score: float

    @property
    def subject(self) -> str:
        return str(self.metadata.get("subject", "general"))

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", "unknown"))

    @property
    def file_name(self) -> str:
        return str(self.metadata.get("file_name") or Path(self.source).name)

    @property
    def page(self) -> int | None:
        page = self.metadata.get("page")
        return int(page) if isinstance(page, (int, float, str)) and str(page).isdigit() else None

    def citation_label(self, index: int) -> str:
        """Compact ``[1] file.pdf p.12`` label used in prompts and the UI."""
        page = f" p.{self.page}" if self.page else ""
        return f"[{index}] {self.file_name}{page}"


@dataclass(slots=True)
class IndexStats:
    """Aggregated view of the persisted index."""

    chunks: int = 0
    files: int = 0
    subjects: dict[str, int] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunks": self.chunks,
            "files": self.files,
            "subjects": self.subjects,
            "sources": self.sources,
        }

    @property
    def is_empty(self) -> bool:
        return self.chunks == 0


class VectorStoreManager:
    """Thread-safe owner of the FAISS index and its embedding model."""

    def __init__(
        self,
        settings: Settings,
        *,
        embeddings_factory: EmbeddingsFactory | None = None,
    ) -> None:
        self.settings = settings
        self._embeddings_factory = embeddings_factory
        self._embeddings: Embeddings | None = None
        self._store: Any | None = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ paths
    @property
    def index_dir(self) -> Path:
        return self.settings.index_dir

    @property
    def index_file(self) -> Path:
        return self.index_dir / f"{INDEX_NAME}.faiss"

    @property
    def metadata_file(self) -> Path:
        return self.settings.index_metadata_path

    def is_ready(self) -> bool:
        """True when a persisted index exists on disk."""
        return self.index_file.exists() and (self.index_dir / f"{INDEX_NAME}.pkl").exists()

    # ------------------------------------------------------------- embeddings
    @property
    def embeddings(self) -> Embeddings:
        """Lazily create (and cache) the embedding client."""
        with self._lock:
            if self._embeddings is None:
                if self._embeddings_factory is None:
                    from rag.embeddings import build_embeddings

                    self._embeddings_factory = lambda: build_embeddings(self.settings)
                self._embeddings = self._embeddings_factory()
            return self._embeddings

    def set_embeddings(self, embeddings: Embeddings) -> None:
        """Inject a pre-built embedding client (used by tests)."""
        with self._lock:
            self._embeddings = embeddings

    # ------------------------------------------------------------------ store
    def load(self, *, force: bool = False) -> Any:
        """Load the FAISS index from disk (cached in memory)."""
        with self._lock:
            if self._store is not None and not force:
                return self._store
            if not self.is_ready():
                raise VectorStoreNotReadyError(
                    f"No FAISS index found at {self.index_dir}. "
                    "Run 'python -m rag.ingest --rebuild' first."
                )
            try:
                from langchain_community.vectorstores import FAISS

                started = datetime.now(timezone.utc)
                self._store = FAISS.load_local(
                    folder_path=str(self.index_dir),
                    embeddings=self.embeddings,
                    index_name=INDEX_NAME,
                    **_deserialize_flag_kwarg(FAISS.load_local, self.settings.faiss_allow_dangerous_deserialize),
                )
                elapsed = (datetime.now(timezone.utc) - started).total_seconds()
                logger.info(
                    "Loaded FAISS index from %s (vectors=%s) in %.2fs",
                    self.index_dir,
                    self._safe_total(),
                    elapsed,
                )
            except VectorStoreError:
                raise
            except Exception as exc:
                raise VectorStoreError(f"Unable to load FAISS index: {exc}") from exc
            return self._store

    def reload(self) -> None:
        """Drop the in-memory index so the next access re-reads disk."""
        with self._lock:
            self._store = None
            logger.info("In-memory FAISS index cleared")

    def build(self, documents: Sequence[Document], *, persist: bool = True) -> Any:
        """Create a brand new index from ``documents`` (replacing any on disk)."""
        if not documents:
            raise VectorStoreError("Cannot build a vector store from zero documents")
        try:
            from langchain_community.vectorstores import FAISS

            logger.info("Building FAISS index from %d chunk(s)...", len(documents))
            store = FAISS.from_documents(list(documents), self.embeddings)
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Unable to build FAISS index: {exc}") from exc

        with self._lock:
            self._store = store
        if persist:
            self.save()
        return store

    def add(self, documents: Sequence[Document], *, persist: bool = True) -> int:
        """Append documents to the existing index; returns how many were added."""
        if not documents:
            return 0
        store = self.load()
        try:
            store.add_documents(list(documents))
        except Exception as exc:
            raise VectorStoreError(f"Unable to add documents to FAISS index: {exc}") from exc
        if persist:
            self.save()
        logger.info("Added %d chunk(s); index now holds %s vector(s)", len(documents), self._safe_total())
        return len(documents)

    def save(self) -> None:
        """Persist the in-memory index to ``vectordb/<name>/``."""
        with self._lock:
            if self._store is None:
                raise VectorStoreError("Nothing to save - the index is not loaded")
            try:
                self.index_dir.mkdir(parents=True, exist_ok=True)
                self._store.save_local(folder_path=str(self.index_dir), index_name=INDEX_NAME)
                logger.info("Saved FAISS index to %s (%s vectors)", self.index_dir, self._safe_total())
            except Exception as exc:
                raise VectorStoreError(f"Unable to save FAISS index: {exc}") from exc

    # ------------------------------------------------------------------ search
    def search(
        self,
        query: str,
        *,
        k: int | None = None,
        subject: str | None = None,
        score_threshold: float | None = None,
        fetch_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Return the top-k passages, optionally restricted to one subject.

        Args:
            query: Student question or re-written search query.
            k: Number of chunks to return.
            subject: ``physics`` / ``chemistry`` / ``biology``; ``None`` or
                ``general`` searches the whole index.
            score_threshold: Minimum cosine similarity in ``[0, 1]``.
            fetch_k: Candidate pool size used for metadata filtering.
        """
        query = (query or "").strip()
        if not query:
            return []

        store = self.load()
        top_k = int(k or self.settings.top_k)
        threshold = (
            self.settings.score_threshold if score_threshold is None else float(score_threshold)
        )
        candidates = int(fetch_k or max(self.settings.fetch_k, top_k * 4))
        filter_subject = None if not subject or subject == "general" else subject

        pairs: list[tuple[Document, float]]
        try:
            if filter_subject:
                pairs = store.similarity_search_with_score(
                    query,
                    k=top_k,
                    filter={"subject": filter_subject},
                    fetch_k=candidates,
                )
            else:
                pairs = store.similarity_search_with_score(query, k=top_k)
        except TypeError:
            # Older LangChain builds without the ``filter`` keyword.
            logger.debug("Filtered FAISS search unsupported; filtering in Python")
            pairs = store.similarity_search_with_score(query, k=candidates)
            if filter_subject:
                pairs = [
                    (doc, score)
                    for doc, score in pairs
                    if doc.metadata.get("subject") == filter_subject
                ][:top_k]
        except Exception as exc:
            raise VectorStoreError(f"FAISS search failed: {exc}") from exc

        chunks: list[RetrievedChunk] = []
        seen: set[str] = set()
        for document, distance in pairs:
            similarity = self._to_similarity(float(distance))
            if similarity < threshold:
                continue
            key = str(document.metadata.get("chunk_id") or document.page_content[:120])
            if key in seen:
                continue
            seen.add(key)
            chunks.append(
                RetrievedChunk(
                    content=document.page_content,
                    metadata=dict(document.metadata),
                    score=round(similarity, 4),
                )
            )

        chunks.sort(key=lambda item: item.score, reverse=True)
        logger.info(
            "Retrieved %d/%d chunk(s) for subject=%s (best=%s)",
            len(chunks),
            len(pairs),
            filter_subject or "all",
            f"{chunks[0].score:.3f}" if chunks else "n/a",
        )
        return chunks

    # ------------------------------------------------------------------- stats
    def stats(self) -> IndexStats:
        """Aggregate counts per subject/file from the loaded docstore."""
        try:
            store = self.load()
        except VectorStoreNotReadyError:
            return IndexStats()

        documents = list(getattr(store.docstore, "_dict", {}).values())
        subjects: dict[str, int] = {}
        sources: set[str] = set()
        for document in documents:
            metadata = getattr(document, "metadata", {}) or {}
            subject = str(metadata.get("subject", "general"))
            subjects[subject] = subjects.get(subject, 0) + 1
            sources.add(str(metadata.get("source") or metadata.get("file_name") or "unknown"))
        return IndexStats(
            chunks=len(documents),
            files=len(sources),
            subjects=dict(sorted(subjects.items())),
            sources=sorted(sources),
        )

    def document_ids(self) -> set[str]:
        """Existing ``doc_id`` values, used for idempotent incremental ingest."""
        try:
            store = self.load()
        except VectorStoreNotReadyError:
            return set()
        return {
            str(document.metadata.get("doc_id"))
            for document in getattr(store.docstore, "_dict", {}).values()
            if document.metadata.get("doc_id")
        }

    # ----------------------------------------------------------------- helpers
    def _to_similarity(self, distance: float) -> float:
        """Convert a FAISS distance to cosine similarity in ``[0, 1]``.

        With L2-normalised vectors the squared L2 distance ``d`` relates to the
        cosine similarity ``s`` by ``d = 2 - 2s``.  Without normalisation the
        raw distance is not comparable across queries, so it is returned as a
        monotonic fallback score.
        """
        if self.settings.normalize_embeddings:
            return max(0.0, min(1.0, 1.0 - distance / 2.0))
        return 1.0 / (1.0 + max(0.0, distance))

    def _safe_total(self) -> int | str:
        try:
            return int(self._store.index.ntotal)  # type: ignore[union-attr]
        except Exception:
            return "?"


# ---------------------------------------------------------------------------
# Index metadata sidecar (read by the UI without loading any model)
# ---------------------------------------------------------------------------
def write_index_metadata(
    settings: Settings,
    *,
    chunks: int,
    files: int,
    subjects: dict[str, int],
    extra: dict[str, Any] | None = None,
) -> Path:
    """Persist a small JSON summary next to the FAISS files."""
    payload: dict[str, Any] = {
        "index_name": settings.vector_store_name,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "chunks": chunks,
        "files": files,
        "subjects": subjects,
    }
    if extra:
        payload.update(extra)
    path = settings.index_metadata_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote index metadata to %s", path)
    return path


def read_index_metadata(settings: Settings) -> dict[str, Any] | None:
    """Read the sidecar JSON, returning ``None`` when it does not exist."""
    path = settings.index_metadata_path
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Unable to read index metadata %s: %s", path, exc)
        return None


def summarise_chunks(documents: Iterable[Document]) -> dict[str, int]:
    """Count chunks per subject for the ingestion report."""
    counts: dict[str, int] = {}
    for document in documents:
        subject = str(document.metadata.get("subject", "general"))
        counts[subject] = counts.get(subject, 0) + 1
    return dict(sorted(counts.items()))
