"""Load NEET study material from ``data/`` into LangChain documents.

Design notes
------------
* PDFs are parsed with **PyPDF** page by page so each :class:`Document` maps to
  exactly one page - this gives precise page citations in the answer.
* The subject is inferred from the folder layout first
  (``data/biology/*.pdf`` -> ``biology``), then from the filename, and finally
  falls back to ``general``.
* Every page carries a stable ``doc_id`` derived from source + page + content
  hash, which makes re-ingestion idempotent (``--append`` mode).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from langchain_core.documents import Document
from pypdf import PdfReader

from config import Settings
from rag.ncert import build_ncert_metadata
from utils.constants import Subject, subject_from_text
from utils.logging_utils import get_logger
from utils.text_utils import clean_text, hash_text

__all__ = ["PDFDocumentLoader", "SUPPORTED_SUFFIXES", "collect_ncert_chapters"]

logger = get_logger(__name__)

SUPPORTED_SUFFIXES: tuple[str, ...] = (".pdf", ".txt", ".md")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


class PDFDocumentLoader:
    """Discovers and loads study material with rich, citation-ready metadata."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.data_dir = Path(settings.data_dir)
        #: Per-file problems recorded by the most recent :meth:`load_all`.
        self.failures: list[dict[str, str]] = []

    # ------------------------------------------------------------------ public
    def discover(self) -> list[Path]:
        """Return every supported file below ``data_dir`` (deterministic order)."""
        if not self.data_dir.exists():
            logger.warning("Data directory does not exist: %s", self.data_dir)
            return []
        files = [
            path
            for path in self.data_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        ]
        files.sort(key=lambda item: str(item).lower())
        logger.info("Discovered %d source file(s) in %s", len(files), self.data_dir)
        return files

    def load_all(self) -> list[Document]:
        """Load every discovered file, skipping (and logging) failures."""
        documents: list[Document] = []
        self.failures = []
        for path in self.discover():
            try:
                loaded = self.load_file(path)
                if not loaded:
                    self.failures.append({"file": path.name, "error": "no extractable text"})
                documents.extend(loaded)
            except Exception as exc:  # a single corrupt PDF must not abort ingest
                logger.error("Failed to load %s: %s", path.name, exc)
                self.failures.append({"file": path.name, "error": str(exc)})
        logger.info("Loaded %d page-level document(s) (%d failure(s))", len(documents), len(self.failures))
        return documents

    def load_file(self, path: Path) -> list[Document]:
        """Load a single PDF/TXT/MD file into page-level documents."""
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            documents = self._load_pdf(path)
        elif suffix in {".txt", ".md"}:
            documents = self._load_text(path)
        else:
            logger.debug("Skipping unsupported file: %s", path)
            return []

        subject = self._infer_subject(path)
        # NCERT chapter identification (subject / class / title) is derived from
        # the raw page text and the file name; ``None`` keeps non-NCERT files
        # on their original, unchanged metadata path.
        raw_texts = [clean_text(document.page_content) for document in documents]
        ncert = build_ncert_metadata(path, raw_texts, subject_hint=subject)
        enriched: list[Document] = []
        for document in documents:
            content = clean_text(document.page_content)
            if len(content) < self.settings.min_page_chars:
                logger.debug("Skipping near-empty page in %s", path.name)
                continue
            metadata = {
                **document.metadata,
                "subject": subject.value,
                "file_name": path.name,
                "file_stem": path.stem,
                "file_type": suffix.lstrip("."),
                "doc_id": hash_text(f"{metadata_source(path, self.data_dir)}|{document.metadata.get('page')}|{content}"),
                "char_count": len(content),
            }
            if ncert is not None:
                metadata.update(ncert.to_document_metadata())
            enriched.append(Document(page_content=content, metadata=metadata))

        if ncert is not None:
            logger.info(
                "Loaded %-45s pages=%d subject=%s ncert=[class=%s chapter=%s]",
                path.name, len(enriched), subject.value, ncert.klass, ncert.chapter,
            )
        else:
            logger.info("Loaded %-45s pages=%d subject=%s", path.name, len(enriched), subject.value)
        return enriched

    # ------------------------------------------------------------------ loaders
    def _load_pdf(self, path: Path) -> list[Document]:
        """Parse a PDF page by page with PyPDF."""
        reader = PdfReader(str(path))
        if getattr(reader, "is_encrypted", False):
            # Many textbook PDFs are "encrypted" with an empty user password.
            try:
                reader.decrypt("")
            except Exception as exc:
                raise ValueError(f"encrypted PDF cannot be opened: {exc}") from exc

        relative_source = metadata_source(path, self.data_dir)
        documents: list[Document] = []
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                raw_text = page.extract_text() or ""
            except Exception as exc:  # malformed page inside an otherwise valid PDF
                logger.warning("Page %d of %s could not be parsed: %s", page_number, path.name, exc)
                continue
            text = _MULTI_SPACE_RE.sub(" ", clean_text(raw_text))
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": relative_source,
                        "page": page_number,
                        "total_pages": len(reader.pages),
                    },
                )
            )
        return documents

    def _load_text(self, path: Path) -> list[Document]:
        """Load plain text / markdown as a single 'page'."""
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [
            Document(
                page_content=text,
                metadata={"source": metadata_source(path, self.data_dir), "page": 1, "total_pages": 1},
            )
        ]

    # ------------------------------------------------------------------ helpers
    def _infer_subject(self, path: Path) -> Subject:
        """Infer the subject from parent folders, then from the filename."""
        try:
            parts = list(path.relative_to(self.data_dir).parts[:-1])
        except ValueError:  # path outside data_dir
            parts = list(path.parent.parts)
        for part in reversed(parts):  # closest folder wins
            subject = subject_from_text(part)
            if subject is not None:
                return subject
        return subject_from_text(path.stem) or Subject.GENERAL


def metadata_source(path: Path, data_dir: Path) -> str:
    """Return a portable, forward-slash relative source path for metadata."""
    try:
        return path.relative_to(data_dir).as_posix()
    except ValueError:
        return path.name


def group_by_subject(documents: Iterable[Document]) -> dict[str, int]:
    """Count documents per subject (used by the ingestion report)."""
    counts: dict[str, int] = {}
    for document in documents:
        subject: Any = document.metadata.get("subject", Subject.GENERAL.value)
        counts[str(subject)] = counts.get(str(subject), 0) + 1
    return counts


def collect_ncert_chapters(documents: Iterable[Document]) -> list[dict[str, Any]]:
    """Return one NCERT chapter record per source file, de-duplicated.

    Each record has the exact ingestion shape::

        {"subject": "Biology", "class": "11", "source": "NCERT",
         "chapter": "<title>", "file": "<pdf>"}

    Files without NCERT metadata are ignored, so a mixed corpus reports only its
    recognised chapters.
    """
    chapters: dict[str, dict[str, Any]] = {}
    for document in documents:
        metadata = document.metadata or {}
        if not metadata.get("ncert"):
            continue
        file_name = str(metadata.get("file_name") or metadata.get("source") or "")
        if not file_name or file_name in chapters:
            continue
        chapters[file_name] = {
            "subject": str(metadata.get("ncert_subject") or "").capitalize(),
            "class": str(metadata.get("class", "")),
            "source": str(metadata.get("ncert_source", "NCERT")),
            "chapter": str(metadata.get("chapter", "")),
            "file": file_name,
        }
    return sorted(chapters.values(), key=lambda item: (item["subject"], item["class"], item["file"]))
