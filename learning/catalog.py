"""Chapter catalog derived from the FAISS index - never hardcoded.

The platform needs a subject -> chapter hierarchy for the Learn module, the
Practice module's chapter tests and the Revision Hub's target picker.  The
only honest source of that hierarchy is the student's own material: whatever
PDFs were ingested into the index.  This module turns the index statistics
(one source file per textbook chapter in the NEET folder layout) into a
browseable catalog, and degrades to an empty catalog - with a clear message -
when nothing is indexed yet.

Nothing here imports FAISS or torch, so the catalog stays cheap to build in
tests and in ``scripts/offline_check.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from utils.constants import Subject, normalise_token, subject_from_text

__all__ = ["Chapter", "ChapterCatalog", "humanise"]

#: Display order for the three NEET subjects; anything else follows.
_SUBJECT_ORDER: tuple[str, ...] = (
    Subject.PHYSICS.value,
    Subject.CHEMISTRY.value,
    Subject.BIOLOGY.value,
)

_EXTENSION_RE = re.compile(r"\.(pdf|txt|md|docx?|pptx?|html?)$", re.IGNORECASE)
_LEADING_NUMBER_RE = re.compile(r"^(?:chapter[\s_-]*)?\d{1,3}[\s_.-]+", re.IGNORECASE)
_TRAILING_NOISE_RE = re.compile(r"[\s_-]*(ncert|class\s*1[12]|part\s*\d+)[\s_-]*$", re.IGNORECASE)
_MULTI_SPACE_RE = re.compile(r"\s{2,}")


def _stem(source: str) -> str:
    """File name without its extension, for use as a chapter label."""
    return _EXTENSION_RE.sub("", Path(str(source).replace("\\", "/")).name).strip()


#: Acronyms that must stay capitalised in a chapter title.
_ACRONYMS: frozenset[str] = frozenset(
    {
        "ph",
        "dna",
        "rna",
        "atp",
        "adp",
        "nadh",
        "nhp",
        "shm",
        "si",
        "rc",
        "lcr",
        "dcp",
        "pdb",
        "pcb",
        "ncert",
        "mcq",
        "pyq",
        "ocr",
        "cct",
        "cvp",
    }
)


def humanise(stem: str) -> str:
    """``04_moving_charges_and_magnetism`` -> ``Moving Charges And Magnetism``."""
    text = _EXTENSION_RE.sub("", stem)
    text = text.replace("_", " ").replace("-", " ").replace(".", " ")
    text = _LEADING_NUMBER_RE.sub("", text)
    text = _TRAILING_NOISE_RE.sub("", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    if not text:
        return stem.strip() or "Untitled chapter"
    words: list[str] = []
    for word in text.split(" "):
        if word.lower() in _ACRONYMS:
            words.append(word.upper())
        else:
            words.append(word[:1].upper() + word[1:])
    return " ".join(words)


@dataclass(frozen=True, slots=True)
class Chapter:
    """One indexed unit of study material (normally one PDF)."""

    subject: str
    name: str
    source: str
    file_name: str
    chunks: int = 0

    @property
    def key(self) -> str:
        """Stable, comparable identifier (the indexed source path)."""
        return str(self.source)

    @property
    def label(self) -> str:
        """UI caption: ``Physics - Moving Charges And Magnetism``."""
        return f"{self.subject.title()} - {self.name}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "name": self.name,
            "source": self.source,
            "file_name": self.file_name,
            "chunks": self.chunks,
        }


@dataclass(slots=True)
class ChapterCatalog:
    """Subject -> chapter hierarchy built from the index sources."""

    chapters: list[Chapter] = field(default_factory=list)
    subject_chunks: dict[str, int] = field(default_factory=dict)

    # ------------------------------------------------------------------ build
    @classmethod
    def empty(cls) -> "ChapterCatalog":
        return cls()

    @classmethod
    def from_sources(
        cls,
        sources: Iterable[str],
        *,
        subject_chunks: Mapping[str, int] | None = None,
        chunk_counts: Mapping[str, int] | None = None,
    ) -> "ChapterCatalog":
        """Build a catalog from indexed source paths.

        The subject comes from the folder the file was ingested from (the
        ``data/<subject>/`` layout), falling back to keyword inference.
        """
        items: list[Chapter] = []
        seen: set[str] = set()
        for raw in sources:
            source = str(raw or "").replace("\\", "/").strip()
            if not source or source in seen or source == "unknown":
                continue
            seen.add(source)
            items.append(
                Chapter(
                    subject=_subject_of(source),
                    name=humanise(_stem(source)),
                    source=source,
                    file_name=Path(source).name,
                    chunks=int((chunk_counts or {}).get(source, 0)),
                )
            )
        items.sort(key=lambda chapter: (_SUBJECT_ORDER.index(chapter.subject)
                                       if chapter.subject in _SUBJECT_ORDER else len(_SUBJECT_ORDER),
                                        chapter.name.lower()))
        return cls(chapters=items, subject_chunks=dict(subject_chunks or {}))

    @classmethod
    def from_manager(cls, manager: Any) -> "ChapterCatalog":
        """Read the live index statistics (returns empty when unavailable)."""
        try:
            stats = manager.stats()
        except Exception:  # pragma: no cover - depends on the environment
            return cls.empty()
        payload = stats.as_dict() if hasattr(stats, "as_dict") else dict(stats or {})
        return cls.from_sources(
            payload.get("sources") or [],
            subject_chunks=payload.get("subjects") or {},
        )

    @classmethod
    def from_metadata(cls, payload: Mapping[str, Any] | None) -> "ChapterCatalog | None":
        """Build from the index sidecar JSON, so no FAISS load is needed.

        Returns ``None`` when the payload predates the ``sources`` key, which is
        the caller's signal to fall back to :meth:`from_manager`.
        """
        if not isinstance(payload, dict):
            return None
        sources = payload.get("sources")
        if not isinstance(sources, list) or not sources:
            return None
        return cls.from_sources(sources, subject_chunks=payload.get("subjects") or {})

    # ------------------------------------------------------------------- query
    @property
    def is_empty(self) -> bool:
        return not self.chapters

    def __bool__(self) -> bool:  # pragma: no cover - readability helper
        return bool(self.chapters)

    @property
    def subjects(self) -> tuple[str, ...]:
        """Subjects present in the index, in NEET order."""
        present = {chapter.subject for chapter in self.chapters}
        ordered = [subject for subject in _SUBJECT_ORDER if subject in present]
        extra = sorted(present - set(_SUBJECT_ORDER))
        return tuple(ordered + extra)

    def list(self, subject: str | None = None) -> list[Chapter]:
        """Chapters of one subject (all chapters when ``subject`` is falsy)."""
        if not subject or subject == "all":
            return list(self.chapters)
        wanted = normalise_token(str(subject))
        return [chapter for chapter in self.chapters if chapter.subject == wanted]

    # ``chapters`` is the dataclass field, so expose a read-only alias too.
    def of(self, subject: str | None = None) -> list[Chapter]:
        """Alias of :meth:`list` for call sites that prefer prose."""
        return self.list(subject)

    def find(self, name: str, subject: str | None = None) -> Chapter | None:
        """Locate a chapter by exact source, file name or fuzzy title."""
        needle = normalise_token(_EXTENSION_RE.sub("", str(name or "")))
        if not needle:
            return None
        pool = self.list(subject)
        for chapter in pool:
            if chapter.source == name or chapter.file_name == name:
                return chapter
        for chapter in pool:
            haystacks = (normalise_token(chapter.name), normalise_token(_stem(chapter.source)))
            if needle in haystacks or any(needle in value or value in needle for value in haystacks):
                return chapter
        tokens = set(needle.split())
        scored = [
            (
                len(tokens & (set(normalise_token(chapter.name).split()) | set(_stem(chapter.source).split()))),
                chapter,
            )
            for chapter in pool
        ]
        scored = [item for item in scored if item[0]]
        if not scored:
            return None
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]

    def names(self, subject: str | None = None) -> list[str]:
        """Chapter titles for a Gradio dropdown."""
        return [chapter.name for chapter in self.list(subject)]

    def size_of(self, subject: str) -> int:
        """Indexed chunks for one subject (from the index stats)."""
        return int(self.subject_chunks.get(subject, 0))

    def count_for(self, subject: str) -> int:
        """Indexed chapters for one subject."""
        return len(self.list(subject))

    def summary(self) -> dict[str, Any]:
        return {
            "chapters": len(self.chapters),
            "subjects": {
                subject: {
                    "chapters": self.count_for(subject),
                    "chunks": self.size_of(subject),
                }
                for subject in self.subjects
            },
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "empty": self.is_empty,
            "subjects": list(self.subjects),
            "summary": self.summary(),
            "chapters": [chapter.as_dict() for chapter in self.chapters],
        }


def _subject_of(source: str) -> str:
    """Folder first, keyword second, ``general`` last."""
    parts = [part for part in str(source).split("/") if part]
    for part in parts[:-1]:
        candidate = normalise_token(part)
        if candidate in ("physics", "chemistry", "biology"):
            return candidate
        if candidate in ("phy", "chem", "bio"):
            return {"phy": "physics", "chem": "chemistry", "bio": "biology"}[candidate]
    inferred = subject_from_text(source)
    return inferred.value if inferred else Subject.GENERAL.value


def chapter_labels(chapters) -> list[str]:  # pragma: no cover - UI convenience
    """Titles of ``chapters`` in display order."""
    return [chapter.name for chapter in chapters]
