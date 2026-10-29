"""NCERT-aware metadata extraction for chapter-wise textbook PDFs.

The rest of the ingestion stack (:mod:`rag.document_loader`,
:mod:`rag.ingest`) already turns *any* PDF into page-level LangChain documents
and persists them through the FAISS :class:`~rag.vector_store.VectorStoreManager`.
This module adds the one thing a NEET aspirant's library needs but a generic
loader cannot know: **which NCERT chapter a PDF actually is**.

Given a file such as ``data/biology/kebo101.pdf`` it derives::

    {
      "subject": "Biology",
      "class":   "11",
      "source":  "NCERT",
      "chapter": "The Living World",   # from the PDF text, else a code map
      "file":    "kebo101.pdf"
    }

Everything here is **pure and offline**: it uses only the standard library plus
the shared :mod:`utils.constants` vocabulary.  It never imports pypdf, torch or
LangChain, so it is cheap to unit-test and safe to call from
``scripts/offline_check.py``.  No OCR - the chapter title is read from the PDF's
existing text layer (already extracted by the loader and passed in as strings).

Two independent signals are combined, content first, filename second:

* **filename code** - NCERT's own e-book scheme (``kebo101`` = Class **11**
  **B**i**o**logy chapter **01**; ``cebo201`` = Class 12).  Gives subject, class
  and chapter *number* even when the text layer is sparse.
* **page text** - the chapter opener (``Chapter 1: The Living World``) gives the
  human title.  A small built-in catalogue is the fallback for the Biology
  chapters students actually upload.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from utils.constants import Subject, subject_from_text

__all__ = [
    "NcertMetadata",
    "SOURCE_NAME",
    "parse_ncert_filename",
    "extract_chapter_title",
    "build_ncert_metadata",
    "is_ncert_file",
]

#: Publisher recorded for every chapter that carries NCERT metadata.
SOURCE_NAME = "NCERT"

# --- filename code grammar -------------------------------------------------
# NCERT e-book codes look like:  [class][medium][subject][NNN]
#   class  : k = Class XI, c = Class XII
#   medium : e = English, h = Hindi (we keep it but do not branch on it)
#   subject: ph = Physics, ch = Chemistry, bo = Biology
#   NNN    : leading digit(s) = volume/part, last two digits = chapter number
_CODE_RE = re.compile(
    r"^(?P<class>[kc])(?P<medium>[eh])(?P<subject>ph|ch|bo|bi|zo)(?P<number>\d{3,4})$",
    re.IGNORECASE,
)
_CLASS_LETTER: dict[str, str] = {"k": "11", "c": "12"}
_SUBJECT_CODE: dict[str, Subject] = {
    "ph": Subject.PHYSICS,
    "ch": Subject.CHEMISTRY,
    "bo": Subject.BIOLOGY,
    "bi": Subject.BIOLOGY,
    "zo": Subject.BIOLOGY,
}

# --- descriptive names (a friendlier alternative to the raw codes) ---------
# e.g. "Class-11-Biology-Chapter-1-The-Living-World", "01_The_Living_World"
_DESC_SUBJECT_RE = re.compile(r"(physics|chemistry|biology|botany|zoology)", re.IGNORECASE)
_DESC_CLASS_RE = re.compile(r"(?:class[\s_-]*|xi|xiv)?(\b1[12]\b)|(?:class[\s_-]*)(\d{1,2})", re.IGNORECASE)
_DESC_CHAPTER_RE = re.compile(r"chapter[\s_-]*(\d{1,2})", re.IGNORECASE)
_ANY_NUM_TRAILING_RE = re.compile(r"(\d{1,3})\s*$")

# --- title extraction from page text ---------------------------------------
# "Chapter 1: The Living World" / "Chapter 7 - Structural Organisation in Animals"
_INLINE_TITLE_RE = re.compile(
    r"chapter[\s_-]*\d{1,3}[\s]*[:\-–][\s]*([^\n]{3,80})",
    re.IGNORECASE,
)
# A "Chapter N" line whose title is on the following non-empty line.
_CHAPTER_LINE_RE = re.compile(r"^\s*chapter[\s_-]*(\d{1,3})\s*:?\s*$", re.IGNORECASE)
# A lone ALL-CAPS heading (>= 2 words) is the classic NCERT chapter opener.
_CAPS_HEADING_RE = re.compile(r"^\s*([A-Z][A-Z ,&\-()/]{6,80}[A-Z])\s*$")

# --- fallback catalogue ----------------------------------------------------
#: Rationalised NCERT Biology titles, indexed by chapter number (0 unused).
_BIOLOGY_TITLES: dict[str, list[str]] = {
    "11": [
        "",
        "The Living World",
        "Biological Classification",
        "Plant Kingdom",
        "Animal Kingdom",
        "Morphology of Flowering Plants",
        "Anatomy of Flowering Plants",
        "Structural Organisation in Animals",
        "Cell: The Unit of Life",
        "Biomolecules",
        "Cell Cycle and Cell Division",
    ],
    "12": [
        "",
        "Sexual Reproduction in Flowering Plants",
        "Human Reproduction",
        "Reproductive Health",
        "Principles of Inheritance and Variation",
        "Molecular Basis of Inheritance",
        "Genetic Information and Gene Expression",
        "Evolution",
        "Human Health and Disease",
        "Microbes in Human Welfare",
        "Biotechnology: Principles and Processes",
        "Biotechnology and Its Applications",
        "Organisms and Populations",
        "Ecosystem",
        "Biodiversity and Conservation",
    ],
}


@dataclass(frozen=True, slots=True)
class NcertMetadata:
    """The five facts the ingestion pipeline records per chapter.

    ``to_document_metadata`` flattens this into the scalar keys merged onto
    every page/chunk of the PDF (so FAISS can still filter on ``subject``);
    ``as_dict`` renders the exact shape required by the ingestion report and
    the index sidecar.
    """

    subject: str          # display form, e.g. "Biology"
    klass: str            # "11" / "12"
    chapter: str          # human title, e.g. "The Living World"
    file: str             # PDF file name
    chapter_number: int | None = None
    code: str | None = None   # raw stem, e.g. "kebo101"

    def as_dict(self) -> dict[str, object]:
        return {
            "subject": self.subject,
            "class": self.klass,
            "source": SOURCE_NAME,
            "chapter": self.chapter,
            "file": self.file,
        }

    def to_document_metadata(self) -> dict[str, object]:
        """Flat, scalar metadata safe for the existing retrieval path.

        The retrieval-critical ``subject`` (lowercase) and ``source`` (relative
        path) keys are owned by the loader and are *not* touched here; these
        ``ncert_*`` names plus ``class``/``chapter`` are additive.
        """
        return {
            "ncert": True,
            "ncert_subject": self.subject,
            "ncert_source": SOURCE_NAME,
            "class": self.klass,
            "chapter": self.chapter,
            "chapter_number": self.chapter_number,
            "book_code": self.code,
        }


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------
def parse_ncert_filename(stem: str) -> tuple[Subject, str, int | None, str | None] | None:
    """Return ``(subject, class, chapter_number, code)`` from a file stem.

    Understands both the compact NCERT codes (``kebo101``) and descriptive
    names (``Class-11-Biology-Chapter-1``).  ``None`` means "not recognisable
    as an NCERT chapter file" - a perfectly valid outcome, since the loader then
    keeps its original, non-NCERT behaviour.
    """
    text = stem.strip()
    if not text:
        return None

    code = text.lower()
    match = _CODE_RE.match(code)
    if match:
        subject = _SUBJECT_CODE[match.group("subject").lower()]
        klass = _CLASS_LETTER[match.group("class").lower()]
        digits = match.group("number")
        chapter_number = int(digits[-2:] if len(digits) >= 2 else digits)
        return subject, klass, chapter_number, text

    # Descriptive fallback: needs at least a subject to count as NCERT.
    subject = _descriptive_subject(text)
    if subject is None:
        return None
    klass = _descriptive_class(text) or _class_from_subject(subject)
    chapter_number = _descriptive_chapter(text)
    return subject, klass, chapter_number, None


def _descriptive_subject(text: str) -> Subject | None:
    found = _DESC_SUBJECT_RE.search(text)
    if found:
        return Subject.coerce(found.group(1))
    return subject_from_text(text)


def _descriptive_class(text: str) -> str | None:
    match = re.search(r"class[\s_-]*(1[12])", text, re.IGNORECASE)
    if match:
        return match.group(1)
    if re.search(r"\b11\b", text):
        return "11"
    if re.search(r"\b12\b", text):
        return "12"
    return None


def _descriptive_chapter(text: str) -> int | None:
    match = _DESC_CHAPTER_RE.search(text)
    if match:
        return int(match.group(1))
    trailing = _ANY_NUM_TRAILING_RE.search(text)
    if trailing:
        number = int(trailing.group(1))
        return number if 1 <= number <= 30 else None
    return None


def _class_from_subject(subject: Subject) -> str:  # pragma: no cover - cosmetic
    """Descriptive files without an explicit class default to Class 11."""
    return "11"


# ---------------------------------------------------------------------------
# Title extraction
# ---------------------------------------------------------------------------
def extract_chapter_title(page_texts: Sequence[str], *, subject: Subject, klass: str,
                          chapter_number: int | None) -> str | None:
    """Find the human chapter title in a PDF's first pages (text only).

    Tries, in order: an inline ``Chapter N: Title``, a ``Chapter N`` line
    followed by a title line, then a lone ALL-CAPS heading.  Returns ``None``
    when nothing convincing is present so the caller can use the code map.
    """
    window = "\n".join(page_texts[:3])
    if not window.strip():
        return None

    inline = _INLINE_TITLE_RE.search(window)
    if inline:
        title = _clean_title(inline.group(1))
        if title:
            return title

    lines = [line.strip() for line in window.splitlines()]
    for index, line in enumerate(lines):
        if _CHAPTER_LINE_RE.match(line):
            for follower in lines[index + 1: index + 4]:
                if follower and not _CHAPTER_LINE_RE.match(follower):
                    title = _clean_title(follower)
                    if title:
                        return title

    for line in lines:
        if _CHAPTER_LINE_RE.match(line) or "chapter" in line.lower():
            continue
        if _CAPS_HEADING_RE.match(line):
            title = _clean_title(line)
            if title:
                return title

    return _catalogue_title(subject, klass, chapter_number)


def _clean_title(raw: str) -> str | None:
    title = re.sub(r"\s{2,}", " ", raw).strip(" .:-–—_")
    # Drop a leading "Chapter N" if the capture accidentally kept it.
    title = re.sub(r"^chapter[\s_-]*\d{1,3}[\s:|-]*", "", title, flags=re.IGNORECASE).strip()
    if not title or len(title) < 3:
        return None
    # Preserve any explicit sub-title casing but normalise shouty ALL-CAPS.
    if title.upper() == title:
        return _smart_title(title)
    return title


_SMALL_WORDS = {"of", "and", "the", "in", "a", "an", "to", "for", "on", "with"}


def _smart_title(text: str) -> str:
    words = text.lower().split()
    out: list[str] = []
    for position, word in enumerate(words):
        if word in _SMALL_WORDS and position not in (0, len(words) - 1):
            out.append(word)
        else:
            out.append(word[:1].upper() + word[1:])
    return " ".join(out)


def _catalogue_title(subject: Subject, klass: str, chapter_number: int | None) -> str | None:
    """Biology fallback titles for the codes students actually upload."""
    if subject is not Subject.BIOLOGY or chapter_number is None:
        return None
    titles = _BIOLOGY_TITLES.get(klass, [])
    if 0 < chapter_number < len(titles):
        return titles[chapter_number]
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def is_ncert_file(path: Path) -> bool:
    """True when the file name carries an NCERT chapter code or descriptor."""
    return parse_ncert_filename(Path(path).stem) is not None


def build_ncert_metadata(
    path: Path,
    page_texts: Iterable[str],
    *,
    subject_hint: Subject | None = None,
) -> NcertMetadata | None:
    """Assemble NCERT metadata for one PDF, or ``None`` if it is not NCERT.

    Args:
        path: The PDF file (only ``.pdf`` files are considered chapters).
        page_texts: The extracted text of the file's pages, in order (already
            produced by :class:`rag.document_loader.PDFDocumentLoader`).
        subject_hint: The subject the folder layout already implied - used when
            the file name is descriptive but omits the subject.
    """
    parsed = parse_ncert_filename(Path(path).stem)
    if parsed is None:
        return None
    subject, klass, chapter_number, code = parsed
    if subject is None:
        subject = subject_hint or Subject.GENERAL

    texts = [text for text in page_texts if text]
    title = extract_chapter_title(
        texts, subject=subject, klass=klass, chapter_number=chapter_number
    )
    if not title:
        title = f"Chapter {chapter_number}" if chapter_number else Path(path).stem

    return NcertMetadata(
        subject=subject.label,           # "Biology"
        klass=klass,
        chapter=title,
        file=Path(path).name,
        chapter_number=chapter_number,
        code=code,
    )
