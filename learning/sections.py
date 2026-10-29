"""Section contracts for the platform's study material (pure standard library).

The chat answer modes (see :mod:`utils.answer_format`) already guarantee a
predictable shape for doubt answers.  The Learn and Revision Hub modules
promise their *own* shapes: a chapter page always has the same four headings,
a formula sheet always has the same three.  Students revise from these pages
later, so a section that the model skipped must be filled rather than lost.

:func:`stamp_sections` is the platform-side sibling of ``enforce_format``:

1. recognised headings are re-labelled to the canonical emoji title,
2. missing sections are appended in order with coach-authored guidance that
   points back to the retrieved passages instead of inventing facts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from utils.constants import RevisionDeliverable

__all__ = [
    "CHAPTER_SECTIONS",
    "REVISION_SECTIONS",
    "sections_for_deliverable",
    "chapter_sections",
    "stamp_sections",
    "SectionStamp",
    "format_section_contract",
]

# ---------------------------------------------------------------------------
# Canonical section titles
# ---------------------------------------------------------------------------
#: Every indexed chapter is summarized with exactly these four sections.
CHAPTER_SECTIONS: tuple[str, ...] = (
    "\U0001f4d8 Chapter Summary",
    "\U0001f3af High-Yield Concepts",
    "\U0001f4dd Important NEET Points",
    "\u26a1 Quick Revision Notes",
)

#: One fixed shape per revision deliverable.
REVISION_SECTIONS: dict[RevisionDeliverable, tuple[str, ...]] = {
    RevisionDeliverable.FLASH_NOTES: (
        "\U0001f5d2\ufe0f Flash Cards",
        "\U0001f9e0 Trigger -> Recall",
        "\u2705 Verify In NCERT",
    ),
    RevisionDeliverable.FORMULA_SHEET: (
        "\U0001f9ee Formula Sheet",
        "\U0001f4d0 Symbols And Units",
        "\u26a0 Conditions And Traps",
    ),
    RevisionDeliverable.HIGH_YIELD: (
        "\u26a1 High-Yield Points",
        "\U0001f501 Repeatedly Asked Lines",
        "\u2705 Must Memorise",
    ),
    RevisionDeliverable.LAST_MINUTE: (
        "\u23f1\ufe0f 60-Second Recap",
        "\u26a0 Final Mistakes To Avoid",
        "\u2705 Exam-Hall Checklist",
    ),
}

#: Plain-language filler used only when the model skipped a section.  Nothing
#: here states a fact about the chapter, so a repair can never invent content.
_FALLBACK_BODIES: dict[str, str] = {
    CHAPTER_SECTIONS[0]: (
        "- The passages above cover this chapter; re-read them and note the "
        "definition, the condition and the one exception for each topic."
    ),
    CHAPTER_SECTIONS[1]: (
        "- Anything NCERT states as a definition, a range or a named exception is "
        "high-yield: NEET reuses those lines almost verbatim."
    ),
    CHAPTER_SECTIONS[2]: (
        "- Mark the exact textbook wording while you re-read, because the options "
        "in NEET are built from that wording."
    ),
    CHAPTER_SECTIONS[3]: (
        "- Close the book and recall one line per topic, then check it against the "
        "cited passages."
    ),
}


def _revision_fallback(title: str) -> str:
    """Safe generic guidance for a revision section the model skipped."""
    if "Formula" in title:
        return "- Write each formula with its symbols and SI units before using it."
    if "Symbols" in title or "Units" in title:
        return "- State what every symbol means and the unit it must be in."
    if "Conditions" in title or "Traps" in title:
        return "- Note where the formula stops being valid, and the usual sign or unit slip."
    if "Recall" in title:
        return "- Cover the right-hand column and recite from the cue on the left."
    if "Cards" in title:
        return "- One idea per card, in the textbook's own words."
    if "Repeatedly" in title or "Must Memorise" in title:
        return "- Lines that keep returning as statement questions deserve a flash card."
    if "Recap" in title:
        return "- Read this once, then recall it from memory before the paper."
    if "Mistakes" in title:
        return "- Re-check units, signs and the condition stated in the question."
    if "Checklist" in title:
        return "- Formula sheet, unit list and one pass of the marked NCERT lines."
    return "- Revisit the cited passages and summarise them in one line each."


def chapter_sections() -> tuple[str, ...]:
    """Section shape of a generated chapter study page."""
    return CHAPTER_SECTIONS


def sections_for_deliverable(deliverable: RevisionDeliverable | str) -> tuple[str, ...]:
    """Section shape of one Revision Hub deliverable."""
    resolved = deliverable if isinstance(deliverable, RevisionDeliverable) else RevisionDeliverable.coerce(
        deliverable
    )
    return REVISION_SECTIONS[resolved]


def format_section_contract(sections: tuple[str, ...]) -> str:
    """Render the contract as a single line for the UI (`A  |  B  |  C`)."""
    return "  |  ".join(sections)


# ---------------------------------------------------------------------------
# Heading detection + repair
# ---------------------------------------------------------------------------
_STOP_PREFIXES = ("```", ">", "|")


def _plain_title(line: str) -> str:
    """Strip markdown heading marks, bold markers and a trailing colon."""
    text = line.strip()
    if text.startswith("#"):
        text = text.lstrip("#").strip()
    for marker in ("**", "__"):
        if text.startswith(marker) and text.endswith(marker):
            text = text[len(marker) : -len(marker)].strip()
    text = text.rstrip(":").strip()
    return text


def _core(title: str) -> str:
    """Lower-case title with the emoji prefix removed, for fuzzy matching."""
    plain = _plain_title(title).lower()
    return re.sub(r"[^a-z0-9]+", " ", plain).strip()


def _matches(line: str, canonical: str) -> bool:
    """True when ``line`` reads as the heading ``canonical`` promises."""
    text = line.strip()
    if not text or text.startswith(_STOP_PREFIXES):
        return False
    if len(text) > 80:
        return False
    got, want = _core(_plain_title(text)), _core(canonical)
    if not got or not want:
        return False
    return got == want or want in got or got in want


@dataclass(slots=True)
class SectionStamp:
    """Outcome of reconciling model output with the promised section shape."""

    text: str
    added: list[str] = field(default_factory=list)
    relabelled: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.added or self.relabelled)

    def summary(self) -> str:
        parts: list[str] = []
        if self.relabelled:
            parts.append("re-labelled: " + ", ".join(self.relabelled))
        if self.added:
            parts.append("filled in: " + ", ".join(self.added))
        return "; ".join(parts) or "sections already in shape"


def stamp_sections(text: str, sections: tuple[str, ...], *, context: str = "") -> SectionStamp:
    """Guarantee that ``text`` carries every heading in ``sections``.

    Existing headings keep their body and are re-labelled to the canonical
    title.  A skipped section is appended at the end, in contract order, with
    coach guidance (never with invented chapter facts).
    """
    source = (text or "").strip()
    if not source:
        body = "\n\n".join(
            f"### {title}\n\n{_FALLBACK_BODIES.get(title) or _revision_fallback(title)}"
            for title in sections
        )
        return SectionStamp(text=body, added=list(sections))

    lines = source.split("\n")
    relabelled: list[str] = []
    for index, line in enumerate(lines):
        for canonical in sections:
            if _matches(line, canonical) and _core(_plain_title(line)) != _core(canonical):
                lines[index] = f"### {canonical}"
                relabelled.append(canonical)
                break

    body = "\n".join(lines).strip()
    missing = [canonical for canonical in sections if not any(_matches(line, canonical) for line in lines)]

    added: list[str] = []
    if missing:
        blocks = [body, ""]
        for canonical in missing:
            fallback = _FALLBACK_BODIES.get(canonical) or _revision_fallback(canonical)
            blocks.append(f"### {canonical}\n")
            blocks.append(fallback)
            if context.strip():
                blocks.append("_Built from the retrieved passages above; nothing here is invented._")
            blocks.append("")
            added.append(canonical)
        body = "\n".join(blocks).strip()

    return SectionStamp(text=body, added=added, relabelled=relabelled)
