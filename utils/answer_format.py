"""The Neet Genie answer contract.

A NEET coach does not just give an answer, it gives an answer in a *predictable
shape* so a student can revise from it later.  Local models do not always obey
that shape, so this module (pure standard library) reconciles what the model
wrote with the sections the selected :class:`~utils.constants.AnswerMode`
promises:

1. :func:`apply_headings` re-labels recognised headings to the canonical,
   emoji-prefixed titles (``Concept Explanation`` -> ``### 📘 Concept Explanation``).
2. :func:`missing_sections` reports the sections the model skipped.
3. :func:`enforce_format` appends those sections in order, filled from the
   retrieved NCERT passages where possible and from honest, exam-safe coach
   guidance otherwise - never from invented facts, page numbers or statistics.

The fallback text is deliberately conservative: a missing "NEET Importance"
block must not become a fabricated weightage claim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from utils.constants import AnswerMode, Subject, normalise_token

__all__ = [
    "MODE_SECTIONS",
    "SECTION_ALIASES",
    "FormatRepair",
    "sections_for",
    "apply_headings",
    "missing_sections",
    "enforce_format",
    "format_contract",
]

# ---------------------------------------------------------------------------
# Canonical section titles, exactly as promised to the student
# ---------------------------------------------------------------------------
EXPLANATION_SECTIONS: tuple[str, ...] = (
    "\U0001f4d8 Concept Explanation",
    "\U0001f3af NEET Importance",
    "\U0001f9e0 Key Points To Remember",
    "\u26a0 Common Student Mistakes",
    "\u2705 Quick Revision Notes",
)

MCQ_SECTIONS: tuple[str, ...] = (
    "\U0001f9e9 Practice MCQs",
    "\U0001f511 Answer Key With Explanation",
    "\u26a0 Traps Tested In NEET",
    "\u2705 Quick Revision Notes",
)

PYQ_SECTIONS: tuple[str, ...] = (
    "\U0001f4dc How NEET Asks This",
    "\U0001f4d8 Concept Explanation",
    "\U0001f3af NEET Importance",
    "\u26a0 Common Student Mistakes",
    "\u2705 Quick Revision Notes",
)

MODE_SECTIONS: dict[AnswerMode, tuple[str, ...]] = {
    AnswerMode.DETAILED: EXPLANATION_SECTIONS,
    AnswerMode.QUICK_REVISION: EXPLANATION_SECTIONS,
    AnswerMode.MCQ_PRACTICE: MCQ_SECTIONS,
    AnswerMode.PYQ_FOCUS: PYQ_SECTIONS,
}

#: Normalised (lower-case, alphanumeric) titles that count as a section heading.
#: Longest alias wins, so "concept explanation" beats "explanation".
SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "\U0001f4d8 Concept Explanation": (
        "concept explanation",
        "explanation",
        "concept",
        "understanding",
        "what it says",
    ),
    "\U0001f3af NEET Importance": (
        "neet importance",
        "importance for neet",
        "importance",
        "why it matters",
        "significance",
        "weightage",
    ),
    "\U0001f9e0 Key Points To Remember": (
        "key points to remember",
        "key points",
        "important points",
        "key takeaways",
        "takeaways",
    ),
    "\u26a0 Common Student Mistakes": (
        "common student mistakes",
        "common mistakes",
        "student mistakes",
        "common errors",
        "mistakes",
        "errors",
        "confusions",
    ),
    "\u2705 Quick Revision Notes": (
        "quick revision notes",
        "quick revision",
        "revision notes",
        "revision",
        "recall",
    ),
    "\U0001f9e9 Practice MCQs": (
        "practice mcqs",
        "practice questions",
        "multiple choice questions",
        "mcqs",
        "mcq",
    ),
    "\U0001f511 Answer Key With Explanation": (
        "answer key with explanation",
        "answer key and explanation",
        "answer key",
        "answers with explanations",
        "solutions",
    ),
    "\u26a0 Traps Tested In NEET": (
        "traps tested in neet",
        "neet traps",
        "exam traps",
        "traps",
    ),
    "\U0001f4dc How NEET Asks This": (
        "how neet asks this",
        "how neet has asked",
        "neet pattern",
        "previous year questions",
        "pyq focus",
        "pyqs",
        "pyq",
    ),
}

#: Heading forms recognised in model output.
_BOLD_ONLY_RE = re.compile(r"^[ \t]*\*\*(?P<title>.+?)\*\*[ \t]*[:.]?[ \t]*$")
_LABEL_ONLY_RE = re.compile(r"^[ \t]*(?P<title>[^:\n#*]{3,80}):[ \t]*$")
_MD_HEADING_RE = re.compile(r"^#{1,6}[ \t]+(?P<title>.+)$")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_CONTEXT_HEADER_RE = re.compile(r"^\[\d+\][^\n]*$")

#: How much of a line an alias must cover before we treat that line as the
#: section heading.  This stops ordinary sentences that merely mention
#: "mistakes" or "revision" from being re-labelled as headings.
_ALIAS_COVERAGE = 0.55
_MAX_HEADING_CHARS = 60
_SENTENCE_PUNCT = (".", "!", "?")
_BULLET_PREFIXES = ("```", ">", "|", "- ", "* ", "\u2022 ", "1.", "2.", "3.", "Q.")

#: Coach-authored filler, keyed by canonical title.  No invented facts.
_IMPORTANCE_FALLBACK = (
    "- Straight NCERT statement questions plus one application-based question are the usual pattern.\n"
    "- Master the definition, the standard unit or formula, and the exception - that is what is asked.\n"
    "- Practice the NCERT exemplar and back-exercise questions on this topic after reading it."
)

_IMPORTANCE_BY_SUBJECT: dict[Subject, str] = {
    Subject.PHYSICS: (
        "- NEET physics asks this as a formula/application question: know the expression, its units "
        "and the limiting cases.\n"
        "- Dimensions and unit checks solve most of the objective options without a full calculation.\n"
        "- Graphs and ratio-type questions (\u201cif n is doubled, then\u2026\u201d) are the usual twist."
    ),
    Subject.CHEMISTRY: (
        "- NEET chemistry asks this as a factual NCERT line plus one numerical/application item.\n"
        "- Learn the exceptions and the order/trend questions - they separate toppers from the rest.\n"
        "- For physical chemistry keep the units and significant figures exactly as NCERT states them."
    ),
    Subject.BIOLOGY: (
        "- Biology is the most NCERT-verbatim paper: the exact textbook wording becomes the option.\n"
        "- Expect statement-based and assertion-reason items from this section's tables and examples.\n"
        "- Learn the scientists, dates, examples and Latin names given in the chapter text."
    ),
}

_MISTAKES_FALLBACK = (
    "- Reading the chapter once and treating it as revision - recall it in one line without looking.\n"
    "- Confusing this idea with the nearest-looking one; keep a one-line difference note.\n"
    "- Ignoring units, sign conventions or the stated conditions of the formula or statement."
)

_MISTAKES_BY_SUBJECT: dict[Subject, str] = {
    Subject.PHYSICS: (
        "- Substituting values before converting every quantity to SI units.\n"
        "- Dropping the sign convention (work, potential, momentum direction).\n"
        "- Applying a formula outside the condition it was derived for (constant force, small angle, ideal gas)."
    ),
    Subject.CHEMISTRY: (
        "- Balancing an equation but forgetting state symbols or the 1 : 1 mole ratio in the numerical.\n"
        "- Mixing up similar reagents/order rules; write the difference next to each name.\n"
        "- Rounding intermediate values and then losing accuracy in the final answer."
    ),
    Subject.BIOLOGY: (
        "- Paraphrasing NCERT lines in your own words, then losing marks on statement-based options.\n"
        "- Skipping the chapter's figures, tables and boxed items, which are mined for PYQs.\n"
        "- Confusing similar terms (homologous/analogous, xylem/phloem stages, alike-sounding hormones)."
    ),
}

_EXAMPLE_MARKER = "_(the model did not fill this block - re-ask after indexing the chapter)_"


def sections_for(mode: AnswerMode | str | None) -> tuple[str, ...]:
    """Return the canonical section titles promised for ``mode``."""
    return MODE_SECTIONS[AnswerMode.coerce(mode)]


def _canonicalise(raw_title: str, mode_sections: tuple[str, ...]) -> str | None:
    """Map a free-form heading onto a canonical title of this mode (or ``None``).

    The longest alias wins, and it must cover most of the line so that normal
    prose is never mistaken for a section title.
    """
    title = normalise_token(raw_title)
    if not title or len(title) > _MAX_HEADING_CHARS:
        return None
    best: tuple[int, str] | None = None
    for canonical in mode_sections:
        for alias in SECTION_ALIASES.get(canonical, ()):
            if alias not in title or len(alias) / len(title) < _ALIAS_COVERAGE:
                continue
            if best is None or len(alias) > best[0]:
                best = (len(alias), canonical)
    return best[1] if best else None


def _heading_title(line: str) -> str | None:
    """Return the title when ``line`` is plausibly a section heading."""
    stripped = line.strip()
    if not stripped or stripped.startswith(_BULLET_PREFIXES):
        return None
    for pattern in (_BOLD_ONLY_RE, _LABEL_ONLY_RE, _MD_HEADING_RE):
        match = pattern.match(stripped)
        if match:
            return match.group("title")
    # A bare title line: short and free of sentence punctuation.
    if len(stripped) <= _MAX_HEADING_CHARS and not any(mark in stripped for mark in _SENTENCE_PUNCT):
        return stripped
    return None


def _heading_candidates(lines: list[str]) -> list[tuple[int, str]]:
    """Yield ``(index, raw_title)`` for every plausible heading line."""
    candidates: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        title = _heading_title(line)
        if title:
            candidates.append((index, title))
    return candidates


def apply_headings(text: str, mode: AnswerMode | str | None) -> tuple[str, int]:
    """Rewrite recognised headings to the canonical markdown headings.

    Returns the (possibly rewritten) text and how many headings were relabelled.
    Section *bodies* are never touched, so a model that wrote good content under
    "Key Points" simply gets the heading upgraded.
    """
    mode_sections = sections_for(mode)
    lines = text.split("\n")
    relabelled = 0
    for index, raw_title in _heading_candidates(lines):
        canonical = _canonicalise(raw_title, mode_sections)
        if canonical is None:
            continue
        replacement = f"### {canonical}"
        if lines[index] != replacement:
            lines[index] = replacement
            relabelled += 1
    return "\n".join(lines), relabelled


def present_sections(text: str, mode: AnswerMode | str | None) -> list[str]:
    """Canonical titles already present in ``text`` (in prompt order)."""
    mode_sections = sections_for(mode)
    hits: set[str] = set()
    for _, raw_title in _heading_candidates(text.split("\n")):
        canonical = _canonicalise(raw_title, mode_sections)
        if canonical:
            hits.add(canonical)
    return [title for title in mode_sections if title in hits]


def missing_sections(text: str, mode: AnswerMode | str | None) -> list[str]:
    """Canonical titles the model omitted for this mode."""
    present = set(present_sections(text, mode))
    return [title for title in sections_for(mode) if title not in present]


# ---------------------------------------------------------------------------
# Grounded fallback content
# ---------------------------------------------------------------------------
def _context_sentences(context: str, limit: int = 3) -> list[str]:
    """Pick the most definition-like sentences out of the retrieved blocks."""
    sentences: list[str] = []
    for line in context.splitlines():
        stripped = line.strip()
        if not stripped or _CONTEXT_HEADER_RE.match(stripped):
            continue
        for sentence in _SENTENCE_RE.split(stripped):
            candidate = sentence.strip()
            if len(candidate) < 40 or len(candidate) > 260:
                continue
            if candidate not in sentences:
                sentences.append(candidate)
            if len(sentences) >= limit:
                return sentences
    return sentences


def _bullets(sentences: list[str]) -> str:
    return "\n".join(f"- {sentence}" for sentence in sentences)


def _importance_body(subject: Subject | None) -> str:
    return _IMPORTANCE_BY_SUBJECT.get(subject or Subject.GENERAL, _IMPORTANCE_FALLBACK)


def _mistakes_body(subject: Subject | None) -> str:
    return _MISTAKES_BY_SUBJECT.get(subject or Subject.GENERAL, _MISTAKES_FALLBACK)


def fallback_body(
    title: str,
    *,
    subject: Subject | None = None,
    context: str = "",
) -> str:
    """Honest filler for a section the model skipped.

    Content comes from the retrieved passages when they exist; otherwise the
    coach states what to do next instead of inventing NEET facts.
    """
    sentences = _context_sentences(context)
    normalised = normalise_token(title)

    if "concept explanation" in normalised:
        if sentences:
            return _bullets(sentences[:2])
        return (
            "- No textbook passage was retrieved for this question, so a grounded explanation "
            "cannot be written yet.\n"
            "- Index the relevant chapter (`python -m rag.ingest --rebuild`) and ask again."
        )
    if "key points" in normalised:
        if sentences:
            return _bullets(sentences[:4])
        return (
            "- Write one line for the definition, one for the condition it applies under, and one "
            "for the most common exception.\n"
            "- Close the book and recall those three lines before moving to the next topic."
        )
    if "quick revision notes" in normalised:
        if sentences:
            return "\n".join(f"- {sentence[:160]}" for sentence in sentences[:5])
        return "- Re-read the cited passage the same day, then again after three days."
    if "neet importance" in normalised:
        return _importance_body(subject)
    if "common student mistakes" in normalised:
        return _mistakes_body(subject)
    if "practice mcqs" in normalised:
        return _EXAMPLE_MARKER
    if "answer key" in normalised:
        return _EXAMPLE_MARKER
    if "traps" in normalised:
        return _mistakes_body(subject)
    if "how neet asks this" in normalised:
        return (
            "- PYQs on a topic like this are usually re-worded NCERT lines, so the chapter text is "
            "the question bank.\n"
            "- Practise statement-assertion items and match-the-following formats from this section.\n"
            "- Add your PYQ PDFs to `data/<subject>/` and re-ingest to let me quote exact years."
        )
    return "- (no grounded content available for this section)"


@dataclass(slots=True)
class FormatRepair:
    """Outcome of :func:`enforce_format`."""

    text: str
    relabelled: int = 0
    added: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.relabelled or self.added)

    def summary(self) -> str:
        """Short, user-facing description of what was repaired."""
        bits: list[str] = []
        if self.relabelled:
            bits.append(f"standardised {self.relabelled} heading(s)")
        if self.added:
            bits.append(f"added {len(self.added)} missing section(s): " + ", ".join(self.added))
        return "; ".join(bits)


def enforce_format(
    text: str,
    mode: AnswerMode | str | None,
    *,
    subject: str | Subject | None = None,
    context: str = "",
) -> FormatRepair:
    """Guarantee the mode's section contract on the final answer."""
    resolved_subject = subject if isinstance(subject, Subject) else Subject.coerce(subject)
    repaired, relabelled = apply_headings(text, mode)
    missing = missing_sections(repaired, mode)

    for title in missing:
        body = fallback_body(title, subject=resolved_subject, context=context)
        repaired = f"{repaired.rstrip()}\n\n### {title}\n\n{body}\n"

    return FormatRepair(text=repaired.rstrip(), relabelled=relabelled, added=missing)


def format_contract(mode: AnswerMode | str | None) -> str:
    """Render the section list for logs, the UI and diagnostics."""
    return " | ".join(sections_for(mode))
