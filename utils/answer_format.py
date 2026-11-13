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


def _revision_body_from_answer(bodies: dict[str, list[str]]) -> str:
    """Reuse concise topic facts when the model omits its revision recap."""
    candidates: list[str] = []
    for title in (EXPLANATION_SECTIONS[0], EXPLANATION_SECTIONS[2]):
        body, _ = _merge_unique_bodies(bodies[title])
        for line in body.splitlines():
            line = re.sub(r"^[ \t]*(?:[-*+]\s*|\d+[.)]\s*)", "", line).strip()
            line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
            for sentence in _SENTENCE_RE.split(line):
                candidate = sentence.strip()
                if 24 <= len(candidate) <= 150 and candidate not in candidates:
                    candidates.append(candidate)
                if len(candidates) == 3:
                    return _bullets(candidates)
    return _bullets(candidates)


def _importance_body() -> str:
    return (
        "- No question-frequency claim is made without a source; revise the topic's NCERT definition "
        "and stated application."
    )


def _mistakes_body() -> str:
    return "- Check each symbol or term and its stated condition before applying the answer."


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
            "- No relevant textbook passage was retrieved; verify this topic's definition and "
            "formula in the relevant NCERT chapter."
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
        return "- Recall the topic's definition, relation and condition from the explanation above."
    if "neet importance" in normalised:
        return _importance_body()
    if "common student mistakes" in normalised:
        return _mistakes_body()
    if "practice mcqs" in normalised:
        return _EXAMPLE_MARKER
    if "answer key" in normalised:
        return _EXAMPLE_MARKER
    if "traps" in normalised:
        return _mistakes_body()
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
    reordered: bool = False
    deduplicated: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.relabelled or self.added or self.reordered or self.deduplicated)

    def summary(self) -> str:
        """Short, user-facing description of what was repaired."""
        bits: list[str] = []
        if self.relabelled:
            bits.append(f"standardised {self.relabelled} heading(s)")
        if self.added:
            bits.append(f"added {len(self.added)} missing section(s): " + ", ".join(self.added))
        if self.reordered:
            bits.append("put sections in the required order")
        if self.deduplicated:
            bits.append(f"merged {self.deduplicated} duplicate section(s)")
        return "; ".join(bits)


def _section_bodies(text: str, mode: AnswerMode | str | None) -> tuple[dict[str, list[str]], int, bool]:
    """Collect recognized sections in order, keeping preamble with the concept."""
    titles = sections_for(mode)
    lines = text.splitlines()
    starts: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        raw_title = _heading_title(line)
        canonical = _canonicalise(raw_title, titles) if raw_title else None
        if canonical:
            starts.append((index, canonical))

    original_order = [title for _, title in starts]
    reordered = original_order != sorted(original_order, key=titles.index)
    duplicate_count = len(starts) - len(set(original_order))
    bodies: dict[str, list[str]] = {title: [] for title in titles}
    if not starts:
        if text.strip():
            bodies[titles[0]].append(text.strip())
        return bodies, 0, False

    preamble = "\n".join(lines[: starts[0][0]]).strip()
    if preamble:
        bodies[titles[0]].append(preamble)
    for position, (start, title) in enumerate(starts):
        stop = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        body_lines = lines[start + 1 : stop]
        # Keep useful subhead labels, but prevent them from looking like extra
        # top-level sections outside the five-heading contract.
        body_lines = [
            f"**{match.group('title').strip()}**" if (match := _MD_HEADING_RE.match(line.strip())) else line
            for line in body_lines
        ]
        body = "\n".join(body_lines).strip()
        if body:
            bodies[title].append(body)
    return bodies, duplicate_count, reordered


def _merge_unique_bodies(bodies: list[str]) -> tuple[str, int]:
    """Merge repeated section bodies without repeating identical paragraphs."""
    paragraphs: list[str] = []
    seen: set[str] = set()
    removed = 0
    for body in bodies:
        for paragraph in re.split(r"\n[ \t]*\n", body.strip()):
            paragraph = paragraph.strip()
            key = re.sub(r"\s+", " ", paragraph).casefold()
            if not paragraph or key in seen:
                removed += bool(paragraph)
                continue
            seen.add(key)
            paragraphs.append(paragraph)
    return "\n\n".join(paragraphs), removed


def enforce_format(
    text: str,
    mode: AnswerMode | str | None,
    *,
    subject: str | Subject | None = None,
    context: str = "",
) -> FormatRepair:
    """Guarantee one canonical, ordered copy of each required section."""
    resolved_subject = subject if isinstance(subject, Subject) else Subject.coerce(subject)
    repaired, relabelled = apply_headings(text, mode)
    bodies, duplicate_sections, reordered = _section_bodies(repaired, mode)
    added: list[str] = []
    deduplicated = duplicate_sections
    blocks: list[str] = []
    for title in sections_for(mode):
        body, removed_paragraphs = _merge_unique_bodies(bodies[title])
        deduplicated += removed_paragraphs
        if not body:
            body = (
                _revision_body_from_answer(bodies)
                if mode in (AnswerMode.DETAILED, AnswerMode.QUICK_REVISION)
                and title == EXPLANATION_SECTIONS[-1]
                else ""
            )
            if not body:
                body = fallback_body(title, subject=resolved_subject, context=context)
            added.append(title)
        blocks.append(f"### {title}\n\n{body}")

    return FormatRepair(
        text="\n\n".join(blocks).rstrip(),
        relabelled=relabelled,
        added=added,
        reordered=reordered,
        deduplicated=deduplicated,
    )


def format_contract(mode: AnswerMode | str | None) -> str:
    """Render the section list for logs, the UI and diagnostics."""
    return " | ".join(sections_for(mode))
