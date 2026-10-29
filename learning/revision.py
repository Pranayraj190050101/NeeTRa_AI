"""Revision Hub service - last-mile sheets generated from your own index.

Four deliverables, each with a fixed section shape (see
:mod:`learning.sections` and :class:`~utils.constants.RevisionDeliverable`):

* 🗒️ Flash Notes - one-line recall cards
* 🧮 Formula Sheet - formulas with symbols, units and validity conditions
* ⚡ High Yield Points - the lines NCERT turns into questions
* ⏱️ Last Minute Revision - the 60-second recap plus the traps

The hub can also target the student's *weakest* topics, which is how the
platform closes the loop: Learn -> Practice -> Progress -> Revision.
"""

from __future__ import annotations

from typing import Sequence

from config import Settings
from learning.generation import GeneratedDoc, GenieGenerator, Grounding
from learning.progress import ProgressStore, WeakArea
from learning.sections import format_section_contract, sections_for_deliverable, stamp_sections
from utils.constants import RevisionDeliverable, Subject
from utils.logging_utils import get_logger
from utils.text_utils import clean_text, truncate

__all__ = ["RevisionService"]

logger = get_logger(__name__)


class RevisionService:
    """Builds the Revision Hub's study sheets."""

    def __init__(
        self,
        settings: Settings,
        generator: GenieGenerator,
        progress: ProgressStore | None = None,
    ) -> None:
        self.settings = settings
        self.generator = generator
        self.progress = progress
        self.logger = get_logger("learning.revision")

    # ------------------------------------------------------------------ sheets
    def sheet(
        self,
        deliverable: RevisionDeliverable | str = RevisionDeliverable.FLASH_NOTES,
        *,
        subject: str | None = None,
        target: str = "",
        passages: int | None = None,
    ) -> GeneratedDoc:
        """Generate one revision sheet for a chapter, topic or whole subject."""
        item = deliverable if isinstance(deliverable, RevisionDeliverable) else RevisionDeliverable.coerce(
            deliverable
        )
        label = clean_text(target) or Subject.coerce(subject, default=Subject.GENERAL).label
        sections = sections_for_deliverable(item)
        grounding = self.ground(item, subject=subject, target=label, passages=passages)
        return self.render(item, sections, label=label, grounding=grounding, subject=subject)

    def weak_area_sheet(
        self,
        deliverable: RevisionDeliverable | str = RevisionDeliverable.LAST_MINUTE,
        *,
        limit: int = 4,
    ) -> GeneratedDoc:
        """Revise exactly what the student keeps getting wrong."""
        areas = self.weakest_topics(limit=limit)
        if not areas:
            return GeneratedDoc.failure(
                kind="revision",
                title="Weak areas",
                message=(
                    "**No weak areas recorded yet.** Take a test in the Practice tab first - "
                    "this sheet is built from the topics you actually lost marks in."
                ),
            )
        item = deliverable if isinstance(deliverable, RevisionDeliverable) else RevisionDeliverable.coerce(
            deliverable
        )
        names = ", ".join(area.name for area in areas)
        subject = areas[0].subject
        sections = sections_for_deliverable(item)
        query = (
            f"{names} {subject} common mistakes definitions formulas conditions "
            "high yield points for NEET"
        )
        grounding = self.generator.ground(
            query, subject=subject, passages=max(5, int(round(self.settings.top_k * item.retrieval_factor * 1.2)))
        )
        doc = self.render(
            item,
            sections,
            label=f"Weakest topics: {names}",
            grounding=grounding,
            subject=subject,
            extra=(
                "Focus only on the weak areas listed in the request. For each one, state the "
                "fact, the trap and the recall cue."
            ),
        )
        doc.notices.insert(
            0,
            "Targets chosen from your test history: " + "; ".join(
                f"{area.name} ({area.accuracy_percent}%)" for area in areas
            ),
        )
        return doc

    def weakest_topics(self, *, limit: int = 6) -> Sequence[WeakArea]:
        """Progress-derived revision targets (empty before the first test)."""
        if self.progress is None:
            return []
        return self.progress.weak_areas(limit=limit)

    # ---------------------------------------------------------------- plumbing
    def ground(
        self,
        item: RevisionDeliverable,
        *,
        subject: str | None,
        target: str,
        passages: int | None = None,
    ) -> Grounding:
        """Retrieve the material the sheet is written from."""
        query = f"{target} {Subject.coerce(subject, default=Subject.GENERAL).label} " + _QUERY_TAILS[item]
        return self.generator.ground(
            query,
            subject=subject,
            passages=passages or max(4, int(round(self.settings.top_k * item.retrieval_factor))),
            prefer_file=_matching_file_hint(self.generator, target),
        )

    def render(
        self,
        item: RevisionDeliverable,
        sections: tuple[str, ...],
        *,
        label: str,
        grounding: Grounding,
        subject: str | None = None,
        extra: str = "",
    ) -> GeneratedDoc:
        """Ask the model for one deliverable and guarantee its section shape."""
        task = self.generator.build_task(
            task_id="revision_notes",
            system_prompt="revision_system",
            user_prompt="revision_user",
            sections=sections,
            deliverable=item.label,
            sections_line=format_section_contract(sections),
            subject=Subject.coerce(subject, default=Subject.GENERAL).label,
            target=truncate(label, 160, suffix=""),
            guidance=extra or "Cover the requested target completely.",
            context=grounding.context or "(no indexed passages were found)",
        )
        doc = self.generator.generate(
            task,
            kind=item.value,
            title=f"{item.emoji} {item.label} - {truncate(label, 70, suffix='')}",
            grounding=grounding,
        )
        if not doc.ok:
            return doc
        stamp = stamp_sections(doc.markdown, sections, context=grounding.context)
        doc.markdown = stamp.text
        if stamp.changed:
            doc.notices.append(f"Revision sheet sections completed: {stamp.summary()}.")
            self.logger.info("Revision sheet repair for '%s': %s", label, stamp.summary())
        return doc


_QUERY_TAILS: dict[RevisionDeliverable, str] = {
    RevisionDeliverable.FLASH_NOTES: (
        "short definitions terms facts one line recall exceptions named scientists"
    ),
    RevisionDeliverable.FORMULA_SHEET: (
        "formula equation relation derivation units symbols condition of validity"
    ),
    RevisionDeliverable.HIGH_YIELD: (
        "important statements frequently asserted exceptions examples textbook lines"
    ),
    RevisionDeliverable.LAST_MINUTE: (
        "summary key values common mistakes traps final checklist formulas"
    ),
}


def _matching_file_hint(generator: GenieGenerator, target: str) -> str | None:
    """Best effort chapter-file scoping for a revision target."""
    token = clean_text(target).lower().replace(" ", "_")
    if not token:
        return None
    try:
        stats = generator.vector_store.stats()
    except Exception:  # pragma: no cover - index unavailable
        return None
    for source in stats.sources:
        if token and token in clean_text(source).lower().replace(" ", "_"):
            return source.split("/")[-1]
    return None
