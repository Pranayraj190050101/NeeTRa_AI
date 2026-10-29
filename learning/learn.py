"""Learn module service - hierarchical chapter study pages built with RAG.

Physics / Chemistry / Biology -> chapter list -> one generated page per
chapter with a fixed four-section shape:

    📘 Chapter Summary · 🎯 High-Yield Concepts · 📝 Important NEET Points
    ⚡ Quick Revision Notes

Chapter names come from the indexed PDFs (see :mod:`learning.catalog`), so
nothing about the syllabus is hardcoded, and every page is grounded in the
student's own passages with file/page citations.
"""

from __future__ import annotations

from typing import Callable, Sequence

from config import Settings
from learning.catalog import Chapter, ChapterCatalog
from learning.generation import GenieGenerator, GeneratedDoc
from learning.sections import CHAPTER_SECTIONS, format_section_contract, stamp_sections
from utils.constants import Subject
from utils.logging_utils import get_logger
from utils.text_utils import clean_text, truncate

__all__ = ["LearnService"]

logger = get_logger(__name__)


class LearnService:
    """Generates the Learn tab's chapter material."""

    def __init__(
        self,
        settings: Settings,
        generator: GenieGenerator,
        *,
        catalog_provider: Callable[[], ChapterCatalog] | None = None,
    ) -> None:
        self.settings = settings
        self.generator = generator
        self._catalog_provider = catalog_provider
        self.logger = get_logger("learning.learn")

    # ---------------------------------------------------------------- catalog
    def catalog(self) -> ChapterCatalog:
        """Current chapter hierarchy (empty when nothing is indexed)."""
        if self._catalog_provider is None:
            return ChapterCatalog.empty()
        return self._catalog_provider()

    def subjects(self) -> Sequence[str]:
        return self.catalog().subjects

    def chapters(self, subject: str | None = None) -> list[Chapter]:
        return self.catalog().list(subject)

    # -------------------------------------------------------------- generation
    def chapter_page(self, chapter: Chapter | str, subject: str | None = None) -> GeneratedDoc:
        """Build the four-section study page for one indexed chapter."""
        resolved = self._resolve(chapter, subject)
        if resolved is None:
            return GeneratedDoc.failure(
                kind="chapter",
                title=str(chapter),
                message=(
                    "**That chapter is not in your index.** Add the chapter PDF under "
                    f"`data/{subject or '<subject>'}/`, run `python -m rag.ingest --append`, "
                    "and pick it again from the dropdown."
                ),
            )

        passages = max(4, int(round(self.settings.top_k * 2)))
        grounding = self.generator.ground(
            (
                f"{resolved.name} chapter summary definitions formulas "
                "important points for NEET exceptions"
            ),
            subject=resolved.subject,
            passages=passages,
            prefer_file=resolved.file_name,
        )
        task = self.generator.build_task(
            task_id="learn_chapter",
            system_prompt="learn_chapter_system",
            user_prompt="learn_chapter_user",
            sections=CHAPTER_SECTIONS,
            subject=Subject.coerce(resolved.subject, default=Subject.GENERAL).label,
            chapter=resolved.name,
            focus="the whole chapter",
            context=grounding.context or "(no indexed passages were found)",
            sections_line=format_section_contract(CHAPTER_SECTIONS),
        )
        # the guarded generation path (same as revision.py): a model crash
        # becomes an ok=False document, never an exception in front of the student
        doc = self.generator.generate(
            task, kind="chapter", title=resolved.name, grounding=grounding
        )
        return self._finish(doc, context=grounding.context)

    def topic_page(self, topic: str, subject: str | None = None, *, focus: str = "") -> GeneratedDoc:
        """Study page for a free-text topic (works even without a chapter match)."""
        topic = clean_text(topic)
        if not topic:
            return GeneratedDoc.failure("chapter", "(empty)", "Name a topic to study.")
        grounding = self.generator.ground(
            f"{topic} {focus}".strip(),
            subject=subject,
            passages=max(4, int(round(self.settings.top_k * 1.6))),
        )
        task = self.generator.build_task(
            task_id="learn_chapter",
            system_prompt="learn_chapter_system",
            user_prompt="learn_chapter_user",
            sections=CHAPTER_SECTIONS,
            subject=Subject.coerce(subject, default=Subject.GENERAL).label,
            chapter=truncate(topic, 120, suffix=""),
            focus=clean_text(focus) or "this topic only",
            context=grounding.context or "(no indexed passages were found)",
            sections_line=format_section_contract(CHAPTER_SECTIONS),
        )
        doc = self.generator.generate(
            task, kind="topic", title=topic, grounding=grounding
        )
        return self._finish(doc, context=grounding.context)

    # ---------------------------------------------------------------- helpers
    def _resolve(self, chapter: Chapter | str, subject: str | None) -> Chapter | None:
        if isinstance(chapter, Chapter):
            return chapter
        catalog = self.catalog()
        found = catalog.find(str(chapter or ""), subject)
        if found is None and subject in (None, "all", "auto"):
            found = catalog.find(str(chapter or ""))
        return found

    def _finish(self, doc: GeneratedDoc, *, context: str = "") -> GeneratedDoc:
        """Stamp the promised section shape onto the model output."""
        if not doc.ok:
            return doc
        stamp = stamp_sections(doc.markdown, CHAPTER_SECTIONS, context=context)
        doc.markdown = stamp.text
        if stamp.changed:
            doc.notices.append(f"Study page sections completed: {stamp.summary()}.")
            self.logger.info("Section repair for '%s': %s", doc.title, stamp.summary())
        return doc
