"""``GeniePlatform`` - the service facade behind every module tab.

The chatbot becomes one module of a larger product.  This class owns the
platform's services and keeps the boundary honest:

* **reuses** the existing tutor (LangGraph, FAISS RAG, LLM client, prompts) and
  never re-implements a step of it;
* **adds** the modular services the platform needs - catalog, learn, practice,
  revision, progress and capability extension points;
* **exposes plain data** (``dashboard()``, ``platform_status()``) so the UI
  layer stays presentational and everything is testable without Gradio.

Construction is flexible on purpose: the app passes the live ``NEETTutor``,
while tests pass an LLM plus a fake-embedding vector store, which is how the
whole platform runs offline.
"""

from __future__ import annotations

import threading
from datetime import date
from typing import Any, Callable

from config import Settings
from learning.assessment import GradedTest
from learning.capabilities import (
    Capability,
    CapabilityNotAvailable,
    CapabilityRegistry,
    CapabilityState,
    default_registry,
    describe_capabilities,
)
from learning.catalog import ChapterCatalog
from learning.generation import GenieGenerator
from learning.learn import LearnService
from learning.motivation import Motivation, daily_motivation
from learning.practice import PracticeService, PracticeSet
from learning.progress import ProgressStore
from learning.revision import RevisionService
from prompts.loader import PromptLibrary
from rag.vector_store import VectorStoreManager
from utils.constants import AnswerMode, RevisionDeliverable, Subject, TestScope
from utils.logging_utils import get_logger

__all__ = [
    "GeniePlatform",
    "QUICK_ACTIONS",
    "capability_hint",
    "CapabilityNotAvailable",
    "PracticeSet",
]

logger = get_logger(__name__)

#: Dashboard shortcuts.  ``tab`` / ``selector`` / ``value`` are what the UI
#: feeds to the tab switcher and the target tab's first control.
QUICK_ACTIONS: tuple[dict[str, str], ...] = (
    {
        "key": "ask_doubt",
        "label": "Ask Doubt",
        "emoji": "\U0001f4ac",
        "tab": "ask-genie",
        "selector": "mode",
        "value": AnswerMode.DETAILED.value,
        "hint": "Type the doubt; Genie answers in the full coach format.",
        "seed": "Explain this concept the way NEET asks it, with the common trap.",
    },
    {
        "key": "practice_mcqs",
        "label": "Practice MCQs",
        "emoji": "\U0001f9e9",
        "tab": "practice",
        "selector": "scope",
        "value": TestScope.CHAPTER.value,
        "hint": "Pick a chapter and take a short, grounded MCQ test.",
        "seed": "",
    },
    {
        "key": "quick_revision",
        "label": "Quick Revision",
        "emoji": "\u26a1",
        "tab": "ask-genie",
        "selector": "mode",
        "value": AnswerMode.QUICK_REVISION.value,
        "hint": "The same coach answer, compressed into a one-page revision sheet.",
        "seed": "Give me a one-screen revision sheet for this chapter.",
    },
    {
        "key": "mock_test",
        "label": "Start Mock Test",
        "emoji": "\U0001f3af",
        "tab": "practice",
        "selector": "scope",
        "value": TestScope.SUBJECT.value,
        "hint": "A full subject test mixed across every indexed chapter of that subject.",
        "seed": "",
    },
)


class GeniePlatform:
    """Owns the modular services of the NEETra AI learning platform."""

    def __init__(
        self,
        settings: Settings,
        *,
        tutor: Any | None = None,
        vector_store: VectorStoreManager | None = None,
        prompts: PromptLibrary | None = None,
        llm: Any | None = None,
        llm_provider: Callable[[], Any] | None = None,
        progress_path: Any | None = None,
    ) -> None:
        self.settings = settings
        self.tutor = tutor
        self.vector_store = vector_store or getattr(tutor, "vector_store", None) or VectorStoreManager(settings)
        self.prompts = prompts or getattr(tutor, "prompts", None) or PromptLibrary(settings.prompt_dir)

        provider = llm_provider
        if provider is None and tutor is not None:
            def provider() -> Any:  # noqa: D401 - lazy model load, same as the chat
                return tutor.ensure_ready().llm

        self.generator = GenieGenerator(
            settings,
            prompts=self.prompts,
            vector_store=self.vector_store,
            llm=llm,
            llm_provider=provider,
        )
        self.progress = ProgressStore(progress_path or settings.progress_dir)
        self.capabilities: CapabilityRegistry = default_registry()

        self._catalog: ChapterCatalog = ChapterCatalog.empty()
        self._catalog_loaded = False
        self._catalog_lock = threading.RLock()

        self.learn = LearnService(settings, self.generator, catalog_provider=self.catalog)
        self.practice = PracticeService(
            settings, self.generator, self.progress, catalog_provider=self.catalog
        )
        self.revision = RevisionService(settings, self.generator, self.progress)

    # ---------------------------------------------------------------- catalog
    def catalog(self) -> ChapterCatalog:
        """Chapter hierarchy, read on first use and cached from then on.

        The refresh is deferred until a module actually needs the hierarchy so
        that opening the app never blocks on the knowledge base, exactly like
        the chat's lazy model loading.
        """
        with self._catalog_lock:
            if not self._catalog_loaded:
                self._catalog_loaded = True
                # Sidecar only here: reading the live index would load the
                # embedding model, and the app must start instantly.
                self._read_catalog()
            return self._catalog

    def refresh_catalog(self) -> ChapterCatalog:
        """Re-read the catalog (used after an ingestion run or on demand).

        This is the one place that may open the FAISS index, so it belongs to an
        explicit user action - the ``Refresh chapters`` button - and never to
        start-up.
        """
        with self._catalog_lock:
            self._catalog_loaded = True
            return self._read_catalog(live=True)

    def _read_catalog(self, *, live: bool = False) -> ChapterCatalog:
        """Prefer the index sidecar (cheap); fall back to the live index."""
        catalog: ChapterCatalog | None = None
        try:
            from rag.vector_store import read_index_metadata

            catalog = ChapterCatalog.from_metadata(read_index_metadata(self.settings))
        except Exception:  # pragma: no cover - defensive
            logger.debug("Catalog from index metadata failed", exc_info=True)
        if catalog is None and live:
            # Older indexes have no ``sources`` key in the sidecar.
            catalog = ChapterCatalog.from_manager(self.vector_store)
        if catalog is not None:
            self._catalog = catalog
        if self._catalog.is_empty:
            logger.info("Chapter catalog empty - Learn/Practice targets come from free text only")
        return self._catalog

    # ---------------------------------------------------------------- dashboard
    def dashboard(self, *, today: date | None = None) -> dict[str, Any]:
        """Everything the Dashboard tab renders, as plain data."""
        stats = self.progress.stats()
        catalog = self.catalog()
        index = self._index_summary()
        motivation = self.motivation(stats)
        # Keep dashboard metadata aligned with the six student-facing modules.
        modules = ["Dashboard", "Learn", "Practice", "Ask Genie", "Progress", "Revision Hub"]
        return {
            "welcome": {
                "name": self.settings.app_name,
                "tagline": self.settings.app_tagline,
                "role": "Your Personal NEET Doubt Solver, Practice Partner and Rank Improver",
                "modules": modules,
                "index_ready": index["ready"],
                "chapters": len(catalog.chapters),
                "chunks": index["chunks"],
            },
            "motivation": motivation.as_dict(),
            "stats": {
                "tests": stats["tests"],
                "questions": stats["questions"],
                "correct": stats["correct"],
                "accuracy": stats["accuracy"],
                "score": stats["score"],
                "max_score": stats["max_score"],
                "streak_days": stats["streak_days"],
                "days_active": stats["days_active"],
                "chapters": len(catalog.chapters),
                "subjects": len(catalog.subjects),
                "doubts": stats["doubts"],
                "revision_sheets": stats["revision_sheets"],
            },
            "subjects": self.subject_cards(),
            "actions": [dict(action) for action in QUICK_ACTIONS],
            "capabilities": describe_capabilities(self.capabilities),
        }

    def subject_cards(self) -> list[dict[str, Any]]:
        """One card per NEET subject: coverage today, performance so far."""
        catalog = self.catalog()
        performance = self.progress.subject_performance()
        weak = self.progress.weak_areas(limit=50)
        cards: list[dict[str, Any]] = []
        for subject in (Subject.PHYSICS, Subject.CHEMISTRY, Subject.BIOLOGY):
            values = performance.get(subject.value)
            topic_names = sorted({area.name for area in weak if area.subject == subject.value})
            chapters = catalog.count_for(subject.value)
            cards.append(
                {
                    "subject": subject.value,
                    "label": subject.label,
                    "emoji": _SUBJECT_EMOJI[subject.value],
                    "chapters": chapters,
                    "chunks": catalog.size_of(subject.value),
                    "tests": values.tests if values else 0,
                    "questions": values.questions if values else 0,
                    "accuracy": values.accuracy if values else 0.0,
                    "accuracy_percent": values.accuracy_percent if values else 0,
                    "score": values.score if values else 0.0,
                    "weak_topics": topic_names[:3],
                    "status": _card_status(chapters, values),
                }
            )
        return cards

    def motivation(self, stats: dict[str, Any] | None = None, *, today: date | None = None) -> Motivation:
        """Greeting line for the dashboard, informed by the study streak."""
        stats = stats or self.progress.stats()
        return daily_motivation(
            on_date=today,
            streak_days=int(stats.get("streak_days") or 0),
            questions_answered=int(stats.get("questions") or 0),
        )

    # -------------------------------------------------------------- generation
    def study_chapter(self, chapter: str, subject: str | None = None):
        """Learn tab: one chapter's four-section study page."""
        doc = self.learn.chapter_page(chapter, subject)
        if doc.ok:
            self.progress.bump("chapter_pages")
        return doc

    def study_topic(self, topic: str, subject: str | None = None, *, focus: str = ""):
        """Learn tab: study page for a free-text topic inside the index."""
        doc = self.learn.topic_page(topic, subject, focus=focus)
        if doc.ok:
            self.progress.bump("chapter_pages")
        return doc

    def generate_test(
        self,
        *,
        scope: TestScope | str = TestScope.CHAPTER,
        subject: str | None = None,
        target: str = "",
        count: int | None = None,
    ) -> PracticeSet:
        """Practice tab: build one subject / chapter / topic test."""
        return self.practice.generate(scope=scope, subject=subject, target=target, count=count)

    def submit_test(self, test: PracticeSet | str, answers: Any, *, duration_seconds: float = 0.0) -> GradedTest:
        """Practice tab: grade, persist and return the analysed result."""
        return self.practice.submit(test, answers, duration_seconds=duration_seconds)

    def revision_sheet(
        self,
        deliverable: Any = "flash_notes",
        *,
        subject: str | None = None,
        target: str = "",
    ):
        """Revision Hub: flash notes, formula sheet, high-yield or last-minute."""
        doc = self.revision.sheet(deliverable, subject=subject, target=target)
        if doc.ok:
            self.progress.bump("revision_sheets")
        return doc

    def revise_weak_areas(self, deliverable: Any = "last_minute"):
        """Revision Hub: turn the recorded weak topics into a focused sheet."""
        doc = self.revision.weak_area_sheet(deliverable)
        if doc.ok:
            self.progress.bump("revision_sheets")
        return doc

    # ------------------------------------------------------------------- chat
    def ask(
        self,
        question: str,
        *,
        subject: str = "auto",
        mode: AnswerMode | str | None = None,
        top_k: int | None = None,
        history: list[dict[str, Any]] | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        on_token: Callable[[str], None] | None = None,
    ):
        """The single funnel every input channel uses (Ask Genie module)."""
        if self.tutor is None:
            raise RuntimeError("GeniePlatform.ask needs a tutor (the chat pipeline)")
        resolved = AnswerMode.coerce(mode, default=self.settings.answer_mode)
        result = self.tutor.ask(
            question,
            history=history,
            subject=subject,
            top_k=top_k,
            mode=resolved,
            on_event=on_event,
            on_token=on_token,
        )
        if getattr(result, "ok", True):
            self.progress.bump("doubts")
        return result

    # -------------------------------------------------- capability extension
    def listen(self, audio: bytes, *, language: str = "en", **ask_kwargs: Any):
        """Extension point: transcribe speech, then ask through the same funnel.

        Raises :class:`CapabilityNotAvailable` until a ``VoiceTranscriber`` is
        registered in :attr:`capabilities` - which is exactly the seam the
        future voice module plugs into.
        """
        transcriber = self.capabilities.require("voice")
        question = transcriber.transcribe(audio, language=language)
        return self.ask(question, **ask_kwargs)

    def solve_photo(self, image: bytes, *, subject: str | None = None, **ask_kwargs: Any):
        """Extension point: read a photographed question, then ask the coach."""
        solver = self.capabilities.require("image")
        question = solver.extract_question(image, subject=subject)
        return self.ask(question, subject=subject or "auto", **ask_kwargs)

    def capability(self, key: str) -> Capability | None:
        return self.capabilities.get(key)

    # ------------------------------------------------------------------ status
    def _index_summary(self) -> dict[str, Any]:
        ready = False
        chunks = 0
        try:
            ready = self.vector_store.is_ready()
            status = getattr(self.tutor, "index_status", None)
            if callable(status):
                payload = status()
                ready = bool(payload.get("ready", ready))
                chunks = int(payload.get("chunks") or 0)
        except Exception:  # pragma: no cover - defensive
            logger.debug("Index summary failed", exc_info=True)
        return {"ready": ready, "chunks": chunks}

    def platform_status(self) -> dict[str, Any]:
        """Non-invasive module report for the Coach/Architecture tab and --check."""
        catalog = self.catalog()
        return {
            "modules": ["Dashboard", "Learn", "Practice", "Ask Genie", "Progress", "Revision Hub"],
            "catalog": catalog.summary(),
            "progress": {
                "path": str(self.progress.file),
                **self.progress.stats(),
            },
            "practice": {"max_questions_per_test": self.settings.test_question_count},
            "revision": [
                {"value": item.value, "label": item.label, "emoji": item.emoji}
                for item in RevisionDeliverable
            ],
            "capabilities": self.capabilities.as_dict(),
            "extension_hooks": sorted(
                {
                    capability.hook
                    for capability in self.capabilities.list()
                    if capability.state is not CapabilityState.ACTIVE
                }
            ),
        }

    def health_note(self) -> str:
        """One-line summary used by the settings panel."""
        catalog = self.catalog()
        stats = self.progress.stats()
        return (
            f"{len(catalog.chapters)} chapter(s) across {len(catalog.subjects)} subject(s) · "
            f"{stats['tests']} test(s) · {stats['questions']} question(s) logged · "
            f"{len(self.capabilities.planned())} capability hook(s) waiting"
        )


_SUBJECT_EMOJI: dict[str, str] = {
    "physics": "\U0001f576\ufe0f",  # sunglasses (mechanics/optics stand-in)
    "chemistry": "\U0001f9ea",  # test tube
    "biology": "\U0001f331",  # seedling
}


def _card_status(chapters: int, values: Any) -> str:
    """Coach's one-word verdict for a subject card."""
    if not chapters:
        return "Add chapters"
    if values is None or not values.questions:
        return "Start a test"
    if values.accuracy >= 0.8:
        return "Exam ready"
    if values.accuracy >= 0.6:
        return "Revise weak spots"
    return "Needs work"


def capability_hint(capability: Capability) -> str:
    """UI text for a planned capability (also used when a call raises)."""
    return (
        f"### {capability.title} - planned\n\n"
        f"{capability.description}\n\n"
        f"**Extension point:** `{capability.hook}`\n\n"
        f"**Next step:** {capability.roadmap}"
    )
