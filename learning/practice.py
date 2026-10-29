"""Practice module service - subject / chapter / topic tests on your own material.

Pipeline::

    scope + target  ->  RAG grounding  ->  MCQ generation prompt
                    ->  tolerant parser ->  PracticeSet (quiz)
                    ->  student answers ->  GradedTest (score, accuracy, weak topics)
                    ->  ProgressStore   (JSON persistence)

Questions are always generated from the indexed textbook passages - the
service never ships a bundled question bank.  The prompt asks for a strict,
line-oriented format and :func:`learning.assessment.parse_questions` copes with
the ways a local model drifts from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from config import Settings
from learning.assessment import GradedTest, MCQuestion, grade, merge_answer_key, parse_questions
from learning.catalog import Chapter, ChapterCatalog
from learning.generation import GenieGenerator, Grounding
from learning.progress import ProgressStore
from utils.constants import Subject, TestScope
from utils.logging_utils import get_logger
from utils.text_utils import clean_text, truncate

__all__ = ["PracticeService", "PracticeSet", "MAX_TEST_QUESTIONS", "render_review"]

#: Upper bound for one test (also the number of answer widgets the UI pre-creates).
MAX_TEST_QUESTIONS = 15


@dataclass(slots=True)
class PracticeSet:
    """A generated, not-yet-answered test."""

    title: str
    scope: str
    subject: str
    questions: list[MCQuestion] = field(default_factory=list)
    chapter: str = ""
    topic: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)
    seconds: float = 0.0
    ok: bool = True
    error: str | None = None
    raw: str = ""

    @property
    def count(self) -> int:
        return len(self.questions)

    @property
    def answered_count(self) -> int:
        return sum(1 for question in self.questions if question.correct is not None)

    def quiz_markdown(self) -> str:
        """Questions + options, with the answers withheld."""
        scope = TestScope.coerce(self.scope)
        head = [
            f"### {scope.emoji} {scope.label} - {self.title}",
            "",
            f"**{self.count} questions** · grounded in "
            f"{len(self.citations)} indexed passage(s)"
            + (f" · chapter `{self.chapter}`" if self.chapter else ""),
        ]
        if not self.questions:
            head.append("\n_No questions could be extracted from the model output._")
        for question in self.questions:
            head.extend(["", "---", "", question.render(reveal=False)])
        return "\n".join(head).strip()

    def widget_updates(self) -> list[dict[str, Any]]:
        """Gradio-friendly radio specs, one per question (choices as `(text, value)`)."""
        specs: list[dict[str, Any]] = []
        for question in self.questions:
            specs.append(
                {
                    "label": f"Q{question.number}. {truncate(question.stem, 90, suffix='...')}",
                    "choices": [
                        (f"({index}) {clean_text(option)}", index)
                        for index, option in enumerate(question.options, start=1)
                    ],
                    "value": None,
                    "visible": True,
                }
            )
        return specs

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "scope": self.scope,
            "subject": self.subject,
            "chapter": self.chapter,
            "topic": self.topic,
            "questions": [question.as_dict() for question in self.questions],
            "citations": self.citations,
            "notices": self.notices,
            "seconds": round(self.seconds, 2),
            "ok": self.ok,
            "error": self.error,
        }


class PracticeService:
    """Creates and grades the tests of the Practice tab."""

    def __init__(
        self,
        settings: Settings,
        generator: GenieGenerator,
        progress: ProgressStore,
        *,
        catalog_provider: Any | None = None,
    ) -> None:
        self.settings = settings
        self.generator = generator
        self.progress = progress
        self._catalog_provider = catalog_provider
        self._sets: dict[str, PracticeSet] = {}
        self.logger = get_logger("learning.practice")

    # ---------------------------------------------------------------- catalog
    def catalog(self) -> ChapterCatalog:
        if self._catalog_provider is None:
            return ChapterCatalog.empty()
        if callable(self._catalog_provider):
            return self._catalog_provider()
        return self._catalog_provider

    # -------------------------------------------------------------- generation
    def generate(
        self,
        *,
        scope: TestScope | str = TestScope.CHAPTER,
        subject: str | None = None,
        target: str = "",
        count: int | None = None,
    ) -> PracticeSet:
        """Build one test.  ``target`` is a chapter name or a free-text topic."""
        resolved_scope = scope if isinstance(scope, TestScope) else TestScope.coerce(scope)
        wanted = max(2, min(int(count or self.settings.test_question_count), MAX_TEST_QUESTIONS))
        chapter = self._match_chapter(target, subject) if resolved_scope is not TestScope.TOPIC else None
        label = (chapter.name if chapter else "") or clean_text(target) or (
            f"{Subject.coerce(subject, default=Subject.GENERAL).label} full syllabus"
            if resolved_scope is TestScope.SUBJECT
            else "Mixed practice"
        )
        title = f"{label} - {resolved_scope.label}"

        query = _search_query(resolved_scope, label, subject)
        passages = max(4, int(round(self.settings.top_k * resolved_scope.retrieval_factor)))
        grounding = self.generator.ground(
            query,
            subject=subject,
            passages=passages,
            prefer_file=chapter.file_name if chapter else None,
        )

        task = self.generator.build_task(
            task_id="mcq_generation",
            system_prompt="mcq_system",
            user_prompt="mcq_user",
            count=wanted,
            mcq_count=wanted,
            subject=Subject.coerce(subject, default=Subject.GENERAL).label,
            scope=resolved_scope.label,
            target=label,
            context=grounding.context or "(no indexed passages were found)",
        )
        raw = ""
        try:
            raw = self.generator.execute(task)
        except Exception as exc:  # generation crash -> empty, explained test
            self.logger.exception("MCQ generation failed")
            return PracticeSet(
                title=title,
                scope=resolved_scope.value,
                subject=str(subject or "general"),
                chapter=chapter.name if chapter else "",
                topic="" if chapter else label,
                notices=list(grounding.notices)
                + [f"The language model could not produce the test: {exc}"],
                citations=grounding.citations,
                ok=False,
                error=str(exc),
            )

        questions = parse_questions(raw)
        if any(question.correct is None for question in questions):
            questions = merge_answer_key(questions, raw)
        questions = [question for question in questions if question.is_valid()]

        notices = list(grounding.notices)
        if len(questions) < wanted:
            if questions:
                notices.append(
                    f"You asked for {wanted} questions and the model produced {len(questions)} "
                    "usable ones. Ask again or widen the target."
                )
            else:
                notices.append(
                    "No well-formed MCQ could be read from the model output. Try the mock backend "
                    "(`NEET_LLM_BACKEND=mock`) to verify the pipeline, or re-run with a clearer target."
                )
                self.logger.warning("MCQ parse returned nothing (raw %d chars)", len(raw or ""))

        result = PracticeSet(
            title=title,
            scope=resolved_scope.value,
            subject=str(subject or (chapter.subject if chapter else "general")),
            questions=questions[:wanted],
            chapter=chapter.name if chapter else "",
            topic="" if chapter else label,
            citations=grounding.citations,
            notices=notices,
            seconds=0.0,
            ok=bool(questions),
            raw=raw,
        )
        self._sets[result.title] = result
        return result

    # ------------------------------------------------------------------ grading
    def grade_set(self, test: PracticeSet, answers: Mapping[int, Any] | Sequence[Any]) -> GradedTest:
        """Score the student's picks (NEET marking) without persisting."""
        return grade(
            test.questions,
            answers,
            marks_correct=self.settings.marks_correct,
            marks_incorrect=self.settings.marks_incorrect,
            title=test.title,
            scope=test.scope,
            subject=test.subject,
            chapter=test.chapter,
            topic=test.topic,
        )

    def submit(
        self,
        test: PracticeSet | str,
        answers: Mapping[int, Any] | Sequence[Any],
        *,
        duration_seconds: float = 0.0,
    ) -> GradedTest:
        """Grade and persist a finished test - the Progress tab reads this back."""
        resolved = test if isinstance(test, PracticeSet) else self._sets.get(str(test))
        if resolved is None:
            empty = GradedTest(
                test_id="unknown",
                title=str(test),
                scope=TestScope.CHAPTER.value,
                subject="general",
                marks_correct=self.settings.marks_correct,
                marks_incorrect=self.settings.marks_incorrect,
            )
            empty.graded = []
            return empty
        result = self.grade_set(resolved, answers)
        result.duration_seconds = duration_seconds
        self.progress.record(result)
        self.logger.info("Test submitted: %s", result.summary_line)
        return result

    def history(self, *, limit: int = 10, subject: str | None = None) -> list[dict[str, Any]]:
        return self.progress.tests(limit=limit, subject=subject)

    def weak_topics(self, *, threshold: float = 0.75, limit: int = 8):
        """Repeatedly-missed topics across every recorded test."""
        return self.progress.weak_areas(threshold=threshold, limit=limit)

    def last_set(self) -> PracticeSet | None:
        """Most recently generated test (used when the page reloads)."""
        if not self._sets:
            return None
        return list(self._sets.values())[-1]

    def forget(self, title: str | None = None) -> None:
        """Drop one (or all) generated set(s) from memory."""
        if title is None:
            self._sets.clear()
        else:
            self._sets.pop(title, None)

    # ----------------------------------------------------------------- helpers
    def _match_chapter(self, target: str, subject: str | None) -> Chapter | None:
        if not target:
            return None
        return self.catalog().find(target, subject)


def _search_query(scope: TestScope, label: str, subject: str | None) -> str:
    """A retrieval query that favours question-able material."""
    if scope is TestScope.SUBJECT:
        return (
            f"{subject or label} important definitions laws formulas exceptions "
            "frequently asked statements"
        )
    return (
        f"{label} concepts definitions formulas conditions exceptions "
        "statement based questions for NEET"
    )


def render_review(result: GradedTest) -> str:
    """Convenience wrapper kept for the UI (identical to the object's method)."""
    return result.report_markdown(reveal=True)
