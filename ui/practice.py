"""Practice tab - subject / chapter / topic MCQ tests with scoring.

The tab is a thin shell over :class:`learning.practice.PracticeService`:
generate from the index, answer, grade with NEET marking and hand the result to
the progress store.  The widgets are pre-created for ``MAX_TEST_QUESTIONS``
questions and hidden as needed, which keeps the Gradio graph static (the safe
pattern across Gradio 4 and 5).
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from learning.assessment import GradedTest, MCQuestion
from learning.practice import MAX_TEST_QUESTIONS, PracticeSet
from learning.service import GeniePlatform
from ui.components import esc, hide_source_markers, notices_md, refresh, subject_choices, table
from utils.constants import TestScope
from utils.text_utils import truncate

__all__ = [
    "build",
    "collect_answers",
    "options_for",
    "practice_intro_md",
    "score_md",
    "submit",
    "wire",
]


def options_for(question: MCQuestion) -> list[tuple[str, int]]:
    """Radio choices of one question: ``(option text, option number)``."""
    return [
        (f"({index}) {esc(option)}", index)
        for index, option in enumerate(question.options, start=1)
    ]


def question_label(question: MCQuestion) -> str:
    return f"Q{question.number}. {truncate(esc(question.stem), 90, suffix='...')}"


def practice_intro_md(platform: GeniePlatform, settings: Any) -> str:
    """How the module works, with the live marking scheme and chapter counts."""
    catalog = platform.catalog()
    scope_rows = [(scope.emoji + " " + scope.label, scope.description) for scope in TestScope]
    return "\n".join(
        [
            "### \U0001f9e9 Practice - test yourself and learn from every mistake",
            "",
            table(("Test type", "What it does"), scope_rows),
            "",
            f"- Questions per test: 2-{MAX_TEST_QUESTIONS} (default {settings.test_question_count})",
            f"- Marking: +{settings.marks_correct:g} correct, {settings.marks_incorrect:g} wrong "
            "(negative marking is on, exactly as in NEET)",
            f"- Chapters available: {len(catalog.chapters)} across your study library",
            "- Every submitted test updates your Dashboard, weak areas and study streak",
        ]
    )


def score_md(result: GradedTest) -> str:
    """Score, accuracy and weak topics as one block."""
    weak = result.weak_topics()
    rows = [
        (
            result.title,
            f"{result.score:g}/{result.max_score:g}",
            f"{result.correct_count}/{result.total}",
            f"{result.accuracy * 100:.0f}%",
            result.wrong_count,
            result.skipped_count,
            ", ".join(weak) or "none",
        )
    ]
    header = table(
        ("Test", "Score", "Correct", "Accuracy", "Wrong", "Skipped", "Weak topics"), rows
    )
    breakdown = table(
        ("Topic", "Asked", "Correct", "Wrong", "Accuracy"),
        [
            (
                stats["topic"],
                stats["total"],
                stats["correct"],
                stats["wrong"],
                f"{stats['accuracy'] * 100:.0f}%",
            )
            for stats in result.topic_breakdown().values()
        ],
    )
    return f"{header}\n\n**Topic breakdown**\n\n{breakdown}"


def collect_answers(values: Sequence[Any]) -> list[int | None]:
    """Radio values -> the ordered answer list the grader expects."""
    answers: list[int | None] = []
    for value in values:
        if value in (None, "", "None"):
            answers.append(None)
            continue
        try:
            answers.append(int(value))
        except (TypeError, ValueError):
            answers.append(None)
    return answers


def generate(platform: GeniePlatform, *, scope: str, subject: str, target: str, count: float) -> PracticeSet:
    """Ask the Practice service for one test."""
    return platform.generate_test(
        scope=TestScope.coerce(scope),
        subject=None if subject in (None, "", "auto", "general") else subject,
        target=esc(target),
        count=int(count),
    )


def submit(platform: GeniePlatform, test: PracticeSet | None, values: Sequence[Any], *, started: float | None) -> GradedTest | None:
    """Grade the student's picks and persist the attempt."""
    if test is None or not test.questions:
        return None
    duration = round(time.monotonic() - started, 2) if started else 0.0
    return platform.submit_test(test, collect_answers(values), duration_seconds=duration)


# ---------------------------------------------------------------------------
# Gradio layout
# ---------------------------------------------------------------------------
def build(
    gr: Any,
    platform: GeniePlatform,
    settings: Any,
    nav: Any | None = None,
    *,
    targets: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the Practice tab and wire it to :class:`PracticeService`."""
    catalog = platform.catalog()
    intro_md = gr.Markdown(practice_intro_md(platform, settings))

    with gr.Row():
        scope_box = gr.Radio(
            choices=TestScope.ui_choices(),
            value=TestScope.CHAPTER.value,
            label="Test type",
            info="Subject mixes the whole subject; chapter tests one PDF; topic drills a concept.",
        )
        subject_box = gr.Dropdown(
            choices=subject_choices(catalog),
            value=(catalog.subjects[0] if catalog.subjects else "physics"),
            label="Subject",
        )
        target_box = gr.Dropdown(
            choices=_target_choices(catalog, TestScope.CHAPTER.value, subject_box.value),
            value=None,
            allow_custom_value=True,
            label="Chapter / topic",
            info="Choose a chapter or type any topic you want to practice.",
        )
        count_box = gr.Slider(
            minimum=2,
            maximum=MAX_TEST_QUESTIONS,
            step=1,
            value=settings.test_question_count,
            label="Questions",
        )

    with gr.Row():
        generate_btn = gr.Button("Generate test", variant="primary", scale=3)
        reset_btn = gr.Button("Clear answers", size="sm", scale=1)
    status_box = gr.Markdown("_No test generated yet._")

    quiz_box = gr.Markdown()
    with gr.Column():
        answer_boxes = [
            gr.Radio(choices=[], label=f"Q{n}", value=None, visible=False)
            for n in range(1, MAX_TEST_QUESTIONS + 1)
        ]
    submit_btn = gr.Button("Submit answers and score", variant="primary", visible=True)

    review_box = gr.Markdown()
    score_box = gr.Markdown()

    weak_btn = gr.Button("Revise my weak areas in the Revision Hub")
    weak_md = gr.Markdown()

    current: Any = gr.State(value=None)
    started: Any = gr.State(value=None)

    refs: dict[str, Any] = {
        "intro": intro_md,
        "scope": scope_box,
        "subject": subject_box,
        "target": target_box,
        "count": count_box,
        "generate": generate_btn,
        "reset": reset_btn,
        "status": status_box,
        "quiz": quiz_box,
        "answers": answer_boxes,
        "submit": submit_btn,
        "review": review_box,
        "score": score_box,
        "state": current,
        "clock": started,
        "weak": weak_btn,
        "weak_md": weak_md,
    }
    wire(gr, platform, refs, nav=nav, targets=targets or {})
    return refs


def _target_choices(catalog: Any, scope: str, subject: str) -> list[tuple[str, str]]:
    """Choices of the target picker for the current scope + subject."""
    item = TestScope.coerce(scope)
    if item is TestScope.SUBJECT:
        names = [f"{subject.title()} - full syllabus"]
        return [(name, name) for name in names] + [
            (chapter.name, chapter.name) for chapter in catalog.list(subject)
        ]
    if item is TestScope.TOPIC:
        topics = sorted({name for name in catalog.names(subject)})
        return [(name, name) for name in topics]
    return [(chapter.name, chapter.name) for chapter in catalog.list(subject)]


def wire(
    gr: Any,
    platform: GeniePlatform,
    refs: dict[str, Any],
    *,
    nav: Any | None = None,
    targets: Mapping[str, Any] | None = None,
) -> None:
    """Bind the Practice tab's controls to the Practice service."""
    answers = refs["answers"]
    outputs_after_status = [refs["quiz"]] + answers + [refs["state"], refs["clock"]]

    def pick_targets(scope: str, subject: str) -> Any:
        choices = _target_choices(platform.catalog(), scope, subject)
        return refresh(choices=choices, value=choices[0][1] if choices and scope == TestScope.SUBJECT.value else None)

    def hidden() -> list[Any]:
        return [refresh(visible=False, choices=[], value=None, label=f"Q{n}") for n in range(1, MAX_TEST_QUESTIONS + 1)]

    def build_test(scope: str, subject: str, target: str, count: float) -> list[Any]:
        test = generate(platform, scope=scope, subject=subject, target=target, count=count)
        updates = hidden()
        for index, question in enumerate(test.questions[:MAX_TEST_QUESTIONS]):
            updates[index] = refresh(
                choices=options_for(question),
                value=None,
                label=question_label(question),
                visible=True,
            )
        status = (
            f"**Status:** {test.count} question(s) generated in {test.scope} mode. "
            "Answer and press Submit."
            if test.questions
            else "**Status:** no usable questions came back - see the notes below."
        )
        return [status, hide_source_markers(test.quiz_markdown())] + updates + [
            test if test.questions else None,
            time.monotonic() if test.questions else None,
        ]

    def score(test: PracticeSet | None, started: float | None, *picks: Any) -> list[Any]:
        if test is None:
            empty = "**Status:** generate a test first."
            return [empty, empty, empty]
        result = submit(platform, test, picks, started=started)
        if result is None:
            status = "**Status:** that test has no questions left in memory - generate it again."
            return [status, "_Generate a new test to answer._", status]
        return [
            f"**Submitted.** {esc(result.summary_line)}",
            hide_source_markers(result.report_markdown(reveal=True)),
            score_md(result),
        ]

    def clear(*_ignored: Any) -> list[Any]:
        return [refresh(value=None) for _ in answers]

    def revise_weak() -> list[Any]:
        areas = platform.practice.weak_topics(limit=6)
        if not areas:
            return [
                "**No weak areas recorded yet.** Take a test above first - the Revision Hub "
                "builds its sheet from the topics you actually lost marks in."
            ] + (nav.tail("revision") if nav is not None else [])
        lines = "\n".join(f"- {esc(area.name)} ({area.subject}, {area.accuracy_percent}%)" for area in areas)
        return [
            "**Your weakest topics so far**\n\n" + lines + "\n\nOpening the Revision Hub: press "
            "*Revise my weak areas* there to build a sheet from exactly these topics."
        ] + (nav.tail("revision") if nav is not None else [])

    refs["scope"].change(pick_targets, inputs=[refs["scope"], refs["subject"]], outputs=[refs["target"]], queue=False)
    refs["subject"].change(pick_targets, inputs=[refs["scope"], refs["subject"]], outputs=[refs["target"]], queue=False)
    refs["generate"].click(
        build_test,
        inputs=[refs["scope"], refs["subject"], refs["target"], refs["count"]],
        outputs=[refs["status"]] + outputs_after_status,
    )
    refs["submit"].click(
        score,
        inputs=[refs["state"], refs["clock"]] + answers,
        outputs=[refs["status"], refs["review"], refs["score"]],
    )
    refs["reset"].click(clear, inputs=answers, outputs=answers, queue=False)
    weak_outputs = [refs["weak_md"]] + (nav.outputs if nav is not None else [])
    refs["weak"].click(revise_weak, inputs=None, outputs=weak_outputs, queue=False)
