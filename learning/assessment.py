"""The assessment engine: MCQ parsing, grading and analysis.

The Practice module asks the LLM for a strict, line-oriented MCQ format and
this module turns that text into structured questions.  Local models drift from
the instructions, so the parser is deliberately tolerant:

* ``Q1.`` / ``1.`` / ``Question 1:`` all start a question,
* options may be ``(1)``, ``1)``, ``1)``, ``(A)`` or ``A.``,
* the answer may be ``ANSWER: 2``, ``**Answer:** (2)`` or ``Correct option: B``,
* the explanation and the topic line are optional, and an answer key written
  as its own section is merged back in with :func:`merge_answer_key`.

Grading follows the real NEET marking scheme (+4 / -1, configurable), so the
score the student sees means something.  Scoring only counts *wrong* answers as
a penalty; leaving a question blank costs nothing here (the coach tells the
student that skipping is the exam-hall move, not a practice penalty).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from utils.constants import TestScope
from utils.text_utils import clean_text, hash_text

__all__ = [
    "MCQuestion",
    "GradedQuestion",
    "GradedTest",
    "parse_questions",
    "merge_answer_key",
    "grade",
    "OPTION_LABELS",
]

OPTION_LABELS: tuple[str, ...] = ("1", "2", "3", "4")

_LET_TO_NUMBER = {"a": 1, "b": 2, "c": 3, "d": 4}

_QUESTION_RE = re.compile(
    r"^\s*(?:\*\*)?\s*(?P<prefix>q(?:uestion)?\s*)?(?P<number>\d{1,3})\s*[.)::]\s*(?:\*\*)?\s*(?P<stem>.*)$",
    re.IGNORECASE,
)
_OPTION_RE = re.compile(
    r"^\s*(?:\*\*)?\s*\(?\s*(?P<label>[1-4ABCDabcd])\s*[).]\s*(?:\*\*)?\s*(?P<text>.+?)\s*$"
)
_ANSWER_RE = re.compile(
    r"^\s*(?:\*\*)?\s*(?:correct\s+(?:answer|option)|answer|ans)\s*(?:\*\*)?\s*[:=]?\s*"
    r"(?:\*\*)?\s*\(?\s*(?P<label>[1-4ABCDabcd])\s*\)?",
    re.IGNORECASE,
)
_EXPLANATION_RE = re.compile(
    r"^\s*(?:\*\*)?\s*(?:explanation|solution|reason(?:ing)?)\s*(?:\*\*)?\s*[:=]\s*(?P<body>.*)$",
    re.IGNORECASE,
)
_TOPIC_RE = re.compile(
    r"^\s*(?:\*\*)?\s*(?:topic|concept|chapter|subtopic|tag)\s*(?:\*\*)?\s*[:=]\s*(?P<body>.*)$",
    re.IGNORECASE,
)
_KEY_LINE_RE = re.compile(
    r"^\s*(?:[-*\u2022]\s*)?(?:\*\*)?\s*(?:q(?:uestion)?\s*)?(?P<number>\d{1,3})\s*[.)::]?\s*"
    r"(?:\*\*)?\s*(?:->|:|=|is)?\s*(?:\*\*)?\s*\(?\s*(?P<label>[1-4ABCDabcd])\s*\)?\s*[.)::]?\s*(?P<rest>.*)$",
    re.IGNORECASE,
)
#: A ``1: (2) - reason`` line is an answer key, even though ``1:`` also looks like
#: a question number.  Checked against the captured stem of a question line.
_KEY_STEM_RE = re.compile(r"^\s*\(?\s*[1-4ABCDabcd]\s*[).]\s*")
#: ``Answer Key`` / ``Answers`` heading that opens the trailing key block.  The
#: lines after it are answers, not study text, so they never reach a stem or an
#: explanation (they are merged back through :func:`merge_answer_key`).
_KEY_HEADING_RE = re.compile(
    r"^\s*(?:[-*\u2022]\s*)?(?:\*\*)?\s*(?:answer\s*key|answer\s*list|answers|key)\s*:?\s*(?:\*\*)?\s*$",
    re.IGNORECASE,
)


def _label_to_number(raw: str) -> int | None:
    """``'3'``, ``'(b)'`` or ``'C'`` -> a 1-based option number."""
    token = str(raw).strip().lower().strip("().: ")
    if token.isdigit():
        value = int(token)
        return value if 1 <= value <= 4 else None
    return _LET_TO_NUMBER.get(token)


@dataclass(frozen=True, slots=True)
class MCQuestion:
    """One NEET-style single-correct question."""

    number: int
    stem: str
    options: tuple[str, ...]
    correct: int | None = None
    explanation: str = ""
    topic: str = ""

    @property
    def option_labels(self) -> tuple[str, ...]:
        return OPTION_LABELS[: len(self.options)]

    def option_text(self, number: int | None) -> str:
        if number is None or not (1 <= number <= len(self.options)):
            return "-"
        return self.options[number - 1]

    def is_valid(self) -> bool:
        """A usable question needs a stem and at least two distinct options."""
        if not self.stem.strip() or len(self.options) < 2:
            return False
        return self.correct is None or 1 <= self.correct <= len(self.options)

    def render(self, *, reveal: bool = False) -> str:
        """Markdown for the UI - answer/explanation only when revealed."""
        lines = [f"**Q{self.number}.** {self.stem.strip()}", ""]
        for index, option in enumerate(self.options, start=1):
            marker = "  "
            if reveal and self.correct == index:
                marker = "\u2705"
            lines.append(f"{marker} **({index})** {option.strip()}")
        if reveal:
            lines.extend(["", f"**Correct answer:** ({self.correct or '?'})"])
            if self.explanation:
                lines.append(f"**Explanation:** {self.explanation.strip()}")
            if self.topic:
                lines.append(f"**Topic:** {self.topic.strip()}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "stem": self.stem,
            "options": list(self.options),
            "correct": self.correct,
            "explanation": self.explanation,
            "topic": self.topic,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MCQuestion":
        options = tuple(str(option) for option in payload.get("options") or ())
        correct = payload.get("correct")
        return cls(
            number=int(payload.get("number") or 1),
            stem=str(payload.get("stem") or ""),
            options=options,
            correct=int(correct) if isinstance(correct, (int, float, str)) and str(correct).isdigit() else None,
            explanation=str(payload.get("explanation") or ""),
            topic=str(payload.get("topic") or ""),
        )


@dataclass(frozen=True, slots=True)
class GradedQuestion:
    """A question paired with what the student chose."""

    question: MCQuestion
    given: int | None = None

    @property
    def attempted(self) -> bool:
        return self.given is not None

    @property
    def is_correct(self) -> bool:
        return self.given is not None and self.question.correct == self.given

    @property
    def topic(self) -> str:
        return (self.question.topic or "General").strip() or "General"

    def as_dict(self) -> dict[str, Any]:
        return {
            "number": self.question.number,
            "topic": self.topic,
            "given": self.given,
            "correct": self.question.correct,
            "ok": self.is_correct,
            "stem": self.question.stem,
        }


@dataclass(slots=True)
class GradedTest:
    """A completed test: the questions, the answers and the derived analytics."""

    test_id: str
    title: str
    scope: str
    subject: str
    graded: list[GradedQuestion] = field(default_factory=list)
    marks_correct: float = 4.0
    marks_incorrect: float = -1.0
    created_at: str = ""
    duration_seconds: float = 0.0
    chapter: str = ""
    topic: str = ""

    # ------------------------------------------------------------- computation
    @property
    def total(self) -> int:
        return len(self.graded)

    @property
    def correct_count(self) -> int:
        return sum(1 for item in self.graded if item.is_correct)

    @property
    def wrong_count(self) -> int:
        return sum(1 for item in self.graded if item.attempted and not item.is_correct)

    @property
    def skipped_count(self) -> int:
        return sum(1 for item in self.graded if not item.attempted)

    @property
    def score(self) -> float:
        """NEET marking: +marks for a correct answer, -marks for a wrong one."""
        return round(
            self.correct_count * self.marks_correct + self.wrong_count * self.marks_incorrect, 2
        )

    @property
    def max_score(self) -> float:
        return round(self.total * self.marks_correct, 2)

    @property
    def accuracy(self) -> float:
        """Share of *attempted* answers that were right (0.0 when none tried)."""
        attempted = self.total - self.skipped_count
        return round(self.correct_count / attempted, 4) if attempted else 0.0

    @property
    def attempted(self) -> int:
        return self.total - self.skipped_count

    def topic_breakdown(self) -> dict[str, dict[str, Any]]:
        """Per-topic attempts / correctness - the raw material of weak topics."""
        buckets: dict[str, dict[str, Any]] = {}
        for item in self.graded:
            bucket = buckets.setdefault(item.topic, {"topic": item.topic, "total": 0, "correct": 0, "wrong": 0})
            bucket["total"] += 1
            if item.is_correct:
                bucket["correct"] += 1
            elif item.attempted:
                bucket["wrong"] += 1
        for bucket in buckets.values():
            attempted = bucket["correct"] + bucket["wrong"]
            bucket["accuracy"] = round(bucket["correct"] / attempted, 4) if attempted else 0.0
        return dict(sorted(buckets.items(), key=lambda kv: (kv[1]["accuracy"], -kv[1]["total"])))

    def weak_topics(self, *, threshold: float = 0.7) -> list[str]:
        """Topics at or below ``threshold`` accuracy, worst first."""
        return [
            name
            for name, stats in self.topic_breakdown().items()
            if stats["total"] and stats["accuracy"] < threshold
        ]

    def as_record(self) -> dict[str, Any]:
        """JSON-friendly form persisted by the progress store."""
        return {
            "test_id": self.test_id,
            "title": self.title,
            "scope": self.scope,
            "subject": self.subject,
            "chapter": self.chapter,
            "topic": self.topic,
            "created_at": self.created_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "duration_seconds": round(float(self.duration_seconds), 2),
            "marks_correct": self.marks_correct,
            "marks_incorrect": self.marks_incorrect,
            "total": self.total,
            "correct": self.correct_count,
            "wrong": self.wrong_count,
            "skipped": self.skipped_count,
            "score": self.score,
            "max_score": self.max_score,
            "accuracy": self.accuracy,
            "weak_topics": self.weak_topics(),
            "topics": list(self.topic_breakdown().values()),
            "answers": [item.as_dict() for item in self.graded],
        }

    @property
    def summary_line(self) -> str:
        scope = TestScope.coerce(self.scope)
        return (
            f"{scope.emoji} {scope.label} - {self.title}:"
            f" {self.correct_count}/{self.total} correct, accuracy {self.accuracy * 100:.0f}%,"
            f" score {self.score:g}/{self.max_score:g}"
        )

    def report_markdown(self, *, reveal: bool = True) -> str:
        """Score header plus one block per question."""
        head = [
            f"### \U0001f9e9 {self.title}",
            "",
            f"**Score:** {self.score:g} / {self.max_score:g} · "
            f"**Correct:** {self.correct_count}/{self.total} · "
            f"**Accuracy:** {self.accuracy * 100:.0f}% · "
            f"**Skipped:** {self.skipped_count}",
        ]
        weak = self.weak_topics()
        if weak:
            head.append("\n**Weak topics:** " + ", ".join(weak))
        for item in self.graded:
            head.extend(["", "---", "", item.question.render(reveal=reveal)])
            if reveal and item.attempted and not item.is_correct:
                head.append(
                    "\n_You chose ({}) - the trap is in the option you did not rule out._".format(item.given)
                )
        return "\n".join(head).strip()


def parse_questions(text: str) -> list[MCQuestion]:
    """Extract structured questions from model output (best effort, no raise)."""
    body = clean_text(text)
    if not body:
        return []

    questions: list[MCQuestion] = []
    state: dict[str, Any] = {
        "number": None,
        "stem": [],
        "options": [],
        "correct": None,
        "explanation": [],
        "topic": "",
        "field": "stem",
    }
    in_key = False
    key_lines: list[str] = []

    def flush() -> None:
        number = state["number"]
        if number is None:
            return
        stem = clean_text(" ".join(part for part in state["stem"] if part))
        options = [option.strip() for option in state["options"] if option.strip()][:4]
        correct = state["correct"]
        if correct is not None and correct > len(options):
            correct = None
        if not stem and not options:
            state.update({"number": None, "stem": [], "options": [], "correct": None,
                          "explanation": [], "topic": "", "field": "stem"})
            return
        questions.append(
            MCQuestion(
                number=number,
                stem=stem or "(question text missing)",
                options=tuple(options),
                correct=correct,
                explanation=clean_text(" ".join(part for part in state["explanation"] if part)),
                topic=str(state["topic"]),
            )
        )
        state.update({"number": None, "stem": [], "options": [], "correct": None,
                      "explanation": [], "topic": "", "field": "stem"})

    for line in body.split("\n"):
        raw = line.rstrip()
        if not raw.strip():
            continue

        if _KEY_HEADING_RE.match(raw):
            flush()
            in_key = True
            continue
        if in_key:
            key_lines.append(raw)
            continue

        answer_match = _ANSWER_RE.match(raw)
        if answer_match and state["number"] is not None:
            state["correct"] = _label_to_number(answer_match.group("label"))
            state["field"] = "explanation"
            continue

        explanation_match = _EXPLANATION_RE.match(raw)
        if explanation_match and state["number"] is not None:
            state["field"] = "explanation"
            tail = explanation_match.group("body").strip()
            if tail:
                state["explanation"].append(tail)
            continue

        topic_match = _TOPIC_RE.match(raw)
        if topic_match and state["number"] is not None:
            state["topic"] = topic_match.group("body").strip().rstrip("*").strip()
            state["field"] = "explanation"
            continue

        question_match = _QUESTION_RE.match(raw)
        if question_match and _KEY_STEM_RE.match(question_match.group("stem") or ""):
            question_match = None  # ``1: (2) - reason`` belongs to an answer key
        option_match = _OPTION_RE.match(raw)
        candidate = int(question_match.group("number")) if question_match else None
        starts_question = False
        if question_match:
            if state["number"] is None:
                # ``Q1.``/``1.`` opening the first (or next) question.
                starts_question = candidate == len(questions) + 1 or bool(question_match.group("prefix"))
            else:
                # ``1.`` - ``4.`` double as option markers.  A line that continues
                # the option sequence of the question in hand is an option, and an
                # unlabelled line only starts a new question when it advances the
                # numbering past the question being read.
                explicit = bool(question_match.group("prefix"))
                continues_options = candidate == len(state["options"]) + 1 and not explicit
                advances = candidate > int(state["number"]) and len(state["options"]) >= 2
                starts_question = (explicit or advances) and not continues_options

        if starts_question:
            flush()
            state.update(
                {
                    "number": candidate,
                    "stem": [question_match.group("stem").strip()],
                    "options": [],
                    "correct": None,
                    "explanation": [],
                    "topic": "",
                    "field": "stem",
                }
            )
            continue

        if (
            option_match
            and state["number"] is not None
            and len(state["options"]) < 4
            and any(part.strip() for part in state["stem"])
            and _label_to_number(option_match.group("label")) == len(state["options"]) + 1
        ):
            state["options"].append(option_match.group("text"))
            state["field"] = "options"
            continue

        if state["number"] is None:
            continue
        if state["field"] == "explanation":
            state["explanation"].append(raw.strip())
        elif state["field"] == "stem":
            state["stem"].append(raw.strip())
        # a stray heading line (``Answer Key``, ``Hints``) between options is
        # dropped rather than swallowed into the question text

    flush()

    if key_lines:
        questions = merge_answer_key(questions, "\n".join(key_lines))

    ordered: list[MCQuestion] = []
    seen: set[str] = set()
    for question in questions:
        fingerprint = question.stem.lower()[:80]
        if fingerprint and fingerprint in seen:
            continue
        seen.add(fingerprint)
        if question.number != len(ordered) + 1:
            question = MCQuestion(
                number=len(ordered) + 1,
                stem=question.stem,
                options=question.options,
                correct=question.correct,
                explanation=question.explanation,
                topic=question.topic,
            )
        ordered.append(question)
    return ordered


def merge_answer_key(questions: Sequence[MCQuestion], text: str) -> list[MCQuestion]:
    """Fill missing answers/explanations from a separate ``Answer Key`` block.

    The chat's MCQ Practice mode writes questions and key as two sections, so
    the same parser can serve both paths.
    """
    merged: list[MCQuestion] = []
    key: dict[int, tuple[int | None, str]] = {}
    for line in clean_text(text).split("\n"):
        match = _KEY_LINE_RE.match(line)
        if not match:
            continue
        label = _label_to_number(match.group("label"))
        if label is None:
            continue
        rest = (match.group("rest") or "").strip().strip("-").strip()
        number = int(match.group("number"))
        if number not in key:
            key[number] = (label, rest)
    for question in questions:
        entry = key.get(question.number)
        if entry is None:
            merged.append(question)
            continue
        answer, note = entry
        merged.append(
            MCQuestion(
                number=question.number,
                stem=question.stem,
                options=question.options,
                correct=question.correct or answer,
                explanation=question.explanation or note,
                topic=question.topic,
            )
        )
    return merged


def grade(
    questions: Sequence[MCQuestion],
    answers: Mapping[int, Any] | Sequence[Any],
    *,
    marks_correct: float = 4.0,
    marks_incorrect: float = -1.0,
    title: str = "Practice test",
    scope: TestScope | str = TestScope.CHAPTER,
    subject: str = "general",
    chapter: str = "",
    topic: str = "",
    duration_seconds: float = 0.0,
) -> GradedTest:
    """Score ``answers`` against ``questions`` and build the analytics object."""
    picked = _coerce_answers(answers, len(questions))
    graded = [
        GradedQuestion(question=question, given=picked.get(question.number))
        for question in questions
    ]
    resolved_scope = scope if isinstance(scope, TestScope) else TestScope.coerce(scope)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return GradedTest(
        test_id=hash_text(f"{stamp}-{title}-{len(questions)}", length=12),
        title=title,
        scope=resolved_scope.value,
        subject=subject,
        chapter=chapter,
        topic=topic,
        graded=graded,
        marks_correct=float(marks_correct),
        marks_incorrect=float(marks_incorrect),
        created_at=stamp,
        duration_seconds=duration_seconds,
    )


def _coerce_answers(
    answers: Mapping[int, Any] | Sequence[Any] | None, count: int
) -> dict[int, int | None]:
    """Map Gradio checkbox/radio values onto ``{question_number: option|None}``."""
    out: dict[int, int | None] = {}
    if answers is None:
        return out
    if isinstance(answers, Mapping):
        items: Iterable[tuple[Any, Any]] = answers.items()
    else:
        items = enumerate(answers, start=1)
    for key, value in items:
        try:
            number = int(key)
        except (TypeError, ValueError):
            continue
        if not 1 <= number <= count:
            continue
        if isinstance(value, str):
            value = _label_from_option_string(value)
        if isinstance(value, bool):
            value = 1 if value else None
        out[number] = int(value) if isinstance(value, (int, float)) and 1 <= int(value) <= 4 else None
    return out


def _label_from_option_string(value: str) -> int | None:
    """``"(2) Some text"`` or ``"Some text"`` -> the option number, if findable."""
    match = re.match(r"^\s*\(?\s*([1-4ABCDabcd])\s*[).]?", value)
    if match:
        return _label_to_number(match.group(1))
    return None
