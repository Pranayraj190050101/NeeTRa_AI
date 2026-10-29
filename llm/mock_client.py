"""Deterministic offline LLM used for development, tests and demos.

It lets you boot the whole stack (Gradio -> LangGraph -> FAISS) on a laptop
without downloading a 7B model.  It is *not* a tutor: answers are canned, but
they prove that routing, retrieval, the coach answer modes and citations work
end-to-end.

Enable with ``NEET_LLM_BACKEND=mock``.
"""

from __future__ import annotations

import json
import re
from typing import Sequence

from config import Settings
from llm.base import ChatMessage, LLMClient, TokenCallback
from utils.answer_format import sections_for
from utils.constants import AnswerMode, Intent, Subject, intent_from_text, subject_from_text

__all__ = ["MockLLM"]

_QUESTION_MARKERS: tuple[str, ...] = (
    "STUDENT'S LATEST MESSAGE",
    "STUDENT'S QUESTION",
    "Student's question:",
)
_REFERENCE_RE = re.compile(r"^\[(\d+)\]", re.MULTILINE)
_SECTION_BREAK_RE = re.compile(r"\n\s*\n(?=[A-Z][A-Z' ]{4,})")
#: Written into the system prompt by the answer agent (see ``AnswerAgent.system_prompt``).
_MODE_ID_RE = re.compile(r"MODE_ID:[ \t]*([a-z_]+)")
#: Written by the platform services (see ``learning.generation.GenerationTask``).
_TASK_ID_RE = re.compile(r"TASK_ID:[ \t]*([a-z_]+)")
_TASK_SECTIONS_RE = re.compile(r"^SECTIONS:[ \t]*(?P<titles>.+?)[ \t]*$", re.MULTILINE)
_TASK_COUNT_RE = re.compile(r"^COUNT:[ \t]*(?P<count>\d+)[ \t]*$", re.MULTILINE)
_TASK_TARGET_RE = re.compile(r"^(?:Target|Chapter):[ \t]*(?P<target>.+?)[ \t]*$", re.MULTILINE)

#: Canned bodies for the platform modules, keyed by canonical section title.
_PLATFORM_BODIES: dict[str, str] = {
    "\U0001f4d8 Chapter Summary": (
        "- The chapter's argument, rebuilt line by line from the indexed passages.\n"
        "- Mock mode only proves that the Learn page reaches the UI with citations."
    ),
    "\U0001f3af High-Yield Concepts": (
        "- Definition lines and stated conditions become statement questions.\n"
        "- Exceptions are the cheapest marks in the paper."
    ),
    "\U0001f4dd Important NEET Points": (
        "- Keep the exact textbook wording for values, ranges and names.\n"
        "- Every point here is quoted from a numbered reference block."
    ),
    "\u26a1 Quick Revision Notes": (
        "- term -> recall cue\n"
        "- condition -> result\n"
        "- formula -> SI unit"
    ),
    "\U0001f5d2\ufe0f Flash Cards": (
        "- **NCERT line** -> the fact it is testing.\n"
        "- One card per definition, condition or exception."
    ),
    "\U0001f9e0 Trigger -> Recall": (
        "- Name of the law -> statement plus validity condition.\n"
        "- Exception -> the chapter it belongs to."
    ),
    "\u2705 Verify In NCERT": (
        "- Re-read the cited passage before trusting a card.\n"
        "- Anything marked 'not in your index' needs the chapter PDF."
    ),
    "\U0001f9ee Formula Sheet": (
        "- Formula with every symbol defined on the same line.\n"
        "- Mock sheets never invent constants or units."
    ),
    "\U0001f4d0 Symbols And Units": (
        "- symbol -> quantity -> SI unit.\n"
        "- Dimensional checks catch most numerical errors."
    ),
    "\u26a0 Conditions And Traps": (
        "- Where the formula stops applying.\n"
        "- Sign, unit and off-by-one coefficient slips."
    ),
    "\u26a1 High-Yield Points": (
        "- Lines the paper reuses almost verbatim.\n"
        "- Exceptions and named experiments."
    ),
    "\U0001f501 Repeatedly Asked Lines": (
        "- Statement-based questions are built from these sentences."
    ),
    "\u2705 Must Memorise": (
        "- Values, ranges and examples that carry no reasoning."
    ),
    "\u23f1\ufe0f 60-Second Recap": (
        "- One line per topic, read in the order of the chapter."
    ),
    "\u26a0 Final Mistakes To Avoid": (
        "- Answering what is not asked.\n"
        "- Ignoring the stated condition."
    ),
    "\u2705 Exam-Hall Checklist": (
        "- Formula sheet, unit list, marked NCERT lines."
    ),
}
_PLATFORM_DEFAULT_BODY = "- Offline mock line for this section (no language model loaded)."

#: Canned MCQ stems so the Practice pipeline can be graded offline.
_MOCK_TOPIC_FALLBACK = "General mock topic"

#: Canned bodies so the mock reply already satisfies every section contract and
#: the format-repair layer has nothing to invent.
_MOCK_BODIES: dict[str, str] = {
    "\U0001f4d8 Concept Explanation": (
        "- The real explanation would be quoted from the numbered NCERT passages above "
        "and expanded line by line.\n"
        "- Mock mode only proves that the coach section layout reaches the chat UI."
    ),
    "\U0001f3af NEET Importance": (
        "- Topics like this are asked as statement-based and assertion-reason items.\n"
        "- A real run scores each section against your indexed chapters."
    ),
    "\U0001f9e0 Key Points To Remember": (
        "- Definition, condition of validity, one exception - three recall lines.\n"
        "- Keep the NCERT wording, because NEET options reuse it verbatim."
    ),
    "\u26a0 Common Student Mistakes": (
        "- Paraphrasing the textbook and then losing marks on factual options.\n"
        "- Ignoring units, signs and the stated conditions of a formula."
    ),
    "\u2705 Quick Revision Notes": (
        "- Re-read the cited passage today, then again after three days.\n"
        "- Solve the NCERT back-exercise questions on this topic."
    ),
    "\U0001f4dc How NEET Asks This": (
        "- PYQs on a topic like this are usually re-worded NCERT lines.\n"
        "- Index your previous-year papers under `data/<subject>/` to let the coach quote them."
    ),
    "\U0001f9e9 Practice MCQs": (
        "Q1. Which of the following best describes the concept asked about above?\n"
        "(1) Option one - a plausible distractor\n"
        "(2) Option two - the NCERT position\n"
        "(3) Option three - a common misconception\n"
        "(4) Option four - true but irrelevant\n\n"
        "_(mock question - a real run writes the practice set from your indexed pages)_"
    ),
    "\U0001f511 Answer Key With Explanation": (
        "Q1: (2). The correct option restates the textbook line; the others are the usual "
        "near-miss traps."
    ),
    "\u26a0 Traps Tested In NEET": (
        "- 'True but irrelevant' options - answer what was actually asked.\n"
        "- Negative marking: NEET awards +4/-1, so skip what you cannot justify."
    ),
}
_MOCK_DEFAULT_BODY = "- Offline mock placeholder for this section."


class MockLLM(LLMClient):
    """A tiny rule-based stand-in for Qwen that never touches the network."""

    name = "mock"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings

    # ------------------------------------------------------------------ public
    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        on_token: TokenCallback | None = None,
    ) -> str:
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = messages[-1]["content"] if messages else ""

        if self._is_routing_prompt(system, user):
            text = self._routing_json(user)
        else:
            task = self._task_from_system(system)
            if task == "mcq_generation":
                text = self._mock_mcq_set(system, user)
            elif task in ("learn_chapter", "revision_notes"):
                text = self._mock_section_doc(system, task, user)
            else:
                text = self._tutor_answer(user, system)

        if on_token is not None:
            for start in range(0, len(text), 24):
                on_token(text[start : start + 24])
        return text

    def describe(self) -> dict[str, object]:
        return {
            "backend": self.name,
            "model": "rule-based stub",
            "device": "cpu",
            "note": "Offline mock - set NEET_LLM_BACKEND=hf for real answers.",
        }

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _is_routing_prompt(system: str, user: str) -> bool:
        """Routing prompts ask for JSON with ``subject`` and ``intent`` keys."""
        haystack = f"{system}\n{user}".lower()
        return "json" in haystack and '"subject"' in haystack and '"intent"' in haystack

    def _routing_json(self, user: str) -> str:
        question = self._extract_question(user)
        subject = subject_from_text(question) or Subject.GENERAL
        intent = intent_from_text(question) or Intent.DOUBT
        return json.dumps(
            {
                "subject": subject.value,
                "intent": intent.value,
                "confidence": 0.55,
                "reasoning": "mock keyword classifier",
            }
        )

    @staticmethod
    def _extract_question(prompt: str) -> str:
        """Pull the student's message out of a rendered prompt template."""
        for marker in _QUESTION_MARKERS:
            if marker in prompt:
                tail = prompt.split(marker, 1)[1].lstrip()
                match = _SECTION_BREAK_RE.search(tail)
                return (tail[: match.start()] if match else tail).strip()
        return prompt.strip()

    @staticmethod
    def _mode_from_system(system: str) -> AnswerMode:
        """Recover the requested answer mode from the ``MODE_ID:`` marker line."""
        match = _MODE_ID_RE.search(system or "")
        return AnswerMode.coerce(match.group(1) if match else None)

    # -------------------------------------------------------- platform tasks
    @staticmethod
    def _task_from_system(system: str) -> str | None:
        """Recover the platform task id from the ``TASK_ID:`` marker line."""
        match = _TASK_ID_RE.search(system or "")
        return match.group(1) if match else None

    @staticmethod
    def _task_sections(system: str) -> list[str]:
        """Section titles the service promised, read from ``SECTIONS:``."""
        match = _TASK_SECTIONS_RE.search(system or "")
        if not match:
            return []
        return [title.strip() for title in match.group("titles").split("|") if title.strip()]

    @staticmethod
    def _task_count(system: str) -> int:
        """How many MCQs the mock should write (``COUNT:`` marker)."""
        match = _TASK_COUNT_RE.search(system or "")
        return max(2, int(match.group("count"))) if match else 3

    @staticmethod
    def _task_target(user: str) -> str:
        """The chapter / topic / target label the request was about."""
        match = _TASK_TARGET_RE.search(user or "")
        if not match:
            return _MOCK_TOPIC_FALLBACK
        return match.group("target").strip().strip("*").strip() or _MOCK_TOPIC_FALLBACK

    def _mock_section_doc(self, system: str, task: str, user: str = "") -> str:
        """Canned study document in exactly the sections the service promised."""
        sections = self._task_sections(system) or list(_PLATFORM_BODIES)[:3]
        blocks = [
            "### Mock study material (no language model loaded)",
            "",
            f"**Task:** {task} · **Passages available:** "
            f"{len(set(_REFERENCE_RE.findall(user or system)))}",
            "",
            "This page is produced by the offline `mock` backend so the Learn and "
            "Revision Hub pipelines - retrieval, prompting, section contract and "
            "citations - can be verified without a 7B model.",
        ]
        for title in sections:
            body = _PLATFORM_BODIES.get(title, _PLATFORM_DEFAULT_BODY)
            blocks.extend(["", f"### {title}", "", body])
        blocks.extend(
            [
                "",
                "Run with `NEET_LLM_BACKEND=hf` for real NEET-level material (see README).",
                "",
                "_Mock mode writes no facts and must not be used for actual studying._",
            ]
        )
        return "\n".join(blocks)

    def _mock_mcq_set(self, system: str, user: str) -> str:
        """Canned, machine-parseable MCQs so Practice can be graded offline."""
        target = self._task_target(user)
        count = self._task_count(system)
        flavours = (
            "a direct NCERT-statement question",
            "a numerical or unit question",
            "an assertion-reason question",
            "an exception-to-the-rule question",
            "a condition-of-validity question",
        )
        blocks: list[str] = []
        for number in range(1, count + 1):
            answer = ((number - 1) % 4) + 1
            stem = (
                f"{target}: which of the following statements is correct "
                f"({flavours[(number - 1) % len(flavours)]}, mock item)?"
            )
            blocks.append(
                "\n".join(
                    [
                        f"Q{number}. {stem}",
                        "(1) A plausible distractor - a near-miss definition",
                        "(2) The textbook statement as written in the passage",
                        "(3) A common misconception about the condition",
                        "(4) True in general but not what the question asked",
                        f"ANSWER: {answer}",
                        "EXPLANATION: The keyed option keeps the NCERT wording; the "
                        "strongest distractor blurs the stated condition. [1]",
                        f"TOPIC: {target}",
                    ]
                )
            )
        return "\n\n".join(blocks)

    def _tutor_answer(self, user: str, system: str = "") -> str:
        mode = self._mode_from_system(system)
        extracted = self._extract_question(user)
        first_line = extracted.splitlines()[0][:200] if extracted else "(empty)"
        references = len(set(_REFERENCE_RE.findall(user)))

        blocks: list[str] = [
            "### Mock answer (no language model loaded)",
            "",
            f"**Question received:** {first_line}",
            "",
            f"**Answer mode:** {mode.emoji} {mode.label} (offline mock reply).",
            "",
            "This reply comes from the offline `mock` backend, which proves that the "
            "agent graph, FAISS retrieval, the coach section contract and citation "
            f"plumbing are working. Retrieved reference blocks available to the "
            f"generator: **{references}**."
        ]
        for title in sections_for(mode):
            blocks.extend(["", f"### {title}", "", _MOCK_BODIES.get(title, _MOCK_DEFAULT_BODY)])
        blocks.extend(
            [
                "",
                "To get real NEET-level explanations, install the model dependencies and "
                "run the app with `NEET_LLM_BACKEND=hf` (see README).",
                "",
                "_Mock mode performs no reasoning and must not be used for actual studying._",
            ]
        )
        return "\n".join(blocks)
