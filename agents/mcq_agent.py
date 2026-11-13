"""Generate answerable NEET MCQs through the existing Practice pipeline."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from agents.base import BaseAgent, emit_event
from learning.practice import MAX_TEST_QUESTIONS
from utils.constants import TestScope
from utils.text_utils import clean_text

__all__ = ["MCQGeneratorAgent"]

_COUNT_RE = re.compile(r"\b(\d{1,2})\s*(?:mcqs?|questions?)\b", re.IGNORECASE)
_TARGET_RE = re.compile(r"\b(?:on|about|from|covering)\s+(.+)$", re.IGNORECASE)


class MCQGeneratorAgent(BaseAgent):
    """Create a short MCQ set with a separately displayed answer key."""

    name = "mcq_generator"

    def __init__(self, settings, *, platform: Any, llm=None, prompts=None) -> None:
        super().__init__(settings, llm=llm, prompts=prompts)
        self.platform = platform

    def run(self, state: dict[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
        question = clean_text(state.get("question"))
        target = _topic_target(question)
        count = _requested_count(question, self.settings.mcq_count)
        subject = str(state.get("subject") or "general")
        emit_event(config, self.name, f"Generating up to {count} NEET MCQs on {target}")

        test = self.platform.generate_test(
            scope=TestScope.TOPIC,
            subject=subject,
            target=target,
            count=count,
        )
        notices = list(state.get("notices") or [])
        notices.extend(notice for notice in test.notices if notice not in notices)
        if not test.questions:
            answer = (
                f"I couldn't create a reliable MCQ set for **{target}**. "
                "Try narrowing the topic or check that the answer service is available."
            )
        else:
            lines = [f"### MCQ Practice: {target}", ""]
            for item in test.questions:
                lines.extend([item.render(reveal=False), ""])
            lines.extend(["### Answer Key", ""])
            for item in test.questions:
                if item.correct is None:
                    lines.append(f"- **Q{item.number}:** The answer could not be verified.")
                    continue
                correct_option = item.option_text(item.correct)
                line = f"- **Q{item.number}: ({item.correct}) {correct_option}**"
                if item.explanation:
                    line += f" — **Explanation:** {item.explanation.strip()}"
                else:
                    line += " — **Explanation:** Not provided in the generated answer key."
                lines.append(line)
            answer = "\n".join(lines).strip()

        return {
            "answer": answer,
            "mode": str(state.get("mode") or "mcq_practice"),
            "citations": test.citations,
            "notices": notices,
        }


def _topic_target(question: str) -> str:
    """Prefer the topic after a request preposition; retain the full request otherwise."""
    match = _TARGET_RE.search(question)
    target = match.group(1) if match else question
    target = re.sub(r"\s+(?:for|in)\s+NEET\b.*$", "", target, flags=re.IGNORECASE)
    target = re.sub(r"\s+", " ", target).strip(" .?!")
    return target or "NEET Biology, Chemistry and Physics"


def _requested_count(question: str, default: int) -> int:
    match = _COUNT_RE.search(question)
    requested = int(match.group(1)) if match else int(default)
    return max(2, min(requested, MAX_TEST_QUESTIONS))
