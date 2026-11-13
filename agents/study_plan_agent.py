"""Build a syllabus-driven weekly plan, prioritising recorded weak areas."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from agents.base import BaseAgent, emit_event
from utils.constants import normalise_token
from utils.text_utils import clean_text

__all__ = ["StudyPlannerAgent"]

_DURATION_RE = re.compile(r"\b(\d{1,2})\s*-?\s*(weeks?|months?)\b", re.IGNORECASE)
_SUBJECT_ORDER = {"biology": 0, "chemistry": 1, "physics": 2}


class StudyPlannerAgent(BaseAgent):
    """Create a deterministic weekly schedule from indexed chapters and quiz data."""

    name = "study_planner"

    def __init__(self, settings, *, platform: Any, llm=None, prompts=None) -> None:
        super().__init__(settings, llm=llm, prompts=prompts)
        self.platform = platform

    def run(self, state: dict[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
        question = clean_text(state.get("question"))
        match = _DURATION_RE.search(question)
        if match:
            weeks = int(match.group(1)) * (4 if match.group(2).lower().startswith("month") else 1)
        else:
            weeks = 12
        weeks = max(1, min(weeks, 52))
        catalog = self.platform.catalog()
        emit_event(config, self.name, f"Building a {weeks}-week plan from indexed chapters")

        if catalog.is_empty:
            answer = (
                "### Study Plan\n\n"
                "I can't build a syllabus-specific plan until the chapter catalog is indexed. "
                "Add the NEET syllabus chapters to the study library and re-index; I can then "
                "prioritise chapters using your recorded quiz results."
            )
            return {
                "answer": answer,
                "mode": str(state.get("mode") or "detailed"),
                "citations": [],
                "notices": list(state.get("notices") or []),
            }

        weak_areas = self.platform.progress.weak_areas(threshold=0.8, limit=24)
        weak_names = {normalise_token(area.where) for area in weak_areas}
        chapters = sorted(
            catalog.chapters,
            key=lambda chapter: (
                not any(
                    key == normalise_token(chapter.name)
                    or key in normalise_token(chapter.source)
                    for key in weak_names
                ),
                _SUBJECT_ORDER.get(chapter.subject, 3),
                chapter.name.casefold(),
            ),
        )
        chapter_weeks: list[list[Any]] = [[] for _ in range(weeks)]
        for index, chapter in enumerate(chapters):
            week = min(weeks - 1, index * weeks // len(chapters))
            chapter_weeks[week].append(chapter)

        weak_weeks: list[list[Any]] = [[] for _ in range(weeks)]
        for index, area in enumerate(weak_areas):
            week = min(weeks - 1, index * weeks // max(1, len(weak_areas)))
            weak_weeks[week].append(area)

        lines = [
            f"### {weeks}-Week NEET Revision Plan",
            "",
            (
                "Built from your indexed chapter list and recorded quiz results. "
                "Assumption: one focused topic block per study day; adjust the pace to your "
                "available time."
            ),
            "",
        ]
        for index in range(weeks):
            lines.append(f"#### Week {index + 1}")
            assigned = chapter_weeks[index]
            if assigned:
                topics = "; ".join(
                    f"{chapter.subject.title()}: {chapter.name}" for chapter in assigned
                )
                lines.append(f"- **Syllabus:** {topics}")
            else:
                lines.append("- **Syllabus:** Consolidate previously covered chapters.")
            if weak_weeks[index]:
                areas = ", ".join(
                    f"{area.name} ({area.subject})" for area in weak_weeks[index]
                )
                lines.append(f"- **Weak-area focus:** {areas}")
            lines.append(
                "- **Practice:** Use active recall, then review every wrong or skipped answer."
            )
            lines.append("- **Check-in:** Finish with a timed mixed set and log remaining gaps.")
            lines.append("")

        return {
            "answer": "\n".join(lines).strip(),
            "mode": str(state.get("mode") or "detailed"),
            "citations": [],
            "notices": list(state.get("notices") or []),
        }
