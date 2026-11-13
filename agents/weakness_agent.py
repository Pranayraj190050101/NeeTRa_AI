"""Report weak areas from the student's persisted practice history."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agents.base import BaseAgent, emit_event

__all__ = ["WeaknessAnalysisAgent"]


class WeaknessAnalysisAgent(BaseAgent):
    """Surface evidence-backed weak topics without inventing performance data."""

    name = "weakness_analysis"

    def __init__(self, settings, *, platform: Any, llm=None, prompts=None) -> None:
        super().__init__(settings, llm=llm, prompts=prompts)
        self.platform = platform

    def run(self, state: dict[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
        progress = self.platform.progress
        stats = progress.stats()
        areas = progress.weak_areas(threshold=0.8, limit=12)
        weak_chapters = [
            chapter
            for chapter in progress.chapter_performance().values()
            if chapter["accuracy"] < 0.8
        ][:12]
        emit_event(config, self.name, "Reviewing completed practice tests and topic accuracy")

        if not stats["tests"]:
            answer = (
                "### Weakness Analysis\n\n"
                "There isn't enough quiz history to identify weak chapters yet. "
                "Complete and submit a test in Practice; I'll use its topic-level results here."
            )
        elif not areas and not weak_chapters:
            answer = (
                "### Weakness Analysis\n\n"
                f"No topic or chapter is below 80% accuracy across your {stats['tests']} "
                "recorded test(s). "
                "Keep taking mixed tests so new weak spots can be identified."
            )
        else:
            lines = [
                "### Weakness Analysis",
                "",
                f"Based on {stats['tests']} recorded test(s), prioritise these areas:",
                "",
            ]
            for chapter in weak_chapters:
                lines.append(
                    f"- **{chapter['chapter']}** ({chapter['subject'].title()}): "
                    f"{chapter['accuracy']:.0%} accuracy across {chapter['tests']} test(s)."
                )
            if weak_chapters and areas:
                lines.extend(["", "**Topic-level misses**"])
            for area in areas:
                answered = max(0, area.questions)
                lines.append(
                    f"- **{area.name}** ({area.subject.title()} · {area.where}): "
                    f"{area.accuracy_percent}% accuracy; {area.wrong} incorrect "
                    f"across {answered} recorded question(s)."
                )
            lines.extend(
                [
                    "",
                    "Revise the first item, then take a short topic test and compare the next result.",
                ]
            )
            answer = "\n".join(lines)

        return {
            "answer": answer,
            "mode": str(state.get("mode") or "detailed"),
            "citations": [],
            "notices": list(state.get("notices") or []),
        }
