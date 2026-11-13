"""Intent registry for active and future agent handlers.

The router classifies every message into an :class:`~utils.constants.Intent`.
Each intent maps to one agent pipeline.  Each active intent is wired to a specialist node in ``agents/graph.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from utils.constants import Intent

__all__ = ["AgentSpec", "AGENT_REGISTRY", "planned_agents", "spec_for", "describe_registry"]

AgentStatus = Literal["active", "planned"]


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """Metadata about one intent handler."""

    intent: Intent
    title: str
    description: str
    status: AgentStatus

    @property
    def available(self) -> bool:
        return self.status == "active"


AGENT_REGISTRY: dict[Intent, AgentSpec] = {
    Intent.DOUBT: AgentSpec(
        intent=Intent.DOUBT,
        title="Doubt Solver",
        description="Routes to RAG retrieval + grounded answer generation.",
        status="active",
    ),
    Intent.MCQ: AgentSpec(
        intent=Intent.MCQ,
        title="MCQ Generator Agent",
        description="Generates NEET-style MCQs with answer keys and explanations.",
        status="active",
    ),
    Intent.PYQ: AgentSpec(
        intent=Intent.PYQ,
        title="PYQ Retrieval Agent",
        description="Searches identified previous-year-question sources by chapter and year.",
        status="active",
    ),
    Intent.STUDY_PLAN: AgentSpec(
        intent=Intent.STUDY_PLAN,
        title="Study Planner Agent",
        description="Builds a week-by-week revision plan from the indexed syllabus and weak areas.",
        status="active",
    ),
    Intent.WEAKNESS: AgentSpec(
        intent=Intent.WEAKNESS,
        title="Weakness Analysis Agent",
        description="Uses quiz history to surface the weakest chapters per subject and topics.",
        status="active",
    ),
}


def spec_for(intent: Intent | str) -> AgentSpec:
    """Return the registry entry for ``intent`` (defaults to the doubt solver)."""
    resolved = Intent.coerce(intent)
    return AGENT_REGISTRY.get(resolved, AGENT_REGISTRY[Intent.DOUBT])


def planned_agents() -> list[AgentSpec]:
    """All intents that are announced but not implemented yet."""
    return [spec for spec in AGENT_REGISTRY.values() if not spec.available]


def describe_registry() -> list[dict[str, str]]:
    """Serialisable view for the UI / diagnostics."""
    return [
        {
            "intent": spec.intent.value,
            "title": spec.title,
            "description": spec.description,
            "status": spec.status,
        }
        for spec in AGENT_REGISTRY.values()
    ]
