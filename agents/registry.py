"""Intent registry - the extension point for future agents.

The router classifies every message into an :class:`~utils.constants.Intent`.
Each intent maps to one agent pipeline.  Adding the MCQ, PYQ, planner or
weakness agent later is a three-step change:

1. Write ``agents/mcq_agent.py`` with a class deriving from ``BaseAgent``.
2. Instantiate it in ``agents/graph.py`` and add a node + edge.
3. Flip its status in ``AGENT_REGISTRY`` below to ``active``.

Until then the graph routes those intents to a polite "not yet available" node
instead of failing, so the router can already be tuned for them.
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
        description="Will generate NEET-style MCQs with answer keys and explanations.",
        status="planned",
    ),
    Intent.PYQ: AgentSpec(
        intent=Intent.PYQ,
        title="PYQ Retrieval Agent",
        description="Will search a previous-year-question index filtered by chapter and year.",
        status="planned",
    ),
    Intent.STUDY_PLAN: AgentSpec(
        intent=Intent.STUDY_PLAN,
        title="Study Planner Agent",
        description="Will build a week-by-week NEET revision plan from the syllabus and weak areas.",
        status="planned",
    ),
    Intent.WEAKNESS: AgentSpec(
        intent=Intent.WEAKNESS,
        title="Weakness Analysis Agent",
        description="Will track quiz history and surface the weakest chapters per subject.",
        status="planned",
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
