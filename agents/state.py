"""Shared graph state and the value object returned to the UI.

The state is a plain ``TypedDict`` of JSON-friendly values so it can later be
attached to a LangGraph checkpointer (conversation memory, replay, analytics)
without any serialisation changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from utils.constants import AnswerMode

__all__ = ["TutorState", "TutorResult", "Citation", "ChatTurn", "RouteDecision"]


class ChatTurn(TypedDict):
    """One chat message as used by Gradio (``type="messages"``)."""

    role: Literal["user", "assistant"]
    content: str


class Citation(TypedDict):
    """A retrieved passage as presented in the UI."""

    index: int
    source: str
    file_name: str
    page: int | None
    subject: str
    score: float
    preview: str


class RouteDecision(TypedDict):
    """Output of the subject router."""

    subject: str
    intent: str
    confidence: float
    reason: str


class TutorState(TypedDict, total=False):
    """LangGraph state passed between agents.

    Every agent returns a *partial* update which LangGraph merges into the
    running state, so keys are optional.
    """

    # --- input
    question: str
    history: list[ChatTurn]
    subject_hint: str
    top_k: int
    #: Selected :class:`~utils.constants.AnswerMode` value.
    mode: str

    # --- router
    subject: str
    intent: str
    subject_confidence: float
    router_reason: str

    # --- retrieval
    query: str
    retrieved: list[dict[str, Any]]
    context: str
    citations: list[Citation]

    # --- answer
    answer: str
    degraded: bool

    # --- cross-cutting
    notices: list[str]
    timings: dict[str, float]
    error: str | None


@dataclass(slots=True)
class TutorResult:
    """Final, UI-ready result of one tutor turn."""

    answer: str
    subject: str = "general"
    intent: str = "doubt"
    mode: str = "detailed"
    citations: list[Citation] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    router_reason: str = ""
    error: str | None = None
    total_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        """``True`` when the turn produced an answer without a fatal error."""
        return self.error is None and bool(self.answer.strip())

    @property
    def used_context(self) -> bool:
        """``True`` when the answer is grounded in retrieved textbook passages."""
        return bool(self.citations)

    @property
    def answer_mode(self) -> "AnswerMode":
        """The selected answer mode as an enum."""
        return AnswerMode.coerce(self.mode)

    def as_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "subject": self.subject,
            "intent": self.intent,
            "mode": self.mode,
            "citations": self.citations,
            "notices": self.notices,
            "timings": self.timings,
            "router_reason": self.router_reason,
            "error": self.error,
            "total_seconds": round(self.total_seconds, 3),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any], *, total_seconds: float = 0.0) -> "TutorResult":
        """Build a result from the final graph state."""
        return cls(
            answer=str(state.get("answer") or ""),
            subject=str(state.get("subject") or "general"),
            intent=str(state.get("intent") or "doubt"),
            mode=AnswerMode.coerce(state.get("mode")).value,
            citations=list(state.get("citations") or []),
            notices=list(state.get("notices") or []),
            timings=dict(state.get("timings") or {}),
            router_reason=str(state.get("router_reason") or ""),
            error=state.get("error"),
            total_seconds=total_seconds,
        )

    @classmethod
    def failure(cls, message: str, *, error: str | None = None) -> "TutorResult":
        """Convenience constructor for fatal errors shown in the chat."""
        return cls(answer=message, error=error or message)
