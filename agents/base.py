"""Common plumbing for every agent: timing, logging and progress events.

An agent is a callable that receives the graph state and returns a *partial*
state update.  ``BaseAgent.__call__`` adds per-agent timing so the UI (and the
logs) can show where each turn spent its time.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Mapping

from config import Settings
from llm.base import LLMClient, TokenCallback
from prompts.loader import PromptLibrary
from utils.logging_utils import get_logger

__all__ = ["BaseAgent", "AgentError", "emit_event", "token_callback", "configurable", "add_notice"]

EventCallback = Callable[[dict[str, Any]], None]


class AgentError(RuntimeError):
    """Raised when an agent cannot complete its step (safe to show to users)."""


def configurable(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return the ``configurable`` section of a LangGraph ``RunnableConfig``."""
    if not config:
        return {}
    section = config.get("configurable")
    return section if isinstance(section, Mapping) else {}


def emit_event(config: Mapping[str, Any] | None, node: str, message: str) -> None:
    """Push a human-readable progress event to the UI (if a callback exists)."""
    callback = configurable(config).get("on_event")
    if callable(callback):
        try:
            callback({"node": node, "message": message})
        except Exception:  # a broken UI must never break the graph
            get_logger(__name__).debug("on_event callback failed", exc_info=True)


def add_notice(notices: list[str], message: str) -> list[str]:
    """Append ``message`` to ``notices`` without creating duplicates."""
    if message and message not in notices:
        notices.append(message)
    return notices


def token_callback(config: Mapping[str, Any] | None) -> TokenCallback | None:
    """Extract the streaming token callback from the graph config."""
    callback = configurable(config).get("on_token")
    return callback if callable(callback) else None


class BaseAgent(ABC):
    """Template-method base class for all tutor agents."""

    #: Unique node name, also used as the key in ``state["timings"]``.
    name: ClassVar[str] = "agent"

    def __init__(
        self,
        settings: Settings,
        *,
        llm: LLMClient | None = None,
        prompts: PromptLibrary | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.prompts = prompts or PromptLibrary(settings.prompt_dir)
        self.logger = get_logger(f"agents.{self.name}")

    # ------------------------------------------------------------------ public
    def __call__(
        self,
        state: Mapping[str, Any],
        config: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the agent and record its duration in the state timings."""
        started = time.perf_counter()
        self.logger.debug("%s started", self.name)
        update = self.run(dict(state), config or {})
        elapsed = time.perf_counter() - started

        timings = dict(state.get("timings") or {})
        timings.update(update.get("timings") or {})
        timings[self.name] = round(elapsed, 3)
        update["timings"] = timings

        self.logger.debug("%s finished in %.3fs", self.name, elapsed)
        return update

    @abstractmethod
    def run(self, state: dict[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
        """Perform the agent's work and return a partial state update."""

    # ----------------------------------------------------------------- helpers
    def require_llm(self) -> LLMClient:
        """Return the LLM client or fail with an actionable error."""
        if self.llm is None:
            raise AgentError("No language model is configured for this agent.")
        return self.llm

    def append_notice(self, state: Mapping[str, Any], notice: str) -> list[str]:
        """Accumulate notices without losing messages from earlier agents."""
        return add_notice(list(state.get("notices") or []), notice)
