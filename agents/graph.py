"""LangGraph orchestration of the Neet Genie coach.

Flow::

    START -> router -> (doubt)      -> retrieval -> answer -> END
                    -> (served by the selected answer mode)
                    -> (planned)    -> deferred  -> END
                    -> (on error)   -> failed    -> END

The router decides the subject *and* the intent; the student separately picks
an :class:`~utils.constants.AnswerMode` that controls how the answer is
presented.  An explicit mode that can serve the detected intent (e.g. MCQ
Practice for an ``mcq`` intent) keeps the turn inside the implemented
pipeline instead of deferring it.  Only intents that no mode can serve yet
(``pyq`` outside PYQ Focus, ``study_plan``, ``weakness``) reach ``deferred``,
so the graph stays valid while those agents are built.
Adding an agent later means registering a node and one conditional edge - see
``agents/registry.py`` for the contract.
"""

from __future__ import annotations

import time
from typing import Any, Mapping

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from agents.answer_agent import AnswerAgent
from agents.base import AgentError, add_notice, emit_event
from agents.registry import describe_registry, planned_agents, spec_for
from agents.retrieval_agent import RetrievalAgent
from agents.router_agent import RouterAgent
from agents.state import TutorResult, TutorState
from config import Settings
from llm.base import LLMClient
from prompts.loader import PromptLibrary
from rag.vector_store import VectorStoreManager
from utils.constants import AnswerMode, Intent
from utils.logging_utils import get_logger
from utils.text_utils import clean_text

__all__ = ["TutorGraph"]

logger = get_logger(__name__)

# LangGraph node names (kept as constants so the UI can render them).
NODE_ROUTER = "router"
NODE_RETRIEVAL = "retrieval"
NODE_ANSWER = "answer"
NODE_DEFERRED = "deferred"
NODE_FAILED = "failed"

#: ``deferred`` suggestions keyed by the intent the router detected.
_MODE_HINTS: dict[Intent, str] = {
    Intent.MCQ: "**MCQ Practice**",
    Intent.PYQ: "**PYQ Focus**",
}


class TutorGraph:
    """Builds and runs the agent graph."""

    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
        vector_store: VectorStoreManager,
        prompts: PromptLibrary,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.vector_store = vector_store
        self.prompts = prompts

        self.router = RouterAgent(settings, llm=llm, prompts=prompts)
        self.retrieval = RetrievalAgent(settings, vector_store, prompts=prompts)
        self.answer = AnswerAgent(settings, llm=llm, prompts=prompts)

        self._graph = self._build()

    # ------------------------------------------------------------------ build
    def _build(self) -> Any:
        """Wire the nodes and conditional edges, then compile."""
        graph = StateGraph(TutorState)
        graph.add_node(NODE_ROUTER, self._router_node)
        graph.add_node(NODE_RETRIEVAL, self._retrieval_node)
        graph.add_node(NODE_ANSWER, self._answer_node)
        graph.add_node(NODE_DEFERRED, self._deferred_node)
        graph.add_node(NODE_FAILED, self._failed_node)

        graph.set_entry_point(NODE_ROUTER)
        graph.add_conditional_edges(
            NODE_ROUTER,
            self._after_router,
            {
                NODE_RETRIEVAL: NODE_RETRIEVAL,
                NODE_DEFERRED: NODE_DEFERRED,
                NODE_FAILED: NODE_FAILED,
            },
        )
        graph.add_conditional_edges(
            NODE_RETRIEVAL,
            self._after_retrieval,
            {NODE_ANSWER: NODE_ANSWER, NODE_FAILED: NODE_FAILED},
        )
        graph.add_edge(NODE_ANSWER, END)
        graph.add_edge(NODE_DEFERRED, END)
        graph.add_edge(NODE_FAILED, END)

        compiled = graph.compile()
        logger.info("LangGraph compiled with nodes: %s", ", ".join(self.node_names))
        return compiled

    # ------------------------------------------------------------------- nodes
    def _router_node(self, state: TutorState, config: RunnableConfig) -> dict[str, Any]:
        return self._guard(NODE_ROUTER, self.router, state, config)

    def _retrieval_node(self, state: TutorState, config: RunnableConfig) -> dict[str, Any]:
        return self._guard(NODE_RETRIEVAL, self.retrieval, state, config)

    def _answer_node(self, state: TutorState, config: RunnableConfig) -> dict[str, Any]:
        return self._guard(NODE_ANSWER, self.answer, state, config)

    def _deferred_node(self, state: TutorState, config: RunnableConfig) -> dict[str, Any]:
        """Polite response for intents whose agent is not implemented yet."""
        spec = spec_for(state.get("intent"))
        mode = AnswerMode.coerce(state.get("mode"), default=self.settings.answer_mode)
        emit_event(config, NODE_DEFERRED, f"{spec.title} is not available yet")
        notices = list(state.get("notices") or [])
        add_notice(notices, f"{spec.title} is on the roadmap; answering as a doubt instead is available now.")

        suggested = _MODE_HINTS.get(spec.intent)
        already_selected = suggested == f"**{mode.label}**"
        mode_tip = (
            f"Pick {suggested} in the **Answer mode** box and ask again - I will build the "
            "practice set straight from your indexed chapters."
            if suggested and not already_selected
            else "Right now I can solve concept and numerical doubts using your indexed NCERT material."
        )
        roadmap = "\n".join(f"- **{agent.title}** - {agent.description}" for agent in planned_agents())
        answer = (
            f"### {spec.title} - coming soon\n\n"
            f"{spec.description}\n\n"
            f"{mode_tip} Ask me the underlying concept instead, for example: _\"Explain the photoelectric effect "
            "with the stopping potential formula\"_.\n\n"
            f"**Planned agents**\n{roadmap}"
        )
        return {"answer": answer, "mode": mode.value, "citations": [], "context": "", "notices": notices}

    def _failed_node(self, state: TutorState, config: RunnableConfig) -> dict[str, Any]:
        """Terminal node - the error message was already written by ``_guard``."""
        emit_event(config, NODE_FAILED, "The request could not be completed")
        return {}

    # -------------------------------------------------------------- routing
    def _after_router(self, state: TutorState) -> str:
        if state.get("error"):
            return NODE_FAILED
        spec = spec_for(state.get("intent"))
        if spec.available:
            return NODE_RETRIEVAL
        # An explicit answer mode can serve the intent today (MCQ Practice /
        # PYQ Focus), so the turn keeps flowing through retrieval + answer.
        mode = AnswerMode.coerce(state.get("mode"), default=self.settings.answer_mode)
        return NODE_RETRIEVAL if mode.satisfies(spec.intent) else NODE_DEFERRED

    def _after_retrieval(self, state: TutorState) -> str:
        return NODE_FAILED if state.get("error") else NODE_ANSWER

    # -------------------------------------------------------------- execution
    def ask(
        self,
        question: str,
        *,
        history: list[dict[str, Any]] | None = None,
        subject: str = "auto",
        top_k: int | None = None,
        mode: str | AnswerMode | None = None,
        on_event: Any | None = None,
        on_token: Any | None = None,
    ) -> TutorResult:
        """Run one conversation turn through the graph."""
        cleaned = clean_text(question)[: self.settings.max_question_chars]
        if not cleaned:
            return TutorResult.failure("Please type a question first.", error="empty question")

        answer_mode = AnswerMode.coerce(mode, default=self.settings.answer_mode)
        initial: TutorState = {
            "question": cleaned,
            "history": list(history or []),
            "subject_hint": (subject or "auto").lower(),
            "top_k": int(top_k or self.settings.top_k),
            "mode": answer_mode.value,
            "notices": [],
            "timings": {},
        }
        config: dict[str, Any] = {
            "configurable": {"on_event": on_event, "on_token": on_token},
            "recursion_limit": 12,
        }

        started = time.perf_counter()
        logger.info(
            "Turn start: subject_hint=%s top_k=%s mode=%s question=%r",
            initial["subject_hint"],
            initial["top_k"],
            answer_mode.value,
            cleaned[:160],
        )
        try:
            final_state = self._graph.invoke(initial, config=config)
        except Exception as exc:  # graph-level failure (should be rare)
            logger.exception("Graph execution failed")
            return TutorResult.failure(
                "Genie could not finish that answer. Please try again, or ask the doubt in a "
                "shorter way. Your study progress is still safe.",
                error=str(exc),
            )

        elapsed = time.perf_counter() - started
        result = TutorResult.from_state(final_state, total_seconds=elapsed)
        logger.info(
            "Turn done in %.2fs: subject=%s intent=%s mode=%s citations=%d error=%s",
            elapsed,
            result.subject,
            result.intent,
            result.mode,
            len(result.citations),
            result.error,
        )
        return result

    # --------------------------------------------------------------- metadata
    @property
    def node_names(self) -> list[str]:
        return [NODE_ROUTER, NODE_RETRIEVAL, NODE_ANSWER, NODE_DEFERRED, NODE_FAILED]

    def architecture(self) -> str:
        """Human readable description of the graph for the UI."""
        active = "\n".join(
            f"  {index}. {entry['title']} ({entry['status']})"
            for index, entry in enumerate(describe_registry(), start=1)
        )
        modes = "\n".join(
            f"  - {mode.emoji} {mode.label}: {mode.description}"
            for mode in AnswerMode
        )
        return (
            "User Question\n"
            "   -> Subject Router Agent        (subject + intent, JSON)\n"
            "   -> RAG Retrieval Agent         (FAISS top-k, subject filter, mode-scaled)\n"
            "   -> Answer Generation Agent     (Neet Genie persona + mode contract)\n"
            "   -> Coach format repair         (section headings for the mode)\n"
            "   -> Response\n\n"
            "Answer modes (chosen by the student, presentation only):\n"
            f"{modes}\n\n"
            "Intent registry:\n"
            f"{active}"
        )

    # ------------------------------------------------------------- internals
    def _guard(
        self,
        node: str,
        agent: Any,
        state: TutorState,
        config: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Run an agent, converting any exception into a terminal error state."""
        try:
            return agent(state, config)
        except AgentError as exc:
            logger.error("%s failed: %s", node, exc)
            return self._error_update(state, node, str(exc))
        except Exception as exc:  # unexpected - log with traceback
            logger.exception("%s crashed", node)
            return self._error_update(
                state,
                node,
                f"Unexpected error in the {node} step: {exc}",
            )

    def _error_update(self, state: TutorState, node: str, detail: str) -> dict[str, Any]:
        notices = list(state.get("notices") or [])
        add_notice(notices, f"The {node} step failed.")
        return {
            "error": f"{node}: {detail}",
            "mode": AnswerMode.coerce(state.get("mode"), default=self.settings.answer_mode).value,
            "answer": (
                "**Genie could not finish that answer.** Please try again in a moment. "
                "If the problem continues, use **Try reconnecting** on the Ask Genie page. "
                "You can still review Dashboard and Progress meanwhile."
            ),
            "citations": [],
            "notices": notices,
        }
