"""High level facade used by the Gradio UI.

``NEETTutor`` owns the long-lived objects (prompt library, vector store, LLM
and compiled graph) and hides the lazy initialisation order:

    prompts + settings  ->  LLM (downloads weights on first use)
                        ->  FAISS (loads on first search)
                        ->  LangGraph

Startup validates one real completion and one textbook search before the UI is
built, so students never see a false ready state. Each turn also carries the
chosen answer mode, which is a presentation choice - it never changes the
retrieval pipeline.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable

from agents.base import emit_event
from agents.graph import TutorGraph
from agents.registry import describe_registry, planned_agents
from agents.state import Citation, TutorResult
from config import Settings
from llm.base import LLMClient, LLMError
from llm.factory import LLMFactory
from prompts.loader import PromptLibrary
from rag.vector_store import VectorStoreManager, read_index_metadata
from utils.constants import AnswerMode, RevisionDeliverable, TestScope
from utils.device import gpu_summary
from utils.logging_utils import get_logger
from utils.text_utils import format_seconds

__all__ = ["NEETTutor"]

logger = get_logger(__name__)

EventCallback = Callable[[dict[str, Any]], None]


class NEETTutor:
    """Stateful application service behind the chat interface."""

    def __init__(
        self,
        settings: Settings,
        *,
        llm: LLMClient | None = None,
        vector_store: VectorStoreManager | None = None,
        prompts: PromptLibrary | None = None,
    ) -> None:
        self.settings = settings
        self.prompts = prompts or PromptLibrary(settings.prompt_dir)
        self.vector_store = vector_store or VectorStoreManager(settings)
        self._llm = llm
        self._graph: TutorGraph | None = None
        self._platform: Any | None = None
        self._readiness: dict[str, Any] | None = None
        self._lock = threading.RLock()
        self._started_at = time.time()

    # ------------------------------------------------------------------- setup
    def ensure_ready(self, on_event: EventCallback | None = None) -> TutorGraph:
        """Load the LLM (once) and compile the graph (once)."""
        with self._lock:
            if self._graph is not None:
                return self._graph

            if self._llm is None:
                emit_event(
                    {"configurable": {"on_event": on_event}},
                    "llm",
                    f"Loading {self.settings.llm_model} (first run downloads the weights)",
                )
                logger.info("Creating LLM client for %s", self.settings.llm_model)
                self._llm = LLMFactory.create(self.settings)

            loader = getattr(self._llm, "load", None)
            if callable(loader) and not getattr(self._llm, "is_loaded", True):
                try:
                    loader()
                except Exception:
                    self._llm = None  # allow a clean retry on the next message
                    raise

            emit_event(
                {"configurable": {"on_event": on_event}},
                "graph",
                "Wiring the router, retrieval and answer agents",
            )
            self._graph = TutorGraph(
                self.settings,
                self._llm,
                self.vector_store,
                self.prompts,
                platform=self.platform(),
            )
            logger.info(
                "Tutor ready (llm=%s, index_ready=%s)",
                self._llm.name,
                self.vector_store.is_ready(),
            )
            return self._graph

    @property
    def is_ready(self) -> bool:
        """``True`` once the model has been loaded and the graph compiled."""
        return self._graph is not None

    @property
    def llm(self) -> LLMClient | None:
        return self._llm

    def startup_status(self) -> dict[str, Any]:
        """Return the last verified readiness snapshot without doing new I/O."""
        if self._readiness is not None:
            return self._readiness
        index_ready = bool(self.index_status()["ready"])
        checks = {
            "knowledge_base": {"ready": index_ready, "label": "Knowledge Base"},
            "llm": {"ready": False, "label": "LLM", "detail": "Not checked yet"},
            "retrieval": {"ready": False, "label": "Retrieval", "detail": "Not checked yet"},
        }
        return {
            "ready": False,
            "checks": checks,
            "student_message": "Genie is still getting ready. Please check again in a moment.",
            "errors": {},
            "checked_at": None,
        }

    def validate_startup(
        self,
        *,
        verify_generation: bool = True,
        verify_retrieval: bool = True,
    ) -> dict[str, Any]:
        """Prove that the study library, answer service and search actually work.

        This is deliberately called before the UI is built.  A configuration-only
        check is not enough: the Groq SDK can be absent, a key can be rejected, or
        an index can exist on disk but fail to load.  The returned snapshot powers
        the student gate; technical details remain available to Developer Mode.
        """
        with self._lock:
            checks: dict[str, dict[str, Any]] = {
                "knowledge_base": {"ready": False, "label": "Knowledge Base"},
                "llm": {"ready": False, "label": "LLM"},
                "retrieval": {"ready": False, "label": "Retrieval"},
            }
            errors: dict[str, str] = {}

            index = self.index_status()
            checks["knowledge_base"]["ready"] = bool(index["ready"])
            checks["knowledge_base"]["detail"] = (
                "Your NCERT study library is available."
                if index["ready"]
                else "Your NCERT study library is not available yet."
            )
            if not index["ready"]:
                errors["knowledge_base"] = f"No readable index at {self.settings.index_dir}"

            try:
                self.settings.validate_llm_provider()
                prompt_problems = self.prompts.validate()
                if prompt_problems:
                    raise LLMError("Prompt validation failed: " + "; ".join(prompt_problems))
                graph = self.ensure_ready()
                llm = graph.llm
                if verify_generation:
                    probe = llm.generate(
                        [{"role": "user", "content": "Reply with the single word READY."}],
                        max_new_tokens=32,
                        temperature=0.0,
                    )
                    if not str(probe or "").strip():
                        raise LLMError("The provider returned an empty readiness response.")
                checks["llm"].update(ready=True, detail="Genie can generate answers.")
            except Exception as exc:
                logger.exception("LLM startup validation failed")
                errors["llm"] = self._diagnostic_error(exc)
                checks["llm"]["detail"] = self._student_llm_error(exc)

            if not index["ready"]:
                checks["retrieval"]["detail"] = "Textbook search needs the study library."
                errors["retrieval"] = "Skipped because the knowledge base is unavailable"
            else:
                try:
                    if verify_retrieval:
                        self.vector_store.search("cell", k=1, score_threshold=0.0)
                    else:
                        self.vector_store.load()
                    checks["retrieval"].update(
                        ready=True, detail="Genie can find relevant textbook explanations."
                    )
                except Exception as exc:
                    logger.exception("Retrieval startup validation failed")
                    errors["retrieval"] = self._diagnostic_error(exc)
                    checks["retrieval"]["detail"] = (
                        "Textbook search is unavailable. Please try reconnecting."
                    )

            ready = all(bool(item["ready"]) for item in checks.values())
            if ready:
                message = "You're all set. Ask Genie your first NEET doubt."
            elif not checks["llm"]["ready"]:
                message = (
                    str(checks["llm"]["detail"])
                    + " You can still view Dashboard and Progress, or try reconnecting."
                )
            else:
                message = (
                    "Your textbook study tools are not ready yet. You can still view "
                    "Dashboard and Progress, or try reconnecting."
                )
            self._readiness = {
                "ready": ready,
                "checks": checks,
                "student_message": message,
                "errors": errors,
                "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            return self._readiness

    # --------------------------------------------------------------------- ask
    def ask(
        self,
        question: str,
        *,
        history: list[dict[str, Any]] | None = None,
        subject: str = "auto",
        top_k: int | None = None,
        mode: str | AnswerMode | None = None,
        on_event: EventCallback | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> TutorResult:
        """Answer one student message, never raising to the caller."""
        started = time.perf_counter()
        try:
            graph = self.ensure_ready(on_event)
        except Exception as exc:
            logger.exception("Tutor initialisation failed")
            return TutorResult.failure(self._startup_hint(exc), error=str(exc))

        try:
            return graph.ask(
                question,
                history=history,
                subject=subject,
                top_k=top_k,
                mode=mode,
                on_event=on_event,
                on_token=on_token,
            )
        except Exception as exc:  # defensive: graph.ask already handles its own errors
            logger.exception("Unexpected tutor failure after %.1fs", time.perf_counter() - started)
            return TutorResult.failure(
                "Genie could not finish that answer. Please try again or use Try reconnecting.",
                error=str(exc),
            )

    # ------------------------------------------------------------------- state
    def platform(self) -> Any:
        """The learning-platform facade behind the module tabs.

        Created once and reused.  The chat stays the owner of the LLM, the
        prompt library and the vector store: ``GeniePlatform`` wraps them with
        the Learn / Practice / Progress / Revision services and never re-does
        what the graph already does.  Importing :mod:`learning.service` lazily
        keeps app start-up as cheap as it was before the platform existed.
        """
        with self._lock:
            if self._platform is None:
                from learning.service import GeniePlatform

                self._platform = GeniePlatform(
                    self.settings,
                    tutor=self,
                    vector_store=self.vector_store,
                    prompts=self.prompts,
                )
                logger.info("Learning platform services wired")
            return self._platform

    def platform_status(self) -> dict[str, Any]:
        """Module map and capability hooks - cheap enough for ``--check``.

        Nothing here touches FAISS or the LLM: it reports the platform's shape,
        and adds the live catalog/progress numbers once the services exist.
        """
        from learning.capabilities import default_registry, describe_capabilities
        from learning.sections import CHAPTER_SECTIONS

        platform = self._platform
        registry = platform.capabilities if platform is not None else default_registry()
        modules = ["Ask Genie", "Practice", "Progress", "Revision Hub", "Dashboard"]
        if self.settings.learn_enabled:
            modules.append("Learn")
        return {
            "enabled": self.settings.platform_enabled,
            "built": platform is not None,
            "modules": modules,
            "learn_sections": list(CHAPTER_SECTIONS),
            "test_scopes": [scope.value for scope in TestScope],
            "revision_deliverables": [item.value for item in RevisionDeliverable],
            "questions_per_test": self.settings.test_question_count,
            "marking": {
                "correct": self.settings.marks_correct,
                "incorrect": self.settings.marks_incorrect,
            },
            "progress_store": str(platform.progress.file) if platform is not None else str(self.settings.progress_dir),
            "capabilities": describe_capabilities(registry),
        }

    def index_status(self) -> dict[str, Any]:
        """Index information read from the JSON sidecar (no model loading)."""
        metadata = read_index_metadata(self.settings) or {}
        return {
            "ready": self.vector_store.is_ready(),
            "path": str(self.settings.index_dir),
            "chunks": int(metadata.get("chunks") or 0),
            "files": int(metadata.get("files") or 0),
            "subjects": metadata.get("subjects") or {},
            "updated_at": metadata.get("updated_at"),
            "embedding_model": metadata.get("embedding_model", self.settings.embedding_model),
        }

    def reload_index(self) -> None:
        """Drop the cached FAISS index so the next question re-reads disk."""
        self.vector_store.reload()
        logger.info("Index cache cleared; next request will reload from disk")

    def health(self) -> dict[str, Any]:
        """Non-invasive diagnostics for the UI and ``--check`` CLI flag."""
        prompt_problems = self.prompts.validate()
        llm_info: dict[str, Any]
        if self._llm is None:
            # Nothing loaded yet (e.g. ``--check``): report the *resolved*
            # provider + model from config, with the API key masked.
            provider = self.settings.llm_provider_diagnostic()
            llm_info = {
                "backend": provider["provider"],
                "provider": provider["provider"],
                "model": provider["model"],
                "loaded": False,
            }
            if provider["provider"] == "groq":
                llm_info["api_key"] = provider["api_key"]
                llm_info["configured"] = provider.get("configured", False)
        else:
            try:
                llm_info = dict(self._llm.describe())
            except Exception:  # pragma: no cover - defensive
                llm_info = {"backend": self._llm.name, "loaded": True}
        return {
            "app": {"name": self.settings.app_name, "version": self.settings.app_version},
            "settings": self.settings.summary(),
            "llm": llm_info,
            "index": self.index_status(),
            "prompts": {"ok": not prompt_problems, "problems": prompt_problems},
            "platform": self.platform_status(),
            "hardware": gpu_summary(),
            "agents": describe_registry(),
            "coach": self.coach_status(),
            "readiness": self.startup_status(),
            "uptime": format_seconds(time.time() - self._started_at),
        }

    def coach_status(self) -> dict[str, Any]:
        """Persona + answer-mode state, used by ``health()`` and the UI panels."""
        return {
            "persona_enabled": self.settings.persona_enabled,
            "enforce_answer_format": self.settings.enforce_answer_format,
            "default_mode": self.settings.answer_mode.value,
            "mcq_count": self.settings.mcq_count,
            "modes": [
                {
                    "value": mode.value,
                    "label": mode.label,
                    "emoji": mode.emoji,
                    "description": mode.description,
                }
                for mode in AnswerMode
            ],
        }

    def architecture(self) -> str:
        """Graph description used by the "Architecture" tab."""
        if self._graph is not None:
            return self._graph.architecture()
        modes = "\n".join(
            f"  - {mode.emoji} {mode.label}: {mode.description}" for mode in AnswerMode
        )
        return (
            "User Question\n"
            "   -> Subject Router Agent        (subject + intent, JSON)\n"
            "   -> Intent specialist           (MCQ / PYQ / study plan / weakness)\n"
            "   -> RAG Retrieval Agent         (FAISS top-k, subject filter, mode-scaled)\n"
            "   -> Answer Generation Agent     (Neet Genie persona + mode contract)\n"
            "   -> Coach format repair         (section headings for the mode)\n"
            "   -> Response\n\n"
            "Answer modes (chosen by the student, presentation only):\n"
            f"{modes}"
        )

    def planned_agents(self) -> list[dict[str, str]]:
        return [
            {"title": spec.title, "description": spec.description}
            for spec in planned_agents()
        ]

    # --------------------------------------------------------------- internals
    def _startup_hint(self, exc: Exception) -> str:
        """Keep provider failures friendly; logs retain the technical detail."""
        return self._student_llm_error(exc) + " Please try again, or view Dashboard and Progress meanwhile."

    @staticmethod
    def _student_llm_error(exc: Exception) -> str:
        detail = str(exc).lower()
        if "api key" in detail or "authentication" in detail or "401" in detail:
            return "Genie could not sign in to the answer service. The setup needs attention."
        if "rate" in detail or "429" in detail:
            return "Genie is busy right now. Please wait a moment before trying again."
        if "connection" in detail or "network" in detail or "timeout" in detail:
            return "Genie cannot reach the answer service. Check your internet connection and retry."
        return "Genie is not connected yet, so answers are paused rather than showing you an unreliable reply."

    def _diagnostic_error(self, exc: Exception) -> str:
        """Secret-free exception text for Developer Mode."""
        detail = str(exc)
        for secret in (self.settings.groq_api_key, self.settings.hf_token):
            if secret:
                detail = detail.replace(secret, "***")
        return f"{type(exc).__name__}: {detail}"


def citations_to_markdown(citations: list[Citation]) -> str:
    """Render retrieved sources as a compact markdown list for the UI."""
    if not citations:
        return "_No textbook passages were used for this answer._"
    lines: list[str] = []
    for citation in citations:
        page = f" · p.{citation['page']}" if citation.get("page") else ""
        preview = str(citation.get("preview", "")).replace("\n", " ").strip()
        lines.append(
            f"**{citation['index']}.** {citation['file_name']}{page} · "
            f"{str(citation['subject']).title()}\n\n"
            f"> {preview}"
        )
    return "\n\n".join(lines)
