"""Shared generation plumbing for the platform services.

Learn, Practice and Revision Hub all need the same three steps: retrieve
grounding passages from the *existing* FAISS index, render a prompt from the
*existing* prompt library, and ask the *existing* LLM client.  This module owns
that loop so the services stay small and behave consistently.

Deliberate constraints:

* the RAG pipeline is **reused, not modified** - retrieval goes through
  ``VectorStoreManager.search`` and the context block is formatted by the very
  same ``RetrievalAgent.format_context`` the chat uses, so citations look
  identical everywhere;
* the ``TASK_ID`` / ``SECTIONS`` / ``COUNT`` marker header written in front of
  every system prompt lets the offline ``MockLLM`` pick the right canned reply
  without parsing prose (mirrors the ``MODE_ID`` line used by the chat);
* a missing index or a missing model degrades into a clear notice instead of
  an exception.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from agents.base import add_notice
from agents.retrieval_agent import RetrievalAgent
from agents.state import Citation
from config import Settings
from llm.base import ChatMessage, LLMClient
from prompts.loader import PERSONA_PROMPT, PromptLibrary
from rag.vector_store import RetrievedChunk, VectorStoreManager, VectorStoreError
from utils.logging_utils import get_logger
from utils.text_utils import clean_text, truncate

__all__ = ["GenieGenerator", "GeneratedDoc", "GenerationTask", "Grounding"]

logger = get_logger(__name__)

LLMProvider = Callable[[], LLMClient]


@dataclass(frozen=True, slots=True)
class Grounding:
    """Retrieved material for one platform request."""

    context: str = ""
    citations: list[Citation] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)

    @property
    def passages(self) -> int:
        return len(self.citations)

    @property
    def grounded(self) -> bool:
        return bool(self.context.strip())


@dataclass(frozen=True, slots=True)
class GenerationTask:
    """One fully rendered request plus the machine-readable markers."""

    task_id: str
    system: str
    user: str
    sections: tuple[str, ...] = ()
    count: int | None = None
    max_new_tokens: int | None = None

    def system_block(self, *, persona: str | None = None) -> str:
        """Marker header + optional persona + the task instructions."""
        headers = [f"TASK_ID: {self.task_id}"]
        if self.sections:
            headers.append("SECTIONS: " + " | ".join(self.sections))
        if self.count:
            headers.append(f"COUNT: {self.count}")
        blocks = ["\n".join(headers)]
        if persona:
            blocks.append(persona.strip())
        blocks.append(self.system.strip())
        return "\n\n---\n\n".join(block for block in blocks if block)

    def messages(self, *, persona: str | None = None) -> list[ChatMessage]:
        return [
            {"role": "system", "content": self.system_block(persona=persona)},
            {"role": "user", "content": self.user},
        ]


@dataclass(slots=True)
class GeneratedDoc:
    """A finished study document with its provenance."""

    kind: str
    title: str
    markdown: str = ""
    citations: list[Citation] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)
    seconds: float = 0.0
    ok: bool = True
    error: str | None = None

    @property
    def grounded(self) -> bool:
        return bool(self.citations)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "title": self.title,
            "markdown": self.markdown,
            "citations": self.citations,
            "notices": self.notices,
            "seconds": round(self.seconds, 2),
            "ok": self.ok,
            "error": self.error,
        }

    @classmethod
    def failure(cls, kind: str, title: str, message: str, *, error: str | None = None) -> "GeneratedDoc":
        return cls(kind=kind, title=title, markdown=message, ok=False, error=error, notices=[message])


class GenieGenerator:
    """Retrieval + prompt rendering + LLM call, shared by every service."""

    def __init__(
        self,
        settings: Settings,
        *,
        prompts: PromptLibrary | None = None,
        vector_store: VectorStoreManager | None = None,
        llm: LLMClient | None = None,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        self.settings = settings
        self.prompts = prompts or PromptLibrary(settings.prompt_dir)
        self.vector_store = vector_store or VectorStoreManager(settings)
        self._llm = llm
        self._llm_provider = llm_provider
        #: Reuse the chat agent's context/citation formatter verbatim.
        self._formatter = RetrievalAgent(settings, self.vector_store, prompts=self.prompts)

    # ------------------------------------------------------------------ access
    @property
    def llm(self) -> LLMClient:
        """Resolve the model lazily (the provider may load a 7B checkpoint)."""
        if self._llm is None:
            if self._llm_provider is None:
                raise RuntimeError("No language model configured for this generator")
            self._llm = self._llm_provider()
        return self._llm

    def index_ready(self) -> bool:
        try:
            return self.vector_store.is_ready()
        except Exception:  # pragma: no cover - defensive
            return False

    # --------------------------------------------------------------- retrieval
    def ground(
        self,
        query: str,
        *,
        subject: str | None = None,
        passages: int | None = None,
        prefer_file: str | None = None,
    ) -> Grounding:
        """Search the index and format the context block.

        ``prefer_file`` implements chapter scoping without touching the RAG
        layer: the FAISS search still asks for ``k`` passages (optionally
        filtered by subject) and the results are then *ordered* so that the
        requested chapter comes first.  Passages from the rest of the library
        stay in the block as supporting material.
        """
        query = clean_text(query)
        notices: list[str] = []
        if not query:
            return Grounding(notices=["No search term was available for this request."])
        if not self.index_ready():
            add_notice(
                notices,
                "No index found - this document is written from the coach's general NEET knowledge.",
            )
            return Grounding(notices=notices)

        top_k = max(1, int(passages or self.settings.top_k))
        wanted = top_k * (3 if prefer_file else 1)
        try:
            chunks: list[RetrievedChunk] = self.vector_store.search(
                truncate(query, 400, suffix=""),
                k=wanted,
                subject=None if not subject or subject == "general" else subject,
            )
        except VectorStoreError as exc:
            logger.error("Platform retrieval failed: %s", exc)
            add_notice(notices, f"Retrieval failed: {exc}")
            return Grounding(notices=notices)
        except Exception as exc:  # pragma: no cover - missing optional backend
            # A sidecar index can exist without its embeddings package being
            # importable; the platform still answers, just without grounding.
            logger.error("Platform retrieval crashed: %s", exc)
            add_notice(
                notices,
                f"Retrieval could not run ({type(exc).__name__}: {exc}); this document "
                "is written from the coach's general NEET knowledge.",
            )
            return Grounding(notices=notices)

        if prefer_file:
            chunks = _prefer_file(chunks, prefer_file)[:top_k]
        else:
            chunks = chunks[:top_k]

        if not chunks:
            add_notice(
                notices,
                f"No indexed passage matched '{truncate(query, 60, suffix='...')}'. "
                "The document is written from general NEET knowledge.",
            )
            return Grounding(notices=notices)

        context, citations = self._formatter.format_context(chunks)
        return Grounding(context=context, citations=citations, notices=notices)

    # -------------------------------------------------------------- generation
    def build_task(
        self,
        *,
        task_id: str,
        system_prompt: str,
        user_prompt: str,
        sections: Sequence[str] = (),
        count: int | None = None,
        **values: object,
    ) -> GenerationTask:
        """Render two prompt files into a :class:`GenerationTask`."""
        system = self.prompts.render(system_prompt, **values) if values else self.prompts.raw(system_prompt)
        user = self.prompts.render(user_prompt, **values)
        return GenerationTask(
            task_id=task_id,
            system=system,
            user=user,
            sections=tuple(str(section) for section in sections),
            count=count,
        )

    def execute(self, task: GenerationTask, *, temperature: float | None = None) -> str:
        """Run one task against the model and return the raw markdown."""
        persona = None
        if self.settings.persona_enabled:
            try:
                persona = self.prompts.raw(PERSONA_PROMPT)
            except (FileNotFoundError, ValueError) as exc:  # pragma: no cover - startup guard
                logger.warning("Persona unavailable for platform task: %s", exc)
        messages = task.messages(persona=persona)
        budget = task.max_new_tokens or int(self.settings.llm_max_new_tokens * _TASK_BUDGET.get(task.task_id, 1.0))
        return self.llm.generate(
            messages,
            max_new_tokens=max(128, budget),
            temperature=(
                self.settings.llm_temperature if temperature is None else float(temperature)
            ),
        )

    def generate(
        self,
        task: GenerationTask,
        *,
        kind: str,
        title: str,
        grounding: Grounding | None = None,
        temperature: float | None = None,
    ) -> GeneratedDoc:
        """Execute a task and wrap the outcome (never raises to the UI)."""
        started = time.perf_counter()
        notices = list(grounding.notices) if grounding else []
        try:
            raw = self.execute(task, temperature=temperature)
        except Exception as exc:  # model crash must stay a chat-friendly message
            logger.exception("Platform generation failed (task=%s)", task.task_id)
            return GeneratedDoc(
                kind=kind,
                title=title,
                markdown=(
                    "**This could not be generated right now.**\n\n"
                    f"`{type(exc).__name__}: {clean_text(str(exc)) or 'no detail'}`\n\n"
                    "Check `logs/neet_tutor.log`, or run the app with "
                    "`NEET_LLM_BACKEND=mock` to try the offline path."
                ),
                citations=list(grounding.citations) if grounding else [],
                notices=notices,
                seconds=time.perf_counter() - started,
                ok=False,
                error=str(exc),
            )
        return GeneratedDoc(
            kind=kind,
            title=title,
            markdown=clean_text(raw),
            citations=list(grounding.citations) if grounding else [],
            notices=notices,
            seconds=time.perf_counter() - started,
        )

    # ----------------------------------------------------------------- helpers
    def fallback_block(self, reason: str) -> str:
        """A short, honest note shown when a document cannot be grounded."""
        return f"_Ungrounded: {reason}_"


#: Relative token budget per platform task (chapter pages are longer than a
#: revision card deck, MCQ sets need the most room).
_TASK_BUDGET: dict[str, float] = {
    "learn_chapter": 1.3,
    "mcq_generation": 1.6,
    "revision_notes": 1.0,
}


def _prefer_file(chunks: list[RetrievedChunk], file_name: str) -> list[RetrievedChunk]:
    """Stable sort putting the requested chapter's passages first."""
    needle = clean_text(file_name).lower()
    ranked = sorted(
        enumerate(chunks),
        key=lambda pair: (
            0 if needle and needle in str(pair[1].file_name).lower() else 1,
            pair[0],
        ),
    )
    return [chunk for _, chunk in ranked]
