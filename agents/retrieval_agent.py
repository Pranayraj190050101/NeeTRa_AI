"""RAG Retrieval Agent.

Responsibilities:

1. Turn the student's message into a search query (short follow-ups like
   "and its unit?" are merged with the previous question).
2. Query FAISS with a subject filter, falling back to the whole index when the
   filtered search finds nothing.
3. Format the top passages into a numbered, character-budgeted context block
   and produce citation records for the UI.

The answer mode only scales *how many* passages are asked for (through the
existing ``search(k=...)`` parameter) - the RAG layer itself is mode agnostic.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from agents.base import BaseAgent, add_notice, emit_event
from agents.state import Citation
from rag.vector_store import RetrievedChunk, VectorStoreError, VectorStoreManager
from utils.constants import AnswerMode
from utils.text_utils import clean_text, truncate

__all__ = ["RetrievalAgent"]

_DEICTIC_RE = re.compile(
    r"\b(it|its|this|that|these|those|they|them|above|again|same|such|there)\b",
    re.IGNORECASE,
)
_MAX_QUERY_CHARS = 600
_PREVIEW_CHARS = 220


class RetrievalAgent(BaseAgent):
    """Retrieves grounded, citable passages for the answer agent."""

    name = "retrieval"

    def __init__(self, settings, vector_store: VectorStoreManager, *, prompts=None) -> None:
        super().__init__(settings, prompts=prompts)
        self.vector_store = vector_store

    def run(self, state: dict[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
        question = clean_text(state.get("question"))
        subject = str(state.get("subject") or "general")
        mode = AnswerMode.coerce(state.get("mode"), default=self.settings.answer_mode)
        top_k = self._passage_count(state, mode)
        notices = list(state.get("notices") or [])
        query = self._build_query(question, state)

        if not self.vector_store.is_ready():
            add_notice(
                notices,
                "No FAISS index found. Run `python -m rag.ingest --rebuild` to index your PDFs; "
                "until then answers come from general NEET knowledge.",
            )
            emit_event(config, self.name, "No knowledge base found - answering without retrieval")
            return {
                "query": query,
                "retrieved": [],
                "context": "",
                "citations": [],
                "notices": notices,
            }

        emit_event(
            config,
            self.name,
            f"Searching the knowledge base ({subject}) for the top {top_k} passages "
            f"- {mode.emoji} {mode.label}",
        )
        try:
            chunks = self.vector_store.search(
                query,
                k=top_k,
                subject=None if subject == "general" else subject,
            )
        except VectorStoreError as exc:
            self.logger.error("Vector search failed: %s", exc)
            raise

        if not chunks and subject != "general":
            self.logger.info("No %s passage matched; retrying across all subjects", subject)
            chunks = self.vector_store.search(query, k=top_k, subject=None)
            if chunks:
                add_notice(
                    notices,
                    f"No {subject} material matched this question, so the whole library was searched.",
                )

        if not chunks:
            add_notice(
                notices,
                "No relevant passage was found in the indexed books for this question.",
            )
            self.logger.info("Retrieval returned no chunks above the score threshold")

        context, citations = self.format_context(chunks)
        emit_event(
            config,
            self.name,
            f"Found {len(citations)} relevant passage(s)"
            if citations
            else "No relevant passage found in the knowledge base",
        )
        return {
            "query": query,
            "mode": mode.value,
            "retrieved": [
                {
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                    "score": chunk.score,
                }
                for chunk in chunks
            ],
            "context": context,
            "citations": citations,
            "notices": notices,
        }

    # ------------------------------------------------------------------ budget
    def _passage_count(self, state: Mapping[str, Any], mode: AnswerMode) -> int:
        """Scale the requested passage count with the answer mode.

        Quick Revision wants fewer, tighter passages; MCQ and PYQ practice want
        extra passages so distractors and question patterns have material.
        """
        base = int(state.get("top_k") or self.settings.top_k)
        scaled = max(1, round(base * mode.retrieval_factor))
        if scaled != base:
            self.logger.debug("Mode %s scaled top_k %d -> %d", mode.value, base, scaled)
        return scaled

    # ------------------------------------------------------------------ query
    def _build_query(self, question: str, state: Mapping[str, Any]) -> str:
        """Add the previous question when the new message is a short follow-up."""
        history = list(state.get("history") or [])
        if not self._is_followup(question) or not history:
            return truncate(question, _MAX_QUERY_CHARS, suffix="")

        previous = next(
            (
                clean_text(turn.get("content"))
                for turn in reversed(history)
                if str(turn.get("role")) == "user" and turn.get("content")
            ),
            "",
        )
        if not previous:
            return question
        merged = f"{truncate(previous, 300, suffix='')} {question}"
        self.logger.debug("Rewrote follow-up query: %s", merged[:120])
        return truncate(merged, _MAX_QUERY_CHARS, suffix="")

    @staticmethod
    def _is_followup(question: str) -> bool:
        """Short messages with pronouns usually refer to the previous turn."""
        words = question.split()
        return len(words) <= 12 and bool(_DEICTIC_RE.search(question))

    # ---------------------------------------------------------------- context
    def format_context(self, chunks: list[RetrievedChunk]) -> tuple[str, list[Citation]]:
        """Build the numbered context block and matching citation records.

        Blocks are added until ``context_char_budget`` is reached; the last one
        is trimmed so a single huge passage cannot crowd out the rest.
        """
        blocks: list[str] = []
        citations: list[Citation] = []
        used = 0

        for index, chunk in enumerate(chunks, start=1):
            page = f" | page {chunk.page}" if chunk.page else ""
            header = f"[{index}] {chunk.file_name}{page} | subject: {chunk.subject}"
            remaining = self.settings.context_char_budget - used - len(header) - 2
            if remaining < 200:
                self.logger.debug("Context budget reached; dropping %d passage(s)", len(chunks) - index + 1)
                break

            body = truncate(chunk.content.strip(), remaining, suffix=" ...")
            block = f"{header}\n{body}"
            blocks.append(block)
            used += len(block) + 2
            citations.append(
                Citation(
                    index=index,
                    source=chunk.source,
                    file_name=chunk.file_name,
                    page=chunk.page,
                    subject=chunk.subject,
                    score=round(chunk.score, 3),
                    preview=truncate(clean_text(chunk.content), _PREVIEW_CHARS, suffix="..."),
                )
            )

        return "\n\n".join(blocks), citations
