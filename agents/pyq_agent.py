"""Retrieve genuine previous-year questions from explicitly identified sources."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from agents.base import BaseAgent, emit_event
from agents.retrieval_agent import RetrievalAgent
from rag.vector_store import RetrievedChunk
from utils.text_utils import clean_text, truncate

__all__ = ["PYQRetrievalAgent"]

_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_PYQ_SOURCE_RE = re.compile(
    r"(?:pyq|previous[\W_]*year|past[\W_]*paper|aipmt|"
    r"neet[\W_]*(?:19|20)\d{2})",
    re.IGNORECASE,
)


class PYQRetrievalAgent(BaseAgent):
    """Search indexed question sources without turning generated content into PYQs."""

    name = "pyq_retrieval"

    def __init__(self, settings, *, platform: Any, llm=None, prompts=None) -> None:
        super().__init__(settings, llm=llm, prompts=prompts)
        self.platform = platform
        self.formatter = RetrievalAgent(settings, platform.vector_store, prompts=self.prompts)

    def run(self, state: dict[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
        question = clean_text(state.get("question"))
        subject = str(state.get("subject") or "general")
        year_match = _YEAR_RE.search(question)
        requested_year = year_match.group(1) if year_match else None
        query = f"{question} previous year NEET question"
        emit_event(config, self.name, "Searching indexed previous-year question sources")

        if not self.platform.vector_store.is_ready():
            return self._not_found(state, requested_year, "No question index is available yet.")

        chunks = self.platform.vector_store.search(
            query,
            k=max(1, min(int(state.get("top_k") or self.settings.top_k), 12)),
            subject=None if subject == "general" else subject,
        )
        matches = [
            chunk
            for chunk in chunks
            if _is_pyq_source(chunk)
            and (requested_year is None or requested_year in _years(chunk))
        ]
        if not matches:
            detail = (
                f"No indexed PYQ source matched {requested_year}."
                if requested_year
                else "No clearly identified PYQ source matched this topic."
            )
            return self._not_found(state, requested_year, detail)

        context, citations = self.formatter.format_context(matches)
        cited_sources = {(item["source"], item["page"]) for item in citations}
        matches = [
            chunk for chunk in matches if (chunk.source, chunk.page) in cited_sources
        ]
        if not matches:
            return self._not_found(state, requested_year, "No PYQ passages fit the response context.")
        lines = [f"### Previous-Year Questions: {question}", ""]
        for index, chunk in enumerate(matches, start=1):
            years = _years(chunk)
            label = (
                f"NEET {requested_year or years[0]}"
                if (requested_year or years)
                else "Indexed PYQ"
            )
            excerpt = truncate(clean_text(chunk.content), 900, suffix="...")
            lines.extend([f"**{label} · Item {index}**", "", excerpt, ""])
        return {
            "answer": "\n".join(lines).strip(),
            "mode": str(state.get("mode") or "pyq_focus"),
            "context": context,
            "citations": citations,
            "notices": list(state.get("notices") or []),
        }

    @staticmethod
    def _not_found(state: Mapping[str, Any], year: str | None, detail: str) -> dict[str, Any]:
        qualifier = f" for {year}" if year else ""
        answer = (
            f"### Previous-Year Questions{qualifier}\n\n{detail} "
            "I won't label newly generated practice questions as authentic PYQs. "
            "Add previous-year question files to the study library and re-index them."
        )
        return {
            "answer": answer,
            "mode": str(state.get("mode") or "pyq_focus"),
            "context": "",
            "citations": [],
            "notices": list(state.get("notices") or []),
        }


def _is_pyq_source(chunk: RetrievedChunk) -> bool:
    metadata = chunk.metadata
    source = " ".join(
        str(value or "")
        for value in (
            metadata.get("source"),
            metadata.get("file_name"),
            metadata.get("document_type"),
            metadata.get("collection"),
        )
    )
    return bool(_PYQ_SOURCE_RE.search(source))


def _years(chunk: RetrievedChunk) -> list[str]:
    metadata_year = str(chunk.metadata.get("year") or "")
    evidence = " ".join(
        [
            metadata_year,
            str(chunk.metadata.get("source") or ""),
            str(chunk.metadata.get("file_name") or ""),
        ]
    )
    return sorted(set(_YEAR_RE.findall(evidence)))
