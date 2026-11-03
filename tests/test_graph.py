"""End-to-end tests for the LangGraph pipeline (offline, mock LLM)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agents.graph import TutorGraph
from agents.tutor import NEETTutor, citations_to_markdown
from config import Settings
from llm.mock_client import MockLLM
from prompts.loader import PromptLibrary
from rag.vector_store import VectorStoreManager


@pytest.fixture
def graph(settings: Settings, vector_store: VectorStoreManager) -> TutorGraph:
    return TutorGraph(
        settings,
        MockLLM(settings),
        vector_store,
        PromptLibrary(settings.prompt_dir),
    )


def test_graph_answers_with_citations(graph: TutorGraph) -> None:
    result = graph.ask("Why is light required for photosynthesis?")
    assert result.ok
    assert result.subject == "biology"
    assert result.intent == "doubt"
    assert result.citations, "retrieved passages must be cited"
    assert "Mock answer" in result.answer
    assert {"router", "retrieval", "answer"} <= set(result.timings)


def test_graph_citation_metadata(graph: TutorGraph) -> None:
    result = graph.ask("Explain photosynthesis in the chloroplast.")
    citation = result.citations[0]
    assert citation["index"] == 1
    assert citation["file_name"].endswith(".pdf")
    assert citation["page"] in {7, 12, 45}
    assert 0.0 <= citation["score"] <= 1.0
    markdown = citations_to_markdown(result.citations)
    assert "score" not in markdown
    assert "plant_physiology.pdf" in markdown and "p.45" in markdown


def test_graph_streams_tokens(settings: Settings, vector_store: VectorStoreManager) -> None:
    streaming_settings = settings.model_copy(update={"stream_tokens": True})
    graph = TutorGraph(
        streaming_settings,
        MockLLM(streaming_settings),
        vector_store,
        PromptLibrary(streaming_settings.prompt_dir),
    )
    tokens: list[str] = []
    events: list[dict[str, Any]] = []

    result = graph.ask(
        "State Newton's second law of motion.",
        on_token=tokens.append,
        on_event=events.append,
    )
    assert tokens, "the mock LLM must stream fragments through on_token"
    assert "".join(tokens).strip()
    assert result.ok
    assert any(event["node"] == "retrieval" for event in events)


def test_graph_defers_unimplemented_agent(graph: TutorGraph) -> None:
    result = graph.ask("Generate 5 MCQs on kinematics.")
    assert result.intent == "mcq"
    assert "coming soon" in result.answer.lower()
    assert result.citations == []


def test_graph_survives_missing_index(settings: Settings) -> None:
    empty_settings = settings.model_copy(
        update={"vector_store_dir": Path(settings.vector_store_dir) / "empty"}
    )
    graph = TutorGraph(
        empty_settings,
        MockLLM(empty_settings),
        VectorStoreManager(empty_settings),
        PromptLibrary(empty_settings.prompt_dir),
    )
    result = graph.ask("Define electric potential.")
    assert result.answer
    assert any("no faiss index" in notice.lower() for notice in result.notices)
    assert result.citations == []


def test_graph_rejects_empty_question(graph: TutorGraph) -> None:
    result = graph.ask("   ")
    assert not result.ok
    assert "question" in result.answer.lower()


def test_tutor_facade_exposes_health(
    settings: Settings, vector_store: VectorStoreManager
) -> None:
    tutor = NEETTutor(settings, llm=MockLLM(settings), vector_store=vector_store)
    tutor.ensure_ready()

    health = tutor.health()
    assert health["prompts"]["ok"] is True
    assert health["index"]["ready"] is True
    assert health["index"]["chunks"] == 3
    assert health["llm"]["backend"] == "mock"

    result = tutor.ask("What is the mole concept?", subject="chemistry")
    assert result.ok
    assert result.subject == "chemistry"
    assert "Subject Router" in tutor.architecture()
