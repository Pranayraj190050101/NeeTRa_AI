"""End-to-end tests for the LangGraph pipeline (offline, mock LLM)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agents.graph import (
    NODE_MCQ,
    NODE_PYQ,
    NODE_STUDY_PLAN,
    NODE_WEAKNESS,
    TutorGraph,
)
from agents.tutor import NEETTutor, citations_to_markdown
from config import Settings
from llm.mock_client import MockLLM
from prompts.loader import PromptLibrary
from rag.vector_store import RetrievedChunk, VectorStoreManager
from utils.constants import AnswerMode


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


def test_graph_generates_mcqs_in_any_answer_mode(graph: TutorGraph) -> None:
    result = graph.ask("Generate 3 MCQs on Newton's second law.")
    assert result.intent == "mcq"
    assert NODE_MCQ in result.timings
    assert "Answer Key" in result.answer
    assert "Explanation" in result.answer
    assert "coming soon" not in result.answer.lower()


def test_graph_only_returns_indexed_previous_year_questions(graph: TutorGraph) -> None:
    result = graph.ask("Show me 2024 PYQs on photosynthesis.")
    assert result.intent == "pyq"
    assert NODE_PYQ in result.timings
    assert "No indexed PYQ source matched 2024." in result.answer
    assert result.citations == []


def test_graph_returns_only_identified_pyq_source_chunks(
    graph: TutorGraph, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunk = RetrievedChunk(
        content="In which phase of the cell cycle does DNA replication occur?",
        metadata={
            "source": "biology/NEET_2024_PYQ_cell_cycle.pdf",
            "file_name": "NEET_2024_PYQ_cell_cycle.pdf",
            "subject": "biology",
            "page": 8,
        },
        score=0.93,
    )
    monkeypatch.setattr(
        graph.vector_store,
        "search",
        lambda query, *, k, subject=None: [chunk],
    )
    result = graph.ask("Show me 2024 PYQs on the cell cycle.")
    assert "DNA replication occur" in result.answer
    assert len(result.citations) == 1
    assert result.citations[0]["file_name"] == "NEET_2024_PYQ_cell_cycle.pdf"


def test_graph_builds_a_plan_from_the_indexed_catalog(graph: TutorGraph) -> None:
    result = graph.ask("Create a 3-week revision plan from my syllabus.")
    assert result.intent == "study_plan"
    assert NODE_STUDY_PLAN in result.timings
    assert "3-Week NEET Revision Plan" in result.answer
    assert "Mechanics" in result.answer
    assert "Mole Concept" in result.answer
    assert "Plant Physiology" in result.answer


def test_study_plan_prioritizes_weak_chapters_from_practice_history(graph: TutorGraph) -> None:
    test = graph.platform.generate_test(
        scope="chapter", subject="physics", target="Mechanics", count=2
    )
    graph.platform.submit_test(test, [4, 4])
    result = graph.ask("Create a 3-week revision plan from my syllabus.")
    first_week = result.answer.split("#### Week 1", 1)[1].split("#### Week 2", 1)[0]
    assert "Mechanics" in first_week
    assert "Weak-area focus" in first_week


def test_graph_reports_missing_weakness_history(graph: TutorGraph) -> None:
    result = graph.ask("Analyze my weak areas.")
    assert result.intent == "weakness"
    assert NODE_WEAKNESS in result.timings
    assert "Complete and submit a test" in result.answer


def test_weakness_analysis_uses_recorded_quiz_results(graph: TutorGraph) -> None:
    test = graph.platform.generate_test(
        scope="topic", subject="physics", target="Mechanics", count=2
    )
    graph.platform.submit_test(test, [4, 4])
    result = graph.ask("Analyze my weak areas.")
    assert "Mechanics" in result.answer
    assert "incorrect" in result.answer


def test_graph_routes_topic_revision_notes_as_a_doubt(graph: TutorGraph) -> None:
    result = graph.ask(
        "Give me quick revision notes for Molecular Basis of Inheritance.",
        mode=AnswerMode.QUICK_REVISION,
    )
    assert result.intent == "doubt"
    assert result.subject == "biology"
    assert "Quick Revision Notes" in result.answer
    assert "Mock answer" in result.answer


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


def test_study_plan_is_honest_when_the_catalog_is_empty(settings: Settings) -> None:
    empty_settings = settings.model_copy(
        update={"vector_store_dir": Path(settings.vector_store_dir) / "empty-catalog"}
    )
    graph = TutorGraph(
        empty_settings,
        MockLLM(empty_settings),
        VectorStoreManager(empty_settings),
        PromptLibrary(empty_settings.prompt_dir),
    )
    result = graph.ask("Build a week-by-week study plan.")
    assert result.intent == "study_plan"
    assert "until the chapter catalog is indexed" in result.answer
    assert "Mechanics" not in result.answer


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
