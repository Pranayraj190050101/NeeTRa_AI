"""Tests for the Gradio chat handler (no Gradio import required).

``build_responder`` is a plain generator, so the streaming contract - one
``(chat_history, status, sources)`` tuple per UI update - can be verified
offline with the mock LLM.
"""

from __future__ import annotations

from typing import Any

from app import build_responder
from agents.tutor import NEETTutor
from config import Settings
from llm.mock_client import MockLLM
from rag.vector_store import VectorStoreManager


def _tutor(settings: Settings, vector_store: VectorStoreManager) -> NEETTutor:
    return NEETTutor(settings, llm=MockLLM(settings), vector_store=vector_store)


def _run(
    responder: Any,
    question: str,
    stream: bool = True,
    mode: str = "detailed",
) -> list[tuple[Any, ...]]:
    """Drive the handler exactly like Gradio does: message, chat, subject, mode, top_k, stream."""
    return list(responder(question, [], "auto", mode, 3.0, stream))


def test_responder_yields_chat_status_and_sources(
    settings: Settings, vector_store: VectorStoreManager
) -> None:
    responder = build_responder(_tutor(settings, vector_store), settings)

    updates = _run(responder, "Why is light required for photosynthesis?")

    assert len(updates) > 1, "the handler must yield at least a placeholder and a final answer"
    chat, status, sources = updates[-1]
    assert [message["role"] for message in chat] == ["user", "assistant"]
    assert chat[0]["content"] == "Why is light required for photosynthesis?"
    assert "Mock answer" in chat[1]["content"]
    assert "**Subject:**" in status and "textbook reference" in status
    assert "Intent" not in status and "Mode" not in status
    assert "\U0001f4d8 Concept Explanation" in chat[1]["content"]
    assert "plant_physiology.pdf" in sources and "p.45" in sources


def test_responder_streams_partial_answers(
    settings: Settings, vector_store: VectorStoreManager
) -> None:
    streaming_settings = settings.model_copy(update={"stream_tokens": True})
    responder = build_responder(_tutor(streaming_settings, vector_store), streaming_settings)

    updates = _run(responder, "State Newton's second law of motion.")

    partials = [chat[-1]["content"] for chat, _, _ in updates]
    assert partials[0] == ""
    assert any(partial for partial in partials[:-1]), "tokens must reach the chatbot before the end"


def test_responder_honours_the_selected_mode(
    settings: Settings, vector_store: VectorStoreManager
) -> None:
    responder = build_responder(_tutor(settings, vector_store), settings)

    answer = _run(responder, "Give me practice questions on kinematics.", mode="mcq_practice")[-1][0][-1][
        "content"
    ]

    assert "\U0001f9e9 Practice MCQs" in answer, "the MCQ mode must shape the reply"
    assert "\U0001f4d8 Concept Explanation" not in answer, "explanation sections belong to other modes"


def test_cell_cycle_quick_revision_answers_without_an_index(
    settings: Settings, tmp_path: Any
) -> None:
    offline_settings = settings.model_copy(update={"vector_store_dir": tmp_path / "empty-vectordb"})
    tutor = NEETTutor(
        offline_settings,
        llm=MockLLM(offline_settings),
        vector_store=VectorStoreManager(offline_settings),
    )
    readiness = tutor.validate_startup()
    assert readiness["checks"]["llm"]["ready"]
    assert not readiness["checks"]["retrieval"]["ready"]

    answer = _run(
        build_responder(tutor, offline_settings),
        "Give quick revision notes for Cell Cycle",
        mode="quick_revision",
    )[-1][0][-1]["content"]

    assert "Give quick revision notes for Cell Cycle" in answer
    assert "Quick Revision Notes" in answer
    assert "could not finish" not in answer.lower()


def test_responder_rejects_empty_message(
    settings: Settings, vector_store: VectorStoreManager
) -> None:
    responder = build_responder(_tutor(settings, vector_store), settings)

    update = _run(responder, "   ")[0]

    chat, status, sources = update
    assert chat == []
    assert "type a question" in status.lower()
    assert sources == ""


def test_responder_reports_unexpected_failure(settings: Settings) -> None:
    class _ExplodingTutor:
        def ask(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("boom")

    responder = build_responder(_ExplodingTutor(), settings)  # type: ignore[arg-type]

    chat, status, sources = _run(responder, "Explain the mole concept.")[-1]

    assert "could not finish" in chat[-1]["content"]
    assert "boom" not in chat[-1]["content"]
    assert "Failed" in status
    assert sources == ""
