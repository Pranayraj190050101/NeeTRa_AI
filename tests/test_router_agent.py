"""Tests for the Subject Router Agent (LLM JSON + keyword fallback)."""

from __future__ import annotations

from typing import Any, Sequence

import pytest

from agents.router_agent import RouterAgent
from config import Settings
from llm.base import ChatMessage, LLMClient, TokenCallback
from llm.mock_client import MockLLM
from prompts.loader import PromptLibrary


class _BrokenLLM(LLMClient):
    """Returns unparsable text; used to exercise the keyword fallback."""

    name = "broken"

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        on_token: TokenCallback | None = None,
    ) -> str:
        return "I am not sure, but maybe it is physics?"


@pytest.fixture
def prompt_library(settings: Settings) -> PromptLibrary:
    return PromptLibrary(settings.prompt_dir)


def _run(agent: RouterAgent, question: str, subject_hint: str = "auto") -> dict[str, Any]:
    return agent(
        {"question": question, "history": [], "subject_hint": subject_hint, "timings": {}},
        {},
    )


def test_router_uses_llm_json(settings: Settings, prompt_library: PromptLibrary) -> None:
    agent = RouterAgent(settings, llm=MockLLM(settings), prompts=prompt_library)
    update = _run(agent, "Explain Newton's second law of motion with an example.")
    assert update["subject"] == "physics"
    assert update["intent"] == "doubt"
    assert 0.0 <= update["subject_confidence"] <= 1.0
    assert "router" in update["timings"]


def test_router_detects_practice_intent(settings: Settings, prompt_library: PromptLibrary) -> None:
    agent = RouterAgent(settings, llm=MockLLM(settings), prompts=prompt_library)
    update = _run(agent, "Generate 10 MCQs on the periodic table trends.")
    assert update["intent"] == "mcq"
    assert update["subject"] == "chemistry"


def test_router_honours_explicit_subject(settings: Settings, prompt_library: PromptLibrary) -> None:
    agent = RouterAgent(settings, llm=MockLLM(settings), prompts=prompt_library)
    update = _run(agent, "Explain the role of the sinoatrial node.", subject_hint="biology")
    assert update["subject"] == "biology"
    assert update["subject_confidence"] == 1.0
    assert "selected by the student" in update["router_reason"].lower()


def test_router_falls_back_when_json_is_invalid(
    settings: Settings, prompt_library: PromptLibrary
) -> None:
    agent = RouterAgent(settings, llm=_BrokenLLM(), prompts=prompt_library)
    update = _run(agent, "A capacitor of 2 microfarad is charged to 100 V. Find the energy stored.")
    assert update["subject"] == "physics"
    assert update["intent"] == "doubt"
    assert any("keyword" in notice.lower() for notice in update.get("notices", []))


def test_router_handles_general_questions(
    settings: Settings, prompt_library: PromptLibrary
) -> None:
    agent = RouterAgent(settings, llm=MockLLM(settings), prompts=prompt_library)
    update = _run(agent, "Hello, how should I start my revision today?")
    assert update["subject"] in {"general", "biology", "chemistry", "physics"}
    assert update["intent"] in {"doubt", "study_plan"}
