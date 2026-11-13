"""End-to-end coach tests: persona, answer modes and the preserved RAG path.

Everything runs offline (mock LLM + fake embeddings), so these tests assert the
behaviour the student sees: the same LangGraph + FAISS pipeline, wrapped in the
Neet Genie persona and delivered in the selected mode's section format.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from agents.answer_agent import AnswerAgent
from agents.graph import (
    NODE_ANSWER,
    NODE_DEFERRED,
    NODE_RETRIEVAL,
    NODE_ROUTER,
    TutorGraph,
)
from agents.retrieval_agent import RetrievalAgent
from agents.tutor import NEETTutor
from config import Settings
from llm.mock_client import MockLLM
from prompts.loader import PromptLibrary
from rag.vector_store import VectorStoreManager
from utils.answer_format import sections_for
from utils.constants import AnswerMode


@pytest.fixture
def prompts(settings: Settings) -> PromptLibrary:
    return PromptLibrary(settings.prompt_dir)


def _agent(settings: Settings, prompts: PromptLibrary) -> AnswerAgent:
    return AnswerAgent(settings, llm=MockLLM(settings), prompts=prompts)


def _graph(settings: Settings, prompts: PromptLibrary, vector_store: VectorStoreManager) -> TutorGraph:
    return TutorGraph(settings, MockLLM(settings), vector_store, prompts)


# ------------------------------------------------------------------- persona
@pytest.mark.parametrize("mode", list(AnswerMode))
def test_system_prompt_combines_persona_grounding_and_mode(
    settings: Settings, prompts: PromptLibrary, mode: AnswerMode
) -> None:
    text = _agent(settings, prompts).system_prompt(mode)

    # 1. NEET faculty persona
    assert "NEET GENIE" in text
    assert "teach for understanding" in text.lower()
    assert "search results" in text.lower()
    assert "never invent a neet question pattern" in text.lower()
    if mode in (AnswerMode.DETAILED, AnswerMode.QUICK_REVISION):
        assert "do not restate a fact" in text.lower()
        assert "do not cite irrelevant blocks" in text.lower()
    # 2. the existing grounding rules are still injected
    assert "Reference material" in text or "reference" in text.lower()
    # 3. machine readable marker + the mode contract, in prompt order
    assert f"MODE_ID: {mode.value}" in text
    for title in sections_for(mode):
        assert title.split(" ", 1)[1] in text, f"{mode.value} must announce '{title}'"
    # No unresolved template placeholder survives composition (LaTeX dollar signs
    # such as "$v = u + at$" are legitimate, so only check the declared names).
    for token in ("$mcq_count", "$mode", "$question", "$history", "$context", "$subject"):
        assert token not in text


def test_persona_can_be_switched_off_without_losing_the_mode(
    settings: Settings, prompts: PromptLibrary
) -> None:
    quiet = settings.model_copy(update={"persona_enabled": False})
    text = _agent(quiet, prompts).system_prompt(AnswerMode.QUICK_REVISION)

    assert "NEET GENIE" not in text
    assert "MODE_ID: quick_revision" in text


def test_prompt_library_still_validates_with_the_coach_prompts(prompts: PromptLibrary) -> None:
    assert prompts.validate() == []


# ------------------------------------------------------------------- retrieval
def test_mode_scales_the_passage_count_without_touching_rag(
    settings: Settings, vector_store: VectorStoreManager, prompts: PromptLibrary
) -> None:
    agent = RetrievalAgent(settings, vector_store, prompts=prompts)
    state = {"question": "Define inertia.", "subject": "physics", "top_k": 5}

    assert agent._passage_count(state, AnswerMode.DETAILED) == 5
    assert agent._passage_count(state, AnswerMode.QUICK_REVISION) == 4
    assert agent._passage_count(state, AnswerMode.MCQ_PRACTICE) == 7
    assert agent._passage_count(state, AnswerMode.PYQ_FOCUS) == 6
    assert agent._passage_count({"top_k": 1}, AnswerMode.QUICK_REVISION) == 1, "never ask for zero passages"


def test_retrieval_result_is_identical_apart_from_the_count(
    settings: Settings, vector_store: VectorStoreManager, prompts: PromptLibrary
) -> None:
    agent = RetrievalAgent(settings, vector_store, prompts=prompts)
    config: dict[str, Any] = {}
    state = {"question": "Explain photosynthesis in the chloroplast.", "subject": "biology", "top_k": 2}

    detailed = agent.run(dict(state) | {"mode": "detailed"}, config)
    revision = agent.run(dict(state) | {"mode": "quick_revision"}, config)

    assert detailed["mode"] == "detailed"
    assert revision["context"].split("\n\n")[0] == detailed["context"].split("\n\n")[0]
    assert len(revision["citations"]) <= len(detailed["citations"])


# ---------------------------------------------------------------------- graph
def test_mode_reaches_the_state_and_the_final_result(
    settings: Settings, prompts: PromptLibrary, vector_store: VectorStoreManager
) -> None:
    graph = _graph(settings, prompts, vector_store)

    result = graph.ask("Explain photosynthesis in the chloroplast.", mode="quick_revision")

    assert result.mode == "quick_revision"
    assert result.answer_mode is AnswerMode.QUICK_REVISION
    assert {NODE_ROUTER, NODE_RETRIEVAL, NODE_ANSWER} <= set(result.timings)
    assert NODE_DEFERRED not in result.timings


def test_every_mode_delivers_its_own_section_contract(
    settings: Settings, prompts: PromptLibrary, vector_store: VectorStoreManager
) -> None:
    graph = _graph(settings, prompts, vector_store)

    for mode in AnswerMode:
        result = graph.ask("Explain photosynthesis in the chloroplast.", mode=mode)
        headings = [line for line in result.answer.split("\n") if line.startswith("### ")]
        for title in sections_for(mode):
            assert f"### {title}" in headings, f"{mode.value} is missing its '{title}' section"
        assert "Mock answer" in result.answer, "the RAG pipeline and citations stay in place"
        assert result.citations, f"{mode.value} must still cite the retrieved textbook passages"


def test_ui_labels_are_accepted_by_the_graph(
    settings: Settings, prompts: PromptLibrary, vector_store: VectorStoreManager
) -> None:
    graph = _graph(settings, prompts, vector_store)

    result = graph.ask("Explain photosynthesis.", mode="MCQ Practice")

    assert result.mode == "mcq_practice"
    assert "\U0001f9e9 Practice MCQs" in result.answer


def test_an_explicit_mode_serves_an_mcq_request_without_deferring(
    settings: Settings, prompts: PromptLibrary, vector_store: VectorStoreManager
) -> None:
    graph = _graph(settings, prompts, vector_store)

    served = graph.ask("Generate 5 MCQs on Newton's laws.", mode=AnswerMode.MCQ_PRACTICE)
    deferred = graph.ask("Generate 5 MCQs on Newton's laws.", mode=AnswerMode.DETAILED)
    planned = graph.ask("Make a 3 month study plan for NEET.", mode=AnswerMode.MCQ_PRACTICE)

    # MCQ Practice keeps the turn inside the real pipeline (retrieval + answer).
    assert NODE_ANSWER in served.timings and NODE_RETRIEVAL in served.timings
    assert "coming soon" not in served.answer.lower()
    assert served.mode == "mcq_practice"

    # Without a matching mode the polite deferral still happens, as before.
    assert "coming soon" in deferred.answer.lower() and deferred.citations == []
    assert NODE_ANSWER not in deferred.timings
    assert "MCQ Practice" in deferred.answer, "the deferred node should name the mode that can help"

    # No mode can serve a study plan yet.
    assert "coming soon" in planned.answer.lower()
    assert NODE_ANSWER not in planned.timings


def test_format_repair_is_reported_as_a_notice(
    settings: Settings, prompts: PromptLibrary, vector_store: VectorStoreManager
) -> None:
    loose = settings.model_copy(update={"enforce_answer_format": False})
    strict = _graph(settings, prompts, vector_store)
    lenient_prompts = PromptLibrary(loose.prompt_dir)
    lenient = TutorGraph(loose, MockLLM(loose), vector_store, lenient_prompts)

    question = "Explain photosynthesis in the chloroplast."
    repaired = strict.ask(question, mode=AnswerMode.DETAILED)
    raw = lenient.ask(question, mode=AnswerMode.DETAILED)

    assert raw.answer.count("### ") == repaired.answer.count("### ")  # mock already conforms
    assert not any("Coach format applied" in notice for notice in raw.notices)

    # A model that ignores the contract gets repaired, and the student is told.
    ragged = "The chloroplast is where photosynthesis happens. Light reactions make ATP."
    update = _agent(settings, prompts)._postprocess(ragged, mode=AnswerMode.QUICK_REVISION, subject="biology", context="", notices=[])
    assert any("Coach format applied" in notice for notice in update[1])
    for title in sections_for(AnswerMode.QUICK_REVISION):
        assert title in update[0]


def test_answer_postprocess_keeps_only_citations_from_supplied_blocks(
    settings: Settings, prompts: PromptLibrary
) -> None:
    agent = _agent(settings, prompts)
    raw = (
        "### Concept Explanation\n\nClaim [1]. Unrelated claim [2]. Unsupported claim [8]. "
        r"Formula: K_{max}=h\nu-\phiK_{max}=h\nu-\phi." "\n\n"
        "### NEET Importance\n\nA testable point.\n\n"
        "### Key Points\n\n- Point.\n\n"
        "### Common Mistakes\n\n- Mistake.\n\n"
        "### Quick Revision\n\n- Recall."
    )
    context = (
        "[1] photoelectric_effect.pdf | page 4 | subject: physics\nThe photoelectric effect.\n\n"
        "[2] mechanics.pdf | page 11 | subject: physics\nKinetic energy and potential energy."
    )

    answer, _ = agent._postprocess(
        raw,
        mode=AnswerMode.QUICK_REVISION,
        subject="physics",
        context=context,
        question="Explain the photoelectric effect for NEET",
        notices=[],
    )
    no_context_answer, _ = agent._postprocess(
        raw,
        mode=AnswerMode.QUICK_REVISION,
        subject="physics",
        context="",
        notices=[],
    )
    irrelevant_context_answer, _ = agent._postprocess(
        "### Concept Explanation\n\nThe photoelectric effect ejects electrons.",
        mode=AnswerMode.QUICK_REVISION,
        subject="physics",
        context="[1] mechanics.pdf | page 11 | subject: physics\nKinetic energy and potential energy.",
        question="Explain the photoelectric effect for NEET",
        notices=[],
    )

    assert "[1]" in answer
    assert "[8]" not in answer
    assert "[2]" not in answer
    assert answer.count(r"$K_{max}=h\nu-\phi$") == 1
    assert "[1]" not in no_context_answer
    assert "kinetic energy" not in irrelevant_context_answer.lower()


def test_answer_postprocess_repairs_the_reported_formula_corruption_and_limits_revision(
    settings: Settings, prompts: PromptLibrary
) -> None:
    agent = _agent(settings, prompts)
    raw = r"""### 📘 Concept Explanation
The photoelectric effect is the emission of electrons when light of sufficient frequency strikes a metal.
Photon energy is E=hfE=hf, where hh is Planck's constant and ff is frequency.
The maximum kinetic energy is K\_{$max} = hf - \\phi_0$.

### 🎯 NEET Importance
NEET frequently tests KmaxKmax and the stopping potential eV0=Kmax.

### 🧠 Key Points To Remember
- KmaxKmax = hf - ϕ0ϕ0.
- f0=ϕ0hf0=hϕ0.

### ⚠ Common Student Mistakes
- Increasing intensity below threshold frequency causes emission.
- Intensity changes KmaxKmax.

### ✅ Quick Revision Notes
- E=hfE=hf.
- K\_{$max} = eV_0$.
- f<f0 -> no emission."""

    answer, _ = agent._postprocess(
        raw,
        mode=AnswerMode.QUICK_REVISION,
        subject="physics",
        context="",
        notices=[],
        question="Explain the photoelectric effect for NEET.",
    )

    assert answer.count("$E=hf$") == 1
    assert answer.count("$K_{max}= hf - \\phi_0$") == 1
    assert r"K\_{$max}" not in answer
    assert "E=hfE=hf" not in answer
    assert "KmaxKmax" not in answer
    assert "NEET frequently tests" not in answer
    assert sum(len(re.findall(r"[\w]+(?:['’-][\w]+)*", line)) for line in answer.splitlines()) <= 155


def test_mock_backend_follows_the_mode_marker(
    settings: Settings, prompts: PromptLibrary
) -> None:
    agent = _agent(settings, prompts)
    llm = MockLLM(settings)

    for mode in AnswerMode:
        messages = [
            {"role": "system", "content": agent.system_prompt(mode)},
            {"role": "user", "content": "STUDENT'S QUESTION\nExplain photosynthesis."},
        ]
        answer = llm.generate(messages)
        assert f"**Answer mode:** {mode.emoji} {mode.label}" in answer
        assert all(title in answer for title in sections_for(mode))


# ---------------------------------------------------------------------- facade
def test_tutor_facade_passes_the_mode_through(
    settings: Settings, vector_store: VectorStoreManager
) -> None:
    tutor = NEETTutor(settings, llm=MockLLM(settings), vector_store=vector_store)

    result = tutor.ask("Define electric potential.", subject="physics", mode="pyq_focus")

    assert result.ok
    assert result.mode == "pyq_focus"


def test_startup_verifies_generation_and_retrieval(
    settings: Settings, vector_store: VectorStoreManager
) -> None:
    class ProbeLLM(MockLLM):
        def __init__(self, settings: Settings) -> None:
            super().__init__(settings)
            self.readiness_calls = 0

        def generate(self, *args: Any, **kwargs: Any) -> str:
            self.readiness_calls += 1
            return super().generate(*args, **kwargs)

    llm = ProbeLLM(settings)
    tutor = NEETTutor(settings, llm=llm, vector_store=vector_store)

    status = tutor.validate_startup()

    assert llm.readiness_calls == 1
    assert status["ready"] is True
    assert all(check["ready"] for check in status["checks"].values())
    assert "Ask Genie" in status["student_message"]
    assert tutor.startup_status() == status


def test_startup_keeps_study_pages_available_without_a_library(settings: Settings) -> None:
    tutor = NEETTutor(
        settings,
        llm=MockLLM(settings),
        vector_store=VectorStoreManager(settings),
    )

    status = tutor.validate_startup()

    assert status["ready"] is False
    assert status["checks"]["knowledge_base"]["ready"] is False
    assert status["checks"]["llm"]["ready"] is True
    assert status["checks"]["retrieval"]["ready"] is False
    assert "Dashboard and Progress" in status["student_message"]


def test_startup_detects_provider_misconfiguration(
    settings: Settings, vector_store: VectorStoreManager
) -> None:
    broken = settings.model_copy(update={"llm_provider": "groq", "groq_api_key": None})
    tutor = NEETTutor(broken, llm=MockLLM(broken), vector_store=vector_store)

    status = tutor.validate_startup()

    assert status["ready"] is False
    assert status["checks"]["llm"]["ready"] is False
    assert "answers are paused" in status["student_message"]
    assert "llm" in status["errors"]


def test_startup_rejects_an_empty_generation_probe(
    settings: Settings, vector_store: VectorStoreManager
) -> None:
    class EmptyLLM(MockLLM):
        def generate(self, *args: Any, **kwargs: Any) -> str:
            return ""

    tutor = NEETTutor(settings, llm=EmptyLLM(settings), vector_store=vector_store)

    status = tutor.validate_startup()

    assert status["checks"]["llm"]["ready"] is False
    assert status["checks"]["retrieval"]["ready"] is True
    assert "empty readiness response" in status["errors"]["llm"]
    assert "empty readiness response" not in status["student_message"]


def test_startup_reports_retrieval_failure_without_raw_details(
    settings: Settings,
    vector_store: VectorStoreManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_search(*args: Any, **kwargs: Any) -> list[Any]:
        raise RuntimeError("private retrieval failure")

    monkeypatch.setattr(vector_store, "search", fail_search)
    tutor = NEETTutor(settings, llm=MockLLM(settings), vector_store=vector_store)

    status = tutor.validate_startup()

    assert status["checks"]["llm"]["ready"] is True
    assert status["checks"]["retrieval"]["ready"] is False
    assert "private retrieval failure" in status["errors"]["retrieval"]
    assert "private retrieval failure" not in status["student_message"]
    assert "Dashboard and Progress" in status["student_message"]


def test_startup_diagnostics_mask_provider_secrets(
    settings: Settings, vector_store: VectorStoreManager
) -> None:
    secret = "gsk_student_secret_123"
    configured = settings.model_copy(update={"groq_api_key": secret})

    class RejectingLLM(MockLLM):
        def generate(self, *args: Any, **kwargs: Any) -> str:
            raise RuntimeError(f"401 authentication failed for {secret}")

    tutor = NEETTutor(configured, llm=RejectingLLM(configured), vector_store=vector_store)

    status = tutor.validate_startup()

    assert "could not sign in" in status["student_message"]
    assert secret not in str(status)
    assert "***" in status["errors"]["llm"]


def test_coach_settings_are_configurable_and_reported(settings: Settings) -> None:
    custom = Settings(
        **{
            **settings.model_dump(),
            "answer_mode": "mcq practice",
            "persona_enabled": False,
            "enforce_answer_format": False,
            "mcq_count": 8,
        }
    )
    tutor = NEETTutor(custom, llm=MockLLM(custom), vector_store=VectorStoreManager(custom))

    coach = tutor.health()["coach"]

    assert coach["default_mode"] == "mcq_practice"
    assert coach["persona_enabled"] is False
    assert coach["enforce_answer_format"] is False
    assert coach["mcq_count"] == 8
    assert [entry["label"] for entry in coach["modes"]] == [mode.label for mode in AnswerMode]


def test_architecture_panel_documents_the_coach_layers(
    settings: Settings, prompts: PromptLibrary, vector_store: VectorStoreManager
) -> None:
    text = _graph(settings, prompts, vector_store).architecture()

    assert "Subject Router Agent" in text, "the original pipeline is still documented"
    assert "RAG Retrieval Agent" in text and "FAISS" in text
    assert "Neet Genie persona" in text
    for mode in AnswerMode:
        assert mode.label in text
