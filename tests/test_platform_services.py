"""Platform service tests - the modules behind the six tabs, offline.

Everything runs on ``MockLLM`` plus the fake-embedding FAISS index from the
shared fixtures, so the whole learning platform (catalog, Learn, Practice,
Revision, Progress, capabilities) is verifiable without a model download.
"""

from __future__ import annotations

import pytest

from learning.capabilities import (
    CapabilityNotAvailable,
    CapabilityState,
    ImageSolver,
    VoiceTranscriber,
)
from learning.generation import GeneratedDoc
from learning.practice import MAX_TEST_QUESTIONS, PracticeSet
from learning.sections import CHAPTER_SECTIONS, REVISION_SECTIONS
from learning.service import QUICK_ACTIONS, GeniePlatform
from llm.base import LLMError
from llm.mock_client import MockLLM
from utils.constants import AnswerMode, RevisionDeliverable
from utils.constants import TestScope as Scope


class FakeTutor:
    """Minimal stand-in for ``NEETTutor`` - the chat pipeline the platform reuses."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def ask(self, question, **kwargs):
        self.calls.append({"question": question, **kwargs})

        class Result:
            ok = True
            answer = f"coach answer to: {question}"

        return Result()

    def ensure_ready(self):
        return self

    @property
    def llm(self):
        return MockLLM()

    def index_status(self):
        return {"ready": True, "chunks": 3, "files": 3, "subjects": {"physics": 1}}


@pytest.fixture
def tutor() -> FakeTutor:
    return FakeTutor()


@pytest.fixture
def platform(settings, vector_store, tutor) -> GeniePlatform:
    return GeniePlatform(settings, tutor=tutor, vector_store=vector_store, llm=MockLLM(settings))


# --------------------------------------------------------------------- wiring
def test_construction_touches_no_index_and_no_progress_file(platform: GeniePlatform) -> None:
    """Lazy start-up is existing behaviour the platform must not break."""
    assert not platform.progress.file.exists()


def test_catalog_is_read_from_the_index_sidecar(platform: GeniePlatform) -> None:
    catalog = platform.catalog()
    assert sorted(chapter.name for chapter in catalog.chapters) == [
        "Mechanics",
        "Mole Concept",
        "Plant Physiology",
    ]
    assert catalog.subjects == ("physics", "chemistry", "biology")
    assert platform.refresh_catalog() is not None


def test_platform_status_describes_the_modules_and_hooks(platform: GeniePlatform) -> None:
    status = platform.platform_status()
    assert status["modules"] == [
        "Dashboard",
        "Learn",
        "Practice",
        "Ask Genie",
        "Progress",
        "Revision Hub",
    ]
    assert status["catalog"]["chapters"] == 3
    assert status["practice"]["max_questions_per_test"] == platform.settings.test_question_count
    assert {item["value"] for item in status["revision"]} == {item.value for item in RevisionDeliverable}
    assert "VoiceTranscriber.transcribe -> GeniePlatform.ask" in status["extension_hooks"]
    note = platform.health_note()
    assert "3 chapter(s)" in note and "3 capability hook(s)" in note


# ------------------------------------------------------------------ dashboard
def test_dashboard_payload_has_every_section_the_tab_renders(platform: GeniePlatform) -> None:
    payload = platform.dashboard()
    assert set(payload) == {"welcome", "motivation", "stats", "subjects", "actions", "capabilities"}
    assert payload["welcome"]["chapters"] == 3
    assert payload["welcome"]["modules"] == [
        "Ask Genie", "Practice", "Progress", "Revision Hub", "Dashboard"
    ]
    assert payload["stats"]["tests"] == 0
    assert payload["motivation"]["line"]
    assert [card["subject"] for card in payload["subjects"]] == ["physics", "chemistry", "biology"]
    assert len(payload["actions"]) == len(QUICK_ACTIONS)


def test_subject_cards_report_coverage_and_verdict(platform: GeniePlatform) -> None:
    cards = {card["subject"]: card for card in platform.subject_cards()}
    assert cards["physics"]["chapters"] == 1
    assert cards["chemistry"]["status"] == "Start a test"
    assert cards["biology"]["accuracy_percent"] == 0
    platform.submit_test(
        platform.generate_test(scope=Scope.CHAPTER, subject="physics", target="Mechanics", count=2),
        [1, 1],
    )
    refreshed = {card["subject"]: card for card in platform.subject_cards()}
    assert refreshed["physics"]["tests"] == 1
    assert refreshed["physics"]["status"] in {"Exam ready", "Revise weak spots", "Needs work"}


def test_quick_actions_address_real_tabs_and_controls() -> None:
    assert {action["key"] for action in QUICK_ACTIONS} == {
        "ask_doubt",
        "practice_mcqs",
        "quick_revision",
        "mock_test",
    }
    for action in QUICK_ACTIONS:
        assert action["tab"] in {"dashboard", "learn", "practice", "ask-genie", "progress", "revision"}
        if action["selector"] == "mode":
            assert AnswerMode.coerce(action["value"]) is not None
        elif action["selector"] == "scope":
            assert Scope.coerce(action["value"]) is not None


# ----------------------------------------------------------------------- learn
def test_study_chapter_produces_the_four_contract_sections(platform: GeniePlatform) -> None:
    doc = platform.study_chapter("Mechanics", "physics")
    assert isinstance(doc, GeneratedDoc) and doc.ok
    assert doc.kind == "chapter"
    for section in CHAPTER_SECTIONS:
        assert section in doc.markdown
    assert doc.citations and doc.citations[0]["file_name"] == "mechanics.pdf"
    assert platform.progress.stats()["chapter_pages"] == 1


def test_study_chapter_requires_an_indexed_chapter(platform: GeniePlatform) -> None:
    doc = platform.study_chapter("Quantum tunnelling spectroscopy", "physics")
    assert not doc.ok
    assert "not in your index" in doc.markdown
    assert platform.progress.stats()["chapter_pages"] == 0


def test_study_topic_works_from_free_text(platform: GeniePlatform) -> None:
    doc = platform.study_topic("mole concept and molar mass", "chemistry")
    assert doc.ok and doc.kind == "topic"
    assert any(section in doc.markdown for section in CHAPTER_SECTIONS)


class ExplodingLLM:
    """A backend that fails the way a live model does (unset client, CUDA OOM)."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    def generate(self, *args, **kwargs):
        raise self._error


@pytest.mark.parametrize(
    "error",
    [RuntimeError("No language model configured"), LLMError("CUDA out of memory")],
)
def test_learn_pages_never_raise_at_the_ui(settings, vector_store, tutor, error) -> None:
    """P1 regression: a model crash becomes an ok=False page, not a traceback."""
    platform = GeniePlatform(
        settings, tutor=tutor, vector_store=vector_store, llm=ExplodingLLM(error)
    )
    page = platform.study_chapter("Mechanics", "physics")
    assert not page.ok
    assert "could not be generated right now" in page.markdown
    assert page.citations  # the grounding gathered before the crash survives
    topic = platform.study_topic("projectile motion", "physics")
    assert not topic.ok and topic.error
    # a failed page is not counted as studied material
    assert platform.progress.stats()["chapter_pages"] == 0


# -------------------------------------------------------------------- practice
def test_generate_test_returns_answerable_questions(platform: GeniePlatform) -> None:
    test = platform.generate_test(scope=Scope.CHAPTER, subject="physics", target="Mechanics", count=4)
    assert isinstance(test, PracticeSet)
    assert test.count == 4 and test.ok
    assert all(question.is_valid() for question in test.questions)
    assert test.scope == "chapter" and test.chapter == "Mechanics"
    quiz = test.quiz_markdown()
    assert "Correct answer" not in quiz  # the quiz must not spoil the answers
    assert len(test.widget_updates()) == 4


def test_generated_test_count_is_clamped_to_the_widget_budget(platform: GeniePlatform) -> None:
    test = platform.generate_test(scope=Scope.TOPIC, subject="biology", target="heart", count=99)
    assert test.count <= MAX_TEST_QUESTIONS
    empty = platform.generate_test(scope=Scope.CHAPTER, count=0)
    assert empty.count >= 2  # the floor is enforced, never a zero-question test


def test_subject_scope_test_spans_the_subject(platform: GeniePlatform) -> None:
    test = platform.generate_test(scope=Scope.SUBJECT, subject="chemistry", count=3)
    assert test.ok
    assert "full syllabus" in test.title or "Chemistry" in test.title


def test_submit_test_grades_persists_and_reports(platform: GeniePlatform) -> None:
    test = platform.generate_test(scope=Scope.CHAPTER, subject="physics", target="Mechanics", count=5)
    picks = [1, 1, 1, 1, 1]
    result = platform.submit_test(test, picks, duration_seconds=12.5)
    assert result.total == 5
    # the mock keys answers 1,2,3,4,1 so five picks of option 1 score exactly two correct
    assert result.correct_count == 2
    assert result.wrong_count == 3
    assert result.score == pytest.approx(2 * platform.settings.marks_correct + 3 * platform.settings.marks_incorrect)
    record = platform.progress.tests()[0]
    assert record["title"] == test.title
    assert record["duration_seconds"] == 12.5
    assert record["weak_topics"]
    assert platform.progress.stats()["tests"] == 1


def test_submit_accepts_the_stored_title_and_forgets_nothing(platform: GeniePlatform) -> None:
    test = platform.generate_test(scope=Scope.TOPIC, subject="biology", target="osmosis", count=2)
    again = platform.practice.last_set()
    assert again is test
    result = platform.submit_test(test.title, [1, 2])
    assert result.total == 2
    assert platform.submit_test("no such test", [1]).total == 0


def test_weak_topics_feed_the_revision_targets(platform: GeniePlatform) -> None:
    test = platform.generate_test(scope=Scope.CHAPTER, subject="physics", target="Mechanics", count=4)
    platform.submit_test(test, [2, 2, 2, 2])
    areas = platform.practice.weak_topics()
    assert areas and areas[0].wrong >= 1
    assert platform.revision.weakest_topics()


# -------------------------------------------------------------------- revision
@pytest.mark.parametrize("deliverable", list(RevisionDeliverable))
def test_every_revision_deliverable_keeps_its_sections(platform: GeniePlatform, deliverable) -> None:
    doc = platform.revision_sheet(deliverable, subject="physics", target="Mechanics")
    assert doc.ok and doc.kind == deliverable.value
    for section in REVISION_SECTIONS[deliverable]:
        assert section in doc.markdown
    assert doc.citations


def test_revision_sheet_falls_back_to_the_subject_when_target_is_blank(platform: GeniePlatform) -> None:
    doc = platform.revision_sheet(RevisionDeliverable.LAST_MINUTE, subject="chemistry", target="")
    assert doc.ok and "Chemistry" in doc.title


def test_revise_weak_areas_needs_test_history_first(platform: GeniePlatform) -> None:
    doc = platform.revise_weak_areas()
    assert not doc.ok
    assert "No weak areas recorded yet" in doc.markdown

    test = platform.generate_test(scope=Scope.CHAPTER, subject="physics", target="Mechanics", count=3)
    platform.submit_test(test, [2, 2, 2])
    built = platform.revise_weak_areas(RevisionDeliverable.HIGH_YIELD)
    assert built.ok
    assert any("test history" in notice for notice in built.notices)
    assert platform.progress.stats()["revision_sheets"] == 1


# ------------------------------------------------------------------ ask funnel
def test_ask_reuses_the_coach_and_counts_the_doubt(platform: GeniePlatform, tutor: FakeTutor) -> None:
    result = platform.ask("Why does the sky scatter blue light?", subject="physics", mode="quick_revision")
    assert result.answer.startswith("coach answer")
    assert tutor.calls[0]["question"] == "Why does the sky scatter blue light?"
    assert tutor.calls[0]["mode"] is AnswerMode.QUICK_REVISION
    assert platform.progress.stats()["doubts"] == 1


def test_ask_without_a_coach_pipeline_is_an_explicit_error(settings, vector_store) -> None:
    lonely = GeniePlatform(settings, vector_store=vector_store, llm=MockLLM(settings))
    with pytest.raises(RuntimeError):
        lonely.ask("anything")


# ------------------------------------------------------ extension points
def test_photo_ocr_is_available_while_voice_remains_planned(platform: GeniePlatform) -> None:
    voice = platform.capability("voice")
    image = platform.capability("image")
    assert voice.state is CapabilityState.PLANNED
    assert image.available
    with pytest.raises(CapabilityNotAvailable) as raised:
        platform.listen(b"\x00\x01fake wav")
    assert "Voice doubts" in raised.value.user_message()
    assert "VoiceTranscriber.transcribe" in raised.value.user_message()
    with pytest.raises(ValueError, match="empty"):
        platform.extract_photo_question(b"")


def test_registering_a_backend_routes_the_input_through_the_same_funnel(
    platform: GeniePlatform, tutor: FakeTutor
) -> None:
    class Mic:
        def transcribe(self, audio: bytes, *, language: str = "en") -> str:
            return "Explain hydraulic action in rocks"

    class Photo:
        def extract_question(self, image: bytes, *, subject=None) -> str:
            return "Solve this: a body accelerates at 2 m/s^2 for 5 s"

    assert isinstance(Mic(), VoiceTranscriber)
    assert isinstance(Photo(), ImageSolver)

    platform.capabilities.register("voice", Mic())
    platform.capabilities.register("image", Photo())
    assert platform.capability("voice").available

    platform.listen(b"audio")
    platform.solve_photo(b"image", subject="physics")
    assert [call["question"] for call in tutor.calls] == [
        "Explain hydraulic action in rocks",
        "Solve this: a body accelerates at 2 m/s^2 for 5 s",
    ]
    assert platform.progress.stats()["doubts"] == 2


# ------------------------------------------------------------ the student loop
def test_a_student_can_learn_ask_test_see_scores_and_revise(
    platform: GeniePlatform,
) -> None:
    """The five success criteria, in order, on one platform instance."""
    page = platform.study_chapter("Mechanics", subject="physics")  # 1. learn a chapter
    assert page.ok
    assert all(section in page.markdown for section in CHAPTER_SECTIONS)
    assert page.citations  # grounded in the indexed pages, never hardcoded

    platform.ask("Why is the work done by friction negative?", subject="physics")  # 2. ask

    test = platform.generate_test(
        scope=Scope.CHAPTER, subject="physics", target="Mechanics", count=3
    )
    result = platform.submit_test(test, [2, 2, 2])  # 3. take a test
    assert result.total == 3

    stats = platform.progress.stats()  # 4. see the scores
    assert stats["tests"] == 1 and stats["questions"] == 3 and stats["correct"] == 1
    assert stats["score"] == result.score and stats["max_score"] == result.max_score

    areas = platform.progress.weak_areas()
    assert areas and areas[0].name == "Mechanics" and areas[0].accuracy_percent < 75

    sheet = platform.revise_weak_areas(RevisionDeliverable.LAST_MINUTE)  # 5. revise
    assert sheet.ok
    assert all(section in sheet.markdown for section in REVISION_SECTIONS[RevisionDeliverable.LAST_MINUTE])
    assert platform.progress.stats()["revision_sheets"] == 1
