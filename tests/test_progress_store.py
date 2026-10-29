"""Progress store tests - JSON persistence and the analytics built on it."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from learning.assessment import MCQuestion, grade
from learning.progress import SCHEMA_VERSION, ProgressStore


def question(number: int, correct: int, topic: str) -> MCQuestion:
    return MCQuestion(
        number=number,
        stem=f"Q{number}",
        options=("a", "b", "c", "d"),
        correct=correct,
        topic=topic,
    )


@pytest.fixture
def store(settings) -> ProgressStore:
    return ProgressStore(settings.progress_dir)


def submit(store: ProgressStore, *, subject="physics", chapter="Mechanics", answers, topics=None) -> None:
    topics = topics or ["Newton's laws"]
    questions = [
        question(index + 1, correct, topics[index % len(topics)])
        for index, correct in enumerate(answers or [1])
    ]
    picks = [1 if correct == 1 else 3 for correct in answers] if answers else []
    result = grade(questions, picks, subject=subject, chapter=chapter)
    store.record(result)


def test_fresh_store_reports_zeroes_and_creates_no_file(store: ProgressStore) -> None:
    stats = store.stats()
    assert stats["tests"] == 0 and stats["questions"] == 0
    assert stats["accuracy"] == 0.0
    assert store.file.name == "default.json"
    assert not store.file.exists()


def test_record_writes_the_document_and_bumps_the_day_bucket(store: ProgressStore) -> None:
    submit(store, answers=[1, 2, 1])
    document = json.loads(store.file.read_text(encoding="utf-8"))
    assert document["schema"] == SCHEMA_VERSION
    assert document["profile"] == "default"
    assert len(document["tests"]) == 1
    today = datetime.now(timezone.utc).date().isoformat()
    assert document["daily"][today]["tests"] == 1
    assert document["daily"][today]["questions"] == 3
    assert document["updated_at"]


def test_scores_and_accuracy_aggregate_over_records(store: ProgressStore) -> None:
    submit(store, answers=[1, 1])  # both correct (key 1, pick 1)
    submit(store, answers=[2, 2])  # both wrong (key 2, pick 3)
    stats = store.stats()
    assert stats["tests"] == 2
    assert stats["questions"] == 4
    assert stats["correct"] == 2
    assert stats["wrong"] == 2
    assert stats["accuracy"] == pytest.approx(0.5)
    assert stats["score"] == pytest.approx(2 * 4.0 + 2 * -1.0)


def test_subject_and_chapter_performance(store: ProgressStore) -> None:
    submit(store, subject="physics", chapter="Mechanics", answers=[1, 2])
    submit(store, subject="biology", chapter="Genetics", answers=[1, 1], topics=["Inheritance"])
    subjects = store.subject_performance()
    assert set(subjects) == {"biology", "physics"}
    assert subjects["physics"].accuracy == pytest.approx(0.5)
    assert subjects["biology"].accuracy_percent == 100
    chapters = store.chapter_performance()
    assert set(chapters) == {"biology::Genetics", "physics::Mechanics"}
    # weakest first, which is what the Progress tab lists
    assert list(chapters)[0] == "physics::Mechanics"


def test_weak_areas_are_ranked_and_thresholded(store: ProgressStore) -> None:
    submit(store, answers=[1, 2, 1, 2], topics=["Rotational motion", "Thermodynamics"])
    weak = store.weak_areas(threshold=0.75)
    # Rotational motion was answered correctly both times; Thermodynamics never was
    assert [area.name for area in weak] == ["Thermodynamics"]
    assert weak[0].accuracy == pytest.approx(0.0)
    assert weak[0].wrong == 2
    assert store.weak_areas(threshold=0.0) == []
    assert weak[0].as_dict()["subject"] == "physics"


def test_events_are_counted_separately_from_tests(store: ProgressStore) -> None:
    assert store.bump("doubts") == 1
    store.bump("doubts")
    store.bump("chapter_pages")
    store.bump("revision_sheets", count=2)
    stats = store.stats()
    assert (stats["doubts"], stats["chapter_pages"], stats["revision_sheets"]) == (2, 1, 2)
    assert stats["tests"] == 0  # events do not invent test history


def test_corrupt_file_is_repaired_instead_of_crashing(store: ProgressStore) -> None:
    store.file.parent.mkdir(parents=True, exist_ok=True)
    store.file.write_text("{ this is not json", encoding="utf-8")
    assert store.stats()["tests"] == 0
    submit(store, answers=[1])
    assert json.loads(store.file.read_text(encoding="utf-8"))["tests"]


def test_unreadable_history_is_quarantined_not_overwritten(store: ProgressStore) -> None:
    """P1 regression: the only copy of a student's history survives corruption."""
    store.file.parent.mkdir(parents=True, exist_ok=True)
    store.file.write_text("{ this is not json", encoding="utf-8")
    assert store.stats()["tests"] == 0
    quarantined = store.last_quarantine
    assert quarantined is not None and quarantined.exists()
    assert ".corrupt-" in quarantined.name
    assert quarantined.read_text(encoding="utf-8") == "{ this is not json"
    submit(store, answers=[1])  # a fresh log starts on the original path
    assert store.stats()["tests"] == 1
    assert quarantined.exists()  # and the kept copy was never touched


def test_json_that_is_not_a_document_is_quarantined(store: ProgressStore) -> None:
    store.file.parent.mkdir(parents=True, exist_ok=True)
    store.file.write_text(json.dumps(["not", "a", "document"]), encoding="utf-8")
    assert store.stats()["tests"] == 0
    assert store.last_quarantine is not None and store.last_quarantine.exists()


def test_mistyped_document_is_repaired_not_fatal(store: ProgressStore) -> None:
    """P1 regression: valid JSON with the wrong shapes must not crash analytics."""
    store.file.parent.mkdir(parents=True, exist_ok=True)
    store.file.write_text(
        json.dumps(
            {
                "schema": SCHEMA_VERSION,
                "profile": "default",
                "tests": "all history gone",
                "daily": ["2026-01-01"],
                "events": {"doubts": "many", "chapter_pages": None},
            }
        ),
        encoding="utf-8",
    )
    assert store.stats()["tests"] == 0
    assert store.subject_performance() == {}
    assert store.chapter_performance() == {}
    assert store.weak_areas() == []
    assert store.recent_activity() == []
    assert store.streak() == 0
    assert len(store.daily_series(days=3)) == 3
    submit(store, answers=[1])  # record_dict no longer AttributeErrors
    assert store.stats()["tests"] == 1


def test_mistyped_records_survive_reading_and_weak_areas(store: ProgressStore) -> None:
    store.file.parent.mkdir(parents=True, exist_ok=True)
    store.file.write_text(
        json.dumps(
            {
                "schema": SCHEMA_VERSION,
                "profile": "default",
                "tests": [
                    "a line, not a record",
                    {
                        "title": "Mechanics - Chapter Test",
                        "subject": "physics",
                        "chapter": "Mechanics",
                        "created_at": "2026-01-05T10:00:00+00:00",
                        "total": "x",  # not a number
                        "correct": 0,
                        "wrong": None,
                        "max_score": "abc",
                        "topics": [
                            "not a dict",
                            {"topic": "Thermodynamics", "total": 2, "correct": "0", "wrong": 2},
                        ],
                    },
                ],
                "daily": {"2026-01-05": "broken bucket", "2026-01-06": {"tests": "2"}},
                "events": {"doubts": "lots"},
            }
        ),
        encoding="utf-8",
    )
    stats = store.stats()
    assert stats["tests"] == 1  # the string record was dropped, not fatal
    assert stats["questions"] == 0  # "x" repaired to 0 instead of raising ValueError
    assert stats["doubts"] == 0  # "lots" repaired to 0
    assert stats["days_active"] == 1  # only the well-shaped day bucket survived
    weak = store.weak_areas(threshold=0.75)
    assert [area.name for area in weak] == ["Thermodynamics"]
    assert weak[0].wrong == 2
    submit(store, answers=[1])
    assert store.stats()["tests"] == 2


def test_no_temporary_files_are_left_behind(store: ProgressStore) -> None:
    for index in range(5):
        submit(store, answers=[1, 2], topics=[f"Topic {index}"])
    leftovers = list(store.file.parent.glob("*.tmp"))
    assert leftovers == []
    assert len(store.tests()) == 5


def test_clear_empties_history_but_keeps_the_file(store: ProgressStore) -> None:
    submit(store, answers=[1])
    store.clear()
    assert store.file.exists()
    assert store.stats()["tests"] == 0
    assert store.recent_activity() == []


def test_limit_and_subject_filters_on_test_records(store: ProgressStore) -> None:
    for index in range(4):
        submit(store, subject="physics" if index % 2 else "chemistry", answers=[1], topics=[f"T{index}"])
    assert len(store.tests(limit=2)) == 2
    assert {item["subject"] for item in store.tests(subject="chemistry")} == {"chemistry"}
    assert store.tests(limit=0) == []


def test_daily_series_covers_the_window_including_gaps(store: ProgressStore) -> None:
    series = store.daily_series(days=7)
    assert len(series) == 7
    assert all(set(day) == {"date", "tests", "questions", "correct"} for day in series)
    submit(store, answers=[1, 1])
    todays = [day for day in store.daily_series(days=3) if day["questions"]]
    assert todays[-1]["correct"] == 2


def test_profiles_keep_separate_documents(settings) -> None:
    default = ProgressStore(settings.progress_dir)
    other = ProgressStore(settings.progress_dir, profile="Ananya")
    submit(default, answers=[1])
    assert default.stats()["tests"] == 1
    assert other.stats()["tests"] == 0
    # the profile becomes a safe file name
    assert other.profile == "ananya" and other.file.name == "ananya.json"


def test_as_dict_exposes_every_progress_view(store: ProgressStore) -> None:
    submit(store, answers=[1, 2])
    payload = store.as_dict()
    assert set(payload) == {"path", "stats", "subjects", "chapters", "weak_areas", "recent"}
    assert payload["stats"]["questions"] == 2
    assert payload["weak_areas"][0]["accuracy"] == pytest.approx(0.5)
