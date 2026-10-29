"""MCQ parsing, grading and the analytics a graded test produces."""

from __future__ import annotations

import pytest

from learning.assessment import (
    GradedTest,
    MCQuestion,
    grade,
    merge_answer_key,
    parse_questions,
)

STRICT = """
Q1. Which law gives the relation F = m * a?
(1) Newton's first law
(2) Newton's second law
(3) Newton's third law
(4) Law of gravitation
ANSWER: 2
EXPLANATION: The second law states force as the rate of change of momentum.
TOPIC: Newton's laws

Q3. The SI unit of magnetic flux is:
(1) tesla
(2) weber
(3) ampere
(4) volt
ANSWER: (2)
EXPLANATION: Flux is measured in weber; field strength in tesla.
TOPIC: Magnetism
"""

NUMBERED = """
Here are the questions.

1. Photosynthesis takes place mainly in the:
   (a) mitochondria
   (b) chloroplast
   (c) ribosome
   (d) nucleus
   Answer: b
   Explanation: Light reactions happen on the thylakoid membranes.
   Topic: Plant physiology

2. Which statement about the Calvin cycle is correct?
   (a) It needs light directly
   (b) It occurs in the stroma
   (c) It produces water
   (d) It releases oxygen
   Answer: B
   Explanation: It runs in the stroma using ATP and NADPH.
   Topic: Plant physiology
"""


def one(number: int, correct: int, topic: str) -> MCQuestion:
    return MCQuestion(
        number=number,
        stem=f"Question {number}",
        options=("first", "second", "third", "fourth"),
        correct=correct,
        topic=topic,
    )


def test_parse_reads_the_strict_contract() -> None:
    questions = parse_questions(STRICT)
    # numbering is closed up (the model skipped Q2) so the answer widgets line up
    assert [question.number for question in questions] == [1, 2]
    assert [question.topic for question in questions] == ["Newton's laws", "Magnetism"]
    first = questions[0]
    assert first.correct == 2
    assert len(first.options) == 4
    assert "momentum" in first.explanation
    assert all(question.is_valid() for question in questions)


def test_parse_tolerates_letters_and_hole_in_numbering() -> None:
    questions = parse_questions(NUMBERED)
    assert len(questions) == 2
    assert questions[0].correct == 2  # '(b)' -> option 2
    assert questions[1].correct == 2  # 'B'  -> option 2
    assert "thylakoid" in questions[0].explanation
    assert questions[0].topic == "Plant physiology"


def test_parse_ignores_prose_and_returns_nothing_useless() -> None:
    assert parse_questions("") == []
    assert parse_questions("Just a paragraph with no question structure at all.") == []
    only_stems = parse_questions("Q1. What is the unit of force?")
    assert only_stems == [] or not only_stems[0].is_valid()


def test_render_withholds_the_answer_until_it_is_revealed() -> None:
    question = one(1, 2, "Work")
    hidden = question.render(reveal=False)
    revealed = question.render(reveal=True)
    assert "Correct answer" not in hidden
    assert "**Q1.**" in hidden and "(2)" in hidden
    assert "Correct answer:** (2)" in revealed


def test_grading_uses_neet_positive_and_negative_marks() -> None:
    questions = [one(1, 1, "Laws"), one(2, 2, "Laws"), one(3, 3, "Units"), one(4, 4, "Units")]
    result = grade(questions, [1, 2, 1, None], marks_correct=4.0, marks_incorrect=-1.0)
    assert result.total == 4
    assert result.correct_count == 2
    assert result.wrong_count == 1
    assert result.skipped_count == 1
    assert result.score == pytest.approx(2 * 4.0 + 1 * -1.0)
    assert result.max_score == pytest.approx(16.0)
    # accuracy counts attempted answers only, so the skipped question is not a miss
    assert result.accuracy == pytest.approx(2 / 3, abs=1e-3)


def test_weak_topics_rank_worst_accuracy_first() -> None:
    questions = [
        one(1, 1, "Rotational motion"),
        one(2, 1, "Rotational motion"),
        one(3, 1, "Thermodynamics"),
        one(4, 1, "Thermodynamics"),
    ]
    result = grade(questions, [1, 2, 1, 1])
    breakdown = result.topic_breakdown()
    assert breakdown["Rotational motion"]["wrong"] == 1
    assert breakdown["Thermodynamics"]["correct"] == 2
    assert result.weak_topics(threshold=0.75) == ["Rotational motion"]
    assert list(breakdown)[0] == "Rotational motion"


def test_answer_key_block_is_merged_into_parsed_questions() -> None:
    text = """
Q1. Which is the strongest intermolecular force in water?
(1) Dipole-dipole
(2) Hydrogen bonding
(3) Van der Waals
(4) Ionic
Q2. Boiling point of water is higher than H2S because of:
(1) Mass
(2) Hydrogen bonding
(3) Shape
(4) Density

Answer Key
1: (2) - hydrogen bonding is the textbook reason
2: (B)
"""
    questions = parse_questions(text)
    # the key lines are answers, not extra questions
    assert [question.number for question in questions] == [1, 2]
    assert [question.correct for question in questions] == [2, 2]
    assert "hydrogen bonding" in questions[0].explanation.lower()
    # parsing is idempotent: re-merging the same key changes nothing
    merged = merge_answer_key(questions, text)
    assert [question.correct for question in merged] == [2, 2]
    assert merged[0].explanation == questions[0].explanation


def test_numbered_options_without_markers_are_not_new_questions() -> None:
    text = """
Q1. The SI unit of electric charge is:
1. ampere
2. coulomb
3. volt
4. ohm
Answer: (2)
Explanation: Charge is measured in coulombs; ampere is current.

Q2. Which quantity opposes the flow of current?
1. ampere
2. coulomb
3. volt
4. resistance
Answer: (4)
"""
    questions = parse_questions(text)
    assert len(questions) == 2
    assert [len(question.options) for question in questions] == [4, 4]
    assert questions[0].options == ("ampere", "coulomb", "volt", "ohm")
    assert questions[1].stem.lower().startswith("which quantity")
    assert questions[0].correct == 2
    assert "coulombs" in questions[0].explanation


def test_record_is_json_friendly_and_self_describing() -> None:
    result = grade([one(1, 2, "Optics")], [1], subject="physics", chapter="Optics")
    record = result.as_record()
    assert record["subject"] == "physics"
    assert record["chapter"] == "Optics"
    assert record["topics"][0]["topic"] == "Optics"
    assert record["answers"][0]["ok"] is False
    assert {"test_id", "score", "accuracy", "weak_topics", "created_at"} <= set(record)
    assert record["total"] == 1


def test_report_markdown_shows_score_accuracy_and_explanations() -> None:
    result = grade([one(1, 2, "Work")], [2], title="Chapter test")
    report = result.report_markdown(reveal=True)
    assert "**Score:** 4 / 4" in report
    assert "**Accuracy:** 100%" in report
    assert "**Skipped:** 0" in report
    assert "Correct answer:** (2)" in report  # revealed after submission
    assert result.summary_line.endswith("score 4/4")


def test_empty_grade_is_reported_not_raised() -> None:
    result = GradedTest(test_id="x", title="Empty", scope="chapter", subject="general")
    assert result.total == 0
    assert result.accuracy == 0.0
    assert result.score == 0.0
    assert result.topic_breakdown() == {}
