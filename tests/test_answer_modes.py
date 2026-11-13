"""Unit tests for the coach persona contract: answer modes and section repair.

These are pure-python tests (no LangChain, no FAISS) that pin the behaviour the
"Neet Genie" requirements promise the student:

* the four answer modes and their friendly spellings,
* the exact five-section format,
* the mode/intent bridge the graph uses for routing.
"""

from __future__ import annotations

import pytest

from utils.answer_format import (
    EXPLANATION_SECTIONS,
    MCQ_SECTIONS,
    PYQ_SECTIONS,
    enforce_format,
    format_contract,
    missing_sections,
    present_sections,
    sections_for,
)
from utils.constants import AnswerMode, Intent

#: The section list quoted verbatim from the product requirement.
REQUIRED_FORMAT = [
    "\U0001f4d8 Concept Explanation",
    "\U0001f3af NEET Importance",
    "\U0001f9e0 Key Points To Remember",
    "\u26a0 Common Student Mistakes",
    "\u2705 Quick Revision Notes",
]


# --------------------------------------------------------------------------- modes
def test_answer_mode_covers_the_four_required_modes() -> None:
    assert [mode.label for mode in AnswerMode] == [
        "Detailed Explanation",
        "Quick Revision",
        "MCQ Practice",
        "PYQ Focus",
    ]
    assert AnswerMode.ui_choices() == [
        ("Detailed Explanation", "detailed"),
        ("Quick Revision", "quick_revision"),
        ("MCQ Practice", "mcq_practice"),
        ("PYQ Focus", "pyq_focus"),
    ]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("detailed", AnswerMode.DETAILED),
        ("Quick Revision", AnswerMode.QUICK_REVISION),
        ("quick-revision", AnswerMode.QUICK_REVISION),
        ("mcq", AnswerMode.MCQ_PRACTICE),
        ("MCQ Practice", AnswerMode.MCQ_PRACTICE),
        ("PYQ", AnswerMode.PYQ_FOCUS),
        ("revision", AnswerMode.QUICK_REVISION),
        (AnswerMode.PYQ_FOCUS, AnswerMode.PYQ_FOCUS),
    ],
)
def test_modes_are_coerced_from_loose_values(raw: object, expected: AnswerMode) -> None:
    assert AnswerMode.coerce(raw) is expected


def test_mode_coercion_falls_back_to_detailed() -> None:
    assert AnswerMode.coerce(None) is AnswerMode.DETAILED
    assert AnswerMode.coerce("telepathy") is AnswerMode.DETAILED
    assert AnswerMode.coerce("telepathy", default=AnswerMode.QUICK_REVISION) is AnswerMode.QUICK_REVISION


def test_each_mode_declares_its_own_prompt_and_budgets() -> None:
    for mode in AnswerMode:
        assert mode.prompt_name == f"mode_{mode.value}"
        assert mode.emoji
        assert mode.description
        assert mode.retrieval_factor > 0
        assert mode.token_factor > 0
        assert mode.char_factor > 0
    # Revision is terser than a detailed lecture, practice sets are longer.
    assert AnswerMode.QUICK_REVISION.char_factor < AnswerMode.DETAILED.char_factor
    assert AnswerMode.MCQ_PRACTICE.char_factor >= AnswerMode.DETAILED.char_factor


def test_mode_intent_bridge_keeps_unserved_intents_deferred() -> None:
    assert AnswerMode.MCQ_PRACTICE.satisfies(Intent.MCQ)
    assert AnswerMode.PYQ_FOCUS.satisfies(Intent.PYQ)
    assert not AnswerMode.DETAILED.satisfies(Intent.MCQ)
    assert not AnswerMode.QUICK_REVISION.satisfies(Intent.PYQ)
    assert not any(mode.satisfies(Intent.STUDY_PLAN) for mode in AnswerMode)
    assert AnswerMode.for_intent(Intent.MCQ) is AnswerMode.MCQ_PRACTICE
    assert AnswerMode.for_intent(Intent.WEAKNESS) is None


# ------------------------------------------------------------------------ sections
def test_detailed_mode_uses_the_required_five_section_format() -> None:
    assert list(EXPLANATION_SECTIONS) == REQUIRED_FORMAT
    assert format_contract(AnswerMode.DETAILED) == " | ".join(REQUIRED_FORMAT)
    assert sections_for(AnswerMode.QUICK_REVISION) == EXPLANATION_SECTIONS


def test_practice_modes_have_their_own_contracts() -> None:
    assert sections_for(AnswerMode.MCQ_PRACTICE) == MCQ_SECTIONS
    assert sections_for(AnswerMode.PYQ_FOCUS) == PYQ_SECTIONS
    assert "\U0001f9e9 Practice MCQs" in format_contract("mcq_practice")
    assert "\U0001f4dc How NEET Asks This" in PYQ_SECTIONS[0]


def test_headings_are_relabelled_without_touching_the_body() -> None:
    text = "### Concept Explanation\n\nThe chloroplast hosts photosynthesis.\n\n**Key Points**\n- ATP and NADPH"
    present = present_sections(text, AnswerMode.DETAILED)
    assert "\U0001f9e0 Key Points To Remember" in present

    repaired = enforce_format(text, AnswerMode.DETAILED, subject="biology")
    assert "### \U0001f4d8 Concept Explanation" in repaired.text
    assert "### \U0001f9e0 Key Points To Remember" in repaired.text
    assert "The chloroplast hosts photosynthesis." in repaired.text
    assert repaired.relabelled == 2
    assert repaired.summary().startswith("standardised 2 heading(s)")


def test_missing_sections_are_appended_in_canonical_order() -> None:
    text = "### \U0001f4d8 Concept Explanation\n\nDiffusion is the net movement of particles."
    assert missing_sections(text, AnswerMode.DETAILED) == [
        "\U0001f3af NEET Importance",
        "\U0001f9e0 Key Points To Remember",
        "\u26a0 Common Student Mistakes",
        "\u2705 Quick Revision Notes",
    ]

    repaired = enforce_format(text, AnswerMode.DETAILED, subject="biology")
    assert repaired.added == missing_sections(text, AnswerMode.DETAILED)
    order = [title for title in REQUIRED_FORMAT if title in repaired.text]
    assert order == REQUIRED_FORMAT, "sections must keep the promised order"


def test_repair_deduplicates_and_reorders_repeated_sections() -> None:
    text = (
        "### Quick Revision\n\n- One photoelectric-effect recall point.\n\n"
        "### Concept Explanation\n\nThe photoelectric effect ejects electrons.\n\n"
        "### Quick Revision Notes\n\n- One photoelectric-effect recall point.\n\n"
        "### NEET Importance\n\n- Recall the threshold frequency."
    )

    repaired = enforce_format(text, AnswerMode.QUICK_REVISION)
    headings = [line for line in repaired.text.splitlines() if line.startswith("### ")]

    assert headings == [f"### {title}" for title in REQUIRED_FORMAT]
    assert repaired.text.count("One photoelectric-effect recall point.") == 1
    assert repaired.reordered
    assert repaired.deduplicated >= 1
    assert repaired.changed


def test_missing_revision_fallback_stays_on_topic_and_avoids_unsourced_claims() -> None:
    repaired = enforce_format("", AnswerMode.QUICK_REVISION, subject="physics")

    assert "potential energy" not in repaired.text.lower()
    assert "capacitance" not in repaired.text.lower()
    assert "question-frequency claim" in repaired.text


def test_missing_revision_section_reuses_the_answered_topic() -> None:
    answer = (
        "### Concept Explanation\n\n"
        "Threshold frequency is the minimum frequency that causes photoelectric emission.\n\n"
        "### Key Points\n\n"
        "- Below threshold, increasing intensity does not cause emission."
    )

    repaired = enforce_format(answer, AnswerMode.QUICK_REVISION, subject="physics")
    revision = repaired.text.split("### ✅ Quick Revision Notes\n\n", 1)[1]

    assert "Threshold frequency is the minimum frequency" in revision
    assert "potential energy" not in revision.lower()
    assert "capacitance" not in revision.lower()


def test_repair_uses_retrieved_context_instead_of_inventing_facts() -> None:
    context = (
        "[1] mole_concept.pdf | page 7 | subject: chemistry\n"
        "The mole is the amount of substance containing 6.022 x 10^23 entities. "
        "Molar mass is the mass of one mole of a substance expressed in grams per mole."
    )
    repaired = enforce_format(
        "### \U0001f4d8 Concept Explanation\n\nA mole counts entities.",
        AnswerMode.DETAILED,
        subject="chemistry",
        context=context,
    )
    assert "Molar mass is the mass of one mole" in repaired.text
    assert "6.022 x 10^23" in repaired.text
    for forbidden in ("2019", "2020", "weightage of", "NEET 2015"):
        assert forbidden not in repaired.text, f"repair must not fabricate {forbidden!r}"


def test_empty_answer_still_produces_a_complete_contract() -> None:
    repaired = enforce_format("", AnswerMode.MCQ_PRACTICE, subject="physics")
    assert all(title in repaired.text for title in MCQ_SECTIONS)
    assert len(repaired.added) == len(MCQ_SECTIONS)


def test_prose_that_happens_to_mention_mistakes_is_not_treated_as_a_heading() -> None:
    text = (
        "### \U0001f4d8 Concept Explanation\n\n"
        "Students often repeat common mistakes in the lab, so read the procedure twice.\n\n"
        "### \U0001f3af NEET Importance\n\n"
        "Statement based items are asked from this line.\n"
    )
    complete = "\n\n".join(
        [
            text.rstrip(),
            "### \U0001f9e0 Key Points To Remember\n\n- Definition, condition, exception.",
            "### \u26a0 Common Student Mistakes\n\n- Skipping units.",
            "### \u2705 Quick Revision Notes\n\n- Recall in one line.",
        ]
    )
    assert not enforce_format(complete, AnswerMode.DETAILED).changed
    assert enforce_format(text, AnswerMode.DETAILED).added == [
        "\U0001f9e0 Key Points To Remember",
        "\u26a0 Common Student Mistakes",
        "\u2705 Quick Revision Notes",
    ]
