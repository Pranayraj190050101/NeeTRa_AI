"""Tests for the text utilities (JSON extraction, history, truncation)."""

from __future__ import annotations

from utils.text_utils import (
    clean_text,
    extract_json_object,
    format_history,
    strip_answer_prefix,
    truncate,
)


def test_extract_json_object_from_code_fence() -> None:
    raw = '```json\n{"subject": "physics", "intent": "doubt", "confidence": 0.9}\n```'
    parsed = extract_json_object(raw)
    assert parsed is not None
    assert parsed["subject"] == "physics"


def test_extract_json_object_ignores_surrounding_prose() -> None:
    raw = 'Here is my answer:\n{"subject": "biology", "nested": {"a": 1}}\nHope it helps!'
    parsed = extract_json_object(raw)
    assert parsed == {"subject": "biology", "nested": {"a": 1}}


def test_extract_json_object_handles_braces_inside_strings() -> None:
    raw = '{"reasoning": "uses {curly} braces", "subject": "chemistry"}'
    parsed = extract_json_object(raw)
    assert parsed is not None
    assert parsed["subject"] == "chemistry"


def test_extract_json_object_returns_none_for_invalid_payload() -> None:
    assert extract_json_object("no json here") is None
    assert extract_json_object("{broken: true}") is None


def test_format_history_keeps_recent_turns_only() -> None:
    history = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "second answer"},
    ]
    rendered = format_history(history, max_turns=2)
    assert "first question" not in rendered
    assert "Student: second question" in rendered
    assert "Tutor: second answer" in rendered


def test_format_history_handles_empty_input() -> None:
    assert format_history(None) == "(no previous conversation)"
    assert format_history([]) == "(no previous conversation)"


def test_clean_text_collapses_whitespace() -> None:
    assert clean_text("  a \t b \n\n\n c  ") == "a b\n\nc"
    assert clean_text(None) == ""


def test_truncate_respects_limit() -> None:
    text = "x" * 100
    assert len(truncate(text, 20)) <= 20
    assert truncate(text, 200) == text


def test_strip_answer_prefix_removes_artifacts() -> None:
    assert strip_answer_prefix("Answer: The unit is the farad.") == "The unit is the farad."
    assert strip_answer_prefix("```\nThe unit is the farad.\n```") == "The unit is the farad."
