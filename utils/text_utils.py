"""Text helpers: cleaning, truncation, JSON extraction and history rendering.

These functions are intentionally free of LangChain/LLM imports so they can be
unit-tested in isolation and reused by the ingestion CLI.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "clean_text",
    "truncate",
    "strip_code_fences",
    "extract_json_object",
    "format_history",
    "hash_text",
    "clamp",
    "format_seconds",
    "strip_answer_prefix",
    "join_context_blocks",
]

_WHITESPACE_RE = re.compile(r"[ \t\u00a0]+")
_SPACE_BEFORE_NEWLINE_RE = re.compile(r"[ \t]+\n")
_SPACE_AFTER_NEWLINE_RE = re.compile(r"\n[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_ANSWER_PREFIX_RE = re.compile(
    r"^\s*(?:final\s+answer|\*\*answer\*\*|answer|response|solution)\s*[:\-]\s*",
    re.IGNORECASE,
)

# Rough token/character ratio for English + formula heavy NEET material.  Only
# used to pre-trim contexts; the tokenizer enforces the hard limit later.
CHARS_PER_TOKEN: float = 4.0


def clean_text(text: str | None) -> str:
    """Normalise whitespace and strip control characters from model/PDF text."""
    if not text:
        return ""
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("\x00", " ")
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)
    cleaned = _SPACE_BEFORE_NEWLINE_RE.sub("\n", cleaned)
    cleaned = _SPACE_AFTER_NEWLINE_RE.sub("\n", cleaned)
    cleaned = _MULTI_NEWLINE_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def truncate(text: str, limit: int, *, suffix: str = " …", keep_tail: bool = False) -> str:
    """Trim ``text`` to ``limit`` characters, optionally keeping the tail."""
    if limit <= 0 or len(text) <= limit:
        return text
    if keep_tail:
        return suffix.strip() + " " + text[-limit:]
    return text[: max(0, limit - len(suffix))].rstrip() + suffix


def strip_code_fences(text: str) -> str:
    """Remove a surrounding markdown code fence (```json ... ```)."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    first_newline = stripped.find("\n")
    if first_newline == -1:
        return stripped.strip("`").strip()
    body = stripped[first_newline + 1 :]
    if body.rstrip().endswith("```"):
        body = body.rstrip()[:-3]
    return body.strip()


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Return the first balanced JSON object found in ``text``.

    Small open-source models often wrap JSON in prose or code fences, so a
    strict ``json.loads`` is not enough.  We scan for a balanced brace pair
    while tracking string state and escapes, then parse it.
    """
    if not text:
        return None
    candidate = strip_code_fences(text)

    start = candidate.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(candidate)):
            char = candidate[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    chunk = candidate[start : index + 1]
                    try:
                        parsed = json.loads(chunk)
                    except json.JSONDecodeError:
                        break
                    return parsed if isinstance(parsed, dict) else None
        start = candidate.find("{", start + 1)
    return None


def format_history(
    history: Sequence[Mapping[str, Any]] | None,
    *,
    max_turns: int = 4,
    max_chars_per_turn: int = 600,
) -> str:
    """Render recent chat turns as compact ``Student/Tutor`` lines.

    Args:
        history: Gradio-style messages ``{"role": "user"|"assistant", "content": ...}``.
        max_turns: Number of *messages* (not exchanges) to keep.
        max_chars_per_turn: Per-message character cap.
    """
    if not history or max_turns <= 0:
        return "(no previous conversation)"
    tail = [item for item in history if item.get("content")][-max_turns:]
    lines: list[str] = []
    for item in tail:
        role = str(item.get("role", "user")).lower()
        speaker = "Student" if role == "user" else "Tutor"
        content = clean_text(str(item.get("content", ""))).replace("\n", " ")
        lines.append(f"{speaker}: {truncate(content, max_chars_per_turn, suffix='...')}")
    return "\n".join(lines) if lines else "(no previous conversation)"


def hash_text(text: str, *, length: int = 16) -> str:
    """Stable short hash used to build deterministic document/chunk ids."""
    digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
    return digest[:length]


def clamp(value: float, low: float, high: float) -> float:
    """Constrain ``value`` to the ``[low, high]`` interval."""
    return max(low, min(high, value))


def format_seconds(seconds: float) -> str:
    """Human friendly duration rendering (``0.42 s`` / ``1 m 07 s``)."""
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, remainder = divmod(int(seconds), 60)
    return f"{minutes} m {remainder:02d} s"


def strip_answer_prefix(text: str) -> str:
    """Drop artefacts such as ``Answer:`` that some models add on their own."""
    cleaned = strip_code_fences(text)
    cleaned = _ANSWER_PREFIX_RE.sub("", cleaned)
    return cleaned.strip()


def join_context_blocks(blocks: Iterable[str], *, separator: str = "\n\n") -> str:
    """Join already-formatted context blocks, skipping empty ones."""
    return separator.join(block for block in blocks if block.strip())
