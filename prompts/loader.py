"""Prompt templates with upfront validation.

Prompts live in ``prompts/*.txt`` so they can be edited without touching code.
Placeholders use :class:`string.Template` syntax (``$question``) - ``$`` is
rare in the prompt text itself, which avoids the brace-escaping pain of
``str.format``.  :meth:`PromptLibrary.validate` fails fast at startup when a
file or placeholder is missing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Mapping

from config import Settings
from utils.constants import AnswerMode
from utils.logging_utils import get_logger

__all__ = [
    "PromptLibrary",
    "PromptSpec",
    "PROMPT_SPECS",
    "ROUTER_PROMPT",
    "PERSONA_PROMPT",
    "ANSWER_SYSTEM_PROMPT",
    "ANSWER_USER_PROMPT",
    "ANSWER_NO_CONTEXT_PROMPT",
    "MODE_PROMPT_NAMES",
    "PLATFORM_PROMPT_PAIRS",
]

logger = get_logger(__name__)

ROUTER_PROMPT = "router"
PERSONA_PROMPT = "persona"
ANSWER_SYSTEM_PROMPT = "answer_system"
ANSWER_USER_PROMPT = "answer_user"
ANSWER_NO_CONTEXT_PROMPT = "answer_user_no_context"

# Platform (Learn / Practice / Revision Hub) prompt keys.
LEARN_SYSTEM_PROMPT = "learn_chapter_system"
LEARN_USER_PROMPT = "learn_chapter_user"
MCQ_SYSTEM_PROMPT = "mcq_system"
MCQ_USER_PROMPT = "mcq_user"
REVISION_SYSTEM_PROMPT = "revision_system"
REVISION_USER_PROMPT = "revision_user"

#: ``task_id -> (system prompt key, user prompt key)`` for the platform services.
PLATFORM_PROMPT_PAIRS: dict[str, tuple[str, str]] = {
    "learn_chapter": (LEARN_SYSTEM_PROMPT, LEARN_USER_PROMPT),
    "mcq_generation": (MCQ_SYSTEM_PROMPT, MCQ_USER_PROMPT),
    "revision_notes": (REVISION_SYSTEM_PROMPT, REVISION_USER_PROMPT),
}

_PLACEHOLDER_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")


@dataclass(frozen=True, slots=True)
class PromptSpec:
    """Definition of a single prompt file."""

    name: str
    filename: str
    placeholders: frozenset[str]
    description: str


PROMPT_SPECS: Mapping[str, PromptSpec] = {
    spec.name: spec
    for spec in (
        PromptSpec(
            name=ROUTER_PROMPT,
            filename="router.txt",
            placeholders=frozenset({"history", "question", "subject_hint"}),
            description="Subject/intent classifier for the routing agent.",
        ),
        PromptSpec(
            name=PERSONA_PROMPT,
            filename="persona.txt",
            placeholders=frozenset(),
            description="Neet Genie coach persona: 15 years of NCERT faculty habits.",
        ),
        PromptSpec(
            name=ANSWER_SYSTEM_PROMPT,
            filename="answer_system.txt",
            placeholders=frozenset(),
            description="Grounding, citation and style rules for the answer agent.",
        ),
        PromptSpec(
            name=ANSWER_USER_PROMPT,
            filename="answer_user.txt",
            placeholders=frozenset({"history", "context", "question", "subject", "mode"}),
            description="Grounded answer request with retrieved context.",
        ),
        PromptSpec(
            name=ANSWER_NO_CONTEXT_PROMPT,
            filename="answer_user_no_context.txt",
            placeholders=frozenset({"history", "question", "subject", "mode"}),
            description="Fallback request when retrieval returns nothing.",
        ),
        PromptSpec(
            name=AnswerMode.DETAILED.prompt_name,
            filename="mode_detailed.txt",
            placeholders=frozenset(),
            description="Detailed Explanation section contract (five canonical headings).",
        ),
        PromptSpec(
            name=AnswerMode.QUICK_REVISION.prompt_name,
            filename="mode_quick_revision.txt",
            placeholders=frozenset(),
            description="Quick Revision sheet contract (same headings, compressed).",
        ),
        PromptSpec(
            name=AnswerMode.MCQ_PRACTICE.prompt_name,
            filename="mode_mcq_practice.txt",
            placeholders=frozenset({"mcq_count"}),
            description="MCQ Practice drill contract.",
        ),
        PromptSpec(
            name=AnswerMode.PYQ_FOCUS.prompt_name,
            filename="mode_pyq_focus.txt",
            placeholders=frozenset(),
            description="PYQ Focus contract (how the topic is questioned).",
        ),
        PromptSpec(
            name=LEARN_SYSTEM_PROMPT,
            filename="learn_chapter_system.txt",
            placeholders=frozenset({"sections_line"}),
            description="Learn module: four-section chapter study page contract.",
        ),
        PromptSpec(
            name=LEARN_USER_PROMPT,
            filename="learn_chapter_user.txt",
            placeholders=frozenset({"subject", "chapter", "focus", "context"}),
            description="Learn module: grounded chapter request for one indexed chapter.",
        ),
        PromptSpec(
            name=MCQ_SYSTEM_PROMPT,
            filename="mcq_system.txt",
            placeholders=frozenset({"mcq_count"}),
            description="Practice module: machine-readable MCQ output contract.",
        ),
        PromptSpec(
            name=MCQ_USER_PROMPT,
            filename="mcq_user.txt",
            placeholders=frozenset({"subject", "scope", "target", "context", "mcq_count"}),
            description="Practice module: subject / chapter / topic test request.",
        ),
        PromptSpec(
            name=REVISION_SYSTEM_PROMPT,
            filename="revision_system.txt",
            placeholders=frozenset({"deliverable", "sections_line"}),
            description="Revision Hub: dense sheet contract for the chosen deliverable.",
        ),
        PromptSpec(
            name=REVISION_USER_PROMPT,
            filename="revision_user.txt",
            placeholders=frozenset({"subject", "target", "guidance", "context"}),
            description="Revision Hub: grounded request for one revision deliverable.",
        ),
    )
}

#: Prompt key for every answer mode, in UI order.
MODE_PROMPT_NAMES: tuple[str, ...] = tuple(mode.prompt_name for mode in AnswerMode)


class PromptLibrary:
    """Loads, caches and renders the prompt files under ``prompts/``."""

    def __init__(self, prompt_dir: Path | Settings | str) -> None:
        if isinstance(prompt_dir, Settings):
            self.prompt_dir = Path(prompt_dir.prompt_dir)
        else:
            self.prompt_dir = Path(prompt_dir)
        self._cache: dict[str, str] = {}

    # ------------------------------------------------------------------ public
    def path_for(self, name: str) -> Path:
        try:
            spec = PROMPT_SPECS[name]
        except KeyError as exc:
            raise KeyError(f"Unknown prompt '{name}'. Known: {sorted(PROMPT_SPECS)}") from exc
        return self.prompt_dir / spec.filename

    def raw(self, name: str) -> str:
        """Return the prompt file contents (cached per instance)."""
        if name in self._cache:
            return self._cache[name]
        path = self.path_for(name)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file missing: {path}")
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(f"Prompt file is empty: {path}")
        self._cache[name] = text
        return text

    def clear_cache(self) -> None:
        """Forget cached prompt text (useful after hot-editing prompt files)."""
        self._cache.clear()

    def template(self, name: str) -> Template:
        """Return the parsed :class:`Template` for a prompt."""
        return Template(self.raw(name))

    def render(self, name: str, **values: object) -> str:
        """Render a prompt, leaving unknown placeholders untouched."""
        spec = PROMPT_SPECS[name]
        missing = {key for key in spec.placeholders if key not in values}
        if missing:
            raise ValueError(f"Prompt '{name}' is missing values for: {sorted(missing)}")
        return self.template(name).safe_substitute(**values).strip()

    def validate(self) -> list[str]:
        """Return a list of problems; an empty list means the prompts are usable."""
        problems: list[str] = []
        for name, spec in PROMPT_SPECS.items():
            path = self.prompt_dir / spec.filename
            if not path.exists():
                problems.append(f"missing file: {path}")
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except OSError as exc:  # pragma: no cover - permissions
                problems.append(f"unreadable file {path}: {exc}")
                continue
            if not text.strip():
                problems.append(f"empty file: {path}")
                continue
            found = set(_PLACEHOLDER_RE.findall(text))
            for missing in sorted(spec.placeholders - found):
                problems.append(f"{spec.filename}: missing placeholder '${missing}'")
        return problems

    def describe(self) -> dict[str, str]:
        """Map prompt name -> absolute path, for the UI/diagnostics."""
        return {name: str(self.path_for(name)) for name in PROMPT_SPECS}
