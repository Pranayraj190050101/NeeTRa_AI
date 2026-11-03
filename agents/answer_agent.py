"""Answer Generation Agent.

Builds the coach prompt (Neet Genie persona + grounding rules + the selected
answer-mode contract + retrieved context + question), calls the local LLM -
optionally streaming tokens to the UI - and reconciles the output with the
section contract the mode promises.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from agents.base import AgentError, BaseAgent, add_notice, emit_event, token_callback
from llm.base import ChatMessage, LLMError
from prompts.loader import (
    ANSWER_NO_CONTEXT_PROMPT,
    ANSWER_SYSTEM_PROMPT,
    ANSWER_USER_PROMPT,
    PERSONA_PROMPT,
)
from utils.answer_format import enforce_format
from utils.constants import AnswerMode
from utils.text_utils import clean_text, format_history, strip_answer_prefix

__all__ = ["AnswerAgent"]

_SOURCES_TAIL_RE = re.compile(
    r"\n[ \t]*(?:\*\*)?(?:sources?|references?|citations?)(?:\*\*)?[ \t]*:[ \t]*\n.*\Z",
    re.IGNORECASE | re.DOTALL,
)
_SENTENCE_END_RE = re.compile(r"[.!?]\s")


class AnswerAgent(BaseAgent):
    """Generates the final, classroom-style explanation."""

    name = "answer"

    def run(self, state: dict[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
        question = clean_text(state.get("question"))
        subject = str(state.get("subject") or "general")
        context = clean_text(state.get("context"))
        mode = AnswerMode.coerce(state.get("mode"), default=self.settings.answer_mode)
        notices = list(state.get("notices") or [])

        if not context and not self.settings.allow_general_fallback:
            add_notice(
                notices,
                "General-knowledge fallback is disabled (NEET_ALLOW_GENERAL_FALLBACK=false).",
            )
            return {
                "answer": (
                    "I could not find this topic in your indexed NEET material, and general-knowledge "
                    "answers are disabled. Please upload the relevant chapter and re-run ingestion."
                ),
                "mode": mode.value,
                "notices": notices,
            }

        prompt_name = ANSWER_USER_PROMPT if context else ANSWER_NO_CONTEXT_PROMPT
        user_prompt = self.prompts.render(
            prompt_name,
            subject=subject,
            mode=mode.label,
            history=format_history(
                state.get("history"),
                max_turns=self.settings.history_turns,
                max_chars_per_turn=500,
            ),
            context=context,
            question=question,
        )
        messages: list[ChatMessage] = [
            {"role": "system", "content": self.system_prompt(mode)},
            {"role": "user", "content": user_prompt},
        ]

        emit_event(
            config,
            self.name,
            f"{mode.emoji} {mode.label} - "
            + ("coaching from the retrieved passages" if context else "coaching from NEET knowledge"),
        )

        on_token = token_callback(config) if self.settings.stream_tokens else None
        max_new_tokens = max(64, int(round(self.settings.llm_max_new_tokens * mode.token_factor)))
        try:
            raw = self.require_llm().generate(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=self.settings.llm_temperature,
                on_token=on_token,
            )
        except LLMError as exc:
            self.logger.error("Answer generation failed: %s", exc)
            raise AgentError(str(exc)) from exc

        answer, notices = self._postprocess(raw, mode=mode, subject=subject, context=context, notices=notices)
        if not answer:
            answer = (
                "Genie did not receive a complete answer. Please rephrase your doubt or try again."
            )
            add_notice(notices, "The language model produced an empty answer.")

        update: dict[str, Any] = {"answer": answer, "mode": mode.value}
        if notices:
            update["notices"] = notices
        return update

    # ------------------------------------------------------------ prompt build
    def system_prompt(self, mode: AnswerMode) -> str:
        """Persona + grounding rules + the mode's section contract.

        The ``MODE_ID:`` line is machine readable: it lets the offline mock
        backend and the logs identify which contract was requested without
        parsing prose.
        """
        blocks: list[str] = []
        if self.settings.persona_enabled:
            blocks.append(self.prompts.raw(PERSONA_PROMPT))
        blocks.append(self.prompts.raw(ANSWER_SYSTEM_PROMPT))
        mode_block = self.prompts.render(mode.prompt_name, mcq_count=self.settings.mcq_count)
        blocks.append(f"MODE_ID: {mode.value}\n\n{mode_block}")
        return "\n\n---\n\n".join(blocks)

    # ------------------------------------------------------------- postprocess
    def _postprocess(
        self,
        raw: str,
        *,
        mode: AnswerMode,
        subject: str,
        context: str,
        notices: list[str],
    ) -> tuple[str, list[str]]:
        """Normalise the output, then guarantee the mode's section contract."""
        text = strip_answer_prefix(raw)

        # Some models append their own "Sources:" list; the UI shows real ones.
        if len(text) > 600:
            text = _SOURCES_TAIL_RE.sub("", text)

        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        text = text.replace("<br>", "\n").replace("&nbsp;", " ")

        if self.settings.enforce_answer_format:
            repair = enforce_format(text, mode, subject=subject, context=context)
            if repair.changed:
                text = repair.text
                add_notice(notices, f"Coach format applied: {repair.summary()}.")
                self.logger.info("Format repair for mode=%s: %s", mode.value, repair.summary())

        return self._enforce_length(text, mode), notices

    def _enforce_length(self, text: str, mode: AnswerMode | None = None) -> str:
        """Trim runaway answers at a sentence boundary (mode scales the cap)."""
        factor = mode.char_factor if mode else 1.0
        limit = max(500, int(round(self.settings.max_answer_chars * factor)))
        if len(text) <= limit:
            return text
        window = text[:limit]
        matches = list(_SENTENCE_END_RE.finditer(window))
        cut = matches[-1].end() if matches and matches[-1].end() > limit * 0.6 else limit
        self.logger.info("Answer truncated from %d to %d characters", len(text), cut)
        return f"{window[:cut].rstrip()}\n\n_(answer truncated)_"
