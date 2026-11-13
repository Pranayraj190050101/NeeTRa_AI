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
from utils.answer_format import enforce_format, sections_for
from utils.constants import AnswerMode
from utils.text_utils import clean_text, format_history, strip_answer_prefix

__all__ = ["AnswerAgent"]

_SOURCES_TAIL_RE = re.compile(
    r"\n[ \t]*(?:\*\*)?(?:sources?|references?|citations?)(?:\*\*)?[ \t]*:[ \t]*\n.*\Z",
    re.IGNORECASE | re.DOTALL,
)
_SENTENCE_END_RE = re.compile(r"[.!?]\s")
_CONTEXT_REFERENCE_RE = re.compile(r"^\[(\d+)\][^\n]*$", re.MULTILINE)
_INLINE_REFERENCE_RE = re.compile(r"(?<!\w)\[(\d+)\](?!\d)")
_QUESTION_WORD_RE = re.compile(r"[a-z0-9]{4,}", re.IGNORECASE)
_QUESTION_STOPWORDS = {
    "about",
    "explain",
    "give",
    "how",
    "into",
    "neet",
    "please",
    "that",
    "this",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
}
_REPEATED_EQUATION_RE = re.compile(
    r"(?P<expression>[A-Za-z][A-Za-z0-9_{}^]*\s*=\s*[^=.;\n$]+?)(?:\s*)(?P=expression)"
)
_REPEATED_SYMBOL_RE = re.compile(
    r"(?<![\w$])(?P<symbol>[^\W\d_][\w]*(?:_\{?[\w]+\}?)?)(?P=symbol)(?!\w)",
    re.UNICODE,
)
_MALFORMED_SUBSCRIPT_EQUATION_RE = re.compile(
    r"(?P<symbol>[A-Za-z])\\?_\{\$(?P<subscript>[A-Za-z][A-Za-z0-9]*)\}\s*"
    r"(?P<expression>=\s*[^$\n]+?)\$"
)
_BARE_EQUATION_RE = re.compile(
    r"(?<![$\w\\{}])(?P<equation>[\w\\{}^]+\s*=\s*[\w\\{}^]+"
    r"(?:\s*[+\-*/]\s*[\w\\{}^]+)*)(?![=$\w])"
)
_MATH_EXPRESSION_RE = re.compile(r"\$\$.*?\$\$|\$[^$\n]+\$")
_UNSUPPORTED_FREQUENCY_CLAIM_RE = re.compile(
    r"\bNEET\s+(?:frequently|often|commonly|regularly)\s+(?:tests?|asks?)\s+",
    re.IGNORECASE,
)
_UNSUPPORTED_QUESTION_PATTERN_RE = re.compile(
    r"\bquestions?\s+(?:often|frequently|commonly)\s+ask\s+(?:you|students?)\s+to\s+",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[\w]+(?:['’-][\w]+)*", re.UNICODE)
_MODE_WORD_BUDGETS: dict[AnswerMode, tuple[int, ...]] = {
    AnswerMode.DETAILED: (130, 35, 55, 45, 35),
    AnswerMode.QUICK_REVISION: (38, 20, 30, 18, 22),
}


def _normalise_answer_math(text: str) -> str:
    """Repair common model-generated math duplication and malformed delimiters."""

    def repair_malformed_subscript(match: re.Match[str]) -> str:
        return (
            f"${match.group('symbol')}_{{{match.group('subscript')}}}"
            f"{match.group('expression')}$"
        )

    text = _MALFORMED_SUBSCRIPT_EQUATION_RE.sub(repair_malformed_subscript, text)
    text = re.sub(r"\\\\(?=[A-Za-z])", r"\\", text)
    text = re.sub(r"\\_", "_", text)
    text = _REPEATED_SYMBOL_RE.sub(r"\g<symbol>", text)
    text = _REPEATED_EQUATION_RE.sub(r"\g<expression>", text)
    return _BARE_EQUATION_RE.sub(lambda match: f"${match.group('equation').strip()}$", text)


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _trim_line(line: str, remaining_words: int) -> str:
    """Shorten an over-budget prose line without cutting through math."""
    if remaining_words <= 0:
        return ""
    if "$" in line:
        return line if _word_count(line) <= remaining_words + 8 else ""

    sentence_endings = list(re.finditer(r".*?[.!?](?=\s|$)", line))
    complete_sentences: list[str] = []
    for ending in sentence_endings:
        sentence = ending.group(0).strip()
        if _word_count(sentence) > remaining_words:
            break
        complete_sentences.append(sentence)
    if complete_sentences:
        return " ".join(complete_sentences)

    words = list(_WORD_RE.finditer(line))
    if len(words) <= remaining_words:
        return line
    prefix = line[: words[remaining_words - 1].end()].rstrip(" ,;:-")
    return f"{prefix}…"


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

        answer, notices = self._postprocess(
            raw,
            mode=mode,
            subject=subject,
            context=context,
            question=question,
            notices=notices,
        )
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
        question: str = "",
    ) -> tuple[str, list[str]]:
        """Normalise the output, then guarantee the mode's section contract."""
        text = strip_answer_prefix(raw)

        # Some models append their own "Sources:" list; the UI shows real ones.
        if len(text) > 600:
            text = _SOURCES_TAIL_RE.sub("", text)

        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        text = text.replace("<br>", "\n").replace("&nbsp;", " ")
        text = _normalise_answer_math(text)
        relevant_context = self._relevant_context(context, question)
        if not re.search(
            r"\b(?:NEET|exam|question)\b.{0,100}\b(?:frequently|often|year|weightage|marks)\b",
            relevant_context,
            re.IGNORECASE | re.DOTALL,
        ):
            text = _UNSUPPORTED_FREQUENCY_CLAIM_RE.sub("For this topic, focus on ", text)
            text = _UNSUPPORTED_QUESTION_PATTERN_RE.sub("Practise how to ", text)
        text = self._keep_supplied_citations(text, relevant_context)

        if self.settings.enforce_answer_format:
            repair = enforce_format(text, mode, subject=subject, context=relevant_context)
            if repair.changed:
                text = repair.text
                add_notice(notices, f"Coach format applied: {repair.summary()}.")
                self.logger.info("Format repair for mode=%s: %s", mode.value, repair.summary())

        text = self._enforce_mode_word_budget(text, mode)
        return self._enforce_length(text, mode), notices

    @staticmethod
    def _relevant_context(context: str, question: str) -> str:
        """Keep retrieved blocks with a topic term shared with the question."""
        terms = {
            term.lower().rstrip("s")
            for term in _QUESTION_WORD_RE.findall(question)
            if term.lower() not in _QUESTION_STOPWORDS
        }
        if not context or not terms:
            return context

        blocks = re.split(r"(?=^\[\d+\][^\n]*$)", context, flags=re.MULTILINE)
        relevant = [
            block.strip()
            for block in blocks
            if block.strip()
            and any(
                re.search(rf"\b{re.escape(term)}\w*\b", block, re.IGNORECASE)
                for term in terms
            )
        ]
        return "\n\n".join(relevant)

    @staticmethod
    def _keep_supplied_citations(text: str, context: str) -> str:
        """Remove citations that do not identify a relevant supplied block."""
        valid = set(_CONTEXT_REFERENCE_RE.findall(context))

        def keep_or_remove(match: re.Match[str]) -> str:
            return match.group(0) if match.group(1) in valid else ""

        return _INLINE_REFERENCE_RE.sub(keep_or_remove, text)

    @staticmethod
    def _enforce_mode_word_budget(text: str, mode: AnswerMode) -> str:
        """Keep detailed and revision answers compact without removing headings."""
        budgets = _MODE_WORD_BUDGETS.get(mode)
        if budgets is None:
            return text

        word_budgets = dict(zip(sections_for(mode), budgets))
        current_section: str | None = None
        remaining_words: int | None = None
        seen_lines: set[str] = set()
        seen_math: set[str] = set()
        output: list[str] = []
        for line in text.splitlines():
            if line.startswith("### "):
                current_section = line[4:]
                remaining_words = word_budgets.get(current_section)
                output.append(line)
                continue
            if not line.strip():
                if output and output[-1]:
                    output.append("")
                continue

            def keep_first_formula(match: re.Match[str]) -> str:
                expression = match.group(0)
                if "=" not in expression:
                    return expression
                key = re.sub(r"[\s\\]", "", expression[1:-1]).casefold()
                if key in seen_math:
                    return "that relation"
                seen_math.add(key)
                return expression

            line = _MATH_EXPRESSION_RE.sub(keep_first_formula, line)
            if re.fullmatch(
                r"\s*(?:[-*]\s*)?(?:that relation|this relation)[.!]?\s*",
                line,
                re.IGNORECASE,
            ):
                continue
            key = re.sub(r"\s+", " ", line).strip().casefold()
            if current_section is not None and key in seen_lines:
                continue
            if current_section is not None:
                seen_lines.add(key)

            if remaining_words is not None:
                kept = _trim_line(line, remaining_words)
                if not kept:
                    continue
                remaining_words -= _word_count(kept)
                output.append(kept)
            else:
                output.append(line)

        return "\n".join(output).strip()

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
