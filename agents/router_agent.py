"""Subject Router Agent.

Classifies the student's message into a subject (Physics / Chemistry /
Biology / General) and an intent (doubt / mcq / pyq / study_plan / weakness).

The LLM is asked for a strict JSON object.  Because small local models
occasionally return prose, the agent always has a keyword-based fallback and
also honours an explicit subject chosen in the UI.  A failure here must never
block the pipeline: a wrong subject only widens the retrieval filter.
"""

from __future__ import annotations

from typing import Any, Mapping

from agents.base import AgentError, BaseAgent, add_notice, emit_event
from agents.state import RouteDecision
from llm.base import ChatMessage, LLMError
from prompts.loader import ROUTER_PROMPT
from utils.constants import Intent, Subject, intent_from_text, subject_from_text
from utils.text_utils import clean_text, clamp, extract_json_object, format_history

__all__ = ["RouterAgent"]

_ROUTER_SYSTEM = (
    "You are a routing classifier for a NEET tutoring system. "
    "Follow the instructions exactly and reply with a single JSON object, nothing else."
)


class RouterAgent(BaseAgent):
    """Decides *what* the question is about and *which* pipeline should run."""

    name = "router"

    def run(self, state: dict[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
        question = clean_text(state.get("question"))
        if not question:
            raise AgentError("The question is empty - please type your doubt.")

        hint = str(state.get("subject_hint") or "auto").strip().lower()
        notices = list(state.get("notices") or [])
        decision = self._heuristic(question)

        emit_event(config, self.name, "Identifying the subject and the intent")
        if self.llm is not None:
            try:
                raw = self.llm.generate(
                    self._messages(question, state, hint),
                    max_new_tokens=192,
                    temperature=0.0,
                )
                parsed = self._parse(raw)
                if parsed is None:
                    self.logger.warning("Router output not parsable: %r", raw[:200])
                    add_notice(notices, "Subject classified with keyword rules.")
                else:
                    decision = parsed
            except LLMError as exc:
                self.logger.warning("Router LLM call failed: %s", exc)
                add_notice(
                    notices,
                    "Subject classification fell back to keyword rules after a model error.",
                )

        # An explicit UI choice always wins over the model.
        override = Subject.coerce(hint) if hint not in {"", "auto"} else None
        if override is not None:
            decision = {
                "subject": override.value,
                "intent": decision["intent"],
                "confidence": 1.0,
                "reason": "Subject selected by the student in the UI.",
            }

        self.logger.info(
            "Routed to subject=%s intent=%s (confidence=%.2f, reason=%s)",
            decision["subject"],
            decision["intent"],
            decision["confidence"],
            decision["reason"],
        )
        update: dict[str, Any] = {
            "subject": decision["subject"],
            "intent": decision["intent"],
            "subject_confidence": decision["confidence"],
            "router_reason": decision["reason"],
        }
        if notices:
            update["notices"] = notices
        return update

    # ----------------------------------------------------------------- prompts
    def _messages(self, question: str, state: Mapping[str, Any], hint: str) -> list[ChatMessage]:
        """Build the classification messages, including recent chat context."""
        prompt = self.prompts.render(
            ROUTER_PROMPT,
            history=format_history(
                state.get("history"),
                max_turns=self.settings.history_turns,
                max_chars_per_turn=400,
            ),
            question=question,
            subject_hint=hint or "auto",
        )
        return [
            {"role": "system", "content": _ROUTER_SYSTEM},
            {"role": "user", "content": prompt},
        ]

    # ------------------------------------------------------------------ parsing
    def _parse(self, raw: str) -> RouteDecision | None:
        """Validate the model's JSON answer, returning ``None`` when unusable."""
        payload = extract_json_object(raw)
        if not payload:
            return None

        subject = Subject.coerce(payload.get("subject"))
        if subject is None:
            return None

        intent = Intent.coerce(payload.get("intent"))
        try:
            confidence = clamp(float(payload.get("confidence", 0.5)), 0.0, 1.0)
        except (TypeError, ValueError):
            confidence = 0.5

        reason = clean_text(str(payload.get("reasoning") or payload.get("reason") or "")).strip()
        return {
            "subject": subject.value,
            "intent": intent.value,
            "confidence": confidence,
            "reason": reason[:200] or "Classified by the router model.",
        }

    def _heuristic(self, question: str) -> RouteDecision:
        """Deterministic keyword fallback used when the model is unavailable."""
        subject = subject_from_text(question) or Subject.GENERAL
        intent = intent_from_text(question) or Intent.DOUBT
        return {
            "subject": subject.value,
            "intent": intent.value,
            "confidence": 0.35,
            "reason": "Keyword-based fallback classification.",
        }
