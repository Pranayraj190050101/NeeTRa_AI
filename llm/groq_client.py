"""Groq chat client - a first-class, hosted LLM provider.

Groq exposes an OpenAI-compatible ``chat.completions`` API, so this client is a
thin, provider-specific adapter onto the shared :class:`LLMClient` contract.
The rest of the application (agents, learning services, UI) never imports this
module and cannot tell the difference between Groq, a local HuggingFace model
or the offline mock.

Production concerns handled here:

* **Lazy SDK import + connection** - the ``groq`` package is imported and the
  client built on first use, so ``app.py --check`` and the offline test suite
  run without the dependency installed.
* **Key hygiene** - the API key is read from configuration, never logged and
  never included in :meth:`describe` output (only :func:`config.mask_secret`).
* **Actionable errors** - SDK failures are wrapped in :class:`LLMError` so the
  agent layer turns them into a friendly chat message instead of a traceback.
* **Token streaming** - when an ``on_token`` callback is supplied the request is
  streamed and each text fragment is forwarded while the full answer is still
  returned.
"""

from __future__ import annotations

from typing import Any, Sequence

from config import Settings, mask_secret
from llm.base import ChatMessage, LLMClient, LLMError, TokenCallback
from utils.logging_utils import get_logger

__all__ = ["GroqChatClient"]

logger = get_logger(__name__)


class GroqChatClient(LLMClient):
    """Chat completions backed by the official ``groq`` SDK."""

    name = "groq"

    def __init__(self, settings: Settings, *, client: Any | None = None) -> None:
        self.settings = settings
        self.model_id = settings.groq_model
        # ``client`` is injectable for tests; when absent it is built lazily.
        self._client = client
        self._client_built = client is not None

    # ------------------------------------------------------------------ client
    def _ensure_client(self) -> Any:
        """Build the SDK client on first use, failing with a clear message."""
        if self._client_built:
            assert self._client is not None
            return self._client
        if not (self.settings.groq_api_key or "").strip():
            # The factory validates before constructing, so this is defence in
            # depth for anyone building the client directly.
            raise LLMError("GROQ_API_KEY is required to use the Groq provider.")
        try:
            from groq import Groq
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise LLMError(
                "The 'groq' package is not installed. Run "
                "`pip install groq` or choose another LLM_PROVIDER."
            ) from exc
        self._client = Groq(api_key=self.settings.groq_api_key)
        self._client_built = True
        logger.info("Groq client ready (model=%s)", self.model_id)
        return self._client

    # --------------------------------------------------------------- generation
    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        on_token: TokenCallback | None = None,
    ) -> str:
        """Generate an assistant reply, optionally streaming through ``on_token``."""
        if not messages:
            raise LLMError("generate() requires at least one message")

        client = self._ensure_client()
        params: dict[str, Any] = {
            "model": self.model_id,
            "messages": [dict(message) for message in messages],
            "temperature": (
                self.settings.groq_temperature if temperature is None else float(temperature)
            ),
            "max_tokens": int(max_new_tokens or self.settings.groq_max_tokens),
            "top_p": self.settings.groq_top_p,
        }
        try:
            if on_token is None:
                return self._generate_blocking(client, params)
            return self._generate_streaming(client, params, on_token)
        except LLMError:
            raise
        except Exception as exc:  # groq.APIError, network, schema drift, ...
            raise LLMError(f"Groq generation failed: {exc}") from exc

    def _generate_blocking(self, client: Any, params: dict[str, Any]) -> str:
        completion = client.chat.completions.create(**params)
        return self._extract(completion)

    def _generate_streaming(
        self, client: Any, params: dict[str, Any], on_token: TokenCallback
    ) -> str:
        stream = client.chat.completions.create(stream=True, **params)
        pieces: list[str] = []
        for chunk in stream:
            piece = self._extract_delta(chunk)
            if piece:
                pieces.append(piece)
                on_token(piece)
        return "".join(pieces)

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _extract(completion: Any) -> str:
        """Pull the assistant text out of a non-streamed completion."""
        choices = getattr(completion, "choices", None) or []
        if not choices:
            raise LLMError("Groq returned no completion choices")
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None) if message is not None else None
        if content is None:
            raise LLMError("Groq returned an empty completion")
        return content

    @staticmethod
    def _extract_delta(chunk: Any) -> str:
        """Pull the streamed text fragment out of one chunk (may be empty)."""
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            return ""
        delta = getattr(choices[0], "delta", None)
        return (getattr(delta, "content", None) or "") if delta is not None else ""

    # ------------------------------------------------------------------- state
    def describe(self) -> dict[str, Any]:
        """Provider metadata for logs and the health panel - never the key."""
        return {
            "provider": self.name,
            "backend": self.name,
            "model": self.model_id,
            "transport": "hosted api",
            "api_key": mask_secret(self.settings.groq_api_key),
            "client_ready": self._client_built,
        }
