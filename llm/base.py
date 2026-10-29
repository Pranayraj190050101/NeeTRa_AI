"""Abstract LLM contract shared by the real and mock backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Sequence, TypedDict

__all__ = ["ChatMessage", "LLMClient", "LLMError", "TokenCallback"]

#: ``{"role": "system"|"user"|"assistant", "content": "..."}``
ChatMessage = TypedDict("ChatMessage", {"role": str, "content": str})

#: Called with each generated text fragment when streaming is enabled.
TokenCallback = Callable[[str], None]


class LLMError(RuntimeError):
    """Raised when a model cannot be loaded or generation fails."""


class LLMClient(ABC):
    """Minimal chat-completion interface used by the agents."""

    #: Short identifier surfaced in the UI / logs (``hf``, ``mock``).
    name: str = "base"

    @abstractmethod
    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        on_token: TokenCallback | None = None,
    ) -> str:
        """Generate an assistant reply for ``messages``.

        Args:
            messages: Chat history ending with the latest user turn.
            max_new_tokens: Hard cap on generated tokens.
            temperature: ``0`` means greedy decoding.
            on_token: Optional streaming callback invoked with text fragments.
                Implementations must still return the full text.
        """

    def describe(self) -> dict[str, Any]:
        """Metadata for logs and the UI health panel."""
        return {"backend": self.name}

    def unload(self) -> None:  # pragma: no cover - optional hook
        """Release model memory. Safe to call multiple times."""
