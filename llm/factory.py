"""LLM factory - the only place that knows about concrete providers.

Selection is driven by :attr:`config.Settings.resolved_llm_provider`, which
prefers the new ``LLM_PROVIDER`` and falls back to the legacy
``NEET_LLM_BACKEND`` so every pre-provider ``.env`` keeps working. Application
code receives an :class:`LLMClient` and never learns which provider is active.
"""

from __future__ import annotations

from config import Settings
from llm.base import LLMClient
from llm.groq_client import GroqChatClient
from llm.huggingface_client import HuggingFaceChatClient
from llm.mock_client import MockLLM
from utils.logging_utils import get_logger

__all__ = ["LLMFactory"]

logger = get_logger(__name__)


class LLMFactory:
    """Creates the configured :class:`LLMClient` implementation."""

    @staticmethod
    def create(settings: Settings) -> LLMClient:
        """Instantiate the client for the resolved provider.

        Construction is cheap for every provider: HuggingFace loads its weights
        lazily and Groq builds its SDK client on first ``generate`` call.
        """
        provider = settings.resolved_llm_provider

        # Fail fast with a clear, secret-free message (also enforced by the
        # ``--check`` diagnostics) before touching any provider SDK.
        settings.validate_llm_provider()

        if provider == "mock":
            logger.warning(
                "Starting with the MOCK LLM provider - answers are placeholders. "
                "Set LLM_PROVIDER=groq or LLM_PROVIDER=huggingface for real tutoring."
            )
            return MockLLM(settings)
        if provider == "groq":
            logger.info("Using Groq LLM: %s", settings.groq_model)
            return GroqChatClient(settings)
        if provider == "huggingface":
            logger.info("Using HuggingFace LLM: %s", settings.llm_model)
            return HuggingFaceChatClient(settings)

        # Unreachable: validate_llm_provider() rejects unknown providers above.
        raise ValueError(  # pragma: no cover - defensive
            f"Unsupported LLM provider '{provider}' (expected groq | huggingface | mock)"
        )
