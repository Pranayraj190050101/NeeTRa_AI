"""Local LLM clients.

The agents depend on the small :class:`LLMClient` contract rather than on a
concrete library, which keeps the graph testable and makes swapping Qwen for
another HuggingFace chat model a one-line configuration change.
"""

from __future__ import annotations

from llm.base import ChatMessage, LLMClient, LLMError, TokenCallback
from llm.factory import LLMFactory
from llm.groq_client import GroqChatClient
from llm.huggingface_client import HuggingFaceChatClient
from llm.mock_client import MockLLM

__all__ = [
    "ChatMessage",
    "LLMClient",
    "LLMError",
    "TokenCallback",
    "LLMFactory",
    "GroqChatClient",
    "HuggingFaceChatClient",
    "MockLLM",
]
