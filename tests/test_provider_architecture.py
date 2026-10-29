"""Tests for the provider architecture (LLM_PROVIDER / Groq / backward compat).

Covers the four areas required by the provider refactor:

* **provider resolution** - ``LLM_PROVIDER`` then legacy ``NEET_LLM_BACKEND``;
* **environment loading**  - the prefix-free ``LLM_*`` / ``GROQ_*`` variables;
* **Groq validation**       - missing key / unknown provider / secret hygiene;
* **fallback compatibility**- a pre-provider ``NEET_LLM_BACKEND`` ``.env`` and the
  factory still resolve to the right client.

Everything runs offline: the Groq SDK is never imported because the client is
injected, and no network call is made.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from config import (
    PROVIDERS,
    ProviderConfigError,
    Settings,
    mask_secret,
)
from llm.factory import LLMFactory
from llm.groq_client import GroqChatClient
from llm.huggingface_client import HuggingFaceChatClient
from llm.base import LLMError
from llm.mock_client import MockLLM


def make_settings(**kwargs: Any) -> Settings:
    """Build settings from explicit kwargs + os.environ, ignoring any ``.env``."""
    return Settings(_env_file=None, **kwargs)


# ---------------------------------------------------------------------------
# A stand-in for the official ``groq`` SDK client.  ``chat.completions.create``
# mirrors the OpenAI-compatible surface the real client exposes.
# ---------------------------------------------------------------------------
class FakeGroqClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **params: Any) -> Any:
        self.calls.append(params)
        if params.get("stream"):
            return iter(
                [
                    SimpleNamespace(
                        choices=[SimpleNamespace(delta=SimpleNamespace(content="Hi"))]
                    ),
                    SimpleNamespace(
                        choices=[SimpleNamespace(delta=SimpleNamespace(content=" there"))]
                    ),
                ]
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Hi there"))]
        )


# --------------------------------------------------------------------------
# 1. Provider resolution
# --------------------------------------------------------------------------
def test_llm_provider_resolves_groq() -> None:
    assert make_settings(llm_provider="groq", groq_api_key="gsk_abc123").resolved_llm_provider == "groq"


def test_llm_provider_alias_hf_maps_to_huggingface() -> None:
    assert make_settings(llm_provider="hf").resolved_llm_provider == "huggingface"


def test_llm_provider_alias_huggingface_is_canonical() -> None:
    assert make_settings(llm_provider="HuggingFace").resolved_llm_provider == "huggingface"


def test_llm_provider_mock() -> None:
    assert make_settings(llm_provider="mock").resolved_llm_provider == "mock"


def test_provider_defaults_to_three_supported_values() -> None:
    assert set(PROVIDERS) == {"groq", "huggingface", "mock"}


def test_active_model_tracks_provider() -> None:
    assert (
        make_settings(llm_provider="groq", groq_api_key="gsk_x", groq_model="llama-3.3-70b").active_llm_model
        == "llama-3.3-70b"
    )
    assert make_settings(llm_provider="mock").active_llm_model == "rule-based stub"
    assert make_settings(llm_provider="huggingface").active_llm_model == make_settings().llm_model


# --------------------------------------------------------------------------
# 2. Environment loading (prefix-free LLM_* / GROQ_* names)
# --------------------------------------------------------------------------
def test_provider_and_groq_settings_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_from_env")
    monkeypatch.setenv("GROQ_MODEL", "qwen/qwen3-32b")
    monkeypatch.delenv("NEET_LLM_BACKEND", raising=False)

    settings = Settings(_env_file=None)
    assert settings.resolved_llm_provider == "groq"
    assert settings.groq_api_key == "gsk_from_env"
    assert settings.groq_model == "qwen/qwen3-32b"
    assert settings.groq_configured is True


def test_llm_provider_wins_over_legacy_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("NEET_LLM_BACKEND", "hf")

    settings = Settings(_env_file=None)
    assert settings.resolved_llm_provider == "mock"


# --------------------------------------------------------------------------
# 3. Groq validation + secret hygiene
# --------------------------------------------------------------------------
def test_groq_without_api_key_raises_clear_error() -> None:
    settings = make_settings(llm_provider="groq", groq_api_key=None)
    with pytest.raises(ProviderConfigError) as exc:
        settings.validate_llm_provider()
    message = str(exc.value)
    assert "GROQ_API_KEY" in message
    assert "groq" in message.lower()


def test_validation_error_never_leaks_the_secret() -> None:
    secret = "gsk_super_secret_value_123"
    settings = make_settings(llm_provider="groq", groq_api_key=None)
    # sanity: the key really is unset here, but even when a *different* config
    # problem fires, the diagnostic must only ever expose a masked fingerprint.
    with pytest.raises(ProviderConfigError) as exc:
        settings.validate_llm_provider()
    assert secret not in str(exc.value)

    armed = make_settings(llm_provider="groq", groq_api_key=secret)
    armed.validate_llm_provider()  # must not raise
    diagnostic = armed.llm_provider_diagnostic()
    assert diagnostic["api_key"] == mask_secret(secret)
    assert secret not in diagnostic["api_key"]
    assert secret not in str(diagnostic)


def test_unknown_provider_raises() -> None:
    settings = make_settings(llm_provider="openai")
    with pytest.raises(ProviderConfigError):
        settings.validate_llm_provider()


def test_groq_requires_a_configured_model() -> None:
    settings = make_settings(llm_provider="groq", groq_api_key="gsk_x", groq_model="   ")
    with pytest.raises(ProviderConfigError) as exc:
        settings.validate_llm_provider()
    assert "GROQ_MODEL" in str(exc.value)


def test_huggingface_requires_a_configured_model() -> None:
    settings = make_settings(llm_provider="huggingface", llm_model="")
    with pytest.raises(ProviderConfigError) as exc:
        settings.validate_llm_provider()
    assert "NEET_LLM_MODEL" in str(exc.value)


def test_mock_needs_no_model_or_key() -> None:
    # ``mock`` is the offline provider: no key, no real model, still valid.
    make_settings(llm_provider="mock").validate_llm_provider()


def test_mask_secret_behaviour() -> None:
    assert mask_secret(None) == "not set"
    assert mask_secret("") == "not set"
    assert mask_secret("abc") == "****"
    assert mask_secret("gsk_long_key_9876") == "...9876"


def test_groq_diagnostic_reports_provider_and_model() -> None:
    settings = make_settings(llm_provider="groq", groq_api_key="gsk_abc1234", groq_model="llama-3.3-70b")
    info = settings.llm_provider_diagnostic()
    assert info == {
        "provider": "groq",
        "model": "llama-3.3-70b",
        "configured": True,
        "api_key": mask_secret("gsk_abc1234"),
    }


# --------------------------------------------------------------------------
# 4. Fallback compatibility (legacy NEET_LLM_BACKEND)
# --------------------------------------------------------------------------
def test_legacy_backend_hf_still_resolves_when_provider_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("NEET_LLM_BACKEND", "hf")
    assert Settings(_env_file=None).resolved_llm_provider == "huggingface"


def test_legacy_backend_mock_still_resolves_when_provider_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("NEET_LLM_BACKEND", "mock")
    assert Settings(_env_file=None).resolved_llm_provider == "mock"


# --------------------------------------------------------------------------
# Factory wiring - application code receives a client, provider-agnostic
# --------------------------------------------------------------------------
def test_factory_returns_mock_for_mock_provider() -> None:
    assert isinstance(LLMFactory.create(make_settings(llm_provider="mock")), MockLLM)


def test_factory_returns_groq_for_groq_provider() -> None:
    client = LLMFactory.create(make_settings(llm_provider="groq", groq_api_key="gsk_x"))
    assert isinstance(client, GroqChatClient)
    assert client.name == "groq"


def test_factory_returns_huggingface_for_legacy_hf_backend() -> None:
    assert isinstance(LLMFactory.create(make_settings(llm_backend="hf")), HuggingFaceChatClient)


def test_factory_fails_fast_for_groq_without_key() -> None:
    with pytest.raises(ProviderConfigError):
        LLMFactory.create(make_settings(llm_provider="groq", groq_api_key=None))


# --------------------------------------------------------------------------
# GroqChatClient behaviour with an injected SDK client
# --------------------------------------------------------------------------
def _groq_settings() -> Settings:
    return make_settings(llm_provider="groq", groq_api_key="gsk_abcdef123456")


def test_groq_blocking_generation_and_params() -> None:
    fake = FakeGroqClient()
    client = GroqChatClient(_groq_settings(), client=fake)
    answer = client.generate(
        [{"role": "user", "content": "What is the mole?"}], max_new_tokens=128, temperature=0.3
    )
    assert answer == "Hi there"
    params = fake.calls[0]
    assert params["messages"] == [{"role": "user", "content": "What is the mole?"}]
    assert params["max_tokens"] == 128
    assert params["temperature"] == 0.3
    assert params.get("stream") is None


def test_groq_streaming_forwards_tokens_and_returns_full_text() -> None:
    fake = FakeGroqClient()
    client = GroqChatClient(_groq_settings(), client=fake)
    received: list[str] = []
    answer = client.generate(
        [{"role": "user", "content": "hi"}], on_token=received.append
    )
    assert answer == "Hi there"
    assert received == ["Hi", " there"]
    assert fake.calls[0]["stream"] is True


def test_groq_requires_at_least_one_message() -> None:
    client = GroqChatClient(_groq_settings(), client=FakeGroqClient())
    with pytest.raises(LLMError):
        client.generate([])


def test_groq_describe_masks_api_key() -> None:
    secret = "gsk_abcdef123456"
    described = GroqChatClient(_groq_settings(), client=FakeGroqClient()).describe()
    assert described["provider"] == "groq"
    assert described["api_key"] == mask_secret(secret)
    assert secret not in str(described)


def test_groq_missing_key_raises_at_client_level() -> None:
    # Defence in depth: build the client directly with no key and no injection.
    settings = make_settings(llm_provider="groq", groq_api_key=None)
    client = GroqChatClient(settings)
    with pytest.raises(LLMError):
        client.generate([{"role": "user", "content": "hi"}])


def test_groq_sdk_errors_are_wrapped_in_llmerror() -> None:
    class ExplodingClient:
        chat = SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_: (_ for _ in ()).throw(RuntimeError("429 rate limited")))
        )

    client = GroqChatClient(_groq_settings(), client=ExplodingClient())
    with pytest.raises(LLMError):
        client.generate([{"role": "user", "content": "hi"}])
