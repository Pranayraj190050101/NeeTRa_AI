"""Central configuration for NEETra AI, the AI NEET Coach.

Every field of :class:`Settings` can be overridden through an environment
variable prefixed with ``NEET_`` (a local ``.env`` file is loaded
automatically).  Provider selection deliberately uses prefix-free names so a
``.env`` reads naturally::

    LLM_PROVIDER=groq            # groq | huggingface | mock
    GROQ_API_KEY=gsk_...         # required when LLM_PROVIDER=groq
    GROQ_MODEL=qwen/qwen3.8-27b
    NEET_LLM_MODEL=Qwen/Qwen2.5-3B-Instruct
    NEET_TOP_K=6

Backward compatibility: ``LLM_PROVIDER`` wins when set, otherwise the legacy
``NEET_LLM_BACKEND`` (``hf`` | ``mock``) decides the provider, so every existing
``.env`` keeps working untouched.

The module exposes a cached ``get_settings()`` accessor so the rest of the
code base never re-reads the environment.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from utils.constants import AnswerMode

# Absolute path of the ``neet_ai_tutor`` directory.  All relative paths in the
# configuration are resolved against it, which keeps the app runnable from any
# working directory.
BASE_DIR: Path = Path(__file__).resolve().parent

DevicePreference = Literal["auto", "cpu", "cuda", "mps"]
DTypeName = Literal["auto", "float16", "bfloat16", "float32"]
LLMBackend = Literal["hf", "mock"]

#: Canonical provider names of the provider-based architecture.
PROVIDERS: tuple[str, ...] = ("groq", "huggingface", "mock")

#: ``LLM_PROVIDER`` / ``NEET_LLM_BACKEND`` spellings mapped onto a canonical
#: provider.  ``hf`` (the legacy backend value) means ``huggingface``.
_PROVIDER_ALIASES: dict[str, str] = {
    "groq": "groq",
    "huggingface": "huggingface",
    "hugging_face": "huggingface",
    "hf": "huggingface",
    "mock": "mock",
    "stub": "mock",
}


class ProviderConfigError(RuntimeError):
    """Raised when the selected LLM provider is missing required configuration.

    It is intentionally a plain ``RuntimeError`` living in :mod:`config` (not in
    the ``llm`` package) so importing it here creates no dependency cycle; the
    message is safe to show an operator and never contains a secret.
    """


def mask_secret(value: str | None) -> str:
    """Return a display-safe fingerprint of a secret, never the secret itself.

    ``gsk_abc123...xyz9`` -> ``...xyz9``; empty/None -> ``not set``.  Only the
    final four characters are kept, which is enough to tell two keys apart in a
    diagnostic without leaking either of them.
    """
    cleaned = (value or "").strip()
    if not cleaned:
        return "not set"
    if len(cleaned) <= 4:
        return "****"
    return f"...{cleaned[-4:]}"


class Settings(BaseSettings):
    """Typed, validated application settings."""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        env_prefix="NEET_",
        case_sensitive=False,
        extra="ignore",
        # Allow constructing by field name as well as by ``validation_alias``
        # (e.g. ``Settings(llm_provider="groq")`` in tests and embedding code).
        populate_by_name=True,
    )

    # ------------------------------------------------------------------ app
    # Product name shown in the Gradio page title, header and reports.
    app_name: str = "NEETra AI"
    app_tagline: str = "Every NEET Doubt. One Wish Away."
    app_version: str = "1.1.0"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_to_file: bool = True

    # ---------------------------------------------------------------- paths
    data_dir: Path = BASE_DIR / "data"
    vector_store_dir: Path = BASE_DIR / "vectordb"
    vector_store_name: str = "neet_index"
    prompt_dir: Path = BASE_DIR / "prompts"
    log_dir: Path = BASE_DIR / "logs"
    #: Local JSON study history (tests, scores, weak areas) - DB later.
    progress_dir: Path = BASE_DIR / "progress"

    # ---------------------------------------------------------- embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: DevicePreference = "auto"
    embedding_batch_size: int = 32
    # Cosine similarity is only meaningful when vectors are L2 normalised;
    # FAISS then returns squared L2 distances that we convert back.
    normalize_embeddings: bool = True

    # -------------------------------------------------------------- chunking
    chunk_size: int = 900
    chunk_overlap: int = 150
    min_chunk_chars: int = 80
    min_page_chars: int = 40

    # ------------------------------------------------------------- retrieval
    top_k: int = Field(default=5, ge=1, le=12)
    fetch_k: int = Field(default=20, ge=5, le=100)
    score_threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    context_char_budget: int = Field(default=6000, ge=1000, le=40000)

    # ------------------------------------------------------------------- llm
    #: Legacy backend switch, kept for backward compatibility.  Overridden by
    #: ``LLM_PROVIDER`` when that is set (see :attr:`resolved_llm_provider`).
    #: Default to the offline mock backend so the app can boot in local/student
    #: mode without loading a hosted model or downloading weights.
    llm_backend: LLMBackend = "mock"
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    llm_device: DevicePreference = "auto"
    llm_dtype: DTypeName = "auto"
    llm_load_in_4bit: bool = False
    llm_trust_remote_code: bool = True
    llm_max_new_tokens: int = Field(default=768, ge=64, le=4096)
    llm_max_input_tokens: int = Field(default=4096, ge=512, le=32768)
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    llm_top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    llm_repetition_penalty: float = Field(default=1.05, ge=1.0, le=2.0)

    # ------------------------------------------------------------- llm provider
    # Provider-based architecture.  ``LLM_PROVIDER`` (no ``NEET_`` prefix) is the
    # forward-looking switch; the concrete provider keys live beside it.
    #: One of ``groq`` / ``huggingface`` / ``mock``.  ``None`` means "not set",
    #: so the legacy ``NEET_LLM_BACKEND`` decides (backward compatibility).
    llm_provider: str | None = Field(default=None, validation_alias="LLM_PROVIDER")

    # --- Groq (hosted, OpenAI-compatible) ---------------------------------
    #: Never printed; surfaced only through :func:`mask_secret`.
    groq_api_key: str | None = Field(default=None, validation_alias="GROQ_API_KEY")
    groq_model: str = Field(default="qwen/qwen3.8-27b", validation_alias="GROQ_MODEL")
    groq_temperature: float = Field(default=0.2, ge=0.0, le=2.0, validation_alias="GROQ_TEMPERATURE")
    groq_top_p: float = Field(default=0.9, gt=0.0, le=1.0, validation_alias="GROQ_TOP_P")
    groq_max_tokens: int = Field(default=2048, ge=64, le=8192, validation_alias="GROQ_MAX_TOKENS")

    # -------------------------------------------------------------- behaviour
    history_turns: int = Field(default=4, ge=0, le=20)
    max_question_chars: int = Field(default=2000, ge=100, le=8000)
    max_answer_chars: int = Field(default=6000, ge=500, le=20000)
    allow_general_fallback: bool = True
    stream_tokens: bool = True

    # ------------------------------------------------------------ coach persona
    #: Default answer mode; the student can still switch it in the UI.
    answer_mode: AnswerMode = AnswerMode.DETAILED
    #: Prepend ``prompts/persona.txt`` (15-years NCERT faculty) to the system prompt.
    persona_enabled: bool = True
    #: Repair the section contract of the selected mode on the final answer.
    enforce_answer_format: bool = True
    #: Questions generated in MCQ Practice mode.
    mcq_count: int = Field(default=5, ge=2, le=10)

    # ------------------------------------------------------------- platform
    #: Serve the modular tabs (Ask Genie / Dashboard / Learn / Practice /
    #: Progress / Revision).
    platform_enabled: bool = True
    #: Show the Learn module.  ``NEETRA_ENABLE_LEARN=false`` hides the tab and
    #: skips the Learn UI entirely.  The student product defaults to the
    #: Ask → Practice → Progress → Revision → Dashboard flow, with Learn kept
    #: as an opt-in compatibility escape hatch for older chapter-study workflows.
    learn_enabled: bool = Field(default=False, validation_alias="NEETRA_ENABLE_LEARN")
    #: Questions per generated test in the Practice module.
    test_question_count: int = Field(default=5, ge=2, le=15)
    #: NEET marking scheme used for a correct answer.
    marks_correct: float = Field(default=4.0, ge=0.0, le=10.0)
    #: Penalty for a wrong answer (NEET negative marking; ``0`` disables it).
    marks_incorrect: float = Field(default=-1.0, ge=-10.0, le=0.0)
    #: Accuracy below this marks a topic as weak in the Progress analytics.
    weak_accuracy_threshold: float = Field(default=0.75, ge=0.0, le=1.0)

    # ----------------------------------------------------------- integration
    hf_token: str | None = None
    faiss_allow_dangerous_deserialize: bool = True
    gradio_server_name: str = "127.0.0.1"
    gradio_server_port: int = Field(default=7860, ge=1, le=65535)
    gradio_share: bool = False
    chatbot_height: int = Field(default=540, ge=320, le=1200)

    # ------------------------------------------------------------- validators
    @field_validator("data_dir", "vector_store_dir", "prompt_dir", "log_dir", "progress_dir", mode="after")
    @classmethod
    def _resolve_path(cls, value: Path) -> Path:
        """Make every configured path absolute (relative to the project root)."""
        return value if value.is_absolute() else (BASE_DIR / value).resolve()

    @field_validator("vector_store_name")
    @classmethod
    def _safe_index_name(cls, value: str) -> str:
        cleaned = value.strip().replace(" ", "_")
        if not cleaned:
            raise ValueError("vector_store_name must not be empty")
        return cleaned

    @field_validator("answer_mode", mode="before")
    @classmethod
    def _coerce_answer_mode(cls, value: Any) -> AnswerMode:
        """Accept friendly spellings such as ``revision`` or ``MCQ Practice``."""
        return AnswerMode.coerce(value)

    @field_validator("llm_provider", mode="before")
    @classmethod
    def _normalise_provider(cls, value: Any) -> Any:
        """Trim/case-fold ``LLM_PROVIDER``; an empty value means "not set"."""
        if value is None:
            return None
        text = str(value).strip().lower()
        return text or None

    # ------------------------------------------------------------- properties
    @property
    def index_dir(self) -> Path:
        """Directory holding ``index.faiss`` and ``index.pkl``."""
        return self.vector_store_dir / self.vector_store_name

    @property
    def resolved_llm_provider(self) -> str:
        """The canonical provider in use: ``LLM_PROVIDER`` then legacy backend.

        Priority (the whole point of the provider architecture):

        1. ``LLM_PROVIDER`` when set (``groq`` / ``huggingface`` / ``mock``, with
           ``hf`` accepted as a synonym for ``huggingface``);
        2. otherwise the legacy ``NEET_LLM_BACKEND`` (``hf`` -> ``huggingface``,
           ``mock`` -> ``mock``), so every pre-provider ``.env`` still resolves.

        Unknown spellings are returned verbatim so callers (the factory) can
        raise one clear :class:`ProviderConfigError` instead of this property
        ever throwing during logging/summary.
        """
        if self.llm_provider:
            return _PROVIDER_ALIASES.get(self.llm_provider, self.llm_provider)
        return _PROVIDER_ALIASES.get((self.llm_backend or "hf").lower(), "huggingface")

    @property
    def active_llm_model(self) -> str:
        """The model id of whichever provider is resolved."""
        provider = self.resolved_llm_provider
        if provider == "groq":
            return self.groq_model
        if provider == "mock":
            return "rule-based stub"
        return self.llm_model

    @property
    def groq_configured(self) -> bool:
        """True when a non-empty ``GROQ_API_KEY`` is present."""
        return bool((self.groq_api_key or "").strip())

    @property
    def index_metadata_path(self) -> Path:
        """Sidecar JSON with index statistics (read without loading models)."""
        return self.vector_store_dir / f"{self.vector_store_name}.meta.json"

    # ----------------------------------------------------------------- helpers
    def validate_llm_provider(self) -> None:
        """Fail fast and clearly when the resolved provider is misconfigured.

        Raised at startup (factory) and surfaced by ``app.py --check``.  The
        message never contains the API key.
        """
        provider = self.resolved_llm_provider
        if provider not in PROVIDERS:
            raise ProviderConfigError(
                f"Unknown LLM_PROVIDER '{self.llm_provider}'. "
                f"Supported: {', '.join(PROVIDERS)}."
            )
        if provider == "groq" and not self.groq_configured:
            raise ProviderConfigError(
                "LLM_PROVIDER=groq requires GROQ_API_KEY. Set it in your "
                "environment or .env file (never commit .env); get a key at "
                "console.groq.com. Alternatively use LLM_PROVIDER=huggingface "
                "for a local model or LLM_PROVIDER=mock for offline testing."
            )
        # A real provider must have a model to talk to.  ``mock`` always reports
        # the fixed "rule-based stub", so only groq/huggingface can be empty.
        if provider != "mock" and not self.active_llm_model.strip():
            variable = "GROQ_MODEL" if provider == "groq" else "NEET_LLM_MODEL"
            raise ProviderConfigError(
                f"The '{provider}' provider has no model configured. Set "
                f"{variable} to a model id (for example "
                + (
                    "'qwen/qwen3.8-27b')."
                    if provider == "groq"
                    else "'Qwen/Qwen2.5-7B-Instruct')."
                )
            )

    def llm_provider_diagnostic(self) -> dict[str, Any]:
        """Provider + model for ``--check``/the UI, with any secret masked."""
        provider = self.resolved_llm_provider
        info: dict[str, Any] = {"provider": provider, "model": self.active_llm_model}
        if provider == "groq":
            info["configured"] = self.groq_configured
            info["api_key"] = mask_secret(self.groq_api_key)
            if not self.groq_configured:
                info["error"] = "GROQ_API_KEY is required when LLM_PROVIDER=groq"
        return info

    def ensure_directories(self) -> None:
        """Create all runtime directories if they are missing."""
        for path in (self.data_dir, self.vector_store_dir, self.log_dir, self.progress_dir):
            path.mkdir(parents=True, exist_ok=True)

    def summary(self) -> dict[str, Any]:
        """Non-secret configuration snapshot, useful for logs and the UI."""
        return {
            "app_name": self.app_name,
            "app_version": self.app_version,
            "embedding_model": self.embedding_model,
            "embedding_device": self.embedding_device,
            "llm_provider": self.resolved_llm_provider,
            "llm_backend": self.llm_backend,
            "llm_model": self.llm_model,
            "llm_model_active": self.active_llm_model,
            "llm_device": self.llm_device,
            "llm_load_in_4bit": self.llm_load_in_4bit,
            "answer_mode": self.answer_mode.value,
            "persona_enabled": self.persona_enabled,
            "enforce_answer_format": self.enforce_answer_format,
            "mcq_count": self.mcq_count,
            "platform_enabled": self.platform_enabled,
            "learn_enabled": self.learn_enabled,
            "test_question_count": self.test_question_count,
            "marks_correct": self.marks_correct,
            "marks_incorrect": self.marks_incorrect,
            "progress_dir": str(self.progress_dir),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "score_threshold": self.score_threshold,
            "index_dir": str(self.index_dir),
            "stream_tokens": self.stream_tokens,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()


def reload_settings() -> Settings:
    """Drop the cache and rebuild settings (used by tests / config reloads)."""
    get_settings.cache_clear()
    return get_settings()
