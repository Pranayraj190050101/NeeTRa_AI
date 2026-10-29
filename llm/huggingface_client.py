"""HuggingFace Transformers chat client (default: Qwen2.5-7B-Instruct).

Key production concerns handled here:

* **Lazy loading** - the 7B checkpoint is only downloaded/loaded on first use,
  so the Gradio UI starts instantly.
* **Chat template** - messages are rendered with the tokenizer's own template
  (ChatML for Qwen) instead of a hand-rolled prompt format.
* **Memory safety** - dtype is chosen from the device, optional 4-bit NF4
  quantisation, prompt truncation to ``llm_max_input_tokens`` and CUDA OOM
  translated into actionable guidance.
* **Token streaming** - ``TextIteratorStreamer`` powers the typing effect in
  the UI while still returning the complete answer.
"""

from __future__ import annotations

import platform
import threading
import time
from typing import Any, Sequence

from config import Settings
from llm.base import ChatMessage, LLMClient, LLMError, TokenCallback
from utils.device import resolve_device, resolve_torch_dtype
from utils.logging_utils import get_logger

__all__ = ["HuggingFaceChatClient"]

logger = get_logger(__name__)

_SPECIAL_TOKENS: tuple[str, ...] = (
    "<|im_end|>",
    "<|im_start|>",
    "<|endoftext|>",
    "<|eot_id|>",
    "<|end|>",
    "</s>",
)


class HuggingFaceChatClient(LLMClient):
    """A local causal-LM chat client with optional token streaming."""

    name = "hf"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_id = settings.llm_model
        self.device = resolve_device(settings.llm_device)
        self.dtype = resolve_torch_dtype(settings.llm_dtype, self.device)
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._load_lock = threading.Lock()
        self._generate_lock = threading.Lock()
        self._load_seconds: float | None = None
        self._quantized = False

    # ------------------------------------------------------------------ loading
    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def load(self) -> None:
        """Download (if needed) and load the tokenizer + model."""
        with self._load_lock:
            if self.is_loaded:
                return
            started = time.perf_counter()
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:  # pragma: no cover - dependency issue
                raise LLMError(
                    "transformers/torch are not installed. Run "
                    "'pip install -r requirements.txt' (see README for CUDA wheels)."
                ) from exc

            logger.info(
                "Loading LLM %s (device=%s dtype=%s 4bit=%s) - first run downloads weights...",
                self.model_id,
                self.device,
                self.dtype,
                self.settings.llm_load_in_4bit,
            )
            auth = {"token": self.settings.hf_token} if self.settings.hf_token else {}

            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id,
                    trust_remote_code=self.settings.llm_trust_remote_code,
                    **auth,
                )
            except Exception as exc:
                raise LLMError(
                    f"Unable to load tokenizer for '{self.model_id}': {exc}. "
                    "Check the model id, your internet connection or NEET_HF_TOKEN."
                ) from exc

            model_kwargs: dict[str, Any] = {
                "torch_dtype": self.dtype,
                "low_cpu_mem_usage": True,
                "trust_remote_code": self.settings.llm_trust_remote_code,
                **auth,
            }
            if self.device.startswith("cuda"):
                model_kwargs["device_map"] = "auto"
            quantization_config = self._quantization_config()
            if quantization_config is not None:
                model_kwargs["quantization_config"] = quantization_config

            try:
                self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
            except Exception as exc:
                raise LLMError(
                    f"Unable to load model '{self.model_id}': {exc}. "
                    "Try a smaller checkpoint (Qwen/Qwen2.5-3B-Instruct), enable "
                    "NEET_LLM_LOAD_IN_4BIT=true, or set NEET_LLM_BACKEND=mock."
                ) from exc

            if not self.device.startswith("cuda"):
                self._model.to(self.device)
            self._model.eval()

            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
            self._tokenizer.padding_side = "left"

            self._load_seconds = time.perf_counter() - started
            logger.info("LLM ready in %.1fs (device=%s)", self._load_seconds, self.device)

    def _quantization_config(self) -> Any | None:
        """Build a 4-bit NF4 config when requested and supported."""
        if not self.settings.llm_load_in_4bit:
            return None
        if not self.device.startswith("cuda"):
            logger.warning("4-bit loading requires CUDA - continuing in full precision")
            return None
        try:
            import torch
            from transformers import BitsAndBytesConfig

            self._quantized = True
            logger.info("4-bit NF4 quantisation enabled")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        except ImportError:
            logger.warning("bitsandbytes is not installed - continuing in full precision")
        except Exception as exc:  # pragma: no cover - driver/binary mismatch
            logger.warning("Unable to configure 4-bit loading (%s) - full precision", exc)
        if platform.system() == "Windows":
            logger.warning("Note: bitsandbytes 4-bit support on native Windows is experimental")
        return None

    # ---------------------------------------------------------------- generation
    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        on_token: TokenCallback | None = None,
    ) -> str:
        """Generate a reply, optionally streaming fragments through ``on_token``."""
        if not messages:
            raise LLMError("generate() requires at least one message")
        self.load()
        assert self._tokenizer is not None and self._model is not None  # for type checkers

        import torch

        prompt = self._build_prompt(messages)
        new_tokens = int(max_new_tokens or self.settings.llm_max_new_tokens)
        sampling = self.settings.llm_temperature if temperature is None else float(temperature)
        inputs = self._encode(prompt, new_tokens)

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": new_tokens,
            "do_sample": sampling > 0,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if sampling > 0:
            generation_kwargs.update(
                temperature=max(sampling, 1e-5),
                top_p=self.settings.llm_top_p,
                repetition_penalty=self.settings.llm_repetition_penalty,
            )

        try:
            with self._generate_lock:
                if on_token is None:
                    text = self._generate_blocking(inputs, generation_kwargs)
                else:
                    text = self._generate_streaming(inputs, generation_kwargs, on_token)
        except Exception as exc:
            if "out of memory" in str(exc).lower():
                raise LLMError(
                    "GPU out of memory while generating. Reduce NEET_LLM_MAX_NEW_TOKENS, "
                    "enable NEET_LLM_LOAD_IN_4BIT=true, or switch to a smaller model "
                    "(e.g. Qwen/Qwen2.5-3B-Instruct)."
                ) from exc
            raise LLMError(f"Generation failed: {exc}") from exc

        return self._postprocess(text)

    def _generate_blocking(self, inputs: dict[str, Any], kwargs: dict[str, Any]) -> str:
        import torch

        with torch.inference_mode():
            output = self._model.generate(**inputs, **kwargs)
        prompt_length = inputs["input_ids"].shape[-1]
        generated = output[0][prompt_length:]
        return self._tokenizer.decode(generated, skip_special_tokens=True)

    def _generate_streaming(
        self,
        inputs: dict[str, Any],
        kwargs: dict[str, Any],
        on_token: TokenCallback,
    ) -> str:
        import torch
        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=600.0,
        )
        errors: list[BaseException] = []

        def _worker() -> None:
            try:
                with torch.inference_mode():
                    self._model.generate(**inputs, streamer=streamer, **kwargs)
            except BaseException as exc:  # noqa: BLE001 - forwarded to the caller
                errors.append(exc)
                end = getattr(streamer, "end", None)
                if callable(end):
                    end()

        thread = threading.Thread(target=_worker, name="neet-hf-stream", daemon=True)
        thread.start()

        pieces: list[str] = []
        for fragment in streamer:
            pieces.append(fragment)
            try:
                on_token(fragment)
            except Exception:  # a broken UI must not abort generation
                logger.debug("Token callback raised; continuing generation", exc_info=True)

        thread.join(timeout=5.0)
        if errors:
            raise errors[0]
        return "".join(pieces)

    # ------------------------------------------------------------------ helpers
    def _build_prompt(self, messages: Sequence[ChatMessage]) -> str:
        """Render messages with the model's chat template (ChatML fallback)."""
        try:
            return self._tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as exc:  # templates missing on exotic checkpoints
            logger.warning("apply_chat_template failed (%s); using ChatML fallback", exc)
            rendered = "".join(
                f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n" for message in messages
            )
            return rendered + "<|im_start|>assistant\n"

    def _encode(self, prompt: str, max_new_tokens: int) -> dict[str, Any]:
        """Tokenise and truncate the prompt to the configured input budget."""
        import torch

        token_ids: list[int] = self._tokenizer(prompt, add_special_tokens=False)["input_ids"]
        budget = max(256, self.settings.llm_max_input_tokens)
        original_length = len(token_ids)
        if original_length > budget:
            head = min(256, budget // 4)  # keep the instruction header intact
            tail = budget - head
            token_ids = token_ids[:head] + token_ids[-tail:] if tail > 0 else token_ids[:budget]
            logger.warning("Prompt truncated from %d to %d tokens", original_length, len(token_ids))

        tensor = torch.tensor([token_ids], dtype=torch.long)
        device = self._input_device()
        return {
            "input_ids": tensor.to(device),
            "attention_mask": torch.ones_like(tensor, dtype=torch.long).to(device),
        }

    def _input_device(self) -> Any:
        """Where to place input tensors (handles accelerate sharding)."""
        try:
            return self._model.device
        except Exception:  # pragma: no cover - sharded models expose device_map only
            return self.device

    def _postprocess(self, text: str) -> str:
        """Remove special tokens and stray role prefixes from the raw output."""
        cleaned = text or ""
        for token in _SPECIAL_TOKENS:
            cleaned = cleaned.replace(token, "")
        cleaned = cleaned.strip()
        for prefix in ("assistant\n", "assistant:", "Assistant:", "assistant "):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :].lstrip()
        return cleaned.strip()

    # ------------------------------------------------------------------- state
    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "model": self.model_id,
            "device": self.device,
            "dtype": str(self.dtype).replace("torch.", "") if self.dtype is not None else "unknown",
            "quantized_4bit": self._quantized,
            "loaded": self.is_loaded,
            "load_seconds": round(self._load_seconds, 1) if self._load_seconds else None,
        }

    def unload(self) -> None:
        """Free model memory (useful before switching checkpoints)."""
        with self._load_lock:
            self._model = None
            self._tokenizer = None
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # pragma: no cover
                pass
            logger.info("LLM unloaded")
