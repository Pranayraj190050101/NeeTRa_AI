"""Device and dtype resolution for PyTorch based components.

Both the embedding model and the LLM accept ``auto``; this module turns that
preference into a concrete device string and a ``torch.dtype`` while tolerating
machines without CUDA or without ``torch`` installed at all (the config layer
must stay importable in lightweight tooling).
"""

from __future__ import annotations

from typing import Any, Final

from config import DTypeName, DevicePreference
from utils.logging_utils import get_logger

__all__ = ["resolve_device", "resolve_torch_dtype", "cuda_is_available", "gpu_summary"]

logger = get_logger(__name__)


def cuda_is_available() -> bool:
    """Return ``True`` when a CUDA device is usable by PyTorch."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - torch missing or broken install
        return False


def resolve_device(preference: DevicePreference | str = "auto") -> str:
    """Map ``auto`` to the best available accelerator, defaulting to CPU."""
    value = (preference or "auto").lower()
    if value == "auto":
        if cuda_is_available():
            return "cuda"
        try:
            import torch

            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except Exception:  # pragma: no cover
            pass
        return "cpu"
    if value.startswith("cuda") and not cuda_is_available():
        logger.warning("CUDA requested but unavailable - falling back to CPU")
        return "cpu"
    return value


def resolve_torch_dtype(preference: DTypeName | str, device: str) -> Any:
    """Pick a compute dtype that is valid for ``device``.

    * ``auto`` -> bfloat16 on Ampere+ GPUs, float16 on other CUDA GPUs, float32
      on CPU/MPS.
    * Explicit requests that are unsupported degrade gracefully with a warning.
    """
    try:
        import torch
    except Exception:  # pragma: no cover - torch is a hard requirement in prod
        return None

    value = (preference or "auto").lower()
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    if device.startswith("cuda"):
        if value in mapping:
            return mapping[value]
        # bfloat16 requires Ampere (compute capability >= 8.0).
        try:
            major, _ = torch.cuda.get_device_capability(0)
            return torch.bfloat16 if major >= 8 else torch.float16
        except Exception:  # pragma: no cover
            return torch.float16

    if value == "float32":
        return torch.float32
    if value != "auto":
        logger.warning("dtype=%s is not supported on %s - using float32", value, device)
    return torch.float32


# Sensible defaults per GPU memory size, used by the UI health panel.
_VRAM_HINTS: Final[tuple[tuple[int, str], ...]] = (
    (10, "Qwen/Qwen2.5-1.5B-Instruct"),
    (16, "Qwen/Qwen2.5-3B-Instruct"),
    (24, "Qwen/Qwen2.5-7B-Instruct (4-bit)"),
    (40, "Qwen/Qwen2.5-7B-Instruct"),
)


def gpu_summary() -> dict[str, Any]:
    """Lightweight hardware description for logs and the UI."""
    summary: dict[str, Any] = {"cuda": cuda_is_available(), "devices": []}
    try:
        import torch

        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                vram_gb = round(props.total_memory / (1024**3), 1)
                summary["devices"].append({"name": props.name, "vram_gb": vram_gb})
            total = max((device["vram_gb"] for device in summary["devices"]), default=0)
            summary["suggested_model"] = next(
                (model for threshold, model in _VRAM_HINTS if total < threshold),
                "Qwen/Qwen2.5-7B-Instruct",
            )
    except Exception:  # pragma: no cover - torch missing
        pass
    return summary
