"""Logging setup for NEETra AI.

A single call to :func:`setup_logging` configures the root logger with a
console handler plus a rotating file handler.  Modules then simply use
:func:`get_logger` and never touch handler configuration themselves.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:  # pragma: no cover - typing only
    from config import Settings

__all__ = ["get_logger", "setup_logging", "LOG_FORMAT"]

LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"

# Third-party libraries that spam INFO logs during model loading.
_NOISY_LOGGERS: Final[tuple[str, ...]] = (
    "httpx",
    "httpcore",
    "urllib3",
    "filelock",
    "huggingface_hub",
    "sentence_transformers",
    "transformers",
    "faiss",
    "matplotlib",
    "asyncio",
)

_CONFIGURED: bool = False


def setup_logging(settings: "Settings", *, force: bool = False) -> None:
    """Configure process-wide logging (idempotent).

    Args:
        settings: Application settings providing the level, log directory and
            whether file logging is enabled.
        force: Rebuild the handlers even if logging was already configured.
    """
    global _CONFIGURED
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    root = logging.getLogger()

    if _CONFIGURED and not force:
        root.setLevel(level)
        return

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    handlers: list[logging.Handler] = []

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    handlers.append(console)

    if settings.log_to_file:
        try:
            Path(settings.log_dir).mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                filename=Path(settings.log_dir) / "neet_tutor.log",
                maxBytes=5 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)
        except OSError as exc:  # pragma: no cover - depends on the file system
            root.warning("File logging disabled (%s)", exc)

    # Remove handlers installed by a previous call so `force=True` is clean.
    for handler in list(root.handlers):
        if getattr(handler, "_neet_handler", False):
            root.removeHandler(handler)
            handler.close()

    for handler in handlers:
        handler._neet_handler = True  # type: ignore[attr-defined]
        root.addHandler(handler)

    root.setLevel(level)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    _CONFIGURED = True
    logging.getLogger(__name__).debug("Logging configured (level=%s)", settings.log_level)


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger without mutating global logging state."""
    return logging.getLogger(name)
