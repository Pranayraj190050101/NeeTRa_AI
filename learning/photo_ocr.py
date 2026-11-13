"""Local OCR for question photos uploaded to Ask Genie."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from utils.text_utils import clean_text

__all__ = ["LocalImageSolver"]

MAX_IMAGE_BYTES = 20 * 1024 * 1024


def _build_ocr_engine() -> Any:
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR(max_side_len=1600, intra_op_num_threads=1)


class LocalImageSolver:
    """Extract editable question text from a photo without a hosted vision API."""

    def __init__(self, engine_factory: Callable[[], Any] = _build_ocr_engine) -> None:
        self._engine_factory = engine_factory
        self._engine: Any | None = None
        self._lock = threading.Lock()

    def extract_question(self, image: bytes, *, subject: str | None = None) -> str:
        if not image:
            raise ValueError("The uploaded image is empty.")
        if len(image) > MAX_IMAGE_BYTES:
            raise ValueError("The uploaded image exceeds the 20 MB limit.")

        with self._lock:
            if self._engine is None:
                self._engine = self._engine_factory()
            results, _ = self._engine(image)

        lines = [
            str(item[1]).strip()
            for item in (results or [])
            if len(item) > 1 and str(item[1]).strip()
        ]
        return clean_text("\n".join(lines))
