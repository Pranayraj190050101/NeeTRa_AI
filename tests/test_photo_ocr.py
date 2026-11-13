from __future__ import annotations

import pytest

from learning.photo_ocr import LocalImageSolver, MAX_IMAGE_BYTES


class FakeOCR:
    def __call__(self, _image: bytes):
        return (
            [
                [None, "A body starts from rest."],
                [None, "Find its acceleration."],
            ],
            [],
        )


def test_local_image_solver_extracts_question_text() -> None:
    solver = LocalImageSolver(engine_factory=FakeOCR)

    assert solver.extract_question(b"image bytes") == (
        "A body starts from rest.\nFind its acceleration."
    )


def test_local_image_solver_rejects_empty_and_oversized_images() -> None:
    solver = LocalImageSolver(engine_factory=FakeOCR)

    with pytest.raises(ValueError, match="empty"):
        solver.extract_question(b"")
    with pytest.raises(ValueError, match="20 MB"):
        solver.extract_question(b"x" * (MAX_IMAGE_BYTES + 1))
