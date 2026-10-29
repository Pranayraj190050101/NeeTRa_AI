"""Tests for the data/ -> Document loader (text files; PDFs need no fixture)."""

from __future__ import annotations

from pathlib import Path

from config import Settings
from rag.document_loader import PDFDocumentLoader, group_by_subject, metadata_source

BIOLOGY_NOTE = (
    "The mitochondrion is the site of aerobic respiration. "
    "Pyruvate is oxidised in the matrix and the electron transport chain sits "
    "on the inner membrane, producing a proton gradient that drives ATP synthase."
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_text_file_with_subject_and_metadata(settings: Settings) -> None:
    path = _write(settings.data_dir / "biology" / "respiration_notes.txt", BIOLOGY_NOTE)
    loader = PDFDocumentLoader(settings)

    documents = loader.load_file(path)

    assert len(documents) == 1
    metadata = documents[0].metadata
    assert metadata["subject"] == "biology"
    assert metadata["page"] == 1
    assert metadata["file_name"] == "respiration_notes.txt"
    assert metadata["source"] == "biology/respiration_notes.txt"
    assert metadata["doc_id"]
    assert metadata["char_count"] == len(documents[0].page_content)


def test_subject_falls_back_to_file_name_then_general(settings: Settings) -> None:
    named = _write(settings.data_dir / "misc" / "chemistry_basics.md", BIOLOGY_NOTE)
    unknown = _write(settings.data_dir / "misc" / "random_notes.md", BIOLOGY_NOTE)
    loader = PDFDocumentLoader(settings)

    assert loader.load_file(named)[0].metadata["subject"] == "chemistry"
    assert loader.load_file(unknown)[0].metadata["subject"] == "general"


def test_near_empty_pages_are_skipped(settings: Settings) -> None:
    path = _write(settings.data_dir / "physics" / "blank.txt", "   \n \n")
    loader = PDFDocumentLoader(settings)

    assert loader.load_file(path) == []


def test_discover_is_sorted_and_filters_suffixes(settings: Settings) -> None:
    _write(settings.data_dir / "physics" / "b_notes.txt", BIOLOGY_NOTE)
    _write(settings.data_dir / "physics" / "a_notes.md", BIOLOGY_NOTE)
    _write(settings.data_dir / "physics" / "ignored.csv", "a,b\n1,2\n")
    loader = PDFDocumentLoader(settings)

    names = [path.name for path in loader.discover()]

    assert names == ["a_notes.md", "b_notes.txt"]


def test_doc_id_is_stable_across_reloads(settings: Settings) -> None:
    path = _write(settings.data_dir / "chemistry" / "mole.txt", BIOLOGY_NOTE)
    loader = PDFDocumentLoader(settings)

    first = loader.load_file(path)[0].metadata["doc_id"]
    second = PDFDocumentLoader(settings).load_file(path)[0].metadata["doc_id"]

    assert first == second


def test_group_by_subject_counts_metadata(settings: Settings) -> None:
    loader = PDFDocumentLoader(settings)
    documents = loader.load_file(_write(settings.data_dir / "biology" / "a.txt", BIOLOGY_NOTE))
    documents += loader.load_file(_write(settings.data_dir / "biology" / "b.txt", BIOLOGY_NOTE + " Extra."))
    documents += loader.load_file(_write(settings.data_dir / "physics" / "c.txt", BIOLOGY_NOTE))

    assert group_by_subject(documents) == {"biology": 2, "physics": 1}


def test_metadata_source_outside_data_dir(settings: Settings) -> None:
    outside = Path(settings.data_dir).parent / "loose.pdf"

    assert metadata_source(outside, settings.data_dir) == "loose.pdf"
