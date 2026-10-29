"""Tests for NCERT-aware, chapter-wise ingestion.

Covers the whole chain the feature promises:

* **filename codes** (``kebo101.pdf`` -> Biology / Class 11 / chapter 1) and
  descriptive names;
* **chapter-title extraction** from a PDF's text layer, with the Biology
  catalogue as a fallback;
* the **loader** attaching ``ncert_*`` metadata while leaving non-NCERT files
  exactly as before, and recording per-file failures;
* the **end-to-end pipeline** building a real (fake-embedding) FAISS index and
  retrieving grounded chunks that carry their chapter metadata.

The sample PDFs are generated in-process by :func:`make_pdf`, which writes a
minimal but valid single-page PDF with a real text layer - no OCR, no network,
no fixture binary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from config import Settings
from rag.chunker import DocumentChunker
from rag.document_loader import PDFDocumentLoader, collect_ncert_chapters
from rag.ingest import IngestionPipeline
from rag.ncert import (
    build_ncert_metadata,
    extract_chapter_title,
    is_ncert_file,
    parse_ncert_filename,
)
from rag.vector_store import VectorStoreManager
from utils.constants import Subject

# ---------------------------------------------------------------------------
# Sample-PDF builder (stdlib + pypdf-readable; no OCR)
# ---------------------------------------------------------------------------
def make_pdf(lines: Sequence[str]) -> bytes:
    """Assemble a valid one-page PDF whose text layer is ``lines`` joined by NL."""

    def escape(text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    parts = ["BT", "/F1 12 Tf", "14 TL", "72 720 Td"]
    for index, line in enumerate(lines):
        token = f"({escape(line)}) Tj"
        parts.append(token if index == 0 else f"T* {token}")
    parts.append("ET")
    stream = "\n".join(parts).encode("latin-1")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 5 0 R >> >> "
        b"/MediaBox [0 0 612 792] /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"

    xref = len(out)
    size = len(objects) + 1
    out += f"xref\n0 {size}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    )
    return bytes(out)


def write_pdf(path: Path, lines: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(make_pdf(lines))
    return path


LIVING_WORLD = [
    "Chapter 1",
    "THE LIVING WORLD",
    "Living organisms are incredibly diverse and their number is constantly "
    "increasing. We are familiar with about 1.7 million species presently known.",
]
PLANT_KINGDOM_BODY = [
    "The plant kingdom includes algae, bryophytes, pteridophytes, gymnosperms "
    "and angiosperms, arranged by increasingly complex structure and reproduction.",
]

# ---------------------------------------------------------------------------
# 1. Filename-code parsing
# ---------------------------------------------------------------------------
def test_parse_class11_biology_code() -> None:
    subject, klass, chapter, code = parse_ncert_filename("kebo101")
    assert subject is Subject.BIOLOGY
    assert klass == "11"
    assert chapter == 1
    assert code == "kebo101"


def test_parse_next_chapter_and_class12() -> None:
    assert parse_ncert_filename("kebo102")[2] == 2
    subject, klass, chapter, _ = parse_ncert_filename("cebo201")
    assert (subject, klass, chapter) == (Subject.BIOLOGY, "12", 1)


def test_parse_physics_code() -> None:
    subject, klass, chapter, _ = parse_ncert_filename("keph108")
    assert (subject, klass, chapter) == (Subject.PHYSICS, "11", 8)


def test_parse_descriptive_name() -> None:
    subject, klass, chapter, code = parse_ncert_filename(
        "Class-11-Biology-Chapter-5-Plant-Kingdom"
    )
    assert subject is Subject.BIOLOGY
    assert klass == "11"
    assert chapter == 5
    assert code is None


def test_parse_plain_name_is_not_ncert() -> None:
    assert parse_ncert_filename("random_notes") is None
    assert is_ncert_file(Path("data/biology/random_notes.pdf")) is False
    assert is_ncert_file(Path("data/biology/kebo101.pdf")) is True


# ---------------------------------------------------------------------------
# 2. Title extraction (content first, catalogue second)
# ---------------------------------------------------------------------------
def test_extract_title_from_chapter_then_heading_line() -> None:
    title = extract_chapter_title(
        ["\n".join(LIVING_WORLD)], subject=Subject.BIOLOGY, klass="11", chapter_number=1
    )
    assert title == "The Living World"


def test_extract_title_inline_colon() -> None:
    title = extract_chapter_title(
        ["Chapter 2: Biological Classification and its five kingdoms."],
        subject=Subject.BIOLOGY,
        klass="11",
        chapter_number=2,
    )
    assert title == "Biological Classification and its five kingdoms"


def test_extract_title_falls_back_to_biology_catalogue() -> None:
    title = extract_chapter_title(
        PLANT_KINGDOM_BODY, subject=Subject.BIOLOGY, klass="11", chapter_number=3
    )
    assert title == "Plant Kingdom"


# ---------------------------------------------------------------------------
# 3. Metadata shape (the exact contract required)
# ---------------------------------------------------------------------------
def test_build_metadata_exact_shape() -> None:
    meta = build_ncert_metadata(Path("biology/kebo101.pdf"), ["\n".join(LIVING_WORLD)])
    assert meta is not None
    assert meta.as_dict() == {
        "subject": "Biology",
        "class": "11",
        "source": "NCERT",
        "chapter": "The Living World",
        "file": "kebo101.pdf",
    }


def test_build_metadata_returns_none_for_non_ncert() -> None:
    assert build_ncert_metadata(Path("misc/notes.pdf"), ["just some prose here"]) is None


# ---------------------------------------------------------------------------
# 4. Loader integration
# ---------------------------------------------------------------------------
def test_loader_attaches_ncert_metadata(settings: Settings) -> None:
    write_pdf(settings.data_dir / "biology" / "kebo101.pdf", LIVING_WORLD)
    documents = PDFDocumentLoader(settings).load_file(
        settings.data_dir / "biology" / "kebo101.pdf"
    )

    assert documents, "the sample PDF should produce page documents"
    metadata = documents[0].metadata
    assert metadata["ncert"] is True
    assert metadata["class"] == "11"
    assert metadata["chapter"] == "The Living World"
    assert metadata["ncert_subject"] == "Biology"
    assert metadata["book_code"] == "kebo101"
    # the retrieval-critical keys are untouched (backward compatible)
    assert metadata["subject"] == "biology"
    assert metadata["source"] == "biology/kebo101.pdf"


def test_loader_leaves_non_ncert_files_untouched(settings: Settings) -> None:
    path = write_pdf(settings.data_dir / "biology" / "misc_notes.pdf", PLANT_KINGDOM_BODY)
    metadata = PDFDocumentLoader(settings).load_file(path)[0].metadata

    assert "ncert" not in metadata
    assert "chapter" not in metadata


def test_collect_ncert_chapters_from_pages(settings: Settings) -> None:
    loader = PDFDocumentLoader(settings)
    write_pdf(settings.data_dir / "biology" / "kebo101.pdf", LIVING_WORLD)
    write_pdf(settings.data_dir / "biology" / "kebo103.pdf", PLANT_KINGDOM_BODY)
    documents = loader.load_all()

    chapters = collect_ncert_chapters(documents)
    by_file = {chapter["file"]: chapter for chapter in chapters}
    assert set(by_file) == {"kebo101.pdf", "kebo103.pdf"}
    assert by_file["kebo101.pdf"] == {
        "subject": "Biology",
        "class": "11",
        "source": "NCERT",
        "chapter": "The Living World",
        "file": "kebo101.pdf",
    }
    assert by_file["kebo103.pdf"]["chapter"] == "Plant Kingdom"


def test_loader_records_corrupt_file_as_failure(settings: Settings) -> None:
    good = write_pdf(settings.data_dir / "biology" / "kebo101.pdf", LIVING_WORLD)
    broken = settings.data_dir / "biology" / "kebo102.pdf"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_bytes(b"this is not a real PDF at all")

    loader = PDFDocumentLoader(settings)
    documents = loader.load_all()

    assert documents  # the good file still loaded
    failures = {failure["file"] for failure in loader.failures}
    assert "kebo102.pdf" in failures
    assert good.name == "kebo101.pdf"


# ---------------------------------------------------------------------------
# 5. End-to-end ingestion through the (unchanged) vector store
# ---------------------------------------------------------------------------
def test_ingestion_pipeline_indexes_and_retrieves_chapters(
    settings: Settings, fake_embeddings: DeterministicFakeEmbedding
) -> None:
    write_pdf(settings.data_dir / "biology" / "kebo101.pdf", LIVING_WORLD)
    write_pdf(settings.data_dir / "biology" / "kebo103.pdf", PLANT_KINGDOM_BODY)

    store = VectorStoreManager(settings, embeddings_factory=lambda: fake_embeddings)
    pipeline = IngestionPipeline(
        settings,
        vector_store=store,
        loader=PDFDocumentLoader(settings),
        chunker=DocumentChunker(settings),
    )
    report = pipeline.run(rebuild=True)

    assert report.files == 2
    assert report.chunks > 0
    assert report.chapter_count == 2
    assert not report.failures
    assert {c["chapter"] for c in report.chapters} == {"The Living World", "Plant Kingdom"}
    assert store.is_ready()

    results = store.search("diversity of living organisms", subject="biology", k=3)
    assert results
    # every retrieved chunk carries its NCERT chapter identity, so the answer
    # can cite "The Living World, Class 11" rather than a bare file name.
    assert all(chunk.metadata.get("ncert") is True for chunk in results)


def test_ingestion_sidecar_records_chapters(
    settings: Settings, fake_embeddings: DeterministicFakeEmbedding
) -> None:
    from rag.vector_store import read_index_metadata

    write_pdf(settings.data_dir / "biology" / "kebo101.pdf", LIVING_WORLD)
    pipeline = IngestionPipeline(
        settings,
        vector_store=VectorStoreManager(settings, embeddings_factory=lambda: fake_embeddings),
    )
    pipeline.run(rebuild=True)

    payload = read_index_metadata(settings)
    assert payload is not None
    assert payload["chapters"][0]["file"] == "kebo101.pdf"
    assert payload["chapters"][0]["subject"] == "Biology"
    assert payload["last_run"]["chapters"] == 1
