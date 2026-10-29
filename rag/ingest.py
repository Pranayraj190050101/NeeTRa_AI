"""End-to-end ingestion pipeline and command line interface.

Usage
-----
Rebuild the index from everything under ``data/``::

    python -m rag.ingest --rebuild

Add only new pages (idempotent thanks to ``doc_id`` de-duplication)::

    python -m rag.ingest --append

Custom corpus location or chunking::

    python -m rag.ingest --source data/physics --chunk-size 1000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

# Allow ``python rag/ingest.py`` in addition to ``python -m rag.ingest``.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - import-time plumbing
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Settings, get_settings  # noqa: E402
from rag.chunker import DocumentChunker  # noqa: E402
from rag.document_loader import PDFDocumentLoader, collect_ncert_chapters  # noqa: E402
from rag.vector_store import (  # noqa: E402
    VectorStoreError,
    VectorStoreManager,
    summarise_chunks,
    write_index_metadata,
)
from utils.logging_utils import get_logger, setup_logging  # noqa: E402
from utils.text_utils import format_seconds  # noqa: E402

__all__ = ["IngestionPipeline", "IngestionReport", "run_ingestion", "write_ingestion_report", "main"]

logger = get_logger(__name__)


@dataclass(slots=True)
class IngestionReport:
    """Result of a single ingestion run."""

    files: int = 0
    pages: int = 0
    chunks: int = 0
    added: int = 0
    skipped: int = 0
    subjects: dict[str, int] = field(default_factory=dict)
    chapters: list[dict[str, object]] = field(default_factory=list)
    failures: list[dict[str, str]] = field(default_factory=list)
    duration_s: float = 0.0
    index_dir: str = ""
    rebuilt: bool = True

    @property
    def chapter_count(self) -> int:
        return len(self.chapters)

    def as_dict(self) -> dict[str, object]:
        return {
            "files": self.files,
            "pages": self.pages,
            "chunks": self.chunks,
            "added": self.added,
            "skipped": self.skipped,
            "subjects": self.subjects,
            "chapters": self.chapter_count,
            "chapter_list": self.chapters,
            "failures": self.failures,
            "duration_s": round(self.duration_s, 2),
            "index_dir": self.index_dir,
            "rebuilt": self.rebuilt,
        }

    def pretty(self) -> str:
        lines = [
            "",
            "=" * 62,
            "NEETra AI - ingestion complete",
            "=" * 62,
            f"  Mode          : {'rebuild' if self.rebuilt else 'append'}",
            f"  Source files  : {self.files}",
            f"  Pages parsed  : {self.pages}",
            f"  Chunks built  : {self.chunks}",
            f"  Chunks added  : {self.added}",
            f"  Duplicates    : {self.skipped}",
            f"  NCERT chapters: {self.chapter_count}",
            f"  Failures      : {len(self.failures)}",
            f"  Subjects      : {self.subjects or {'general': 0}}",
            f"  Duration      : {format_seconds(self.duration_s)}",
            f"  Index folder  : {self.index_dir}",
            "=" * 62,
            "",
        ]
        return "\n".join(lines)

    def markdown(self) -> str:
        """Render the ingestion report as Markdown (files/chapters/chunks/failures)."""
        lines = [
            "# NCERT Ingestion Report",
            "",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Mode: **{'rebuild' if self.rebuilt else 'append'}**  ",
            f"Index: `{self.index_dir}`",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Files processed | {self.files} |",
            f"| Pages parsed | {self.pages} |",
            f"| Chunks created | {self.chunks} |",
            f"| Chunks added | {self.added} |",
            f"| Duplicates skipped | {self.skipped} |",
            f"| Chapters found | {self.chapter_count} |",
            f"| Failures | {len(self.failures)} |",
            f"| Duration | {format_seconds(self.duration_s)} |",
            "",
            "## Chapters",
            "",
        ]
        if self.chapters:
            lines += ["| Subject | Class | Chapter | File |", "|---|---|---|---|"]
            for chapter in self.chapters:
                lines.append(
                    f"| {chapter['subject']} | {chapter['class']} | "
                    f"{chapter['chapter']} | `{chapter['file']}` |"
                )
        else:
            lines.append("_No NCERT chapters were recognised in this corpus._")
        lines += ["", "## Failures", ""]
        if self.failures:
            lines += ["| File | Error |", "|---|---|"]
            for failure in self.failures:
                error = str(failure.get("error", "")).replace("|", "\\|").replace("\n", " ")
                lines.append(f"| `{failure.get('file', '')}` | {error} |")
        else:
            lines.append("_None - every file ingested cleanly._")
        lines.append("")
        return "\n".join(lines)


class IngestionPipeline:
    """Load -> chunk -> embed -> persist, in one testable object."""

    def __init__(
        self,
        settings: Settings,
        *,
        vector_store: VectorStoreManager | None = None,
        loader: PDFDocumentLoader | None = None,
        chunker: DocumentChunker | None = None,
    ) -> None:
        self.settings = settings
        self.loader = loader or PDFDocumentLoader(settings)
        self.chunker = chunker or DocumentChunker(settings)
        self.vector_store = vector_store or VectorStoreManager(settings)

    def run(
        self,
        *,
        rebuild: bool = True,
        source_dir: Path | None = None,
    ) -> IngestionReport:
        """Execute the pipeline and persist the index.

        Args:
            rebuild: Replace the existing index when ``True``; otherwise append
                only chunks whose ``doc_id`` is not indexed yet.
            source_dir: Optional override for the corpus folder.
        """
        started = time.perf_counter()
        if source_dir is not None:
            self.loader.data_dir = Path(source_dir)

        report = IngestionReport(index_dir=str(self.settings.index_dir), rebuilt=rebuild)
        files = self.loader.discover()
        report.files = len(files)
        if not files:
            raise FileNotFoundError(
                f"No PDF/TXT/MD files found under {self.loader.data_dir}. "
                "Place your NEET material in data/physics, data/chemistry or data/biology."
            )

        pages = self.loader.load_all()
        report.pages = len(pages)
        report.failures = list(getattr(self.loader, "failures", []))
        if not pages:
            raise ValueError(
                "No extractable text found. The PDFs may be scanned images; "
                "run OCR (e.g. ocrmypdf) before ingesting."
            )

        chunks = self.chunker.split(pages)
        report.chunks = len(chunks)
        report.subjects = summarise_chunks(chunks)
        report.chapters = collect_ncert_chapters(pages)
        if not chunks:
            raise ValueError("Chunking produced zero chunks - check chunk_size/min_chunk_chars.")

        if rebuild or not self.vector_store.is_ready():
            self.vector_store.build(chunks, persist=True)
            report.added = len(chunks)
        else:
            existing = self.vector_store.document_ids()
            fresh = [chunk for chunk in chunks if str(chunk.metadata.get("doc_id")) not in existing]
            report.skipped = len(chunks) - len(fresh)
            report.added = self.vector_store.add(fresh, persist=True) if fresh else 0
            if not fresh:
                logger.info("Nothing new to index - all %d chunk(s) already present", len(chunks))

        report.duration_s = time.perf_counter() - started
        write_index_metadata(
            self.settings,
            chunks=report.added if rebuild else len(self.vector_store.document_ids()),
            files=report.files,
            subjects=report.subjects,
            extra={
                # Source paths are recorded so the chapter catalog can be read
                # straight from this sidecar, without loading the FAISS index.
                "sources": sorted(
                    {
                        str(chunk.metadata.get("source") or chunk.metadata.get("file_name") or "unknown")
                        for chunk in chunks
                    }
                ),
                # NCERT chapter hierarchy (subject / class / title / file) so the
                # Learn + Practice modules can list real chapters offline.
                "chapters": report.chapters,
                "last_run": {
                    "mode": "rebuild" if rebuild else "append",
                    "pages": report.pages,
                    "chunks_built": report.chunks,
                    "chunks_added": report.added,
                    "duplicates_skipped": report.skipped,
                    "chapters": report.chapter_count,
                    "failures": len(report.failures),
                    "duration_s": round(report.duration_s, 2),
                }
            },
        )
        logger.info("Ingestion finished: %s", report.as_dict())
        return report


def run_ingestion(
    settings: Settings,
    *,
    rebuild: bool = True,
    source_dir: Path | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> IngestionReport:
    """NCERT-aware ingestion in one call - shared by ``--ingest`` and the script.

    Builds (or appends to) the FAISS index exactly as :class:`IngestionPipeline`
    does, but also resolves chapter metadata so callers receive a report of
    files, chapters, chunks and failures without touching the internals.
    """
    chunk_kwargs: dict[str, int] = {}
    if chunk_size:
        chunk_kwargs["chunk_size"] = chunk_size
    if chunk_overlap is not None:
        chunk_kwargs["chunk_overlap"] = chunk_overlap
    pipeline = IngestionPipeline(settings, chunker=DocumentChunker(settings, **chunk_kwargs))
    return pipeline.run(rebuild=rebuild, source_dir=source_dir)


def write_ingestion_report(
    report: IngestionReport,
    *,
    markdown_path: Path | None = None,
    json_path: Path | None = None,
) -> None:
    """Persist the ingestion report as Markdown and/or JSON."""
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(report.markdown(), encoding="utf-8")
        logger.info("Wrote ingestion report to %s", markdown_path)
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report.as_dict(), indent=2, default=str), encoding="utf-8")
        logger.info("Wrote ingestion JSON to %s", json_path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m rag.ingest",
        description="Build the NEETra AI FAISS index from PDFs.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--rebuild", action="store_true", help="Recreate the index (default).")
    mode.add_argument("--append", action="store_true", help="Add only new pages to the index.")
    parser.add_argument("--source", type=Path, default=None, help="Override the corpus folder.")
    parser.add_argument("--chunk-size", type=int, default=None, help="Characters per chunk.")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="Chunk overlap in characters.")
    parser.add_argument("--report", type=Path, default=None, help="Write a Markdown ingestion report here.")
    parser.add_argument("--json", type=Path, default=None, help="Write a JSON ingestion report here.")
    parser.add_argument("--log-level", default=None, help="Override NEET_LOG_LEVEL.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""
    args = _parse_args(argv)
    settings = get_settings()
    if args.log_level:
        settings = settings.model_copy(update={"log_level": args.log_level.upper()})
    settings.ensure_directories()
    setup_logging(settings)

    try:
        report = run_ingestion(
            settings,
            rebuild=not args.append,
            source_dir=args.source,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    except (FileNotFoundError, ValueError, VectorStoreError) as exc:
        logger.error("Ingestion failed: %s", exc)
        print(f"\n[ERROR] {exc}\n", file=sys.stderr)
        return 1
    except Exception as exc:  # unexpected - keep the traceback for debugging
        logger.exception("Unexpected ingestion failure")
        print(f"\n[ERROR] {exc}\n", file=sys.stderr)
        return 2

    write_ingestion_report(report, markdown_path=args.report, json_path=args.json)
    print(report.pretty())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
