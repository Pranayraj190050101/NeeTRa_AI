"""Chapter-aware NCERT ingestion command.

Build the retrieval index from chapter-wise NCERT PDFs and write an ingestion
report.  This is the standalone entry point equivalent to ``python app.py
--ingest``; both delegate to :func:`rag.ingest.run_ingestion`, which reuses the
existing FAISS :class:`~rag.vector_store.VectorStoreManager` untouched.

Usage::

    python scripts/ingest_kb.py                       # rebuild from data/
    python scripts/ingest_kb.py --source data/biology # one subject folder
    python scripts/ingest_kb.py --append              # index only new chapters
    python scripts/ingest_kb.py --report out.md --json out.json

The report lists files processed, NCERT chapters found (subject / class /
title / file), chunks created and any per-file failures.  A ``.txt``/``.md``
corpus works too, but real ingestion needs the PDF/embedding dependencies
listed in ``requirements.txt`` installed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a plain script from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_settings  # noqa: E402
from rag.ingest import (  # noqa: E402
    VectorStoreError,
    run_ingestion,
    write_ingestion_report,
)
from utils.logging_utils import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)

DEFAULT_REPORT = PROJECT_ROOT / "KB_INGESTION_REPORT.md"
DEFAULT_JSON = PROJECT_ROOT / "KB_INGESTION_REPORT.json"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python scripts/ingest_kb.py",
        description="Ingest chapter-wise NCERT PDFs into the NEETra AI knowledge base.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--rebuild", action="store_true", help="Recreate the index (default).")
    mode.add_argument("--append", action="store_true", help="Index only new chapters.")
    parser.add_argument("--source", type=Path, default=None, help="Corpus folder (default: data/).")
    parser.add_argument("--chunk-size", type=int, default=None, help="Characters per chunk.")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="Chunk overlap in characters.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="Markdown report path.")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON, help="JSON report path.")
    parser.add_argument("--no-report", action="store_true", help="Skip writing report files.")
    parser.add_argument("--log-level", default=None, help="Override NEET_LOG_LEVEL.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
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

    if not args.no_report:
        write_ingestion_report(report, markdown_path=args.report, json_path=args.json)

    print(report.pretty())
    if report.failures:
        print(f"[WARN] {len(report.failures)} file(s) failed; see the report.\n", file=sys.stderr)
    if not args.no_report:
        print(f"Report: {args.report}\nJSON:   {args.json}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
