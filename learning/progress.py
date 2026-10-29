"""Local progress persistence - a JSON store today, a database later.

The platform must remember what a student has attempted, scored and got wrong,
but it should not need a server or a database engine to do that.  This module
keeps one JSON document per student profile on disk
(``progress/<profile>.json`` by default) behind a small, storage-shaped API:

* :meth:`ProgressStore.record` appends one graded test,
* :meth:`ProgressStore.stats` aggregates it for the Dashboard and Progress tab,
* :meth:`ProgressStore.weak_areas` ranks topics and chapters for the Revision
  Hub and the future Weakness agent.

Writes are atomic (temp file + ``os.replace``) and guarded by a lock, so a
Gradio reload or two concurrent tabs cannot corrupt the history.  Swapping the
backend later only means implementing the four methods this class exposes -
the services and the UI never touch the file themselves.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from learning.assessment import GradedTest
from utils.constants import normalise_token
from utils.logging_utils import get_logger

__all__ = ["ProgressStore", "SubjectPerformance", "WeakArea", "SCHEMA_VERSION"]

SCHEMA_VERSION = 1

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SubjectPerformance:
    """Aggregated test performance for one subject."""

    subject: str
    tests: int
    questions: int
    correct: int
    accuracy: float
    score: float
    max_score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "tests": self.tests,
            "questions": self.questions,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "score": self.score,
            "max_score": self.max_score,
        }

    @property
    def accuracy_percent(self) -> int:
        return int(round(self.accuracy * 100))


@dataclass(frozen=True, slots=True)
class WeakArea:
    """One repeatedly-missed topic or chapter."""

    name: str
    subject: str
    where: str
    questions: int
    wrong: int
    accuracy: float

    @property
    def accuracy_percent(self) -> int:
        return int(round(self.accuracy * 100))

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "subject": self.subject,
            "where": self.where,
            "questions": self.questions,
            "wrong": self.wrong,
            "accuracy": self.accuracy,
        }


def _today_key() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _blank(profile: str) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {
        "schema": SCHEMA_VERSION,
        "profile": profile,
        "created_at": now,
        "updated_at": now,
        "tests": [],
        "daily": {},
        "events": {"doubts": 0, "chapter_pages": 0, "revision_sheets": 0},
    }


# --------------------------------------------------------------------- repair
#: Fields the analytics aggregate as counts, marks or timings.
_RECORD_COUNTS = ("total", "correct", "wrong", "skipped")
_RECORD_MEASURES = ("score", "max_score", "accuracy", "duration_seconds")
_RECORD_TEXT = ("test_id", "title", "scope", "subject", "chapter", "topic", "created_at")
_TOPIC_COUNTS = ("total", "correct", "wrong")


def _as_int(value: Any, default: int = 0) -> int:
    """Count stored in any shape the file happens to have (``"x"`` -> default)."""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _coerce_topic(topic: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(topic)
    out["topic"] = str(out.get("topic") or "General")
    for key in _TOPIC_COUNTS:
        out[key] = _as_int(out.get(key))
    return out


def _coerce_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """One stored test result, brought back to the shapes the readers expect."""
    out = dict(record)
    for key in _RECORD_TEXT:
        out[key] = str(out.get(key) or "")
    for key in _RECORD_COUNTS:
        out[key] = _as_int(out.get(key))
    for key in _RECORD_MEASURES:
        out[key] = _as_float(out.get(key))
    topics = out.get("topics")
    out["topics"] = (
        [_coerce_topic(topic) for topic in topics if isinstance(topic, Mapping)]
        if isinstance(topics, list)
        else []
    )
    return out


def _repair_document(document: dict[str, Any], *, source: Path) -> dict[str, Any]:
    """Repair a syntactically valid but mistyped document instead of crashing.

    ``load`` can guarantee a dict, but nothing guarantees that ``tests`` is a
    list, that its entries are dicts, or that the counters are numbers - a
    hand-edited or half-truncated file must degrade to dropped junk, never to
    an ``AttributeError`` rendered in front of the student.
    """
    tests = document.get("tests")
    if isinstance(tests, list):
        records = [_coerce_record(item) for item in tests if isinstance(item, Mapping)]
        if len(records) < len(tests):
            logger.warning(
                "Progress file %s: dropped %d malformed test record(s)",
                source,
                len(tests) - len(records),
            )
        document["tests"] = records
    else:
        if tests:
            logger.warning(
                "Progress file %s: 'tests' was %s, not a list; the history is unusable",
                source,
                type(tests).__name__,
            )
        document["tests"] = []
    daily = document.get("daily")
    buckets: dict[str, Any] = {}
    if isinstance(daily, Mapping):
        for key, value in daily.items():
            if not isinstance(value, Mapping):
                logger.warning("Progress file %s: day '%s' was not a bucket; dropped", source, key)
                continue
            buckets[str(key)] = {
                "tests": _as_int(value.get("tests")),
                "questions": _as_int(value.get("questions")),
                "correct": _as_int(value.get("correct")),
            }
    elif daily:
        logger.warning("Progress file %s: 'daily' was %s, not an object", source, type(daily).__name__)
    document["daily"] = buckets
    events = document.get("events")
    if isinstance(events, Mapping):
        document["events"] = {str(key): _as_int(value) for key, value in events.items()}
    else:
        if events:
            logger.warning("Progress file %s: 'events' was %s, not an object", source, type(events).__name__)
        document["events"] = {}
    return document


class ProgressStore:
    """Append-only study history kept in one JSON document."""

    def __init__(self, path: Path | str, *, profile: str = "default") -> None:
        self.path = Path(path)
        # the profile doubles as the file name, so spaces become underscores
        self.profile = normalise_token(str(profile)).replace(" ", "_") or "default"
        self._lock = threading.RLock()
        self._cache: dict[str, Any] | None = None
        self._cache_stamp: float | None = None
        #: set when an unreadable history file was moved aside (see ``load``)
        self.last_quarantine: Path | None = None

    # ------------------------------------------------------------------- paths
    @property
    def file(self) -> Path:
        """``progress/<profile>.json``."""
        if self.path.suffix == ".json":
            return self.path
        return self.path / f"{self.profile}.json"

    # -------------------------------------------------------------------- I/O
    def load(self, *, force: bool = False) -> dict[str, Any]:
        """Read (and repair) the stored document; never raises."""
        with self._lock:
            stamp = self._stat_stamp()
            if not force and self._cache is not None and self._cache_stamp == stamp:
                return self._cache
            document: dict[str, Any] | None = None
            raw: str | None = None
            try:
                raw = self.file.read_text(encoding="utf-8")
            except FileNotFoundError:
                raw = None
            except (OSError, UnicodeDecodeError) as exc:
                logger.error("Progress file %s unreadable (%s); starting a fresh log", self.file, exc)
                raw = None
                # the bytes may still be worth keeping; renaming can work when reading cannot
                self._quarantine("unreadable progress file")
            if raw is not None:
                parsed: Any = None
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError as exc:
                    logger.error(
                        "Progress file %s is not valid JSON (%s); starting a fresh log", self.file, exc
                    )
                if parsed is not None and not isinstance(parsed, dict):
                    logger.error(
                        "Progress file %s holds a JSON %s, not a document; starting a fresh log",
                        self.file,
                        type(parsed).__name__,
                    )
                    parsed = None
                if isinstance(parsed, dict):
                    document = _repair_document(parsed, source=self.file)
                elif parsed is None and raw.strip():
                    # unreadable *content* (not a read error): keep the only copy
                    self._quarantine("unreadable progress document")
            if document is None:
                document = _blank(self.profile)
            document.setdefault("schema", SCHEMA_VERSION)
            document.setdefault("profile", self.profile)
            document.setdefault("tests", [])
            document.setdefault("daily", {})
            events = document.setdefault("events", {})
            for key in ("doubts", "chapter_pages", "revision_sheets"):
                events.setdefault(key, 0)
            self._cache = document
            self._cache_stamp = stamp
            return document

    def save(self, document: Mapping[str, Any]) -> Path:
        """Write the document atomically (temp file + rename)."""
        with self._lock:
            target = self.file
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = dict(document)
            payload["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            temp = target.with_suffix(f".{os.getpid()}.tmp")
            temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(temp, target)
            self._cache = payload
            self._cache_stamp = self._stat_stamp()
            return target

    def _stat_stamp(self) -> float | None:
        try:
            return self.file.stat().st_mtime
        except OSError:
            return None

    def _quarantine(self, reason: str) -> "Path | None":
        """Move an unreadable history file aside so the next save cannot erase it.

        The student's progress may be the only copy of months of study; when the
        document cannot be parsed it is renamed to ``<profile>.corrupt-<utc>.json``
        and recorded on :attr:`last_quarantine` instead of being silently replaced.
        """
        source = self.file
        if not source.exists():
            return None
        utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        target = source.with_name(f"{source.stem}.corrupt-{utc}.json")
        try:
            if target.exists():
                target = source.with_name(f"{source.stem}.corrupt-{utc}-{os.getpid()}.json")
            os.replace(source, target)
        except OSError as exc:
            logger.error("Could not quarantine %s (%s)", source, exc)
            return None
        self.last_quarantine = target
        logger.error("%s: kept the unreadable history copy at %s", reason, target)
        return target

    # ------------------------------------------------------------------ record
    def record(self, result: GradedTest) -> dict[str, Any]:
        """Persist one graded test and return its stored record."""
        return self.record_dict(result.as_record())

    def record_dict(self, record: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            document = self.load(force=True)
            entry = dict(record)
            entry.setdefault("created_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
            document["tests"].append(entry)
            day = document["daily"].setdefault(
                str(entry["created_at"])[:10], {"tests": 0, "questions": 0, "correct": 0}
            )
            day["tests"] = int(day.get("tests", 0)) + 1
            day["questions"] = int(day.get("questions", 0)) + int(entry.get("total") or 0)
            day["correct"] = int(day.get("correct", 0)) + int(entry.get("correct") or 0)
            self.save(document)
            logger.info(
                "Recorded %s test '%s' (%s/%s correct, accuracy %.0f%%)",
                entry.get("scope"),
                entry.get("title"),
                entry.get("correct"),
                entry.get("total"),
                float(entry.get("accuracy") or 0.0) * 100,
            )
            return entry

    def bump(self, kind: str, *, count: int = 1) -> int:
        """Count a non-test activity (doubt asked, chapter studied, sheet made)."""
        with self._lock:
            document = self.load(force=True)
            events = document.setdefault("events", {})
            events[kind] = int(events.get(kind, 0)) + count
            self.save(document)
            return int(events[kind])

    def tests(self, *, limit: int | None = None, subject: str | None = None) -> list[dict[str, Any]]:
        """Stored test records, newest first."""
        records = list(self.load().get("tests") or [])
        records.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        if subject and subject != "all":
            records = [item for item in records if item.get("subject") == normalise_token(subject)]
        # ``limit=0`` means "show none"; only ``None`` means "no limit"
        return records[: max(0, int(limit))] if limit is not None else records

    def clear(self) -> None:
        """Delete the history (keeps the file, empty - used by the UI reset)."""
        with self._lock:
            self.save(_blank(self.profile))
            logger.info("Progress history cleared for profile '%s'", self.profile)

    # --------------------------------------------------------------- analytics
    def stats(self) -> dict[str, Any]:
        """Headline numbers for the Dashboard and the Progress tab."""
        document = self.load()
        records = document.get("tests") or []
        questions = sum(int(item.get("total") or 0) for item in records)
        correct = sum(int(item.get("correct") or 0) for item in records)
        wrong = sum(int(item.get("wrong") or 0) for item in records)
        skipped = sum(int(item.get("skipped") or 0) for item in records)
        score = round(sum(float(item.get("score") or 0.0) for item in records), 2)
        max_score = round(sum(float(item.get("max_score") or 0.0) for item in records), 2)
        events = document.get("events") or {}
        return {
            "profile": document.get("profile", self.profile),
            "tests": len(records),
            "questions": questions,
            "correct": correct,
            "wrong": wrong,
            "skipped": skipped,
            "accuracy": round(correct / questions, 4) if questions else 0.0,
            "score": score,
            "max_score": max_score,
            "doubts": int(events.get("doubts") or 0),
            "chapter_pages": int(events.get("chapter_pages") or 0),
            "revision_sheets": int(events.get("revision_sheets") or 0),
            "days_active": len(document.get("daily") or {}),
            "streak_days": self.streak(),
            "last_studied": (
                max((str(item.get("created_at") or "")[:10] for item in records), default="")
                or ""
            ),
        }

    def streak(self) -> int:
        """Consecutive study days ending today (or yesterday)."""
        days = set((self.load().get("daily") or {}).keys())
        if not days:
            return 0
        cursor = date.today()
        if cursor.isoformat() not in days:
            cursor -= timedelta(days=1)  # a streak survives until the next day
            if cursor.isoformat() not in days:
                return 0
        count = 0
        while cursor.isoformat() in days:
            count += 1
            cursor -= timedelta(days=1)
        return count

    def subject_performance(self) -> dict[str, SubjectPerformance]:
        """Per-subject questions, accuracy and marks."""
        buckets: dict[str, dict[str, float]] = {}
        for item in self.load().get("tests") or []:
            subject = str(item.get("subject") or "general")
            bucket = buckets.setdefault(
                subject,
                {"tests": 0, "questions": 0, "correct": 0, "score": 0.0, "max_score": 0.0},
            )
            bucket["tests"] += 1
            bucket["questions"] += int(item.get("total") or 0)
            bucket["correct"] += int(item.get("correct") or 0)
            bucket["score"] += float(item.get("score") or 0.0)
            bucket["max_score"] += float(item.get("max_score") or 0.0)
        return {
            subject: SubjectPerformance(
                subject=subject,
                tests=int(values["tests"]),
                questions=int(values["questions"]),
                correct=int(values["correct"]),
                accuracy=round(values["correct"] / values["questions"], 4) if values["questions"] else 0.0,
                score=round(values["score"], 2),
                max_score=round(values["max_score"], 2),
            )
            for subject, values in sorted(buckets.items())
        }

    def chapter_performance(self) -> dict[str, dict[str, Any]]:
        """Per-chapter attempts and accuracy (keyed by ``subject::chapter``)."""
        out: dict[str, dict[str, Any]] = {}
        for item in self.load().get("tests") or []:
            chapter = str(item.get("chapter") or "").strip()
            if not chapter:
                continue
            subject = str(item.get("subject") or "general")
            key = f"{subject}::{chapter}"
            entry = out.setdefault(
                key,
                {
                    "subject": subject,
                    "chapter": chapter,
                    "tests": 0,
                    "questions": 0,
                    "correct": 0,
                    "accuracy": 0.0,
                    "last_attempt": "",
                },
            )
            entry["tests"] += 1
            entry["questions"] += int(item.get("total") or 0)
            entry["correct"] += int(item.get("correct") or 0)
            entry["last_attempt"] = max(entry["last_attempt"], str(item.get("created_at") or ""))
        for entry in out.values():
            entry["accuracy"] = (
                round(entry["correct"] / entry["questions"], 4) if entry["questions"] else 0.0
            )
        return dict(sorted(out.items(), key=lambda kv: (kv[1]["accuracy"], -kv[1]["questions"])))

    def weak_areas(self, *, threshold: float = 0.75, limit: int = 8) -> list[WeakArea]:
        """Topics and chapters the student keeps losing marks in, worst first."""
        topics: dict[str, dict[str, Any]] = {}
        for item in self.load().get("tests") or []:
            subject = str(item.get("subject") or "general")
            chapter = str(item.get("chapter") or "")
            for topic in item.get("topics") or []:
                name = str(topic.get("topic") or "General").strip() or "General"
                bucket = topics.setdefault(
                    f"{subject}::{name}",
                    {
                        "name": name,
                        "subject": subject,
                        "where": chapter or "whole syllabus",
                        "questions": 0,
                        "wrong": 0,
                        "correct": 0,
                    },
                )
                bucket["questions"] += int(topic.get("total") or 0)
                bucket["wrong"] += int(topic.get("wrong") or 0)
                bucket["correct"] += int(topic.get("correct") or 0)
                if chapter:
                    bucket["where"] = chapter
        areas: list[WeakArea] = []
        for bucket in topics.values():
            answered = bucket["correct"] + bucket["wrong"]
            if not answered:
                continue
            accuracy = round(bucket["correct"] / answered, 4)
            if accuracy >= threshold:
                continue
            areas.append(
                WeakArea(
                    name=bucket["name"],
                    subject=bucket["subject"],
                    where=bucket["where"],
                    questions=bucket["questions"],
                    wrong=bucket["wrong"],
                    accuracy=accuracy,
                )
            )
        areas.sort(key=lambda area: (area.accuracy, -area.wrong))
        return areas[:limit]

    def recent_activity(self, *, limit: int = 6) -> list[dict[str, Any]]:
        """Compact rows for the Progress timeline."""
        return [
            {
                "created_at": item.get("created_at"),
                "title": item.get("title"),
                "scope": item.get("scope"),
                "subject": item.get("subject"),
                "total": item.get("total"),
                "correct": item.get("correct"),
                "accuracy": item.get("accuracy"),
                "score": item.get("score"),
            }
            for item in self.tests(limit=limit)
        ]

    def daily_series(self, *, days: int = 14) -> list[dict[str, Any]]:
        """Last ``days`` calendar buckets, oldest first (gaps included)."""
        store = self.load().get("daily") or {}
        series: list[dict[str, Any]] = []
        cursor = date.today() - timedelta(days=days - 1)
        for offset in range(days):
            key = (cursor + timedelta(days=offset)).isoformat()
            bucket = store.get(key) or {}
            series.append(
                {
                    "date": key,
                    "tests": int(bucket.get("tests") or 0),
                    "questions": int(bucket.get("questions") or 0),
                    "correct": int(bucket.get("correct") or 0),
                }
            )
        return series

    def as_dict(self) -> dict[str, Any]:
        """Everything the UI needs in one JSON-friendly payload."""
        return {
            "path": str(self.file),
            "stats": self.stats(),
            "subjects": {name: values.as_dict() for name, values in self.subject_performance().items()},
            "chapters": list(self.chapter_performance().values()),
            "weak_areas": [area.as_dict() for area in self.weak_areas()],
            "recent": self.recent_activity(),
        }


def summarise(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Small helper for tests and reports: totals over an arbitrary record list."""
    items = list(records)
    questions = sum(int(item.get("total") or 0) for item in items)
    correct = sum(int(item.get("correct") or 0) for item in items)
    return {
        "tests": len(items),
        "questions": questions,
        "correct": correct,
        "accuracy": round(correct / questions, 4) if questions else 0.0,
    }
