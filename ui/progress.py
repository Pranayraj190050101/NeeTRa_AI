"""Progress tab - the local study history, read back from the JSON store.

Everything on this page comes from :class:`learning.progress.ProgressStore`:
tests attempted, scores, accuracy, per-subject and per-chapter performance and
the ranked weak areas.  The reset button is the only writer, and it writes
through the store so the schema and the atomic rename stay in one place.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from learning.progress import ProgressStore
from learning.service import GeniePlatform
from ui.components import bar, chip_line, esc, pct, table, weak_areas_md

__all__ = [
    "build",
    "headline_md",
    "subject_table_md",
    "chapter_table_md",
    "weak_areas_table_md",
    "activity_md",
    "daily_md",
    "progress_report_md",
    "wire",
]


def headline_md(store: ProgressStore) -> str:
    """Totals plus where the history file lives."""
    stats = store.stats()
    row = table(
        ("Tests", "Questions", "Correct", "Wrong", "Skipped", "Accuracy", "Marks", "Active days", "Streak"),
        [
            (
                stats["tests"],
                stats["questions"],
                stats["correct"],
                stats["wrong"],
                stats["skipped"],
                f"{bar(stats['accuracy'], width=8)} {pct(stats['accuracy'])}",
                f"{stats['score']:g}/{stats['max_score']:g}",
                stats["days_active"],
                f"{stats['streak_days']} day(s)",
            )
        ],
    )
    return (
        row
        + "\n\n"
        + f"_History file: `{store.file}` · last activity {esc(stats['last_studied']) or 'never'}_\n\n"
        + f"**Non-test activity:** {stats['doubts']} doubt(s) asked · "
        f"{stats['chapter_pages']} chapter page(s) studied · {stats['revision_sheets']} revision sheet(s)"
    )


def subject_table_md(store: ProgressStore) -> str:
    """Per-subject performance (the Progress module's subject breakdown)."""
    values = store.subject_performance()
    if not values:
        return "_No tests recorded yet - subject accuracy appears here after your first submission._"
    return table(
        ("Subject", "Tests", "Questions", "Correct", "Accuracy", "Marks"),
        [
            (
                performance.subject.title(),
                performance.tests,
                performance.questions,
                performance.correct,
                f"{bar(performance.accuracy, width=8)} {performance.accuracy_percent}%",
                f"{performance.score:g}/{performance.max_score:g}",
            )
            for performance in values.values()
        ],
    )


def chapter_table_md(store: ProgressStore) -> str:
    """Per-chapter performance, weakest chapter first."""
    chapters = store.chapter_performance()
    if not chapters:
        return "_No chapter-level history yet. Chapter tests fill this table._"
    return table(
        ("Chapter", "Subject", "Tests", "Questions", "Accuracy", "Last attempt"),
        [
            (
                entry["chapter"],
                entry["subject"].title(),
                entry["tests"],
                entry["questions"],
                f"{bar(entry['accuracy'], width=6)} {pct(entry['accuracy'])}",
                str(entry["last_attempt"])[:10] or "-",
            )
            for entry in chapters.values()
        ],
    )


def weak_areas_table_md(store: ProgressStore, *, threshold: float = 0.75) -> str:
    """Topics the student keeps losing marks in - the Revision Hub's targets."""
    areas = store.weak_areas(threshold=threshold)
    return weak_areas_md(
        areas,
        empty_hint=(
            "_No topic is below the threshold yet. Either your accuracy is solid or no test "
            "has been recorded - the Practice tab is the next step._"
        ),
    )


def activity_md(store: ProgressStore, *, limit: int = 10) -> str:
    """The test timeline."""
    rows = []
    for record in store.tests(limit=limit):
        rows.append(
            (
                str(record.get("created_at") or "")[:16].replace("T", " "),
                record.get("scope"),
                record.get("subject"),
                record.get("title"),
                f"{record.get('correct')}/{record.get('total')}",
                pct(record.get("accuracy")),
                f"{float(record.get('score') or 0):g}",
                chip_line(record.get("weak_topics") or [], empty="-"),
            )
        )
    if not rows:
        return "_No test has been submitted yet._"
    return table(("When", "Scope", "Subject", "Test", "Correct", "Accuracy", "Marks", "Weak topics"), rows)


def daily_md(store: ProgressStore, *, days: int = 14) -> str:
    """Last two weeks of activity, so a gap is visible."""
    series = [day for day in store.daily_series(days=days) if day["questions"]]
    if not series:
        return f"_No activity in the last {days} days._"
    return table(
        ("Date", "Tests", "Questions", "Correct", "Accuracy"),
        [
            (
                day["date"],
                day["tests"],
                day["questions"],
                day["correct"],
                f"{bar(day['correct'] / day['questions'] if day['questions'] else 0, width=6)}",
            )
            for day in series
        ],
    )


def progress_report_md(platform: GeniePlatform) -> str:
    """The whole tab as one document (also used by tests and the CLI)."""
    store = platform.progress
    parts = [
        "### \U0001f4c8 Progress",
        headline_md(store),
        "**Subject performance**\n\n" + subject_table_md(store),
        "**Chapter performance**\n\n" + chapter_table_md(store),
        "**Weak areas**\n\n" + weak_areas_table_md(store, threshold=platform.settings.weak_accuracy_threshold),
        "**Recent tests**\n\n" + activity_md(store),
        "**Last 14 days**\n\n" + daily_md(store),
    ]
    if store.last_quarantine is not None:
        # an unreadable history was kept, not erased - tell the student where it is
        parts.insert(
            1,
            "\u26a0\ufe0f An unreadable progress file was saved to **"
            f"`{store.last_quarantine}`** before starting a fresh log. "
            "If it holds study history you want back, fix the JSON and restart the app.",
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Gradio layout
# ---------------------------------------------------------------------------
def build(
    gr: Any,
    platform: GeniePlatform,
    settings: Any,
    nav: Any | None = None,
    *,
    targets: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the Progress tab widgets (all read-only apart from the reset)."""
    store = platform.progress
    gr.Markdown("### \U0001f4c8 Progress - what you have done and where marks are leaking")
    headline_box = gr.Markdown(headline_md(store))
    with gr.Row():
        refresh_btn = gr.Button("Refresh", size="sm", scale=2)
        practice_btn = gr.Button("Practice a weak chapter", variant="primary", scale=3)
        reset_btn = gr.Button("Reset history", size="sm", scale=1)

    with gr.Accordion("Subject performance", open=True):
        subjects_box = gr.Markdown(subject_table_md(store))
    with gr.Accordion("Chapter performance", open=False):
        chapters_box = gr.Markdown(chapter_table_md(store))
    with gr.Accordion("Weak areas", open=False):
        weak_box = gr.Markdown(weak_areas_table_md(store, threshold=settings.weak_accuracy_threshold))
    with gr.Accordion("Test timeline", open=False):
        activity_box = gr.Markdown(activity_md(store))
        daily_box = gr.Markdown(daily_md(store))
    with gr.Accordion("Raw stored document (JSON)", open=False):
        gr.Markdown(f"_Location:_ `{store.file}`")
        raw_box = gr.Markdown(_raw_md(store))
        show_btn = gr.Button("Reload JSON", size="sm")

    refs: dict[str, Any] = {
        "headline": headline_box,
        "subjects": subjects_box,
        "chapters": chapters_box,
        "weak": weak_box,
        "activity": activity_box,
        "daily": daily_box,
        "raw": raw_box,
        "refresh": refresh_btn,
        "practice": practice_btn,
        "reset": reset_btn,
        "show": show_btn,
    }
    wire(gr, platform, settings, refs, nav=nav)
    return refs


def _raw_md(store: ProgressStore) -> str:
    """The stored document itself, so the JSON format is inspectable."""
    text = json.dumps(store.load(force=True), indent=2, sort_keys=True, default=str)
    if len(text) > 6000:
        text = text[:6000] + "\n... (truncated - open the file for the whole history)"
    return f"```json\n{text}\n```"


def wire(
    gr: Any,
    platform: GeniePlatform,
    settings: Any,
    refs: dict[str, Any],
    *,
    nav: Any | None = None,
) -> None:
    """Refresh / reset handlers for the Progress tab."""

    def reload() -> list[Any]:
        store = platform.progress
        store.load(force=True)
        return [
            headline_md(store),
            subject_table_md(store),
            chapter_table_md(store),
            weak_areas_table_md(store, threshold=settings.weak_accuracy_threshold),
            activity_md(store),
            daily_md(store),
        ]

    def show_raw() -> str:
        return _raw_md(platform.progress)

    def nudge_practice() -> list[Any]:
        areas = platform.progress.weak_areas(limit=3)
        if not areas:
            text = "**Nothing to fix yet** - submit a test in the Practice tab first."
        else:
            text = (
                "**Weakest right now:** "
                + chip_line(f"{area.name} ({area.accuracy_percent}%)" for area in areas)
                + "\n\nOpening Practice: pick the matching chapter in the target list."
            )
        return [text] + (nav.tail("practice") if nav is not None else [])

    reload_outputs = [
        refs["headline"],
        refs["subjects"],
        refs["chapters"],
        refs["weak"],
        refs["activity"],
        refs["daily"],
    ]
    refs["refresh"].click(reload, inputs=None, outputs=reload_outputs, queue=False)
    refs["show"].click(show_raw, inputs=None, outputs=[refs["raw"]], queue=False)

    practice_outputs = [refs["weak"]] + (nav.outputs if nav is not None else [])
    refs["practice"].click(nudge_practice, inputs=None, outputs=practice_outputs, queue=False)

    def reset() -> list[Any]:
        platform.progress.clear()
        store = platform.progress
        return [
            "**History cleared.** The JSON file now holds an empty document - "
            f"`{store.file}`",
            subject_table_md(store),
            chapter_table_md(store),
            weak_areas_table_md(store, threshold=settings.weak_accuracy_threshold),
            activity_md(store),
            daily_md(store),
        ]

    refs["reset"].click(reset, inputs=None, outputs=[refs["headline"]] + reload_outputs[1:], queue=False)
