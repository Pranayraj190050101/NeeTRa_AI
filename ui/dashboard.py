"""Dashboard tab - the landing page of the platform.

Shows the welcome section, the daily motivation, quick statistics, one card per
subject and the four shortcuts the product brief asks for.  Every number comes
from :class:`learning.service.GeniePlatform` (i.e. the progress store and the
chapter catalog) - the dashboard ships no static content of its own, so it is
equally honest on day one of a fresh index.
"""

from __future__ import annotations

from typing import Any, Mapping

from learning.service import GeniePlatform, QUICK_ACTIONS
from ui.components import (
    bar,
    chip_line,
    dashboard_heading,
    esc,
    pct,
    table,
)

__all__ = [
    "build",
    "quick_action_plan",
    "wire",
    "actions_md",
    "dashboard_markdown",
    "stats_md",
    "subject_cards_md",
]

#: refs keys each quick action pre-configures, in the order its values are fed.
_ACTION_TARGETS: dict[str, tuple[str, ...]] = {
    "ask_doubt": ("ask_mode", "ask_message"),
    "practice_mcqs": ("practice_scope",),
    "quick_revision": ("ask_mode", "ask_message"),
    "mock_test": ("practice_scope",),
}


def action_by_key(key: str) -> dict[str, str]:
    """One shipped quick action by key."""
    for action in QUICK_ACTIONS:
        if action["key"] == key:
            return dict(action)
    raise KeyError(f"Unknown quick action '{key}'. Known: {[a['key'] for a in QUICK_ACTIONS]}")


def quick_action_plan(platform: GeniePlatform, key: str) -> dict[str, Any]:
    """What a shortcut does: where it lands, what it configures, what it says."""
    action = action_by_key(key)
    stats = platform.progress.stats()
    catalog = platform.catalog()
    lines = [
        f"### {action['emoji']} {action['label']}",
        "",
        action["hint"],
        "",
        f"**Opens:** the _{action['tab'].replace('-', ' ').title()}_ module",
    ]
    if action["selector"] == "mode":
        from utils.constants import AnswerMode

        mode = AnswerMode.coerce(action["value"], default=platform.settings.answer_mode)
        lines.append(f"**Answer mode set to:** {mode.emoji} {mode.label}")
    elif action["selector"] == "scope":
        from utils.constants import TestScope

        scope = TestScope.coerce(action["value"])
        lines.append(f"**Test scope set to:** {scope.emoji} {scope.label}")
    if action["seed"]:
        lines.append(f"**Question drafted:** _{action['seed']}_")
    if catalog.is_empty:
        lines.append(
            "\n**Note:** no chapters are indexed yet, so the first stop should be "
            "`data/<subject>/` + `python -m rag.ingest --rebuild`."
        )
    if stats["tests"] and action["selector"] == "scope":
        weak = platform.progress.weak_areas(limit=3)
        if weak:
            lines.append(
                "\n**Weakest topics so far:** " + chip_line(area.name for area in weak)
            )
    lines.append("\nSwitch to the tab and press Enter / Generate.")
    return {
        "action": action,
        "markdown": "\n".join(lines),
        "tab": action["tab"],
        "selector": action["selector"],
        "value": action["value"],
        "seed": action["seed"],
    }


def stats_md(stats: Mapping[str, Any]) -> str:
    """Quick statistics block."""
    return table(
        ("Tests", "Questions", "Correct", "Accuracy", "Marks", "Streak", "Doubts", "Sheets"),
        [
            (
                stats.get("tests", 0),
                stats.get("questions", 0),
                stats.get("correct", 0),
                f"{bar(stats.get('accuracy', 0.0), width=8)} {pct(stats.get('accuracy', 0.0))}",
                f"{stats.get('score', 0):g}/{stats.get('max_score', 0):g}",
                f"{stats.get('streak_days', 0)} day(s)",
                stats.get("doubts", 0),
                stats.get("revision_sheets", 0),
            )
        ],
    )


def subject_cards_md(cards: list[Mapping[str, Any]]) -> str:
    """One row per NEET subject: coverage plus performance."""
    return table(
        ("Subject", "Chapters", "Indexed chunks", "Tests", "Accuracy", "Verdict", "Weak topics"),
        [
            (
                f"{card['emoji']} {card['label']}",
                card["chapters"],
                card["chunks"],
                card["tests"],
                f"{bar(card['accuracy'], width=6)} {card['accuracy_percent']}%",
                card["status"],
                chip_line(card["weak_topics"], empty="-"),
            )
            for card in cards
        ],
    )


def actions_md(actions: list[Mapping[str, Any]]) -> str:
    """Human list of the shortcuts below the buttons."""
    return "\n".join(
        f"- **{action['emoji']} {action['label']}** - {action['hint']}" for action in actions
    )


def dashboard_markdown(platform: GeniePlatform) -> str:
    """The whole dashboard as one markdown document (used by tests and CLI)."""
    payload = platform.dashboard()
    welcome = payload["welcome"]
    stats = payload["stats"]
    head = dashboard_heading(welcome["name"], welcome["tagline"])
    facts = table(
        ("Module", "Ready"),
        [
            ("Ask Genie - the coach chat solves your doubt", "yes"),
            ("Practice - subject / chapter / topic tests", "yes" if welcome["index_ready"] else "ungrounded"),
            ("Progress - local JSON history", "yes"),
            ("Revision Hub - flash notes, formulas, high yield", "yes"),
        ]
        + (
            [
                (
                    "Learn - chapter pages from your index",
                    "yes" if welcome["chapters"] else "index empty",
                )
            ]
            if getattr(platform.settings, "learn_enabled", True)
            else []
        ),
    )
    return "\n\n".join(
        [
            head,
            f"**Role:** {welcome['role']}  \n"
            f"**Chapters indexed:** {welcome['chapters']} across {stats['subjects']} subject(s)  \n"
            f"**Passages available:** {welcome['chunks']}",
            "### Quick statistics\n\n" + stats_md(payload["stats"]),
            "### Subjects\n\n" + subject_cards_md(payload["subjects"]),
            "### Quick actions\n\n" + actions_md(payload["actions"]),
            "### What this build can do\n\n" + str(payload["capabilities"]),
        ]
    )


# ---------------------------------------------------------------------------
# Gradio layout
# ---------------------------------------------------------------------------
def build(gr: Any, platform: GeniePlatform, settings: Any, nav: Any | None = None) -> dict[str, Any]:
    """Create the Dashboard tab contents and return the widget references."""
    payload = platform.dashboard()
    welcome = payload["welcome"]

    gr.Markdown(dashboard_heading(welcome["name"], welcome["tagline"]))
    gr.Markdown(
        f"**{esc(welcome['role'])}** - open **Ask Genie** and type the doubt; "
        "practice, mock tests, analytics and revision sit around the chat. "
        f"{payload['stats']['chapters']} chapter(s) are indexed."
    )
    with gr.Row():
        with gr.Column(scale=2):
            stats_md_widget = gr.Markdown("### Quick statistics\n\n" + stats_md(payload["stats"]))
            gr.Markdown("### Subjects\n\n" + subject_cards_md(payload["subjects"]))
        with gr.Column(scale=1):
            gr.Markdown("### Daily motivation")
            motivation_md = gr.Markdown(f"> {esc(payload['motivation']['line'])}")
            gr.Markdown("### Quick actions")
            with gr.Row():
                ask_btn = gr.Button("💬 Ask Doubt", variant="primary")
                practice_btn = gr.Button("🧩 Practice MCQs")
            with gr.Row():
                revise_btn = gr.Button("⚡ Quick Revision")
                mock_btn = gr.Button("🎯 Start Mock Test")
            guidance_md = gr.Markdown(actions_md(payload["actions"]))
            refresh_btn = gr.Button("Refresh dashboard", size="sm")

    with gr.Accordion("Platform status and roadmap", open=False):
        status_md = gr.Markdown(str(payload["capabilities"]))

    buttons = {
        "ask_doubt": ask_btn,
        "practice_mcqs": practice_btn,
        "quick_revision": revise_btn,
        "mock_test": mock_btn,
    }
    return {
        "stats": stats_md_widget,
        "motivation": motivation_md,
        "guidance": guidance_md,
        "status": status_md,
        "buttons": buttons,
        "refresh": refresh_btn,
        "nav": nav,
    }


def wire(
    gr: Any,
    platform: GeniePlatform,
    refs: Mapping[str, Any],
    nav: Any,
    dashboard_refs: Mapping[str, Any],
) -> None:
    """Attach the quick-action handlers once every tab exists.

    Each shortcut configures the module it opens (answer mode or test scope,
    plus a drafted question where one makes sense) and, on Gradio builds that
    support it, switches to that tab.  The guidance panel always explains the
    next click, so the shortcut is usable even without tab switching.
    """
    targets = {
        "ask_mode": refs.get("ask_mode"),
        "ask_message": refs.get("ask_message"),
        "practice_scope": refs.get("practice_scope"),
    }
    for key, button in dashboard_refs["buttons"].items():
        names = tuple(name for name in _ACTION_TARGETS[key] if targets.get(name) is not None)
        outputs = [dashboard_refs["guidance"]] + [targets[name] for name in names] + nav.outputs

        def handler(action_key: str = key, wanted: tuple[str, ...] = names) -> list[Any]:
            plan = quick_action_plan(platform, action_key)
            values: list[Any] = [plan["markdown"]]
            for name in wanted:
                if name == "ask_mode":
                    values.append(gr.update(value=plan["value"]))
                elif name == "ask_message":
                    values.append(gr.update(value=plan["seed"]))
                elif name == "practice_scope":
                    values.append(gr.update(value=plan["value"]))
            return values + nav.tail(plan["tab"])

        button.click(handler, inputs=None, outputs=outputs, queue=False)

    def refresh() -> list[Any]:
        payload = platform.dashboard()
        return [
            "### Quick statistics\n\n" + stats_md(payload["stats"]),
            f"> {esc(payload['motivation']['line'])}",
            actions_md(payload["actions"]),
            str(payload["capabilities"]),
        ]

    dashboard_refs["refresh"].click(
        refresh,
        inputs=None,
        outputs=[dashboard_refs["stats"], dashboard_refs["motivation"],
                 dashboard_refs["guidance"], dashboard_refs["status"]],
        queue=False,
    )
