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
    weak_areas_md,
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
            "\n**Note:** your study library is still being prepared. You can explore "
            "your Dashboard and Progress while it is made ready."
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
    """Physics, Chemistry and Biology progress in student language."""
    return table(
        ("Subject", "Progress", "Practice", "Current focus"),
        [
            (
                f"{card['emoji']} **{card['label']} Progress**",
                (
                    f"{bar(card['accuracy'], width=8)} {card['accuracy_percent']}% accuracy"
                    if card["questions"]
                    else "Ready to begin"
                ),
                f"{card['questions']} questions · {card['tests']} test(s)",
                chip_line(card["weak_topics"], empty="Take a practice test"),
            )
            for card in cards
        ],
    )


def mock_stats_md(stats: Mapping[str, Any]) -> str:
    """Mock-test statistics without internal scoring or storage details."""
    return table(
        ("Mock tests", "Questions practiced", "Overall accuracy", "Marks earned"),
        [
            (
                stats.get("tests", 0),
                stats.get("questions", 0),
                f"{bar(stats.get('accuracy', 0.0), width=8)} {pct(stats.get('accuracy', 0.0))}",
                f"{stats.get('score', 0):g} / {stats.get('max_score', 0):g}",
            )
        ],
    )


def student_profile_md(store: Any, *, daily_goal: int = 20) -> str:
    """Compact profile for the Dashboard and Ask Genie sidebars."""
    stats = store.stats()
    today = store.daily_series(days=1)[0]
    completed = min(int(today["questions"]), daily_goal)
    recent = store.recent_activity(limit=1)
    activity = (
        f"{recent[0].get('title') or recent[0].get('subject') or 'Practice test'} · "
        f"{pct(recent[0].get('accuracy'))}"
        if recent
        else "No activity yet — your next study step will appear here."
    )
    return (
        "### 👋 Student Profile\n\n"
        f"**Progress:** {stats['questions']} questions · {pct(stats['accuracy'])} accuracy  \n"
        f"**Daily Goal:** {completed}/{daily_goal} questions  \n"
        f"**Study Streak:** 🔥 {stats['streak_days']} day(s)  \n"
        f"**Recent Activity:** {esc(activity)}"
    )


def recommended_topic_md(platform: GeniePlatform) -> str:
    """One clear next step, based on weak areas before catalog coverage."""
    weak = platform.progress.weak_areas(limit=1)
    if weak:
        area = weak[0]
        return (
            f"### 🎯 Recommended Next Study Topic\n\n"
            f"**Revise {esc(area.name)}** in {esc(area.subject.title())}. "
            f"Your current accuracy is {area.accuracy_percent}%. Start with Quick Revision, "
            "then try a short practice set."
        )
    chapters = platform.catalog().chapters
    if chapters:
        chapter = chapters[0]
        return (
            "### 🎯 Recommended Next Study Topic\n\n"
            f"**Start with {esc(chapter.name)}** in {esc(chapter.subject.title())}. "
            "Open Learn for a guided study page, then test yourself in Practice."
        )
    return (
        "### 🎯 Recommended Next Study Topic\n\n"
        "**Ask your first doubt.** Tell Genie which chapter you are studying and it will "
        "help you choose the next step."
    )


def actions_md(actions: list[Mapping[str, Any]]) -> str:
    """Human list of the shortcuts below the buttons."""
    return "\n".join(
        f"- **{action['emoji']} {action['label']}** - {action['hint']}" for action in actions
    )


def dashboard_markdown(platform: GeniePlatform) -> str:
    """The complete student dashboard as one document (used by tests and CLI)."""
    payload = platform.dashboard()
    return "\n\n".join(
        [
            "## Your NEET preparation, made simple.\n"
            "Learn a chapter, clear doubts, practice questions and revise weak topics.",
            "### Quick actions\n\n" + actions_md(payload["actions"]),
            student_profile_md(platform.progress),
            "### Subject progress\n\n" + subject_cards_md(payload["subjects"]),
            "### Mock test statistics\n\n" + mock_stats_md(payload["stats"]),
            "### Weak areas\n\n" + weak_areas_md(platform.progress.weak_areas(limit=5)),
            recommended_topic_md(platform),
        ]
    )


# ---------------------------------------------------------------------------
# Gradio layout
# ---------------------------------------------------------------------------
def build(gr: Any, platform: GeniePlatform, settings: Any, nav: Any | None = None) -> dict[str, Any]:
    """Create the landing dashboard and return refs for its study shortcuts."""
    payload = platform.dashboard()

    with gr.Column(elem_classes=["genie-hero"]):
        gr.Markdown(
            "## Study smarter. Clear every doubt.\n"
            "Your personal NEET study companion for learning, practice, revision and progress."
        )
        gr.Markdown("**New here? Start with Ask a Doubt or choose a chapter in Learn.**")
    with gr.Row(elem_classes=["genie-actions"]):
        ask_btn = gr.Button("✅ Ask a Doubt", variant="primary")
        practice_btn = gr.Button("✅ Start Practice")
        revise_btn = gr.Button("✅ Quick Revision")
        mock_btn = gr.Button("✅ Mock Test")
    guidance_md = gr.Markdown(
        "**How to study:** Learn a topic → ask doubts → practice MCQs → revise weak areas."
    )

    with gr.Row():
        with gr.Column(scale=2, elem_classes=["genie-main-card"]):
            subjects_md = gr.Markdown("### 📚 Subject Progress\n\n" + subject_cards_md(payload["subjects"]))
            stats_widget = gr.Markdown("### 📝 Mock Test Statistics\n\n" + mock_stats_md(payload["stats"]))
            weak_widget = gr.Markdown(
                "### 🧠 Weak Areas\n\n"
                + weak_areas_md(
                    platform.progress.weak_areas(limit=5),
                    empty_hint="_Take your first practice test to discover topics that need more revision._",
                )
            )
            recommendation_widget = gr.Markdown(recommended_topic_md(platform))
        with gr.Column(scale=1, elem_classes=["genie-profile-card"]):
            profile_widget = gr.Markdown(student_profile_md(platform.progress))
            gr.Markdown("### 🌟 Today’s Motivation")
            motivation_md = gr.Markdown(f"> {esc(payload['motivation']['line'])}")
            refresh_btn = gr.Button("Refresh dashboard", size="sm")

    buttons = {
        "ask_doubt": ask_btn,
        "practice_mcqs": practice_btn,
        "quick_revision": revise_btn,
        "mock_test": mock_btn,
    }
    return {
        "stats": stats_widget,
        "subjects": subjects_md,
        "weak": weak_widget,
        "recommendation": recommendation_widget,
        "profile": profile_widget,
        "motivation": motivation_md,
        "guidance": guidance_md,
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
            "### 📝 Mock Test Statistics\n\n" + mock_stats_md(payload["stats"]),
            "### 📚 Subject Progress\n\n" + subject_cards_md(payload["subjects"]),
            "### 🧠 Weak Areas\n\n" + weak_areas_md(platform.progress.weak_areas(limit=5)),
            recommended_topic_md(platform),
            student_profile_md(platform.progress),
            f"> {esc(payload['motivation']['line'])}",
        ]

    dashboard_refs["refresh"].click(
        refresh,
        inputs=None,
        outputs=[
            dashboard_refs["stats"], dashboard_refs["subjects"], dashboard_refs["weak"],
            dashboard_refs["recommendation"], dashboard_refs["profile"], dashboard_refs["motivation"],
        ],
        queue=False,
    )
