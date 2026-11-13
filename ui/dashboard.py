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
            "Ask Genie for a quick explanation, then test yourself in Practice."
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
    stats = payload["stats"]
    weak = platform.progress.weak_areas(limit=3)
    weak_topics = chip_line((area.name for area in weak), empty="Take a practice test to reveal focus topics")
    return "\n\n".join(
        [
            "## NEETra AI\nYour Personal NEET Doubt Solver, Practice Partner and Rank Improver",
            "### Your progress\n\n"
            + table(
                ("Questions answered", "Practice attempted", "Accuracy", "Weak topics"),
                [(stats["questions"], stats["tests"], pct(stats["accuracy"]), weak_topics)],
            ),
            "### Quick actions\n\n"
            + "Ask a Doubt · Practice Tests · Revision Planner · Mock Tests · Progress Dashboard",
            recommended_topic_md(platform),
        ]
    )


# ---------------------------------------------------------------------------
# Gradio layout
# ---------------------------------------------------------------------------
def build(gr: Any, platform: GeniePlatform, settings: Any, nav: Any | None = None) -> dict[str, Any]:
    """Create the landing dashboard and return refs for its study shortcuts."""
    payload = platform.dashboard()
    stats = payload["stats"]
    weak_areas = platform.progress.weak_areas(limit=5)

    with gr.Column(elem_classes=["genie-hero"]):
        gr.Markdown(
            "## Your Personal NEET Doubt Solver, Practice Partner and Rank Improver"
        )
    gr.Markdown("### Quick actions")
    with gr.Row(elem_classes=["genie-dashboard-actions"]):
        action_cards: dict[str, Any] = {}
        for key, title, description in (
            ("ask_doubt", "💬 Ask a Doubt", "Get a clear, NEET-focused explanation."),
            ("practice_tests", "🎯 Practice Tests", "Build confidence with focused questions."),
            ("revision_planner", "🔄 Revision Planner", "Create a revision sheet or review weak topics."),
            ("mock_tests", "📝 Mock Tests", "Attempt a timed-style practice set."),
            ("progress_dashboard", "📈 Progress Dashboard", "Review accuracy, activity and focus areas."),
        ):
            with gr.Column(elem_classes=["genie-action-card"]):
                action_cards[key] = gr.Button(
                    title,
                    variant="primary" if key == "ask_doubt" else "secondary",
                )
                gr.Markdown(description)

    stats_widget = gr.Markdown(
        "### Your progress\n\n"
        + table(
            ("Questions answered", "Practice attempted", "Accuracy", "Weak topics"),
            [
                (
                    stats["questions"],
                    stats["tests"],
                    pct(stats["accuracy"]),
                    chip_line(
                        (area.name for area in weak_areas),
                        empty="Take a practice test to reveal focus topics",
                    ),
                )
            ],
        ),
        elem_classes=["genie-main-card"],
    )
    weak_widget = gr.Markdown(
        "### Focus topics\n\n"
        + weak_areas_md(
            weak_areas,
            empty_hint="_Your focus topics will appear after your first practice test._",
        )
    )
    recommendation_widget = gr.Markdown(recommended_topic_md(platform))
    guidance_md = gr.Markdown()
    refresh_btn = gr.Button("Refresh dashboard", size="sm")
    return {
        "stats": stats_widget,
        "weak": weak_widget,
        "recommendation": recommendation_widget,
        "guidance": guidance_md,
        "buttons": action_cards,
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
    action_map = {
        "ask_doubt": ("ask_doubt", "ask-genie"),
        "practice_tests": ("practice_mcqs", "practice"),
        "revision_planner": (None, "revision"),
        "mock_tests": ("mock_test", "practice"),
        "progress_dashboard": (None, "progress"),
    }
    for key, button in dashboard_refs["buttons"].items():
        action_key, target_tab = action_map[key]
        action_names = _ACTION_TARGETS.get(action_key or "", ())
        names = tuple(name for name in action_names if targets.get(name) is not None)
        outputs = [dashboard_refs["guidance"]] + [targets[name] for name in names] + nav.outputs

        def handler(
            selected_action: str | None = action_key,
            wanted: tuple[str, ...] = names,
            destination: str = target_tab,
        ) -> list[Any]:
            if selected_action:
                plan = quick_action_plan(platform, selected_action)
                guidance = plan["markdown"]
            else:
                guidance = (
                    "**Opening the Revision Hub.** Choose a sheet type and topic to get started."
                    if destination == "revision"
                    else "**Opening Progress.** Review your practice history and accuracy."
                )
                plan = {"value": None, "seed": None}
            values: list[Any] = [guidance]
            for name in wanted:
                if name == "ask_mode":
                    values.append(gr.update(value=plan["value"]))
                elif name == "ask_message":
                    values.append(gr.update(value=plan["seed"]))
                elif name == "practice_scope":
                    values.append(gr.update(value=plan["value"]))
            return values + nav.tail(destination)

        button.click(handler, inputs=None, outputs=outputs, queue=False)

    def refresh() -> list[Any]:
        payload = platform.dashboard()
        weak = platform.progress.weak_areas(limit=5)
        current_stats = payload["stats"]
        return [
            "### Your progress\n\n"
            + table(
                ("Questions answered", "Practice attempted", "Accuracy", "Weak topics"),
                [
                    (
                        current_stats["questions"],
                        current_stats["tests"],
                        pct(current_stats["accuracy"]),
                        chip_line(
                            (area.name for area in weak),
                            empty="Take a practice test to reveal focus topics",
                        ),
                    )
                ],
            ),
            "### Focus topics\n\n"
            + weak_areas_md(
                weak,
                empty_hint="_Your focus topics will appear after your first practice test._",
            ),
            recommended_topic_md(platform),
        ]

    dashboard_refs["refresh"].click(
        refresh,
        inputs=None,
        outputs=[
            dashboard_refs["stats"],
            dashboard_refs["weak"],
            dashboard_refs["recommendation"],
        ],
        queue=False,
    )
