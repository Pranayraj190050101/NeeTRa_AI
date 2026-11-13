"""Reusable presentation helpers shared by every NEETra AI module tab.

The tab modules in this package split in two halves:

* **content functions** - pure Python that turns service data into the markdown
  a tab shows.  They need no Gradio, which is how the whole platform stays
  testable (and importable by ``scripts/offline_check.py``) without a UI
  dependency installed;
* **``build()`` functions** - the Gradio widgets themselves, created with a
  lazily imported ``gradio`` so ``app.py --check`` keeps reporting a readable
  "install Gradio" message instead of an import crash.

Keeping that line in this package means a new module (voice, image solving,
the planner agent) only has to reuse the formatters below to look native.
"""

from __future__ import annotations

import inspect
from datetime import date
from typing import Any, Iterable, Mapping, Sequence

from learning.motivation import daily_motivation
from utils.constants import Subject, TestScope
from utils.text_utils import clean_text

__all__ = [
    "PLATFORM_CSS",
    "MODULE_TABS",
    "TabNav",
    "bar",
    "bullets",
    "chip_line",
    "citations_md",
    "dashboard_heading",
    "doc_md",
    "esc",
    "inject_gradio",
    "lazy_gradio",
    "notices_md",
    "pct",
    "refresh",
    "subject_choices",
    "table",
    "weak_areas_md",
]

#: Appended to the chat's own stylesheet so module tabs share its look.
PLATFORM_CSS = """
body {background: #f6f8fd;}
.gradio-container {background: #f6f8fd; max-width: 1320px !important;}
.gradio-container footer,
.gradio-container .footer,
.gradio-container [data-testid='footer'],
.gradio-container .footer-links,
.gradio-container [class*='footer'],
.gradio-container .icon-button,
.gradio-container button[title='Settings'],
.gradio-container button[aria-label='Settings'],
.gradio-container button[aria-label='Developer mode'],
.gradio-container .developer-toggle {
  display: none !important;
  visibility: hidden !important;
}
.tabs > .tab-nav {gap: 0.35rem; border-bottom: 1px solid #e2e8f0; margin-bottom: 1.25rem;}
.tabs > .tab-nav button {border-radius: 0.75rem 0.75rem 0 0; font-weight: 650; color: #526079;}
.tabs > .tab-nav button.selected {color: #4f46b8; background: #eeecff;}
.genie-card, .genie-main-card, .genie-profile-card, .genie-side-card, .genie-chat-card {
  border: 1px solid #e4e9f2; border-radius: 1.15rem; padding: 1.15rem 1.25rem;
  background: #ffffff; box-shadow: 0 8px 28px rgba(39, 50, 87, 0.055); line-height: 1.5;}
.genie-hero {border-radius: 1.35rem; padding: 1.2rem 1.4rem; margin-bottom: 1rem;
  background: linear-gradient(125deg, #eeecff 0%, #e8f4ff 58%, #e9fbf3 100%);
  border: 1px solid #dfe4fa;}
.genie-hero h2 {font-size: clamp(1.45rem, 3vw, 2rem); color: #2f276f; margin-bottom: 0.45rem;}
.genie-hero p {color: #526079;}
.genie-actions {gap: 0.75rem; margin: 0.5rem 0 1rem;}
.genie-actions button {min-height: 3.1rem; border-radius: 0.85rem; font-weight: 700;}
.genie-dashboard-actions {gap: 0.75rem; align-items: stretch; margin: 0.5rem 0 1rem;}
.genie-action-card {min-width: 0; border: 1px solid #e4e9f2; border-radius: 1rem;
  padding: 0.75rem; background: #fff; box-shadow: 0 6px 20px rgba(39, 50, 87, 0.045);}
.genie-action-card button {min-height: 3rem; width: 100%; border-radius: 0.8rem; font-weight: 700;}
.genie-action-card p {font-size: 0.86rem; color: #64748b; margin: 0.55rem 0 0;}
.genie-suggestions {gap: 0.45rem; flex-wrap: wrap; margin: 0.35rem 0 0.75rem;}
.genie-prompt-chip {min-height: 2rem; border-radius: 999px !important; font-size: 0.82rem;}
.genie-suggestion-label {margin: 0.7rem 0 0;}
.genie-kpi {font-size: 1.35rem; font-weight: 700; color: #17233c;}
.genie-kpi-label {font-size: 0.8rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em;}
.genie-hint {font-size: 0.9rem; color: #526079;}
.developer-toggle {max-width: 13rem; margin-top: 2rem; opacity: 0.72;}
button.primary {background: linear-gradient(110deg, #6356d9, #4e46b8); border: none;}
input, textarea, .wrap {border-radius: 0.75rem !important;}
@media (max-width: 800px) {
  .genie-main-card, .genie-profile-card, .genie-side-card, .genie-chat-card {padding: 0.85rem;}
  .genie-hero {padding: 1.15rem;}
  .genie-actions {flex-direction: column;}
  .genie-dashboard-actions {flex-wrap: wrap;}
  .genie-action-card {flex: 1 1 100% !important;}
  .tabs > .tab-nav {overflow-x: auto; flex-wrap: nowrap;}
  .tabs > .tab-nav button {white-space: nowrap; font-size: 0.86rem;}
}
"""

#: Navigation contract of the platform. ``id`` values are the Gradio tab ids.
#: The student-facing product defaults to Ask → Practice → Progress → Revision
#: → Dashboard, with Learn remaining opt-in.
MODULE_TABS: tuple[tuple[str, str], ...] = (
    ("Ask Genie", "ask-genie"),
    ("Practice", "practice"),
    ("Progress", "progress"),
    ("Revision Hub", "revision"),
    ("Dashboard", "dashboard"),
)


#: A stand-in ``gradio`` module, injected by the test suite (and usable by any
#: headless renderer) so the handlers below can run where Gradio is not installed.
_injected_gradio: Any = None


def inject_gradio(module: Any) -> None:
    """Use ``module`` instead of the real Gradio (tests, headless builds)."""
    global _injected_gradio
    _injected_gradio = module


def lazy_gradio() -> Any:
    """Import Gradio on demand with an actionable error message."""
    if _injected_gradio is not None:
        return _injected_gradio
    try:
        import gradio as gr
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on the env
        raise RuntimeError(
            "Gradio is not installed. Run 'pip install -r requirements.txt' first."
        ) from exc
    return gr


# ---------------------------------------------------------------------------
# Text formatters
# ---------------------------------------------------------------------------
def esc(value: Any) -> str:
    """Make text safe for a markdown table cell."""
    text = value if isinstance(value, str) else ("" if value is None else str(value))
    return clean_text(text).replace("|", "\\|").replace("\n", " ")


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """Markdown table (``rows`` may be empty - the header still explains why)."""
    body = [[esc(cell) for cell in row] for row in rows]
    if not body:
        return "_Nothing here yet._"
    head = " | ".join(esc(header) for header in headers)
    line = "\n".join(f"| {' | '.join(row)} |" for row in body)
    return f"| {head} |\n| {' | '.join(['---'] * len(headers))} |\n{line}"


def bullets(items: Iterable[Any], *, limit: int = 20) -> str:
    values = [value for value in (esc(item) for item in items) if value]
    if not values:
        return "_None yet._"
    return "\n".join(f"- {value}" for value in values[:limit])


def chip_line(values: Iterable[Any], *, empty: str = "none yet") -> str:
    picked = [value for value in (esc(item) for item in values) if value]
    return " · ".join(picked) if picked else empty


def bar(ratio: float, *, width: int = 10) -> str:
    """Text progress bar for an accuracy figure."""
    try:
        filled = int(round(max(0.0, min(1.0, float(ratio))) * width))
    except (TypeError, ValueError):
        filled = 0
    return "\u2588" * filled + "\u2591" * (width - filled)


def pct(value: Any) -> str:
    """``0.7532`` -> ``75%`` (tolerates ``None`` and strings)."""
    try:
        return f"{float(value) * 100:.0f}%"
    except (TypeError, ValueError):
        return "0%"


def subject_choices(catalog: Any | None = None, *, include_general: bool = True) -> list[tuple[str, str]]:
    """Dropdown choices for the subjects actually present in the index."""
    choices: list[tuple[str, str]] = []
    for subject in Subject:
        if subject is Subject.GENERAL and not include_general:
            continue
        choices.append((f"{_SUBJECT_EMOJI[subject.value]} {subject.label}", subject.value))
    if catalog is not None and not getattr(catalog, "is_empty", True):
        known = {value for _, value in choices}
        for name in catalog.subjects:
            if name not in known:
                choices.append((f"\U0001f4da {name.title()}", name))
    return choices


def weak_areas_md(areas: Sequence[Any], *, empty_hint: str = "") -> str:
    """Weak-topic table used by the Progress and Revision tabs."""
    if not areas:
        return empty_hint or "_No weak areas recorded yet - take a test in the Practice tab._"
    return table(
        ("Topic", "Subject", "Where", "Asked", "Wrong", "Accuracy"),
        [
            (
                area.name,
                area.subject,
                area.where,
                area.questions,
                area.wrong,
                f"{area.accuracy_percent}%",
            )
            for area in areas
        ],
    )


def notices_md(notices: Sequence[str], *, title: str = "Coach notes") -> str:
    """Grounding / section-repair notices, deduplicated and ordered."""
    seen: list[str] = []
    for notice in notices or ():
        text = esc(notice)
        if text and text not in seen:
            seen.append(text)
    if not seen:
        return ""
    lines = [f"**{title}**", ""] + [f"- {text}" for text in seen]
    return "\n".join(lines)


def citations_md(citations: Sequence[Mapping[str, Any]] | None) -> str:
    """Render grounding citations with the chat's own formatter."""
    if not citations:
        return "_No textbook reference was used for this document._"
    try:  # reuse the coach's renderer so both halves of the app look identical
        from agents.tutor import citations_to_markdown

        return citations_to_markdown(list(citations))
    except Exception:  # pragma: no cover - agents unavailable
        return table(
            ("#", "File", "Page", "Subject", "Score"),
            [
                (
                    item.get("index"),
                    item.get("file_name"),
                    item.get("page") or "-",
                    item.get("subject"),
                    f"{float(item.get('score') or 0.0):.2f}",
                )
                for item in citations
            ],
        )


def doc_md(doc: Any, *, kicker: str = "") -> str:
    """One generated document (Learn page or Revision sheet) as tab markdown."""
    if doc is None:
        return "_Nothing generated yet._"
    if not getattr(doc, "ok", True):
        return esc(getattr(doc, "markdown", "")) or "**Generation failed.**"
    parts: list[str] = []
    if kicker:
        parts.append(kicker)
    if getattr(doc, "title", ""):
        parts.append(f"## {esc(doc.title)}")
    parts.append(clean_text(str(doc.markdown or "")))
    notices = notices_md(getattr(doc, "notices", ()))
    if notices:
        parts.append(notices)
    return "\n\n".join(piece for piece in parts if piece).strip()


def dashboard_heading(app_name: str, tagline: str = "", *, today: date | None = None) -> str:
    """Welcome line + the daily motivation the dashboard opens with."""
    stamp = today or date.today()
    line = daily_motivation(on_date=stamp).line
    intro = (
        f"### \U0001f9de {esc(app_name)} - your personal NEET doubt solver, "
        "practice partner and rank improver"
    )
    if tagline:
        intro += f"\n\n_{esc(tagline)}_"
    return f"{intro}\n\n> **Today ({stamp:%d %b %Y}):** {esc(line)}"


def scope_hint(scope: TestScope | str | None) -> str:
    """What a test scope actually retrieves - shown next to the picker."""
    item = scope if isinstance(scope, TestScope) else TestScope.coerce(scope)
    return f"{item.emoji} **{item.label}** - {item.description}"


def refresh(**props: Any) -> Any:
    """``gr.update`` wrapper kept in one place (and a no-op without Gradio)."""
    gr = lazy_gradio()
    return gr.update(**props)


_SUBJECT_EMOJI: dict[str, str] = {
    "physics": "\U0001f576\ufe0f",
    "chemistry": "\U0001f9ea",
    "biology": "\U0001f331",
    "general": "\U0001f4da",
}


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
class TabNav:
    """Programmatic tab switching that degrades across Gradio versions.

    ``gr.Tabs`` only learned a ``selected`` prop recently, so the quick-action
    buttons ask this helper whether a jump is possible.  When it is not, the
    guidance markdown still tells the student where to go and the target tab's
    controls are pre-configured, so the shortcut keeps working.
    """

    def __init__(self, gr: Any, *, ids: Mapping[str, str] | None = None) -> None:
        self.gr = gr
        self.ids = dict(ids or {key: key for key, _ in MODULE_TABS})
        self.tabs: Any = None
        try:
            self.supports_selected = "selected" in inspect.signature(gr.Tabs.__init__).parameters
        except (TypeError, ValueError):  # pragma: no cover - exotic gradio build
            self.supports_selected = False

    def attach(self, tabs: Any) -> "TabNav":
        """Remember the ``gr.Tabs`` component created by the app."""
        self.tabs = tabs
        return self

    def tab(self, label: str, key: str) -> Any:
        """Create one module tab carrying the id :meth:`jump` selects.

        Called inside the ``gr.Tabs`` context, in navigation order.  A Gradio
        build that rejects ``id`` on a tab cannot be jumped to, so switching is
        disabled for the whole app rather than half-wiring the shortcuts.
        """
        try:
            return self.gr.Tab(label, id=key)
        except TypeError:  # pragma: no cover - older gradio
            self.supports_selected = False
            return self.gr.Tab(label)

    @property
    def outputs(self) -> list[Any]:
        """Outputs a handler must declare to be allowed to switch tabs."""
        return [self.tabs] if (self.supports_selected and self.tabs is not None) else []

    def jump(self, key: str) -> Any | None:
        """Update value selecting tab ``key`` (``None`` when unsupported)."""
        if not self.outputs:
            return None
        target = self.ids.get(key, key)
        try:
            return self.gr.Tabs(selected=target)
        except TypeError:  # pragma: no cover - defensive across versions
            return None

    def tail(self, key: str) -> list[Any]:
        """Values to append when :attr:`outputs` is declared last in a handler."""
        return [self.jump(key)] if self.outputs else []
