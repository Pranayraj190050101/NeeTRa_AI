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
.genie-card {border: 1px solid #e5e7eb; border-radius: 12px; padding: 0.85rem 1rem;
  background: #ffffff; line-height: 1.4;}
.genie-kpi {font-size: 1.35rem; font-weight: 600; color: #111827;}
.genie-kpi-label {font-size: 0.8rem; color: #6b7280; text-transform: uppercase;
  letter-spacing: 0.04em;}
.genie-hint {font-size: 0.85rem; color: #4b5563;}
#genie-dashboard .gradio-container {max-width: 1240px !important;}
"""

#: Navigation contract of the platform. ``id`` values are the Gradio tab ids.
MODULE_TABS: tuple[tuple[str, str], ...] = (
    ("Dashboard", "dashboard"),
    ("Learn", "learn"),
    ("Practice", "practice"),
    ("Ask Genie", "ask-genie"),
    ("Progress", "progress"),
    ("Revision Hub", "revision"),
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
        return "_No indexed passage was used for this document._"
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
        if self.supports_selected:
            try:  # some builds accept the prop in the signature but not at init
                self.gr.Tabs(selected="__probe__")
            except TypeError:  # pragma: no cover - older gradio
                self.supports_selected = False
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
