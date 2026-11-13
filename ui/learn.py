"""Learn tab - subject -> chapter -> generated study page.

The hierarchy is read from the index (``learning.catalog``) and the page body is
generated with RAG by ``learning.learn.LearnService``: Summary, High-Yield
Concepts, Important NEET Points and Quick Revision Notes.  Nothing here holds
study text of its own, so the tab is only as good as the student's corpus - and
says so when the corpus is empty.
"""

from __future__ import annotations

from typing import Any, Mapping

from learning.catalog import ChapterCatalog
from learning.service import GeniePlatform
from learning.sections import CHAPTER_SECTIONS, format_section_contract
from ui.components import esc, hide_source_markers, notices_md, refresh, subject_choices, table

__all__ = [
    "build",
    "chapter_choices",
    "empty_catalog_md",
    "learn_intro_md",
    "page_md",
    "study",
    "wire",
]

INTRO = (
    "Pick a subject and chapter for a guided study page with a summary, high-yield "
    "concepts, NEET points and quick revision notes."
)


def learn_intro_md(catalog: ChapterCatalog) -> str:
    """Section contract plus the chapters that are actually available."""
    lines = [
        f"### \U0001f4da Learn - {len(catalog.chapters)} chapter(s) in your study library",
        "",
        INTRO,
        "",
        "**Every chapter page has these sections:**",
        format_section_contract(CHAPTER_SECTIONS),
    ]
    if not catalog.is_empty:
        rows = [
            (
                chapter.subject.title(),
                chapter.name,
                "Guided study page",
            )
            for chapter in catalog.chapters
        ]
        lines.append("\n**Available chapters**\n\n" + table(("Subject", "Chapter", "Study sections"), rows))
    return "\n".join(lines)


def empty_catalog_md() -> str:
    """What to tell a student when nothing is indexed."""
    return (
        "**Your study library is being prepared.**\n\n"
        "You can still type a topic below or ask Genie a doubt. Once your NCERT books are "
        "available, their chapters will appear here automatically."
    )


def chapter_choices(catalog: ChapterCatalog, subject: str | None) -> list[tuple[str, str]]:
    """Chapter dropdown choices for one subject (all subjects when blank)."""
    return [
        (
            chapter.name,
            chapter.name,
        )
        for chapter in catalog.list(subject)
    ]


def study(platform: GeniePlatform, subject: str | None, chapter: str, topic: str) -> Any:
    """Run the Learn service for the tab's current selection."""
    chapter_name = esc(chapter)
    topic_name = esc(topic)
    if topic_name:
        return platform.study_topic(topic_name, subject, focus=chapter_name)
    if chapter_name:
        return platform.study_chapter(chapter_name, subject)
    return None


def status_md(doc: Any) -> str:
    """One-line result of a generation."""
    if doc is None:
        return "**Status:** nothing generated - select a chapter or name a topic."
    icon = "\U0001f4d8" if doc.kind == "chapter" else "\U0001f50e"
    state = "ready" if doc.ok else "please try again"
    return f"**Status:** {icon} {esc(doc.title)} — {state}"


def page_md(doc: Any) -> str:
    """Generated chapter page with its notices."""
    if doc is None:
        return "_Nothing generated yet._"
    if not getattr(doc, "ok", True):
        return esc(doc.markdown) or "**Generation failed.**"
    body = f"## {esc(doc.title)}\n\n{hide_source_markers(doc.markdown)}"
    return "\n\n".join(part for part in (body, notices_md(doc.notices)) if part)


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
    autowire: bool = True,
) -> dict[str, Any]:
    """Create the Learn tab widgets and return the widget references.

    ``autowire=False`` only builds the widgets: ``app.py`` uses that because the
    Ask Genie tab is created after Learn, so its cross-tab targets are not known
    yet and :func:`wire` is called once the whole navigation exists.
    """
    catalog = platform.catalog()
    intro_md = gr.Markdown(learn_intro_md(catalog))
    if catalog.is_empty:
        gr.Markdown(empty_catalog_md())

    with gr.Row():
        subject_box = gr.Dropdown(
            choices=subject_choices(catalog),
            value=(catalog.subjects[0] if catalog.subjects else "physics"),
            label="Subject",
        )
        chapter_box = gr.Dropdown(
            choices=chapter_choices(catalog, subject_box.value),
            value=None,
            allow_custom_value=True,
            label="Chapter",
            info="Choose from your study library or type a chapter name.",
        )
    topic_box = gr.Textbox(
        label="Or study a free-text topic",
        placeholder="e.g. photoelectric effect, mole concept conversions, human heart physiology",
        lines=1,
    )
    with gr.Row():
        generate_btn = gr.Button("Generate study page", variant="primary", scale=3)
        refresh_btn = gr.Button("Refresh chapters", size="sm", scale=1)
    status_box = gr.Markdown("_Pick a chapter (or type a topic), then press Generate._")
    page_box = gr.Markdown()
    ask_btn = gr.Button("Ask Genie about this chapter")
    ask_md = gr.Markdown()

    refs: dict[str, Any] = {
        "intro": intro_md,
        "subject": subject_box,
        "chapter": chapter_box,
        "topic": topic_box,
        "generate": generate_btn,
        "refresh": refresh_btn,
        "status": status_box,
        "page": page_box,
        "ask": ask_btn,
        "ask_md": ask_md,
    }
    if autowire:
        wire(gr, platform, refs, nav=nav, targets=targets or {})
    return refs


def wire(
    gr: Any,
    platform: GeniePlatform,
    refs: dict[str, Any],
    *,
    nav: Any | None = None,
    targets: Mapping[str, Any] | None = None,
) -> None:
    """Bind the Learn tab's controls to :class:`learning.learn.LearnService`."""
    targets = targets or {}

    def pick_chapters(subject: str) -> Any:
        choices = chapter_choices(platform.catalog(), subject)
        return refresh(choices=choices, value=choices[0][1] if choices else None)

    def reload_catalog() -> list[Any]:
        catalog = platform.refresh_catalog()
        subjects = subject_choices(catalog)
        first = subjects[0][1] if subjects else "physics"
        return [
            refresh(choices=subjects, value=first),
            refresh(choices=chapter_choices(catalog, first), value=None),
            learn_intro_md(catalog),
        ]

    def generate(subject: str, chapter: str, topic: str) -> list[Any]:
        doc = study(platform, subject, chapter, topic)
        if doc is None:
            hint = empty_catalog_md() if platform.catalog().is_empty else "_Pick a chapter or name a topic._"
            return [hint, status_md(None)]
        return [page_md(doc), status_md(doc)]

    def draft_question(chapter: str, topic: str) -> list[Any]:
        focus = esc(chapter) or esc(topic) or "this chapter"
        seed = f"Explain {focus} the way NEET asks it, with the common trap and one MCQ to check me."
        values: list[Any] = [f"**Drafted for Ask Genie:** _{seed}_"]
        if "ask_message" in targets:
            values.append(refresh(value=seed))
        if "ask_mode" in targets:
            values.append(refresh(value="detailed"))
        return values + (nav.tail("ask-genie") if nav is not None else [])

    refs["subject"].change(pick_chapters, inputs=[refs["subject"]], outputs=[refs["chapter"]], queue=False)
    refs["refresh"].click(
        reload_catalog,
        inputs=None,
        outputs=[refs["subject"], refs["chapter"], refs["intro"]],
        queue=False,
    )
    refs["generate"].click(
        generate,
        inputs=[refs["subject"], refs["chapter"], refs["topic"]],
        outputs=[refs["page"], refs["status"]],
    )

    ask_outputs = [refs["ask_md"]]
    for name in ("ask_message", "ask_mode"):
        if name in targets:
            ask_outputs.append(targets[name])
    ask_outputs.extend(nav.outputs if nav is not None else [])
    refs["ask"].click(draft_question, inputs=[refs["chapter"], refs["topic"]], outputs=ask_outputs)
