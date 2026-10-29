"""NEETra AI - Gradio application for the NEETra AI preparation assistant.

Run::

    python app.py                      # start the UI (models load on first question)
    python app.py --check              # validate configuration and exit
    python app.py --ingest             # build the knowledge base from NCERT chapter PDFs
    python app.py --host 0.0.0.0 --port 7860 --share

The product is a NEET preparation assistant with **Ask Genie - the doubt-
solving coach chat - as its primary module**.  Its navigation is **Ask Genie**,
**Dashboard**, **Practice**, **Progress** and **Revision Hub**, followed by the
three diagnostic tabs the tutor already shipped with (Knowledge base, Coach,
Architecture).  The optional **Learn** module stays available behind the
``NEETRA_ENABLE_LEARN`` feature flag (on by default; ``false`` hides the tab and
never builds its widgets).  Ask Genie *is* the original coach chat with all
four answer modes; it was not rewritten, it simply leads the app.

The chat still streams three things while a question is being answered:

1. live status from every agent (routing -> retrieval -> generation),
2. the answer tokens themselves (``TextIteratorStreamer``),
3. the textbook passages that grounded the answer, with page numbers.

The sidebar of Ask Genie carries the coach controls: the **answer mode**
(Detailed Explanation / Quick Revision / MCQ Practice / PYQ Focus) that decides
the section shape of every reply.

All module content comes from the ``learning/`` services (chapter catalog,
generator, progress store, capability registry) and every widget from ``ui/``,
which is why this file only arranges tabs and hands the tabs their services.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
import warnings
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as metadata_version
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - import plumbing
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.state import TutorResult  # noqa: E402
from agents.tutor import NEETTutor, citations_to_markdown  # noqa: E402
from agents.registry import describe_registry  # noqa: E402
from config import BASE_DIR, ProviderConfigError, Settings, get_settings  # noqa: E402
from learning.capabilities import default_registry, describe_capabilities  # noqa: E402
from utils.answer_format import format_contract  # noqa: E402
from utils.constants import AnswerMode  # noqa: E402
from utils.logging_utils import get_logger, setup_logging  # noqa: E402
from utils.text_utils import format_seconds  # noqa: E402
from ui import components as genie_ui  # noqa: E402
from ui import dashboard as dashboard_ui  # noqa: E402
from ui import learn as learn_ui  # noqa: E402
from ui import practice as practice_ui  # noqa: E402
from ui import progress as progress_ui  # noqa: E402
from ui import revision as revision_ui  # noqa: E402

logger = get_logger(__name__)

SUBJECT_CHOICES: list[str] = ["auto", "physics", "chemistry", "biology", "general"]
STREAM_THROTTLE_S: float = 0.06

EXAMPLES: list[list[str]] = [
    ["Explain the photoelectric effect and write Einstein's photoelectric equation."],
    ["Why is the boiling point of water higher than that of hydrogen sulphide?"],
    ["Differentiate between mitosis and meiosis with the number of divisions."],
    ["Solve: a body starts from rest and accelerates at 2 m/s^2 for 5 s. Find its final velocity."],
    ["Give me the whole of chemical bonding as a one-screen revision sheet."],
    ["Frame practice MCQs on Newton's laws and show where students lose marks."],
]

CSS = """
.gradio-container {max-width: 1240px !important;}
#neet-header {margin-bottom: 0.25rem;}
#neet-sub {color: #6b7280; margin-top: 0; margin-bottom: 1rem;}
#neet-status {min-height: 2.2rem; font-size: 0.92rem;}
#neet-mode-help {font-size: 0.86rem; color: #4b5563; margin-top: -0.4rem;}
.source-card {line-height: 1.35;}
"""


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def index_panel(tutor: NEETTutor) -> str:
    """Knowledge base summary shown in the chat sidebar."""
    status = tutor.index_status()
    if not status["ready"]:
        return (
            "**Knowledge base: empty**\n\n"
            "No FAISS index found. Add PDFs to `data/physics`, `data/chemistry` or "
            "`data/biology`, then run:\n\n"
            "```bash\npython -m rag.ingest --rebuild\n```\n\n"
            "Until then the tutor answers from general NEET knowledge."
        )

    subjects = status.get("subjects") or {}
    subject_line = (
        " · ".join(f"{name}: {count}" for name, count in sorted(subjects.items()))
        if subjects
        else "n/a"
    )
    updated = status.get("updated_at") or "unknown"
    return (
        "**Knowledge base: ready**\n\n"
        f"- Chunks: **{status['chunks']}**\n"
        f"- Source files: **{status['files']}**\n"
        f"- Chunks per subject: {subject_line}\n"
        f"- Embeddings: `{status['embedding_model']}`\n"
        f"- Last ingested: {updated}\n"
        f"- Folder: `{status['path']}`"
    )


def settings_panel(settings: Settings, tutor: NEETTutor) -> str:
    """Model + prompt diagnostics for the sidebar."""
    problems = tutor.prompts.validate()
    prompt_line = "prompts OK" if not problems else f"{len(problems)} prompt problem(s)"
    persona = "on (15-yr NCERT faculty)" if settings.persona_enabled else "off"
    contract = "enforced" if settings.enforce_answer_format else "as written by the model"
    provider = settings.llm_provider_diagnostic()
    provider_line = f"- Provider: `{provider['provider']}`"
    if provider["provider"] == "groq":
        provider_line += f" - API key `{provider['api_key']}`"
        if not provider.get("configured", True):
            provider_line += " ⚠ GROQ_API_KEY missing"
    return (
        "**Engine**\n\n"
        f"{provider_line}\n"
        f"- Active model: `{provider['model']}`\n"
        f"- Embeddings: `{settings.embedding_model}`\n"
        f"- Chunk size / overlap: {settings.chunk_size} / {settings.chunk_overlap}\n"
        f"- Retrieval: top-{settings.top_k}, score >= {settings.score_threshold}\n"
        f"- {prompt_line}\n\n"
        "**Coach**\n\n"
        f"- Persona: {persona}\n"
        f"- Default mode: {settings.answer_mode.emoji} {settings.answer_mode.label}\n"
        f"- Section contract: {contract}\n"
        f"- MCQs per practice set: {settings.mcq_count}"
    )


def coach_panel(settings: Settings) -> str:
    """Persona + the answer-format contract of every mode (Coach tab)."""
    persona = (
        "on - 15 years of NEET faculty, NCERT-first, names the trap, flags the "
        "high-yield line" if settings.persona_enabled else "off (NEET_PERSONA_ENABLED=false)"
    )
    contract = (
        "repaired to the mode shape before it reaches the chat"
        if settings.enforce_answer_format
        else "left as the model wrote it (NEET_ENFORCE_ANSWER_FORMAT=false)"
    )
    rows = "\n".join(
        f"| {mode.emoji} {mode.label} | `{mode.value}` | {mode.description} | "
        f"{format_contract(mode).replace(' | ', '<br>')} |"
        for mode in AnswerMode
    )
    return (
        "### Tutor persona\n\n"
        f"- Faculty persona: {persona}\n"
        f"- Default mode: {settings.answer_mode.emoji} {settings.answer_mode.label}\n"
        f"- Section contract: {contract}\n"
        f"- MCQ Practice set size: {settings.mcq_count}\n\n"
        "### Answer modes\n\n"
        "| Mode | Key | What it is for | Sections produced |\n"
        "| --- | --- | --- | --- |\n"
        f"{rows}\n\n"
        "_Modes only change presentation. Retrieval, the FAISS index and the agent "
        "graph are identical for all four, so switching modes never costs a re-index._"
    )


def architecture_panel(tutor: NEETTutor) -> str:
    """Markdown overview of the agent graph and the planned agents."""
    registry = describe_registry()
    rows = "\n".join(
        f"| {entry['title']} | `{entry['intent']}` | {entry['status']} | {entry['description']} |"
        for entry in registry
    )
    return (
        "### Agent graph\n\n"
        "```text\n"
        f"{tutor.architecture()}\n"
        "```\n\n"
        "### Agent registry\n\n"
        "| Agent | Intent | Status | Responsibility |\n"
        "| --- | --- | --- | --- |\n"
        f"{rows}\n\n"
        "### Adding a new agent\n\n"
        "1. Create `agents/mcq_agent.py` with a class deriving from `agents.base.BaseAgent`.\n"
        "2. Instantiate it in `agents/graph.py` and add the node plus a conditional edge.\n"
        "3. Flip its status to `active` in `agents/registry.py`.\n\n"
        "The router already emits `mcq`, `pyq`, `study_plan` and `weakness` intents, and the "
        "shared state (`agents/state.py`) already carries retrieved passages, so new agents can "
        "reuse the retrieval layer without changes."
    )


def platform_panel(platform: Any) -> str:
    """Module map of the platform, its data sources and its extension points."""
    status = platform.platform_status()
    catalog = status["catalog"]
    progress = status["progress"]
    subjects = catalog.get("subjects") or {}
    sheets = " · ".join(f"{item['emoji']} {item['label']}" for item in status["revision"])
    return (
        "### Modules\n\n"
        "| Tab | Backed by | Data source |\n"
        "| --- | --- | --- |\n"
        "| Dashboard | `learning.service.GeniePlatform.dashboard()` | progress store + chapter catalog |\n"
        "| Learn | `learning.learn.LearnService` | FAISS retrieval, four-section contract |\n"
        "| Practice | `learning.practice.PracticeService` + `learning.assessment` | generated MCQs, NEET marking |\n"
        "| Ask Genie | `agents.tutor.NEETTutor` (LangGraph, unchanged) | retrieval + the coach persona |\n"
        f"| Progress | `learning.progress.ProgressStore` (JSON now, DB later) | `{progress.get('path')}` |\n"
        "| Revision Hub | `learning.revision.RevisionService` | same RAG, condensed deliverables |\n\n"
        "### Syllabus and history\n\n"
        f"- Chapters recognised: **{catalog.get('chapters', 0)}** across "
        f"{len(subjects)} subject(s)\n"
        f"- Tests stored: **{progress.get('tests', 0)}** · questions answered: "
        f"{progress.get('questions', 0)} · doubts asked: {progress.get('doubts', 0)}\n"
        f"- Revision deliverables: {sheets}\n\n"
        "### Extension points (not implemented yet)\n\n"
        "Voice and photo solving are registered capabilities with a typed "
        "protocol each; both funnel into `GeniePlatform.ask()`, so they reuse "
        "the coach pipeline as it is.\n\n"
        f"{describe_capabilities(platform.capabilities)}\n\n"
        "_Registering a backend is one call - "
        "`platform.capabilities.register('voice', my_transcriber)` - and the "
        "Ask Genie tab's hint disappears on its own._"
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _status_line(message: str) -> str:
    return f"**Status:** {message}"


def _mode_help(mode: AnswerMode) -> str:
    """Sidebar hint describing the sections the selected mode produces."""
    return (
        f"{mode.emoji} **{mode.label}** - {mode.description}\n\n"
        f"Sections: {format_contract(mode)}"
    )


def _history_for_model(chat: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chat turns excluding the current message, trimmed to the configured depth."""
    return [
        {"role": str(turn.get("role", "user")), "content": str(turn.get("content", ""))}
        for turn in chat
        if turn.get("content")
    ]


def _completion_status(result: TutorResult, settings: Settings) -> str:
    """Final status line: routing decision, mode, timings and warnings."""
    mode = AnswerMode.coerce(getattr(result, "mode", None), default=settings.answer_mode)
    parts = [
        f"**Subject:** {result.subject}",
        f"**Intent:** {result.intent}",
        f"**Mode:** {mode.emoji} {mode.label}",
        f"**Sources:** {len(result.citations)}",
        f"**Time:** {format_seconds(result.total_seconds)}",
        f"**Model:** `{settings.llm_model}`" if settings.llm_backend == "hf" else "**Model:** mock backend",
    ]
    line = " · ".join(parts)
    if result.notices:
        warnings = "\n".join(f"- {notice}" for notice in result.notices)
        line += f"\n\n**Notes**\n{warnings}"
    if result.error:
        line += f"\n\n**Error:** `{result.error}`"
    return line


# ---------------------------------------------------------------------------
# Chat handler (generator -> streams status, tokens and sources)
# ---------------------------------------------------------------------------
def build_responder(tutor: NEETTutor, settings: Settings, on_result: Any = None):
    """Create the Gradio callback bound to a tutor instance.

    ``on_result`` is an optional observer called with the finished
    :class:`agents.state.TutorResult`; the platform uses it to count a doubt in
    the progress store.  It never changes the answer, and a failure inside the
    observer cannot break the chat turn.
    """

    def respond(
        message: str,
        chat: list[dict[str, Any]] | None,
        subject: str,
        mode: str,
        top_k: float,
        stream_enabled: bool,
    ) -> Iterator[tuple[list[dict[str, Any]], str, str]]:
        history = list(chat or [])
        text = (message or "").strip()

        if not text:
            yield history, _status_line("Please type a question."), ""
            return
        if len(text) > settings.max_question_chars:
            text = text[: settings.max_question_chars]
            logger.info("Question truncated to %d characters", settings.max_question_chars)

        answer_mode = AnswerMode.coerce(mode, default=settings.answer_mode)
        prior_turns = _history_for_model(history)
        history.append({"role": "user", "content": text})

        def snapshot(answer: str) -> list[dict[str, Any]]:
            return [*history, {"role": "assistant", "content": answer}]

        streaming = bool(stream_enabled and settings.stream_tokens)
        events: "queue.Queue[tuple[str, Any]]" = queue.Queue()
        box: dict[str, Any] = {}

        def on_event(payload: dict[str, Any]) -> None:
            events.put(("event", payload))

        def on_token(fragment: str) -> None:
            events.put(("token", fragment))

        def worker() -> None:
            try:
                box["result"] = tutor.ask(
                    text,
                    history=prior_turns,
                    subject=subject or "auto",
                    top_k=int(top_k),
                    mode=answer_mode,
                    on_event=on_event,
                    on_token=on_token if streaming else None,
                )
                if on_result is not None:
                    try:
                        on_result(box["result"])
                    except Exception:  # pragma: no cover - bookkeeping only
                        logger.exception("Result observer failed")
            except Exception as exc:  # never let the UI hang silently
                logger.exception("Chat handler crashed")
                box["error"] = exc
            finally:
                events.put(("done", None))

        thread = threading.Thread(target=worker, name="neet-chat-turn", daemon=True)
        thread.start()

        partial = ""
        status = _status_line(f"{answer_mode.emoji} {answer_mode.label} - starting the agent graph")
        last_emit = 0.0
        yield snapshot(partial), status, ""

        while True:
            kind, payload = events.get()
            if kind == "done":
                break
            if kind == "token":
                partial += str(payload)
                now = time.monotonic()
                if now - last_emit >= STREAM_THROTTLE_S:
                    last_emit = now
                    yield snapshot(partial), status, ""
            elif kind == "event" and isinstance(payload, dict):
                status = _status_line(str(payload.get("message", "Working")))
                yield snapshot(partial), status, ""

        thread.join(timeout=5.0)
        result = box.get("result")
        if result is None:
            error = box.get("error")
            logger.error("Turn produced no result: %s", error)
            message = (
                "**The request failed unexpectedly.**\n\n"
                f"`{type(error).__name__ if error else 'UnknownError'}: {error}`\n\n"
                "Please retry; if it persists check `logs/neet_tutor.log`."
            )
            yield snapshot(message), _status_line("Failed"), ""
            return

        sources = citations_to_markdown(result.citations)
        yield snapshot(result.answer), _completion_status(result, settings), sources

    return respond


# ---------------------------------------------------------------------------
# Ask Genie - the coach interface, reused as one module of the platform
# ---------------------------------------------------------------------------
def build_ask_genie_tab(
    gr: Any,
    settings: Settings,
    tutor: NEETTutor,
    responder: Any,
    capabilities: Any,
) -> dict[str, Any]:
    """Create the Ask Genie module - the original chat, widget for widget.

    The brief is explicit: reuse the coach interface, do not replace it.  The
    chatbot, the question box, the four answer modes, the subject override, the
    passage slider, the streaming switch, the knowledge-base sidebar and every
    handler are the ones the tutor shipped with; only the tab name changed.  The
    refs are returned so the module tabs (Learn, Revision Hub) and the Dashboard
    shortcuts can pre-fill a doubt once the whole navigation exists.
    """
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation",
                type="messages",
                height=settings.chatbot_height,
                show_copy_button=True,
                latex_delimiters=[
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": "$", "right": "$", "display": False},
                ],
            )
            message_box = gr.Textbox(
                placeholder="Ask any NEET doubt, e.g. 'Why does SN1 favour tertiary carbocations?'",
                label="",
                autofocus=True,
                lines=1,
                max_lines=4,
            )
            with gr.Row():
                send_btn = gr.Button("Ask", variant="primary", scale=3)
                clear_btn = gr.ClearButton([chatbot], value="New chat", scale=1)
            status_box = gr.Markdown(
                _status_line("Ready. Ask your first doubt."), elem_id="neet-status"
            )
            with gr.Accordion("Sources from your textbooks", open=False):
                sources_box = gr.Markdown(
                    "_Sources appear here after the first answer._",
                    elem_classes=["source-card"],
                )
            gr.Examples(examples=EXAMPLES, inputs=[message_box], label="Try one")

        with gr.Column(scale=1):
            gr.Markdown("### Session settings")
            subject_box = gr.Dropdown(
                choices=SUBJECT_CHOICES,
                value="auto",
                label="Subject",
                info="'auto' lets the router agent decide. Picking a subject overrides it.",
            )
            mode_box = gr.Radio(
                choices=AnswerMode.ui_choices(),
                value=settings.answer_mode.value,
                label="Answer mode",
                info="How the coach shapes every reply - "
                + " / ".join(f"{mode.emoji} {mode.label}" for mode in AnswerMode),
            )
            mode_md = gr.Markdown(
                _mode_help(settings.answer_mode), elem_id="neet-mode-help"
            )
            top_k_box = gr.Slider(
                minimum=2,
                maximum=10,
                value=settings.top_k,
                step=1,
                label="Passages to retrieve",
            )
            stream_box = gr.Checkbox(
                value=settings.stream_tokens,
                label="Stream the answer while it is written",
                # The setting is the operator's kill switch for slow machines.
                interactive=settings.stream_tokens,
            )
            kb_md = gr.Markdown(index_panel(tutor))
            engine_md = gr.Markdown(settings_panel(settings, tutor))
            reload_btn = gr.Button("Reload knowledge base", size="sm")

    mode_box.change(
        lambda choice: _mode_help(AnswerMode.coerce(choice, default=settings.answer_mode)),
        inputs=[mode_box],
        outputs=[mode_md],
        queue=False,
    )

    send_btn.click(
        responder,
        inputs=[message_box, chatbot, subject_box, mode_box, top_k_box, stream_box],
        outputs=[chatbot, status_box, sources_box],
    ).then(lambda: "", outputs=[message_box], queue=False)

    message_box.submit(
        responder,
        inputs=[message_box, chatbot, subject_box, mode_box, top_k_box, stream_box],
        outputs=[chatbot, status_box, sources_box],
    ).then(lambda: "", outputs=[message_box], queue=False)

    clear_btn.click(
        lambda: (_status_line("New chat started. Ask your doubt."), ""),
        outputs=[status_box, sources_box],
        queue=False,
    )

    reload_btn.click(
        lambda: (index_panel(tutor), _status_line("Knowledge base reloaded from disk.")),
        outputs=[kb_md, status_box],
        queue=False,
    )

    with gr.Accordion("Voice and photo doubt - extension points", open=False):
        gr.Markdown(
            "Two more input channels are planned in `learning/capabilities.py`, "
            "and both would reuse this exact pipeline through "
            "`GeniePlatform.ask()` / `listen()` / `solve_photo()`:\n\n"
            f"{describe_capabilities(capabilities)}\n\n"
            "_Nothing here is clickable yet - the coach works with text, which is "
            "all this build claims._"
        )

    return {
        "chatbot": chatbot,
        "message": message_box,
        "send": send_btn,
        "clear": clear_btn,
        "status": status_box,
        "sources": sources_box,
        "subject": subject_box,
        "mode": mode_box,
        "mode_help": mode_md,
        "top_k": top_k_box,
        "stream": stream_box,
        "knowledge": kb_md,
        "engine": engine_md,
        "reload": reload_btn,
    }


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def build_app(settings: Settings, tutor: NEETTutor | None = None) -> "gr.Blocks":
    """Construct the Gradio Blocks app - the module tabs plus the diagnostics.

    Gradio is resolved here rather than at module level, so ``app.py --check``
    reports a missing UI dependency instead of crashing on import, and so a test
    can inject a stand-in module (``ui.components.inject_gradio``) and build the
    whole navigation on a machine where Gradio is not installed.

    ``NEET_PLATFORM_ENABLED=false`` serves the original single-chat app: the
    module tabs and the progress store are then never created, which is the
    operator's kill switch if a 7B model needs every bit of RAM.

    ``NEETRA_ENABLE_LEARN=false`` keeps the platform but drops the Learn tab:
    its widgets are never instantiated.  ``learning/learn.py`` and its tests
    are untouched, and the environment can bring the module back.
    """
    gr = genie_ui.lazy_gradio()

    tutor = tutor or NEETTutor(settings)
    modular = bool(settings.platform_enabled)
    platform = tutor.platform() if modular else None
    registry = platform.capabilities if platform is not None else default_registry()

    def count_doubt(result: TutorResult) -> None:
        """Ask Genie keeps the platform's statistics honest."""
        if platform is not None and getattr(result, "ok", True):
            platform.progress.bump("doubts")

    responder = build_responder(tutor, settings, on_result=count_doubt)
    nav = genie_ui.TabNav(gr)
    learn_refs: dict[str, Any] | None = None  # set only when the Learn tab is built

    with gr.Blocks(
        title=settings.app_name, theme=gr.themes.Soft(), css=CSS + genie_ui.PLATFORM_CSS
    ) as demo:
        gr.Markdown(f"# {settings.app_name}", elem_id="neet-header")
        gr.Markdown(
            (
                "**Doubt solving** with a grounded coach chat, **practice tests** and "
                "**full mock exams** with NEET marking, **progress analytics** that find "
                "the weak topics, **revision planning** in one click - your personal "
                "NEET preparation assistant, grounded in **your** textbooks with chapter "
                "and page citations."
            )
            if modular
            else (
                f"{settings.app_tagline} - a NEET faculty with 15 years of NCERT teaching, answering in "
                "a fixed coach format and grounded in **your** textbooks, with chapter and page "
                "citations. _Module tabs are switched off (`NEET_PLATFORM_ENABLED=false`)._"
            ),
            elem_id="neet-sub",
        )

        with gr.Tabs() as tabs:
            nav.attach(tabs)

            if modular:
                # ------------------------------------------------------ ask genie
                # the primary module: doubt solving leads the navigation
                with nav.tab("Ask Genie", "ask-genie"):
                    genie_refs = build_ask_genie_tab(gr, settings, tutor, responder, registry)

                # -------------------------------------------------- dashboard
                with nav.tab("Dashboard", "dashboard"):
                    dashboard_refs = dashboard_ui.build(gr, platform, settings, nav)

                # --------------------------------------------------- practice
                with nav.tab("Practice", "practice"):
                    practice_refs = practice_ui.build(gr, platform, settings, nav)

                # ------------------------------------------------------- learn
                if settings.learn_enabled:
                    # optional study pages: hidden by NEETRA_ENABLE_LEARN=false
                    # (wired below, once the Ask Genie controls exist)
                    with nav.tab("Learn", "learn"):
                        learn_refs = learn_ui.build(gr, platform, settings, nav, autowire=False)

                # --------------------------------------------------- progress
                with nav.tab("Progress", "progress"):
                    progress_ui.build(gr, platform, settings, nav)

                # ----------------------------------------------- revision hub
                with nav.tab("Revision Hub", "revision"):
                    revision_refs = revision_ui.build(gr, platform, settings, nav, autowire=False)

            else:
                # ------------------------------------------------------ ask genie
                with gr.Tab("Ask Genie"):
                    genie_refs = build_ask_genie_tab(gr, settings, tutor, responder, registry)

            # --------------------------------------------- knowledge base
            with gr.Tab("Knowledge base"):
                gr.Markdown("## Indexed study material")
                gr.Markdown(
                    "Drop textbook PDFs into the subject folders and re-run ingestion:\n\n"
                    "```bash\n"
                    "python -m rag.ingest --rebuild     # full rebuild\n"
                    "python -m rag.ingest --append      # add only new pages\n"
                    "```\n\n"
                    "Folder layout drives the subject metadata:\n\n"
                    "```text\n"
                    "data/\n"
                    "├── physics/      *.pdf\n"
                    "├── chemistry/    *.pdf\n"
                    "└── biology/      *.pdf\n"
                    "```\n\n"
                    "Scanned PDFs without a text layer must be OCR'd first "
                    "(`ocrmypdf input.pdf output.pdf`)."
                )
                kb_full = gr.Markdown(index_panel(tutor) + "\n\n" + settings_panel(settings, tutor))
                kb_refresh = gr.Button("Refresh", size="sm")
                kb_refresh.click(
                    lambda: index_panel(tutor) + "\n\n" + settings_panel(settings, tutor),
                    outputs=[kb_full],
                )

            # ----------------------------------------------------- coach persona
            with gr.Tab("Coach"):
                gr.Markdown("## NEETra AI - the Neet Genie coach persona, format and answer modes")
                gr.Markdown(coach_panel(settings))

            # ----------------------------------------------------- architecture
            with gr.Tab("Architecture"):
                gr.Markdown(
                    "## How a question is answered\n\n"
                    "1. **Subject Router Agent** - asks the LLM for strict JSON "
                    "(`subject`, `intent`, `confidence`) with a keyword fallback.\n"
                    "2. **RAG Retrieval Agent** - FAISS similarity search filtered by subject, "
                    "score thresholding, deduplication and a character-budgeted context block "
                    "(the answer mode only scales how many passages are requested).\n"
                    "3. **Answer Generation Agent** - the Neet Genie coach persona + grounding rules + the "
                    "selected mode's section contract, with token streaming and post-processing.\n"
                    "4. **Coach format repair** - the mode's section headings are standardised and "
                    "any skipped section is filled from the retrieved passages (never invented).\n\n"
                    "Every step is timed and logged; failures are converted into chat messages instead "
                    "of tracebacks."
                )
                gr.Markdown(architecture_panel(tutor))
                gr.Markdown("## Platform modules")
                gr.Markdown(
                    platform_panel(platform)
                    if platform is not None
                    else "_Module tabs are switched off (`NEET_PLATFORM_ENABLED=false`), "
                    "so no platform services were created._"
                )

        # ------------------------------------------------- cross-tab wiring
        # Every module tab is built now, so the shortcuts may reach into the
        # Ask Genie controls and the Practice scope picker.
        if modular:
            shared_targets = {
                "ask_mode": genie_refs["mode"],
                "ask_message": genie_refs["message"],
                "practice_scope": practice_refs["scope"],
            }
            if learn_refs is not None:
                # NEETRA_ENABLE_LEARN=false: the tab was never built, so neither is its wiring
                learn_ui.wire(gr, platform, learn_refs, nav=nav, targets=shared_targets)
            revision_ui.wire(gr, platform, revision_refs, nav=nav, targets=shared_targets)
            dashboard_ui.wire(gr, platform, shared_targets, nav, dashboard_refs)

    demo.queue(default_concurrency_limit=1)
    return demo


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NEETra AI - the NEET preparation assistant (Ask Genie doubt "
        "solving, Dashboard, Practice, Progress, Revision Hub; optional Learn "
        "module behind NEETRA_ENABLE_LEARN)."
    )
    parser.add_argument("--host", default=None, help="Bind address (default: NEET_GRADIO_SERVER_NAME).")
    parser.add_argument("--port", type=int, default=None, help="Port (default: NEET_GRADIO_SERVER_PORT).")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    parser.add_argument("--check", action="store_true", help="Print a diagnostics report and exit.")
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Build the knowledge base from chapter-wise NCERT PDFs in data/ and exit.",
    )
    parser.add_argument(
        "--ingest-append",
        action="store_true",
        help="With --ingest, index only chapters not yet in the index.",
    )
    parser.add_argument("--source", type=Path, default=None, help="With --ingest, override the corpus folder.")
    parser.add_argument("--log-level", default=None, help="Override NEET_LOG_LEVEL.")
    return parser.parse_args(argv)


REQUIRED_MODULES: tuple[str, ...] = (
    "pydantic",
    "pydantic_settings",
    "langchain_core",
    "langchain_community",
    "langgraph",
    "faiss",
    "pypdf",
    "gradio",
)
# Only needed when a local HuggingFace model is the active provider.
HF_MODULES: tuple[str, ...] = ("transformers", "sentence_transformers", "torch")
# Only needed when Groq is the active provider (LLM_PROVIDER=groq).
GROQ_MODULES: tuple[str, ...] = ("groq",)
# Module name -> distribution name (everything else matches after normalisation).
DISTRIBUTION_NAMES: dict[str, str] = {"faiss": "faiss-cpu"}


def _module_version(module: str) -> str:
    """Return the installed version of ``module`` or a ``MISSING (...)`` marker."""
    try:
        with warnings.catch_warnings():  # keep --check output free of deprecations
            warnings.simplefilter("ignore", DeprecationWarning)
            __import__(module)
    except Exception as exc:  # pragma: no cover - depends on the environment
        return f"MISSING ({type(exc).__name__})"
    try:
        return metadata_version(DISTRIBUTION_NAMES.get(module, module))
    except PackageNotFoundError:  # namespace/compiled modules without metadata
        return "installed"


def _dependency_report(settings: Settings | None = None) -> dict[str, str]:
    """Import-time versions of the heavy dependencies (for --check only).

    The provider decides which optional stack is required: a Groq setup needs
    the ``groq`` SDK but not ``torch``/``transformers``; ``mock`` needs neither.
    """
    modules = list(REQUIRED_MODULES)
    provider = settings.resolved_llm_provider if settings is not None else "huggingface"
    if provider == "huggingface":
        modules += list(HF_MODULES)
    elif provider == "groq":
        modules += list(GROQ_MODULES)
    return {module: _module_version(module) for module in modules}


def run_config_check(settings: Settings) -> int:
    """Validate provider, prompts, paths and dependencies; return an exit code."""
    # Provider configuration is checked first and reported without any secret.
    provider_problems: list[str] = []
    try:
        settings.validate_llm_provider()
    except ProviderConfigError as exc:
        provider_problems.append(str(exc))

    tutor = NEETTutor(settings)
    health = tutor.health()
    health["dependencies"] = _dependency_report(settings)
    print(json.dumps(health, indent=2, default=str))

    problems = list(provider_problems)
    problems += list(health["prompts"]["problems"])
    missing = [name for name, value in health["dependencies"].items() if value.startswith("MISSING")]
    if missing:
        problems.append(f"missing dependencies: {', '.join(missing)}")
    if problems:
        print("\n[FAILED] " + "; ".join(problems), file=sys.stderr)
        return 1
    print("\n[OK] configuration, prompts and dependencies look good.", file=sys.stderr)
    return 0


def run_ingest_command(
    settings: Settings,
    *,
    append: bool = False,
    source: Path | None = None,
) -> int:
    """Build the knowledge base from chapter-wise NCERT PDFs (``--ingest``).

    Delegates to the shared :func:`rag.ingest.run_ingestion`, which reuses the
    existing FAISS vector store unchanged and adds NCERT chapter metadata. The
    heavy RAG imports are lazy so importing ``app.py`` (and ``--check``) stays
    light; an ingestion failure returns a non-zero exit code instead of a
    traceback into the console.
    """
    from rag.ingest import VectorStoreError, run_ingestion, write_ingestion_report

    try:
        report = run_ingestion(settings, rebuild=not append, source_dir=source)
    except (FileNotFoundError, ValueError, VectorStoreError) as exc:
        logger.error("Ingestion failed: %s", exc)
        print(f"\n[ERROR] {exc}\n", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - unexpected
        logger.exception("Unexpected ingestion failure")
        print(f"\n[ERROR] {exc}\n", file=sys.stderr)
        return 2

    write_ingestion_report(
        report,
        markdown_path=BASE_DIR / "KB_INGESTION_REPORT.md",
        json_path=BASE_DIR / "KB_INGESTION_REPORT.json",
    )
    print(report.pretty())
    if report.failures:
        print(f"[WARN] {len(report.failures)} file(s) failed; see KB_INGESTION_REPORT.md\n", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Application entry point."""
    args = _parse_args(argv)
    settings = get_settings()
    if args.log_level:
        settings = settings.model_copy(update={"log_level": args.log_level.upper()})

    settings.ensure_directories()
    setup_logging(settings)
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)
    logger.info("Configuration: %s", json.dumps(settings.summary(), default=str))

    if args.check:
        return run_config_check(settings)

    if args.ingest:
        return run_ingest_command(settings, append=args.ingest_append, source=args.source)

    tutor = NEETTutor(settings)
    if not tutor.index_status()["ready"]:
        logger.warning("No FAISS index found - the tutor will answer without retrieval")
    if tutor.prompts.validate():
        logger.error("Prompt validation failed: %s", tutor.prompts.validate())
    logger.info(
        "Platform modules: %s (%s)",
        ", ".join(tutor.platform_status()["modules"]),
        "enabled" if settings.platform_enabled else "disabled",
    )

    demo = build_app(settings, tutor)
    demo.launch(
        server_name=args.host or settings.gradio_server_name,
        server_port=args.port or settings.gradio_server_port,
        share=bool(args.share or settings.gradio_share),
        inbrowser=False,
        show_error=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
