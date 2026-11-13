"""NEETra AI - a student-first NEET learning application.

Run::

    python app.py                      # verify study services, then start the UI
    python app.py --check              # run startup diagnostics and exit
    python app.py --ingest             # build the NCERT study library

The student interface focuses on Ask Genie, Practice, Progress, Revision Hub
and Dashboard. Learn remains opt-in for compatibility. Startup verifies the
answer provider and textbook search before presenting the Ask Genie status.
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
from agents.tutor import NEETTutor  # noqa: E402
from agents.registry import describe_registry  # noqa: E402
from config import BASE_DIR, ProviderConfigError, Settings, get_settings  # noqa: E402
from learning.capabilities import describe_capabilities  # noqa: E402
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

STREAM_THROTTLE_S: float = 0.06

CSS = """
:root {--genie-primary: #5b4bdb; --genie-ink: #17233c; --genie-muted: #64748b;}
.gradio-container {max-width: 1320px !important; padding: 1.25rem 2rem 2.5rem !important;
  color: var(--genie-ink); font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;}
#neet-header {margin: 0; color: #35268f; letter-spacing: -0.035em;}
#neet-header h1 {font-size: clamp(2rem, 4vw, 2.8rem); margin-bottom: 0.2rem;}
#neet-sub {color: var(--genie-muted); font-size: 1.12rem; margin: 0 0 1.25rem;}
#neet-status {min-height: 2.2rem; font-size: 0.94rem; color: #475569;}
#neet-mode-help {font-size: 0.88rem; color: #64748b;}
.source-card {line-height: 1.5;}
@media (max-width: 700px) {
  .gradio-container {padding: 0.85rem !important;}
  #neet-sub {font-size: 1rem;}
}
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
        "### Input channels\n\n"
        "Ask Genie accepts typed questions and uploaded doubt photos. Photo text is "
        "read locally and shown for review before the student sends it. Voice input "
        "is still planned.\n\n"
        f"{describe_capabilities(platform.capabilities)}\n\n"
        "_Uploaded questions use the same answer pipeline as typed questions._"
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _status_line(message: str) -> str:
    return f"**Status:** {message}"


def _mode_help(mode: AnswerMode) -> str:
    """A short student-facing explanation of the selected answer style."""
    return f"{mode.emoji} **{mode.label}** — {mode.description}"


def _ask_genie_available(status: dict[str, Any], settings: Settings) -> bool:
    """Allow student answers only when a real model provider is ready."""
    checks = status.get("checks") or {}
    llm_ready = bool((checks.get("llm") or {}).get("ready"))
    retrieval_ready = bool((checks.get("retrieval") or {}).get("ready"))
    return (
        settings.resolved_llm_provider != "mock"
        and llm_ready
        and (retrieval_ready or settings.allow_general_fallback)
    )


def _ask_genie_status_message(status: dict[str, Any], settings: Settings) -> str:
    if settings.resolved_llm_provider == "mock":
        return "Genie's AI service isn't connected yet, so answers are unavailable."
    if _ask_genie_available(status, settings):
        return "Ready for your doubt."
    return str(status.get("student_message") or "Answers are paused until Genie reconnects.")


def _student_progress(payload: dict[str, Any]) -> str:
    node = str(payload.get("node") or "")
    return {
        "router": "Understanding your question…",
        "retrieval": "Finding the right textbook explanation…",
        "answer": "Writing a clear NEET-focused answer…",
        "llm": "Connecting to Genie…",
        "graph": "Preparing your study session…",
    }.get(node, "Genie is working on your answer…")


def _history_for_model(chat: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chat turns excluding the current message, trimmed to the configured depth."""
    return [
        {"role": str(turn.get("role", "user")), "content": str(turn.get("content", ""))}
        for turn in chat
        if turn.get("content")
    ]


def _completion_status(result: TutorResult, settings: Settings) -> str:
    """A concise student receipt, without routing or model diagnostics."""
    if result.error:
        return "**Status:** Genie could not finish this answer. Please try again."
    parts = [f"**Subject:** {str(result.subject).title()}"]
    parts.append(f"Answered in {format_seconds(result.total_seconds)}")
    return " · ".join(parts)


# ---------------------------------------------------------------------------
# Chat handler (generator -> streams status and tokens)
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
        top_k: float | None = None,
        stream_enabled: bool | None = None,
    ) -> Iterator[tuple[list[dict[str, Any]], str]]:
        history = list(chat or [])
        text = (message or "").strip()

        if not text:
            yield history, _status_line("Please type a question.")
            return
        if len(text) > settings.max_question_chars:
            text = text[: settings.max_question_chars]
            logger.info("Question truncated to %d characters", settings.max_question_chars)

        answer_mode = AnswerMode.coerce(mode, default=settings.answer_mode)
        logger.info(
            "Ask Genie request started (mode=%s, subject=%s, chars=%d)",
            answer_mode.value,
            subject or "auto",
            len(text),
        )
        prior_turns = _history_for_model(history)
        history.append({"role": "user", "content": text})

        def snapshot(answer: str) -> list[dict[str, Any]]:
            return [
                *history,
                {"role": "assistant", "content": genie_ui.hide_source_markers(answer)},
            ]

        streaming = bool(settings.stream_tokens if stream_enabled is None else stream_enabled)
        streaming = streaming and settings.stream_tokens
        retrieval_count = int(top_k or settings.top_k)
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
                    top_k=retrieval_count,
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
        status = _status_line("Genie is thinking about your doubt…")
        last_emit = 0.0
        yield snapshot(partial), status

        while True:
            kind, payload = events.get()
            if kind == "done":
                break
            if kind == "token":
                partial += str(payload)
                now = time.monotonic()
                if now - last_emit >= STREAM_THROTTLE_S:
                    last_emit = now
                    yield snapshot(partial), status
            elif kind == "event" and isinstance(payload, dict):
                status = _status_line(_student_progress(payload))
                yield snapshot(partial), status

        thread.join(timeout=5.0)
        result = box.get("result")
        if result is None:
            error = box.get("error")
            logger.error("Turn produced no result: %s", error)
            message = (
                "**Genie could not finish that answer.** Please try again in a moment. "
                "You can also try a shorter question or use Try reconnecting."
            )
            yield snapshot(message), _status_line("Failed")
            return

        logger.info(
            "Ask Genie request completed (ok=%s, citations=%d)",
            getattr(result, "ok", True),
            len(result.citations),
        )
        yield snapshot(result.answer), _completion_status(result, settings)

    return respond


# ---------------------------------------------------------------------------
# Ask Genie - the coach interface, reused as one module of the platform
# ---------------------------------------------------------------------------
def build_ask_genie_tab(
    gr: Any,
    settings: Settings,
    tutor: NEETTutor,
    responder: Any,
    platform: Any,
) -> dict[str, Any]:
    """A calm chat workspace with only student-relevant controls."""
    readiness = tutor.startup_status()
    ready = _ask_genie_available(readiness, settings)

    gr.Markdown(
        "## 🤖 Ask Genie\n"
        "Ask a concept, numerical, MCQ or revision question. Genie explains it in a "
        "NEET-focused way."
    )
    with gr.Row():
        with gr.Column(scale=3, elem_classes=["genie-chat-card"]):
            chatbot = gr.Chatbot(
                label="Chat with Genie",
                type="messages",
                height=settings.chatbot_height,
                show_copy_button=True,
                latex_delimiters=[
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": "$", "right": "$", "display": False},
                ],
            )
            message_box = gr.Textbox(
                placeholder="Type your NEET doubt here…",
                label="Your question",
                autofocus=True,
                lines=2,
                max_lines=5,
                interactive=ready,
            )
            with gr.Accordion("📷 Ask with a photo", open=False):
                photo_box = gr.Image(
                    type="filepath",
                    sources=["upload"],
                    label="Upload your question photo",
                    show_download_button=False,
                    height=240,
                )
                gr.Markdown(
                    "Choose a clear, upright photo of your question. Review the extracted text for "
                    "handwriting, equations, or diagram details before sending."
                )
                read_photo_btn = gr.Button("Read uploaded photo", size="sm")
            with gr.Row():
                send_btn = gr.Button(
                    "Send to Genie", variant="primary", scale=3, interactive=ready
                )
                clear_btn = gr.ClearButton([chatbot], value="New chat", scale=1)
            status_box = gr.Markdown(
                    _status_line(_ask_genie_status_message(readiness, settings)),
                elem_id="neet-status",
            )
            gr.Markdown("**Try asking**", elem_classes=["genie-suggestion-label"])
            with gr.Row(elem_classes=["genie-suggestions"]):
                for label, prompt in (
                    ("Explain Cell Cycle", "Explain the Cell Cycle and its key checkpoints."),
                    ("Explain Photoelectric Effect", "Explain the photoelectric effect for NEET."),
                    (
                        "10 MCQs: Thermodynamics",
                        "Generate 10 MCQs from Thermodynamics.",
                    ),
                    (
                        "Revise Molecular Basis of Inheritance",
                        "Give me quick revision notes for Molecular Basis of Inheritance.",
                    ),
                ):
                    suggestion = gr.Button(
                        label,
                        size="sm",
                        elem_classes=["genie-prompt-chip"],
                    )
                    suggestion.click(
                        lambda value=prompt: gr.update(value=value),
                        inputs=None,
                        outputs=[message_box],
                        queue=False,
                    )
        with gr.Column(scale=1, elem_classes=["genie-side-card"]):
            gr.Markdown(dashboard_ui.student_profile_md(platform.progress))
            subject_box = gr.Dropdown(
                choices=[
                    ("✨ Detect automatically", "auto"),
                    ("🕶️ Physics", "physics"),
                    ("🧪 Chemistry", "chemistry"),
                    ("🌱 Biology", "biology"),
                    ("📚 General", "general"),
                ],
                value="auto",
                label="Subject",
            )
            mode_box = gr.Radio(
                choices=AnswerMode.ui_choices(),
                value=settings.answer_mode.value,
                label="How should Genie answer?",
            )
            mode_md = gr.Markdown(
                _mode_help(settings.answer_mode), elem_id="neet-mode-help"
            )
            reconnect_btn = gr.Button("Try reconnecting", size="sm")

    mode_box.change(
        lambda choice: _mode_help(AnswerMode.coerce(choice, default=settings.answer_mode)),
        inputs=[mode_box],
        outputs=[mode_md],
        queue=False,
    )

    send_btn.click(
        responder,
        inputs=[message_box, chatbot, subject_box, mode_box],
        outputs=[chatbot, status_box],
    ).then(lambda: "", outputs=[message_box], queue=False)

    def read_photo(image_path: str | None, subject: str) -> tuple[str, str]:
        if not image_path:
            return "", _status_line("Upload a photo first.")

        path = Path(image_path)
        if not path.is_file():
            return "", _status_line("That photo is no longer available. Upload it again.")
        if path.stat().st_size > 20 * 1024 * 1024:
            return "", _status_line("This photo is too large. Choose an image under 20 MB.")

        try:
            question = platform.extract_photo_question(
                path.read_bytes(),
                subject=None if subject in (None, "", "auto", "general") else subject,
            )
        except Exception:
            logger.exception("Could not read text from uploaded doubt photo")
            return "", _status_line(
                "We couldn't read this photo. Try a clearer image or type your question."
            )
        if not question.strip():
            return "", _status_line(
                "No readable question text was found. Try a clearer photo or type your question."
            )
        return question, _status_line(
            "Text is ready. Review or edit your question above, then send it to Genie."
        )

    read_photo_btn.click(
        read_photo,
        inputs=[photo_box, subject_box],
        outputs=[message_box, status_box],
    )

    message_box.submit(
        responder,
        inputs=[message_box, chatbot, subject_box, mode_box],
        outputs=[chatbot, status_box],
    ).then(lambda: "", outputs=[message_box], queue=False)

    clear_btn.click(
        lambda: _status_line("New chat started. What would you like to understand?"),
        outputs=[status_box],
        queue=False,
    )

    def reconnect() -> list[Any]:
        status = tutor.validate_startup()
        available = _ask_genie_available(status, settings)
        return [
            _status_line(_ask_genie_status_message(status, settings)),
            gr.update(interactive=available),
            gr.update(interactive=available),
        ]

    reconnect_btn.click(
        reconnect,
        inputs=None,
        outputs=[status_box, message_box, send_btn],
    )

    return {
        "chatbot": chatbot,
        "message": message_box,
        "send": send_btn,
        "clear": clear_btn,
        "status": status_box,
        "subject": subject_box,
        "mode": mode_box,
        "mode_help": mode_md,
        "reconnect": reconnect_btn,
    }


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def build_app(settings: Settings, tutor: NEETTutor | None = None) -> "gr.Blocks":
    """Build the student-facing product without exposing developer diagnostics."""
    gr = genie_ui.lazy_gradio()
    tutor = tutor or NEETTutor(settings)
    platform = tutor.platform()

    def count_doubt(result: TutorResult) -> None:
        if getattr(result, "ok", True):
            platform.progress.bump("doubts")

    responder = build_responder(tutor, settings, on_result=count_doubt)
    nav = genie_ui.TabNav(gr)

    with gr.Blocks(
        title="NEETra AI",
        theme=gr.themes.Soft(),
        css=CSS + genie_ui.PLATFORM_CSS,
        analytics_enabled=False,
    ) as demo:
        gr.Markdown("# NEETra AI", elem_id="neet-header")

        with gr.Tabs(selected="ask-genie") as tabs:
            nav.attach(tabs)
            with nav.tab("🤖 Ask Genie", "ask-genie"):
                genie_refs = build_ask_genie_tab(gr, settings, tutor, responder, platform)
            with nav.tab("🎯 Practice", "practice"):
                practice_refs = practice_ui.build(gr, platform, settings, nav)
            with nav.tab("📈 Progress", "progress"):
                progress_ui.build(gr, platform, settings, nav)
            with nav.tab("📝 Revision Hub", "revision"):
                revision_refs = revision_ui.build(gr, platform, settings, nav, autowire=False)
            with nav.tab("🏠 Dashboard", "dashboard"):
                dashboard_refs = dashboard_ui.build(gr, platform, settings, nav)
            if settings.learn_enabled:
                with nav.tab("📚 Learn", "learn"):
                    learn_refs = learn_ui.build(gr, platform, settings, nav, autowire=False)

        if settings.learn_enabled:
            learn_ui.wire(
                gr,
                platform,
                learn_refs,
                nav=nav,
                targets={"ask_message": genie_refs["message"], "ask_mode": genie_refs["mode"]},
            )

        shared_targets = {
            "ask_mode": genie_refs["mode"],
            "ask_message": genie_refs["message"],
            "practice_scope": practice_refs["scope"],
        }
        revision_ui.wire(gr, platform, revision_refs, nav=nav, targets=shared_targets)
        dashboard_ui.wire(gr, platform, shared_targets, nav, dashboard_refs)

    demo.queue(default_concurrency_limit=1)
    return demo


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NEETra AI - Ask Genie, Practice, Progress, Revision Hub and Dashboard."
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
    readiness = tutor.validate_startup()
    health = tutor.health()
    health["dependencies"] = _dependency_report(settings)
    print(json.dumps(health, indent=2, default=str))

    problems = list(provider_problems)
    problems += list(health["prompts"]["problems"])
    missing = [name for name, value in health["dependencies"].items() if value.startswith("MISSING")]
    if missing:
        problems.append(f"missing dependencies: {', '.join(missing)}")
    if not readiness["ready"]:
        problems += [f"{name}: {detail}" for name, detail in readiness["errors"].items()]
    if problems:
        print("\n[FAILED] " + "; ".join(dict.fromkeys(problems)), file=sys.stderr)
        return 1
    print("\n[OK] knowledge base, answer generation and retrieval are ready.", file=sys.stderr)
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
    readiness = tutor.validate_startup()
    if readiness["ready"] and settings.resolved_llm_provider != "mock":
        logger.info("Startup verified: knowledge base, LLM and retrieval are ready")
    elif settings.resolved_llm_provider == "mock":
        logger.warning(
            "The offline mock provider is active; Ask Genie is disabled until a real model is configured"
        )
    else:
        # The learning shell can still show progress, but Ask Genie remains gated.
        logger.error("Startup readiness failed: %s", readiness["errors"])

    demo = build_app(settings, tutor)
    demo.launch(
        server_name=args.host or settings.gradio_server_name,
        server_port=args.port or settings.gradio_server_port,
        share=bool(args.share or settings.gradio_share),
        inbrowser=False,
        show_error=False,
        show_api=False,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
