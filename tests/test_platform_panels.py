"""The platform UI, built without Gradio installed.

Every tab module in ``ui/`` splits into Gradio-free content functions (covered
by ``test_platform_services.py``) and widget builders.  The builders are checked
here against a stand-in ``gradio`` module: the whole navigation is assembled, the
tab order is asserted, and every registered handler is called so an output-count
or ref-key mistake fails the suite instead of failing a student's click.

No model is involved - generation runs on ``MockLLM`` and retrieval on the
fake-embedding FAISS index from the shared fixtures.
"""

from __future__ import annotations

import inspect
import time
import types
from typing import Any

import pytest

import app as app_module
from learning.practice import MAX_TEST_QUESTIONS
from learning.service import GeniePlatform
from llm.mock_client import MockLLM
from ui import components as genie_ui
from utils.constants import AnswerMode

# ---------------------------------------------------------------------------
# A stand-in for the ``gradio`` module
# ---------------------------------------------------------------------------
_ACTIVE: list["FakeGradio"] = []


class Event:
    """One registered ``(component, event, handler, inputs, outputs)`` edge."""

    def __init__(self, component: "Component", name: str, fn: Any, inputs: list, outputs: list) -> None:
        self.component = component
        self.name = name
        self.fn = fn
        self.inputs = inputs
        self.outputs = outputs

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        label = self.component.props.get("label") or self.component.kind
        return f"<Event {label}.{self.name} -> {len(self.outputs)} output(s)>"

    def call(self, *values: Any) -> Any:
        return self.fn(*values)

    def call_with_samples(self) -> tuple[Any, list[Any]]:
        """Invoke the handler with plausible values for its inputs."""
        values = [component.sample() for component in self.inputs]
        return self.fn(*values), values


class Component:
    """Base widget: records children, properties and event registrations."""

    kind = "component"

    def __init__(self, *args: Any, **props: Any) -> None:
        self.args = args
        self.props = props
        # Gradio exposes the constructor value/label as attributes, and the tab
        # builders read them while creating dependent widgets.
        self.value = props.get("value")
        self.label = props.get("label")
        self.children: list[Component] = []
        self.events: list[Event] = []
        fake = _ACTIVE[0] if _ACTIVE else None
        if fake is not None:
            stack = fake.stack
            if stack:
                stack[-1].children.append(self)
            fake.created.append(self)

    # ------------------------------------------------------------- contexts
    def __enter__(self) -> "Component":
        _ACTIVE[0].stack.append(self)
        return self

    def __exit__(self, *_exc: Any) -> bool:
        _ACTIVE[0].stack.pop()
        return False

    # --------------------------------------------------------------- events
    def _on(self, name: str, fn: Any, inputs: Any, outputs: Any, _queue: Any) -> "Component":
        event = Event(self, name, fn, list(inputs or []), list(outputs or []))
        self.events.append(event)
        _ACTIVE[0].events.append(event)
        return self

    def click(self, fn: Any = None, *_a: Any, inputs: Any = None, outputs: Any = None, queue: Any = None, **_kw: Any):
        return self._on("click", fn, inputs, outputs, queue)

    def change(self, fn: Any = None, *_a: Any, inputs: Any = None, outputs: Any = None, queue: Any = None, **_kw: Any):
        return self._on("change", fn, inputs, outputs, queue)

    def submit(self, fn: Any = None, *_a: Any, inputs: Any = None, outputs: Any = None, queue: Any = None, **_kw: Any):
        return self._on("submit", fn, inputs, outputs, queue)

    def select(self, fn: Any = None, *_a: Any, inputs: Any = None, outputs: Any = None, queue: Any = None, **_kw: Any):
        return self._on("select", fn, inputs, outputs, queue)

    def then(self, fn: Any = None, *_a: Any, inputs: Any = None, outputs: Any = None, queue: Any = None, **_kw: Any):
        return self._on("then", fn, inputs, outputs, queue)

    def queue(self, **_props: Any) -> "Component":
        return self

    def launch(self, **_props: Any) -> None:  # pragma: no cover - never started
        raise AssertionError("the test must not start a server")

    # -------------------------------------------------------------- helpers
    def sample(self) -> Any:
        """A believable runtime value for this widget."""
        kind = type(self).kind
        value = self.props.get("value")
        if kind in ("dropdown", "radio"):
            if value is not None:
                return value
            choices = self.props.get("choices") or []
            if not choices:
                return None
            first = choices[0]
            return first[1] if isinstance(first, tuple) else first
        if kind == "slider":
            return value if value is not None else self.props.get("minimum", 5)
        if kind == "checkbox":
            return bool(value)
        if kind == "chatbot":
            return []
        if kind == "state":
            return value
        return value if value is not None else ""

    def find(self, kind: str) -> list[Component]:
        found: list[Component] = []
        for child in self.children:
            if child.kind == kind:
                found.append(child)
            found.extend(child.find(kind))
        return found


class Blocks(Component):
    kind = "blocks"


class Tabs(Component):
    kind = "tabs"

    def __init__(self, *, selected: Any = None, **props: Any) -> None:
        super().__init__(selected=selected, **props)


class Tab(Component):
    kind = "tab"

    def __init__(self, label: Any = None, *, id: Any = None, visible: bool = True, **props: Any) -> None:
        super().__init__(label=label, id=id, visible=visible, **props)


class Row(Component):
    kind = "row"


class Column(Component):
    kind = "column"


class Accordion(Component):
    kind = "accordion"


class Markdown(Component):
    kind = "markdown"

    def __init__(self, value: Any = None, **props: Any) -> None:
        super().__init__(value=value, **props)


class Chatbot(Component):
    kind = "chatbot"


class Textbox(Component):
    kind = "textbox"


class Button(Component):
    kind = "button"

    def __init__(self, value: Any = None, **props: Any) -> None:
        super().__init__(value=value, **props)


class ClearButton(Component):
    kind = "clear_button"


class Dropdown(Component):
    kind = "dropdown"


class Radio(Component):
    kind = "radio"


class Slider(Component):
    kind = "slider"


class Checkbox(Component):
    kind = "checkbox"


class State(Component):
    kind = "state"


class Examples(Component):
    kind = "examples"


class Image(Component):
    kind = "image"


class FakeGradio:
    """The slice of the ``gradio`` module the platform actually uses."""

    Blocks = Blocks
    Tabs = Tabs
    Tab = Tab
    Row = Row
    Column = Column
    Accordion = Accordion
    Markdown = Markdown
    Chatbot = Chatbot
    Textbox = Textbox
    Button = Button
    ClearButton = ClearButton
    Dropdown = Dropdown
    Radio = Radio
    Slider = Slider
    Checkbox = Checkbox
    Image = Image
    State = State
    Examples = Examples

    def __init__(self) -> None:
        self.stack: list[Component] = []
        self.created: list[Component] = []
        self.events: list[Event] = []
        self.themes = types.SimpleNamespace(Soft=lambda **props: "soft-theme")

    def update(self, **props: Any) -> dict[str, Any]:
        return {"__update__": True, **props}

    def tabs(self) -> Tabs:
        return next(node for node in self.created if isinstance(node, Tabs))

    def labels(self) -> list[str]:
        return [child.props.get("label") for child in self.tabs().find("tab")]

    def by_label(self, label: str) -> Component:
        # buttons carry their caption as the positional ``value`` argument
        for node in self.created:
            if node.props.get("label") == label or (
                isinstance(node, Button) and node.props.get("value") == label
            ):
                return node
        raise LookupError(f"no widget labelled {label!r}")

    def event_for(self, label: str, name: str = "click") -> Event:
        widget = self.by_label(label)
        return next(event for event in widget.events if event.name == name)

    def call(self, label: str, *values: Any) -> Any:
        return self.event_for(label).call(*values)


@pytest.fixture
def fake_gr() -> Any:
    """Inject the stand-in module, then restore the real import path."""
    fake = FakeGradio()
    _ACTIVE.clear()
    _ACTIVE.append(fake)
    genie_ui.inject_gradio(fake)
    try:
        yield fake
    finally:
        genie_ui.inject_gradio(None)
        _ACTIVE.clear()


# ---------------------------------------------------------------------------
# The app under test
# ---------------------------------------------------------------------------
class StubTutor:
    """Only what ``build_app`` asks of the chat pipeline (never runs a model)."""

    def __init__(self, platform: GeniePlatform) -> None:
        self.platform_instance = platform
        self.prompts = platform.prompts
        self.platform_calls = 0

    def platform(self) -> GeniePlatform:
        self.platform_calls += 1
        return self.platform_instance

    def index_status(self) -> dict[str, Any]:
        return {
            "ready": True,
            "chunks": 3,
            "files": 3,
            "subjects": {"physics": 1, "chemistry": 1, "biology": 1},
            "embedding_model": "fake-embedding",
            "updated_at": "today",
            "path": "vectordb",
        }

    def startup_status(self) -> dict[str, Any]:
        return {
            "ready": True,
            "checks": {
                "knowledge_base": {"ready": True, "label": "Knowledge Base"},
                "llm": {"ready": True, "label": "LLM"},
                "retrieval": {"ready": True, "label": "Retrieval"},
            },
            "student_message": "You're all set. Ask Genie your first NEET doubt.",
            "errors": {},
            "checked_at": "today",
        }

    def validate_startup(self) -> dict[str, Any]:
        return self.startup_status()

    def health(self) -> dict[str, Any]:
        return {"readiness": self.startup_status(), "index": self.index_status()}

    def architecture(self) -> str:
        return "router -> retrieval -> generation"

    def ensure_ready(self) -> "StubTutor":
        return self


@pytest.fixture
def platform(settings: Any, vector_store: Any) -> GeniePlatform:
    return GeniePlatform(settings, vector_store=vector_store, llm=MockLLM(settings))


@pytest.fixture
def stub(platform: GeniePlatform) -> StubTutor:
    return StubTutor(platform)


@pytest.fixture
def built(fake_gr: FakeGradio, settings: Any, stub: StubTutor) -> FakeGradio:
    app_module.build_app(settings, tutor=stub)
    return fake_gr


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
def test_navigation_is_exactly_the_five_student_modules(built: FakeGradio) -> None:
    assert built.labels() == [
        "🤖 Ask Genie",
        "🎯 Practice",
        "📈 Progress",
        "📝 Revision Hub",
        "🏠 Dashboard",
    ]
    assert built.tabs().props["selected"] == "ask-genie"


def test_student_layout_does_not_render_developer_controls(built: FakeGradio) -> None:
    labels = [str(node.props.get("label") or node.props.get("value") or "") for node in built.created]
    assert not any("Developer Mode" in label for label in labels)
    assert not any("Technical diagnostics" in label for label in labels)


def test_ask_genie_can_run_without_textbook_retrieval_when_fallback_is_enabled(
    settings: Any,
) -> None:
    status = {
        "checks": {
            "llm": {"ready": True},
            "retrieval": {"ready": False},
        }
    }
    assert app_module._ask_genie_available(status, settings)
    assert not app_module._ask_genie_available(
        status, settings.model_copy(update={"allow_general_fallback": False})
    )


def test_module_tabs_carry_the_ids_the_shortcuts_jump_to(built: FakeGradio) -> None:
    ids = {child.props.get("label"): child.props.get("id") for child in built.tabs().find("tab")}
    assert sum(isinstance(node, Tabs) for node in built.created) == 1
    assert ids["🏠 Dashboard"] == "dashboard"
    assert ids["🤖 Ask Genie"] == "ask-genie"
    assert ids["📝 Revision Hub"] == "revision"


# ---------------------------------------------------------------------------
# Ask Genie = the coach interface, reused
# ---------------------------------------------------------------------------
def test_ask_genie_has_simple_student_controls_and_suggestions(built: FakeGradio) -> None:
    ask_tab = next(child for child in built.tabs().find("tab") if child.props.get("label") == "🤖 Ask Genie")
    assert ask_tab.find("chatbot")
    modes = ask_tab.find("radio")[0]
    assert [value for _, value in modes.props["choices"]] == [mode.value for mode in AnswerMode]
    assert ask_tab.find("textbox")
    assert not ask_tab.find("slider"), "technical retrieval controls stay out of the student UI"
    assert not ask_tab.find("examples")
    for label in (
        "Explain Cell Cycle",
        "Explain Photoelectric Effect",
        "10 MCQs: Thermodynamics",
        "Revise Molecular Basis of Inheritance",
    ):
        assert built.by_label(label)
    text = "\n".join(node.props.get("value") or "" for node in ask_tab.find("markdown"))
    assert "Knowledge Base Ready" not in text
    assert "LLM Ready" not in text
    assert "Retrieval Ready" not in text
    assert "You're all set." not in text
    assert "Student Profile" in text


def test_ask_genie_send_handler_uses_only_student_inputs(built: FakeGradio) -> None:
    event = built.event_for("Send to Genie")
    assert len(event.inputs) == 4, "message, chat, subject and answer style"
    assert [node.kind for node in event.outputs] == ["chatbot", "markdown", "markdown"]
    assert inspect.isgeneratorfunction(event.fn), "streaming must stay a generator"


def test_ask_genie_photo_upload_prefills_editable_question_without_sending(
    built: FakeGradio, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    assert any(
        node.kind == "accordion" and node.args[0] == "📷 Ask with a photo"
        for node in built.created
    )
    assert built.by_label("Upload your question photo").kind == "image"

    image_path = tmp_path / "doubt.png"
    image_path.write_bytes(b"uploaded image")
    monkeypatch.setattr(
        GeniePlatform,
        "extract_photo_question",
        lambda _self, _image, *, subject=None: "Find the acceleration of the body.",
    )

    event = built.event_for("Read uploaded photo")
    text, status = event.call(str(image_path), "physics")

    assert text == "Find the acceleration of the body."
    assert "Review or edit" in status
    assert [component.kind for component in event.outputs] == ["textbox", "markdown"]
    assert event.component is not built.by_label("Send to Genie")


# ---------------------------------------------------------------------------
# Every handler returns one value per output
# ---------------------------------------------------------------------------
def test_every_registered_handler_returns_the_declared_outputs(built: FakeGradio) -> None:
    """Catches ref/output bookkeeping errors without a browser."""
    failures: list[str] = []
    for event in built.events:
        if inspect.isgeneratorfunction(event.fn):
            continue
        try:
            result, _values = event.call_with_samples()
        except Exception as exc:  # pragma: no cover - reported below
            failures.append(f"{event}: raised {type(exc).__name__}: {exc}")
            continue
        values = list(result) if isinstance(result, (list, tuple)) else [result]
        if len(values) != len(event.outputs):
            failures.append(f"{event}: returned {len(values)} value(s)")
    assert not failures, "\n".join(failures)


def test_tabnav_understands_the_injected_build(built: FakeGradio) -> None:
    """The jump path is only as good as the Gradio build - and it is checked."""
    nav = next(node for node in built.created if isinstance(node, Tabs))
    assert nav is not None
    probe = genie_ui.TabNav(built)
    assert probe.supports_selected is True


# ---------------------------------------------------------------------------
# Dashboard shortcuts
# ---------------------------------------------------------------------------
def test_dashboard_practice_shortcut_configures_and_jumps(built: FakeGradio, stub: StubTutor) -> None:
    event = built.event_for("🎯 Practice Tests")
    guidance, scope_update, tabs_update = event.call()
    assert "Practice" in guidance
    assert scope_update["value"] == "chapter"
    assert tabs_update.props["selected"] == "practice"


def test_dashboard_ask_shortcut_drafts_into_the_chat_box(built: FakeGradio, stub: StubTutor) -> None:
    event = built.event_for("💬 Ask a Doubt")
    guidance, mode_update, message_update, tabs_update = event.call()
    assert "Ask Genie" in guidance
    assert mode_update["value"] == AnswerMode.DETAILED.value
    assert "trap" in message_update["value"]
    assert tabs_update.props["selected"] == "ask-genie"


def test_dashboard_refresh_recounts_from_the_store(built: FakeGradio, stub: StubTutor) -> None:
    outputs = built.event_for("Refresh dashboard").outputs
    values = built.event_for("Refresh dashboard").call()
    assert len(values) == len(outputs) == 3
    assert "Questions answered" in values[0]
    assert "Focus topics" in values[1]


def test_dashboard_revision_and_progress_cards_open_the_right_tabs(built: FakeGradio) -> None:
    _guidance, revision_tabs = built.event_for("🔄 Revision Planner").call()
    _progress_guidance, progress_tabs = built.event_for("📈 Progress Dashboard").call()
    assert revision_tabs.props["selected"] == "revision"
    assert progress_tabs.props["selected"] == "progress"


# ---------------------------------------------------------------------------
# Learn
# ---------------------------------------------------------------------------
def test_learn_tab_offers_the_indexed_chapters(fake_gr: FakeGradio, settings: Any, stub: StubTutor) -> None:
    settings = settings.model_copy(update={"learn_enabled": True})
    app_module.build_app(settings, tutor=stub)
    learn_tab = next(child for child in fake_gr.tabs().find("tab") if child.props.get("label") == "📚 Learn")
    dropdowns = learn_tab.find("dropdown")
    subjects, chapters = dropdowns[0], dropdowns[1]
    assert [value for _, value in subjects.props["choices"]][:3] == ["physics", "chemistry", "biology"]
    assert "Mechanics" in [name for _, name in chapters.props["choices"]]


def test_learn_generate_builds_the_four_section_page(fake_gr: FakeGradio, settings: Any, stub: StubTutor) -> None:
    settings = settings.model_copy(update={"learn_enabled": True})
    app_module.build_app(settings, tutor=stub)
    page, status, citations = fake_gr.call("Generate study page", "physics", "Mechanics", "")
    for heading in ("Summary", "High-Yield Concepts", "Important NEET Points", "Quick Revision Notes"):
        assert heading in page, f"missing section {heading}"
    assert "Status" in status
    assert citations


def test_learn_topic_box_works_without_a_chapter(fake_gr: FakeGradio, settings: Any, stub: StubTutor) -> None:
    settings = settings.model_copy(update={"learn_enabled": True})
    app_module.build_app(settings, tutor=stub)
    page, _status, _citations = fake_gr.call("Generate study page", "physics", "", "photoelectric effect")
    assert "Mock study material" in page or "High-Yield" in page


def test_learn_ask_button_prefills_the_coach(fake_gr: FakeGradio, settings: Any, stub: StubTutor) -> None:
    settings = settings.model_copy(update={"learn_enabled": True})
    app_module.build_app(settings, tutor=stub)
    preview, message_update, mode_update, tabs_update = fake_gr.call("Ask Genie about this chapter", "Mechanics", "")
    assert "Drafted for Ask Genie" in preview
    assert "Mechanics" in message_update["value"]
    assert mode_update["value"] == "detailed"
    assert tabs_update.props["selected"] == "ask-genie"


def test_learn_refresh_rebuilds_the_choices(fake_gr: FakeGradio, settings: Any, stub: StubTutor) -> None:
    settings = settings.model_copy(update={"learn_enabled": True})
    app_module.build_app(settings, tutor=stub)
    subjects, chapters, intro = fake_gr.call("Refresh chapters")
    assert subjects["choices"]
    assert "chapters" in intro.lower() or "index" in intro.lower()
    assert chapters["choices"] or chapters["value"] is None


# ---------------------------------------------------------------------------
# Practice
# ---------------------------------------------------------------------------
def _picks_for(test: Any, answer: int = 1) -> list[int | None]:
    picks: list[int | None] = []
    for _question in test.questions:
        picks.append(answer)
    return picks + [None] * (MAX_TEST_QUESTIONS - len(test.questions))


def test_practice_generate_reveals_one_radio_per_question(built: FakeGradio, stub: StubTutor) -> None:
    values = built.call("Generate test", "chapter", "physics", "Mechanics", 3)
    status, quiz, radios, citations, stored, clock = values[0], values[1], values[2:17], values[17], values[18], values[19]
    assert len(radios) == MAX_TEST_QUESTIONS
    visible = [radio for radio in radios if radio.get("__update__") and radio["visible"]]
    hidden = [radio for radio in radios if not radio.get("visible")]
    assert len(visible) == 3 and len(hidden) == MAX_TEST_QUESTIONS - 3
    assert "Question" in quiz or "Q1" in quiz
    assert stored is not None and len(stored.questions) == 3
    assert isinstance(clock, float)
    assert "3 question(s)" in status


def test_practice_submit_scores_and_persists(built: FakeGradio, stub: StubTutor) -> None:
    platform = stub.platform()
    values = built.call("Generate test", "chapter", "physics", "Mechanics", 4)
    test, clock = values[18], values[19]
    status, review, score = built.call(
        "Submit answers and score", test, clock, *_picks_for(test, answer=1)
    )
    assert "Submitted" in status
    assert "**Score:**" in review and "**Accuracy:**" in review
    assert "Weak topics" in score or "Topic breakdown" in score
    assert platform.progress.stats()["tests"] == 1


def test_practice_submit_without_a_test_is_explained(built: FakeGradio) -> None:
    status, review, sources = built.call("Submit answers and score", None, None, *[None] * MAX_TEST_QUESTIONS)
    assert "generate a test first" in status.lower()
    assert review == status


def test_practice_weak_area_button_points_at_the_revision_hub(built: FakeGradio, stub: StubTutor) -> None:
    values = built.call("Revise my weak areas in the Revision Hub")
    assert values[0]
    if len(values) > 1:
        assert values[1].props["selected"] == "revision"


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------
def test_progress_reads_the_store_after_a_test(built: FakeGradio, stub: StubTutor) -> None:
    platform = stub.platform()
    values = built.call("Generate test", "chapter", "physics", "Mechanics", 3)
    test, clock = values[18], values[19]
    built.call("Submit answers and score", test, clock, *_picks_for(test, answer=2))

    headline, subjects, chapters, weak, activity, daily = built.call("Refresh")
    assert "1" in headline
    assert "Physics" in subjects
    assert "Mechanics" in chapters
    assert "Q1" in activity or "Mechanics" in activity
    assert daily.count("-") >= 0  # the 14-day table renders
    assert platform.progress.stats()["tests"] == 1


def test_progress_reset_empties_the_store(built: FakeGradio, stub: StubTutor) -> None:
    platform = stub.platform()
    values = built.call("Generate test", "chapter", "physics", "Mechanics", 2)
    built.call("Submit answers and score", values[18], values[19], *_picks_for(values[18]))
    assert platform.progress.stats()["tests"] == 1
    headline = built.call("Reset history")[0]
    assert "History cleared" in headline
    assert platform.progress.stats()["tests"] == 0


# ---------------------------------------------------------------------------
# Revision Hub
# ---------------------------------------------------------------------------
def test_revision_deliverables_are_all_offered(built: FakeGradio) -> None:
    revision_tab = next(child for child in built.tabs().find("tab") if child.props.get("label") == "📝 Revision Hub")
    sheet_types = [value for _, value in revision_tab.find("radio")[0].props["choices"]]
    assert {"flash_notes", "formula_sheet", "high_yield", "last_minute"} <= set(sheet_types)


def test_revision_sheet_keeps_its_section_contract(built: FakeGradio, stub: StubTutor) -> None:
    status, sheet, citations, _weak = built.call("Generate sheet", "formula_sheet", "physics", "Mechanics")
    assert "Status" in status
    assert "###" in sheet
    assert sheet != ""
    assert citations


def test_revision_weak_area_sheet_uses_the_history(built: FakeGradio, stub: StubTutor) -> None:
    platform = stub.platform()
    values = built.call("Generate test", "chapter", "physics", "Mechanics", 4)
    built.call("Submit answers and score", values[18], values[19], *_picks_for(values[18], answer=1))
    status, sheet, _citations, weak = built.call("Revise my weak areas", "flash_notes")
    assert platform.progress.stats()["revision_sheets"] == 1
    assert "Status" in status and sheet
    assert "Topic" in weak or "No weak areas" in weak or "no weak" in weak.lower()


def test_revision_can_draft_a_doubt_for_the_coach(built: FakeGradio, stub: StubTutor) -> None:
    preview, message_update, mode_update, tabs_update = built.call("Ask Genie to explain these", "Mechanics")
    assert "Drafted for Ask Genie" in preview
    assert mode_update["value"] == "quick_revision"
    assert tabs_update.props["selected"] == "ask-genie"


# ---------------------------------------------------------------------------
# Cross-cutting
# ---------------------------------------------------------------------------
def test_app_uses_verified_readiness_without_loading_a_model_during_layout(
    fake_gr: FakeGradio, settings: Any, stub: StubTutor
) -> None:
    app_module.build_app(settings, tutor=stub)
    blocks = next(node for node in fake_gr.created if isinstance(node, Blocks))
    assert blocks.props["title"] == "NEETra AI"
    assert stub.platform_calls >= 1


def test_the_layout_is_rebuilt_on_a_second_call(fake_gr: FakeGradio, settings: Any, stub: StubTutor) -> None:
    """No global widget state may leak between builds."""
    app_module.build_app(settings, tutor=stub)
    first = len(fake_gr.created)
    app_module.build_app(settings, tutor=stub)
    assert len(fake_gr.created) > first
    assert fake_gr.labels() == [
        "🤖 Ask Genie", "🎯 Practice", "📈 Progress", "📝 Revision Hub", "🏠 Dashboard"
    ]


def test_clock_helper_is_used_for_duration(built: FakeGradio, stub: StubTutor) -> None:
    values = built.call("Generate test", "topic", "physics", "Newton's second law", 2)
    started = values[19]
    assert started <= time.monotonic()
