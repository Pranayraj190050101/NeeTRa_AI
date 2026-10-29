"""Offline smoke test for the dependency-free helpers.

Runs with the standard library only (no LangChain / pydantic needed):

    python scripts/offline_check.py

It verifies text normalisation, JSON extraction, keyword routing, the prompt
placeholder contract and the learning-platform layer that is likewise
dependency-free (chapter catalog, MCQ parser and marking, the JSON progress
store, the capability hooks and the tab layout of ``ui/``).  Intended as a fast
sanity check in CI before the heavy dependencies are installed.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Section headings and motivation lines carry emoji; a Windows console on a
# legacy code page must not crash while printing a failure detail.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from utils.answer_format import (  # noqa: E402
    EXPLANATION_SECTIONS,
    enforce_format,
    format_contract,
    missing_sections,
    sections_for,
)
from utils.constants import (  # noqa: E402
    AnswerMode,
    Intent,
    RevisionDeliverable,
    Subject,
    TestScope,
    intent_from_text,
    subject_from_text,
)
from utils.text_utils import (  # noqa: E402
    clean_text,
    extract_json_object,
    format_history,
    strip_answer_prefix,
    truncate,
)
from learning.assessment import grade, merge_answer_key, parse_questions  # noqa: E402
from learning.capabilities import (  # noqa: E402
    CapabilityNotAvailable,
    CapabilityState,
    default_registry,
    describe_capabilities,
)
from learning.catalog import ChapterCatalog, humanise  # noqa: E402
from learning.motivation import daily_motivation  # noqa: E402
from learning.progress import ProgressStore  # noqa: E402
from learning.sections import (  # noqa: E402
    CHAPTER_SECTIONS,
    REVISION_SECTIONS,
    chapter_sections,
    format_section_contract,
    sections_for_deliverable,
    stamp_sections,
)

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}{(' -> ' + detail) if detail and not condition else ''}")
    if not condition:
        FAILURES.append(label)


def main() -> int:
    # --- text normalisation -------------------------------------------------
    check("clean_text collapses spaces", clean_text("  a \t b \n\n\n c  ") == "a b\n\nc")
    check("clean_text handles None", clean_text(None) == "")
    check("truncate caps length", len(truncate("x" * 100, 20)) <= 20)
    check(
        "strip_answer_prefix removes prefixes",
        strip_answer_prefix("Answer: Farad") == "Farad",
    )

    # --- JSON extraction ----------------------------------------------------
    fenced = '```json\n{"subject": "physics", "confidence": 0.9}\n```'
    check("extract_json_object handles fences", (extract_json_object(fenced) or {}).get("subject") == "physics")
    nested = 'text {"a": {"b": 1}, "c": "{}"} trailing'
    check("extract_json_object handles nesting", extract_json_object(nested) == {"a": {"b": 1}, "c": "{}"})
    check("extract_json_object rejects junk", extract_json_object("nothing here") is None)

    # --- history rendering --------------------------------------------------
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "third"},
    ]
    rendered = format_history(history, max_turns=2)
    check("format_history keeps the tail", "first" not in rendered and "third" in rendered)
    check("format_history handles empty", format_history(None) == "(no previous conversation)")

    # --- subject / intent keywords -----------------------------------------
    cases = [
        ("Explain Newton's second law of motion.", Subject.PHYSICS),
        ("What is the hybridisation of carbon in benzene?", Subject.CHEMISTRY),
        ("Why is light required for photosynthesis?", Subject.BIOLOGY),
        ("Balance the redox reaction in acidic medium.", Subject.CHEMISTRY),
        ("Explain the s-block elements trend.", Subject.CHEMISTRY),
        ("A particle moves in a circular motion with constant speed.", Subject.PHYSICS),
        ("Which hormone regulates blood glucose?", Subject.BIOLOGY),
    ]
    for question, expected in cases:
        actual = subject_from_text(question)
        check(f"subject_from_text: {question[:38]}", actual == expected, f"expected {expected}, got {actual}")

    check("subject_from_text returns None for greetings", subject_from_text("Hello there") is None)
    check("intent: MCQs", intent_from_text("Generate 5 MCQs on kinematics") == Intent.MCQ)
    check("intent: PYQs", intent_from_text("Give me PYQs from NEET 2019") == Intent.PYQ)
    check("intent: study plan", intent_from_text("Make a study plan for 3 months") == Intent.STUDY_PLAN)
    check("intent: doubt default", intent_from_text("Why is the sky blue?") is None)

    # --- answer modes -------------------------------------------------------
    check(
        "the five required sections are the Detailed contract",
        EXPLANATION_SECTIONS
        == (
            "\U0001f4d8 Concept Explanation",
            "\U0001f3af NEET Importance",
            "\U0001f9e0 Key Points To Remember",
            "\u26a0 Common Student Mistakes",
            "\u2705 Quick Revision Notes",
        ),
        format_contract(AnswerMode.DETAILED),
    )
    check("mode coercion accepts UI labels", AnswerMode.coerce("Quick Revision") is AnswerMode.QUICK_REVISION)
    check("mode coercion falls back", AnswerMode.coerce("nonsense") is AnswerMode.DETAILED)
    check("mode labels match the requirement", [label for label, _ in AnswerMode.ui_choices()] == [
        "Detailed Explanation", "Quick Revision", "MCQ Practice", "PYQ Focus",
    ], str(AnswerMode.ui_choices()))
    check("MCQ mode serves the mcq intent", AnswerMode.MCQ_PRACTICE.satisfies(Intent.MCQ))
    check("PYQ mode serves the pyq intent", AnswerMode.PYQ_FOCUS.satisfies(Intent.PYQ))
    check("Detailed mode does not absorb mcq", not AnswerMode.DETAILED.satisfies(Intent.MCQ))
    check("planner intent has no mode yet", AnswerMode.for_intent(Intent.STUDY_PLAN) is None)
    check("every mode has a retrieval factor", all(mode.retrieval_factor > 0 for mode in AnswerMode))

    # --- format repair ------------------------------------------------------
    messy = "Concept Explanation\nThe cell is the basic unit of life.\n\nRevision\nRecall it daily."
    check(
        "missing_sections lists what the model skipped",
        missing_sections(messy, AnswerMode.DETAILED)
        == [
            "\U0001f3af NEET Importance",
            "\U0001f9e0 Key Points To Remember",
            "\u26a0 Common Student Mistakes",
        ],
        str(missing_sections(messy, AnswerMode.DETAILED)),
    )
    repaired = enforce_format(messy, AnswerMode.DETAILED, subject="biology", context="")
    check("headings are standardised", repaired.relabelled >= 1, repaired.summary())
    check("missing sections are appended", repaired.added == [
        "\U0001f3af NEET Importance",
        "\U0001f9e0 Key Points To Remember",
        "\u26a0 Common Student Mistakes",
    ], str(repaired.added))
    check("repaired text carries every canonical heading", all(
        title in repaired.text for title in sections_for(AnswerMode.DETAILED)
    ))
    check(
        "a complete answer is left untouched",
        not enforce_format(
            "\n\n".join(f"### {title}\n\n- point" for title in sections_for(AnswerMode.QUICK_REVISION)),
            AnswerMode.QUICK_REVISION,
        ).changed,
    )
    check("mcq contract differs from the explanation contract",
          format_contract(AnswerMode.MCQ_PRACTICE) != format_contract(AnswerMode.DETAILED))

    # --- prompt placeholder contract (stdlib only) --------------------------
    prompt_dir = PROJECT_ROOT / "prompts"
    required = {
        "router.txt": {"$history", "$question", "$subject_hint"},
        "answer_user.txt": {"$history", "$context", "$question", "$subject", "$mode"},
        "answer_user_no_context.txt": {"$history", "$question", "$subject", "$mode"},
        "answer_system.txt": set(),
        "persona.txt": set(),
        "mode_detailed.txt": set(),
        "mode_quick_revision.txt": set(),
        "mode_mcq_practice.txt": {"$mcq_count"},
        "mode_pyq_focus.txt": set(),
        "learn_chapter_user.txt": {"$subject", "$chapter", "$context", "$focus"},
        "learn_chapter_system.txt": set(),
        "mcq_user.txt": {"$subject", "$scope", "$target", "$mcq_count", "$context"},
        "mcq_system.txt": set(),
        "revision_user.txt": {"$subject", "$target", "$context", "$guidance"},
        "revision_system.txt": set(),
    }
    for filename, placeholders in required.items():
        path = prompt_dir / filename
        exists = path.exists()
        check(f"prompt exists: {filename}", exists)
        if exists:
            text = path.read_text(encoding="utf-8")
            missing = [token for token in placeholders if token not in text]
            check(f"placeholders in {filename}", not missing, f"missing {missing}")

    # Each mode template must name the section headings it is held to.
    for mode in AnswerMode:
        path = prompt_dir / f"mode_{mode.value}.txt"
        if path.exists():
            text = path.read_text(encoding="utf-8")
            absent = [title for title in sections_for(mode) if title.split(" ", 1)[1] not in text]
            check(f"mode_{mode.value}.txt lists its sections", not absent, f"missing {absent}")

    # --- platform: assessment + revision vocabulary -------------------------
    check(
        "Practice offers exactly the three test scopes",
        [scope.value for scope in TestScope] == ["subject", "chapter", "topic"],
        str([scope.value for scope in TestScope]),
    )
    check("scope coercion accepts UI labels", TestScope.coerce("Chapter Tests") is TestScope.CHAPTER)
    check("scope coercion has a safe default", TestScope.coerce("nonsense") is TestScope.CHAPTER)
    check(
        "Revision Hub offers the four required deliverables",
        {item.value for item in RevisionDeliverable}
        == {"flash_notes", "formula_sheet", "high_yield", "last_minute"},
        str(sorted(item.value for item in RevisionDeliverable)),
    )
    check(
        "deliverable coercion accepts labels",
        RevisionDeliverable.coerce("Formula Sheet") is RevisionDeliverable.FORMULA_SHEET,
    )

    # --- platform: section contracts ----------------------------------------
    check(
        "a chapter page is the four brief sections, in order",
        tuple(title.split(" ", 1)[1] for title in chapter_sections())
        == ("Chapter Summary", "High-Yield Concepts", "Important NEET Points", "Quick Revision Notes"),
        str(CHAPTER_SECTIONS),
    )
    check(
        "every deliverable has its own section shape",
        len({tuple(sections) for sections in REVISION_SECTIONS.values()}) == 4
        and all(len(sections) >= 3 for sections in REVISION_SECTIONS.values()),
    )
    check(
        "sections_for_deliverable accepts the stored key",
        sections_for_deliverable("last_minute") == REVISION_SECTIONS[RevisionDeliverable.LAST_MINUTE],
    )
    check(
        "the contract line is renderable",
        "|" in format_section_contract(CHAPTER_SECTIONS),
        format_section_contract(CHAPTER_SECTIONS),
    )
    repaired_sections = stamp_sections("### Chapter Summary\n\nFrom the passages.\n", CHAPTER_SECTIONS)
    check(
        "a skipped study section is filled, never invented",
        len(repaired_sections.added) == 3
        and all(title.split(" ", 1)[1] in repaired_sections.text for title in CHAPTER_SECTIONS),
        repaired_sections.summary(),
    )

    # --- platform: chapter catalog (no hard-coded syllabus) -----------------
    catalog = ChapterCatalog.from_sources(
        ["physics/mechanics.pdf", "physics/plant_physiology.pdf", "chemistry/mole_concept.pdf"],
        subject_chunks={"physics": 2, "chemistry": 1},
    )
    check("catalog groups sources into chapters", len(catalog.chapters) == 3, str(catalog.summary()))
    check(
        "chapter names are humanised from the file name",
        humanise("plant_physiology") == "Plant Physiology",
        humanise("plant_physiology"),
    )
    check("catalog knows its subjects", sorted(catalog.subjects) == ["chemistry", "physics"], str(catalog.subjects))
    check("catalog counts per subject", catalog.count_for("physics") == 2, str(catalog.summary()))
    check("an empty index is an empty catalog", ChapterCatalog.from_sources([]).is_empty)
    check(
        "the sidecar payload shape is understood",
        ChapterCatalog.from_metadata({"sources": ["biology/heart.pdf"]}).chapters[0].subject == "biology",
    )
    check("a sidecar without sources means 'read later'", ChapterCatalog.from_metadata({"chunks": 3}) is None)

    # --- platform: MCQ parsing, marking and analytics -----------------------
    raw_questions = """
Q1. Which quantity is measured in the newton metre?
(1) Force
(2) Work
(3) Power
(4) Pressure
Answer: (2)
Explanation: Work is force times displacement in the direction of the force.
TOPIC: Units and Measurement

Q2. The SI unit of electric charge is:
1. ampere
2. coulomb
3. volt
4. ohm
Explanation: Charge is measured in coulombs.
TOPIC: Current Electricity

Answer Key
1: (2) - definition of work
2: (2) - coulomb is the unit of charge
"""
    parsed = parse_questions(raw_questions)
    check("the parser finds both questions", len(parsed) == 2, f"got {len(parsed)}")
    check("options are captured", all(len(question.options) == 4 for question in parsed))
    check("explanations are captured", parsed[0].explanation.startswith("Work is force"), parsed[0].explanation)
    check("topics are captured for weak-area analytics", parsed[1].topic == "Current Electricity", parsed[1].topic)
    check("an answer key is not mistaken for a question", not any("definition of work" in q.stem for q in parsed))
    filled = merge_answer_key(parsed, "1: 2\n2: 2")
    check("the answer key is merged in", all(question.correct == 2 for question in filled), str(filled))
    graded = grade(
        filled,
        [2, 1],
        marks_correct=4.0,
        marks_incorrect=-1.0,
        scope=TestScope.CHAPTER,
        subject="physics",
        chapter="Mechanics",
    )
    check("NEET marking is applied", graded.score == 3.0 and graded.max_score == 8.0, f"{graded.score}/{graded.max_score}")
    check("accuracy is computed", abs(graded.accuracy - 0.5) < 1e-9, str(graded.accuracy))
    check("weak topics are named", graded.weak_topics() == ["Current Electricity"], str(graded.weak_topics()))
    check("a report can be rendered", "**Score:**" in graded.report_markdown(reveal=True))

    # --- platform: JSON progress store (the database seam) ------------------
    with tempfile.TemporaryDirectory() as tmp:
        store = ProgressStore(Path(tmp), profile="Offline Check")
        store.record(graded)
        store.bump("doubts")
        stats = store.stats()
        check("the store writes one JSON document per profile", store.file.name == "offline_check.json", store.file.name)
        check("tests attempted are tracked", stats["tests"] == 1, str(stats))
        check("scores and accuracy are tracked", stats["score"] == 3.0 and stats["accuracy"] == 0.5, str(stats))
        check("non-test activity is counted", stats["doubts"] == 1, str(stats))
        check("subject performance is aggregated", "physics" in store.subject_performance(), str(store.subject_performance()))
        check("chapter performance is aggregated", "physics::Mechanics" in store.chapter_performance(), str(store.chapter_performance()))
        check("weak areas are ranked", [area.name for area in store.weak_areas()] == ["Current Electricity"], str(store.weak_areas()))
        check("the daily series covers two weeks", len(store.daily_series()) == 14)
        check("a re-opened store sees the history", ProgressStore(Path(tmp), profile="Offline Check").stats()["tests"] == 1)
        store.clear()
        check("history can be reset", store.stats()["tests"] == 0)

    # --- platform: capability extension points (voice / image not built) ----
    registry = default_registry()
    check("voice and image are planned, not active", len(registry.planned()) >= 2, str(registry.as_dict()))
    try:
        registry.require("voice")
        check("an unregistered capability raises", False)
    except CapabilityNotAvailable as exc:
        check("an unregistered capability raises", True)
        message = exc.user_message().lower()
        check(
            "the failure message is student-friendly",
            "voice" in message and "roadmap" in message and "genieplatform.ask" in message,
            exc.user_message(),
        )

    class _Mic:
        def transcribe(self, audio: bytes, *, language: str = "en") -> str:
            return "What is the unit of permittivity?"

    registered = registry.register("voice", _Mic())
    check("registering a backend activates the capability", registered.state is CapabilityState.ACTIVE)
    check("the funnel then gets a transcript", registry.require("voice").transcribe(b"") != "")
    table_text = describe_capabilities(registry)
    check("the extension points are documentable", "VoiceTranscriber.transcribe" in table_text, table_text)

    # --- platform: navigation contract (Gradio-free half of ui/) ------------
    from ui import components as genie_ui  # noqa: PLC0415 - stdlib-only import

    check(
        "the navigation is the six modules in brief order",
        [label for label, _ in genie_ui.MODULE_TABS]
        == ["Dashboard", "Learn", "Practice", "Ask Genie", "Progress", "Revision Hub"],
        str(genie_ui.MODULE_TABS),
    )
    check(
        "tab ids are stable for programmatic jumps",
        [key for _, key in genie_ui.MODULE_TABS]
        == ["dashboard", "learn", "practice", "ask-genie", "progress", "revision"],
        str(genie_ui.MODULE_TABS),
    )
    check("table cells are escaped", genie_ui.esc("a | b\nc") == "a \\| b c", genie_ui.esc("a | b\nc"))
    check("markdown tables render", genie_ui.table(("A",), ((1, "x"),)).count("|") >= 6)
    check("accuracy bars are bounded", len(genie_ui.bar(1.9)) == 10 and genie_ui.bar(0.0).startswith("\u2591"))
    check("percent formatting tolerates junk", genie_ui.pct(None) == "0%" and genie_ui.pct(0.7532) == "75%")
    check("the empty-catalog notice is honest", "yet" in genie_ui.weak_areas_md([]))
    check("daily motivation always has a line", bool(daily_motivation().line))

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
        return 1
    print("All offline checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
