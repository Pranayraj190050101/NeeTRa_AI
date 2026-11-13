"""Domain vocabulary shared by every layer of the application.

Keeping subjects, intents and keyword tables in one place means the router,
the retrieval filter, the prompts and the future agents always agree on the
exact string values that circulate through the LangGraph state.
"""

from __future__ import annotations

import re
from enum import Enum

__all__ = [
    "Subject",
    "Intent",
    "AnswerMode",
    "TestScope",
    "RevisionDeliverable",
    "MODE_ALIASES",
    "SUBJECT_KEYWORDS",
    "SUBJECT_HINT_OVERRIDES",
    "INTENT_KEYWORDS",
    "subject_from_text",
    "intent_from_text",
    "normalise_token",
]


class Subject(str, Enum):
    """NEET subjects plus a catch-all bucket."""

    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    GENERAL = "general"

    @classmethod
    def coerce(cls, value: object, default: "Subject | None" = None) -> "Subject | None":
        """Best-effort conversion of an arbitrary value into a :class:`Subject`."""
        if isinstance(value, cls):
            return value
        if value is None:
            return default
        text = str(value).strip().lower()
        if not text:
            return default
        for member in cls:
            if text == member.value:
                return member
        return subject_from_text(text) or default

    @property
    def label(self) -> str:
        """Human readable name for the UI."""
        return self.value.capitalize()


class Intent(str, Enum):
    """What the student wants to do with the tutor."""

    DOUBT = "doubt"
    MCQ = "mcq"
    PYQ = "pyq"
    STUDY_PLAN = "study_plan"
    WEAKNESS = "weakness"

    @classmethod
    def coerce(cls, value: object, default: "Intent | None" = None) -> "Intent":
        """Best-effort conversion, falling back to ``DOUBT``."""
        fallback = default or cls.DOUBT
        if isinstance(value, cls):
            return value
        if value is None:
            return fallback
        text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "question": cls.DOUBT,
            "doubt": cls.DOUBT,
            "explain": cls.DOUBT,
            "mcq": cls.MCQ,
            "quiz": cls.MCQ,
            "practice": cls.MCQ,
            "pyq": cls.PYQ,
            "previous_year": cls.PYQ,
            "past_paper": cls.PYQ,
            "study_plan": cls.STUDY_PLAN,
            "planner": cls.STUDY_PLAN,
            "schedule": cls.STUDY_PLAN,
            "weakness": cls.WEAKNESS,
            "analysis": cls.WEAKNESS,
            "weak_area": cls.WEAKNESS,
        }
        return aliases.get(text, fallback)


class AnswerMode(str, Enum):
    """How the coach presents its answer - chosen by the student per question.

    Modes are a *presentation* concern and live next to :class:`Intent`, which
    is a *request* concern detected by the router. A mode can satisfy a
    compatible intent (``MCQ_PRACTICE`` satisfies ``Intent.MCQ``); specialist
    intents are routed to their dedicated agents.
    """

    DETAILED = "detailed"
    QUICK_REVISION = "quick_revision"
    MCQ_PRACTICE = "mcq_practice"
    PYQ_FOCUS = "pyq_focus"

    # ------------------------------------------------------------------ parse
    @classmethod
    def coerce(cls, value: object, default: "AnswerMode | None" = None) -> "AnswerMode":
        """Best-effort conversion, falling back to ``DETAILED``."""
        fallback = default or cls.DETAILED
        if isinstance(value, cls):
            return value
        if value is None:
            return fallback
        text = normalise_token(str(value)).replace(" ", "_")
        if text in MODE_ALIASES:
            return MODE_ALIASES[text]
        try:
            return cls(text)
        except ValueError:
            return fallback

    # ------------------------------------------------------------- presentation
    @property
    def label(self) -> str:
        """Label shown on the Gradio radio button."""
        return _MODE_LABELS[self]

    @property
    def emoji(self) -> str:
        return _MODE_EMOJI[self]

    @property
    def description(self) -> str:
        """One-line help text for the UI and the architecture panel."""
        return _MODE_DESCRIPTIONS[self]

    @property
    def prompt_name(self) -> str:
        """Key of this mode's template in :mod:`prompts.loader`."""
        return f"mode_{self.value}"

    # ------------------------------------------------------------- budget hints
    @property
    def retrieval_factor(self) -> float:
        """How many more/fewer passages the mode wants (multiplies ``top_k``)."""
        return {
            self.DETAILED: 1.0,
            self.QUICK_REVISION: 0.8,
            self.MCQ_PRACTICE: 1.4,
            self.PYQ_FOCUS: 1.2,
        }[self]

    @property
    def token_factor(self) -> float:
        """Multiplier on ``llm_max_new_tokens`` - revision sheets are short."""
        return {
            self.DETAILED: 0.8,
            self.QUICK_REVISION: 0.8,
            self.MCQ_PRACTICE: 1.5,
            self.PYQ_FOCUS: 1.1,
        }[self]

    @property
    def char_factor(self) -> float:
        """Multiplier on ``max_answer_chars`` used by the post-processing cap."""
        return {
            self.DETAILED: 1.0,
            self.QUICK_REVISION: 0.5,
            self.MCQ_PRACTICE: 1.6,
            self.PYQ_FOCUS: 1.1,
        }[self]

    # ------------------------------------------------------------ intent bridge
    def satisfies(self, intent: object) -> bool:
        """``True`` when this mode can serve ``intent`` right now."""
        return intent in _MODE_INTENTS[self]

    @classmethod
    def for_intent(cls, intent: object) -> "AnswerMode | None":
        """The mode that would serve ``intent`` (used for UI hints)."""
        resolved = Intent.coerce(intent) if not isinstance(intent, Intent) else intent
        for mode in cls:
            if resolved in _MODE_INTENTS[mode]:
                return mode
        return None

    @classmethod
    def ui_choices(cls) -> list[tuple[str, str]]:
        """``(label, value)`` pairs for the Gradio answer-mode selector."""
        return [(mode.label, mode.value) for mode in cls]


#: Free-text spellings accepted by :meth:`AnswerMode.coerce`.
MODE_ALIASES: dict[str, AnswerMode] = {
    "detailed": AnswerMode.DETAILED,
    "detailed_explanation": AnswerMode.DETAILED,
    "explanation": AnswerMode.DETAILED,
    "concept": AnswerMode.DETAILED,
    "default": AnswerMode.DETAILED,
    "quick": AnswerMode.QUICK_REVISION,
    "quick_revision": AnswerMode.QUICK_REVISION,
    "revision": AnswerMode.QUICK_REVISION,
    "revision_sheet": AnswerMode.QUICK_REVISION,
    "one_pager": AnswerMode.QUICK_REVISION,
    "mcq": AnswerMode.MCQ_PRACTICE,
    "mcqs": AnswerMode.MCQ_PRACTICE,
    "mcq_practice": AnswerMode.MCQ_PRACTICE,
    "practice": AnswerMode.MCQ_PRACTICE,
    "quiz": AnswerMode.MCQ_PRACTICE,
    "pyq": AnswerMode.PYQ_FOCUS,
    "pyqs": AnswerMode.PYQ_FOCUS,
    "pyq_focus": AnswerMode.PYQ_FOCUS,
    "previous_year": AnswerMode.PYQ_FOCUS,
}

_MODE_LABELS: dict[AnswerMode, str] = {
    AnswerMode.DETAILED: "Detailed Explanation",
    AnswerMode.QUICK_REVISION: "Quick Revision",
    AnswerMode.MCQ_PRACTICE: "MCQ Practice",
    AnswerMode.PYQ_FOCUS: "PYQ Focus",
}

_MODE_EMOJI: dict[AnswerMode, str] = {
    AnswerMode.DETAILED: "\U0001f4d8",  # blue book
    AnswerMode.QUICK_REVISION: "\u26a1",  # high voltage
    AnswerMode.MCQ_PRACTICE: "\U0001f9e9",  # puzzle piece
    AnswerMode.PYQ_FOCUS: "\U0001f4dc",  # scroll
}

_MODE_DESCRIPTIONS: dict[AnswerMode, str] = {
    AnswerMode.DETAILED: (
        "Full coach-style answer: concept, NEET importance, key points, mistakes, revision notes."
    ),
    AnswerMode.QUICK_REVISION: (
        "Same structure, compressed into a one-page revision sheet you can skim before the exam."
    ),
    AnswerMode.MCQ_PRACTICE: (
        "NEET-style single-correct MCQs from the retrieved chapters, with answer key and trap notes."
    ),
    AnswerMode.PYQ_FOCUS: (
        "How NEET has historically framed this topic, the lines NCERT turns into questions and the "
        "exceptions to memorise."
    ),
}

#: Which router intents each answer mode can already serve.
_MODE_INTENTS: dict[AnswerMode, frozenset[Intent]] = {
    AnswerMode.DETAILED: frozenset({Intent.DOUBT}),
    AnswerMode.QUICK_REVISION: frozenset({Intent.DOUBT}),
    AnswerMode.MCQ_PRACTICE: frozenset({Intent.DOUBT, Intent.MCQ}),
    AnswerMode.PYQ_FOCUS: frozenset({Intent.DOUBT, Intent.PYQ}),
}


#: What a practice test covers.  The three scopes share one generation and
#: grading pipeline; only the retrieval target and the label differ.
class TestScope(str, Enum):
    """Granularity of an assessment in the Practice module."""

    SUBJECT = "subject"
    CHAPTER = "chapter"
    TOPIC = "topic"

    @classmethod
    def coerce(cls, value: object, default: "TestScope | None" = None) -> "TestScope":
        """Accept ``chapter test``, ``Chapter``, ``syllabus`` and similar spellings."""
        fallback = default or cls.CHAPTER
        if isinstance(value, cls):
            return value
        if value is None:
            return fallback
        text = normalise_token(str(value)).replace(" ", "_")
        aliases = {
            "subject": cls.SUBJECT,
            "subject_test": cls.SUBJECT,
            "full_syllabus": cls.SUBJECT,
            "chapter": cls.CHAPTER,
            "chapter_test": cls.CHAPTER,
            "topic": cls.TOPIC,
            "topic_test": cls.TOPIC,
            "concept": cls.TOPIC,
        }
        return aliases.get(text, fallback)

    @property
    def label(self) -> str:
        return _SCOPE_LABELS[self]

    @property
    def emoji(self) -> str:
        return _SCOPE_EMOJI[self]

    @property
    def description(self) -> str:
        return _SCOPE_DESCRIPTIONS[self]

    @property
    def retrieval_factor(self) -> float:
        """How much source material the scope needs relative to ``top_k``."""
        return {
            self.SUBJECT: 2.0,
            self.CHAPTER: 1.4,
            self.TOPIC: 1.0,
        }[self]

    @classmethod
    def ui_choices(cls) -> list[tuple[str, str]]:
        """``(label, value)`` pairs for the Gradio scope selector."""
        return [(f"{scope.emoji} {scope.label}", scope.value) for scope in cls]


_SCOPE_LABELS: dict[TestScope, str] = {
    TestScope.SUBJECT: "Subject Test",
    TestScope.CHAPTER: "Chapter Test",
    TestScope.TOPIC: "Topic Test",
}

_SCOPE_EMOJI: dict[TestScope, str] = {
    TestScope.SUBJECT: "\U0001f3af",  # target
    TestScope.CHAPTER: "\U0001f4da",  # books
    TestScope.TOPIC: "\U0001f9e9",  # puzzle piece
}

_SCOPE_DESCRIPTIONS: dict[TestScope, str] = {
    TestScope.SUBJECT: "Mixed questions across every indexed chapter of one subject.",
    TestScope.CHAPTER: "A test built from a single chapter of your own textbook PDF.",
    TestScope.TOPIC: "A drill narrowed to one concept, formula or topic line.",
}


#: The four revision deliverables of the Revision Hub.  Each one promises a
#: fixed section shape (see ``learning.sections``) so the sheets stay reusable.
class RevisionDeliverable(str, Enum):
    """What kind of revision material the student asks the hub to produce."""

    FLASH_NOTES = "flash_notes"
    FORMULA_SHEET = "formula_sheet"
    HIGH_YIELD = "high_yield"
    LAST_MINUTE = "last_minute"

    @classmethod
    def coerce(cls, value: object, default: "RevisionDeliverable | None" = None) -> "RevisionDeliverable":
        """Accept friendly spellings such as ``formulas`` or ``one liner``."""
        fallback = default or cls.FLASH_NOTES
        if isinstance(value, cls):
            return value
        if value is None:
            return fallback
        text = normalise_token(str(value)).replace(" ", "_")
        aliases = {
            "flash": cls.FLASH_NOTES,
            "flash_notes": cls.FLASH_NOTES,
            "flashcard": cls.FLASH_NOTES,
            "flash_cards": cls.FLASH_NOTES,
            "one_pager": cls.FLASH_NOTES,
            "formula": cls.FORMULA_SHEET,
            "formulas": cls.FORMULA_SHEET,
            "formula_sheet": cls.FORMULA_SHEET,
            "equations": cls.FORMULA_SHEET,
            "high_yield": cls.HIGH_YIELD,
            "high_yield_points": cls.HIGH_YIELD,
            "important_points": cls.HIGH_YIELD,
            "last_minute": cls.LAST_MINUTE,
            "last_minute_revision": cls.LAST_MINUTE,
            "exam_day": cls.LAST_MINUTE,
            "night_before": cls.LAST_MINUTE,
        }
        return aliases.get(text, fallback)

    @property
    def label(self) -> str:
        return _DELIVERABLE_LABELS[self]

    @property
    def emoji(self) -> str:
        return _DELIVERABLE_EMOJI[self]

    @property
    def description(self) -> str:
        return _DELIVERABLE_DESCRIPTIONS[self]

    @property
    def retrieval_factor(self) -> float:
        """Passage multiplier - formula and last-minute sheets want tight input."""
        return {
            self.FLASH_NOTES: 1.0,
            self.FORMULA_SHEET: 1.2,
            self.HIGH_YIELD: 1.4,
            self.LAST_MINUTE: 0.8,
        }[self]

    @classmethod
    def ui_choices(cls) -> list[tuple[str, str]]:
        """``(label, value)`` pairs for the Gradio deliverable selector."""
        return [(f"{item.emoji} {item.label}", item.value) for item in cls]


_DELIVERABLE_LABELS: dict[RevisionDeliverable, str] = {
    RevisionDeliverable.FLASH_NOTES: "Flash Notes",
    RevisionDeliverable.FORMULA_SHEET: "Formula Sheet",
    RevisionDeliverable.HIGH_YIELD: "High Yield Points",
    RevisionDeliverable.LAST_MINUTE: "Last Minute Revision",
}

_DELIVERABLE_EMOJI: dict[RevisionDeliverable, str] = {
    RevisionDeliverable.FLASH_NOTES: "\U0001f5d2\ufe0f",  # memo
    RevisionDeliverable.FORMULA_SHEET: "\U0001f9ee",  # abacus
    RevisionDeliverable.HIGH_YIELD: "\u26a1",  # high voltage
    RevisionDeliverable.LAST_MINUTE: "\u23f1\ufe0f",  # stopwatch
}

_DELIVERABLE_DESCRIPTIONS: dict[RevisionDeliverable, str] = {
    RevisionDeliverable.FLASH_NOTES: (
        "One-line recall cards - term on the left, the exact NCERT fact on the right."
    ),
    RevisionDeliverable.FORMULA_SHEET: (
        "Every formula for the target with symbols, units and the condition it holds under."
    ),
    RevisionDeliverable.HIGH_YIELD: (
        "The lines NCERT turns into statements, assertion-reason and numericals - with why."
    ),
    RevisionDeliverable.LAST_MINUTE: (
        "The 60-second recap to read outside the exam hall, including the traps to avoid."
    ),
}


# Explicit folder / dropdown names win over fuzzy matching.
SUBJECT_HINT_OVERRIDES: dict[str, Subject] = {
    "physics": Subject.PHYSICS,
    "phy": Subject.PHYSICS,
    "chemistry": Subject.CHEMISTRY,
    "chem": Subject.CHEMISTRY,
    "biology": Subject.BIOLOGY,
    "bio": Subject.BIOLOGY,
    "botany": Subject.BIOLOGY,
    "zoology": Subject.BIOLOGY,
    "general": Subject.GENERAL,
}

# Order matters: chemistry is checked before physics so that
# "physical chemistry" is not classified as physics.
SUBJECT_KEYWORDS: dict[Subject, tuple[str, ...]] = {
    Subject.CHEMISTRY: (
        "chemistry",
        "chemical",
        "mole",
        "molar",
        "molarity",
        "stoichiometry",
        "organic",
        "inorganic",
        "equilibrium",
        "electrochemistry",
        "hydrocarbon",
        "aldehyde",
        "ketone",
        "alcohol",
        "amine",
        "carboxylic",
        "phenol",
        "alkane",
        "alkene",
        "alkyne",
        "benzene",
        "isomer",
        "periodic table",
        "electronegativity",
        "hybridisation",
        "hybridization",
        "titration",
        "ph",
        "buffer",
        "redox",
        "oxidation",
        "reduction",
        "catalyst",
        "polymer",
        "biomolecule",
        "coordination compound",
        "d-block",
        "p-block",
        "s-block",
        "chemical bonding",
        "avogadro",
        "gibbs",
        "enthalpy",
        "entropy",
    ),
    Subject.PHYSICS: (
        "physics",
        "mechanics",
        "kinematics",
        "newton",
        "velocity",
        "acceleration",
        "momentum",
        "impulse",
        "friction",
        "gravitation",
        "work energy",
        "power",
        "rotational",
        "torque",
        "moment of inertia",
        "oscillation",
        "shm",
        "wave",
        "sound",
        "optics",
        "lens",
        "mirror",
        "refraction",
        "interference",
        "diffraction",
        "electrostatics",
        "coulomb",
        "capacitor",
        "capacitance",
        "current",
        "resistance",
        "ohm",
        "kirchhoff",
        "magnetic",
        "induction",
        "transformer",
        "alternating current",
        "photoelectric",
        "nuclear",
        "radioactivity",
        "semiconductor",
        "diode",
        "transistor",
        "thermodynamics",
        "calorimetry",
        "circular motion",
        "projectile",
        "electric field",
        "potential difference",
    ),
    Subject.BIOLOGY: (
        "biology",
        "botany",
        "zoology",
        "cell",
        "cell cycle",
        "mitosis",
        "meiosis",
        "photosynthesis",
        "respiration",
        "enzyme",
        "biomolecule",
        "genetics",
        "inheritance",
        "molecular basis of inheritance",
        "dna",
        "rna",
        "gene",
        "allele",
        "mendel",
        "evolution",
        "ecology",
        "ecosystem",
        "biodiversity",
        "plant",
        "animal",
        "tissue",
        "organ",
        "digestion",
        "circulation",
        "excretion",
        "neuron",
        "hormone",
        "endocrine",
        "reproduction",
        "embryo",
        "biotechnology",
        "immunity",
        "pathogen",
        "virus",
        "bacteria",
        "fungi",
        "taxonomy",
        "kingdom",
        "human health",
        "nutrition",
        "osmoregulation",
        "respiration in plants",
    ),
}

INTENT_KEYWORDS: dict[Intent, tuple[str, ...]] = {
    Intent.MCQ: (
        "mcq",
        "mcqs",
        "multiple choice",
        "objective question",
        "quiz me",
        "generate questions",
        "practice questions",
        "test me",
        "mock test",
    ),
    Intent.PYQ: (
        "pyq",
        "pyqs",
        "previous year",
        "past paper",
        "last year question",
        "neet 20",
        "aipmt",
    ),
    Intent.STUDY_PLAN: (
        "study plan",
        "revision plan",
        "weekly plan",
        "week by week plan",
        "week by week",
        "planner",
        "timetable",
        "schedule",
        "roadmap",
        "how should i prepare",
        "study strategy",
    ),
    Intent.WEAKNESS: (
        "weak",
        "weakness",
        "improve my score",
        "analyse my performance",
        "analyze my performance",
        "where am i lacking",
        "score analysis",
    ),
}


def normalise_token(text: str) -> str:
    """Lower-case and collapse punctuation/whitespace for keyword matching."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _has_phrase(text: str, phrase: str) -> bool:
    r"""Whole-phrase match that also works for hyphenated keywords (``s-block``).

    The haystack is expected to be normalised (see :func:`normalise_token`), so
    every non-alphanumeric separator in both sides is widened to ``[\s-]+``.
    """
    words = [re.escape(part) for part in re.split(r"[^a-z0-9]+", phrase.lower()) if part]
    if not words:
        return False
    pattern = r"(?<![a-z0-9])" + r"[\s\-]+".join(words) + r"(?![a-z0-9])"
    return re.search(pattern, text) is not None


def subject_from_text(text: str | None) -> Subject | None:
    """Infer a subject from free text, a filename or a folder name.

    Exact tokens such as ``bio`` or ``physics`` are checked first, then the
    keyword tables.  ``None`` is returned when nothing matches so callers can
    decide on their own default.
    """
    if not text:
        return None
    normalised = normalise_token(text)
    if normalised in SUBJECT_HINT_OVERRIDES:
        return SUBJECT_HINT_OVERRIDES[normalised]
    for subject, keywords in SUBJECT_KEYWORDS.items():
        if any(_has_phrase(normalised, keyword) for keyword in keywords):
            return subject
    return None


def intent_from_text(text: str | None) -> Intent | None:
    """Infer the student's intent from keyword cues (LLM-free fallback)."""
    if not text:
        return None
    normalised = normalise_token(text)
    planning_request = any(
        _has_phrase(normalised, phrase)
        for phrase in (
            "study plan",
            "revision plan",
            "weekly plan",
            "week by week",
            "timetable",
            "schedule",
            "planner",
        )
    )
    if not planning_request and any(
        _has_phrase(normalised, phrase)
        for phrase in ("quick revision notes", "revision notes", "revision sheet")
    ):
        return Intent.DOUBT
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(_has_phrase(normalised, keyword) for keyword in keywords):
            return intent
    return None
