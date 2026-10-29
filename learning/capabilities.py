"""Extension points for the capabilities the platform is *designed* for.

Voice doubts and photo-based question solving are the next two input channels
of a NEET learning platform.  They are deliberately **not implemented** here;
what is implemented is the seam they will slot into, so that adding them later
does not reshape the services or the UI:

* ``VoiceTranscriber`` / ``ImageSolver`` are ``runtime_checkable`` protocols -
  any object with the right method can be registered.
* A capability reports ``state`` (``active`` / ``planned`` / ``disabled``) plus
  a ``hook`` name, which is what the UI prints on its "coming soon" card and
  what the future implementation registers against.
* Until a real backend is registered, the platform's ``listen()`` /
  ``solve_photo()`` entry points raise :class:`CapabilityNotAvailable`, which
  the UI turns into a friendly message instead of a traceback.

The existing chat path is unaffected: ``GeniePlatform.ask`` is the single
funnel every input channel - text, voice or image - will use.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from utils.constants import AnswerMode

__all__ = [
    "CapabilityState",
    "Capability",
    "CapabilityNotAvailable",
    "CapabilityRegistry",
    "VoiceTranscriber",
    "ImageSolver",
    "DEFAULT_CAPABILITIES",
    "default_registry",
    "describe_capabilities",
]


class CapabilityState(str, Enum):
    """Whether a capability can be used right now."""

    ACTIVE = "active"
    PLANNED = "planned"
    DISABLED = "disabled"

    @property
    def badge(self) -> str:
        return {"active": "\u2705 live", "planned": "\U0001f5d7\ufe0f planned",
                "disabled": "\u26d4 turned off"}[self.value]


@runtime_checkable
class VoiceTranscriber(Protocol):
    """Contract a future speech-to-text backend must satisfy."""

    def transcribe(self, audio: bytes, *, language: str = "en") -> str:
        """Return the spoken words found in ``audio``."""
        ...


@runtime_checkable
class ImageSolver(Protocol):
    """Contract a future OCR + vision-language backend must satisfy."""

    def extract_question(self, image: bytes, *, subject: str | None = None) -> str:
        """Read the question text (and any figure caption) out of a photo."""
        ...


@dataclass(frozen=True, slots=True)
class Capability:
    """One input/output channel of the platform."""

    key: str
    title: str
    description: str
    state: CapabilityState
    hook: str
    roadmap: str = ""

    @property
    def available(self) -> bool:
        return self.state is CapabilityState.ACTIVE

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "description": self.description,
            "state": self.state.value,
            "badge": self.state.badge,
            "hook": self.hook,
            "roadmap": self.roadmap,
            "available": self.available,
        }


class CapabilityNotAvailable(RuntimeError):
    """Raised when a student triggers a capability that is not wired up yet."""

    def __init__(self, capability: Capability) -> None:
        self.capability = capability
        super().__init__(
            f"{capability.title} is not available yet. {capability.roadmap}".strip()
        )

    def user_message(self) -> str:
        """Chat-ready explanation with the concrete next step."""
        return (
            f"**{self.capability.title} is on the roadmap, not in this build.**\n\n"
            f"- What it will do: {self.capability.description}\n"
            f"- Extension point already in place: `{self.capability.hook}`\n"
            f"- Until then: {self.capability.roadmap}"
        )


#: The channels the platform knows about.  Text is live; the rest are seams.
DEFAULT_CAPABILITIES: tuple[Capability, ...] = (
    Capability(
        key="chat",
        title="Ask Genie (text)",
        description="Type a doubt and get a grounded, coach-format answer.",
        state=CapabilityState.ACTIVE,
        hook="GeniePlatform.ask",
    ),
    Capability(
        key="voice",
        title="Voice doubts",
        description="Speak a doubt; a speech-to-text backend turns it into a question and "
        "the same coach pipeline answers it.",
        state=CapabilityState.PLANNED,
        hook="VoiceTranscriber.transcribe -> GeniePlatform.ask",
        roadmap="register a VoiceTranscriber in `learning/capabilities.py` and wire a "
        "`gr.Audio` input in the Ask Genie tab.",
    ),
    Capability(
        key="image",
        title="Image question solving",
        description="Photograph a textbook or exam question; OCR plus a vision model extract "
        "the stem and the coach solves it with citations.",
        state=CapabilityState.PLANNED,
        hook="ImageSolver.extract_question -> GeniePlatform.ask",
        roadmap="register an ImageSolver and add a `gr.Image` input; retrieval and grading "
        "reuse the existing pipeline unchanged.",
    ),
    Capability(
        key="study_plan",
        title="Study planner",
        description="Week-by-week plan built from the syllabus coverage and the progress store.",
        state=CapabilityState.PLANNED,
        hook="agents/registry.py -> Intent.STUDY_PLAN",
        roadmap="implement the planner agent; the router already emits this intent.",
    ),
    Capability(
        key="weakness",
        title="Weakness analysis",
        description="Longitudinal weakness report over every recorded test.",
        state=CapabilityState.PLANNED,
        hook="agents/registry.py -> Intent.WEAKNESS",
        roadmap="the Progress tab already aggregates the data this agent needs.",
    ),
)


class CapabilityRegistry:
    """Small name -> backend registry with a friendly failure mode."""

    def __init__(self, capabilities: tuple[Capability, ...] = DEFAULT_CAPABILITIES) -> None:
        self._capabilities = {capability.key: capability for capability in capabilities}
        self._backends: dict[str, Any] = {}

    def register(self, key: str, backend: Any) -> Capability:
        """Attach a real implementation and mark the capability active."""
        capability = self._capabilities.get(key)
        if capability is None:
            raise KeyError(f"Unknown capability '{key}'. Known: {sorted(self._capabilities)}")
        self._capabilities[key] = Capability(
            key=capability.key,
            title=capability.title,
            description=capability.description,
            state=CapabilityState.ACTIVE,
            hook=capability.hook,
            roadmap=capability.roadmap,
        )
        self._backends[key] = backend
        return self._capabilities[key]

    def get(self, key: str) -> Capability | None:
        return self._capabilities.get(key)

    def list(self) -> list[Capability]:
        return list(self._capabilities.values())

    def planned(self) -> list[Capability]:
        return [item for item in self._capabilities.values() if not item.available]

    def backend(self, key: str) -> Any | None:
        return self._backends.get(key)

    def require(self, key: str) -> Any:
        """Return the backend or raise :class:`CapabilityNotAvailable`."""
        capability = self._capabilities.get(key)
        if capability is None:
            raise KeyError(f"Unknown capability '{key}'")
        backend = self._backends.get(key)
        if backend is None or not capability.available:
            raise CapabilityNotAvailable(capability)
        return backend

    def as_dict(self) -> dict[str, Any]:
        return {
            "capabilities": [capability.as_dict() for capability in self.list()],
            "planned": [item.key for item in self.planned()],
        }


def default_registry() -> CapabilityRegistry:
    """A registry holding the shipped capability set (text only is active)."""
    return CapabilityRegistry()


def describe_capabilities(registry: CapabilityRegistry) -> str:
    """Markdown table of every channel and its state (used by the UI)."""
    rows = "\n".join(
        f"| {item.title} | {item.state.badge} | `{item.hook}` |" for item in registry.list()
    )
    return (
        "| Channel | State | Extension point |\n"
        "| --- | --- | --- |\n"
        f"{rows}"
    )


#: The answer mode a voice or image turn falls back to when the channel has no
#: selector of its own yet (see :class:`CapabilityRegistry`).
DEFAULT_ANSWER_MODE: AnswerMode = AnswerMode.DETAILED
