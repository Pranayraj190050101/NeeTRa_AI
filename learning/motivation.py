"""Daily motivation for the Dashboard.

A coach greets a student, but a product must not invent quotations attributed
to real people, so every line here is Genie's own - no fake NEET toppers, no
fabricated quotes.  The line is chosen deterministically from the calendar day
so the dashboard is stable within a day and refreshes overnight.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Sequence

__all__ = ["Motivation", "daily_motivation", "MOTIVATION_LINES"]

MOTIVATION_LINES: tuple[str, ...] = (
    "One chapter today beats ten chapters on the plan. Pick one and finish it.",
    "NCERT is not reading material - it is a scoring system. Read it with a pen.",
    "Accuracy you build today is marks you do not lose in May.",
    "Revision beats volume: five passages recalled beat fifty highlighted.",
    "Every mistake you analyse now is a repeat exam question you cancel.",
    "You do not need to be fast. You need to be right under pressure.",
    "The trap in a NEET question is usually the sentence you skimmed twice.",
    "Solve, check, note why. That loop is the whole preparation.",
    "A formula without its condition is a half answer - NEET marks the condition.",
    "Consistency is a skill: show up, do the set, log the score, repeat.",
    "Weak chapters are not warnings, they are the shortest route to marks.",
    "Teach the concept out loud once and you keep it a week longer.",
    "Marks live in the exceptions. Read the lines you find boring.",
    "Today's ten questions are next month's confidence.",
)

#: Lines used when the student has evidence of progress to celebrate.
_STREAK_LINES: tuple[str, ...] = (
    "You showed up yesterday too. That habit is worth more than any one score.",
    "Momentum is on your side - keep the streak alive with one more set.",
    "A streak is just proof that revision is working. Add today to it.",
)


@dataclass(frozen=True, slots=True)
class Motivation:
    """One dashboard greeting line with the context it was chosen from."""

    line: str
    day_of_year: int
    kind: str = "daily"

    def as_dict(self) -> dict[str, Any]:
        return {"line": self.line, "day_of_year": self.day_of_year, "kind": self.kind}


def daily_motivation(
    *,
    on_date: date | None = None,
    streak_days: int = 0,
    questions_answered: int = 0,
    extra_lines: Sequence[str] | None = None,
) -> Motivation:
    """Return the motivation line for ``on_date`` (today by default).

    A live study streak unlocks a small pool of streak-specific lines; the
    index into the pool still moves with the day so the line is stable.
    """
    today = on_date or date.today()
    day_index = today.timetuple().tm_yday
    pool: tuple[str, ...] = tuple(extra_lines or ()) or MOTIVATION_LINES

    if streak_days >= 2 and day_index % 2 == 0:
        line = _STREAK_LINES[day_index % len(_STREAK_LINES)]
        kind = "streak"
    elif questions_answered >= 20 and day_index % 5 == 0:
        line = (
            f"You have logged {questions_answered} practice questions so far. "
            "Analyse the wrong ones before adding more."
        )
        kind = "stats"
    else:
        line = pool[day_index % len(pool)]
        kind = "daily"
    return Motivation(line=line, day_of_year=day_index, kind=kind)


def days_until(target: date, *, today: date | None = None) -> int | None:
    """Whole days left to ``target`` (``None`` once it has passed)."""
    reference = today or date.today()
    delta = (target - reference).days
    return delta if delta >= 0 else None


def week_of_day(on_date: date | None = None) -> str:
    """Friendly ``Mon-Wed`` style label used by the dashboard header."""
    reference = on_date or date.today()
    monday = reference - timedelta(days=reference.weekday())
    return f"week of {monday.strftime('%d %b')}"
