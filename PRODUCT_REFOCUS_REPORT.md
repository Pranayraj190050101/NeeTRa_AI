# PRODUCT_REFOCUS_REPORT.md — NEETra AI: from learning platform to NEET preparation assistant

**Date:** 2026-10-28
**Change class:** product positioning + one opt-out feature flag.
**Behavioural contract:** no code deleted, no package/module/path renamed, every
existing test kept (navigation-order assertions updated to the new tab order).

---

## 1. Product rationale

NEETra AI shipped as a "modular NEET learning platform" whose navigation opened
on a **Dashboard** and treated six modules as equals, with the chat (**Ask
Genie**) sitting fourth. That framing undersold the thing students actually use
the product for: getting unstuck on a specific doubt, drilling MCQs, and fixing
the topics that cost them marks.

The refocus restates the product as a **NEET preparation assistant** and puts
the highest-value behaviour — grounded doubt solving — first:

> **NEETra AI**
> **Your Personal NEET Doubt Solver,**
> **Practice Partner and Rank Improver.**

Three concrete moves implement this:

1. **Ask Genie becomes the primary module** — it is now the first tab in the
   navigation instead of the fourth.
2. **Learn is demoted to an optional module** behind the
   `NEETRA_ENABLE_LEARN` feature flag, because auto-generated chapter study
   pages are the least-used surface and the one most likely to be replaced by a
   future content pipeline. It defaults to **on** so today's users see no
   change; operators set it `false` to hide it.
3. **All landing / dashboard copy** now leads with the five pillars the product
   actually promises: doubt solving, practice tests, mock exams, progress
   analytics, revision planning.

Nothing was removed: `learning/learn.py`, `ui/learn.py` and their tests are
untouched, and the `LearnService` is still instantiated by `GeniePlatform` even
when the tab is hidden, so re-enabling is a pure config change.

## 2. Feature flag

| Flag | Default | Effect when `false` |
|---|---|---|
| `NEETRA_ENABLE_LEARN` | `true` | Learn tab is not rendered; its widgets are never built and `learn_ui.wire()` is skipped. The service (`platform.learn`) still exists. |

* Field: `Settings.learn_enabled` (`config.py`), declared with
  `validation_alias="NEETRA_ENABLE_LEARN"` so the flag deliberately drops the
  `NEET_` prefix (matching the requested name). Verified end to end: unset →
  `True`, `NEETRA_ENABLE_LEARN=false` → `False`, `=true` → `True`.
* The flag also appears in `Settings.summary()` / `app.py --check` output.
* `.env.example` documents it in the "Platform modules" block.

## 3. New module hierarchy

**Primary**

- **Ask Genie** — the coach chat: type any doubt, get a retrieval-grounded
  answer with chapter/page citations, four answer modes, token streaming.

**Supporting (always on)**

- **Dashboard** — welcome, daily motivation, live stats, subject cards, four quick actions.
- **Practice** — Subject / Chapter / Topic MCQ tests with NEET marking and weak-topic detection.
- **Progress** — JSON study history, accuracy, per-subject/chapter performance, two-week series.
- **Revision Hub** — Flash Notes, Formula Sheet, High-Yield Points, Last-Minute Notes, weak-area recap.

**Optional (feature-flagged, on by default)**

- **Learn** — four-section chapter study pages, hidden when `NEETRA_ENABLE_LEARN=false`.

**Diagnostics (unchanged, after the modules)** — Knowledge base · Coach ·
Architecture.

Rendered navigation order with the flag **on**:
`Ask Genie · Dashboard · Practice · Learn · Progress · Revision Hub · (Knowledge base · Coach · Architecture)`.
With the flag **off**: `Ask Genie · Dashboard · Practice · Progress · Revision Hub · (diagnostics)`.

> `ui.components.MODULE_TABS` is kept as the stable id registry (it drives
> `TabNav` jump ids and is asserted by `scripts/offline_check.py`); the tab
> *build order* in `app.py` is what changed, so the navigation contract and the
> offline check remain valid.

## 4. User journey

1. A student opens the app and lands on **Ask Genie** (first tab). They paste the
   exact doubt that blocked them; Genie answers from *their* NCERT pages with
   citations.
2. Curious about the wider picture, they glance at the **Dashboard**: today's
   motivation, live accuracy/streak, and one-tap shortcuts — *Ask Doubt*,
   *Practice MCQs*, *Quick Revision*, *Start Mock Test* — each pre-configures and
   jumps to the right tab.
3. To convert the cleared doubt into retention they take a **Practice** test
   (chapter now, subject-level **mock exam** later) and get Score / Accuracy /
   Weak Topics with per-question explanations.
4. **Progress** shows what the tests revealed: ranked weak areas and a two-week
   trend.
5. **Revision Hub** turns those weak areas into flash notes, formula sheets and a
   last-minute recap — all through the same grounding pipeline.
6. (Optional) with `NEETRA_ENABLE_LEARN=true` they can still build full chapter
   study pages; with it `false` that whole path is simply absent from navigation.

The loop is **Doubt → Practice → Diagnose → Revise**, with the coach chat as the
entry point instead of a browsable library.

## 5. Copy changes (user-facing)

| Surface | File | Now reads |
|---|---|---|
| App landing sub-headline | `app.py` | "Doubt solving … practice tests and full mock exams … progress analytics … revision planning — your personal NEET preparation assistant…" |
| Dashboard welcome line | `learning/service.py` (`role`) | "Your Personal NEET Doubt Solver, Practice Partner and Rank Improver" |
| Dashboard heading | `ui/components.py` (`dashboard_heading`) | "🧞 {app_name} — your personal NEET doubt solver, practice partner and rank improver" |
| Dashboard intro | `ui/dashboard.py` (`build`) | "…open **Ask Genie** and type the doubt; practice, mock tests, analytics and revision sit around the chat." |
| "What this build can do" facts | `ui/dashboard.py` (`dashboard_markdown`) | Ask Genie row listed first; **Learn row only when the flag is on** |
| CLI `--help` description | `app.py` | "…the NEET preparation assistant (Ask Genie doubt solving, Dashboard, Practice, Progress, Revision Hub; optional Learn module behind NEETRA_ENABLE_LEARN)." |
| Module docstring | `app.py` | Ask Genie named as the primary module; Learn described as flag-gated |
| README hero + module section | `README.md` | New three-line positioning; "The assistant: Ask Genie first" table reordered, Learn marked *(optional)* |

## 6. Backward compatibility

- **No deletions or renames** of packages, modules, directories, classes
  (`GeniePlatform`, `LearnService`, …), env var prefix (`NEET_` for everything
  except the explicitly-requested `NEETRA_ENABLE_LEARN`), log file
  (`neet_tutor.log`), or the `vectordb/neet_index` / `progress/` data paths.
- **Default behaviour is unchanged**: `learn_enabled=True`, so a clean
  environment renders the same six modules (only reordered) and the same
  dashboard facts table.
- **Re-enabling Learn is config-only**: `NEETRA_ENABLE_LEARN=true` (the default)
  brings back the tab and wiring; the underlying service was never disabled.
- **`NEET_PLATFORM_ENABLED=false`** still yields the original single-chat app;
  its test (`test_platform_disabled_serves_the_original_chat_only`) passes
  unchanged because the non-modular branch keeps Ask Genie first.
- **Coach persona** ("Neet Genie"), the **"Ask Genie"** label/`ask-genie` id, and
  the `MODULE_TABS` id contract are preserved.

## 7. Tests

- Added `test_learn_flag_off_hides_the_tab_and_skips_its_widgets` — with the flag
  off the navigation is the five modules + diagnostics, no `learn` tab id.
- Added `test_learn_flag_off_still_builds_the_learn_service` — proves the service
  survives the hidden UI (backward-compatibility guard).
- Updated the two navigation-order assertions to the new Ask-Genie-first order
  (`test_navigation_leads_with_ask_genie_and_keeps_learn`, and the rebuild test).
- **Full suite: `183 passed`, 1 warning** (pre-existing langchain-community
  deprecation). `scripts/offline_check.py`: **all passed** (exit 0).
  `compileall`: exit 0. `app.py --check`: exit 1 solely from the pre-existing
  `gradio: MISSING`; `learn_enabled` now appears in its config report.

## 8. Future roadmap

1. **Content pipeline for Learn** — decide whether Learn pages stay generated,
   move to curated NCERT summaries, or are retired; until then it remains
   feature-flagged so the default can flip to `false` without code changes.
2. **Full-length mock exam engine** — a timed, multi-subject, full-syllabus
   paper (the current "mock" is a single-subject test); surface it as its own
   action on the Dashboard.
3. **Adaptive revision planner** — turn Progress weak-area analytics into a
   dated revision schedule (spaced repetition) inside the Revision Hub.
4. **Doubt → test loop** — one click from an Ask Genie answer to a targeted
   mini-test on exactly the cited concepts.
5. **Auth + `share` hardening** — the leading open P2 from `REVIEW.md`
   (`share=True` without auth); pair with the existing `REVIEW.md` remediation.
6. **Capability hooks** — voice input and photo question solving are already
   stubbed in `learning/capabilities.py`; wire a real transcriber / OCR solver.

---

### Files touched

`config.py` · `app.py` · `learning/service.py` · `ui/components.py` ·
`ui/dashboard.py` · `README.md` · `.env.example` ·
`tests/test_platform_panels.py` · (new) `PRODUCT_REFOCUS_REPORT.md`.
