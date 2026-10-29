# RENAME_REPORT.md — "NEET AI Tutor" → "NEETra AI"

**Date:** 2026-10-26
**Scope:** product / project naming only. No Python package, module, directory,
environment-variable, or file-name was renamed. No behaviour changed.

---

## 1. Two name layers found in the codebase

The repo carried **two distinct brands**, and the rename treats them differently:

| Layer | Old value | Where it lived | Action |
|---|---|---|---|
| Project / technical name | `NEET AI Tutor` | ingest banner, requirements header, package docstrings | **Renamed** to `NEETra AI` |
| Product name (`Settings.app_name`) | `Neet Genie` | Gradio page title, header, startup log, dashboard heading, welcome/health payloads | **Renamed** to `NEETra AI` (overridable via `NEET_APP_NAME`) |
| Coach persona | `Neet Genie` | `prompts/persona.txt`, `prompts/answer_system.txt`, prompt loader spec, agent docstrings, contract tests | **Preserved** — it is the in-product coach character, not the product name |

Because every UI title (Gradio page title via `gr.Blocks(title=settings.app_name)`,
the `# {app_name}` header, the dashboard heading, the welcome message and the
`/health` payload) already reads from `Settings.app_name`, one config default
change renames the entire UI-title surface at once.

## 2. Renamed surfaces (before → after)

| File | Line | Before | After |
|---|---|---|---|
| `config.py` | 1 | docstring "…NEET AI Tutor…" | "Central configuration for NEETra AI, the AI NEET Coach." |
| `config.py` | 52 | `app_name: str = "Neet Genie"` | `app_name: str = "NEETra AI"` (with comment: persona stays "Neet Genie", override via `NEET_APP_NAME`) |
| `app.py` | 1 | module docstring "NEET AI Tutor…" | "NEETra AI - Gradio application for the NEETra AI learning platform." |
| `app.py` | 642 | Coach tab heading "Neet Genie" | "## NEETra AI - the Neet Genie coach persona, format and answer modes" |
| `app.py` | 654 | architecture bullet "Neet Genie persona…" | "the Neet Genie coach persona + grounding rules…" (persona wording clarified) |
| `app.py` | ~692 | CLI description "Run NEET AI Tutor…" | "Run NEETra AI - the AI NEET mentor platform…" |
| `rag/ingest.py` | 80 | banner "NEET AI Tutor - ingestion complete" | "NEETra AI - ingestion complete" |
| `rag/ingest.py` | 196 | "Build the NEET AI Tutor FAISS index from PDFs." | "Build the NEETra AI FAISS index from PDFs." |
| `requirements.txt` | 2 | "# NEET AI Tutor - runtime dependencies" | "# NEETra AI - runtime dependencies" |
| `utils/logging_utils.py` | 1 | "Logging setup for the NEET AI Tutor." | "Logging setup for NEETra AI." |
| `utils/__init__.py` | 1 | "…helpers used across the NEET AI Tutor." | "…helpers used across NEETra AI." |
| `learning/__init__.py` | 1 | "NEET Genie platform services." | "NEETra AI platform services." |
| `learning/service.py` | 103 | "…services of the NEET Genie learning platform." | "…services of the NEETra AI learning platform." |
| `ui/__init__.py` | 1 | "NEET Genie UI layer - one module per product tab." | "NEETra AI UI layer - …" |
| `ui/components.py` | 1 | "…shared by every NEET Genie module tab." | "…shared by every NEETra AI module tab." |
| `README.md` | 1 | "# Neet Genie - Your Personal AI NEET Mentor" | "# NEETra AI - Your Personal AI NEET Mentor" |
| `README.md` | 352 | `NEET_APP_NAME` default shown as `Neet Genie` | `NEETra AI`, with note that the coach persona is still "Neet Genie" |
| `.env.example` | 2 | "# Neet Genie - … environment template" | "# NEETra AI - … environment template" |
| `.env.example` | 7 | `# NEET_APP_NAME=Neet Genie` | `# NEET_APP_NAME=NEETra AI` |
| `REVIEW.md` | 1 | "# Production-Readiness Review — NEET AI Tutor / Neet Genie" | "# Production-Readiness Review — NEETra AI (formerly NEET AI Tutor / Neet Genie product brand)" |

## 3. Progress / report text

The student-facing progress report (`ui/progress.py::progress_report_md`), the
dashboard heading (`ui/components.py`), and the welcome payload
(`learning/service.py`) all derive their titles from `settings.app_name`, so
they now render **"NEETra AI"** automatically with no per-string edits. The
coach's own voice inside generated answers still says "Neet Genie" (persona).

## 4. Example screenshot references

**None exist.** A full-repo search for image/screenshot references
(`![`, `.png`, `.jpg`, `screenshot`, `docs/`) found zero matches — the README
embeds no images and there is no `docs/` folder, so there was nothing to update
on this point.

## 5. Deliberately preserved (backward compatibility)

| Surface | Why untouched |
|---|---|
| `neet_ai_tutor/` directory and every package (`app.py`, `ui/`, `learning/`, `agents/`, `rag/`, `llm/`, `prompts/`, `utils/`, `scripts/`, `tests/`) | Instruction: no package renames unless necessary — it isn't |
| Class/function names: `GeniePlatform`, `GenieGenerator`, `build_ask_genie_tab`, etc. | Code identifiers, not user-facing brand |
| "Ask Genie" tab label and `ask-genie` element id | User-visible coach feature name; asserted by `scripts/offline_check.py:382` and panel tests |
| Coach persona "Neet Genie" in `prompts/persona.txt`, `prompts/answer_system.txt`, `prompts/loader.py:84`, agent docstrings, and the `test_coach_pipeline.py` / `test_answer_modes.py` persona-contract tests | Persona is a character inside the product; renaming it would change model output and break contract tests |
| `NEET_` environment-variable prefix (`NEET_APP_NAME`, `NEET_LLM_MODEL`, …) | Existing `.env` files keep working; `NEET_APP_NAME` still overrides the display name (verified: setting it to "Legacy Name" wins over the new default) |
| Log file name `logs/neet_tutor.log` and rotation path (`utils/logging_utils.py`) | Existing log history and any external log shippers keep working |
| `vectordb/neet_index` index path and `progress/` data dir | Re-pointing them would orphan existing indexes and study histories |
| Historical lines in `logs/neet_tutor.log` containing "NEET AI Tutor" | Logs are append-only history and must not be rewritten |
| `app_version` `1.1.0` | A rename is not a version bump; bump separately if desired |

## 6. Verification

All run on 2026-10-26 after the rename:

| Check | Result |
|---|---|
| `python -m pytest -q` | **181 passed**, 1 warning (pre-existing langchain-community deprecation) — includes `test_platform_panels.py`'s `title == settings.app_name` assertion, which now verifies "NEETra AI" |
| `python scripts/offline_check.py` | **All offline checks passed** (exit 0), including the "Ask Genie" module assertion |
| `python -m compileall` on all packages | exit 0 |
| `python app.py --check` | exit 1 solely from the pre-existing `gradio: MISSING` in this venv; config/dependency report otherwise unchanged |
| `Settings().app_name` | `'NEETra AI'` by default; `NEET_APP_NAME` override confirmed working |
| Repo-wide grep for `NEET AI Tutor` | only historical log lines + the REVIEW.md "formerly" note remain |

## 7. What operators should know

- Users who want the old title back need **no code change**: set
  `NEET_APP_NAME=Neet Genie` in `.env`.
- Any saved `.env` without `NEET_APP_NAME` silently moves from "Neet Genie" to
  "NEETra AI" in the UI title — intended, and reversible as above.
- The ingestion banner (`python -m rag.ingest`), `--help` text for `app.py` and
  `rag.ingest`, and the README/REVIEW headers now read "NEETra AI".
