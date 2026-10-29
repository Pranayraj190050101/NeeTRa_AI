# Production-Readiness Review — NEETra AI (formerly NEET AI Tutor / Neet Genie product brand)

**Scope:** the whole repository (`app.py`, `ui/`, `learning/`, `agents/`, `rag/`, `llm/`,
`prompts/`, `utils/`, `config.py`, `scripts/`, `tests/`) — roughly 10k lines of first-party
Python across 40 modules, covered by 175 tests.

**Method:** static analysis only. No behaviour was changed and no feature was added.
Two throwaway `ast`-based analysers were used to build the unused-symbol, exception and
import-graph inventories (the project venv has no `ruff`, `pyflakes`, `mypy` or `pytest-cov`,
so coverage claims are name-based, not line-based, and are marked as such).

**Verdict:** the codebase is well above the average for a single-developer local app. Layer
boundaries are real and enforced by discipline, the failure model is genuinely
"never show a traceback" in the chat path, and persistence writes are atomic. It is **not**
production-ready in the multi-user sense. All three **P1** remediation items were fixed on
2026-10-26 (status column below; the share-without-auth row is the leading P2, still open):

| # | Severity | Finding | Status |
| --- | --- | --- | --- |
| 1 | **P1** | The Learn tab can raise straight into Gradio (`learning/learn.py:95`, `:119` use the unguarded `execute()`), and the app is launched with `show_error=True`. | ✅ **Resolved** - Learn now uses the guarded `generate()` path like `revision.py`; regression test `test_learn_pages_never_raise_at_the_ui`. |
| 2 | **P1** | `ProgressStore.load()` validates JSON syntax but not document *shape*; a syntactically valid but wrong-typed file escapes as `AttributeError` / `ValueError` from every analytics method. | ✅ **Resolved** - `_repair_document()` coerces the document on read; regression tests `test_mistyped_*`. |
| 3 | **P2** | `demo.launch()` has no `auth=` while `--share` / `NEET_GRADIO_SHARE` / `--host 0.0.0.0` can expose the app (and the raw progress JSON) publicly. | ⏳ Open (P2). |

Everything else is consolidation, hardening and test debt.

---

## 1. Code Quality Audit

### 1.1 Unused imports

| Location | Symbol | Note |
| --- | --- | --- |
| `learning/practice.py:24` | `Grounding` | imported from `learning.generation`, never referenced; `grounding` local variables are inferred. |
| `ui/practice.py:18` | `notices_md` | imported from `ui.components`, never used in this module (the Practice tab renders `score_md` instead). |

Both are zero-risk deletions.

### 1.2 Dead code (definitions with no consumer anywhere in the repo)

| Location | Symbol | Evidence |
| --- | --- | --- |
| `utils/text_utils.py:38` | `CHARS_PER_TOKEN = 4.0` | only occurrence in the tree; the token estimator uses its own inline constant. |
| `learning/capabilities.py:239` | `DEFAULT_ANSWER_MODE` | single occurrence (its own definition). The voice/photo hooks never consult it. |
| `learning/progress.py:94` | `_today_key()` | single occurrence. `record_dict()` re-derives the day with `str(entry["created_at"])[:10]` instead (`:193`) — the two are **not** equivalent (local date vs UTC date), so the leftover helper is also a trap. |
| `learning/catalog.py:293` | `chapter_labels()` | single occurrence; every UI module uses `catalog.list()` + `.name`. |
| `learning/motivation.py:85`, `:92` | `days_until()`, `week_of_day()` | single occurrence each; the dashboard only uses `daily_motivation()`. |
| `config.py:207` | `reload_settings()` | single occurrence; nothing re-reads settings at runtime. |
| `ui/components.py:214` | `doc_md()` | exported in `__all__` (`:36`) and defined, but no UI module imports it — `ui/learn.page_md` and `ui/revision.sheet_md` each re-implement it (§1.4). |
| `ui/components.py:242` | `scope_hint()` | single occurrence; the Practice tab writes its own scope line. |
| `prompts/loader.py:52` | `PLATFORM_PROMPT_PAIRS` | in `__all__`, zero readers — the only export with no consumer. Its `task_id` keys still match what the services send (`learn_chapter` at `learning/learn.py:85`, `mcq_generation` at `learning/practice.py:166`, `revision_notes` at `learning/revision.py:149`), but nothing enforces that: the services hard-code the same names at `learn.py:86-87`, `practice.py:167-168`, `revision.py:150-151`, so the table and reality can drift silently. The six `*_PROMPT` constants it is built from (`loader.py:44-49`) are **not** dead — `PROMPT_SPECS` uses them at `:129-159`. |
| `prompts/loader.py:168` | `MODE_PROMPT_NAMES` | in `__all__`, zero readers. |

Note: `doc_md`, `scope_hint`, `_today_key`, `days_until`, `week_of_day`, `chapter_labels`,
`reload_settings`, `DEFAULT_ANSWER_MODE`, `CHARS_PER_TOKEN` and `MODE_PROMPT_NAMES` are
**public-looking API with no caller**. Removing them is safe today; keeping them means the
next reader assumes they are wired up. `PLATFORM_PROMPT_PAIRS` is the borderline case: the
map is unread while the strings it contains are duplicated at three call sites.

### 1.3 Duplicated logic

| Duplication | Locations | Risk |
| --- | --- | --- |
| **Two section-contract engines.** Heading detection + "fill the section the model skipped". | `utils/answer_format.py` (`_MD_HEADING_RE`, `_heading_title`, `_canonicalise`, `apply_headings`, `missing_sections`, `enforce_format`, `fallback_body`) vs `learning/sections.py` (`_plain_title`, `_core`, `_matches`, `stamp_sections`). | High. They use **different matching rules** (60-char heading cap + alias table vs. normalised substring containment), so identical model output can be repaired in the Ask Genie tab and left broken in the Learn tab. One engine, two section vocabularies. |
| **Three catalog → `(label, value)` dropdown mappers.** | `ui/learn.py:71 chapter_choices`, `ui/revision.py:63 target_choices`, `ui/practice.py:222 _target_choices`. | Medium. Same job, three filter orders. `ui/revision.target_choices`' docstring promises "the chapters of one subject, **then its raw source names**" but returns chapters only (`:64-68`) — stale contract. |
| **Two document renderers.** | `ui/components.py:214 doc_md` (dead) vs `ui/learn.py:105 page_md` + `ui/revision.py:71 sheet_md`. | Medium: three near-identical `ok`/title/notices assemblies. |
| **Test result rendered twice.** | `ui/practice.py:285` returns `result.report_markdown(reveal=True)` **and** `score_md(result)` into two different Markdown components. | Low but visible to students: the same score/accuracy numbers appear twice on one tab. |
| **Subject emoji maps.** | `learning/service.py:416` (3 keys) and `ui/components.py:254` (4 keys, incl. `general`); `app.py:67 SUBJECT_CHOICES` is a third source of subject vocabulary. | Medium: adding a subject means editing three places, and only one of them raises. |
| **Weak-area aggregation.** | `learning/assessment.py:254 GradedTest.weak_topics`, `learning/practice.py:273 PracticeService.weak_topics`, `learning/progress.py:340 ProgressStore.weak_areas`. | Medium — see §7.4 for the threshold/denominator inconsistency this creates. |

### 1.4 Single-responsibility hotspots

- **`utils/constants.py` (682 lines).** Six enums, twelve parallel `_X_LABELS` / `_X_EMOJI` /
  `_X_DESCRIPTIONS` dicts, keyword banks and the intent bridge in one module. Every enum
  property indexes a module dict with `dict[self]` (e.g. `_MODE_INTENTS[self]` at `:181`,
  `:252-257`), so **adding an enum member without adding the dict entry is a `KeyError` at
  import-adjacent runtime**, with no compiler or test guard. Splitting into
  `constants/enums.py`, `constants/vocabulary.py`, `constants/keywords.py` and making the
  per-enum maps fields of the enum (or deriving them with a `.get()` + explicit default) is
  mechanical and low-risk.
- **`app.py` (798 lines).** CLI parsing, config check, logging setup, the streaming chat
  responder, six panel renderers and the tab shell. `_dependency_report`,
  `run_config_check` and the panels are pure presentation and belong next to `ui/`.
- **`learning/generation.py`.** Owns grounding, prompt assembly and the guarded
  generate/failure-message contract — reasonable — but also instantiates
  `RetrievalAgent` purely to reuse its context formatter. That is a service layer reaching
  into the agent layer for a string helper; the formatter should be a free function in
  `rag/` (see §2.4).
- **`learning/capabilities.py:225 describe_capabilities()`.** Hand-builds a Markdown table
  inside a domain module. It is the only place in `learning/` that emits presentation
  markup; move it to `ui/components.py` alongside `table()`.

### 1.5 Safe refactoring opportunities (ordered by risk/benefit)

1. Delete §1.1 + §1.2 outright. Zero behavioural surface, removes ~15 misleading symbols.
2. Collapse the three dropdown mappers into one `ui/catalog_view.py::choices_for(catalog, scope, subject)`; have `learn`/`revision` pass the scope they need. Fix the stale `target_choices` docstring at the same time.
3. Keep `doc_md` and delete `page_md`/`sheet_md` (or the reverse) so exactly one renderer exists; `ui/learn.py:119` already calls `notices_md` which `doc_md` also calls.
4. Merge `stamp_sections` into `enforce_format` behind one "section contract" object that takes a title list instead of an `AnswerMode`. Both callers keep their current signatures via thin wrappers.
5. Make `weak_topics`/`weak_areas` accept one shared `WeakArea` type and one threshold source (see §7.4).
6. Merge the duplicated score rendering on the Practice tab into a single report.

---

## 2. Architecture Review

### 2.1 Layer separation — good, and verified

| Boundary | Evidence | Verdict |
| --- | --- | --- |
| `learning/` must not know about Gradio | the only `gr.`-shaped text in `learning/*.py` is inside two docstrings (`capabilities.py:137`, `:146`). | **Clean.** The service tier is importable headless, which is what makes `scripts/offline_check.py` and the 23 service tests possible. |
| `agents/` must not know about the UI | no `from ui` / `import ui` in `agents/*.py`. | **Clean.** |
| `ui/` must not reach past the service facade | one exception: `ui/components.py:195` does an inline `from agents.tutor import citations_to_markdown` inside a `try/except Exception`. | **Acceptable but inconsistent** — the presentation layer depends on the agent layer for one formatter. Either re-implement the citation table locally (it already has a fallback `table(...)` at `:199`) or inject the formatter. |
| `rag/`, `llm/`, `prompts/`, `utils/` | no upward imports found. | **Clean.** |

### 2.2 Circular dependencies

Two **package-level** cycles exist; **no module-level cycle** exists.

- `agents ⇄ learning`: `agents/tutor.py` imports `learning.service` *lazily inside*
  `platform()` / `platform_status()`, while `learning/generation.py` and `learning/service.py`
  import `agents.*` at module level (`agents.state`, `agents.retrieval_agent`). The lazy
  import is deliberate and correct, but it is invisible to a reader and one careless
  top-level import in `agents/tutor.py` would break the app at import time.
- `config ⇄ utils`: `config.py` imports from `utils.*`; `utils/*` reaches back into
  `config.Settings` for type hints. Currently safe because every such import is
  `from __future__ import annotations`-guarded or inside a function.

**Recommendation:** keep the cycle-breaking imports where they are, but make them
*self-documenting* — the `agents/tutor.py` lazy import has no comment explaining why it must
stay inside the method. This is the single most likely future source of an
`ImportError`-at-startup regression, and it is not covered by any test (importing `app.py`
is only exercised through `--check`, which is itself untested).

### 2.3 Interfaces / protocols — mostly respected

- `learning/capabilities.py` defines `runtime_checkable` `VoiceTranscriber` and
  `ImageSolver` Protocols and funnels both through `GeniePlatform.ask()`. Extension points
  are real: `CapabilityRegistry.register()` flips state to `ACTIVE`, `require()` raises a
  typed `CapabilityNotAvailable`, and `as_dict()` feeds the diagnostics panel.
- Gaps: `DEFAULT_ANSWER_MODE` (`:239`) is the documented fallback for a voice/photo turn and
  is never read; `register()` rebuilds the whole `Capability` rather than storing availability
  alongside the backend, so `register()` is not idempotent w.r.t. `roadmap` text; and there
  is no Protocol for the LLM seam at the *platform* level — `GenieGenerator` accepts
  `LLMClient | None` and raises `RuntimeError("No language model configured…")` from
  `.llm`, which is exactly the exception that reaches the Learn tab (§3.1).

### 2.4 Scalability bottlenecks

| Bottleneck | Location | Consequence |
| --- | --- | --- |
| `demo.queue(default_concurrency_limit=1)` | `app.py:683` | One generation at a time, by design (a 7B model on one GPU). Correct locally; fatal for anything multi-student. |
| `HF` generation serialised behind `self._generate_lock` | `llm/huggingface_client.py:195` | Same story at the client level. |
| Whole history re-read and re-aggregated on every render | `learning/progress.py:234-418`, `ui/progress.py` (4 call sites per refresh) | O(all tests ever taken) per tab refresh; grows linearly forever (§4.4). |
| `docstore` materialisation for stats | `rag/vector_store.py:349` | O(index) for a summary; see §8.2. |
| Single JSON document per profile | `learning/progress.py:124-128` | Fine at `SCHEMA_VERSION = 1` scale; the write path is already the seam a DB will replace, which is the right design. |
| No query-level retrieval cache | `learning/generation.py:190-201` | Repeated identical chapter/topic lookups re-embed and re-search (§8.1). |

---

## 3. Error Handling Review

### 3.1 Where exceptions can still escape to the UI (P1 - ✅ resolved 2026-10-26)

> **Resolution:** `learning/learn.py` now calls `GenieGenerator.generate()` (the same
> never-raises path `revision.py` uses); a model crash becomes an `ok=False`
> `GeneratedDoc` rendered by the tab's existing failure branch. Regression test:
> `tests/test_platform_services.py::test_learn_pages_never_raise_at_the_ui` (parametrised
> over `RuntimeError` and `LLMError`). The text below is kept as the original finding.

The app is launched with **`show_error=True` (`app.py:791`)**, so any exception that reaches a
Gradio handler is rendered to the student as an error banner.

The chat path is properly defended — `app.py:342` wraps the worker thread,
`agents/graph.py:213` catches graph-level failure, `agents/tutor.py:124`/`:138` convert
startup and runtime failures into `TutorResult.failure(...)`, and each node has its own
`except AgentError` / `except Exception` at `agents/graph.py:273-286` that appends
`"The {node} step failed."`.

**The Learn tab is the hole:**

```python
# learning/learn.py:95  (chapter_page)   and :119  (topic_page)
raw = self.generator.execute(task)
```

`GenieGenerator.execute()` is the *raw* call. Two exceptions are reachable from it with no
local handler:

1. `GenieGenerator.llm` → `RuntimeError("No language model configured…")` when the LLM is
   unset or failed to load (`agents/tutor.py:80-84` deliberately leaves `self._llm = None`
   for a retry).
2. `HuggingFaceClient.generate` → `LLMError` (OOM, CUDA failure, tokenizer error —
   `llm/huggingface_client.py:200-207`).

Its two siblings already use the guarded path:

- `learning/revision.py:160` calls `self.generator.generate(...)`, which wraps `execute()` in
  `try/except Exception` and returns a `GeneratedDoc(ok=False, error=...)`
  (`learning/generation.py:281-292`).
- `learning/practice.py:177-192` wraps `execute()` itself and returns an empty `PracticeSet`
  carrying a notice.

So `Learn` is the only module tab that can put a Python exception class name in front of a
student. Fix: call `self.generator.generate(task, ...)` in `learn.py` (or add the same
try/except as `practice.py`), returning `GeneratedDoc.failure(...)`.

**Also on the escape list, lower priority:**

| Call path | Exposure |
| --- | --- |
| `ui/dashboard.py:245 handler(...)` → `platform.*` | Inherits whatever the underlying service does; safe today only because Learn is the sole unguarded one. Fixing §3.1 makes this airtight. |
| `app.py:635 kb_refresh.click(lambda: index_panel(...) + settings_panel(...))` | `index_panel`/`settings_panel` read `tutor` internals; a raise here lands in the banner. Cheap to wrap in the same "panel never raises" convention used by `learning/service.py:374-377`. |
| `ui/progress.py:221 _raw_md` → `json.dumps(store.load(force=True), ...)` | `default=str` makes it robust, but it depends on §4.2 shape validation. |

### 3.2 Broad exception handlers

43 `except Exception` / `except BaseException` handlers exist in first-party code. The
majority are **correct and commented** (defensive boundaries around optional dependencies,
e.g. `learning/catalog.py:170`, `learning/service.py:180`/`:376`, `ui/components.py:198`,
`rag/vector_store.py:201-255` which re-wrap into `VectorStoreError`). Three are worth
narrowing or annotating:

| Location | Comment |
| --- | --- |
| `rag/vector_store.py:310` | `except Exception` around FAISS search also swallows a genuine internal `TypeError`/`AttributeError` bug and reports it as "FAISS search failed". Add `except (VectorStoreError, OSError, RuntimeError)` and let programming errors surface to the caller's log. |
| `llm/huggingface_client.py:241` | `except BaseException` in the streaming worker is **justified** (a `KeyboardInterrupt`/`SystemExit` raised inside the generation thread must still be forwarded to the caller), and the `# noqa: BLE001` says so. Keep, but consider `except Exception` + `finally: end()` if you ever drop the noqa. |
| `learning/generation.py:202` | `except Exception` for "index present but embeddings package missing" is right, but it is the only place that can hide a missing-failed import behind a generic notice. `logger.exception` is present — acceptable. |

`# pragma: no cover` appears 37 times in first-party code, concentrated in the HF client and
the optional-dependency branches. That is honest labelling, not evasion, but it means the
live-LLM path has effectively no automated proof of life (§6.4).

### 3.3 Actionability of user-facing failures

Good: the OOM message names three concrete remedies (`huggingface_client.py:202-206`); the
no-index notice tells the student the document is ungrounded rather than failing
(`generation.py:184-188`); capability hooks answer "not in this build" with the extension
point name.

Weak: the generic generation failure embeds the exception identity and a log path in the
Markdown the student sees (`learning/generation.py:288-292`, and the same pattern in
`app.py:375-379` for chat, and `learning/practice.py:188` —
`"The language model could not produce the test: {exc}"`). Showing
`RuntimeError: CUDA out of memory…` to a 17-year-old is not actionable and leaks environment
detail. **Recommendation:** keep the technical string in
the log and show a fixed two-line message plus a "copy this into the diagnostics tab" hint.
Relatedly, `assert self._tokenizer is not None` at `huggingface_client.py:172` is a
type-checker aid, not a guard: under `python -O` it vanishes and the next line fails with
`AttributeError` on `None`. Prefer an explicit `if ... is None: raise LLMError(...)`.

---

## 4. Persistence Review

### 4.1 What is right

- **Atomic write.** `save()` writes `.<pid>.tmp` then `os.replace()` (`progress.py:161-173`) —
  no torn documents on a crash mid-write.
- **Never-raises read.** `load()` handles `FileNotFoundError`, `json.JSONDecodeError` and
  `OSError`, logs, and falls back to a blank document (`:138-147`).
- **Schema marker + repair.** `SCHEMA_VERSION` and the `setdefault` ladder (`:150-156`) mean a
  hand-edited file missing `events` still loads.
- **Path safety.** The file name derives from the profile via `normalise_token(...)` with
  spaces → underscores (`:117`); a `path` with a `.json` suffix is honoured verbatim
  (`:126-128`). No user-supplied path reaches `open()` unmediated (§5.1).

### 4.2 Corruption handling is syntax-only, not shape-aware (P1 - ✅ resolved 2026-10-26)

> **Resolution:** `load()` now runs every parsed document through `_repair_document()`
> (`learning/progress.py`), which coerces `tests` to a list of well-shaped records
> (`_coerce_record`/`_coerce_topic`, numeric fields via `_as_int`/`_as_float`), `daily` to
> `{str: bucket}` and `events` to `{str: int}`, logging what was dropped. Every analytics
> method is now total. Regression tests: `test_mistyped_document_is_repaired_not_fatal`,
> `test_mistyped_records_survive_reading_and_weak_areas`. The text below is the original finding.

`load()` guarantees a `dict`; it does not guarantee that `tests` is a `list`, that entries are
`dict`s, or that `total`/`correct` are numbers. `setdefault("tests", [])` only repairs a
*missing* key — an existing `"tests": "abc"` passes straight through, and then:

| Method | Failure on a mistyped document |
| --- | --- |
| `stats()` `:238` | `AttributeError: 'str' object has no attribute 'get'` |
| `subject_performance()` / `chapter_performance()` `:312` | same |
| `weak_areas()` `:343-346` | `AttributeError` on `topic.get`, plus `int("x")` → `ValueError` |
| `record_dict()` `:191` | `AttributeError` (no `.append` on a str) — i.e. **submitting a test after corruption crashes** |
| `daily_series()` `:403` | `AttributeError` if `daily` is a list |

Every one of those reaches the UI banner because of `show_error=True`. A partially
hand-edited or half-truncated JSON file is a realistic scenario for a local app whose
progress file is documented in the README.

**Recommendation:** one `_coerce(document)` step inside `load()` — `tests` → list of dicts
(drop others with a warning), numeric fields → `int(..., or 0)` via a small
`_as_int(value)` helper, `daily` → dict of dict, `events` → dict of int. This is hardening,
not a feature: it changes no correct-document behaviour and makes every analytics method
total. `summarise()` (`:431-441`) already demonstrates the defensive pattern.

### 4.3 Corruption is silently destroyed (✅ resolved 2026-10-26)

> **Resolution:** an unparseable (or non-dict) progress file is now moved to
> `<profile>.corrupt-<utc>.json` by `ProgressStore._quarantine()` before a blank document is
> started; the copy is recorded on `store.last_quarantine`, and `ui/progress.py::
> progress_report_md` shows the student where it went. Read errors (`OSError`,
> `UnicodeDecodeError`) quarantine too; a missing or empty file does not. Regression tests:
> `test_unreadable_history_is_quarantined_not_overwritten`,
> `test_json_that_is_not_a_document_is_quarantined`.

On unreadable JSON, `load()` logs and returns a blank document (`:145-147`); the next
`save()` then overwrites the student's entire history with no backup. For an app where the
only record of a month of study is one JSON file, **quarantine before discarding**: rename the
bad file to `<profile>.corrupt-<utc>.json` and surface a notice ("your history could not be
read; it was saved to X") instead of only a log line.

### 4.4 Concurrency assumptions

`self._lock = threading.RLock()` (`:118`) makes the store safe **within one process**, and
`record_dict()` correctly holds it across read-modify-write (`:187-198`). The assumption that
does not hold: **one process**. Two `app.py` instances (or a `scripts/` run plus the UI)
sharing `progress/<profile>.json` will lost-update each other, because `load()` → `save()` is
not serialised at the OS level. Combined with the mtime-only cache stamp (`:175-179`), a
write from another process with the same mtime second is served from cache.

These are fine for a documented single-student local tool; they are **not** fine if the app is
ever served to a class (§5.2). Worth doing regardless: an exclusive lock file
(`msvcrt.locking` / `fcntl.flock`), or at minimum a `warnings.warn`/log line when
`_stat_stamp()` changes under a held cache, plus a README note that one profile = one
process. Two smaller items: `temp` is not cleaned up if `write_text`/`os.replace` raises
(`:169-170`) — wrap in `try/finally` with `unlink(missing_ok=True)`; and `clear()` (`:227`)
rewrites a blank document but is not undoable, so the UI reset button should confirm and,
ideally, back up first (same helper as §4.3).

### 4.5 Unbounded growth

`tests` grows forever and `as_dict()`/`stats()`/`subject_performance()`/
`chapter_performance()`/`weak_areas()` each do a **separate full pass** over it, and the
Progress tab calls four renderers per refresh (`ui/progress.py:163`, `:195`, `:246`, `:288`).
Recommendation (no behaviour change): cap `daily` to a rolling window, add an optional
`archive_before(iso)` maintenance method, and compute the aggregates in one pass in
`as_dict()`.

### 4.6 Non-atomic sibling: `write_index_metadata`

`rag/vector_store.py:399-410` uses `path.write_text(...)` where `ProgressStore.save()` uses
temp+replace. `read_index_metadata()` is guarded, so a torn sidecar degrades to "no catalog"
rather than crashing — but the failure is silent and the catalog is what the Learn dropdown
shows. Make it atomic for consistency (same 4-line pattern).

---

## 5. Security Review

### 5.1 File operations and traversal — clean

- Prompt loading is whitelist-driven: `PROMPT_SPECS` fixes the file name per prompt (`prompts/loader.py:61-99`), so a caller cannot name a file; `PromptLibrary.raw()` raises on an unknown name. `agents/tutor.py:77-84` catches load failure and leaves the LLM unset for a clean retry.
- The progress path is derived from a normalised profile token, not from a request field (§4.1).
- Index/data directories come from `Settings` (`config.py`), i.e. env-var or CLI, never from a browser-supplied value. `data/` layout is documented as operator-controlled.
- Uploads: ingestion is an out-of-band CLI (`python -m rag.ingest`), not a web endpoint, so no arbitrary-file-write surface exists in the app.

### 5.2 Exposure — the real finding (P2)

```python
# app.py:786-792
demo.launch(server_name=..., server_port=...,
            share=bool(args.share or settings.gradio_share),
            inbrowser=False, show_error=True)   # no auth=
```

`--share` / `NEET_GRADIO_SHARE=true` create a public gradio.app URL, and `--host` can bind
`0.0.0.0`. There is **no `auth=`, no rate limiting, and no profile concept at the transport
level** — every visitor shares the single `default` progress document, can submit tests into
it, can press the Progress reset (`ui/progress.py:280`), and can press *Show raw history*
(`:251` → `_raw_md` `:221-223`) which dumps the entire stored document — every test title, chapter and
score the student has ever recorded. (The absolute path of the file is shown separately, by
`as_dict()`/`progress_report_md`, not by this panel.)

This is out of scope for a local study tool and dangerous the moment it is shared.
Recommendation, cheapest first: (a) refuse to start with `share=True` unless
`NEET_GRADIO_AUTH` (or `--username/--password`) is set, with a clear message; (b) set
`show_error=False` and render failures as chat text (pairs with §3.3); (c) gate the raw-JSON
panel behind an explicit "diagnostics" env flag; (d) document that the deployment model is
one student per process.

### 5.3 Injection risk — low by construction, with one hygiene issue

Markdown is the only sink and there is no HTML/JS execution, no `eval`, no shell
(`rag/ingest.py` uses no subprocess with user input) and no SQL. The only injection-adjacent
surface is **prompt and retrieval text**, and it is bounded: `clean_text()` + `truncate(query, 400)`
before search (`generation.py:179`, `:194`), `truncate(topic, 120)` for chapter labels
(`learn.py:114`), and the section contract is appended after the persona so user text is
data, not instructions. Model output is post-processed (`huggingface_client.py:209`) and
repaired (§1.3), so a prompt-injected answer still has to fit the section shape.

**The hygiene issue is the reverse direction:** `esc()` (`ui/components.py`) is a
*markdown-table-cell* escaper (`|` → `\|`, newlines → spaces). It is being used as if it were
a sanitizer **on the data path**:

- `ui/learn.py:84-85` escapes the chapter and topic before handing them to
  `platform.study_chapter/study_topic`, so an apostrophe in a chapter name no longer matches
  the catalog (`_resolve` → `catalog.find`) and the student silently gets the
  "chapter not found" page.
- `ui/practice.py:119` sends `target=esc(target)` into `PracticeService.generate()`, where it
  is used for `_match_chapter` → `catalog.find()` and, via `label`, as the retrieval query and
  the prompt's `target` field (`learning/practice.py:148-175`). The search query therefore
  carries escaped pipes, and a chapter whose name contains `|` stops matching the catalog.

Escaping at the boundary is the right rule; the boundary is the render, not the request.
Recommendation: keep `esc()` for table cells only, and pass raw text to services (the service
layer already normalises via `clean_text`). This is a correctness bug and a security smell at
the same time.

### 5.4 Environment variables — acceptable

`config.py` uses Pydantic `Field` with bounds (`score_threshold` `ge/le`,
`weak_accuracy_threshold` `ge/le`, `:81`/`:124`), so a bad `NEET_*` value fails at startup
with a typed error rather than producing nonsense at runtime. `NEET_LLM_BACKEND=mock` is a
whitelist switch. `HF_TOKEN` is not read into a `Settings` field (it stays with
`huggingface_hub`), which is correct — no secret is ever stored in the progress document, the
index sidecar, or a log line. Nothing logs the environment wholesale. The one gap: no
`SecretStr` usage, so a future `NEET_*_API_KEY` field would be printable by
`settings_panel(...)`/`--check`; add `SecretStr` before introducing any such field.

---

## 6. Test Coverage Review

175 tests / 15 files, and they are well-written: fixtures over factories, behaviour-named
tests, a genuine acceptance test for the five-criteria student journey
(`tests/test_platform_services.py::test_a_student_can_learn_ask_test_see_scores_and_revise`),
and `tests/test_platform_panels.py` (485 lines) exercising the tab wiring against an injected
fake Gradio so the suite runs with Gradio uninstalled.

### 6.1 Untested entry points (highest value first)

| Target | Why it matters |
| --- | --- |
| `app.py::run_config_check`, `_dependency_report`, `_parse_args`, `main` | The operator's only self-diagnostic path. It is the thing that tells a user why the app exits 1. Zero tests. A subprocess test on `--check` with `NEET_LLM_BACKEND=mock` plus direct tests of `_dependency_report` would close it cheaply. |
| The five panel builders (`index_panel`, `settings_panel`, `platform_panel`, `coach_panel`, `architecture_panel`) | They run at every app start-up and read live tutor state; a `KeyError` here is a start-up crash. |
| `ProgressStore` **mistyped** documents (§4.2) | The exact scenario that currently crashes the Progress tab. |
| `ProgressStore.save()` I/O failure (read-only dir, locked file on Windows) | `os.replace` raising is untested; today it propagates into a submit handler. |
| `MCQuestion.from_dict` / record round-trip | The persistence contract between grading and storage. |
| `render_review`, `grade_set`, `PracticeService.weak_topics` | Grading edge paths (skipped marks, all-wrong, empty topics). |
| `VectorStoreManager.reload_index` / `unload` / `stats()` / `document_ids()` | Index lifecycle behind the Knowledge base tab and incremental ingest. |
| `PromptLibrary.path_for`, `clear_cache`, `validate()` failures | The prompt contract that `app.py:777` logs as an error but no test asserts. |
| `utils/text_utils` (the half not used by `CHARS_PER_TOKEN`'s module), `utils/device`, `apply_headings`, `fallback_body`, `FormatRepair` | Pure functions, trivially testable, currently reached only indirectly. |
| `ProgressStore.record_dict` / `summarise` | `summarise` has no production caller and no test — it should be one or the other. |

### 6.2 Edge cases not covered

- Empty `answers` list against a non-empty test; extra answers beyond `total`.
- A graded record with `total=0` (accuracy division guard exists but is unproven).
- `weak_areas()` when `topic["total"]` ≠ `correct + wrong` — the mixed denominators (§7.4) are
  precisely the un-tested case.
- A progress file whose `schema` is **newer** than `SCHEMA_VERSION` (forward compatibility: currently ignored silently).
- Catalog with two chapters of the same name in different subjects (dropdown collisions in the three mappers).
- Retrieval returning zero passages with `prefer_file` set (the 3× over-fetch re-ordering path, `generation.py:191`).
- Concurrent `record()` calls from two threads on one store (the `RLock` promise is asserted by design, not by test).
- Unicode/Devanagari textbook content through `clean_text` → chunker → FAISS metadata → markdown cell.

### 6.3 Highest-value additions (short list)

1. `test_progress_store_survives_a_mistyped_document` — `"tests": "abc"`, string records, `"total": "x"`, `daily` as a list; assert every analytics method returns and `record_dict` still works.
2. `test_progress_store_keeps_a_corrupt_file` — invalid JSON is quarantined, not overwritten.
3. `test_learn_tab_never_raises` — `GenieGenerator.llm` raising `RuntimeError` and the client raising `LLMError` both produce `GeneratedDoc(ok=False)` (the regression test for §3.1).
4. `test_check_command_reports_missing_dependency_without_crashing` (subprocess, `--check`, mock backend).
5. `test_save_failure_leaves_previous_document_intact` (patch `os.replace` to raise).
6. `test_weak_area_threshold_is_honoured_end_to_end` — set `NEET_WEAK_ACCURACY_THRESHOLD` and assert the Progress tab and `Revise my weak areas` agree.
7. `test_the_same_section_contract_is_repaired_the_same_way_everywhere` — one model output through `enforce_format` and `stamp_sections` (the regression test for §1.3).

### 6.4 Structural gap

The live LLM path (`llm/huggingface_client.py`, ~340 lines, 10 broad handlers, 37
`# pragma: no cover` markers across the tree) has no automated coverage that can run offline.
Recommendation without adding a feature: extract `_build_prompt` / `_postprocess` (already
pure) into tested units, and add one test that instantiates the client with a fake
`AutoModelForCausalLM` so the OOM→`LLMError` translation at `:200-207` is proven. Also: there
is no coverage measurement installed, so "untested branch" claims here are name-based. Adding
`pytest-cov` to the dev requirements is a one-line change that turns this section into
numbers.

---

## 7. Performance Review

### 7.1 Repeated vector lookups

- `GenieGenerator.ground()` over-fetches by design: `wanted = top_k * (3 if prefer_file else 1)` (`generation.py:191`) and `learning/learn.py:74` requests `top_k * 2` passages for a chapter page. Worst case a single Learn render asks FAISS for **6 × top_k** vectors, then discards all but the budgeted context. Correct output, ~3× the search cost needed for chapter scoping.
- No memoisation: opening the same chapter twice, or Learn → "ask about this chapter" from the dashboard, re-embeds an identical query. Retrieval is the dominant latency term on CPU.
- **Recommendation:** an LRU on `(query, subject, top_k)` → chunks (8–16 entries) inside `VectorStoreManager.search()`, and a per-request `Grounding` object reused across the tab's refresh path. Both are pure caching; no behaviour change.

### 7.2 Redundant document loading

- `VectorStoreManager.stats()` materialises the entire docstore (`rag/vector_store.py:349`, `list(store.docstore._dict.values())`) and `document_ids()` (`:372`) does it again. `ChapterCatalog.build()` (`learning/catalog.py:168`) calls `stats()`; the UI calls the catalog *and* the panel per refresh.
- Mitigated today by the catalog cache and the manager's single loaded store; the residual cost is the O(index) Python iteration per stats call.
- **Recommendation:** derive `IndexStats` from the index metadata sidecar (`read_index_metadata`) instead of the loaded store — it already contains per-subject counts — and keep the docstore walk as the fallback. That also removes the "load the model to count documents" coupling, which matters on a 4 GB GPU.

### 7.3 Avoidable recomputation

`ProgressStore.as_dict()` (`:419-428`) calls `stats()`, `subject_performance()`,
`chapter_performance()` and `weak_areas()` — four independent full passes over the same list.
`ui/progress.py` renders four panels per refresh, each calling `load()` (cache-hit, so no
re-parse, but each re-aggregates). For 500 stored tests that is ~2000 dict visits per click.
**Recommendation:** one `_aggregate(records)` pass producing the shared buckets, with the four
public methods as views over it.

### 7.4 Consistency defect found while measuring (P2)

The same "weak area" concept has three thresholds and two denominators:

| Site | Default threshold | Denominator |
| --- | --- | --- |
| `learning/assessment.py:254 GradedTest.weak_topics` | **0.7** | `stats["total"]` |
| `learning/practice.py:273 → ProgressStore.weak_areas` | 0.75 | `correct + wrong` (`progress.py:366`) while `WeakArea.questions` is `topic["total"]` (`:359`) |
| `learning/progress.py:340 weak_areas` | 0.75 | `correct + wrong` |
| `ui/progress.py:163/195/246/288` | `settings.weak_accuracy_threshold` | — |

`settings.weak_accuracy_threshold` (`config.py:124`, default 0.75) is honoured **only** by the
Progress tab. `PracticeService.weak_topics()` (used by `ui/practice.py:291` *revise weak* and
by the Revision Hub's "weakest topics" preview) silently uses the hard-coded 0.75, and the
per-test view uses 0.7. So the Progress tab and the Practice tab can disagree about what is
weak, and a topic's displayed `questions` count may not match the accuracy shown next to it
because the two use different denominators. **Recommendation:** thread `settings.weak_accuracy_threshold`
through `weak_topics()`/`weak_areas()` call sites, keep one denominator (`answered = correct + wrong`,
with `total` reported separately), and add test 6 from §6.3.

### 7.5 Model-tier note (not a code defect)

`default_concurrency_limit=1` and the `_generate_lock` mean throughput is one generation at a
time — that is the right choice for a local GPU, but it makes §7.1/§7.2 caching the highest
leverage change available, since every saved FAISS query is real wall-clock for the next
student action.

---

## 8. Remediation Plan

Nothing here adds a feature. Each step is behaviour-preserving except where it explicitly
fixes a defect.

### P1 — before the next release (all done 2026-10-26)

1. ✅ **Learn tab exception escape** — `learning/learn.py:95`, `:119`: use the guarded
   `generate()` path (or the `practice.py:177-192` pattern). Add regression test §6.3-3.
2. ✅ **ProgressStore shape validation** — add `_coerce()` in `learning/progress.py::load()`;
   numeric fields via `_as_int`; drop non-dict records with a log line. Test §6.3-1.
3. ✅ **Quarantine corrupt history instead of discarding** — `load()` `:145-147` renames the bad
   file and emits a user-facing notice. Test §6.3-2.

### P2 — next iteration

4. **Share without auth** — refuse `share=True` unless credentials are configured; set `show_error=False` and move technical exception text out of student-facing Markdown (`generation.py:288-292`, `app.py:375-379`).
5. **Stop using `esc()` on the data path** — `ui/learn.py:84-85` and `ui/practice.py:119`; keep it at the render boundary only.
6. **One weak-area definition** — thread `settings.weak_accuracy_threshold` everywhere; align denominator; single `WeakArea` type across `assessment`/`practice`/`progress`.
7. **Cross-process write safety** — lock file or documented single-process rule (§4.4); clean up the temp file in a `finally`.
8. **`assert` → explicit raise** at `huggingface_client.py:172`; narrow `except Exception` at `vector_store.py:310`.

### P3 — consolidation (do as one refactor branch, with tests from §6.3 first)

9. Delete §1.1 and §1.2 (unused imports and dead symbols) — **or** wire the platform services to consume `PLATFORM_PROMPT_PAIRS` instead of hard-coding the prompt names at their call sites, so the task → prompt map stops being a second, unread source of truth. Pick one; do not leave both states.
10. Unify the section-repair engines (§1.3-1) and the three dropdown mappers (§1.3-2); fix the stale `ui/revision.py:64` docstring.
11. One document renderer (`doc_md` vs `page_md`/`sheet_md`); remove the duplicate Practice-tab score rendering.
12. Single-source subject vocabulary + emoji (`learning/service.py:416`, `ui/components.py:254`, `app.py:67`); turn the `utils/constants.py` parallel dicts into enum-owned data or `.get()`-with-default.
13. Split `utils/constants.py` and move the panel builders out of `app.py`; comment the two package cycles (§2.2) where the lazy imports live.
14. Performance: query-level LRU (§7.1), sidecar-based `IndexStats` (§7.2), single-pass progress aggregation (§7.3).
15. Move `describe_capabilities()` to `ui/`; inject or re-implement `citations_to_markdown` so `ui/` stops importing `agents/` (§2.1).
16. Add `pytest-cov` to dev requirements and replace the name-based claims in §6 with measured numbers.

### Explicitly fine as-is

The atomic temp+replace write, the never-raises `load()`, the chat-path exception funnel in
`app.py`/`agents/graph.py`/`agents/tutor.py`, the `CapabilityRegistry` Protocol seam for
voice/photo, the mock-LLM test harness, the injected-fake-Gradio panel tests, prompt-name
whitelisting, bounds-checked Pydantic settings, and the FAISS search dedupe/threshold/filter
fallback.

---

## Appendix — how this was produced

- `ast`-based scanners over all first-party modules for unused imports, module-level symbols
  with no reference, `except Exception`/`BaseException` sites, unreachable statements, and the
  local import graph (package- and module-level cycles). Every candidate was then confirmed by
  reading the file and by a repository-wide `Select-String` occurrence count.
- Targeted reads of `learning/*` (services), `ui/*` (presentation), `agents/*` (graph + tutor),
  `rag/vector_store.py`, `llm/huggingface_client.py`, `prompts/loader.py`, `utils/*` and
  `app.py` for each review goal.
- Test inventory by file (`tests/*.py`: 175 tests over 15 files) plus a production-symbol vs.
  test-reference cross-check.
- **Limitations:** no linter, type checker or coverage tool is installed in
  `f:\Project\neet_ai_tutor\.venv`, and **Gradio is not installed**, so no real Gradio render
  was observed; §3.1 and §5.2 are reasoned from the launch configuration and the handler code,
  not from a reproduced traceback. The repository is not a git working tree, so no change
  history was available.
