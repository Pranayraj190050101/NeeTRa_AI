# NEETra AI

> **Your Personal NEET Doubt Solver,**
> **Practice Partner and Rank Improver.**

A **NEET preparation assistant** built on an agentic RAG core. Its primary module is the coach chat:
Genie answers a student's doubt from *their own* NCERT textbooks, cites chapter and page numbers,
teaches like a NEET faculty member and always replies in a fixed, revisable section format - streamed
token by token into a Gradio UI. Around the chat sit a Practice engine, a Progress tracker, a Revision
Hub and a Dashboard; the optional **Learn** library (chapter study pages) ships behind the
`NEETRA_ENABLE_LEARN` feature flag. Everything is preserved and stays one environment variable away.

The app and retrieval index run locally. The default offline mock provider returns placeholders;
configure Hugging Face for local model answers or Groq for hosted answers. A hosted provider receives
the prompts sent to it.

```text
User Question  (+ the Answer mode chosen in the sidebar)
      |
      v
Subject Router Agent        (Qwen2.5 -> strict JSON: subject + intent)
      |
      v
RAG Retrieval Agent         (FAISS top-k over the subject, score threshold, dedupe)
      |
      v
Answer Generation Agent     (Neet Genie persona + mode contract, grounded prompt, citations)
      |
      v
Coach format repair         (the mode's section headings are guaranteed, never invented facts)
      |
      v
Response  +  Sources panel (file, page, similarity score)
```

The router still emits the intents `mcq`, `pyq`, `study_plan` and `weakness`, and those pipelines stay
registered as *planned* in `agents/registry.py`. The coach layer below sits on top of the same graph:
**MCQ Practice** and **PYQ Focus** modes can already serve an `mcq` / `pyq` request inside the existing
retrieval + answer path, so nothing had to be removed or rewritten to add them.

---

## The assistant: Ask Genie first

`python app.py` opens the student modules in this order:

| Tab | What a student does | Service layer | UI module |
| --- | --- | --- | --- |
| **Ask Genie** | the primary module - the coach chat: type any doubt and get a grounded answer with citations, same widgets, same four answer modes, same streaming | `agents/tutor.py` | `app.py::build_ask_genie_tab` |
| **Practice** | takes a Subject, Chapter or Topic test, answers the MCQs, submits, and gets Score / Accuracy / Weak Topics with the correct answer and explanation for every question | `learning/practice.py` + `learning.assessment.py` | `ui/practice.py` |
| **Progress** | reviews tests attempted, scores, accuracy, per-subject and per-chapter performance, ranked weak areas and a two-week daily series | `learning/progress.py` | `ui/progress.py` |
| **Revision Hub** | generates Flash Notes, a Formula Sheet, High-Yield Points or Last-Minute Revision Notes for any chapter, plus a *Revise my weak areas* pass | `learning/revision.py` | `ui/revision.py` |
| **Dashboard** | compact product landing page with five shortcuts (Ask a Doubt, Practice Tests, Revision Planner, Mock Tests, Progress Dashboard) plus questions answered, practice attempts, accuracy and weak topics | `learning/service.py` (`dashboard()`) | `ui/dashboard.py` |
| **Learn** *(optional; off by default)* | picks subject -> chapter and gets the four-section study page (📘 Chapter Summary / 🎯 High-Yield Concepts / 📝 Important NEET Points / ⚡ Quick Revision Notes) built from retrieved pages - shown only when `NEETRA_ENABLE_LEARN=true` | `learning/learn.py` + `learning/catalog.py` | `ui/learn.py` |

Every generated page, test and revision sheet goes through the **same** retrieval pipeline as the chat:
`GeniePlatform` calls the tutor's graph, so content always comes from the indexed NCERT pages with
citations. Nothing is hardcoded - an empty index produces an honest *"no chapters indexed yet"* notice
instead of invented material. Set `NEET_PLATFORM_ENABLED=false` to serve the original single-chat layout
and `NEETRA_ENABLE_LEARN=true` to opt into the optional Learn tab (its code and tests stay in the repo).

---

## Coach layer: persona, standard format and answer modes

**1. NEET tutor persona** (`prompts/persona.txt`, toggled by `NEET_PERSONA_ENABLED`) - the system
prompt is composed as *persona + grounding rules + the selected mode's contract*, so the model
behaves as a NEET faculty member with 15 years of NCERT teaching: it teaches from the NCERT position
first, marks **HIGH-YIELD** lines that must be memorised verbatim, names the **TRAP** in every
explanation, uses NEET exam vocabulary (statement-based, assertion-reason, match-the-following,
negative marking) and gives the exam shortcut. It never invents a page number, paper, year or statistic.

**2. Standard answer format** (`utils/answer_format.py`) - every explanation reply is delivered in the
same five sections so a student can revise straight from the chat history:

```text
📘 Concept Explanation
🎯 NEET Importance
🧠 Key Points To Remember
⚠ Common Student Mistakes
✅ Quick Revision Notes
```

Local models do not always obey that shape, so the answer agent *repairs* it before the answer reaches
the chat: recognised headings (`Key Points`, `**Common mistakes:**`, `### Revision notes`, ...) are
relabelled to the canonical emoji titles, and a section the model skipped is appended - filled from the
retrieved NCERT passages when they exist, otherwise from conservative coach guidance. The repair is
reported as a note under the answer (`Coach format applied: ...`), never silently.

**3. Answer modes** (`utils/constants.AnswerMode`, chosen in the Gradio sidebar):

| Mode | Key | Sections produced | Tuning |
| --- | --- | --- | --- |
| 📘 Detailed Explanation | `detailed` | the five standard sections | ~300 words, top-k as configured |
| ⚡ Quick Revision | `quick_revision` | the same five sections, compressed | 0.5x length, 0.8x passages, ~90-second skim |
| 🧩 MCQ Practice | `mcq_practice` | 🧩 Practice MCQs / 🔑 Answer Key With Explanation / ⚠ Traps Tested In NEET / ✅ Quick Revision Notes | `NEET_MCQ_COUNT` questions, 1.4x passages |
| 📜 PYQ Focus | `pyq_focus` | 📜 How NEET Asks This + the standard sections | 1.2x passages, no invented years or marks |

Modes are a **presentation** axis; the router's intents are a **request** axis. The two meet in one
rule inside `agents/graph.py::_after_router`: a mode that can serve the detected intent keeps the turn
in the implemented pipeline (`MCQ Practice` + `mcq` intent -> retrieval -> answer), otherwise the
polite *deferred* node answers with the roadmap and points at the mode that would help.

---

## 1. Tech stack

| Layer | Choice |
| --- | --- |
| Language | Python 3.10 - 3.12 |
| UI | Gradio (`gr.Blocks`, `gr.Chatbot(type="messages")`) |
| Orchestration | LangGraph `StateGraph` |
| RAG plumbing | LangChain (`Document`, text splitters, FAISS wrapper) |
| Vector store | FAISS (persisted to `vectordb/`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-d, L2 normalised) |
| LLM | `Qwen/Qwen2.5-7B-Instruct` via HuggingFace Transformers |
| PDF parsing | PyPDF (`pypdf.PdfReader`) |
| Config | pydantic-settings + `.env` |

The orchestration layer is verified against both LangChain 0.3 and 1.x (tested with `langgraph`
1.2.x, `langchain-core` 1.7.x, `langchain-community` 0.4.x); the only breaking difference between
those majors, the FAISS pickle-trust keyword, is detected at runtime in `rag/vector_store.py`.

---

## 2. Project structure

```text
neet_ai_tutor/
├── app.py                     # module tabs, streaming chat handler, --check diagnostics
├── config.py                  # pydantic settings, all NEET_* env vars
├── requirements.txt
├── requirements-dev.txt
├── .env.example
│
├── agents/
│   ├── base.py                # BaseAgent (timing, logging, progress events)
│   ├── state.py               # TutorState (TypedDict) + TutorResult
│   ├── registry.py            # intent -> agent registry (extension point)
│   ├── router_agent.py        # Question -> subject + intent (JSON, keyword fallback)
│   ├── retrieval_agent.py     # FAISS search -> context blocks + citations (mode-scaled k)
│   ├── answer_agent.py        # persona + mode prompt -> streamed, format-repaired answer
│   ├── graph.py               # LangGraph wiring, conditional edges, error nodes
│   └── tutor.py               # facade used by the UI (lazy model loading, health)
│
├── rag/
│   ├── document_loader.py     # PyPDF -> page-level Documents (+ subject metadata)
│   ├── chunker.py             # RecursiveCharacterTextSplitter wrapper
│   ├── embeddings.py          # HuggingFaceEmbeddings factory
│   ├── vector_store.py        # FAISS build/save/load/search + stats sidecar
│   └── ingest.py              # CLI: python -m rag.ingest
│
├── llm/
│   ├── base.py                # LLMClient contract + ChatMessage
│   ├── huggingface_client.py  # Qwen chat template, 4-bit option, streaming
│   ├── mock_client.py         # offline stub for UI/plumbing tests
│   └── factory.py
│
├── prompts/
│   ├── loader.py              # Template-based loader with startup validation
│   ├── router.txt
│   ├── persona.txt            # Neet Genie: 15 years of NCERT faculty habits
│   ├── answer_system.txt      # grounding + citation rules
│   ├── answer_user.txt        # + answer_user_no_context.txt (both carry $mode)
│   ├── mode_detailed.txt      # one contract per answer mode
│   ├── mode_quick_revision.txt
│   ├── mode_mcq_practice.txt
│   ├── mode_pyq_focus.txt
│   ├── learn_chapter_*.txt    # the four-section study page contract
│   ├── mcq_*.txt              # test generation for a subject / chapter / topic scope
│   └── revision_*.txt         # Revision Hub deliverable contracts
│
├── learning/                  # the platform services (no Gradio import anywhere)
│   ├── service.py             # GeniePlatform facade: catalog, dashboard, ask(), stats
│   ├── catalog.py             # ChapterCatalog from the index sidecar / metadata (no FAISS at startup)
│   ├── sections.py            # CHAPTER_SECTIONS + REVISION_SECTIONS and their repair stamp
│   ├── learn.py               # chapter / topic study pages (retrieval-grounded)
│   ├── assessment.py          # MCQ parsing, NEET marking, GradedTest analytics
│   ├── practice.py            # PracticeSet generation + submission per TestScope
│   ├── revision.py            # flash notes, formula sheets, high-yield, last-minute, weak areas
│   ├── progress.py            # ProgressStore: one JSON document per profile (the DB seam)
│   ├── motivation.py          # daily motivation line, chosen from live study statistics
│   └── capabilities.py        # CapabilityRegistry: chat active, voice/image are declared seams
│
├── ui/                        # Gradio widgets, thin over learning/ (content logic stays testable)
│   ├── components.py          # TabNav (version-safe tab jumps), tables, bars, lazy gradio import
│   ├── dashboard.py           # welcome, motivation, stats, subject cards, quick actions
│   ├── learn.py               # subject -> chapter navigation and the study page
│   ├── practice.py            # scope picker, static question widgets, submit and report
│   ├── progress.py            # scorecards, subject/chapter tables, weak areas, reset
│   └── revision.py            # deliverable picker and revision sheets
│
├── utils/
│   ├── constants.py            # Subject / Intent / AnswerMode / TestScope / RevisionDeliverable enums
│   ├── answer_format.py        # the chat section contract and its repair engine
│   ├── text_utils.py           # normalisation, JSON extraction, history rendering
│   ├── device.py               # CUDA/CPU selection helpers
│   └── logging_utils.py
│
├── data/                      # your PDFs, grouped by subject
│   ├── physics/
│   ├── chemistry/
│   └── biology/
│
├── vectordb/                  # FAISS index + meta.json (generated)
├── progress/                  # study history, one JSON per profile (generated)
├── logs/                      # rotating log file (generated)
├── scripts/offline_check.py   # stdlib-only smoke test (no ML dependencies)
└── tests/                     # offline pytest suite (mock LLM + fake embeddings)
```

---

## 3. Installation

> Use **Python 3.10, 3.11 or 3.12 (64-bit)**. Wheels for `torch` and `faiss-cpu` are not published
> for every brand-new Python release (for example 3.13/3.14 at the time of writing).

### Windows (PowerShell)

```powershell
cd neet_ai_tutor
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# GPU build of torch first (skip this line for CPU only)
pip install torch --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
Copy-Item .env.example .env
```

### Linux / macOS

```bash
cd neet_ai_tutor
python3 -m venv .venv
source .venv/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cu124   # optional, GPU
pip install -r requirements.txt
cp .env.example .env
```

Hardware guidance for `Qwen2.5-7B-Instruct`:

| Hardware | Recommendation |
| --- | --- |
| 24 GB+ VRAM | `NEET_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct` (bf16, default) |
| 12-16 GB VRAM | 7B with `NEET_LLM_LOAD_IN_4BIT=true`, or `Qwen/Qwen2.5-3B-Instruct` |
| 8 GB VRAM / CPU only | `Qwen/Qwen2.5-1.5B-Instruct` (slower, still usable) |
| No model download wanted | `NEET_LLM_BACKEND=mock` (plumbing demo, not a tutor) |

---

## 4. Add your study material

Put PDFs into the subject folders - the folder name becomes the retrieval filter:

```text
data/
├── physics/      class11_mechanics.pdf, current_electricity.pdf ...
├── chemistry/    mole_concept.pdf, organic_basics.pdf ...
└── biology/      cell_biology.pdf, human_physiology.pdf ...
```

`.txt` and `.md` files are also accepted. Scanned PDFs without a text layer must be OCR'd first.

### Build the index

```bash
python -m rag.ingest --rebuild     # full rebuild (recommended after adding books)
python -m rag.ingest --append      # only newly added pages (idempotent, uses doc_id)
python -m rag.ingest --source data/biology --chunk-size 1000 --chunk-overlap 200
```

This creates:

```text
vectordb/neet_index/index.faiss
vectordb/neet_index/index.pkl
vectordb/neet_index.meta.json      # stats shown in the UI (no model loading)
```

---

## 5. Run the tutor

```bash
python app.py                                  # http://127.0.0.1:7860
python app.py --host 0.0.0.0 --port 7861
python app.py --share                          # temporary public Gradio link
python app.py --check                          # validate config, prompts, dependencies
```

`--check` prints a JSON report (paths, index statistics, prompt validation, installed versions) and
exits non-zero when something is missing. It can be run even before the ML stack is installed:
Gradio is imported lazily, so the report still tells you exactly which packages are absent. With
`NEET_LLM_BACKEND=mock` the Transformers/Torch checks are skipped, which makes it usable on a
machine that only runs the offline demo.

First question triggers the model download (~15 GB for the 7B checkpoint) and prints progress in the
terminal; the UI stays responsive and shows a status line such as *"Loading Qwen/Qwen2.5-7B-Instruct"*.

The sidebar carries the coach controls: pick the **Answer mode** (📘 Detailed Explanation /
⚡ Quick Revision / 🧩 MCQ Practice / 📜 PYQ Focus), pin the subject, change the number of
retrieved passages and toggle token streaming. Below the controls you get a live one-liner about the
sections the chosen mode will produce. Every answer exposes its sources with file name, page number and
similarity score, and the **Coach** tab documents the persona, the format contract and all four modes.

The answer mode is remembered per question, not per session: switch it whenever you want a revision
sheet instead of a lecture, and the same retrieved pages are reused.

With the platform on, the app opens on **Ask Genie**. A student can do the main study loop without
leaving the app: **Ask Genie** about a topic -> **Practice** a test on it -> read the **Progress**
scorecards -> use the **Revision Hub** for topics that lost marks. The Dashboard's quick actions jump
straight into those tabs, and *Revise my weak areas* pulls the
weak list from the progress store. Nothing blocks start-up: the index, the LLM and the chapter catalog
are all resolved on first use, so the tabs render before a single model file is loaded.

---

## 6. Configuration

All settings live in `config.py` and can be overridden by environment variables prefixed `NEET_`
(or in `.env`, see `.env.example`).

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_PROVIDER` | unset; resolves to `mock` by default | `groq`, `huggingface` or `mock`; takes precedence over the legacy backend |
| `NEET_LLM_BACKEND` | `mock` | legacy provider setting: `hf` (local Transformers model) or `mock` (offline stub) |
| `NEET_LLM_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | any HuggingFace chat model |
| `NEET_LLM_DEVICE` | `auto` | `auto` / `cpu` / `cuda` / `mps` |
| `NEET_LLM_DTYPE` | `auto` | `float16` / `bfloat16` / `float32` |
| `NEET_LLM_LOAD_IN_4BIT` | `false` | NF4 quantisation (CUDA + `bitsandbytes`) |
| `NEET_LLM_MAX_NEW_TOKENS` | `768` | answer length cap |
| `NEET_LLM_MAX_INPUT_TOKENS` | `4096` | prompt token budget |
| `NEET_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | swap for a domain-tuned encoder |
| `NEET_DATA_DIR` / `NEET_VECTOR_STORE_DIR` | `data` / `vectordb` | corpus and index locations |
| `NEET_CHUNK_SIZE` / `NEET_CHUNK_OVERLAP` | `900` / `150` | chunking |
| `NEET_MIN_CHUNK_CHARS` | `80` | drop fragments too short to be useful |
| `NEET_TOP_K` | `5` | passages retrieved |
| `NEET_FETCH_K` | `20` | candidate pool before the subject filter |
| `NEET_SCORE_THRESHOLD` | `0.15` | minimum cosine similarity |
| `NEET_CONTEXT_CHAR_BUDGET` | `6000` | max characters of context per answer |
| `NEET_MAX_QUESTION_CHARS` / `NEET_MAX_ANSWER_CHARS` | `2000` / `6000` | input and output guards |
| `NEET_HISTORY_TURNS` | `4` | chat messages sent to the model |
| `NEET_ALLOW_GENERAL_FALLBACK` | `true` | answer without retrieval (with disclaimer) |
| `NEET_STREAM_TOKENS` | `true` | stream answer tokens to the UI |
| `NEET_ANSWER_MODE` | `detailed` | default mode: `detailed` / `quick_revision` / `mcq_practice` / `pyq_focus` |
| `NEET_PERSONA_ENABLED` | `true` | prepend the NEET faculty persona (`prompts/persona.txt`) |
| `NEET_ENFORCE_ANSWER_FORMAT` | `true` | standardise headings, fill skipped sections |
| `NEET_MCQ_COUNT` | `5` | questions generated in MCQ Practice mode |
| `NEET_PLATFORM_ENABLED` | `true` | serve the five student tabs (`false` = chat-only layout) |
| `NEET_TEST_QUESTION_COUNT` | `5` | questions per generated Practice test (2-15) |
| `NEET_MARKS_CORRECT` / `NEET_MARKS_INCORRECT` | `4.0` / `-1.0` | NEET marking used when a test is submitted |
| `NEET_WEAK_ACCURACY_THRESHOLD` | `0.75` | below this accuracy a topic is reported as weak |
| `NEET_PROGRESS_DIR` | `progress` | where the JSON study history is written |
| `NEET_APP_NAME` / `NEET_APP_TAGLINE` | `NEETra AI` / `Every NEET Doubt. One Wish Away.` | UI branding (the coach persona inside the prompts is still "Neet Genie") |
| `NEETRA_ENABLE_LEARN` | `false` | opt into the Learn tab; the flag name has no `NEET_` prefix by design |
| `NEET_HF_TOKEN` | - | token for gated HuggingFace repos |

---

## 7. How the agents work

**Router** - renders `prompts/router.txt`, asks the model for one JSON object
(`subject`, `intent`, `confidence`, `reasoning`), and validates it. Unparsable output falls back to a
keyword classifier built from `utils/constants.py`; a subject chosen in the UI always wins. Failures
here degrade quality, never the pipeline.

**Retrieval** - expands short follow-ups ("and its unit?") with the previous question, searches FAISS
with `filter={"subject": ...}`, falls back to the whole index when the subject filter is too narrow,
converts squared L2 distances of normalised vectors back to cosine similarity, applies the score
threshold, de-duplicates chunks and packs them into a character-budgeted numbered context block. The
answer mode only scales *how many* passages are requested (through the existing `search(k=...)`
argument) - the RAG layer itself has no idea what a mode is.

**Answer** - composes the system prompt as *persona + grounding rules + the selected mode's contract*
(with a machine readable `MODE_ID:` line), renders `answer_user.txt` (or the no-context variant, which
now names the mode too), calls the model with the Qwen chat template, streams tokens through
`TextIteratorStreamer`, then strips role prefixes, a model-written "Sources:" tail and over-long output
(the length cap and the token budget both scale with the mode). Finally `utils/answer_format.py`
fixes the section contract: relabelled headings and appended missing sections are reported as a
note under the answer.

Failures are turned into chat messages (`failed` node) instead of tracebacks; every step writes its
duration into `state["timings"]` and into `logs/neet_tutor.log`.

---

## 8. Platform internals

**Persistence - JSON now, a database later.** `learning/progress.py::ProgressStore` keeps the whole
study history in `progress/<profile>.json`: a versioned document (`"schema": 1`) with a `tests` list of
graded-test records, a `daily` map (questions / correct / score per day, feeding the two-week series) and
an `events` counter for non-test activity (`doubts`, `chapter_pages`, `revision_sheets`). Writes are
atomic (temp file + `os.replace`) behind an `RLock`; reads are cached on the file's mtime. Every analytics
method the UI uses - `stats()`, `subject_performance()`, `chapter_performance()`, `weak_areas()`,
`daily_series()`, `tests()` - is a store method, not a Gradio concern, so replacing the JSON file with
SQLite or Postgres later means swapping one class and keeping the same call signatures.

**Input channels.** Typed chat and photo-question text extraction are active; voice input, study plans and
weakness analysis remain planned. Ask Genie accepts an uploaded photo, extracts text locally with
`rapidocr-onnxruntime`, and places it in the editable question box. The student reviews or corrects the
text before sending it through the same router -> retrieval -> answer graph as typed questions. OCR runs
on the local machine; handwriting, equations and diagrams may need manual correction.

---

## 9. Tests

GitHub Actions runs the test suite and offline smoke checks on Python 3.10, 3.11 and 3.12 for pushes and pull requests.

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest -q
```

The suite runs fully offline: `MockLLM` stands in for Qwen (it reads the `MODE_ID:` marker and answers
in the selected mode's shape, and now also serves the Learn / MCQ / Revision generation prompts) and
`DeterministicFakeEmbedding` stands in for MiniLM, so it covers document discovery/loading, chunking
metadata, routing, FAISS persistence/filtering, citation formatting, streaming callbacks, the deferred-agent
path, the tutor facade, the Gradio chat handler, the answer-mode contract (`tests/test_answer_modes.py`),
the coach pipeline end to end (`tests/test_coach_pipeline.py`) and the platform: the chapter catalog
(`test_chapter_catalog.py`), assessment and grading (`test_assessment.py`), the JSON progress store
(`test_progress_store.py`), the five platform services (`test_platform_services.py`) and all six tabs
against a stand-in Gradio (`test_platform_panels.py`) - 175 tests in a few seconds.
`python scripts/offline_check.py` runs a smaller, stdlib-only subset - including the five required chat
section titles, every mode's prompt contract, the chapter and revision section shapes, the catalog,
grading, the progress store, the capability registry and the tab navigation - that works before any
dependency is installed.

---

## 10. Extending the system

Adding the **MCQ Generator Agent** (same pattern for PYQ, planner, weakness):

1. Create `agents/mcq_agent.py`:

   ```python
   from agents.base import BaseAgent

   class MCQAgent(BaseAgent):
       name = "mcq"

       def run(self, state, config):
           retrieved = state.get("retrieved", [])   # reuse the retrieval layer
           ...
           return {"answer": mcq_markdown}
   ```

2. In `agents/graph.py` add `graph.add_node("mcq", ...)` and extend `_after_router` to return
   `"mcq"` for `Intent.MCQ`.
3. In `agents/registry.py` flip the MCQ entry's `status` to `"active"`.

Because the state already carries `retrieved`, `citations`, `subject`, `mode` and `history`, a new
agent only needs its own prompt and generation logic.

Until such an agent exists, the **MCQ Practice** and **PYQ Focus** answer modes already cover those
requests inside the current pipeline - they reuse retrieval and change only the presentation, which is
why `AnswerMode.satisfies(intent)` gates the deferred node in `agents/graph.py`.

Adding a **new module tab** (or a new input channel) follows the pattern the six tabs already use:

1. Put the logic in `learning/<module>.py` with no Gradio import - plain functions and dataclasses that
   take a `Settings` plus the retrieval/generation callable, so it is testable and reusable from the CLI.
2. Put the widgets in `ui/<module>.py`: content helpers next to `build(gr, ...)` / `wire(gr, ...)`, and
   register the tab in `ui/components.py::MODULE_TABS` (label + stable id) so `TabNav` can jump to it
   from the Dashboard's quick actions.
3. Register the capability in `learning/capabilities.py::DEFAULT_CAPABILITIES` if the module introduces a
   new channel - a `PLANNED` entry is honest UI text today and the activation point tomorrow.

To add voice input, implement `VoiceTranscriber`, register it with
`platform.capabilities.register("voice", MyTranscriber())`, and wire a `gr.Audio` input to
`GeniePlatform.listen()` in the Ask Genie tab.

---

## 11. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `No FAISS index found` in the answer | Run `python -m rag.ingest --rebuild` after adding PDFs |
| `No extractable text found` | The PDF is scanned: run `ocrmypdf in.pdf out.pdf` first |
| `GPU out of memory` | Lower `NEET_LLM_MAX_NEW_TOKENS`, set `NEET_LLM_LOAD_IN_4BIT=true`, or use the 3B/1.5B model |
| Very slow answers on CPU | Use `Qwen/Qwen2.5-1.5B-Instruct`, reduce `NEET_LLM_MAX_NEW_TOKENS`; retrieval stays fast |
| 4-bit loading fails on Windows | `bitsandbytes` support on native Windows is experimental - use WSL2 or full precision |
| Answers ignore your books | Check `python app.py --check`; the index may be empty or the subject filter too narrow (`NEET_SCORE_THRESHOLD=0.05`) |
| Model download is slow/blocked | Set `HF_ENDPOINT` to a mirror, or pre-download with `huggingface-cli download` |
| `Prompt file missing` at startup | Run from the project folder so `prompts/` is found, or set `NEET_PROMPT_DIR` |
| Learn / Revision Hub says no chapters are listed | The catalog is read from the index sidecar - run `python -m rag.ingest --rebuild`, then press **Refresh chapters** in the Learn tab |
| Practice shows fewer questions than configured | `NEET_TEST_QUESTION_COUNT` is a target: the parser keeps only questions it could read with four options, and the status line says how many arrived |
| Progress looks empty after a reset | History is per profile in `progress/<profile>.json`; check `NEET_PROGRESS_DIR` and the profile name before assuming the data is lost |

---

## 12. Notes and limitations

- The tutor is a study aid. It cites retrieved passages, but always verify formulas and values against
  NCERT before relying on them.
- `all-MiniLM-L6-v2` is an English model; NEET material in English works best.
- FAISS indexes are pickled (`.pkl`). Only load indexes produced by this application
  (`NEET_FAISS_ALLOW_DANGEROUS_DESERIALIZE=true` is only safe for your own files).
- Source PDFs are git-ignored by default - they are usually large and copyrighted.
