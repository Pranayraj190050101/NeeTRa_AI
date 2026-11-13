# Ask Genie debug report

## Root cause and fix

The chat submit handlers already connected both **Send to Genie** and Enter to the streaming responder, which invokes `NEETTutor.ask()`, streams answer tokens to the chatbot, and renders citations separately. The blocker for a fresh local setup was the UI readiness gate: it disabled Ask Genie whenever textbook retrieval was unavailable, even though retrieval and generation handle an absent index and `NEET_ALLOW_GENERAL_FALLBACK` is enabled by default.

Ask Genie now stays enabled when the answer provider is verified and general-knowledge fallback is allowed. The status panel explains that textbook sources are unavailable while general answers remain possible. If the provider is unavailable, or fallback is disabled and retrieval is unavailable, the input remains disabled. Diagnostic logs record request mode, subject, length, completion state and citation count without logging student question text.

## Verification

- Added a regression test for “Give quick revision notes for Cell Cycle” with no index; it verifies a generated chat response and the Quick Revision section.
- Added readiness-gate tests for enabled and disabled general fallback.
- Existing handler tests continue to verify button/submit streaming, answer modes, citations and error rendering.

The offline mock provider only verifies the end-to-end UI and response contract; it does not produce study-quality answers. Real NEET content and textbook citations require a configured real provider and, for citations, an ingested index.
