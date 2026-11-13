# UI redesign report

## Before

- The Dashboard used a tall hero, profile and motivation panels, multiple tables, and Learn-specific copy.
- The default navigation and Dashboard metadata did not follow the intended student flow.
- Developer Mode and technical diagnostics remained in the rendered component tree, even when hidden by CSS.
- Ask Genie was disabled if the NCERT index was missing, despite general-knowledge fallback being supported.

## After

- The app opens on Ask Genie and presents Ask Genie, Practice, Progress, Revision Hub and Dashboard in the requested order. Learn is omitted by default and remains available only with `NEETRA_ENABLE_LEARN=true`.
- The Dashboard is compact: five direct-action cards and a progress summary for answered questions, practice attempts, accuracy and weak topics. Student-facing Dashboard copy no longer tells students to use Learn.
- Developer Mode and diagnostic panels are no longer rendered in the student UI.
- Ask Genie has four suggested-question buttons and remains usable without an index when general-knowledge fallback is enabled.
- Gradio footer/developer chrome remains suppressed, and mobile layouts wrap the dashboard actions and navigation.

## Components changed

- `app.py`: Ask Genie fallback gate, suggested question handlers, request diagnostics, navigation shell and removal of developer-only widgets.
- `ui/dashboard.py`: compact cards, progress indicators and direct navigation.
- `ui/components.py`: responsive action-card and prompt-chip styling, plus student navigation contract.
- `config.py`, `.env.example`, `learning/service.py`, `agents/tutor.py`: product defaults and visible module metadata.
- Tests and `README.md`: regression coverage and current product behavior.

No before/after screenshots were captured in this run.
