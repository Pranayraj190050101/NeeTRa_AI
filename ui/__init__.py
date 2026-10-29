"""NEETra AI UI layer - one module per product tab.

Layout mirrors the six navigation tabs of the platform:

======================  =========================================
Tab                     Module
======================  =========================================
Dashboard               :mod:`ui.dashboard`
Learn                   :mod:`ui.learn`
Practice                :mod:`ui.practice`
Ask Genie               ``app.py`` (the existing coach chat, reused)
Progress                :mod:`ui.progress`
Revision Hub            :mod:`ui.revision`
shared formatters       :mod:`ui.components`
======================  =========================================

Every module splits pure content functions (no Gradio needed - these are what
the tests and the CLI print) from a ``build(gr, ...)`` function that creates the
widgets.  Gradio is always passed in or imported lazily, so ``app.py --check``
and the offline checks never require the UI dependency.
"""

from __future__ import annotations

__all__ = [
    "components",
    "dashboard",
    "learn",
    "practice",
    "progress",
    "revision",
]
