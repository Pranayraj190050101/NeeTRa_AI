"""NEETra AI platform services.

Modular, Gradio-free building blocks that sit on top of the chat pipeline:

* :mod:`learning.catalog` - subject/chapter hierarchy read from the FAISS index
* :mod:`learning.learn` - chapter study pages (summary, high-yield, NEET points)
* :mod:`learning.practice` - subject / chapter / topic tests and their grading
* :mod:`learning.revision` - flash notes, formula sheets and last-minute recap
* :mod:`learning.progress` - JSON persistence of everything attempted
* :mod:`learning.capabilities` - extension points for voice and image solving
* :mod:`learning.service` - the ``GeniePlatform`` facade the UI talks to

Submodules are imported explicitly (``from learning.catalog import Chapter``)
so that a stdlib-only tool such as ``scripts/offline_check.py`` can pull in the
cheap modules without loading LangChain.
"""

__all__ = [
    "assessment",
    "capabilities",
    "catalog",
    "generation",
    "learn",
    "motivation",
    "practice",
    "progress",
    "revision",
    "sections",
    "service",
]
