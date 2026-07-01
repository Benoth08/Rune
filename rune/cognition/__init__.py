"""Cognitive phases of the Hippocampe orchestrator.

The hippocampe was originally a ~1100-line monolith. This package
splits its responsibilities along the 5 theoretical phases of the
Lythéa cognitive cycle:

* :mod:`encoding`       — input → latent + entropies + entities
* :mod:`storage`        — write to SDM / KG / MHN / Chroma
* :mod:`surprise`       — composite biomimetic surprise (4 signals)
* :mod:`retrieval`      — KG identity + MHN + Chroma → RAG context
* :mod:`consolidation`  — microsleep + deep sleep plumbing

Plus a small :mod:`generation` helper module for text-stream cleanup
and the two-pass reasoning prompt — these are not phases per se, but
share the cognition concern and would otherwise pollute hippocampe.py.

Each phase is a class composed (not inherited) into Hippocampe and
takes its dependencies via the constructor (DI). This keeps every
phase trivially mockable in tests.
"""
