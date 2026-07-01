"""V6 activation-steering layer (BETA).

CAA-style steering on the in-process core model: computes contrastive
vectors, auto-selects layers, caches per model, injects ``alpha *
scale_l * v̂_l`` via forward hooks. See ``engine.py`` (torch bridge),
``vectors.py`` (pure math), ``axes.py`` (shipped contrastive pairs).
"""

from __future__ import annotations

from rune.steering.axes import AXES
from rune.steering.engine import SteeringEngine

__all__ = ["SteeringEngine", "AXES"]
