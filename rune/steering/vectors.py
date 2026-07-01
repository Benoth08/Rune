"""Pure numerical core of activation steering (CAA).

Kept torch-free (numpy only) so the math is unit-testable without a loaded
model. The torch bridge (``engine.py``) captures activations and converts
them to numpy arrays before calling these functions.

The per-layer recipe:
  v_l         = mean(pos_l) - mean(neg_l)        # difference of means
  score_l     = cohens_d(proj onto v̂_l)          # how linearly separable
  selected    = top-k layers within a mid-late band, by score
  scale_l     = mean ||activation|| at layer l    # to make alpha portable
At inference the hook adds:  h_l += alpha * scale_l * v̂_l
"""

from __future__ import annotations

import numpy as np

__all__ = ["diff_of_means", "unit", "cohens_d", "select_layers",
           "blend_scores", "clamp_norm", "axis_cosine"]


def axis_cosine(a: dict, b: dict) -> float | None:
    """Mean cosine between two axes' steering vectors over their SHARED layers.

    ``a``/``b`` map ``layer_index -> unit_vector`` (lists or arrays). Layers
    must be compared in the same activation space, hence the shared-layer
    restriction. Returns ``None`` if the axes share no layer. Used to detect
    collinear axes before they are combined (the Gram-matrix orthogonality
    check the steering library documents but must enforce)."""
    shared = sorted(set(a) & set(b))
    if not shared:
        return None
    cs = []
    for l in shared:
        va = np.asarray(a[l], dtype=np.float64)
        vb = np.asarray(b[l], dtype=np.float64)
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na > 0 and nb > 0:
            cs.append(float(va @ vb / (na * nb)))
    return float(np.mean(cs)) if cs else None


def clamp_norm(vec: np.ndarray, cap: float) -> np.ndarray:
    """Scale ``vec`` down so its L2 norm never exceeds ``cap`` (direction kept).

    Used as the GLOBAL budget when several steering axes are combined: the
    summed per-layer coefficient is clamped so the total nudge stays a bounded
    fraction of the activation, regardless of how many axes stack.
    """
    vec = np.asarray(vec, dtype=np.float64)
    if cap <= 0:
        return np.zeros_like(vec)
    n = float(np.linalg.norm(vec))
    if n > cap and n > 0:
        return vec * (cap / n)
    return vec


def diff_of_means(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Contrastive direction: mean(positive) - mean(negative).

    ``pos``/``neg`` are (n, d) / (m, d) arrays of activations. The mean is
    deliberately preferred over a top-PC of the paired differences: on the
    small contrastive sets used here, an uncentered PCA chases a single
    high-magnitude outlier, whereas the mean stays robust.
    """
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    return pos.mean(axis=0) - neg.mean(axis=0)


def unit(v: np.ndarray) -> np.ndarray:
    """Return v / ||v||; a zero vector is returned unchanged."""
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def cohens_d(pos: np.ndarray, neg: np.ndarray, direction: np.ndarray) -> float:
    """Effect size of class separation when projecting onto ``direction``.

    A large value means the axis is strongly, linearly encoded at this
    layer — a good steering site. Returns 0.0 for degenerate inputs.
    """
    d = unit(direction)
    if not np.any(d):
        return 0.0
    p = np.asarray(pos, dtype=np.float64) @ d
    n = np.asarray(neg, dtype=np.float64) @ d
    if len(p) < 2 or len(n) < 2:
        return abs(float(p.mean() - n.mean()))
    sp, sn = p.std(ddof=1), n.std(ddof=1)
    pooled = np.sqrt(((len(p) - 1) * sp**2 + (len(n) - 1) * sn**2)
                     / (len(p) + len(n) - 2))
    if pooled <= 1e-12:
        return abs(float(p.mean() - n.mean()))
    return abs(float(p.mean() - n.mean()) / pooled)


def select_layers(
    scores: dict[int, float],
    num_layers: int,
    *,
    band: tuple[float, float] = (0.4, 0.7),
    top_k: int = 3,
) -> list[int]:
    """Pick the best-separating layers within a mid-late depth band.

    ``scores`` maps a 0-indexed decoder-layer to its cohens_d. The band is
    expressed as fractions of ``num_layers`` (default 40-70 %), which is
    where behavioural directions steer best without wrecking fluency. The
    band scales with the model automatically.
    """
    if num_layers <= 0 or not scores:
        return []
    lo = int(round(band[0] * num_layers))
    hi = int(round(band[1] * num_layers))
    lo = max(0, min(lo, num_layers - 1))
    hi = max(lo, min(hi, num_layers - 1))
    candidates = [l for l in scores if lo <= l <= hi]
    if not candidates:  # band empty (tiny model) → fall back to all layers
        candidates = list(scores)
    candidates.sort(key=lambda l: scores[l], reverse=True)
    return sorted(candidates[: max(1, top_k)])


def blend_scores(
    sep: dict[int, float],
    causal: dict[int, float],
    *,
    w_causal: float = 0.6,
) -> dict[int, float]:
    """Fuse a separability score map and a causal-effect score map.

    Each map is min-max normalised to [0, 1] independently, then combined as
    ``(1 - w_causal) * sep + w_causal * causal``. Causal weight dominates by
    default: separability says where the axis is *readable*, the causal score
    says where injecting it actually *moves the output* — the latter is what we
    ultimately want. Layers absent from ``causal`` keep their separability score
    only (so the blend never invents a causal value).
    """
    def _norm(d: dict[int, float]) -> dict[int, float]:
        if not d:
            return {}
        vals = list(d.values())
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-12:
            return {k: 0.0 for k in d}
        return {k: (v - lo) / (hi - lo) for k, v in d.items()}

    sn = _norm(sep)
    cn = _norm(causal)
    out: dict[int, float] = {}
    for l in sep:
        s = sn.get(l, 0.0)
        c = cn.get(l)
        out[l] = (1.0 - w_causal) * s + w_causal * c if c is not None else s
    return out
