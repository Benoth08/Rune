"""Tests for the steering numerical core (no torch needed).

Covers the CAA math (vectors.py) and the shipped axes integrity. The
torch bridge (engine.py) is validated on a live model on the pod.
"""

from __future__ import annotations

import numpy as np

from rune.steering.axes import AXES
from rune.steering.vectors import (
    axis_cosine, blend_scores, clamp_norm, cohens_d, diff_of_means, select_layers, unit,
)


def _two_clusters(seed=0, d=16, n=8, sep=3.0):
    rng = np.random.default_rng(seed)
    pos = rng.normal(0, 1, (n, d)); pos[:, 0] += sep
    neg = rng.normal(0, 1, (n, d)); neg[:, 0] -= sep
    return pos, neg


def test_diff_of_means_points_along_separating_axis():
    pos, neg = _two_clusters()
    assert diff_of_means(pos, neg).argmax() == 0


def test_unit_is_normalised_and_zero_safe():
    assert abs(np.linalg.norm(unit(np.array([3.0, 4.0]))) - 1.0) < 1e-9
    assert np.all(unit(np.zeros(5)) == 0)


def test_cohens_d_clean_axis_beats_noise():
    pos, neg = _two_clusters()
    clean = cohens_d(pos, neg, diff_of_means(pos, neg))
    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 1, (8, 16)), rng.normal(0, 1, (8, 16))
    noise = cohens_d(a, b, diff_of_means(a, b))
    assert clean > noise


def test_select_layers_band_and_topk():
    scores = {l: (l if 12 <= l <= 20 else 0.1) for l in range(28)}
    sel = select_layers(scores, 28, band=(0.4, 0.7), top_k=3)
    lo, hi = round(0.4 * 28), round(0.7 * 28)
    assert len(sel) == 3 and sel == sorted(sel)
    assert all(lo <= l <= hi for l in sel)


def test_select_layers_small_model_fallback():
    sel = select_layers({0: 1.0, 1: 2.0, 2: 0.5}, 3, top_k=2)
    assert len(sel) == 2


def test_select_layers_empty_inputs():
    assert select_layers({}, 0) == []
    assert select_layers({}, 10) == []


def test_axes_are_well_formed():
    assert len(AXES) >= 10, "expected the full axis library"
    for name, ax in AXES.items():
        assert ax.get("label") and ax.get("description"), name
        pairs = ax.get("pairs")
        # sweep needs >= 4 prompts per class for reliable separability
        assert pairs and len(pairs) >= 4, f"{name}: too few pairs"
        for pos, neg in pairs:
            assert isinstance(pos, str) and pos.strip(), f"{name}: empty pos"
            assert isinstance(neg, str) and neg.strip(), f"{name}: empty neg"
            assert pos != neg, f"{name}: identical poles"


def test_blend_scores_favours_causal():
    sep = {10: 1.0, 12: 0.0}      # 10 most separable
    causal = {10: 0.0, 12: 1.0}   # 12 most causal
    blended = blend_scores(sep, causal, w_causal=0.6)
    assert blended[12] > blended[10]  # causal weight dominates


def test_blend_scores_missing_causal_preserves_order():
    sep = {5: 0.2, 6: 0.9}
    blended = blend_scores(sep, {}, w_causal=0.6)
    assert blended[6] > blended[5]  # separability ordering kept (rescaled)


def test_blend_scores_handles_flat_inputs():
    out = blend_scores({1: 2.0, 2: 2.0}, {1: 5.0, 2: 5.0}, w_causal=0.6)
    assert set(out) == {1, 2}


def test_clamp_norm_caps_large_vectors():
    v = np.array([3.0, 4.0])               # norm 5
    c = clamp_norm(v, 1.0)
    assert abs(np.linalg.norm(c) - 1.0) < 1e-9
    # direction preserved
    assert np.allclose(c / np.linalg.norm(c), v / np.linalg.norm(v))


def test_clamp_norm_leaves_small_vectors():
    v = np.array([0.1, 0.0])
    assert np.allclose(clamp_norm(v, 1.0), v)


def test_clamp_norm_zero_cap():
    assert np.allclose(clamp_norm(np.array([1.0, 2.0]), 0.0), [0.0, 0.0])
    for name, ax in AXES.items():
        assert ax["label"] and ax["description"]
        assert len(ax["pairs"]) >= 3
        for pos, neg in ax["pairs"]:
            assert pos.strip() and neg.strip() and pos != neg


def test_no_unsafe_axis_present():
    # steering is style/tone only — guard against a safety-bypass axis
    banned = {"refusal", "safety", "jailbreak", "uncensored", "harmful"}
    assert not (set(AXES) & banned)


def test_package_imports_without_torch():
    # engine uses lazy torch imports, so the package must import fine here
    from rune.steering import AXES as A, SteeringEngine  # noqa: F401
    assert A is AXES


def test_axis_cosine_aligned_and_orthogonal():
    e0 = {18: [1.0, 0.0], 19: [1.0, 0.0]}
    e0b = {18: [1.0, 0.0], 19: [1.0, 0.0]}
    e1 = {18: [0.0, 1.0], 19: [0.0, 1.0]}
    assert abs(axis_cosine(e0, e0b) - 1.0) < 1e-9      # identical → 1
    assert abs(axis_cosine(e0, e1)) < 1e-9             # orthogonal → 0
    assert axis_cosine({1: [1.0]}, {2: [1.0]}) is None  # no shared layer


def test_axis_cosine_shared_layers_only():
    a = {18: [1.0, 0.0], 20: [1.0, 0.0]}
    b = {18: [1.0, 0.0], 19: [0.0, 1.0]}   # only layer 18 is shared
    assert abs(axis_cosine(a, b) - 1.0) < 1e-9
