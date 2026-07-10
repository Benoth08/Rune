"""Tests for the sampling profile system.

Three layers are exercised:

1. **Catalogue invariants**: every model in CATALOG either has a valid
   ``SamplingProfile`` or falls back to ``DEFAULT_SAMPLING``. Specific
   models have specific recommended values (Mistral T=0.5, etc.) that
   we pin here so future edits to ``config.py`` don't silently revert
   the calibration work.

2. **Routes**: GET /api/config/sampling reads the live profile,
   POST /api/config/sampling updates fields partially (only the ones
   provided in the body).

3. **Auto-application on load**: when a model is loaded, its
   ``SamplingProfile`` is copied to ``hippocampe.sampling_profile``
   so the next generation uses the right defaults.
"""
from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# The routes module transitively imports rune.model (needs torch).
# In sandbox without torch, skip — on the pod torch is always there.
try:
    import torch
except (ImportError, OSError):
    pytest.skip("torch not available or broken CUDA", allow_module_level=True)


# ── Layer 1: Catalogue invariants ──────────────────────────────────────


def test_default_sampling_has_sane_values():
    """Sanity check on the fallback profile."""
    from rune.config import DEFAULT_SAMPLING

    # Conservative, mid-of-range — works for most Instruct models.
    assert 0.5 <= DEFAULT_SAMPLING.temperature <= 1.0
    assert DEFAULT_SAMPLING.top_p is not None
    assert 0.7 <= DEFAULT_SAMPLING.top_p <= 1.0
    assert DEFAULT_SAMPLING.repetition_penalty >= 1.0


def test_every_catalog_entry_has_a_profile_or_defaults():
    """No entry should silently rely on legacy hardcoded defaults.
    Either an explicit profile is set, or ``None`` (which falls back
    to DEFAULT_SAMPLING at load time)."""
    from rune.config import CATALOG, SamplingProfile

    for model_id, spec in CATALOG.items():
        # Must be SamplingProfile or None — never a dict or random thing.
        assert spec.sampling is None or isinstance(spec.sampling, SamplingProfile), (
            f"{model_id}: invalid sampling type {type(spec.sampling)}"
        )


def test_qwen25_models_share_canonical_profile():
    """All Qwen2.5-Instruct sizes use the same recommended profile
    (T=0.7, top_p=0.8, top_k=20, rep=1.05) per HuggingFace model card.

    Note: Qwen2.5-14B was removed from the CATALOG in v9 (28 GB VRAM
    requirement exceeds typical pod hardware). Only 3B and 7B remain.
    """
    from rune.config import CATALOG

    for size in ("3B", "7B"):
        model_id = f"Qwen/Qwen2.5-{size}-Instruct"
        spec = CATALOG[model_id]
        assert spec.sampling is not None, f"{model_id} missing profile"
        assert spec.sampling.temperature == pytest.approx(0.7)
        assert spec.sampling.top_p == pytest.approx(0.8)
        assert spec.sampling.top_k == 20
        assert spec.sampling.repetition_penalty == pytest.approx(1.05)


def test_thinking_models_use_lower_temperature():
    """Qwen3-thinking variants recommend T=0.6 to keep the <think>
    trace focused. Pinning this so we don't drift.

    Note: DeepSeek-R1 distills and Jamba-Reasoning-3B were removed
    from the CATALOG in v9 (R1: redundant with Qwen3, Jamba: runtime
    blocked by mamba-ssm/CUDA mismatch — see BACKLOG.md).
    """
    from rune.config import CATALOG

    thinking_ids = [
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-4B",
    ]
    for model_id in thinking_ids:
        spec = CATALOG[model_id]
        assert spec.sampling is not None
        assert spec.sampling.temperature == pytest.approx(0.6), (
            f"{model_id}: thinking models should run at T=0.6"
        )


def test_lfm2_uses_min_p_not_top_p():
    """Liquid AI's LFM2 family is documented to use min_p as its
    nucleus filter, not top_p.

    Note: LFM2.5-1.2B and LFM2-8B-A1B were removed from the CATALOG
    in v9 (never tested in prod, dispensable). Only LFM2-2.6B remains
    as the Liquid architectural representative.
    """
    from rune.config import CATALOG

    spec = CATALOG["LiquidAI/LFM2-2.6B"]
    assert spec.sampling is not None
    assert spec.sampling.top_p is None, "LFM2-2.6B should not use top_p"
    assert spec.sampling.min_p == pytest.approx(0.15)
    # Low temperature is intrinsic to the Liquid architecture.
    assert spec.sampling.temperature <= 0.4


def test_v9_catalog_inventory():
    """V9 catalogue refactor — verify the 8 expected entries are present
    and the 5 removed entries are gone.

    The refactor was driven by empirical findings from the v8 sprint:
    - Mistral structurally fails rules 9+10 (4 sessions of evidence)
    - Qwen2.5-14B too big for 22.8 GB pods
    - DeepSeek-R1 distills redundant with Qwen3 family
    - LFM2.5/LFM2-8B never tested in prod, dispensable
    - Jamba runtime blocked (see BACKLOG)

    New entries: Phi-4-mini, SmolLM3-3B (dual-mode), Qwen1.5-MoE.
    """
    from rune.config import CATALOG

    expected_present = [
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "microsoft/Phi-4-mini-instruct",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-8B",
        "HuggingFaceTB/SmolLM3-3B",
        "Qwen/Qwen1.5-MoE-A2.7B-Chat",
        "LiquidAI/LFM2-2.6B",
    ]
    for model_id in expected_present:
        assert model_id in CATALOG, f"{model_id} missing from CATALOG"

    # Modèles historiquement retirés qui ne doivent pas réapparaître.
    # (La liste v9 d'origine incluait aussi les distills DeepSeek-R1 et
    # Jamba ; le catalogue a depuis réintégré certains d'entre eux, donc
    # on ne teste plus que ceux qui restent définitivement absents.)
    expected_absent = [
        "mistralai/Mistral-7B-Instruct-v0.3",
        "Qwen/Qwen2.5-14B-Instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "LiquidAI/LFM2.5-1.2B-Instruct",
        "LiquidAI/LFM2-8B-A1B",
    ]
    for model_id in expected_absent:
        assert model_id not in CATALOG, (
            f"{model_id} should not be in CATALOG — see CHANGELOG"
        )

    # Invariant structurel : le catalogue n'est jamais vide et chaque
    # entrée a un ModelSpec valide (label + profil de sampling).
    assert len(CATALOG) > 0, "CATALOG ne doit pas être vide"
    for model_id, spec in CATALOG.items():
        assert spec.label, f"{model_id} a un label vide"
        assert spec.sampling is not None, f"{model_id} n'a pas de profil de sampling"



# ── Layer 2: Routes ────────────────────────────────────────────────────


def _make_request(profile=None):
    """Fake request whose ``app.state.lythea`` exposes a hippocampe
    with a sampling_profile attribute."""
    from rune.config import DEFAULT_SAMPLING

    if profile is None:
        profile = DEFAULT_SAMPLING

    hippocampe = SimpleNamespace(sampling_profile=profile)
    model = SimpleNamespace(is_loaded=True, model_id="test-model")
    lythea_app = SimpleNamespace(hippocampe=hippocampe, model=model)
    state = SimpleNamespace(lythea=lythea_app)
    fake_app = SimpleNamespace(state=state)
    return SimpleNamespace(app=fake_app)


def test_get_sampling_returns_profile_fields():
    """GET returns all profile fields plus the source model_id."""
    from rune.config import SamplingProfile
    from rune.server.routes import get_sampling

    profile = SamplingProfile(
        temperature=0.42, top_p=0.88, top_k=15, min_p=None,
        repetition_penalty=1.08, max_new_tokens=512,
    )
    request = _make_request(profile)
    result = asyncio.run(get_sampling(request))

    assert result["temperature"] == pytest.approx(0.42)
    assert result["top_p"] == pytest.approx(0.88)
    assert result["top_k"] == 15
    assert result["min_p"] is None
    assert result["repetition_penalty"] == pytest.approx(1.08)
    assert result["max_new_tokens"] == 512
    assert result["model_id"] == "test-model"


def test_post_sampling_partial_update():
    """POST with only ``temperature`` set must update only that field
    and leave the others alone."""
    from rune.config import SamplingProfile
    from rune.server.routes import config_sampling
    from rune.server.schemas import SamplingConfigRequest

    initial = SamplingProfile(
        temperature=0.7, top_p=0.9, top_k=20, repetition_penalty=1.0,
    )
    request = _make_request(initial)
    body = SamplingConfigRequest(temperature=0.3)  # only one field set
    result = asyncio.run(config_sampling(body, request))

    # Updated:
    assert result["temperature"] == pytest.approx(0.3)
    # Untouched:
    assert result["top_p"] == pytest.approx(0.9)
    assert result["top_k"] == 20
    assert result["repetition_penalty"] == pytest.approx(1.0)


def test_post_sampling_explicit_none_disables_top_p():
    """POSTing ``top_p: null`` must DISABLE top_p (set the field to
    None), not be ignored as an absent field."""
    from rune.config import SamplingProfile
    from rune.server.routes import config_sampling
    from rune.server.schemas import SamplingConfigRequest

    initial = SamplingProfile(
        temperature=0.7, top_p=0.9, top_k=None, repetition_penalty=1.0,
    )
    request = _make_request(initial)
    body = SamplingConfigRequest(top_p=None)
    # Pydantic strips defaults; we need to ensure the field is in fields_set.
    # Manually craft the body via __init__ with explicit None doesn't work
    # because the field default is also None. Use model_construct as escape:
    body = SamplingConfigRequest.model_validate({"top_p": None})
    result = asyncio.run(config_sampling(body, request))

    assert result["top_p"] is None
    # Other fields preserved.
    assert result["temperature"] == pytest.approx(0.7)


def test_post_sampling_validates_bounds():
    """The Pydantic schema must reject out-of-range values."""
    from pydantic import ValidationError
    from rune.server.schemas import SamplingConfigRequest

    with pytest.raises(ValidationError):
        SamplingConfigRequest(temperature=3.0)  # > 2.0
    with pytest.raises(ValidationError):
        SamplingConfigRequest(temperature=-0.1)
    with pytest.raises(ValidationError):
        SamplingConfigRequest(repetition_penalty=0.5)  # < 1.0
    with pytest.raises(ValidationError):
        SamplingConfigRequest(max_new_tokens=10)  # < 16


# ── Layer 3: Auto-application on load (integration sketch) ────────────

# A full integration test of /api/models/load is too heavy (it needs
# real torch + a real model download). The behaviour we care about
# is testable in isolation: given a model_id, the right profile is
# picked from the catalogue. See test_post_sampling_partial_update
# for the actual mutation logic. The wiring in routes.py::load_model
# is exercised end-to-end manually on the pod (see CHANGELOG fix #11).
