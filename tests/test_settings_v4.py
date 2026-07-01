"""V4 settings tests.

Validate:
1. Every V4 master flag (``enable_*``) is False by default →
   activating V4 requires explicit opt-in.
2. Sub-flag defaults are conservative (anti-sycophant cap on
   contagion, strict N1 inhibition, neutral starting trust, etc.).
3. Pydantic bounds reject aberrant values (negative half-life,
   contagion >1, max_steps=0).
4. Environment overrides work via the LYTHEA_ prefix.
5. **Sentinel**: a sample of V3 fields keep their V3 values, proving
   the V4 patch did not regress anything.

These tests intentionally use ``LytheaSettings(**overrides)`` rather
than ``get_settings()`` so the ``lru_cache`` does not pollute results.
"""

import os
from contextlib import contextmanager

import pytest
from pydantic import ValidationError

from rune.settings import LytheaSettings, get_settings


# ── Helpers ──────────────────────────────────────────────────────────


@contextmanager
def env(**overrides):
    """Temporarily set env vars, restore previous values on exit."""
    saved = {k: os.environ.get(k) for k in overrides}
    try:
        for k, v in overrides.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        get_settings.cache_clear()
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        get_settings.cache_clear()


# ── 1. All V4 enable_* flags are OFF by default ──────────────────────


@pytest.mark.parametrize(
    "flag",
    [
        "enable_cognitive_state",
        "enable_inhibition",
        "enable_planning",
        "enable_predictive_coding",
    ],
)
def test_v4_master_flag_off_by_default(flag):
    s = LytheaSettings()
    assert getattr(s, flag) is False, f"{flag} must default to False"


def test_v4_timeline_off_by_default():
    s = LytheaSettings()
    assert s.enable_timeline is False


def test_v41_affect_modulates_off_by_default():
    s = LytheaSettings()
    assert s.affect_modulates_consolidation is False


def test_v42_pc_apply_gating_off_by_default():
    s = LytheaSettings()
    assert s.pc_apply_gating is False


# ── 2. Conservative inhibition sub-defaults ──────────────────────────


def test_inhibition_n1_strict_by_default():
    s = LytheaSettings()
    assert s.inhibition_n1_strict is True


def test_inhibition_n2_disabled_by_default():
    s = LytheaSettings()
    assert s.inhibition_n2_enabled is False  # placeholder, no model bundled


def test_inhibition_n3_enabled_by_default():
    s = LytheaSettings()
    assert s.inhibition_n3_enabled is True  # cheap, no deps


def test_inhibition_default_action_is_annotate():
    s = LytheaSettings()
    assert s.inhibition_default_action == "annotate"


# ── 3. Affect: anti-sycophant defaults ───────────────────────────────


def test_affect_decay_300s():
    s = LytheaSettings()
    assert s.affect_decay_half_life_sec == 300.0


def test_affect_contagion_strictly_below_one():
    """ANTI-SYCOPHANT: contagion cap <1 prevents Lythéa from mirroring."""
    s = LytheaSettings()
    assert 0.0 <= s.affect_contagion_max < 1.0
    # Stronger: anything ≥ 0.5 starts to feel sycophantic
    assert s.affect_contagion_max <= 0.5


def test_affect_inertia_in_unit_interval():
    s = LytheaSettings()
    assert 0.0 <= s.affect_inertia <= 1.0


def test_affect_detector_default_lexical():
    s = LytheaSettings()
    assert s.affect_detector == "lexical"


# ── 4. Planning bounds ───────────────────────────────────────────────


def test_planning_max_steps_in_reasonable_range():
    s = LytheaSettings()
    assert 1 <= s.planning_max_steps <= 30
    assert s.planning_max_steps == 7  # cognitive sweet spot


def test_planning_block_cap_reasonable():
    s = LytheaSettings()
    assert 50 <= s.planning_prompt_block_max_chars <= 4000


# ── 5. Whitelist seeded with FR technical vocabulary ─────────────────


def test_inhibition_whitelist_contains_industrial_vocab():
    s = LytheaSettings()
    wl = s.inhibition_domain_whitelist.lower()
    for term in ("fissure", "défaut", "anomalie", "spectroscopie"):
        assert term in wl, f"{term!r} missing from default whitelist"


# ── 6. Env overrides via LYTHEA_ prefix ──────────────────────────────


def test_env_override_enable_planning():
    with env(LYTHEA_ENABLE_PLANNING="true"):
        s = get_settings()
        assert s.enable_planning is True


def test_env_override_affect_contagion():
    with env(LYTHEA_AFFECT_CONTAGION_MAX="0.25"):
        s = get_settings()
        assert s.affect_contagion_max == 0.25


def test_env_override_affect_detector():
    with env(LYTHEA_AFFECT_DETECTOR="llm"):
        s = get_settings()
        assert s.affect_detector == "llm"


# ── 7. Pydantic bounds reject aberrant values ────────────────────────


def test_reject_contagion_above_one():
    with pytest.raises(ValidationError):
        LytheaSettings(affect_contagion_max=1.5)


def test_reject_zero_or_negative_decay():
    with pytest.raises(ValidationError):
        LytheaSettings(affect_decay_half_life_sec=0.0)
    with pytest.raises(ValidationError):
        LytheaSettings(affect_decay_half_life_sec=-1.0)


def test_reject_zero_max_steps():
    with pytest.raises(ValidationError):
        LytheaSettings(planning_max_steps=0)


# ── 8. Sentinel — V3 fields unchanged by V4 patch ────────────────────


def test_v3_fields_still_present():
    """If this fails, the V4 patch silently broke a V3 setting.

    Sample of 9 V3 fields covering memory, microsleep, salience, web,
    cascade. Compare values, not just existence.
    """
    s = LytheaSettings()
    expected = {
        "sdm_dim": 1024,
        "mhn_beta": 8.0,
        "salience_min_score": 0.25,
        "ripple_surprise_threshold": 0.5,
        "ripple_boost_multiplier": 2.0,
        "web_max_rounds": 3,
        "cascade_synthesis_max_tokens": 120,
        "coreference_window_sec": 1800.0,
        "caption_max_tokens": 256,
    }
    for field, want in expected.items():
        assert hasattr(s, field), f"V3 field {field!r} disappeared"
        got = getattr(s, field)
        assert got == want, f"V3 field {field!r}: expected {want!r}, got {got!r}"


# ── 9. V4.1 microsleep modulation bounds ─────────────────────────────


def test_v41_arousal_threshold_in_unit():
    s = LytheaSettings()
    assert 0.0 <= s.affect_ripple_arousal_threshold <= 1.0


def test_v41_boost_factor_at_least_one():
    s = LytheaSettings()
    assert s.affect_consolidation_boost_factor >= 1.0


def test_v41_reject_boost_below_one():
    with pytest.raises(ValidationError):
        LytheaSettings(affect_consolidation_boost_factor=0.5)


# ── 10. V4.2 predictive coding bounds ────────────────────────────────


def test_v42_low_below_high_threshold():
    s = LytheaSettings()
    assert s.pc_low_threshold < s.pc_high_threshold


def test_v42_history_size_bounded():
    s = LytheaSettings()
    assert 1 <= s.pc_history_size <= 64


def test_v42_reject_oversized_history():
    with pytest.raises(ValidationError):
        LytheaSettings(pc_history_size=128)


# ── 11. V4.3 timeline bounds ─────────────────────────────────────────


def test_v43_max_events_bounded():
    s = LytheaSettings()
    assert 1 <= s.timeline_max_events <= 64


def test_v43_block_max_chars_bounded():
    s = LytheaSettings()
    assert 50 <= s.timeline_block_max_chars <= 4000
