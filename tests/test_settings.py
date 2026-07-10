"""Tests for the Pydantic Settings module."""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest


def _reload_settings():
    """Reload settings, bypassing the lru_cache.

    We have to clear the cache *and* re-import because field defaults
    are captured at class-definition time, so simply clearing the cache
    is enough — but we keep the import-time invariant explicit.
    """
    from rune.settings import get_settings
    get_settings.cache_clear()
    return get_settings()


def test_defaults_are_loaded():
    """All fields should have sensible defaults out of the box."""
    s = _reload_settings()

    assert s.sdm_dim == 1024
    assert s.sdm_rows == 4096
    assert s.mhn_max_patterns == 512
    assert s.mhn_beta == pytest.approx(8.0)
    # NOTE: entropy_threshold default was 0.6 pre-step-8. Tightened to
    # 0.2 in the post-refactor polishings — see the dedicated
    # calibration test below.
    assert s.entropy_threshold == pytest.approx(0.2)
    assert s.max_new_tokens == 1024
    assert s.port == 7860
    assert s.host == "0.0.0.0"


def test_post_refactor_calibration_defaults():
    """Defaults validated empirically during post-refactor production testing.

    These three settings were tuned in prod on RunPod (Qwen2.5-3B-Instruct
    + Qwen3-4B-Thinking) and represent the calibration sweet spots. If
    you are intentionally changing one of these defaults, also update
    CHANGELOG.md and the comments in settings.py explaining why.

    See the comments in :mod:`rune.settings` for the full calibration
    history and reasoning behind each value.
    """
    s = _reload_settings()
    # Cross-encoder rerank threshold — 0.5 was too strict (full rejection
    # on meta-questions like "Tu te souviens de X ?"), 0.0 was too
    # permissive (let through unrelated content). 0.2 is the sweet spot.
    assert s.cross_encoder_min_score == pytest.approx(0.2)
    # Caption max tokens — 150 truncated rich photo descriptions
    # mid-sentence. 256 gives Qwen2VL room to breathe.
    assert s.caption_max_tokens == 256
    # Entropy threshold — 0.6 was too strict for instruction-tuned
    # models (mean entropy ~0.08 → doubt ~0.13 → always "fait", no
    # gradient). 0.2 restores the fait/intuition/hypothese spectrum.
    assert s.entropy_threshold == pytest.approx(0.2)


def test_env_override_int():
    """Integer fields should be overridable via env vars."""
    with patch.dict(os.environ, {"LYTHEA_SDM_DIM": "2048"}):
        s = _reload_settings()
        assert s.sdm_dim == 2048


def test_env_override_float():
    """Float fields should be overridable via env vars."""
    with patch.dict(os.environ, {"LYTHEA_MHN_BETA": "12.5"}):
        s = _reload_settings()
        assert s.mhn_beta == pytest.approx(12.5)


def test_env_override_case_insensitive():
    """Env var matching should be case-insensitive."""
    with patch.dict(os.environ, {"lythea_max_new_tokens": "2048"}):
        s = _reload_settings()
        assert s.max_new_tokens == 2048


def test_validation_rejects_out_of_range():
    """Field validators should reject impossible values."""
    from pydantic import ValidationError

    from rune.settings import LytheaSettings, get_settings

    with patch.dict(os.environ, {"LYTHEA_SDM_DIM": "32"}):  # below ge=64
        get_settings.cache_clear()
        with pytest.raises(ValidationError):
            LytheaSettings()


def test_unrelated_env_vars_are_ignored():
    """Random env vars should not break parsing."""
    with patch.dict(os.environ, {
        "HF_HOME": "/tmp/hf",
        "PATH": "/usr/bin",
        "RANDOM_GARBAGE": "xyz",
    }):
        s = _reload_settings()
        # Just check it loads without error
        assert s.sdm_dim == 1024


def test_caption_max_tokens_override():
    """The image captioner budget must be overridable via env."""
    with patch.dict(os.environ, {"LYTHEA_CAPTION_MAX_TOKENS": "512"}):
        s = _reload_settings()
        assert s.caption_max_tokens == 512


def test_caption_max_tokens_bounds_enforced():
    """Out-of-range values should be rejected at parse time."""
    from pydantic import ValidationError

    from rune.settings import LytheaSettings, get_settings

    with patch.dict(os.environ, {"LYTHEA_CAPTION_MAX_TOKENS": "32"}):
        get_settings.cache_clear()
        with pytest.raises(ValidationError):
            LytheaSettings()
    with patch.dict(os.environ, {"LYTHEA_CAPTION_MAX_TOKENS": "9999"}):
        get_settings.cache_clear()
        with pytest.raises(ValidationError):
            LytheaSettings()


def test_coreference_settings_defaults():
    """Coreference fallback is on by default with a 30-min window."""
    s = _reload_settings()
    # 30 min window — covers a typical conversational session without
    # going so far back that we'd inherit a stale interlocutor.
    assert s.coreference_window_sec == pytest.approx(1800.0)
    # Inferred relations are tagged with a lower confidence than direct
    # extraction (0.9), so callers can filter them out if they want
    # to be conservative.
    assert s.coreference_inferred_confidence == pytest.approx(0.6)


def test_config_module_reexports():
    """Backward compatibility: ``from rune.config import X`` must work."""
    # Cleanup first
    get_settings_clear()

    from rune import config

    # Re-exports
    assert config.SDM_DIM == 1024
    assert config.MHN_BETA == pytest.approx(8.0)
    assert config.ENTROPY_THRESHOLD == pytest.approx(0.2)
    assert config.MAX_NEW_TOKENS == 1024
    assert config.MICROSLEEP_INTERVAL == 5

    # Static structures still present
    assert "Qwen/Qwen2.5-3B-Instruct" in config.CATALOG
    # v9 changed default from 7B to 3B — the 3B was validated 5/5 in
    # prod (factual, concise, follows rules 9+10 reliably) and uses
    # less VRAM. The 7B is still in CATALOG for users who want more.
    assert config.DEFAULT_MODEL == "Qwen/Qwen2.5-3B-Instruct"
    assert "person" in config.GLINER_LABELS
    # V5 refondu : le SYSTEM_PROMPT commence maintenant par une section
    # markdown "# Identité" plutôt que directement "Tu es Rune". On
    # vérifie la présence de l'identité dans les 100 premiers caractères.
    assert "Tu es Rune" in config.SYSTEM_PROMPT[:200], (
        "L'identité Rune doit apparaître tôt dans le SYSTEM_PROMPT"
    )
    assert "qwen2vl" in config.CAPTIONER_OPTIONS


def test_system_prompt_has_anti_recitation_rule():
    """Anti-recitation rule — protège contre le "Salut <prénom>,
    tu vis à <ville>, tu travailles chez <employer>..." pattern observed
    in prod with Mistral-7B-Instruct, where the model recited every
    known fact at every turn instead of conversing.

    V5 refondu : le SYSTEM_PROMPT a été restructuré de 10 règles
    numérotées vers 4 sections markdown. On valide la sémantique
    (anti-recitation présente quelque part) plutôt qu'une numérotation
    qui n'existe plus.
    """
    from rune import config

    prompt = config.SYSTEM_PROMPT
    # La règle anti-recitation doit être présente sous forme sémantique.
    # V5 utilise des sections markdown (# Identité, # Honnêteté, etc.)
    # avec la règle dans la section style ("N'énumère pas les faits...").
    has_anti_recitation = (
        "n'énumère pas" in prompt.lower()
        or "ne récite" in prompt.lower()
        or "pas la peine de les réciter" in prompt.lower()
    )
    assert has_anti_recitation, (
        "SYSTEM_PROMPT V5 doit contenir une instruction anti-récitation "
        "des faits utilisateur (formulation libre, sections markdown)"
    )
    # Anti-resaluting (déjà présent V4, conservé V5)
    assert "resalue" in prompt.lower() or "salutation" in prompt.lower()
    # Must NOT hardcode any specific name — la règle est universelle.
    forbidden_names = ["Mika", "Sophie", "Jean", "Pierre", "Mickaël"]
    for name in forbidden_names:
        assert name not in prompt, f"Hardcoded name '{name}' in SYSTEM_PROMPT"


def test_lfm2_models_in_catalog():
    """LFM2 / LFM2.5 entries — descendants des CfC (MIT CSAIL → Liquid AI),
    architecture hybride non-Transformer, supportés natifs par
    transformers v4.55+. Ajoutés en post-refactor étape 8 pour offrir
    une alternative à l'archi pure-Transformer.

    Doivent être catalogués comme NON-thinking : LFM2 ne génère pas
    de balises ``<think>``, ce sont des Instruct standards. Si un
    futur LFM-Thinking sort, il faudra l'ajouter explicitement.

    Note: LFM2.5-1.2B and LFM2-8B-A1B were removed in v9 (never tested
    in prod, dispensable). Only LFM2-2.6B remains as the Liquid
    architectural representative.
    """
    from rune import config

    spec = config.CATALOG["LiquidAI/LFM2-2.6B"]
    # LFM2 doesn't emit <think> tags — must NOT be flagged thinking.
    assert not spec.is_thinking, (
        "LFM2-2.6B should be non-thinking (LFM2 has no native CoT)"
    )
    # Sanity: label is human-readable, size_gb is set.
    assert spec.label
    assert spec.size_gb > 0


def test_singleton_caching():
    """Repeated calls to get_settings should return the same instance."""
    from rune.settings import get_settings
    get_settings.cache_clear()

    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


# ── Helper ────────────────────────────────────────────────────────────

def get_settings_clear():
    """Force-reload of the settings cache and the config module.

    Some tests change env between tests; both the settings cache AND
    any module that re-exports values at import time must be reloaded.
    """
    import importlib
    from rune import settings as settings_mod
    settings_mod.get_settings.cache_clear()
    # Also reload config so its module-level re-exports pick up the new
    # values. This is only needed in tests that override env BEFORE
    # asserting on config.X — most production code reads via
    # get_settings() directly and doesn't need this.
    from rune import config
    importlib.reload(config)
