"""V4.0.a — Tests for cognitive_state module.

Sections
--------
1. AffectVector : clamping, intensity, label grid, JSON roundtrip.
2. Lexical detection : FR / EN positive/negative, neutral, punctuation,
   technical vocabulary sentinel, ellipsis & CAPS edge cases.
3. AffectState : decay, empathic_update (contagion + compassion +
   intrinsic), inertia, reset latch, anti-sycophant cap.
4. UserKnowledgeState : EMA, threshold, normalization.
5. UserAffectiveState : EMA on both axes.
6. UserTrustState : gain on neutral, loss on friction, bounds, labels.
7. CognitiveState integration : observe + render blocks.
8. **Anti-sycophant** : critical tests on contagion cap.
9. **Friction** : explicit vs external negative affect.
10. Persistence : roundtrip, missing, corrupted, unknown detector.
11. reset_session_scoped : preserves knowledge + trust.
"""

import json
import time
from pathlib import Path

import pytest

from rune.memory.cognitive_state import (
    AffectState,
    AffectVector,
    CognitiveState,
    CognitiveStateConfig,
    UserAffectiveState,
    UserKnowledgeState,
    UserTrustState,
    _detect_lexical_affect,
)


# ════════════════════════════════════════════════════════════════════
# 1. AffectVector
# ════════════════════════════════════════════════════════════════════


def test_affect_vector_clamping_valence():
    v = AffectVector(valence=2.0)
    assert v.valence == 1.0
    v = AffectVector(valence=-3.0)
    assert v.valence == -1.0


def test_affect_vector_clamping_arousal():
    v = AffectVector(arousal=2.0)
    assert v.arousal == 1.0
    v = AffectVector(arousal=-0.5)
    assert v.arousal == 0.0


def test_affect_vector_clamping_confidence():
    v = AffectVector(confidence=2.5)
    assert v.confidence == 1.0
    v = AffectVector(confidence=-0.5)
    assert v.confidence == 0.0


def test_affect_vector_invalid_target_falls_back_to_world():
    v = AffectVector(target="bogus")
    assert v.target == "world"


def test_affect_vector_intensity_formula():
    # |v| × a
    v = AffectVector(valence=-0.8, arousal=0.5)
    assert abs(v.intensity - 0.4) < 1e-9
    v = AffectVector(valence=0.0, arousal=1.0)
    assert v.intensity == 0.0
    v = AffectVector(valence=-1.0, arousal=1.0)
    assert v.intensity == 1.0


@pytest.mark.parametrize(
    "valence,arousal,expected",
    [
        (-0.5, 0.1, "tristesse calme"),
        (-0.5, 0.5, "préoccupation"),
        (-0.5, 0.9, "détresse"),
        (0.0, 0.1, "neutre"),
        (0.5, 0.5, "intérêt"),
        (0.5, 0.9, "enthousiasme"),
    ],
)
def test_affect_vector_label_grid(valence, arousal, expected):
    assert AffectVector(valence=valence, arousal=arousal).label() == expected


def test_affect_vector_json_roundtrip():
    v = AffectVector(valence=0.3, arousal=0.7, target="user", confidence=0.5)
    rt = AffectVector.from_dict(v.to_dict())
    assert rt.valence == v.valence
    assert rt.arousal == v.arousal
    assert rt.target == v.target
    assert rt.confidence == v.confidence


def test_affect_vector_from_dict_handles_garbage():
    assert isinstance(AffectVector.from_dict("not a dict"), AffectVector)
    assert isinstance(AffectVector.from_dict(None), AffectVector)


# ════════════════════════════════════════════════════════════════════
# 2. Lexical detection
# ════════════════════════════════════════════════════════════════════


def test_lexical_detects_fr_negative():
    v = _detect_lexical_affect("je suis vraiment triste aujourd'hui")
    assert v.valence < 0
    assert v.confidence > 0.3


def test_lexical_detects_fr_positive_genial():
    v = _detect_lexical_affect("c'est génial !")
    assert v.valence > 0
    assert v.confidence > 0.3


def test_lexical_detects_fr_positive_merci():
    v = _detect_lexical_affect("merci beaucoup")
    assert v.valence > 0


def test_lexical_detects_fr_positive_parfait():
    v = _detect_lexical_affect("c'est parfait pour moi")
    assert v.valence > 0


def test_lexical_detects_en_positive_happy():
    v = _detect_lexical_affect("I am happy with this")
    assert v.valence > 0


def test_lexical_detects_en_positive_awesome():
    v = _detect_lexical_affect("this is awesome")
    assert v.valence > 0


def test_lexical_detects_en_negative_sad():
    v = _detect_lexical_affect("I'm so sad")
    assert v.valence < 0


def test_lexical_detects_en_negative_tired():
    v = _detect_lexical_affect("I'm exhausted and tired today")
    assert v.valence < 0


def test_lexical_neutral_text_low_confidence():
    v = _detect_lexical_affect("le tableau contient cinq colonnes")
    assert v.confidence < 0.3


def test_lexical_empty_text_zero_confidence():
    v = _detect_lexical_affect("")
    assert v.confidence == 0.0


def test_lexical_exclamation_amplifies_arousal():
    v_calm = _detect_lexical_affect("c'est génial")
    v_loud = _detect_lexical_affect("c'est génial !!!")
    assert v_loud.arousal > v_calm.arousal


def test_lexical_caps_with_excl_signals_anger_when_no_lexical():
    # No lexical hit but punctuation pattern → ambient anger
    v = _detect_lexical_affect("VOILÀ !")
    assert v.valence < 0


def test_lexical_ellipsis_without_lexical_hit():
    v = _detect_lexical_affect("je ne sais pas...")
    assert v.valence < 0
    assert v.arousal < 0.3


def test_lexical_multi_word_term_jen_ai_marre():
    v = _detect_lexical_affect("j'en ai marre de tout ça")
    assert v.valence < -0.5
    assert v.confidence > 0.3


# ── Critical: technical vocabulary must NOT trigger affect ──────────


@pytest.mark.parametrize(
    "text",
    [
        "j'ai détecté un défaut de fissure et une anomalie spectrale",
        "L'analyse spectroscopique révèle une corrosion industrielle",
        "Le défaut de surface est cohérent avec une anomalie de soudure",
        "spectroscopie infrarouge sur fissure de fatigue",
    ],
)
def test_lexical_does_not_detect_technical_terms(text):
    """SENTINEL: industrial vocabulary must not trigger negative affect.

    These terms are common in Mika's professional vocabulary
    (chemometrics, NDT, metrology). False positives here would make
    Lythéa "feel sad" every time the user discusses their work.
    """
    v = _detect_lexical_affect(text)
    # Either low confidence OR positive/neutral valence is acceptable.
    assert v.confidence < 0.3 or v.valence > -0.3, (
        f"Technical text {text!r} triggered affect: "
        f"v={v.valence:.3f}, conf={v.confidence:.3f}"
    )


# ════════════════════════════════════════════════════════════════════
# 3. AffectState
# ════════════════════════════════════════════════════════════════════


def test_affect_state_decay_halves_at_half_life():
    state = AffectState(decay_half_life_sec=10.0)
    state.current = AffectVector(
        valence=0.8,
        arousal=0.6,
        target="user",
        confidence=0.7,
        timestamp=time.time() - 10.0,  # exactly one half-life ago
    )
    state.decay_to(time.time())
    assert abs(state.current.valence - 0.4) < 0.05
    assert abs(state.current.arousal - 0.3) < 0.05


def test_affect_state_compassion_on_user_sad():
    """User valence < -0.3 with confidence > 0.3 → compassion target=user."""
    state = AffectState(inertia=0.0)  # snap, no smoothing for clarity
    user_sad = AffectVector(valence=-0.7, arousal=0.4, confidence=0.6)
    state.empathic_update(user_sad)
    # Compassion produces softened negative valence + target=user
    assert state.current.valence < 0
    assert state.current.target == "user"


def test_affect_state_contagion_capped():
    """ANTI-SYCOPHANT: even at user_conf=1, contagion ≤ contagion_max."""
    state = AffectState(contagion_max=0.4, inertia=0.0)
    user_max = AffectVector(valence=1.0, arousal=1.0, confidence=1.0)
    state.empathic_update(user_max)
    # Without compassion (positive valence), only contagion fires.
    # Cap is 0.4 → final valence ≤ 0.4 + small target-priority adjustments.
    assert state.current.valence < 0.5


def test_affect_state_inertia_smoothing():
    state = AffectState(inertia=0.5, contagion_max=1.0)  # uncapped for test
    state.current = AffectVector(valence=0.0, arousal=0.0, confidence=0.0)
    user = AffectVector(valence=0.8, arousal=0.8, confidence=1.0)
    state.empathic_update(user)
    # With inertia=0.5: new = 0.5·target + 0.5·current = 0.5·0.8 + 0 = 0.4
    assert 0.3 < state.current.valence < 0.5


def test_affect_state_reset_latch_after_n_quiet_turns():
    state = AffectState(reset_latch_turns=3, contagion_max=1.0)
    # First, push to a non-neutral state
    state.empathic_update(AffectVector(valence=0.7, arousal=0.5, confidence=0.5))
    assert state.current.valence > 0
    # Now N quiet turns (no signal)
    for _ in range(3):
        state.empathic_update(AffectVector(confidence=0.0))
    # State should have reset to neutral
    assert abs(state.current.valence) < 0.1


def test_affect_state_intrinsic_signal_full_weight():
    """Intrinsic curiosity should drive Lythéa's affect even with no user signal."""
    state = AffectState(inertia=0.0)
    intr = AffectVector(valence=0.5, arousal=0.7, target="topic", confidence=0.8)
    state.empathic_update(
        user_affect=AffectVector(confidence=0.0),
        intrinsic=intr,
    )
    assert state.current.valence > 0.3
    assert state.current.target == "topic"


# ════════════════════════════════════════════════════════════════════
# 4. UserKnowledgeState
# ════════════════════════════════════════════════════════════════════


def test_knowledge_observe_accumulates():
    k = UserKnowledgeState()
    for _ in range(10):
        k.observe("python", evidence_strength=0.9)
    assert k.is_known("python")


def test_knowledge_normalize_case():
    k = UserKnowledgeState()
    for _ in range(10):
        k.observe("PYTHON", 0.9)
    assert k.is_known("python")
    assert k.is_known("PyThOn")


def test_knowledge_ignores_empty():
    k = UserKnowledgeState()
    k.observe("", 0.9)
    k.observe("   ", 0.9)
    assert k.known_concepts() == []


def test_knowledge_known_concepts_sorted():
    k = UserKnowledgeState()
    for _ in range(10):
        k.observe("zeta", 0.9)
        k.observe("alpha", 0.9)
        k.observe("mu", 0.9)
    assert k.known_concepts() == ["alpha", "mu", "zeta"]


# ════════════════════════════════════════════════════════════════════
# 5. UserAffectiveState
# ════════════════════════════════════════════════════════════════════


def test_user_affective_ema_progresses():
    ua = UserAffectiveState(smoothing=0.4)
    ua.update(AffectVector(valence=0.8, arousal=0.6, confidence=0.5))
    first = ua.smoothed.valence
    ua.update(AffectVector(valence=0.8, arousal=0.6, confidence=0.5))
    second = ua.smoothed.valence
    # EMA pulls toward target → second > first
    assert second > first


# ════════════════════════════════════════════════════════════════════
# 6. UserTrustState
# ════════════════════════════════════════════════════════════════════


def test_trust_gains_on_neutral_exchange():
    t = UserTrustState(score=0.3)
    for _ in range(5):
        t.observe_exchange(0.0)
    assert t.score > 0.3


def test_trust_loses_on_friction():
    t = UserTrustState(score=0.5)
    for _ in range(3):
        t.observe_exchange(-0.7)
    assert t.score < 0.5


def test_trust_bounds_respected_over_many_iterations():
    t = UserTrustState()
    for _ in range(1000):
        t.observe_exchange(0.0)
    assert t.score <= 1.0
    for _ in range(1000):
        t.observe_exchange(-0.5)
    assert t.score >= 0.0


def test_trust_label_bands():
    assert UserTrustState(score=0.1).label() == "réservée"
    assert UserTrustState(score=0.3).label() == "modérée"
    assert UserTrustState(score=0.6).label() == "établie"
    assert UserTrustState(score=0.9).label() == "forte"


# ════════════════════════════════════════════════════════════════════
# 7. CognitiveState integration
# ════════════════════════════════════════════════════════════════════


def test_cognitive_state_observe_user_sad_drives_lythea_to_user_target():
    cs = CognitiveState()
    cs.observe_user_message("je suis vraiment triste, c'est horrible")
    # Compassion should fire → target=user, valence<0
    assert cs.lythea_affect.current.target == "user"
    assert cs.lythea_affect.current.valence < 0


def test_render_user_state_block_empty_at_start():
    cs = CognitiveState()
    block = cs.render_user_state_block()
    assert block == ""


def test_render_user_state_block_contains_familiarity_after_3_exchanges():
    cs = CognitiveState()
    for _ in range(3):
        cs.observe_user_message("hello")
    block = cs.render_user_state_block()
    assert "Familiarité" in block


def test_render_self_affect_block_empty_when_neutral():
    cs = CognitiveState()
    block = cs.render_self_affect_block()
    assert block == ""


def test_render_self_affect_block_contains_user_target_on_compassion():
    cs = CognitiveState()
    cs.observe_user_message("je suis vraiment triste, c'est horrible")
    block = cs.render_self_affect_block()
    # Block should mention "à propos de ton interlocuteur"
    assert "interlocuteur" in block


def test_render_blocks_respect_max_chars():
    cs = CognitiveState()
    for i in range(20):
        cs.observe_user_message(f"je connais le sujet_{i}_avec_un_nom_assez_long")
    block = cs.render_user_state_block(max_chars=80)
    assert len(block) <= 80


# ════════════════════════════════════════════════════════════════════
# 8. ANTI-SYCOPHANT — critical tests
# ════════════════════════════════════════════════════════════════════


def test_anti_sycophant_positive_burst_does_not_full_mirror():
    """5 enthusiastic messages must not push Lythéa to full positive mirror."""
    cs = CognitiveState(config=CognitiveStateConfig(contagion_max=0.4))
    for _ in range(5):
        cs.observe_user_message("c'est absolument parfait, génial, j'adore !!!")
    # Even after many bursts, valence is capped well below 1.0
    assert cs.lythea_affect.current.valence < 0.95, (
        f"Sycophantic mirror detected: v={cs.lythea_affect.current.valence:.3f}"
    )


def test_anti_sycophant_negative_burst_does_not_full_mirror():
    """5 angry messages must not push Lythéa to full negative mirror."""
    cs = CognitiveState(config=CognitiveStateConfig(contagion_max=0.4))
    for _ in range(5):
        cs.observe_user_message("c'est horrible, je suis furieux, putain merde !!!")
    # Compassion + capped contagion → Lythéa softens, doesn't fully match
    assert cs.lythea_affect.current.valence > -0.95, (
        f"Negative mirror detected: v={cs.lythea_affect.current.valence:.3f}"
    )


# ════════════════════════════════════════════════════════════════════
# 9. Friction detection
# ════════════════════════════════════════════════════════════════════


def test_friction_explicit_drops_trust():
    cs = CognitiveState()
    # Build trust first
    for _ in range(5):
        cs.observe_user_message("merci beaucoup")
    high = cs.user_trust.score
    # Explicit friction
    cs.observe_user_message("non c'est faux, tu te trompes complètement")
    assert cs.user_trust.score < high


def test_friction_external_sadness_does_not_drop_trust():
    """Sad message about an external topic must not erode trust in Lythéa."""
    cs = CognitiveState()
    for _ in range(5):
        cs.observe_user_message("merci")
    high = cs.user_trust.score
    # Sad about politics, not about Lythéa
    cs.observe_user_message("je suis triste à cause de la situation politique")
    # Trust should not drop because no 2nd-person marker
    assert cs.user_trust.score >= high - 0.01  # allow normal +0.02 gain


# ════════════════════════════════════════════════════════════════════
# 10. Persistence
# ════════════════════════════════════════════════════════════════════


def test_persistence_roundtrip(tmp_path: Path):
    cs = CognitiveState(storage_dir=tmp_path)
    cs.observe_user_message("je suis vraiment frustré par ce bug")
    cs.user_knowledge.observe("python", 0.9)
    cs.user_knowledge.observe("python", 0.9)
    cs.user_knowledge.observe("python", 0.9)
    saved = cs.save("session1")
    assert saved is True

    cs2 = CognitiveState(storage_dir=tmp_path)
    loaded = cs2.load("session1")
    assert loaded is True
    # Knowledge survived
    assert "python" in cs2.user_knowledge.mastery
    # Trust count survived
    assert cs2.user_trust.exchange_count == cs.user_trust.exchange_count


def test_persistence_missing_file_returns_false(tmp_path: Path):
    cs = CognitiveState(storage_dir=tmp_path)
    assert cs.load("nonexistent_session") is False


def test_persistence_corrupted_file_returns_false(tmp_path: Path):
    cs = CognitiveState(storage_dir=tmp_path)
    (tmp_path / "bad.json").write_text("{not valid json", encoding="utf-8")
    assert cs.load("bad") is False
    # State should remain at defaults after corrupted load
    assert cs.lythea_affect.current.valence == 0.0


def test_persistence_atomic_write_uses_tmp(tmp_path: Path):
    cs = CognitiveState(storage_dir=tmp_path)
    cs.save("atomic_test")
    target = tmp_path / "atomic_test.json"
    assert target.exists()
    # No leftover .tmp file
    assert not (tmp_path / "atomic_test.json.tmp").exists()


def test_persistence_unknown_detector_falls_back_to_lexical():
    cs = CognitiveState(config=CognitiveStateConfig(detector="bogus"))
    # Should not crash, should still detect via lexical fallback
    cs.observe_user_message("je suis triste")
    assert cs.user_affect.smoothed.valence < 0


# ════════════════════════════════════════════════════════════════════
# 11. reset_session_scoped
# ════════════════════════════════════════════════════════════════════


def test_reset_session_scoped_keeps_knowledge_and_trust():
    cs = CognitiveState()
    cs.observe_user_message("je suis fatigué")
    for _ in range(5):
        cs.user_knowledge.observe("python", 0.9)
    cs.user_trust.observe_exchange(0.0)
    cs.user_trust.observe_exchange(0.0)

    knowledge_before = dict(cs.user_knowledge.mastery)
    trust_count_before = cs.user_trust.exchange_count

    cs.reset_session_scoped()

    # Affect resets
    assert cs.lythea_affect.current.valence == 0.0
    # Knowledge + trust persist
    assert cs.user_knowledge.mastery == knowledge_before
    assert cs.user_trust.exchange_count == trust_count_before
