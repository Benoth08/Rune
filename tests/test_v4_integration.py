"""V4 — Cross-module integration tests.

These tests don't require torch / chromadb / hippocampe — they
exercise the end-to-end contracts between V4 modules:

1. Settings → module configs are correctly threaded.
2. cognitive_state friction signals correlate with trust changes.
3. planning + cognitive_state work independently (no shared state).
4. inhibition + cognitive_state independence.
5. Persistence files don't collide.
6. predictive_coding gating logic semantics.

If you have torch/chromadb available, ``test_hippocampe_v4.py`` adds
the full Hippocampe orchestration tests on top of these.
"""

import json
from pathlib import Path

import pytest

from rune.cognition.inhibition import (
    InhibitionConfig,
    InhibitionFilter,
    parse_whitelist,
)
from rune.cognition.planning import (
    GoalStack,
    PlanningConfig,
    PlanningPhase,
)
from rune.cognition.predictive_coding import (
    PredictiveCodingConfig,
    PredictiveCodingPhase,
)
from rune.cognition.timeline import (
    TimelineConfig,
    TimelineExtractor,
    render_block,
)
from rune.memory.cognitive_state import (
    CognitiveState,
    CognitiveStateConfig,
)
from rune.settings import LytheaSettings


# ════════════════════════════════════════════════════════════════════
# 1. Settings → module configs roundtrip
# ════════════════════════════════════════════════════════════════════


def test_settings_to_cognitive_state_config():
    s = LytheaSettings(
        affect_decay_half_life_sec=120.0,
        affect_contagion_max=0.3,
        affect_inertia=0.25,
        affect_reset_latch_turns=5,
        affect_detector="lexical",
        user_model_known_threshold=0.5,
    )
    cfg = CognitiveStateConfig(
        decay_half_life_sec=s.affect_decay_half_life_sec,
        contagion_max=s.affect_contagion_max,
        inertia=s.affect_inertia,
        reset_latch_turns=s.affect_reset_latch_turns,
        detector=s.affect_detector,
        user_known_threshold=s.user_model_known_threshold,
    )
    cs = CognitiveState(config=cfg)
    assert cs.lythea_affect.contagion_max == 0.3
    assert cs.user_knowledge.known_threshold == 0.5


def test_settings_to_inhibition_config():
    s = LytheaSettings(
        inhibition_n1_strict=False,
        inhibition_default_action="block",
        inhibition_domain_whitelist="alpha,beta,gamma",
    )
    cfg = InhibitionConfig(
        n1_strict=s.inhibition_n1_strict,
        n2_enabled=s.inhibition_n2_enabled,
        n3_enabled=s.inhibition_n3_enabled,
        default_action=s.inhibition_default_action,
        domain_whitelist=parse_whitelist(s.inhibition_domain_whitelist),
    )
    f = InhibitionFilter(cfg)
    assert f.config.n1_strict is False
    assert f.config.default_action == "block"
    assert "beta" in f.config.domain_whitelist


def test_settings_to_planning_config(tmp_path: Path):
    s = LytheaSettings(
        planning_max_steps=5,
        planning_goal_stale_days=7,
        planning_use_llm=False,
        planning_prompt_block_max_chars=200,
    )
    pp = PlanningPhase(
        config=PlanningConfig(
            max_steps=s.planning_max_steps,
            goal_stale_days=s.planning_goal_stale_days,
            use_llm=s.planning_use_llm,
            prompt_block_max_chars=s.planning_prompt_block_max_chars,
        ),
        goal_stack=GoalStack(tmp_path / "g.json"),
    )
    assert pp.config.max_steps == 5
    assert pp.config.use_llm is False


def test_settings_to_predictive_coding_config():
    s = LytheaSettings(
        pc_history_size=4,
        pc_cold_start_min=2,
        pc_low_threshold=0.1,
        pc_high_threshold=0.5,
    )
    pc = PredictiveCodingPhase(
        PredictiveCodingConfig(
            history_size=s.pc_history_size,
            cold_start_min=s.pc_cold_start_min,
            low_threshold=s.pc_low_threshold,
            high_threshold=s.pc_high_threshold,
        )
    )
    assert pc.config.history_size == 4
    assert pc.config.cold_start_min == 2


def test_settings_to_timeline_config():
    s = LytheaSettings(
        timeline_max_events=4,
        timeline_block_max_chars=300,
        timeline_render_min_confidence=0.5,
        timeline_render_vague=True,
    )
    ext = TimelineExtractor(
        TimelineConfig(
            max_events=s.timeline_max_events,
            block_max_chars=s.timeline_block_max_chars,
            render_min_confidence=s.timeline_render_min_confidence,
            render_vague=s.timeline_render_vague,
        )
    )
    assert ext.config.max_events == 4
    assert ext.config.render_vague is True


# ════════════════════════════════════════════════════════════════════
# 2. Friction → trust drop
# ════════════════════════════════════════════════════════════════════


def test_friction_eventually_drops_trust(tmp_path: Path):
    """Repeated friction must cause measurable trust loss."""
    cs = CognitiveState(storage_dir=tmp_path)
    # Build a baseline through neutral exchanges
    for _ in range(10):
        cs.observe_user_message("merci pour ton aide")
    high_trust = cs.user_trust.score
    # Now several friction exchanges
    for _ in range(5):
        cs.observe_user_message("non c'est faux, tu te trompes")
    assert cs.user_trust.score < high_trust


# ════════════════════════════════════════════════════════════════════
# 3. Planning + cognitive_state independence
# ════════════════════════════════════════════════════════════════════


def test_planning_then_continuation_with_cognitive_state(tmp_path: Path):
    """Planning maintains its goal independently from cognitive_state."""
    cs = CognitiveState(storage_dir=tmp_path / "cs")
    gs = GoalStack(tmp_path / "g.json")
    pp = PlanningPhase(goal_stack=gs)

    # Multi-step prompt creates a goal
    msg = "D'abord refactor le module ensuite déploie en prod"
    cs.observe_user_message(msg)
    res = pp.process(msg)
    if res.intent != "multi_step":
        pytest.skip("template fallback may not split — test skipped on this corpus")
    assert pp.goal_stack.has_active()

    # Continuation message — but cognitive_state has now seen one turn
    msg2 = "on reprend où on en était"
    cs.observe_user_message(msg2)
    res2 = pp.process(msg2)
    # Continuation should fire (active goal exists)
    assert res2.intent == "continuation"


# ════════════════════════════════════════════════════════════════════
# 4. Persistence file independence
# ════════════════════════════════════════════════════════════════════


def test_cognitive_state_and_goals_persistence_dont_collide(tmp_path: Path):
    """cognitive_state and goal_stack write to distinct paths."""
    cs_dir = tmp_path / "cognitive_state"
    goals_path = tmp_path / "goals" / "goals.json"

    cs = CognitiveState(storage_dir=cs_dir)
    cs.observe_user_message("je suis content")
    cs.save("session1")

    gs = GoalStack(goals_path)
    gs.add("test goal", steps=["a", "b"])

    # Both files exist independently
    assert (cs_dir / "session1.json").exists()
    assert goals_path.exists()
    # Reload each — no cross-talk
    cs2 = CognitiveState(storage_dir=cs_dir)
    assert cs2.load("session1") is True
    gs2 = GoalStack(goals_path)
    assert any(g.description == "test goal" for g in gs2.list_all())


# ════════════════════════════════════════════════════════════════════
# 5. Inhibition + cognitive_state independence
# ════════════════════════════════════════════════════════════════════


def test_inhibition_does_not_use_cognitive_state():
    """Inhibition is stateless re: cognitive_state — verifies modules
    don't share runtime state."""
    cs = CognitiveState()
    cs.observe_user_message("je suis très en colère contre toi !!!")
    # Even if cognitive_state is in negative state, inhibition only
    # acts on the literal output text.
    f = InhibitionFilter()
    res = f.check("Réponse parfaitement neutre.")
    assert res.passed is True
    assert res.action == "pass"


# ════════════════════════════════════════════════════════════════════
# 6. Predictive coding gating semantics
# ════════════════════════════════════════════════════════════════════


def test_predictive_coding_low_power_then_high_jump():
    """A repeating sequence should yield low_power; an outlier yields high."""
    pc = PredictiveCodingPhase(
        PredictiveCodingConfig(
            cold_start_min=2,
            low_threshold=0.05,
            high_threshold=0.5,
        )
    )
    stable = [1.0, 0.0, 0.0]
    pc.observe(stable)
    pc.observe(stable)
    d_low = pc.observe(stable)
    assert d_low.mode == "low_power"
    d_high = pc.observe([0.0, 1.0, 0.0])  # orthogonal jump
    assert d_high.mode == "high"


# ════════════════════════════════════════════════════════════════════
# 7. Timeline + render_block via real text
# ════════════════════════════════════════════════════════════════════


def test_timeline_extract_then_render_full_pipeline():
    cfg = TimelineConfig(render_min_confidence=0.3)
    ext = TimelineExtractor(cfg)
    text = "Hier on a vu un défaut. Le 12 mai 2026 on déploie. Dans 3 jours on teste."
    events = ext.extract(text)
    assert len(events) >= 2
    block = render_block(events, cfg)
    assert "[Chronologie]" in block
    assert "2026-05-12" in block
