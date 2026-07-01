"""V4.0.c — Tests for planning module.

Sections
--------
1. IntentClassifier : chitchat / one_shot / multi_step / continuation,
   short messages, active goal context.
2. **Anti-over-planning** : critical sentinel — chitchat stays chitchat
   even with an active goal.
3. GoalStack : add / get_active / advance / set_status / archive_stale
   / persistence / corrupted file resilience.
4. extract_plan_json : 3 strategies, malformed inputs.
5. PlanGenerator : LLM mode, template fallback, malformed LLM output,
   max_steps cap.
6. PlanningPhase : full pipeline by intent, block rendering.
"""

import json
import time
from pathlib import Path

import pytest

from rune.cognition.planning import (
    Goal,
    GoalStack,
    IntentClassifier,
    IntentResult,
    PlanGenerator,
    PlanGeneratorConfig,
    PlanningConfig,
    PlanningPhase,
    PlanningResult,
    extract_plan_json,
)


# ════════════════════════════════════════════════════════════════════
# 1. IntentClassifier
# ════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "text",
    ["salut", "bonjour", "ok merci", "yo", "hey", "thanks", "ça va ?"],
)
def test_classify_chitchat(text):
    res = IntentClassifier().classify(text)
    assert res.intent == "chitchat"


@pytest.mark.parametrize(
    "text",
    [
        "Quelle est la capitale de la France ?",
        "Explique-moi la photosynthèse.",
        "Calcule 12 × 7 pour moi.",
    ],
)
def test_classify_one_shot(text):
    res = IntentClassifier().classify(text)
    assert res.intent == "one_shot"


@pytest.mark.parametrize(
    "text",
    [
        "D'abord, refactorise le module ensuite déploie en production",
        "Je veux construire une API complète, étape par étape, avec tests",
        "Build the dashboard step by step then deploy and finally test",
    ],
)
def test_classify_multi_step(text):
    res = IntentClassifier().classify(text)
    assert res.intent == "multi_step"


def test_classify_continuation_requires_active_goal():
    text = "on reprend où on en était"
    # No active goal → falls through (one_shot or chitchat)
    res = IntentClassifier().classify(text, has_active_goal=False)
    assert res.intent != "continuation"
    # Active goal → continuation fires
    res = IntentClassifier().classify(text, has_active_goal=True)
    assert res.intent == "continuation"


def test_classify_empty_is_chitchat():
    assert IntentClassifier().classify("").intent == "chitchat"
    assert IntentClassifier().classify("   ").intent == "chitchat"


def test_classify_returns_matched_markers_for_multi_step():
    res = IntentClassifier().classify(
        "D'abord développer puis déployer ensuite tester"
    )
    assert res.intent == "multi_step"
    assert len(res.matched_markers) >= 2


# ════════════════════════════════════════════════════════════════════
# 2. ANTI-OVER-PLANNING — critical sentinels
# ════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("text", ["salut", "ça va ?", "ok merci", "yo"])
def test_anti_over_planning_short_messages_never_multi_step(text):
    """SENTINEL: trivial openers must never trigger plan generation."""
    res = IntentClassifier().classify(text)
    assert res.intent != "multi_step"


def test_anti_over_planning_chitchat_with_active_goal_stays_chitchat():
    """SENTINEL: 'salut' with an active goal still classifies as chitchat,
    not continuation."""
    res = IntentClassifier().classify("salut", has_active_goal=True)
    assert res.intent == "chitchat"


def test_anti_over_planning_phase_chitchat_emits_no_block(tmp_path: Path):
    """SENTINEL: chitchat through the full phase must not produce a plan block."""
    pp = PlanningPhase(goal_stack=GoalStack(tmp_path / "g.json"))
    res = pp.process("salut")
    assert res.intent == "chitchat"
    assert res.prompt_block == ""
    assert res.is_new_goal is False
    assert res.active_goal is None


# ════════════════════════════════════════════════════════════════════
# 3. GoalStack
# ════════════════════════════════════════════════════════════════════


def test_goal_only_one_active(tmp_path: Path):
    gs = GoalStack(tmp_path / "g.json")
    g1 = gs.add("A", steps=["s1"])
    assert g1.status == "active"
    g2 = gs.add("B", steps=["s3"])
    assert g2.status == "active"
    # g1 should now be blocked (single-active invariant)
    g1_after = next(g for g in gs.list_all() if g.id == g1.id)
    assert g1_after.status == "blocked"


def test_goal_advance_step_progresses(tmp_path: Path):
    gs = GoalStack(tmp_path / "g.json")
    g = gs.add("test", steps=["s1", "s2", "s3"])
    assert g.current_step == 0
    gs.advance_step(g.id)
    g_after = next(g for g in gs.list_all() if g.id == g.id)
    assert g_after.current_step == 1


def test_goal_advance_to_done(tmp_path: Path):
    gs = GoalStack(tmp_path / "g.json")
    g = gs.add("test", steps=["s1", "s2"])
    gs.advance_step(g.id)  # → 1
    gs.advance_step(g.id)  # → 2 == len → done
    g_after = next(g for g in gs.list_all() if g.id == g.id)
    assert g_after.status == "done"


def test_goal_set_status(tmp_path: Path):
    gs = GoalStack(tmp_path / "g.json")
    g = gs.add("x", steps=["s"])
    gs.set_status(g.id, "abandoned")
    g_after = next(g for g in gs.list_all() if g.id == g.id)
    assert g_after.status == "abandoned"


def test_goal_archive_stale(tmp_path: Path):
    gs = GoalStack(tmp_path / "g.json")
    g = gs.add("old", steps=["s"])
    # Manually backdate
    g.updated_at = time.time() - 30 * 86400  # 30 days ago
    n = gs.archive_stale(stale_after_sec=14 * 86400)
    assert n == 1
    g_after = next(g for g in gs.list_all() if g.id == g.id)
    assert g_after.status == "abandoned"


def test_goal_persistence_roundtrip(tmp_path: Path):
    p = tmp_path / "g.json"
    gs1 = GoalStack(p)
    g = gs1.add("persistent", steps=["a", "b", "c"])
    gs1.advance_step(g.id)
    # New stack instance loads from disk
    gs2 = GoalStack(p)
    g_loaded = next(g for g in gs2.list_all() if g.id == g.id)
    assert g_loaded.description == "persistent"
    assert g_loaded.current_step == 1


def test_goal_stack_corrupted_file_no_crash(tmp_path: Path):
    p = tmp_path / "g.json"
    p.write_text("{ not valid json }", encoding="utf-8")
    gs = GoalStack(p)  # must not raise
    assert gs.list_all() == []


def test_goal_stack_atomic_no_tmp_leftover(tmp_path: Path):
    p = tmp_path / "g.json"
    gs = GoalStack(p)
    gs.add("x", steps=["s"])
    # No .tmp file left after successful save
    assert not (tmp_path / "g.json.tmp").exists()
    assert p.exists()


def test_goal_stack_clear(tmp_path: Path):
    gs = GoalStack(tmp_path / "g.json")
    gs.add("x", steps=["s"])
    gs.clear()
    assert gs.list_all() == []


def test_goal_stack_no_active_returns_none(tmp_path: Path):
    gs = GoalStack(tmp_path / "g.json")
    assert gs.get_active() is None
    assert gs.has_active() is False


# ════════════════════════════════════════════════════════════════════
# 4. extract_plan_json
# ════════════════════════════════════════════════════════════════════


def test_json_pure():
    raw = '{"description": "X", "steps": ["a", "b"]}'
    d = extract_plan_json(raw)
    assert d == {"description": "X", "steps": ["a", "b"]}


def test_json_fenced():
    raw = '''Here is the plan:
```json
{"description": "X", "steps": ["a", "b"]}
```
That's it.'''
    d = extract_plan_json(raw)
    assert d is not None
    assert d.get("steps") == ["a", "b"]


def test_json_with_extra_text():
    raw = 'Some preamble {"description": "X", "steps": ["a"]} and trailing text'
    d = extract_plan_json(raw)
    assert d is not None
    assert d.get("steps") == ["a"]


def test_json_invalid_returns_none():
    assert extract_plan_json("not json at all") is None
    assert extract_plan_json("") is None
    assert extract_plan_json(None) is None  # type: ignore[arg-type]


def test_json_dict_without_steps_returns_none():
    raw = '{"description": "no steps here"}'
    assert extract_plan_json(raw) is None


# ════════════════════════════════════════════════════════════════════
# 5. PlanGenerator
# ════════════════════════════════════════════════════════════════════


def test_plan_generator_no_llm_template_path():
    pg = PlanGenerator(config=PlanGeneratorConfig(use_llm=False))
    r = pg.generate("faire X puis faire Y puis faire Z")
    assert len(r["steps"]) >= 2


def test_plan_generator_llm_returns_valid_json():
    def fake_llm(prompt: str) -> str:
        return '```json\n{"description": "from LLM", "steps": ["a", "b"]}\n```'

    pg = PlanGenerator(llm_call=fake_llm)
    r = pg.generate("anything")
    assert r["description"] == "from LLM"
    assert r["steps"] == ["a", "b"]


def test_plan_generator_llm_malformed_falls_back_to_template():
    def bad_llm(prompt: str) -> str:
        return "I refuse to output JSON, sorry"

    pg = PlanGenerator(llm_call=bad_llm)
    # Request has multi-step structure → template can split
    r = pg.generate("faire A puis faire B puis faire C")
    assert len(r["steps"]) >= 2  # template fallback worked


def test_plan_generator_llm_crash_no_propagate():
    def crashing_llm(prompt: str) -> str:
        raise RuntimeError("LLM down")

    pg = PlanGenerator(llm_call=crashing_llm)
    # Must not raise — fall back to template
    r = pg.generate("faire A puis faire B")
    assert isinstance(r, dict)
    assert "steps" in r


def test_plan_generator_max_steps_capped():
    def big_llm(prompt: str) -> str:
        return json.dumps(
            {
                "description": "big",
                "steps": [f"step {i}" for i in range(20)],
            }
        )

    pg = PlanGenerator(llm_call=big_llm, config=PlanGeneratorConfig(max_steps=5))
    r = pg.generate("anything")
    assert len(r["steps"]) == 5


def test_plan_generator_empty_request():
    pg = PlanGenerator()
    r = pg.generate("")
    assert r == {"description": "", "steps": []}


# ════════════════════════════════════════════════════════════════════
# 6. PlanningPhase end-to-end
# ════════════════════════════════════════════════════════════════════


def test_phase_chitchat_no_goal(tmp_path: Path):
    pp = PlanningPhase(goal_stack=GoalStack(tmp_path / "g.json"))
    res = pp.process("salut")
    assert res.intent == "chitchat"
    assert res.active_goal is None
    assert res.prompt_block == ""


def test_phase_one_shot_no_goal(tmp_path: Path):
    pp = PlanningPhase(goal_stack=GoalStack(tmp_path / "g.json"))
    res = pp.process("Combien font 12 fois 7 ?")
    assert res.intent == "one_shot"
    assert res.active_goal is None
    assert res.prompt_block == ""


def test_phase_multi_step_creates_goal(tmp_path: Path):
    def llm(prompt: str) -> str:
        return json.dumps(
            {"description": "Refactor + deploy", "steps": ["s1", "s2", "s3"]}
        )

    gs = GoalStack(tmp_path / "g.json")
    pg = PlanGenerator(llm_call=llm)
    pp = PlanningPhase(goal_stack=gs, plan_generator=pg)
    res = pp.process("D'abord refactor puis déploie ensuite teste tout")
    assert res.intent == "multi_step"
    assert res.is_new_goal is True
    assert res.active_goal is not None
    assert "But:" in res.prompt_block


def test_phase_continuation_surfaces_existing_goal(tmp_path: Path):
    gs = GoalStack(tmp_path / "g.json")
    g = gs.add("Existing", steps=["a", "b"])
    pp = PlanningPhase(goal_stack=gs)
    res = pp.process("on reprend là où on en était")
    assert res.intent == "continuation"
    assert res.is_new_goal is False
    assert res.active_goal is not None
    assert res.active_goal.id == g.id


def test_phase_block_renders_check_arrow_dot(tmp_path: Path):
    gs = GoalStack(tmp_path / "g.json")
    g = gs.add("test", steps=["s1", "s2", "s3"])
    gs.advance_step(g.id)  # current = 1
    pp = PlanningPhase(goal_stack=gs)
    block = pp._render_block(gs.get_active())
    assert "✓ s1" in block
    assert "→ s2" in block
    assert "· s3" in block


def test_phase_block_respects_max_chars(tmp_path: Path):
    gs = GoalStack(tmp_path / "g.json")
    g = gs.add("X" * 200, steps=["step " + "Y" * 100 for _ in range(7)])
    pp = PlanningPhase(
        config=PlanningConfig(prompt_block_max_chars=120),
        goal_stack=gs,
    )
    block = pp._render_block(gs.get_active())
    assert len(block) <= 120
    assert block.endswith("…")


def test_phase_persistence_across_instances(tmp_path: Path):
    p = tmp_path / "g.json"
    pp1 = PlanningPhase(goal_stack=GoalStack(p))
    res1 = pp1.process("on reprend")  # no goal yet → not continuation
    assert res1.intent != "continuation"

    # Force-create a goal via direct add
    pp1.goal_stack.add("persistent goal", steps=["a", "b"])

    # New PlanningPhase instance, same disk file
    pp2 = PlanningPhase(goal_stack=GoalStack(p))
    res2 = pp2.process("on reprend")
    assert res2.intent == "continuation"
    assert res2.active_goal is not None
    assert res2.active_goal.description == "persistent goal"


def test_phase_multi_step_failure_degrades_gracefully(tmp_path: Path):
    """If plan_generator returns no steps, intent degrades to one_shot."""

    def empty_llm(prompt: str) -> str:
        return '{"description": "", "steps": []}'

    pp = PlanningPhase(
        goal_stack=GoalStack(tmp_path / "g.json"),
        plan_generator=PlanGenerator(llm_call=empty_llm),
    )
    res = pp.process("D'abord refactor puis déploie ensuite teste")
    assert res.intent == "one_shot"  # degraded
    assert res.active_goal is None


def test_phase_handles_internal_crash():
    """Defensive: if anything inside crashes, return neutral PlanningResult."""

    class BrokenStack:
        def archive_stale(self, *a, **kw):
            return 0

        def get_active(self):
            raise RuntimeError("disk error")

        def add(self, *a, **kw):
            raise RuntimeError("disk error")

    pp = PlanningPhase()
    pp.goal_stack = BrokenStack()  # type: ignore[assignment]
    res = pp.process("salut")
    # Should not raise; returns neutral default
    assert isinstance(res, PlanningResult)


# ════════════════════════════════════════════════════════════════════
# 7. Integration: classifier + stack invariant
# ════════════════════════════════════════════════════════════════════


def test_continuation_requires_actually_active_goal(tmp_path: Path):
    pp = PlanningPhase(goal_stack=GoalStack(tmp_path / "g.json"))
    # No goal at all
    res = pp.process("on reprend où on en était")
    # Without goal, "on reprend" is not continuation; should be one_shot or chitchat
    assert res.intent != "continuation"
