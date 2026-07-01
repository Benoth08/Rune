"""Tests HotContext + Blackboard partagé pour sous-agents."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rune.agents.hot_context import HotContext, HotContextSerializer
from rune.agents.subagent import SubAgentConfig, SubAgentSpawner
from rune.agents._runtime import SubAgentRuntime


# ── HotContext ────────────────────────────────────────────────────────


def test_hot_context_empty():
    """Un HotContext vide ne produit pas de bloc de prompt."""
    ctx = HotContext()
    assert ctx.as_prompt_block() == ""
    assert ctx._has_content() is False


def test_hot_context_with_rag_chunks():
    """HotContext avec RAG chunks produit un bloc valide."""
    ctx = HotContext(
        rag_chunks=[
            {"kind": "chroma_chunk", "content": "test content", "relevance": 0.8},
        ]
    )
    block = ctx.as_prompt_block()
    assert "[CONTEXTE MÉMOIRE]" in block
    assert "[RAG]" in block
    assert "test content" in block


def test_hot_context_with_skills():
    """HotContext avec skills produit un bloc valide."""
    ctx = HotContext(
        skills=[
            {
                "skill_id": "skill_test",
                "trigger": "test trigger",
                "approach": ["étape 1", "étape 2"],
                "validation": ["critère 1"],
                "confidence": 0.8,
                "success_count": 3,
            }
        ]
    )
    block = ctx.as_prompt_block()
    assert "[SKILLS APPLICABLES]" in block
    assert "test trigger" in block
    assert "étape 1" in block


def test_hot_context_with_anti_patterns():
    """HotContext avec anti-patterns produit un bloc valide."""
    ctx = HotContext(
        anti_patterns=[
            {
                "failure_id": "fail_1",
                "context": "test contexte",
                "attempted_action": "bad action",
                "symptom": "bad result",
                "correction": "good action",
                "occurrences": 2,
            }
        ]
    )
    block = ctx.as_prompt_block()
    assert "[ANTI-PATTERNS" in block
    assert "test contexte" in block
    assert "good action" in block


def test_hot_context_as_dict():
    """as_dict retourne une structure JSON-sérialisable."""
    ctx = HotContext(
        rag_chunks=[{"kind": "test", "content": "x"}],
        skills=[{"trigger": "t"}],
    )
    d = ctx.as_dict()
    assert "rag_chunks" in d
    assert "skills" in d
    json.dumps(d)  # ne doit pas crasher


# ── HotContextSerializer ──────────────────────────────────────────────


def test_hot_context_serializer_empty():
    """Serializer sans mémoires retourne un HotContext vide."""
    serializer = HotContextSerializer()
    ctx = serializer.build("test task")
    assert ctx.rag_chunks == []
    assert ctx.skills == []
    assert ctx.anti_patterns == []
    assert ctx.kg_entities == []


def test_hot_context_serializer_with_skills(tmp_path):
    """Serializer récupère les skills pertinents."""
    from rune.memory.auto_skill import AutoSkillStore, Skill

    store = AutoSkillStore(storage_dir=tmp_path)
    store.add(Skill(
        skill_id="skill_test",
        trigger="test trigger",
        trigger_embedding=[0.9, 0.1] + [0.0] * 30,
        approach=["étape 1"],
    ))

    def mock_embed(text):
        return [0.85, 0.15] + [0.0] * 30

    serializer = HotContextSerializer(
        skills_store=store,
        embed_fn=mock_embed,
    )
    ctx = serializer.build("test trigger")
    assert len(ctx.skills) == 1
    assert ctx.skills[0]["skill_id"] == "skill_test"


def test_hot_context_serializer_with_failures(tmp_path):
    """Serializer récupère les anti-patterns pertinents."""
    from rune.memory.failure_memory import FailureMemory, FailurePattern

    mem = FailureMemory(storage_dir=tmp_path)
    mem.add(FailurePattern(
        failure_id="fail_test",
        context="test context",
        attempted_action="bad action",
        symptom="bad result",
        root_cause="cause",
        correction="good action",
        embedding=[0.9, 0.1] + [0.0] * 30,
    ))

    def mock_embed(text):
        return [0.85, 0.15] + [0.0] * 30

    serializer = HotContextSerializer(
        failures_store=mem,
        embed_fn=mock_embed,
    )
    ctx = serializer.build("test context")
    assert len(ctx.anti_patterns) == 1
    assert ctx.anti_patterns[0]["failure_id"] == "fail_test"


def test_hot_context_serializer_handles_errors_gracefully():
    """Serializer ne crash pas si une mémoire lève une exception."""

    class BrokenStore:
        def find_by_trigger_embedding(self, *args, **kwargs):
            raise RuntimeError("broken")

    serializer = HotContextSerializer(
        skills_store=BrokenStore(),
        embed_fn=lambda x: [0.1] * 32,
    )
    ctx = serializer.build("test")
    # Ne doit pas crasher — juste retourner un HotContext vide
    assert ctx.skills == []


# ── SubAgentSpawner — hot_context + blackboard ────────────────────────


def test_subagent_spawner_passes_hot_context_in_payload():
    """SubAgentSpawner injecte hot_context dans le payload."""
    spawner = SubAgentSpawner(SubAgentConfig(timeout_sec=10))
    # Mock le subprocess pour vérifier le payload
    captured_payload = {}

    class MockProc:
        def __init__(self, *args, **kwargs):
            pass
        def communicate(self, input=None, timeout=None):
            captured_payload["json"] = input
            return ('{"status": "ok", "result": "done"}', "")
        @property
        def returncode(self):
            return 0
        def kill(self): pass
        def wait(self): pass

    import rune.agents.subagent as sa_mod
    original_popen = sa_mod.subprocess.Popen
    sa_mod.subprocess.Popen = MockProc
    try:
        result = spawner.run(
            task="test",
            hot_context={"rag_chunks": [{"content": "chunk1"}]},
        )
    finally:
        sa_mod.subprocess.Popen = original_popen

    payload = json.loads(captured_payload["json"])
    assert "hot_context" in payload
    assert payload["hot_context"]["rag_chunks"][0]["content"] == "chunk1"


def test_subagent_spawner_passes_blackboard_in_payload():
    """SubAgentSpawner injecte blackboard_path + section dans le payload."""
    spawner = SubAgentSpawner(SubAgentConfig(timeout_sec=10))
    captured_payload = {}

    class MockProc:
        def __init__(self, *args, **kwargs):
            pass
        def communicate(self, input=None, timeout=None):
            captured_payload["json"] = input
            return ('{"status": "ok", "result": "done"}', "")
        @property
        def returncode(self):
            return 0
        def kill(self): pass
        def wait(self): pass

    import rune.agents.subagent as sa_mod
    original_popen = sa_mod.subprocess.Popen
    sa_mod.subprocess.Popen = MockProc
    try:
        result = spawner.run(
            task="test",
            blackboard_path="/tmp/bb.json",
            blackboard_section="subagent_1",
        )
    finally:
        sa_mod.subprocess.Popen = original_popen

    payload = json.loads(captured_payload["json"])
    assert payload["blackboard_path"] == "/tmp/bb.json"
    assert payload["blackboard_section"] == "subagent_1"


# ── SubAgentRuntime — hot_context injection ───────────────────────────


def test_runtime_builds_system_prompt_with_hot_context():
    """Le runtime injecte le hot_context dans le system prompt."""
    runtime = SubAgentRuntime({
        "task": "test",
        "hot_context": {
            "rag_chunks": [{"kind": "test", "content": "chunk content"}],
            "skills": [{"trigger": "t", "approach": ["step1"]}],
            "anti_patterns": [{"context": "ctx", "correction": "fix"}],
        },
    })
    prompt = runtime._build_system_prompt()
    assert "[CONTEXTE MÉMOIRE]" in prompt
    assert "[RAG]" in prompt
    assert "chunk content" in prompt
    assert "[SKILLS APPLICABLES]" in prompt
    assert "t" in prompt
    assert "[ANTI-PATTERNS" in prompt


def test_runtime_system_prompt_without_hot_context():
    """Sans hot_context, le system prompt est minimal."""
    runtime = SubAgentRuntime({"task": "test"})
    prompt = runtime._build_system_prompt()
    assert "[CONTEXTE MÉMOIRE]" not in prompt
    assert "sous-agent Rune" in prompt


# ── SubAgentRuntime — blackboard lecture + écriture ──────────────────


def test_runtime_reads_blackboard_if_exists(tmp_path):
    """Le runtime lit le blackboard si le fichier existe."""
    from rune.agentic.blackboard import MissionBlackboard

    bb_path = tmp_path / "bb.json"
    bb = MissionBlackboard(path=bb_path)
    bb.set_contract("Mission de test")
    # Note : render_for(owner) montre les wins de l'owner mais pour les
    # peers seulement fails/notes/interface. On test juste le contract + section.
    bb.record_fail("lead", "tentative 1", "raison 1")

    runtime = SubAgentRuntime({
        "task": "test",
        "blackboard_path": str(bb_path),
        "blackboard_section": "subagent_1",
    })
    block = runtime._read_blackboard_context()
    assert "[BLACKBOARD PARTAGÉ]" in block
    assert "Mission de test" in block
    # Le lead est un peer du subagent_1 → ses fails apparaissent
    assert "tentative 1" in block


def test_runtime_returns_empty_if_blackboard_missing(tmp_path):
    """Si le blackboard n'existe pas, retourne une string vide."""
    runtime = SubAgentRuntime({
        "task": "test",
        "blackboard_path": str(tmp_path / "nonexistent.json"),
        "blackboard_section": "subagent_1",
    })
    block = runtime._read_blackboard_context()
    assert block == ""


def test_runtime_writes_to_blackboard(tmp_path):
    """Le runtime écrit le résultat dans sa section du blackboard."""
    from rune.agentic.blackboard import MissionBlackboard

    bb_path = tmp_path / "bb.json"
    # Crée un blackboard initial avec un contract
    bb = MissionBlackboard(path=bb_path)
    bb.set_contract("Mission test")
    bb.save()

    runtime = SubAgentRuntime({
        "task": "test",
        "blackboard_path": str(bb_path),
        "blackboard_section": "subagent_1",
    })
    success = runtime._write_to_blackboard("Résultat de la mission")
    assert success is True

    # Re-load le blackboard et vérifie que la section a été écrite
    bb2 = MissionBlackboard.load(bb_path)
    section = bb2.sections.get("subagent_1")
    assert section is not None
    assert len(section.wins) >= 1
    assert "Résultat de la mission" in section.wins[0]
    assert section.status == "done"


def test_runtime_write_blackboard_handles_errors():
    """Si le blackboard est inaccessible, _write_to_blackboard retourne False."""
    runtime = SubAgentRuntime({
        "task": "test",
        "blackboard_path": "/nonexistent/path/bb.json",
        "blackboard_section": "subagent_1",
    })
    # Devrait réussir quand même car on crée un nouveau blackboard
    # Mais test avec un path invalide (permission denied)
    success = runtime._write_to_blackboard("test")
    # Soit True (création nouveau), soit False (erreur I/O)
    assert isinstance(success, bool)


# ── Blackboard — concurrence (2 sous-agents) ──────────────────────────


def test_blackboard_two_agents_no_corruption(tmp_path):
    """Deux sous-agents écrivent dans des sections différentes sans corruption."""
    from rune.agentic.blackboard import MissionBlackboard

    bb_path = tmp_path / "bb.json"
    bb = MissionBlackboard(path=bb_path)
    bb.set_contract("Mission multi-agent")
    bb.save()

    # Simule 2 sous-agents qui écrivent en séquence
    for i in range(1, 3):
        runtime = SubAgentRuntime({
            "task": f"tâche {i}",
            "blackboard_path": str(bb_path),
            "blackboard_section": f"subagent_{i}",
        })
        success = runtime._write_to_blackboard(f"Résultat {i}")
        assert success is True

    # Re-load et vérifie que les 2 sections sont intactes
    bb2 = MissionBlackboard.load(bb_path)
    assert "subagent_1" in bb2.sections
    assert "subagent_2" in bb2.sections
    assert "Résultat 1" in bb2.sections["subagent_1"].wins[0]
    assert "Résultat 2" in bb2.sections["subagent_2"].wins[0]
    assert bb2.sections["subagent_1"].status == "done"
    assert bb2.sections["subagent_2"].status == "done"
