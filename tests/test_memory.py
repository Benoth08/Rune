"""Tests mémoire — WorkingMemory + TieredRetriever + AutoSkill + FailureMemory."""
from __future__ import annotations

import time

import pytest

from rune.memory.auto_skill import AutoSkillStore, Skill, SkillExtractor
from rune.memory.failure_memory import FailureAnalyzer, FailureMemory, FailurePattern
from rune.memory.tiered_retriever import TieredRetriever
from rune.memory.working_memory import WorkingMemoryBuffer, WorkingMemoryChunk


# ── WorkingMemoryBuffer ───────────────────────────────────────────────


def test_working_memory_basic_add_get():
    buf = WorkingMemoryBuffer(capacity=3)
    buf.add(WorkingMemoryChunk(kind="user_message", content="hello", relevance=0.9))
    chunks = buf.get()
    assert len(chunks) == 1
    assert chunks[0].content == "hello"


def test_working_memory_eviction_when_full():
    buf = WorkingMemoryBuffer(capacity=2)
    buf.add(WorkingMemoryChunk(kind="msg", content="a", relevance=0.5))
    buf.add(WorkingMemoryChunk(kind="msg", content="b", relevance=0.9))
    evicted = buf.add(WorkingMemoryChunk(kind="msg", content="c", relevance=0.3))
    # 'c' a la plus basse relevance → doit être évincé
    assert evicted is not None
    assert evicted.content == "c"
    chunks = buf.get()
    assert len(chunks) == 2
    contents = {c.content for c in chunks}
    assert contents == {"a", "b"}


def test_working_memory_clear():
    buf = WorkingMemoryBuffer()
    buf.add(WorkingMemoryChunk(kind="msg", content="x"))
    assert len(buf.get()) == 1
    buf.clear()
    assert len(buf.get()) == 0


def test_working_memory_freshness_decay():
    buf = WorkingMemoryBuffer(freshness_half_life_sec=0.1)  # 100ms
    chunk = WorkingMemoryChunk(kind="msg", content="old", relevance=0.5)
    buf.add(chunk)
    time.sleep(0.3)  # 3 half-lives → freshness ≈ 0.125
    chunks = buf.get()
    assert chunks[0].freshness < 0.2


def test_working_memory_as_prompt_block():
    buf = WorkingMemoryBuffer(capacity=5)
    buf.add(WorkingMemoryChunk(kind="user_message", content="Bonjour", relevance=1.0))
    buf.add(WorkingMemoryChunk(kind="skill", content="Faire X", relevance=0.8))
    block = buf.as_prompt_block()
    assert "[USER_MESSAGE]" in block
    assert "[SKILL]" in block
    assert "Bonjour" in block


def test_working_memory_status():
    buf = WorkingMemoryBuffer(capacity=3)
    buf.add(WorkingMemoryChunk(kind="msg", content="x"))
    status = buf.status()
    assert status["capacity"] == 3
    assert status["used"] == 1


# ── TieredRetriever ───────────────────────────────────────────────────


def test_tiered_retriever_core_sufficient():
    """Si le Core a un user_message récent, on s'arrête là."""
    buf = WorkingMemoryBuffer()
    buf.add(WorkingMemoryChunk(
        kind="user_message", content="dernier message", relevance=0.9
    ))
    retriever = TieredRetriever(working_memory=buf)
    result = retriever.retrieve(query="dernier message")
    assert result.max_level_reached == "core"
    assert "core" in result.sources_consulted


def test_tiered_retriever_low_doubt_skips_kg_chroma():
    """Avec doubt_index < gate, on skip KG et Chroma même si vides."""
    buf = WorkingMemoryBuffer()
    retriever = TieredRetriever(
        working_memory=buf,
        doubt_gate=0.5,
    )
    result = retriever.retrieve(query="test", doubt_index=0.1)
    # Ne doit pas avoir consulté kg ni chroma
    assert "kg" not in result.sources_consulted
    assert "chroma" not in result.sources_consulted


def test_tiered_retriever_high_doubt_consults_all():
    """Avec doubt_index élevé, on descend jusqu'au bout."""
    buf = WorkingMemoryBuffer()

    class FakeChroma:
        def query(self, query_embeddings, n_results=5):
            return {"documents": [["doc1", "doc2"]], "distances": [[0.3, 0.5]]}

    retriever = TieredRetriever(
        working_memory=buf,
        chroma=FakeChroma(),
        doubt_gate=0.05,  # très bas pour forcer le passage
    )
    result = retriever.retrieve(
        query="question complexe",
        query_embedding=[0.1] * 64,
        doubt_index=0.8,  # haut → on descend
    )
    assert "chroma" in result.sources_consulted
    assert result.max_level_reached == "chroma"


def test_tiered_retriever_handles_backend_errors_gracefully():
    """Si un backend lève, on ne crash pas."""
    buf = WorkingMemoryBuffer()

    class BrokenSDM:
        def read(self, address, top_k=3):
            raise RuntimeError("SDM broken")

    retriever = TieredRetriever(working_memory=buf, sdm=BrokenSDM())
    result = retriever.retrieve(
        query="test",
        query_embedding=[0.1] * 32,
        doubt_index=0.5,
    )
    # Le retriever ne doit pas crasher
    assert result.error is None or "broken" in (result.error or "")


# ── AutoSkillStore ────────────────────────────────────────────────────


def test_auto_skill_add_and_get(tmp_path):
    store = AutoSkillStore(storage_dir=tmp_path)
    skill = Skill(
        skill_id="skill_test1",
        trigger="Quand l'utilisateur demande un calcul",
        approach=["Utiliser python_executor"],
        validation=["Résultat numérique correct"],
        trigger_embedding=[0.1] * 32,
    )
    added = store.add(skill)
    assert added.skill_id == "skill_test1"
    retrieved = store.get("skill_test1")
    assert retrieved is not None
    assert retrieved.trigger == "Quand l'utilisateur demande un calcul"


def test_auto_skill_dedup_by_embedding(tmp_path):
    store = AutoSkillStore(storage_dir=tmp_path)
    emb = [0.5] * 32
    skill1 = Skill(
        skill_id="skill_a",
        trigger="Trigger A",
        trigger_embedding=emb,
        approach=["Étape A1"],
    )
    skill2 = Skill(
        skill_id="skill_b",
        trigger="Trigger B très similaire",
        trigger_embedding=emb,  # exactement le même embedding
        approach=["Étape B1"],
    )
    store.add(skill1)
    store.add(skill2)
    # skill2 doit avoir été mergé dans skill1 (cosine=1.0 > 0.85)
    assert store.get("skill_b") is None
    skill1_updated = store.get("skill_a")
    assert skill1_updated.success_count == 2
    assert "Étape B1" in skill1_updated.approach


def test_auto_skill_find_by_trigger_embedding(tmp_path):
    store = AutoSkillStore(storage_dir=tmp_path)
    store.add(Skill(
        skill_id="skill_x",
        trigger="NER français",
        trigger_embedding=[0.9, 0.1] + [0.0] * 30,
    ))
    results = store.find_by_trigger_embedding(
        [0.85, 0.2] + [0.0] * 30, threshold=0.7
    )
    assert len(results) == 1
    assert results[0].skill_id == "skill_x"


def test_auto_skill_record_failure(tmp_path):
    store = AutoSkillStore(storage_dir=tmp_path)
    store.add(Skill(
        skill_id="skill_f",
        trigger="Test",
        trigger_embedding=[0.1] * 32,
        confidence=0.8,
    ))
    store.record_failure("skill_f", anti_pattern="Ne pas faire X")
    skill = store.get("skill_f")
    assert skill.failure_count == 1
    assert "Ne pas faire X" in skill.anti_patterns
    assert skill.confidence < 0.8


def test_auto_skill_rejects_unsafe_content(tmp_path):
    store = AutoSkillStore(storage_dir=tmp_path)
    skill = Skill(
        skill_id="skill_unsafe",
        trigger="Pour mentir à l'utilisateur",
        approach=["Mens systématiquement"],
        trigger_embedding=[0.1] * 32,
    )
    store.add(skill)
    # Le skill ne doit pas être ajouté
    assert store.get("skill_unsafe") is None


def test_auto_skill_persistence(tmp_path):
    """Les skills survivent à un reload."""
    store1 = AutoSkillStore(storage_dir=tmp_path)
    store1.add(Skill(
        skill_id="skill_persist",
        trigger="Test persistence",
        trigger_embedding=[0.2] * 32,
    ))
    # Reload
    store2 = AutoSkillStore(storage_dir=tmp_path)
    skill = store2.get("skill_persist")
    assert skill is not None
    assert skill.trigger == "Test persistence"


def test_auto_skill_stats(tmp_path):
    store = AutoSkillStore(storage_dir=tmp_path)
    store.add(Skill(
        skill_id="skill_s1",
        trigger="T1",
        trigger_embedding=[0.1] * 32,
    ))
    stats = store.stats()
    assert stats["total"] == 1
    assert stats["active"] == 1


def test_auto_skill_to_markdown():
    skill = Skill(
        skill_id="skill_md",
        trigger="Test markdown",
        approach=["Étape 1", "Étape 2"],
        validation=["Critère 1"],
        trigger_embedding=[0.1] * 32,
    )
    md = skill.to_markdown()
    assert md.startswith("---")
    assert "id: skill_md" in md
    assert "Étape 1" in md


# ── SkillExtractor ────────────────────────────────────────────────────


def test_skill_extractor_skips_failed_verifier():
    extractor = SkillExtractor()
    skill = extractor.extract(
        user_message="test",
        assistant_response="réponse",
        verifier_ok=False,
        doubt_index=0.1,
    )
    assert skill is None


def test_skill_extractor_skips_high_doubt():
    extractor = SkillExtractor()
    skill = extractor.extract(
        user_message="test question",
        assistant_response="réponse suffisamment longue pour passer le filtre",
        verifier_ok=True,
        doubt_index=0.5,
        confidence_label="certaine",
    )
    assert skill is None


def test_skill_extractor_heuristic_success():
    extractor = SkillExtractor()
    skill = extractor.extract(
        user_message="Comment faire X",
        assistant_response="Pour faire X, il faut d'abord Y. Ensuite, on fait Z. Enfin, on valide.",
        verifier_ok=True,
        doubt_index=0.1,
        confidence_label="très_certaine",
        trigger_embedding=[0.1] * 32,
    )
    assert skill is not None
    assert len(skill.approach) > 0
    assert len(skill.validation) > 0


# ── FailureMemory ─────────────────────────────────────────────────────


def test_failure_memory_add_and_find(tmp_path):
    mem = FailureMemory(storage_dir=tmp_path)
    pattern = FailurePattern(
        failure_id="fail_1",
        context="Recommander un modèle NER",
        attempted_action="Citer spaCy sans vérif",
        symptom="Variantes inventées",
        root_cause="Généralisation abusive",
        correction="Vérifier chaque variante",
        embedding=[0.9, 0.1] + [0.0] * 30,
    )
    mem.add(pattern)
    results = mem.find_by_embedding([0.85, 0.15] + [0.0] * 30, threshold=0.7)
    assert len(results) == 1
    assert results[0].failure_id == "fail_1"


def test_failure_memory_dedup(tmp_path):
    mem = FailureMemory(storage_dir=tmp_path)
    emb = [0.5] * 32
    p1 = FailurePattern(
        failure_id="fail_a",
        context="Context A",
        attempted_action="Action A",
        symptom="Symptom",
        root_cause="Cause",
        correction="Correction A",
        embedding=emb,
    )
    p2 = FailurePattern(
        failure_id="fail_b",
        context="Context B très similaire",
        attempted_action="Action B",
        symptom="Symptom",
        root_cause="Cause",
        correction="Correction B",
        embedding=emb,  # same embedding
    )
    mem.add(p1)
    mem.add(p2)
    # p2 doit être merge dans p1
    assert mem.get("fail_b") is None
    p1_updated = mem.get("fail_a")
    assert p1_updated.occurrences == 2
    assert "Correction B" in p1_updated.correction


def test_failure_memory_warning_block(tmp_path):
    mem = FailureMemory(storage_dir=tmp_path)
    mem.add(FailurePattern(
        failure_id="fail_w",
        context="Test contexte",
        attempted_action="Mauvaise action",
        symptom="Mauvais résultat",
        root_cause="Cause racine",
        correction="Bonne action",
        embedding=[0.5] * 32,
    ))
    block = mem.as_warning_block(embedding=[0.5] * 32, top_k=1)
    assert "ANTI-PATTERNS" in block
    assert "Test contexte" in block
    assert "Bonne action" in block


def test_failure_memory_persistence(tmp_path):
    mem1 = FailureMemory(storage_dir=tmp_path)
    mem1.add(FailurePattern(
        failure_id="fail_p",
        context="Persist",
        attempted_action="A",
        symptom="S",
        root_cause="C",
        correction="Corr",
        embedding=[0.3] * 32,
    ))
    mem2 = FailureMemory(storage_dir=tmp_path)
    pattern = mem2.get("fail_p")
    assert pattern is not None
    assert pattern.context == "Persist"


# ── FailureAnalyzer ───────────────────────────────────────────────────


def test_failure_analyzer_no_reasons_returns_none():
    analyzer = FailureAnalyzer()
    pattern = analyzer.analyze(
        context="test",
        attempted_action="action",
        verifier_reasons=[],
        user_message="msg",
        assistant_response="resp",
    )
    assert pattern is None


def test_failure_analyzer_heuristic_short_response():
    analyzer = FailureAnalyzer()
    pattern = analyzer.analyze(
        context="test",
        attempted_action="action",
        verifier_reasons=["trop court (5 mots < 80 attendus)"],
        user_message="msg",
        assistant_response="réponse courte",
        context_embedding=[0.1] * 32,
    )
    assert pattern is not None
    assert "court" in pattern.symptom.lower()
