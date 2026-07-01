"""Tests SkillComposer — composition de skills existantes."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rune.memory.auto_skill import AutoSkillStore, Skill
from rune.memory.skill_composer import (
    CompositionStrategy,
    SkillComposer,
    _cosine_similarity,
    _lexical_similarity,
)


# ── Helpers ───────────────────────────────────────────────────────────


def _make_skill(
    skill_id: str,
    trigger: str,
    approach: list[str],
    success_count: int = 5,
    confidence: float = 0.8,
    trigger_embedding: list[float] | None = None,
) -> Skill:
    """Crée un Skill avec success_count suffisant pour is_reliable()."""
    return Skill(
        skill_id=skill_id,
        trigger=trigger,
        approach=approach,
        validation=["Réponse fournie"],
        trigger_embedding=trigger_embedding or [0.1] * 32,
        success_count=success_count,
        confidence=confidence,
    )


@pytest.fixture
def store_with_skills(tmp_path):
    """Store avec 3 skills fiables (embeddings très distincts pour éviter dédup)."""
    store = AutoSkillStore(storage_dir=tmp_path)
    store.add(_make_skill(
        "skill_fib", "Calculer fibonacci",
        ["Utiliser récursion", "Ajouter mémoïsation"],
        trigger_embedding=[0.9, 0.1, 0.0] + [0.0] * 29,
    ))
    store.add(_make_skill(
        "skill_test", "Écrire tests pytest",
        ["Créer fixtures", "Écrire asserts", "Lancer pytest"],
        trigger_embedding=[0.1, 0.9, 0.0] + [0.0] * 29,
    ))
    store.add(_make_skill(
        "skill_debug", "Débugger Python",
        ["Utiliser pdb", "Ajouter prints", "Inspecter variables"],
        trigger_embedding=[0.0, 0.0, 0.9] + [0.0] * 29,
    ))
    return store


# ── Tests validation ─────────────────────────────────────────────────


def test_compose_needs_at_least_2_skills(store_with_skills):
    """Erreur si moins de 2 skills à composer."""
    result = store_with_skills.compose(skill_ids=["skill_fib"])
    assert result["status"] == "skipped"
    assert "at least 2" in result["reason"].lower()


def test_compose_skill_not_found(store_with_skills):
    """Erreur si une skill n'existe pas."""
    result = store_with_skills.compose(
        skill_ids=["skill_fib", "nonexistent"]
    )
    assert result["status"] == "error"
    assert "not found" in result["reason"]


def test_compose_skips_unreliable_skill(tmp_path):
    """Une skill non fiable est refusée sans force=True."""
    store = AutoSkillStore(storage_dir=tmp_path)
    # success_count = 1 → is_reliable() retourne False (besoin de 2)
    # Embeddings différents pour éviter la dédup
    store.add(_make_skill(
        "skill_a", "trigger A", ["step A"],
        success_count=1,
        trigger_embedding=[0.9, 0.1] + [0.0] * 30,
    ))
    store.add(_make_skill(
        "skill_b", "trigger B", ["step B"],
        success_count=1,
        trigger_embedding=[0.1, 0.9] + [0.0] * 30,
    ))
    result = store.compose(skill_ids=["skill_a", "skill_b"])
    assert result["status"] == "skipped"
    assert "not reliable" in result["reason"]


def test_compose_force_overrides_reliability(tmp_path):
    """force=True compose même si skills non fiables."""
    store = AutoSkillStore(storage_dir=tmp_path)
    store.add(_make_skill(
        "skill_a", "trigger A", ["step A"], success_count=1,
        trigger_embedding=[0.9, 0.1] + [0.0] * 30,
    ))
    store.add(_make_skill(
        "skill_b", "trigger B", ["step B"], success_count=1,
        trigger_embedding=[0.1, 0.9] + [0.0] * 30,
    ))
    result = store.compose(
        skill_ids=["skill_a", "skill_b"],
        force=True,
    )
    assert result["status"] == "ok"
    assert result["composed_skill_id"] is not None


# ── Tests stratégies ─────────────────────────────────────────────────


def test_compose_sequential(store_with_skills):
    """Stratégie sequential concatène les approaches dans l'ordre."""
    result = store_with_skills.compose(
        skill_ids=["skill_fib", "skill_test"],
        strategy=CompositionStrategy.SEQUENTIAL,
    )
    assert result["status"] == "ok"
    composed = store_with_skills.get(result["composed_skill_id"])
    assert composed is not None
    # Doit contenir les étapes des 2 skills
    all_approach = " ".join(composed.approach)
    assert "récursion" in all_approach or "mémoïsation" in all_approach
    assert "fixtures" in all_approach or "pytest" in all_approach


def test_compose_parallel(store_with_skills):
    """Stratégie parallel intercale les étapes."""
    result = store_with_skills.compose(
        skill_ids=["skill_fib", "skill_test"],
        strategy=CompositionStrategy.PARALLEL,
    )
    assert result["status"] == "ok"
    composed = store_with_skills.get(result["composed_skill_id"])
    assert composed is not None
    # Vérifie qu'on a bien des étapes des 2 skills
    assert len(composed.approach) >= 2


def test_compose_conditional(store_with_skills):
    """Stratégie conditional structure en if/else."""
    result = store_with_skills.compose(
        skill_ids=["skill_fib", "skill_test"],
        strategy=CompositionStrategy.CONDITIONAL,
    )
    assert result["status"] == "ok"
    composed = store_with_skills.get(result["composed_skill_id"])
    assert composed is not None
    all_approach = " ".join(composed.approach).lower()
    assert "condition" in all_approach or "sinon" in all_approach


def test_compose_pipeline(store_with_skills):
    """Stratégie pipeline structure en étapes génère/valide."""
    result = store_with_skills.compose(
        skill_ids=["skill_fib", "skill_test", "skill_debug"],
        strategy=CompositionStrategy.PIPELINE,
    )
    assert result["status"] == "ok"
    composed = store_with_skills.get(result["composed_skill_id"])
    assert composed is not None
    all_approach = " ".join(composed.approach)
    # Doit contenir des marqueurs de rôle
    assert any(role in all_approach for role in ["Générer", "Valider", "Corriger", "Étape"])


# ── Tests métadonnées ────────────────────────────────────────────────


def test_composed_skill_marked_as_composed(store_with_skills):
    """La skill composée a metadata.composed = True."""
    result = store_with_skills.compose(
        skill_ids=["skill_fib", "skill_test"],
    )
    composed = store_with_skills.get(result["composed_skill_id"])
    assert composed is not None
    assert composed.metadata.get("composed") is True
    assert composed.metadata.get("strategy") == "sequential"
    assert "skill_fib" in composed.metadata.get("source_skill_ids", [])
    assert "skill_test" in composed.metadata.get("source_skill_ids", [])


def test_composed_skill_inherits_anti_patterns(store_with_skills):
    """La skill composée hérite des anti_patterns des sources."""
    # Ajoute un anti-pattern à une skill source
    store_with_skills.record_failure("skill_fib", anti_pattern="Ne pas oublier le cas base")
    result = store_with_skills.compose(
        skill_ids=["skill_fib", "skill_test"],
    )
    composed = store_with_skills.get(result["composed_skill_id"])
    assert composed is not None
    assert "Ne pas oublier le cas base" in composed.anti_patterns


def test_composed_skill_embedding_is_average(store_with_skills):
    """L'embedding de la skill composée est la moyenne des embeddings sources."""
    result = store_with_skills.compose(
        skill_ids=["skill_fib", "skill_test"],
    )
    composed = store_with_skills.get(result["composed_skill_id"])
    assert composed is not None
    # emb_fib = [0.9, 0.1, 0.0, 0, 0, ...], emb_test = [0.1, 0.9, 0.0, 0, 0, ...]
    # moyenne = [0.5, 0.5, 0.0, 0, 0, ...]
    assert abs(composed.trigger_embedding[0] - 0.5) < 0.01
    assert abs(composed.trigger_embedding[1] - 0.5) < 0.01
    assert abs(composed.trigger_embedding[2] - 0.0) < 0.01


def test_compose_persists_to_disk(store_with_skills, tmp_path):
    """La skill composée est persistée — survive au reload."""
    result = store_with_skills.compose(
        skill_ids=["skill_fib", "skill_test"],
    )
    composed_id = result["composed_skill_id"]

    # Reload le store
    store2 = AutoSkillStore(storage_dir=tmp_path)
    composed = store2.get(composed_id)
    assert composed is not None
    assert composed.metadata.get("composed") is True


# ── Tests find_composable_candidates ─────────────────────────────────


def test_find_composable_candidates_returns_pairs(store_with_skills):
    """find_composable_candidates retourne des paires triées par potential."""
    composer = SkillComposer(store_with_skills)
    # Les embeddings fib/test/debug sont très orthogonaux (similarité ~0.18)
    # On override le seuil min via les paramètres de recherche
    pairs = composer.find_composable_candidates(max_pairs=5)
    # Si pas de paires (similarité trop basse), on teste avec des embeddings proches
    if not pairs:
        # Ajoute 2 skills avec embeddings dans la zone composable [0.3, 0.85]
        # Emb A = [1, 0, 0, ...], Emb B = [0.5, 0.5, 0, ...] → cos = 0.5
        store_with_skills.add(_make_skill(
            "skill_x", "Tâche X maths",
            ["step X"],
            trigger_embedding=[1.0, 0.0, 0.0] + [0.0] * 29,
        ))
        store_with_skills.add(_make_skill(
            "skill_y", "Tâche Y maths",
            ["step Y"],
            trigger_embedding=[0.5, 0.5, 0.0] + [0.0] * 29,
        ))
        pairs = composer.find_composable_candidates(max_pairs=5)
    assert len(pairs) > 0
    # Tri décroissant par potential
    potentials = [p[2] for p in pairs]
    assert potentials == sorted(potentials, reverse=True)


def test_find_composable_candidates_via_store(store_with_skills):
    """L'API facade du store retourne des dicts."""
    pairs = store_with_skills.find_composable_candidates(max_pairs=3)
    assert isinstance(pairs, list)
    for p in pairs:
        assert "skill_a" in p
        assert "skill_b" in p
        assert "potential" in p


# ── Tests composition LLM ────────────────────────────────────────────


def test_compose_with_llm_callback(store_with_skills):
    """La composition LLM utilise le callback si fourni."""
    def mock_llm(prompt: str) -> dict:
        return {
            "trigger": "Calculer et tester fibonacci",
            "approach": [
                "1. Implémenter fibonacci avec récursion",
                "2. Ajouter mémoïsation",
                "3. Écrire tests pytest",
                "4. Lancer pytest pour valider",
            ],
            "validation": ["Tests passent", "Résultat correct"],
        }

    composer = SkillComposer(store_with_skills, llm_callback=mock_llm)
    result = composer.compose(
        skill_ids=["skill_fib", "skill_test"],
        strategy="sequential",
    )
    assert result.status == "ok"
    composed = result.skill
    assert composed.trigger == "Calculer et tester fibonacci"
    assert len(composed.approach) == 4
    assert "récursion" in composed.approach[0].lower()


def test_compose_llm_fallback_to_heuristic(store_with_skills):
    """Si le LLM lève une exception, fallback sur heuristique."""
    def broken_llm(prompt: str) -> dict:
        raise RuntimeError("LLM broken")

    composer = SkillComposer(store_with_skills, llm_callback=broken_llm)
    result = composer.compose(
        skill_ids=["skill_fib", "skill_test"],
    )
    # Fallback réussit
    assert result.status == "ok"
    assert result.skill is not None


def test_compose_llm_skip_falls_back_to_heuristic(store_with_skills):
    """Si le LLM retourne {skip: true}, fallback sur heuristique."""
    def skip_llm(prompt: str) -> dict:
        return {"skip": True}

    composer = SkillComposer(store_with_skills, llm_callback=skip_llm)
    result = composer.compose(
        skill_ids=["skill_fib", "skill_test"],
    )
    assert result.status == "ok"


# ── Tests helpers ────────────────────────────────────────────────────


def test_cosine_similarity_identical():
    a = [1.0, 0.0, 0.0]
    assert abs(_cosine_similarity(a, a) - 1.0) < 0.01


def test_cosine_similarity_orthogonal():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert abs(_cosine_similarity(a, b)) < 0.01


def test_cosine_similarity_empty():
    assert _cosine_similarity([], []) == 0.0


def test_lexical_similarity_identical():
    assert _lexical_similarity("hello world", "hello world") == 1.0


def test_lexical_similarity_no_overlap():
    assert _lexical_similarity("foo", "bar") == 0.0


def test_lexical_similarity_partial():
    sim = _lexical_similarity("hello world foo", "hello world bar")
    assert 0.3 < sim < 0.8  # 2 mots sur 3 en commun


# ── Test CLI ─────────────────────────────────────────────────────────


def test_cli_skills_compose_help():
    """La commande 'skills compose --help' doit fonctionner."""
    from typer.testing import CliRunner
    from rune.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["skills", "compose", "--help"])
    assert result.exit_code == 0
    assert "strategy" in result.stdout.lower()


def test_cli_skills_candidates_help():
    """La commande 'skills candidates --help' doit fonctionner."""
    from typer.testing import CliRunner
    from rune.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["skills", "candidates", "--help"])
    assert result.exit_code == 0
