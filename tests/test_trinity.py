"""Tests Trinity — pool multi-modèles optionnel."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rune.trinity import (
    TrinityConfig,
    TrinityModelSpec,
    TrinityPool,
    TrinityRole,
    load_trinity_config,
)
from rune.trinity.config import save_default_config_template, _parse_config


# ── Config ────────────────────────────────────────────────────────────


def test_trinity_default_config_is_disabled():
    """TrinityConfig par défaut doit être désactivé."""
    config = TrinityConfig()
    assert config.enabled is False


def test_trinity_load_missing_file_returns_default(tmp_path):
    """Si le fichier n'existe pas, retourne la config par défaut."""
    config = load_trinity_config(tmp_path / "nonexistent.yaml")
    assert config.enabled is False


def test_trinity_load_none_path_returns_default():
    """Si path=None, retourne la config par défaut."""
    config = load_trinity_config(None)
    assert config.enabled is False


def test_trinity_load_valid_config(tmp_path):
    """Charge un fichier trinity.yaml valide."""
    config_path = tmp_path / "trinity.yaml"
    config_path.write_text("""
enabled: true
thinker:
  model_id: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
  quant_4bit: true
  max_new_tokens: 1024
worker:
  model_id: "Qwen/Qwen2.5-7B-Instruct"
  quant_4bit: true
critic:
  model_id: "Qwen/Qwen2.5-3B-Instruct"
  quant_4bit: true
routing:
  thinker_threshold_steps: 3
  critic_always: false
""", encoding="utf-8")

    config = load_trinity_config(config_path)
    assert config.enabled is True
    assert config.thinker.model_id == "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    assert config.thinker.max_new_tokens == 1024
    assert config.worker.model_id == "Qwen/Qwen2.5-7B-Instruct"
    assert config.critic.model_id == "Qwen/Qwen2.5-3B-Instruct"
    assert config.routing.thinker_threshold_steps == 3
    assert config.routing.critic_always is False


def test_trinity_load_invalid_yaml_returns_default(tmp_path):
    """Si le YAML est invalide, retourne la config par défaut."""
    config_path = tmp_path / "trinity.yaml"
    config_path.write_text("not: valid: yaml: [", encoding="utf-8")
    config = load_trinity_config(config_path)
    assert config.enabled is False


def test_trinity_save_default_template(tmp_path):
    """Génère un template de config par défaut."""
    path = tmp_path / "trinity.yaml"
    save_default_config_template(path)
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "enabled: false" in content
    assert "thinker:" in content
    assert "worker:" in content
    assert "critic:" in content


def test_trinity_config_as_dict():
    """as_dict retourne une structure JSON-sérialisable."""
    config = TrinityConfig()
    d = config.as_dict()
    assert "enabled" in d
    assert "thinker" in d
    assert "worker" in d
    assert "critic" in d
    assert "routing" in d
    # JSON-serializable
    import json
    json.dumps(d)


# ── Pool ──────────────────────────────────────────────────────────────


@pytest.fixture
def mock_pool():
    """Pool Trinity avec 3 modèles mockés (pas de GPU)."""
    config = TrinityConfig()
    config.enabled = True

    pool = TrinityPool(config)

    class MockModel:
        def __init__(self, name):
            self.name = name
        def load(self, model_id): return True
        def unload(self): pass

    pool.models[TrinityRole.THINKER] = MockModel("thinker")
    pool.models[TrinityRole.WORKER] = MockModel("worker")
    pool.models[TrinityRole.CRITIC] = MockModel("critic")
    pool._loaded = True
    return pool


def test_trinity_pool_disabled_returns_none():
    """Si Trinity désactivé, pick_model retourne le Worker (ou None)."""
    config = TrinityConfig()
    config.enabled = False
    pool = TrinityPool(config)
    # Pas de modèle chargé
    assert pool.pick_model(phase="execution") is None


def test_trinity_pool_routing_reasoning_complex_uses_thinker(mock_pool):
    """Reasoning + complexité élevée → Thinker."""
    model = mock_pool.pick_model(
        complexity_steps=4, surprise=0.3, phase="reasoning"
    )
    assert model is not None
    assert model.name == "thinker"


def test_trinity_pool_routing_reasoning_simple_uses_worker(mock_pool):
    """Reasoning + complexité basse → Worker."""
    model = mock_pool.pick_model(
        complexity_steps=2, surprise=0.3, phase="reasoning"
    )
    assert model is not None
    assert model.name == "worker"


def test_trinity_pool_routing_high_surprise_uses_thinker(mock_pool):
    """Surprise élevée → Thinker (même si complexité basse)."""
    model = mock_pool.pick_model(
        complexity_steps=1, surprise=0.7, phase="reasoning"
    )
    assert model is not None
    assert model.name == "thinker"


def test_trinity_pool_routing_execution_uses_worker(mock_pool):
    """Phase exécution → toujours Worker."""
    model = mock_pool.pick_model(phase="execution")
    assert model is not None
    assert model.name == "worker"


def test_trinity_pool_routing_verification_uses_critic(mock_pool):
    """Phase vérification → Critic (critic_always=true par défaut)."""
    model = mock_pool.pick_model(phase="verification", doubt_index=0.1)
    assert model is not None
    assert model.name == "critic"


def test_trinity_pool_routing_critic_only_on_high_doubt():
    """Si critic_always=false, Critic seulement si doubt élevé."""
    config = TrinityConfig()
    config.enabled = True
    config.routing.critic_always = False
    config.routing.critic_threshold_doubt = 0.4
    pool = TrinityPool(config)

    class MockModel:
        def __init__(self, name): self.name = name

    pool.models[TrinityRole.THINKER] = MockModel("thinker")
    pool.models[TrinityRole.WORKER] = MockModel("worker")
    pool.models[TrinityRole.CRITIC] = MockModel("critic")
    pool._loaded = True

    # doubt bas → Worker
    m1 = pool.pick_model(phase="verification", doubt_index=0.2)
    assert m1.name == "worker"

    # doubt haut → Critic
    m2 = pool.pick_model(phase="verification", doubt_index=0.6)
    assert m2.name == "critic"


def test_trinity_pool_fallback_to_worker_if_role_missing():
    """Si le rôle choisi n'est pas chargé, fallback sur Worker."""
    config = TrinityConfig()
    config.enabled = True
    pool = TrinityPool(config)

    class MockModel:
        def __init__(self, name): self.name = name

    # Seulement le Worker est chargé
    pool.models[TrinityRole.WORKER] = MockModel("worker")
    pool._loaded = True

    # Demandons le Thinker (pas chargé) → fallback Worker
    model = pool.pick_model(
        complexity_steps=5, surprise=0.8, phase="reasoning"
    )
    assert model is not None
    assert model.name == "worker"


def test_trinity_pool_degraded_mode_uses_worker():
    """En mode dégradé, toujours le Worker."""
    config = TrinityConfig()
    config.enabled = True
    pool = TrinityPool(config)
    pool.degraded_mode = True

    class MockModel:
        def __init__(self, name): self.name = name

    pool.models[TrinityRole.WORKER] = MockModel("worker")
    pool._loaded = True

    # Même en phase reasoning complex → Worker (dégradé)
    model = pool.pick_model(
        complexity_steps=5, surprise=0.8, phase="reasoning"
    )
    assert model is not None
    assert model.name == "worker"


def test_trinity_pool_status(mock_pool):
    """status retourne la bonne structure."""
    status = mock_pool.status()
    assert status["enabled"] is True
    assert status["loaded"] is True
    assert status["degraded_mode"] is False
    assert "roles" in status
    assert status["roles"]["thinker"]["loaded"] is True
    assert status["roles"]["worker"]["loaded"] is True
    assert status["roles"]["critic"]["loaded"] is True
    assert "routing" in status


def test_trinity_pool_load_all_disabled():
    """Si config.enabled=False, load_all retourne {status: disabled}."""
    config = TrinityConfig()
    config.enabled = False
    pool = TrinityPool(config)
    report = pool.load_all()
    assert report["status"] == "disabled"
    assert pool._loaded is False


def test_trinity_pool_get_model(mock_pool):
    """get_model retourne le modèle pour un rôle donné."""
    thinker = mock_pool.get_model(TrinityRole.THINKER)
    assert thinker is not None
    assert thinker.name == "thinker"

    worker = mock_pool.get_model(TrinityRole.WORKER)
    assert worker is not None
    assert worker.name == "worker"


def test_trinity_pool_get_model_not_loaded():
    """get_model retourne None si le rôle n'est pas chargé."""
    config = TrinityConfig()
    pool = TrinityPool(config)
    assert pool.get_model(TrinityRole.THINKER) is None


def test_trinity_pool_unload_all(mock_pool):
    """unload_all vide tous les modèles."""
    mock_pool.unload_all()
    assert mock_pool.get_model(TrinityRole.THINKER) is None
    assert mock_pool.get_model(TrinityRole.WORKER) is None
    assert mock_pool.get_model(TrinityRole.CRITIC) is None
    assert mock_pool._loaded is False
