"""Trinity configuration — fichier YAML optionnel.

Format du fichier `trinity.yaml` :

    enabled: true
    thinker:
      model_id: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
      quant_4bit: true
      max_new_tokens: 1024
    worker:
      model_id: "Qwen/Qwen2.5-7B-Instruct"
      quant_4bit: true
      max_new_tokens: 512
    critic:
      model_id: "Qwen/Qwen2.5-3B-Instruct"
      quant_4bit: true
      max_new_tokens: 256
    routing:
      thinker_threshold_steps: 4
      thinker_threshold_surprise: 0.6
      critic_always: true

Si le fichier n'existe pas ou `enabled: false`, Trinity reste désactivé
et Rune utilise le mode single-model standard (Lythea d'origine).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrinityModelSpec:
    """Spécification d'un modèle Trinity (un par rôle)."""
    model_id: str
    quant_4bit: bool = True
    max_new_tokens: int = 512
    temperature: float = 0.7
    # Métadonnées optionnelles
    label: str = ""
    notes: str = ""


@dataclass
class TrinityRouting:
    """Règles de routing entre les 3 modèles."""
    # Thinker se déclenche si assess_complexity >= threshold_steps
    thinker_threshold_steps: int = 4
    # Thinker se déclenche si surprise >= threshold_surprise
    thinker_threshold_surprise: float = 0.6
    # Critic tourne toujours (true) ou seulement si doute élevé (false)
    critic_always: bool = True
    # Critic se déclenche si doubt_index >= threshold
    critic_threshold_doubt: float = 0.4


@dataclass
class TrinityConfig:
    """Config Trinity complète."""
    enabled: bool = False
    thinker: TrinityModelSpec = field(default_factory=lambda: TrinityModelSpec(
        model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        label="Thinker (raisonnement pur)",
        notes="Distill R1 — excellent pour décomposer et raisonner",
    ))
    worker: TrinityModelSpec = field(default_factory=lambda: TrinityModelSpec(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        label="Worker (exécution)",
        notes="Instruct standard — code, outils, génération",
    ))
    critic: TrinityModelSpec = field(default_factory=lambda: TrinityModelSpec(
        model_id="Qwen/Qwen2.5-3B-Instruct",
        label="Critic (vérification)",
        notes="Petit modèle rapide pour valider les livrables",
    ))
    routing: TrinityRouting = field(default_factory=TrinityRouting)

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "thinker": {
                "model_id": self.thinker.model_id,
                "quant_4bit": self.thinker.quant_4bit,
                "max_new_tokens": self.thinker.max_new_tokens,
                "label": self.thinker.label,
            },
            "worker": {
                "model_id": self.worker.model_id,
                "quant_4bit": self.worker.quant_4bit,
                "max_new_tokens": self.worker.max_new_tokens,
                "label": self.worker.label,
            },
            "critic": {
                "model_id": self.critic.model_id,
                "quant_4bit": self.critic.quant_4bit,
                "max_new_tokens": self.critic.max_new_tokens,
                "label": self.critic.label,
            },
            "routing": {
                "thinker_threshold_steps": self.routing.thinker_threshold_steps,
                "thinker_threshold_surprise": self.routing.thinker_threshold_surprise,
                "critic_always": self.routing.critic_always,
                "critic_threshold_doubt": self.routing.critic_threshold_doubt,
            },
        }


# Config par défaut — Trinity désactivé
DEFAULT_CONFIG = TrinityConfig()


def load_trinity_config(config_path: str | Path | None = None) -> TrinityConfig:
    """Charge la config Trinity depuis un fichier YAML.

    Parameters
    ----------
    config_path : str | Path | None
        Chemin du fichier trinity.yaml. Si None, lit la variable
        d'environnement RUNE_TRINITY_CONFIG. Si toujours None, utilise
        le défaut (désactivé).

    Returns
    -------
    TrinityConfig
        La config chargée. Si le fichier n'existe pas ou est invalide,
        retourne DEFAULT_CONFIG (Trinity désactivé) avec un warning.
    """
    path = config_path or os.environ.get("RUNE_TRINITY_CONFIG")
    if not path:
        return DEFAULT_CONFIG

    path = Path(path)
    if not path.exists():
        return DEFAULT_CONFIG

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:
        import logging
        logging.getLogger("rune.trinity").warning(
            "Failed to load trinity config %s: %s — using default", path, exc
        )
        return DEFAULT_CONFIG

    try:
        return _parse_config(data)
    except Exception as exc:
        import logging
        logging.getLogger("rune.trinity").warning(
            "Invalid trinity config %s: %s — using default", path, exc
        )
        return DEFAULT_CONFIG


def _parse_config(data: dict) -> TrinityConfig:
    """Parse un dict YAML en TrinityConfig."""
    config = TrinityConfig()
    config.enabled = bool(data.get("enabled", False))

    for role in ("thinker", "worker", "critic"):
        role_data = data.get(role, {})
        if not isinstance(role_data, dict):
            continue
        spec = getattr(config, role)
        if "model_id" in role_data:
            spec.model_id = str(role_data["model_id"])
        if "quant_4bit" in role_data:
            spec.quant_4bit = bool(role_data["quant_4bit"])
        if "max_new_tokens" in role_data:
            spec.max_new_tokens = int(role_data["max_new_tokens"])
        if "temperature" in role_data:
            spec.temperature = float(role_data["temperature"])
        if "label" in role_data:
            spec.label = str(role_data["label"])
        if "notes" in role_data:
            spec.notes = str(role_data["notes"])

    routing_data = data.get("routing", {})
    if isinstance(routing_data, dict):
        r = config.routing
        if "thinker_threshold_steps" in routing_data:
            r.thinker_threshold_steps = int(routing_data["thinker_threshold_steps"])
        if "thinker_threshold_surprise" in routing_data:
            r.thinker_threshold_surprise = float(routing_data["thinker_threshold_surprise"])
        if "critic_always" in routing_data:
            r.critic_always = bool(routing_data["critic_always"])
        if "critic_threshold_doubt" in routing_data:
            r.critic_threshold_doubt = float(routing_data["critic_threshold_doubt"])

    return config


def save_default_config_template(path: str | Path) -> None:
    """Écrit un template de config trinity.yaml à l'emplacement donné.

    Utile pour bootstrap — l'utilisateur peut éditer le template
    au lieu de tout retaper.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    template = """# Trinity config — pool multi-modèles optionnel pour Rune
# Désactivé par défaut. Passe à `enabled: true` pour activer.
#
# Trinity charge 3 modèles spécialisés en VRAM :
# - Thinker : raisonnement pur (décomposition, chaîne de pensée)
# - Worker : exécution (code, outils, génération)
# - Critic : vérification des livrables
#
# Contrainte VRAM : 3 modèles simultanés. Sur 24 GB, utiliser NF4.
# Sur 40+ GB, on peut passer quant_4bit: false (qualité meilleure).

enabled: false

thinker:
  model_id: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
  quant_4bit: true
  max_new_tokens: 1024
  temperature: 0.6
  label: "Thinker (raisonnement pur)"
  notes: "Distill R1 — excellent pour décomposer et raisonner"

worker:
  model_id: "Qwen/Qwen2.5-7B-Instruct"
  quant_4bit: true
  max_new_tokens: 512
  temperature: 0.7
  label: "Worker (exécution)"
  notes: "Instruct standard — code, outils, génération"

critic:
  model_id: "Qwen/Qwen2.5-3B-Instruct"
  quant_4bit: true
  max_new_tokens: 256
  temperature: 0.3
  label: "Critic (vérification)"
  notes: "Petit modèle rapide pour valider les livrables"

routing:
  # Thinker se déclenche si assess_complexity >= threshold_steps
  thinker_threshold_steps: 4
  # Thinker se déclenche si surprise >= threshold_surprise
  thinker_threshold_surprise: 0.6
  # Critic tourne toujours (true) ou seulement si doute élevé (false)
  critic_always: true
  # Critic se déclenche si doubt_index >= threshold (et critic_always=false)
  critic_threshold_doubt: 0.4
"""
    path.write_text(template, encoding="utf-8")
