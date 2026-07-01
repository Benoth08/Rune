"""Trinity — pool multi-modèles optionnel pour Rune.

Inspiré du concept "Trinity" mentionné dans le BACKLOG Lythea V6+ :
un pool de 3 modèles spécialisés qui collaborent sur les missions
complexes.

Architecture
------------
- **Thinker** : modèle de raisonnement pur (DeepSeek-R1 distill, Qwen3-Thinking).
  Décompose les problèmes, génère des chaînes de pensée.
- **Worker** : modèle d'exécution (Qwen2.5-Instruct, Qwen2.5-Coder).
  Implémente, code, exécute les outils.
- **Critic** : modèle de vérification (modèle plus petit, rapide).
  Évalue les livrables du Worker, détecte les erreurs.

Activation
----------
**Désactivé par défaut.** Pour activer :

1. Créer un fichier de config `trinity.yaml` (ou utiliser le défaut) :
   ```yaml
   enabled: true
   thinker:
     model_id: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
     quant_4bit: true
   worker:
     model_id: "Qwen/Qwen2.5-7B-Instruct"
     quant_4bit: true
   critic:
     model_id: "Qwen/Qwen2.5-3B-Instruct"
     quant_4bit: true
   routing:
     # Quand utiliser le Thinker (vs Worker seul)
     thinker_threshold_steps: 4   # assess_complexity >= 4
     thinker_threshold_surprise: 0.6
     # Quand utiliser le Critic (toujours si activé)
     critic_always: true
   ```

2. Set `RUNE_TRINITY_CONFIG=data/trinity.yaml` dans `.env`.

3. Rune charge les 3 modèles au boot (si VRAM suffisante) et les
   utilise selon le routing.

Contraintes VRAM
----------------
Trinity nécessite assez de VRAM pour 3 modèles simultanés. Sur 24 GB :
- Thinker 7B NF4 (~5 GB) + Worker 7B NF4 (~5 GB) + Critic 3B NF4 (~2.5 GB)
- Total ~12.5 GB → OK sur 24 GB
- Sur 40 GB : on peut monter en bf16 (qualité meilleure)

Si VRAM insuffisante, Trinity tombe en mode "single model" (Worker seul)
avec un warning au boot.

Différence avec le mode single (Lythea d'origine)
-------------------------------------------------
- **Single** : 1 modèle fait tout (think + execute + verify)
- **Trinity** : 3 modèles spécialisés collaborent via handoffs

Trinity ne remplace PAS le cycle cognitif Lythea — il change juste
quel modèle génère à chaque phase. Le cycle (encode → surprise →
retrieve → generate) reste identique. C'est une couche en dessous.

Cohabitation avec Rune extensions
---------------------------------
Trinity est orthogonal à AutoSkill / FailureMemory / SubAgent :
- AutoSkill continue de s'extraire après succès
- FailureMemory continue d'apprendre des échecs
- SubAgent continue d'isoler les tâches

Quand un SubAgent tourne en mode Trinity, il utilise le Worker par
défaut (le Thinker et Critic sont réservés au parent).
"""
from __future__ import annotations

from .config import TrinityConfig, TrinityModelSpec, load_trinity_config
from .pool import TrinityPool, TrinityRole, TrinityHandoff

__all__ = [
    "TrinityConfig",
    "TrinityModelSpec",
    "TrinityPool",
    "TrinityRole",
    "TrinityHandoff",
    "load_trinity_config",
]
