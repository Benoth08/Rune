# Rune

> Agent IA cognitif **local**, **headless**, **open-weights only** — combine les **42 000 lignes** du cycle cognitif biomimétique de [Lythea](https://github.com/Benoth08/Lythea-Reasoning-Agent) avec les capacités agentiques d'[Rune](https://rune-agent.nousresearch.com/) (auto-skill, subagents, cron).

[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-63%20passed-brightgreen)](#tests)

---

## Ce qu'est Rune

**Lythea v5.8** est un assistant LLM local avec un cycle cognitif biomimétique remarquable (encodage → surprise → retrieval → génération → consolidation), une mémoire multi-étages (SDM + MHN + KG + Chroma), du steering CAA, de la métacognition, du deep reasoning, du vision active, etc.

**Rune** (Nous Research, 2026) est un agent autonome avec auto-skill (SKILL.md), subagents isolés, cron de fond.

**Rune** = Lythea (intégralement conservé) + extensions Rune :

| Feature | Lythea v5.8 | Rune | **Rune** |
|---|---|---|---|
| Cycle cognitif biomimétique (5 phases) | ✅ | ❌ | ✅ |
| Mémoire SDM + MHN + KG + Chroma | ✅ | ❌ | ✅ |
| Steering CAA (hooks PyTorch) | ✅ | ❌ | ✅ |
| Métacognition multi-signaux | ✅ (V4.4) | ❌ | ✅ |
| Deep reasoning (chaîne 4 étapes) | ✅ | ❌ | ✅ |
| Vision active (zoom cognitif VLM) | ✅ | ❌ | ✅ |
| Cascade Gemini (draft-then-refine) | ✅ | ❌ | ✅ |
| Web providers (Brave, DDG, SearXNG…) | ✅ | ❌ | ✅ |
| Inhibition (3 niveaux) | ✅ | ❌ | ✅ |
| Predictive coding (Friston) | ✅ | ❌ | ✅ |
| Timeline + incohérence temporelle | ✅ | ❌ | ✅ |
| Planning (PFC latéral) | ✅ | ❌ | ✅ |
| Reflection + auto-calibration | ✅ | ❌ | ✅ |
| CRAG (Corrective RAG) | ✅ | ❌ | ✅ |
| GraphRAG (communities) | ✅ | ❌ | ✅ |
| Consolidation microsleep/deep sleep | ✅ | ❌ | ✅ |
| MCP support | ✅ | ❌ | ✅ |
| Agent Taëlys (orchestrator + workers) | ✅ | ❌ | ✅ |
| **AutoSkill (SKILL.md auto)** | ❌ (procédural passif) | ✅ | ✅ **+ métacognition filtre** |
| **FailureMemory (anti-patterns)** | ❌ | ❌ | ✅ **unique** |
| **SubAgent spawner (subprocess)** | ❌ (workers in-process) | ✅ | ✅ |
| **Cron + consolidation post-run** | ❌ | ✅ (sans consolidation) | ✅ **+ microsleep** |
| **TieredRetriever strict (5 niveaux)** | partiel | ✅ (3 niveaux) | ✅ |
| **WorkingMemoryBuffer (Core 4±1)** | ❌ | ✅ | ✅ |
| **Headless (CLI + API, pas d'UI)** | ❌ (UI lourde) | ✅ | ✅ |
| **Speculative decoding** | ❌ | n/a | ✅ |
| Modèles open-source + accès poids | ✅ | ❌ (API) | ✅ |

---

## Architecture

```
rune/                       # Package principal (anciennement lythea/)
├── hippocampe.py                    # ← Cœur cognitif Lythea (3526 lignes, INTACT)
├── cognition/                       # ← 26 modules cognitifs Lythea (INTACTS)
│   ├── encoding.py                  #   cortex entorhinal
│   ├── storage.py                   #   hippocampe CA3
│   ├── surprise.py                  #   ripples + dopamine
│   ├── retrieval.py                 #   pattern completion
│   ├── generation.py                #   two-pass reasoning
│   ├── consolidation.py             #   microsleep + deep sleep
│   ├── metacognition.py             #   mPFC + dACC
│   ├── inhibition.py                #   contrôle inhibiteur 3 niveaux
│   ├── planning.py                  #   PFC latéral
│   ├── predictive_coding.py         #   Friston
│   ├── timeline.py                  #   chronologie narrative
│   ├── deep_reasoning.py            #   chaîne 4 étapes
│   ├── reflection.py                #   auto-critique
│   ├── cascade.py                   #   Gemini draft-then-refine
│   ├── crag.py                      #   Corrective RAG
│   ├── graph_communities.py         #   GraphRAG
│   ├── semantic_router.py           #   multi-tool routing
│   ├── vision_semantic.py           #   zoom cognitif VLM
│   ├── ... (26 modules au total)
│   └── *_rune.py                  # ← Extensions Rune (consolidation, surprise, metacog)
│
├── memory/                          # ← Mémoire Lythea (INTACTE) + extensions Rune
│   ├── sdm.py                       #   Sparse Distributed Memory (Kanerva)
│   ├── mhn.py                       #   Modern Hopfield Network
│   ├── kg.py                        #   Knowledge Graph (GLiNER + RapidFuzz)
│   ├── retrieval.py                 #   HybridRetriever (cross-encoder)
│   ├── procedural.py                #   mémoire procédurale Lythea
│   ├── cognitive_state.py           #   état affectif (valence/arousal)
│   ├── salience.py                  #   filtre de saillance
│   ├── visual_working_memory.py     #   tampon visuel
│   ├── health.py                    #   health monitoring
│   ├── working_memory.py            # ← NOUVEAU Rune : Core 4±1 chunks (Cowan)
│   ├── tiered_retriever.py          # ← NOUVEAU Rune : Core→SDM→MHN→KG→Chroma
│   ├── auto_skill.py                # ← NOUVEAU Rune : SKILL.md auto-extraction
│   └── failure_memory.py            # ← NOUVEAU Rune : anti-patterns
│
├── steering/                        # ← Steering CAA Lythea (INTACT)
├── agentic/                         # ← Agent Taëlys Lythea (INTACT)
├── web_providers/                   # ← Web search multi-fournisseurs Lythea (INTACT)
├── mcp/                             # ← MCP support Lythea (INTACT)
├── tools/                           # ← Python executor Lythea (INTACT)
├── external/                        # ← Cascade Gemini Lythea (INTACT)
├── server/                          # ← API FastAPI Lythea (sans static/)
│   ├── app.py                       #   (modifié : pas de StaticFiles)
│   └── routes.py                    #   (2755 lignes, INTACT)
│
├── cortex_ext/                      # ← Couche d'intégration Rune (NOUVEAU)
│   └── integration.py               #   RuneCortex wrap Hippocampe + extensions
│
├── perf/                            # ← Backend modèle abstrait (NOUVEAU)
│   ├── backend.py                   #   interface ModelBackend
│   ├── mock.py                      #   MockBackend (tests)
│   ├── transformers_backend.py      #   HF in-process + speculative decoding
│   └── speculative.py               #   SpeculativeDecoder (draft + verify)
│
├── agents/                          # ← SubAgent + Cron (NOUVEAU)
│   ├── subagent.py                  #   subprocess isolé + sandbox
│   ├── cron.py                      #   scheduler + consolidation post-run
│   └── _runtime.py                  #   runtime subagent
│
├── channels/                        # ← Adaptateurs omnicanal (NOUVEAU)
│   ├── base.py                      #   ChannelAdapter abstrait
│   ├── console.py                   #   CLI interactif
│   ├── telegram_channel.py          #   Telegram
│   └── slack_channel.py             #   Slack
│
├── cli/                             # ← CLI Typer (NOUVEAU)
│   └── main.py                      #   chat, run, serve, skills, cron, consolidate
│
└── utils/                           # ← Config + logging
```

**Volume total** : ~45 000 lignes Python (42 000 Lythea + 3 000 extensions Rune).

---

## Installation

```bash
# Installation complète (avec GPU)
pip install -e ".[channels,dev]"

# Installation minimale (tests sans GPU)
pip install -e ".[dev]"

# Sans channels (Telegram/Slack)
pip install -e .
```

## Démarrage rapide

### Mode console (nécessite GPU + modèle)

```bash
rune chat
```

### Mode API (FastAPI complet Lythea)

```bash
rune serve --port 7860
# API sur http://localhost:7860
```

Puis :

```bash
curl -X POST http://localhost:7860/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Bonjour"}'
```

### Gestion des skills (sans boot complet)

```bash
rune skills list
rune skills show skill_abc123
rune skills archive skill_abc123
```

### Gestion du cron

```bash
rune cron list
rune cron add --id veille --schedule "every:300s" --action "Veille tech"
rune cron run veille
rune cron start  # bloquant
```

### Status global

```bash
rune status
```

### Subagent isolé

```bash
rune run "Calcule fibonacci(10)" --lightweight
```

---

## CLI complet

```
rune chat            # mode interactif console (boot Hippocampe)
rune run "task"      # exécute une mission (subagent)
rune run "task" --lightweight  # sans boot Hippocampe
rune serve           # démarre l'API HTTP Lythea (FastAPI complet)
rune status          # statut global (léger, sans boot)

rune skills list     # liste les skills auto-appris
rune skills show ID  # détail d'un skill
rune skills archive ID

rune cron list       # liste les tâches cron
rune cron add ...    # ajoute une tâche
rune cron run ID     # exécute une tâche immédiatement
rune cron start      # démarre le scheduler (bloquant)

rune consolidate     # force un microsleep
rune deep-sleep      # force un deep sleep
```

---

## Ce qui différencie Rune

### 1. Lythea est conservé intégralement
Tous les modules cognitifs de Lythea v5.8 sont présents et fonctionnels :
- Cycle cognitif 5 phases (encode → store → retrieve → generate → consolidate)
- Mémoire multi-étages (SDM + MHN + KG + Chroma + procedural + cognitive_state)
- Steering CAA (axes, vectors, engine — hooks PyTorch)
- Métacognition (mPFC + dACC, Brier calibration)
- Deep reasoning (chaîne 4 étapes : décomposer → explorer → critiquer → synthétiser)
- Vision active (zoom cognitif VLM, 50+ langues)
- Cascade Gemini (draft-then-refine)
- Inhibition (3 niveaux en cascade)
- Predictive coding (Friston)
- Timeline + détection d'incohérence temporelle
- Planning (PFC latéral, GoalStack)
- Reflection + auto-calibration
- CRAG (Corrective RAG)
- GraphRAG (communautés thématiques)
- Consolidation microsleep/deep sleep (ripples + replay)
- MCP support
- Agent Taëlys (orchestrator + workers + verifier + blackboard + skills + sandbox)
- Web providers (Brave, DDG, SearXNG, Serper, Tavily)

### 2. Extensions Rune ajoutées
- **RuneCortex** (`cortex_ext/integration.py`) — wrap Hippocampe avec hooks pre/post generation
- **AutoSkill** (`memory/auto_skill.py`) — SKILL.md auto-extraction après succès vérifié, filtrée par métacognition
- **FailureMemory** (`memory/failure_memory.py`) — anti-patterns appris des échecs (unique)
- **TieredRetriever** (`memory/tiered_retriever.py`) — fallback strict Core → SDM → MHN → KG → Chroma
- **WorkingMemoryBuffer** (`memory/working_memory.py`) — Core 4±1 chunks (Cowan 2001)
- **SubAgentSpawner** (`agents/subagent.py`) — subprocess isolé + sandbox
- **CronScheduler** (`agents/cron.py`) — tâches de fond + consolidation post-run
- **SpeculativeDecoder** (`perf/speculative.py`) — draft + verify pour 2-3× latence
- **Channels** (`channels/`) — Console, Telegram, Slack
- **CLI Typer** (`cli/main.py`) — headless, pas d'UI

### 3. UI supprimée
- `server/static/` (HTML/CSS/JS) supprimé
- `server/app.py` modifié : plus de StaticFiles ni d'index.html
- Accès via CLI + API REST uniquement

---

## Tests

```bash
# Tests des modules Rune (sans GPU)
python -m pytest tests/test_memory.py tests/test_agents.py tests/test_backend.py tests/test_rune_integration.py -v

# Tous les tests (inclut tests Lythea qui nécessitent torch)
python -m pytest tests/ -v
```

63 tests couvrent : backend mock, working memory, tiered retriever, auto-skill,
failure memory, surprise meter, metacognition, subagents, cron scheduler,
et l'intégration RuneCortex (avec mock d'Hippocampe).

Les tests Lythea originaux (~166 tests sans GPU) nécessitent torch et sont
disponibles dans `tests/test_*.py` (66 fichiers).

---

## Configuration

Rune réutilise la config Lythea (variables d'environnement avec préfixe
`LYTHEA_`) + ajoute ses propres variables `HERMES_LYTHEA_` :

| Variable | Défaut | Description |
|---|---|---|
| `LYTHEA_*` | (cf. `.env.example`) | Config Lythea originale |
| `HERMES_LYTHEA_AUTH_TOKEN` | `` | Token d'auth API |
| `HERMES_LYTHEA_TELEGRAM_TOKEN` | `` | Token bot Telegram |
| `HERMES_LYTHEA_SLACK_APP_TOKEN` | `` | Token Slack Socket Mode |
| `HERMES_LYTHEA_SLACK_BOT_TOKEN` | `` | Token bot Slack |

---

## Limitations honnêtes

- **Backend transformers** : code présent mais non testé sur GPU dans ce sprint.
  Sur un pod RunPod avec CUDA, `rune chat` doit fonctionner mais
  demande validation.
- **Tests Lythea originaux** : 66 fichiers de tests Lythea sont présents mais
  nécessitent torch. À exécuter sur le pod.
- **CLI `chat`** : nécessite le boot complet Hippocampe (model + SDM + MHN +
  Chroma + KG). À tester sur GPU.
- **Subagent runtime** : utilise MockBackend par défaut. Pour un vrai subagent
  avec modèle, override `HERMES_SUBAGENT_BACKEND=transformers`.
- **Steering CAA** : présent dans `steering/` mais non branché dans
  `RuneCortex`. À connecter si besoin.

---

## Roadmap

### Court terme
- Validation backend transformers sur GPU réel
- Brancher steering CAA dans RuneCortex
- Tests end-to-end avec vrai modèle sur pod RunPod

### Moyen terme
- MCP client complet (cf. BACKLOG Lythea V6)
- Channels WhatsApp (Twilio), iMessage (BlueBubbles)
- Streaming SSE vrai (TextIteratorStreamer)

### Long terme
- Auto-calibration des seuils métacognition par modèle
- Multi-user (isolation mémoire par utilisateur)

---

## Licence

MIT — voir `LICENSE`.

Inspiré de :
- [Lythea](https://github.com/Benoth08/Lythea-Reasoning-Agent) v5.8 par Michaël Féré (42 000 lignes)
- [Rune](https://rune-agent.nousresearch.com/) par Nous Research
- [OpenClaw](https://openclaw.ai/) pour le concept omnicanal
