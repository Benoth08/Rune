# Rune

```
  ____    _   _   _   _   _____       _       ____   _____   _   _   _____
 |  _ \  | | | | | \ | | | ____|     / \     / ___| | ____| | \ | | |_   _|
 | |_) | | | | | |  \| | |  _|      / _ \   |  _  |  _|   |  \| |   | |
 |  _ <  | |_| | | |\  | | |___    / ___ \  | |_| | | |___  | |\  |   | |
 |_| \_\  \___/  |_| \_| |_____|  /_/   \_\  \____| |_____| |_| \_|  |_|
```

> **Agent cognitif local, headless, *open-weights only*.**
> Une boucle cognitive biomimétique (encodage -> surprise -> rappel -> génération -> consolidation) posée sur une mémoire persistante multi-étages, avec extension agentique (auto-apprentissage de compétences, sous-agents, cron).

**Statut : v0.1.0 — projet en développement actif.** Le cœur cognitif est fonctionnel et vérifié sur GPU ; plusieurs briques agentiques sont câblées mais pas encore validées en production (voir [État du projet](#état-du-projet)). Ce README distingue explicitement **ce qui marche** de **ce qui reste à faire**.

---

## Table des matières

- [En une phrase](#en-une-phrase)
- [Ce que Rune apporte](#ce-que-rune-apporte)
- [Différence avec les harnais agentiques classiques](#différence-avec-les-harnais-agentiques-classiques)
- [Aperçu — le dashboard live](#aperçu--le-dashboard-live)
- [État du projet](#état-du-projet)
- [Architecture](#architecture)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Configuration](#configuration)
- [Roadmap](#roadmap)
- [Tests](#tests)
- [Lignage](#lignage)

---

## En une phrase

La plupart des harnais agentiques sont des **orchestrateurs sans mémoire** qui pilotent une API LLM cloud. Rune est l'inverse : un **agent local sur modèles open-weights** qui exécute des missions outillées (écrire/tester du code, lancer des commandes), apprend des compétences réutilisables au fil des sessions, et coordonne son travail via un tableau noir partagé.

---

## Ce que Rune apporte

**1. Un auto-apprentissage de compétences (AutoSkill).**
Quand un échange révèle une **méthode réutilisable** (résoudre un type de problème), Rune en extrait un *skill* structuré (déclencheur, approche, validation, anti-patterns), le persiste, et le réinjecte quand une situation similaire se présente. Un filtre à deux couches (heuristique + jugement LLM) évite de transformer une simple conversation en compétence. Des garde-fous anti-injection protègent le contenu réinjecté dans le prompt.

**2. Une exécution agentique outillée.**
Rune ne se contente pas de générer du texte : il **agit**. Boucle ReAct plan → action → observation, avec des outils réels (écriture de fichiers, exécution de tests pytest, commandes shell sandboxées, recherche web). Des garde-fous encadrent l'exécution : détection de boucle avec redirection, blocage des commandes non autorisées, venv isolé par mission. Les sous-agents permettent de déléguer une tâche ponctuelle à un modèle dédié.

**3. Un tableau noir partagé (blackboard).**
Chaque mission écrit son déroulé dans un blackboard structuré (sections par agent, succès/échecs/notes) persisté sur disque pour l'audit, et suivi en direct dans le dashboard. C'est la mémoire de travail d'une mission : ce qui a été tenté, ce qui a réussi ou échoué, les notes laissées en cours de route. À chaque run, l'agent distille aussi une **leçon réutilisable** (déclencheur → approche) rangée dans une mémoire procédurale partagée entre le chat et l'agent, pour s'améliorer sur des tâches similaires.

**4. Le tout en local, sans dépendance cloud obligatoire.**
Modèles open-weights (Qwen, Mistral, Gemma, Phi, LFM...), quantification 4-bit NF4, chargement paresseux. Aucune clé API requise pour le fonctionnement de base. Une cascade optionnelle vers des modèles cloud existe mais est **désactivée par défaut**.

> **Note sur la mémoire.** Rune est bâti sur le socle cognitif de Lythéa (mémoire multi-étages : épisodique, sémantique vectorielle, graphe de connaissances). Ce qui est **actif et utilisé aujourd'hui** : le KG (graphe de connaissances, tenu à jour), la mémoire vectorielle Chroma (RAG hybride + reranking), et les skills. Les étages plus avancés — mémoire associative MHN exploitée à plein, **surprise prédictive**, **codage prédictif** (Friston), consolidation par micro-sleep — sont **présents dans le code mais pas encore mis en service** ; ils sont dormants par défaut et **arriveront dans une prochaine étape** (voir Roadmap). Le cœur de Rune aujourd'hui, c'est l'**agent** : skills, exécution outillée, blackboard, sous-agents.

---

## Différence avec les harnais agentiques classiques

> **Note d'honnêteté.** Rune s'inspire d'un harnais interne (« Hermès ») dont le code n'est pas public ici ; les comparaisons ci-dessous portent sur la **catégorie** des harnais agentiques (orchestrateurs type AutoGPT / SWE-agent / OpenHands et similaires), pas sur un produit précis dont on reproduirait des specs. L'objectif est de situer la **philosophie** de Rune, pas de dénigrer d'autres outils.

| Axe | Harnais agentiques classiques | **Rune** |
|---|---|---|
| **Exécution** | API LLM cloud (OpenAI, Anthropic...) | **Local, open-weights** ; cloud optionnel et *off* par défaut |
| **État entre sessions** | Sans mémoire, ou historique brut re-collé | **Mémoire persistante hiérarchisée** (MHN + KG + Chroma) avec cycle de vie |
| **Apprentissage** | Aucun — recommence à zéro | **AutoSkill** : extrait et réutilise des compétences vérifiées |
| **Sélection mémoire** | Fenêtre de contexte / RAG naïf | **Surprise cognitive** gouverne quoi encoder ; RAG correctif (CRAG) |
| **Objet central** | Une *tâche* à accomplir | Un *agent* avec un vécu qui s'accumule |
| **Confidentialité** | Données envoyées au cloud | **Rien ne sort de la machine** (mode local) |

**Ce que Rune ne cherche PAS à être.** Ce n'est pas un agent de codage autonome type « résous ce ticket GitHub tout seul ». C'est un **substrat cognitif** : un agent local qui apprend de ses interactions et construit une mémoire durable. Les deux approches sont complémentaires, pas concurrentes.

---

## Aperçu — le dashboard live

Rune est *headless* (pas d'interface web), mais fournit un **dashboard terminal** temps réel (`rune dashboard`) qui suit un serveur `rune serve`. Il affiche la mission en cours, les sous-agents, le blackboard partagé, l'état mémoire, la métacognition et les compétences apprises :

```
+-------------------------------------------------------------------------------+
|   ____    _   _   _   _   _____       _       ____   _____   _   _   _____      |
|  |  _ \  | | | | | \ | | | ____|     / \     / ___| | ____| | \ | | |_   _|     |
|  | |_) | | | | | |  \| | |  _|      / _ \   |  _  |  _|   |  \| |   | |          |
|  | |_ <  | |_| | | |\  | | |___    / ___ \  | |_| | | |___  | |\  |   | |        |
|  |_| \_\  \___/  |_| \_| |_____|  /_/   \_\  \____| |_____| |_| \_|  |_|         |
|                                                                               |
|         Rune v0.1.0  .  Trinity: OFF (single-model)  .  19:04:09              |
+-------------------------------------------------------------------------------+
+- Mission courante -----------------------+ +- Mémoire -------------------------+
| Task  : Analyser le fichier de logs      | | MHN    : 128 / 512  patterns      |
| Phase : retrieval -> qwen2.5-7b          | | KG     : 87 entités . 143 rel.    |
| Elapsed : 2.4s   |   Tokens : 312        | | Chroma : 1 204 chunks             |
+------------------------------------------+ | Skills : 2 actifs . 1 fiable      |
+- Subagents ------------------------------+ +-----------------------------------+
|  #  Section        Statut     Modèle     | +- Métacognition -------------------+
|  1  analyse_logs   done       qwen-7b    | | Doute       : 0.18  (certaine)    |
|  2  synthèse       run        qwen-7b    | | Surprise    : 0.42                |
+------------------------------------------+ |  struct 0.31 . épisod 0.55        |
+- Blackboard -----------------------------+ |  prédict --  . chroma -0.12       |
| Section      Wins  Fails  Notes  Statut  | +-----------------------------------+
| _contract    -     -      -    objectif  | +- Skills actifs -------------------+
| analyse_logs 3     0      1    validé    | | ID          Trigger      Succ Conf|
| synthèse     1     0      0    en cours  | | skill_e9d9  débug segf.   5  0.87 |
+------------------------------------------+ | skill_20de  requête SQL   1  0.70 |
+- Logs -----------------------------------+ +-----------------------------------+
| 19:04:07  Retention GC: 0 stale chunks   |
| 19:04:08  Skill applied: skill_e9d9.     |
| 19:04:09  Consolidation: 2 -> Chroma     |
+------------------------------------------+
```

> Illustration représentative de la disposition réelle (`rune/cli/dashboard.py`). Les valeurs dépendent de l'état courant. `prédict --` reflète que la composante prédictive de la surprise est inerte (voir Roadmap).

---

## État du projet

L'honnêteté sur la maturité est un principe du projet. **« Couvert par les tests » et « vérifié en production » sont deux choses différentes** : la suite compte **1485 tests unitaires au vert**, mais plusieurs briques n'ont pas encore été exercées avec un vrai modèle sur GPU.

### Fonctionnel et vérifié bout-en-bout (sur GPU)

| Brique | Détail |
|---|---|
| **Boot complet** | Chargement Hippocampe (MHN, KG, Chroma), modèle open-weights, quantif 4-bit |
| **Chat cognitif** | Streaming, rappel mémoire ciblé (KG + Chroma + MHN), identité stable |
| **Mémoire persistante** | KG + Chroma opérationnels (RAG hybride + reranking + CRAG) ; MHN câblé. *Voir la note sur la mémoire ci-dessus pour les étages avancés à venir.* |
| **AutoSkill** | Extraction, persistance JSON, réinjection, filtre anti-conversation, garde-fous sécurité |
| **Blackboard** | Sections par agent (succès/échecs/notes), persistance JSON pour l'audit, suivi live au dashboard |
| **Rétention Chroma** | Cycle de vie création->accès->purge, consolidés permanents, mode dry-run de sûreté |
| **KG à jour** | Dédoublonnage des faits + supersession des prédicats fonctionnels (un employeur récent périme l'ancien) |
| **Agent + missions** | Boucle ReAct outillée (fichiers, tests, shell sandboxé), sous-agents, orchestrateur, suivi live |
| **CLI** | `chat`, `serve`, `status`, `skills`, `dashboard`, `run` |

### Câblé mais non vérifié en production

| Brique | Statut |
|---|---|
| **Sous-agents** | Spawner câblé ; `--lightweight` utilise un backend *mock* — jamais exercé avec un vrai modèle |
| **Cron / scheduler** | `CronScheduler` intégré au runtime, non déclenché en conditions réelles |
| **Dashboard live** | Rendu implémenté ; nécessite `rune serve` actif en parallèle |

### Présent mais dormant (par conception ou configuration)

| Brique | Raison |
|---|---|
| **Trinity** (multi-modèles) | `OFF` par défaut — mode single-model |
| **Cascade Claude/Gemini** | Désactivée par défaut (`enable_cascade=False`) — nécessite des clés API |
| **Recherche web** | Chaîne composite en place (SearXNG -> DDG...) mais SearXNG public peu fiable ; recommandé : clé Tavily/Serper |
| **Predictive coding** (Friston) | Désactivé par défaut |
| **SDM / surprise prédictive** | Inerte — mais un *juge d'ablation* est câblé (`pc_gating_w_sdm`, off par défaut) pour mesurer si le signal SDM apporte de la variance discriminante avant de trancher |

---

## Architecture

```
rune/
├── hippocampe.py            # Orchestrateur du cycle cognitif
├── model.py                 # Wrapper modèles HF (4-bit, streaming, cascade)
├── cognition/
│   ├── encoding.py          #   Encodage + filtre de salience
│   ├── surprise.py          #   Surprise composite (structurelle + épisodique + prédictive)
│   ├── retrieval.py         #   RAG hybride + CRAG + rafraîchissement d'accès
│   ├── generation.py        #   Génération
│   ├── storage.py           #   Écriture mémoire pondérée par la surprise
│   ├── consolidation.py     #   Micro-sleep : MHN -> Chroma, GC de rétention
│   └── predictive_coding.py #   Codage prédictif (Friston, désactivé par défaut)
├── memory/
│   ├── mhn.py               #   Modern Hopfield Network (épisodique)
│   ├── sdm.py               #   Sparse Distributed Memory (inerte, en réévaluation)
│   ├── tiered_retriever.py  #   Fallback Core -> SDM(inerte) -> MHN -> KG -> Chroma
│   ├── auto_skill.py        #   AutoSkill : extraction + garde-fous
│   ├── failure_memory.py    #   Patterns d'échec réinjectés en avertissement
│   └── salience.py          #   Cascade anti-bruit N1/N2/N3
├── agents/
│   ├── subagent.py          #   Sous-agents (spawner)
│   └── cron.py              #   Tâches planifiées
├── cortex_ext/
│   └── integration.py       #   RuneCortex : couche agentique sur Hippocampe
├── server/                  #   API HTTP headless (FastAPI) + auth
└── cli/
    ├── main.py              #   Commandes Typer
    └── dashboard.py         #   Dashboard terminal live
```

---

## Installation

Testé sur RunPod (image CUDA, Python 3.12, PyTorch 2.8).

```bash
# 1. Décompresser puis installer (crée la commande `rune`)
tar -xzf rune_v0_1_1_tar.gz
cd rune
pip install -e . --no-deps --break-system-packages

# 2. (Optionnel) déploiement complet des dépendances ML
bash deploy.sh
```

> Sans `pip install -e .`, la commande `rune` n'existe pas — repli : `python3 -m rune.cli <commande>`.

---

## Utilisation

```bash
# Chat interactif (charge un modèle par défaut ; --searxng pour la recherche web locale)
rune chat --model Qwen/Qwen2.5-7B-Instruct

# Serveur HTTP headless (API /api/*, doc interactive sur /docs)
rune serve

# Dashboard live (dans un 2e terminal, pendant que le serveur tourne)
rune dashboard

# Compétences apprises
rune skills list
rune skills show <skill_id>

# État du système
rune status
```

**Interagir sans UI :** Rune est headless. On l'utilise via la CLI (`rune chat`), l'API (`POST /api/chat`, `POST /api/models/load`), ou la doc interactive Swagger sur `/docs`.

---

## Configuration

Tout se règle via un fichier `.env` (préfixes `RUNE_` et `LYTHEA_`). Extraits utiles :

```bash
# Modèle par défaut du chat
RUNE_DEFAULT_MODEL=Qwen/Qwen2.5-7B-Instruct

# Recherche web : "auto" = chaîne composite avec fallback (recommandé)
LYTHEA_WEB_PROVIDER=auto
# Pour une recherche fiable (gratuit, sans CB) : https://tavily.com
# LYTHEA_TAVILY_API_KEY=tvly-...

# Rétention mémoire (jours avant purge d'un chunk non consulté)
RUNE_RETENTION_TTL_DAYS=30
# Premier passage en observation, sans rien supprimer :
# RETENTION_GC_DRY_RUN=1

# Token HuggingFace (accélère les téléchargements, supprime le rate-limit)
# HF_TOKEN=hf_...
```

---

## Roadmap

Prochaines étapes, par ordre de priorité :

1. **Valider les sous-agents avec un vrai modèle** — remplacer le chemin mock par une exécution réelle, tester une mission multi-sections.
2. **Exercer le cron** — vérifier le déclenchement de tâches planifiées en conditions réelles.
3. **Recherche web fiable** — intégration Tavily/Serper de première classe, SearXNG local automatique.
4. **Trancher le sort de la SDM** — soit ablation propre du prédictif, soit recâblage causal de la surprise dans la cascade de salience (boost N2 / assouplissement N3, jamais d'override du filtre anti-bruit N1).
5. **Consolidation MHN -> Chroma** — durcir la promotion des souvenirs marquants.
6. **Cascade cloud optionnelle** — chemin Claude/Gemini pour les cas où le modèle local plafonne.

---

## Tests

```bash
# Suite complète
python3 -m pytest tests/ -q

# 1485 tests au vert (couverture unitaire)
```

> Rappel : les tests unitaires ne remplacent pas la validation GPU. Plusieurs bugs réels (format d'events, sérialisation numpy, config de logging, entry point CLI) n'étaient visibles qu'à l'exécution sur matériel — d'où l'importance de la section État du projet.

---

## Lignage

Rune étend **Lythea** (assistant LLM local à cycle cognitif biomimétique) d'une couche agentique inspirée d'un harnais interne. Le cœur cognitif (mémoire, surprise, consolidation, RAG/CRAG, steering) vient de Lythea ; Rune y ajoute l'AutoSkill, les sous-agents, le cron et le mode headless.

---

*Projet en développement actif — les interfaces et le périmètre peuvent évoluer.*
