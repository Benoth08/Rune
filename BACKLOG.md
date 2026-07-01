# Backlog — Améliorations différées

Ce fichier liste les items identifiés pendant le sprint étape 8 mais
**volontairement reportés** : soit parce qu'ils sortent du scope du
refactor, soit parce qu'ils nécessitent des conditions externes
(autre pod, plus de modèles à comparer, plus de données utilisateur)
qui ne sont pas réunies aujourd'hui.

---

## 🔌 V6+ Couche MCP — Services tiers (Slack, Drive, GitHub, Notion)

**Statut** : identifié sur recommandation utilisateur (mai 2026, suite à
l'analyse panorama RAG 2026).

**Constat 2026** :

Le Model Context Protocol (MCP) a explosé depuis sa sortie en
novembre 2024. État du marché à fin mai 2026 :

- **10 000+ serveurs MCP** publics en production
- **97M téléchargements SDK / mois** sur le 2e semestre 2025
- **Anthropic, OpenAI, Google, Microsoft, AWS** adoptent tous le standard
- Donation à la **Linux Foundation** (Agentic AI Foundation, déc 2025)
  → gouvernance neutre, plus de risque vendor lock-in
- Surnom officieux : *"USB-C de l'IA"*

**Pourquoi ça change la donne pour Lythéa** :

Aujourd'hui, chaque outil que tu ajoutes (web_search, python_executor)
demande son module Python custom, son intégration dans le router, ses
tests. Avec MCP, tu pourrais :

- Hériter d'un **catalogue de centaines de serveurs déjà testés**
  (Slack, Drive, GitHub, Notion, PostgreSQL, Jira, Salesforce…)
- Brancher en **quelques heures au lieu de semaines** chaque nouveau
  service tiers
- **Architecture future-proof** : tout l'industrie converge dessus,
  Lythéa parlerait le même protocole que Claude Desktop, Cursor,
  ChatGPT Apps, etc.

**Bénéfice symétrique** : tu pourrais aussi exposer Lythéa **comme
serveur MCP** (mémoire SDM/MHN/KG/Chroma → outils MCP queryables par
d'autres clients). Lythéa deviendrait un *"persistent memory MCP
server"*, utilisable depuis Claude Desktop par exemple.

**Coûts honnêtes à connaître** :

Ce n'est pas trivial. Pour faire du MCP propre, il faut :

1. **Un client MCP côté Lythéa** qui parle JSON-RPC 2.0 — donc soit
   `mcp` SDK Python (officiel), soit implem manuelle
2. **Transport** : stdio pour les serveurs locaux (simple) OU HTTP/SSE
   pour les serveurs distants (OAuth 2.1, sessions persistantes)
3. **Auth** : OAuth 2.1 obligatoire pour les serveurs distants
   (Slack, Drive, etc.) — flow de consent, refresh tokens, scope
   par tool
4. **Sécurité** : la spec exige une **séparation host/protocol** :
   c'est Lythéa qui doit gérer le consent utilisateur, pas les
   serveurs MCP eux-mêmes
5. **Gestion de processus** : les serveurs stdio sont des subprocess
   à orchestrer (lifecycle, restart, logs)

**Limite spécifique au déploiement RunPod** :

Tu héberges Lythéa sur un pod RunPod **ponctuel**. Les MCP servers
HTTP/SSE de production (genre Slack officiel) sont conçus pour des
deployments stables avec OAuth persistant. Pour le stdio MCP local
ça marche très bien (Git, filesystem, etc.), mais l'aspect "remote
SaaS connecté" demande un setup OAuth qui est moins évident en
self-hosted ponctuel.

**Solutions** :
- Pour services locaux (Git, Postgres local, filesystem) : **stdio**,
  ça roule
- Pour SaaS (Slack, Drive, Notion) : soit auto-héberger les serveurs
  MCP sur un VPS stable séparé, soit utiliser les serveurs hébergés
  par les éditeurs SaaS (modèle qui se généralise en 2026)

**Plan d'implem proposé (sprint dédié V6)** :

**Phase 1 — Client MCP côté Lythéa (~3-4 jours)**
- Module `lythea/mcp/client.py` utilisant le SDK officiel `mcp`
- Refactor du router V5.1 pour qu'un outil puisse être *soit* un
  module Python interne (comme aujourd'hui), *soit* une référence
  vers un tool MCP
- Configuration `mcp_servers.yaml` listant les serveurs à lancer/contacter
- Lifecycle : démarrage des serveurs stdio en subprocess, health checks,
  restart auto

**Phase 2 — 2-3 serveurs pilotes (~2-3 jours)**
- **Git MCP** (officiel Anthropic, stdio) : pour permettre à Lythéa
  d'interagir avec ses propres repos
- **Filesystem MCP** (officiel) : lire/écrire dans des dossiers
  désignés
- **Fetch MCP** (officiel) : remplace partiellement `lythea/web.py` —
  test si l'expérience est meilleure

**Phase 3 — Sélection sémantique des tools MCP (~2 jours)**
- Le router V5.1 doit étendre les routes pour découvrir les tools
  MCP dynamiquement (au lieu de hardcoder web/python/none)
- Au démarrage, query chaque serveur MCP pour son manifest, génère
  des embeddings d'exemples
- Cache au démarrage, refresh sur changement de config

**Phase 4 — Documentation + tests (~1-2 jours)**

**Total estimé** : 1-2 semaines de sprint dédié, à faire quand tu
auras un cas d'usage concret. *"Je voudrais que Taëlys puisse lire
mes notes Notion / fichiers Drive"* est le déclencheur typique.

**Validation préalable conseillée** :

Avant d'investir, joue avec un client MCP existant (Claude Desktop)
pendant quelques semaines pour te faire une idée concrète des
patterns d'usage. Beaucoup de gens découvrent que ce qu'ils croyaient
vouloir (intégration Drive complète) se réduit en pratique à 2-3
opérations spécifiques (chercher un doc, lire son contenu) qui ne
justifient pas forcément MCP.

**Si tu n'as pas besoin d'apps SaaS tierces** : reste sur ton archi
actuelle. MCP est une couche d'**intégration**, pas une amélioration
fondamentale de la cognition de Lythéa.

**Fichiers concernés (futurs)** :
- Nouveau package `lythea/mcp/` (client, manager, lifecycle)
- Modif `lythea/cognition/semantic_router.py` (routes dynamiques)
- Modif `lythea/cognition/tool_dispatcher.py` (extension VALID_TOOLS)
- Modif `lythea/hippocampe.py` (`_route_tool_v5_1` → délégation MCP)
- Config `mcp_servers.yaml` à la racine

**Références** :
- Spec officielle : https://modelcontextprotocol.io
- Registre serveurs : https://modelcontextprotocol.io/servers
- SDK Python : `pip install mcp`
- Anthropic blog : https://www.anthropic.com/engineering/code-execution-with-mcp

---

## 🎨 V6+ Multimodal RAG — Index unifié vision + texte

**Statut** : identifié sur le panorama LinkedIn 5 RAG Architectures
(Brij Kishore Pandey, 2026) — Lythéa couvre cette archi à 30%.

**État actuel V5.2** :
- ✅ Captioner BLIP (image → texte) déjà branché dans le flow chat
- ❌ Index multimodal partagé (CLIP, ColPali) avec embeddings
  vision + texte dans le même espace
- ❌ Retrieval direct sur images sans passer par le captioning
- ❌ Tables comme citoyens de première classe (extraction structurée)

**Limitation** : aujourd'hui le flow image perd l'info visuelle riche.
Une image → caption texte → texte indexé. Pas moyen de retrouver une
image par sa similarité visuelle pure.

**Solution** :
- Backend embedding multimodal (CLIP-ViT-B-32 ou ColPali si dispo)
- Doubler l'index Chroma ou utiliser collection multimodale
- Reranking adaptatif selon le type de query

**Effort estimé** : plusieurs semaines (changement backend vectoriel).
À envisager si le use-case devient visuel-heavy. Sinon BACKLOG long
terme.

---

## 🧠 V6+ Agentic RAG — Chaînage d'outils autonome

**Statut** : V5.1 a posé la fondation (router multi-outils web/python),
V6 visera le **chaînage** automatique.

**État actuel V5.2** :
- ✅ Router sémantique (MiniLM) + JSON dispatcher (Qwen)
- ✅ 2 outils branchés (web search, python interpreter)
- ❌ Loop "agent_loops_until_confident" pas implem
- ❌ Pas de planner agent qui décompose une question en plan
- ❌ Pas de reasoner final qui synthétise plusieurs résultats outils

**Évolutions à prévoir** :
1. **Function calling natif** Qwen2.5/Qwen3 (déjà supporté par les
   modèles, juste à brancher via le chat template)
2. **Planner LLM** : décompose la question en N étapes (web → python
   → web → final)
3. **Loop ReAct** : à chaque étape, le modèle décide de l'outil suivant
   ou de finaliser. Avec garde-fou max_steps=4 pour éviter les
   boucles infinies.
4. **UI temps-réel** qui affiche la chaîne d'outils en cours
   (pas juste 🐍 ou 🌐 isolés)

**Effort estimé** : sprint dédié 1-2 semaines avec tests fiabilité
function calling par modèle du catalog.

---

## 🔬 V5.3 NLI zero-shot router (alternative au LLM dispatcher)

**Statut** : insight identifié pendant la discussion archi V5.1
(17 mai 2026).

**Constat** : le slow-path LLM dispatcher V5.1 fait ~300-500ms et coûte
des tokens. Un modèle NLI dédié (mDeBERTa-v3-base-mnli-xnli, ~280M,
~700 MB VRAM) ferait la même décision en ~25ms, déterministe.

**Candidats** :
- `mDeBERTa-v3-base-mnli-xnli` (~700 MB, multilingue FR/EN, ~25ms GPU)
- `bart-large-mnli` (~800 MB, EN only)
- `xtremedistil-l6-h256-uncased` (~50 MB, ~3ms, EN principalement)

**Architecture cible** : cascade à 3 étages
```
Regex → NLI (HC: tranché) → LLM dispatcher (zone grise) → fallback
```

**Validation préalable** : utiliser le logging `WEB_DECISION` V5 pour
mesurer combien de fois le slow-path se déclenche réellement. Si
< 5% des questions → pas nécessaire. Si > 15% → ROI clair pour NLI.

**Fichiers concernés** :
- Nouveau module `lythea/cognition/nli_router.py`
- Modif `_route_tool_v5_1` dans `hippocampe.py` (cascade)

---

## 🔗 V4.5 Citation paresseuse en mode direct (tech_reco)

**Statut** : identifié pendant la validation finale du renforcement
contextuel V4.4 (17 mai 2026, Qwen2.5-7B-Instruct, mode direct sans
raisonnement, question *"Recommande-moi un modèle NER en français"*).

**Constat** :

Le renforcement contextuel V4.4 (directive *"[Vigilance accrue]"*
injectée dans le system prompt quand `tech_reco` se déclenche) a
**éliminé les confabulations majeures** (plus de variantes inventées
type `fr_core_news_md/lg` non vérifiées). Citations correctes pour
`flair/ner-french [1]` et `DrBERT [2]`.

Mais le modèle reste **paresseux dans le référencement** : il a cité
`spaCy fr_core_news_sm` sans plaquer `[3]` à côté, alors que `[3]`
dans les sources web pointait vers *"French · spaCy Models
Documentation"* (`spacy.io/models/fr/`) qui mentionne effectivement
ce modèle. Donc :

- Pas une confabulation (le modèle existe, source disponible)
- Juste un référencement incomplet (le modèle a omis le `[N]`
  alors qu'il aurait pu le lier)

La directive V4.4 disait : *"Si tu cites un nom de mémoire sans le
voir dans les résultats, dis-le explicitement"*. Le modèle a appliqué
cette règle mais en mode permissif : il n'a pas dit *"sans source
précise"* (puisqu'une source existait) mais n'a pas non plus lié `[3]`.
Comportement à mi-chemin, fonctionnellement correct mais imparfait.

**Solution proposée** :

Renforcer la directive avec une clause explicite *"liaison
obligatoire des [N] à chaque nom cité quand la source est présente"*.
Risque : durcir trop pourrait rendre la réponse aussi laconique que
le mode raisonnement (qui ne cite qu'un seul modèle, cf. autre item
BACKLOG ci-dessous). Tester d'abord la formulation puis valider qu'on
ne sur-corrige pas.

**Fichiers concernés** :
- `lythea/hippocampe.py` — directive `[Vigilance accrue]` ligne ~1335

---

## 🔍 V4.5 Mode raisonnement trop conservateur sur recommandations

**Statut** : identifié pendant la validation finale (17 mai 2026,
Qwen2.5-7B-Instruct + toggle raisonnement, question *"Recommande-moi
un modèle NER en français"*).

**Constat** :

Le mode raisonnement Lythéa (`DeepReasoningChain`) est **excellent en
précision** mais perd en utilité sur les questions de recommandation.
Comparaison directe :

| Mode | Modèles cités | Doute |
|---|---|---|
| Direct (V4.4 + renforcement) | 3 (`fr_core_news_sm`, `flair`, `DrBERT`) | 11% |
| Raisonnement | 1 (`flair`) | 6% |

La phase critique de `DeepReasoningChain` **élague trop**. Le doute
plus bas (6%) confirme la haute confiance, mais au prix d'une perte
d'information utile : DrBERT et la doc spaCy étaient dans les sources
et auraient mérité d'être mentionnés.

C'est l'inverse du problème de confabulation : ici on a une
sur-prudence qui amputerait les réponses.

**Pistes d'investigation** :

1. **Ajuster les prompts des étapes critique/synthèse** pour
   préserver la diversité quand le contexte web fournit plusieurs
   options pertinentes. Aujourd'hui le prompt de critique force le
   modèle à "filtrer", il faudrait nuancer entre "filtrer les
   inventions" et "préserver les options réelles".
2. **Détecter le type de question** : si la question est ouverte
   (*"recommande-moi"*), la critique doit garder plusieurs candidats.
   Si elle est fermée (*"quel est le meilleur"*), elle peut élaguer.

**Fichiers concernés** :
- `lythea/cognition/deep_reasoning.py` — prompts critique/synthèse
- Tests à enrichir dans `tests/test_deep_reasoning.py`

---

## 🧠 V4.5+ Auto-raisonnement sur questions à risque de confabulation

**Statut** : insight identifié pendant l'A/B test V4.4 (17 mai 2026,
Qwen2.5-7B-Instruct sur 3 questions, mode raisonnement vs mode direct).

**Constat** :

L'A/B test a montré que le mode raisonnement Lythéa (`DeepReasoningChain`)
agit comme **filtre anti-confabulation efficace** sur les questions
techniques. Exemple terrain — *"Recommande-moi un modèle NER en français"* :

- **Avec raisonnement** : ne cite que `flair/ner-french` et `DrBERT`,
  deux modèles vérifiables dans les sources web fournies. La phase
  critique a filtré les extrapolations.
- **Sans raisonnement** : ajoute spaCy `fr_core_news_sm/md/lg` sans
  étayage web. C'est une sur-généralisation depuis un snippet
  partiel — les variantes md/lg ne sont pas vérifiées dans les
  sources mais sont citées comme recommandations.

Pour les autres types de questions (calcul simple, concepts stables),
le raisonnement n'apporte rien de mesurable mais coûte 17-50s vs
réponse instantanée.

**Solution V4.4 (déjà appliquée, partielle)** :

Injection contextuelle d'une mini-directive dans le system prompt
quand `tech_reco` se déclenche. Discipline le modèle sans coût.
Validation à faire sur plus de cas pour mesurer l'efficacité.

**Solution V4.5+ proposée (plus ambitieuse)** :

Activer automatiquement le mode raisonnement Lythéa (toggle UI) quand
le routeur détecte un cas à fort risque de confabulation : `tech_reco`,
`citation académique`, et peut-être les questions où `assess_complexity`
retourne ≥ 3 étapes. L'utilisateur peut désactiver via un nouveau
réglage *« raisonnement auto »* dans le panel sampling.

**Décisions ouvertes** :
- Faut-il un timeout max (genre 30s) pour ne pas faire attendre
  trop longtemps sur des cas légers ?
- Garder le toggle UI manuel en parallèle ou le remplacer ?
- Cumul avec le renforcement contextuel V4.4 ou substitution ?

**Validation préalable nécessaire** :

Mesurer d'abord l'efficacité de l'injection contextuelle V4.4 sur
un échantillon élargi de questions tech avant d'investir dans la
mécanique plus lourde du raisonnement auto. Si le renforcement
prompt suffit dans 80% des cas, le raisonnement auto devient
optionnel.

**Fichiers concernés** :
- `lythea/hippocampe.py` — branchement auto via `web_reason` et
  `assess_complexity`
- `lythea/server/routes.py` — endpoint `/api/config/reasoning-auto`
- `lythea/server/static/app.js` — toggle UI dans panel paramètres

---

## 🎨 V4.5 UX — Feuille de Route affichant le brouillon non corrigé

**Statut** : identifié pendant les tests V4.4 du mode raisonnement Lythéa
sur LLM classiques (17 mai 2026, session debug confabulation).

**Constat** :

La chaîne `DeepReasoningChain` fonctionne par phases successives
(décomposition → exploration → critique → synthèse). La phase critique
joue son rôle : elle détecte et corrige les erreurs d'exploration.
Exemple terrain (Mika, Qwen2.5-7B-Instruct + mode raisonnement) :

> Question : « Combien font 17 × 23 + 89 ÷ 11 ? »

La **Feuille de Route** affichée dans l'UI montre :
> *"Correction : 17 × 23 = 391, 89 ÷ 11 ≈ 8.0909, 391 + 8.0909 = **400.0909**"*

Puis la **réponse finale** affiche correctement :
> *"391 + 8.0909 = **399.0909**"*

L'erreur d'exploration (400 au lieu de 399) a été silencieusement corrigée
par la phase critique. La réponse finale est juste. Mais l'utilisateur voit
le brouillon erroné dans la feuille de route s'il l'ouvre, ce qui peut être
déroutant — il a l'impression que Lythéa s'est trompée alors qu'en réalité
le pipeline a corrigé l'erreur comme prévu.

**Deux écoles possibles pour la solution** :

1. **Filtrage** : n'afficher que la feuille de route post-critique dans
   l'UI, en gardant le brouillon en interne pour les logs/debug. Plus
   propre côté UX mais perd la transparence sur le processus.

2. **Annotation visuelle** : afficher le brouillon ET la correction côte
   à côte, avec une icône qui montre que la critique a corrigé. Plus
   pédagogique (l'utilisateur voit comment Lythéa pense et se corrige)
   mais plus complexe à implémenter.

Décision reportée : à choisir avec Mika selon le positionnement
voulu (cleaner vs transparent).

**Fichiers concernés** :
- `lythea/cognition/deep_reasoning.py` — phase critique produit déjà
  une version corrigée, elle est juste pas exposée distinctement
- `lythea/server/static/app.js` — rendu de la feuille de route
- `lythea/hippocampe.py` — émission des events cognitifs

---

## 📅 V4.3 Timeline — Détection d'incohérence temporelle

**Statut** : identifié pendant les tests manuels V4.0.1 in vivo (10 mai 2026).

**Constat** :

Le module Timeline extrait correctement les dates absolues, relatives
et les durées. Mais il ne **vérifie pas** la cohérence entre dates
multiples mentionnées dans le même message.

Exemple terrain (Mika, 10 mai 2026) :
> "Hier on a soutenu la réunion du 12 mai 2026"

Lythéa paraphrase sans tiquer alors que :
- "hier" = 9 mai 2026 (puisqu'on est le 10)
- "12 mai 2026" = +2 jours dans le futur
- → Incohérence temporelle évidente

Le module connaît la date du jour (via `[Conscience du temps]` V3.9.4)
mais Timeline ne croise pas cette info avec les marqueurs extraits.
Le LLM Qwen2.5-3B est trop petit pour faire l'inférence tout seul.

**Solution proposée** (~30 lignes dans `timeline.py`) :

Quand au moins 2 marqueurs temporels sont extraits dans le même message :
- Calculer leur date ISO normalisée (déjà fait)
- Si une date relative passée ("hier", "il y a X") cohabite avec une
  date absolue future (>0 jours par rapport à `now`) → flagger une
  incohérence dans le bloc rendu :

```
[Chronologie]
📅 2026-05-12 (12 mai 2026)
⏱ il y a 1 jour (hier) — 2026-05-09
⚠️ Incohérence : "hier" (2026-05-09) ne correspond pas à "12 mai 2026"
```

Le LLM voit le warning explicite → il en parle dans sa réponse au
lieu de paraphraser bêtement.

**Pourquoi reporté** : non bloquant en V4.0.1 et représente une vraie
amélioration de produit, pas une régression. À traiter dans V4.0.2.

---

## 🪞 V4.4 Métacognition — Seuils mal calibrés pour Qwen 3B

**Statut** : identifié pendant les tests manuels V4.0.1 (10 mai 2026).

**Constat** :

Les seuils par défaut (`metacog_low_doubt: 0.15`, `high: 0.35`,
`very_high: 0.55`) sont calibrés pour des modèles type 7B+ qui
*savent qu'ils ne savent pas*. Sur Qwen2.5-3B :

- Question triviale ("capitale de la France") : `confidence_label: très_certaine` (doubt ≈ 0.05) ✅
- Question impossible ("cuisine des Sentinelles d'Andamane, peuple isolé sans contact extérieur") : `confidence_label: très_certaine` (doubt ≈ 0.10) 🔴

Qwen 3B **invente avec aplomb** quand il ne sait pas — donc le
`doubt_index` reste bas, donc la métacognition voit toujours "très
certaine".

**Pistes de correction** :

1. **Quick win** — réduire fortement les seuils dans `.env` pour ce modèle :
   ```
   LYTHEA_METACOG_LOW_DOUBT=0.05
   LYTHEA_METACOG_HIGH_DOUBT=0.10
   LYTHEA_METACOG_VERY_HIGH_DOUBT=0.18
   ```

2. **Vraie correction** — enrichir `MetacognitiveDecision` avec d'autres
   signaux que `doubt_index` seul :
   - **KG-orphan** : si l'entité principale de la question est absente
     du KG → drapeau d'incertitude fort
   - **Web-rejected** : si le RAG/web n'a rien retourné pertinent →
     drapeau d'incertitude
   - **Embedding distance** : si la question est sémantiquement loin
     de tout ce qu'il y a en mémoire active

3. **Calibration adaptative** — apprendre les seuils par modèle en
   observant la distribution de `doubt_index` sur les premiers N tours
   et placer les bandes au 25/50/75 percentile.

**Pourquoi reporté** : tracking marche (n_entries: 2 après 2 tours,
calibration_score remonté à 0.99). Le module est **utile pour
l'observabilité** dès maintenant. La discrimination des bandes
d'incertitude est l'amélioration produit suivante.

---

## ⚙️ V4 — Endpoint debug pour exposer l'état affectif courant

**Statut** : identifié pendant les tests manuels V4.0.1.

**Constat** :

`/api/config/v4` expose `cognitive_state.enabled` + sa config, mais
**pas** l'état runtime de `lythea_affect.current` (valence, arousal,
target). Pour valider l'anti-sycophant en in vivo, il a fallu poser
la question "comment te sens-tu ?" et déduire de la réponse texte.

**Solution** (~10 lignes) : ajouter dans `Hippocampe.v4_status()` :
```python
"cognitive_state": {
    ...
    "lythea_affect_now": (
        self._cognitive_state.lythea_affect.current.to_dict()
        if self._cognitive_state else None
    ),
}
```

Ça permettra aux harnais E2E d'asserter directement `valence < 0.6`
après burst sycophant au lieu de lire des réponses LLM ambiguës.

**Pourquoi reporté** : non bloquant, c'est de l'observabilité pure.

---

## 🎯 V4.0.c Planning — Goal advance via commande / bouton UI

**Statut** : identifié dès le design V4.0.c, confirmé in vivo.

**Constat** :

`GoalStack.advance_step()` existe et est testé (53 tests verts) mais
**rien ne l'appelle depuis le chat** : ni le LLM (qui n'a pas la
notion d'API), ni l'UI (pas de bouton). Le `current_step` reste
bloqué à 0 même quand l'utilisateur dit "j'ai fini la première
étape".

**Solutions possibles** :

1. **Commande explicite** `/done` : Lythéa l'intercepte et appelle
   `goal_stack.advance_step()`.
2. **Classifier d'avancement** : pattern matching sur "j'ai fini X",
   "ok pour la 1", "next" → trigger automatique.
3. **Bouton UI** "✓ Marquer fait" à côté du goal actif dans le
   diagnostic Cognition.

La 1+3 combinée est probablement le sweet spot.

**Pourquoi reporté** : V4.0.1 valide que le goal **est créé**
correctement. Le cycle complet (création → avancement → complétion)
est une feature suivante.

---

## 🔬 Reasoning v2 — Repenser le système de chain-of-thought

**Statut** : observations terrain accumulées, pas encore de design solide.

**Constat** :

Le système de two-pass reasoning actuel (toggle "Réflexion" dans l'UI)
applique un seul prompt générique avant la génération finale. Tests
en prod sur 3 modèles ont montré qu'il est :

- ❌ **Nuisible** sur Mistral-7B-Instruct : pass 1 et pass 2 déconnectés,
  Mistral confabule sous reasoning (invente une "conversation il y a
  quelques minutes"), augmente la sycophantie déjà présente
- ❌ **Nuisible** sur LFM2-2.6B : amplifie la confabulation native
  (expositions inventées, etc.)
- ⚠️ **Sans effet visible** sur Qwen2.5-Instruct : les pass 1/2 sont
  redondants
- ✅ **Efficace** sur les modèles thinking-natifs (Qwen3, R1, Jamba),
  mais ils n'utilisent **pas** le two-pass — ils ont leur propre `<think>`

**Pistes pour v2** (par ordre d'effort croissant) :

1. **Gate de pertinence** : avant de déclencher le two-pass, classifier
   la requête. Conversation simple → skip ; calcul/raisonnement
   multi-étape → enable. Heuristiques candidates : tokens-marqueurs
   ("calcule", "explique pourquoi", "compare"), longueur du message,
   score de surprise globale, présence d'un RAG context riche.

2. **Template structuré** : remplacer le prompt libre par un template
   contraint en français ("Rappel des faits / Demande / Plan"). Borne
   la longueur du pass 1, évite la pseudo-philosophie.

3. **RAG dans le pass 1** (le plus impactant) : actuellement le RAG
   context est injecté dans le prompt du pass 2 mais pas du pass 1.
   Donc le modèle "réfléchit sans avoir vu sa mémoire", puis "répond
   avec sa mémoire" — d'où la confabulation. Inverser l'ordre :
   pass 1 reçoit RAG + KG, pass 2 produit la réponse finale en langage
   naturel sans pouvoir s'écarter du cadrage.

4. **Self-refine** : pass 3 d'auto-évaluation ("la réponse est-elle
   sourcée par le contexte ?"). Coûteux en latence (3x), à réserver
   aux questions critiques.

**Critères de succès** : la sycophantie de Mistral disparaît, LFM2
arrête de confabuler, le two-pass devient utile sur Qwen2.5.

**Avant d'attaquer** : faire des **sessions de test multi-modèles**
sur des tâches variées (conversation, calcul, RAG-heavy, raisonnement
multi-étape) avec et sans le toggle, pour avoir une vraie data avant
de redesigner.

---

## 🧩 Jamba Reasoning 3B — Validation runtime

**Statut** : intégré au CATALOG, **chargement validé** (28 layers,
hidden_dim=2560, profil sampling auto-appliqué). **Génération
non validée** : les kernels Mamba CUDA refusent de tourner sur le
pod RunPod actuel (cu124 + torch 2.4.1 + transformers récent).

**Erreurs rencontrées et tentatives** (chronologie complète dans
DEPLOY.md section "Pièges connus") :

1. `causal-conv1d>=1.2.0` plante en compilation (CUDA 12.4 vs 13.0
   tirée par pip dans l'env de build). **Résolu** : pin à
   `==1.5.0.post8` avec wheel précompilé.

2. `mamba-ssm` (latest) plante à l'import (`GreedySearchDecoderOnlyOutput`
   absent de transformers récent). **Contournement sale** : patcher
   `mamba_ssm/utils/generation.py` pour stuber l'import.

3. Au runtime, `Fast Mamba kernels are not available` malgré l'import
   OK. **Pas résolu** dans le temps imparti.

**Action future** :

- Tester sur un pod **CUDA 12.4 propre** (image AI21 officielle si elle
  existe) ou **environnement venv isolé** avec versions verrouillées
  (transformers ~4.45, mamba-ssm 2.0.4, causal-conv1d 1.5.0).
- Alternative : compiler `mamba-ssm` depuis les sources avec
  `pip install git+https://github.com/state-spaces/mamba`. ~20 min de
  build, succès non garanti.
- Si toujours échec : retirer l'entrée Jamba du CATALOG et la mettre
  dans une branche expérimentale séparée.

**Pas urgent** : Jamba est un bonus architectural, pas un manque
fonctionnel. LFM2 valide déjà l'archi-agnosticité du pipeline cognition
sur une famille non-Transformer.

---

## 🩹 Patch propre de mamba-ssm (si Jamba devient prioritaire)

Le contournement actuel (modifier `mamba_ssm/utils/generation.py` à
la main) n'est pas reproductible et casse à chaque réinstallation du
package. Si Jamba devient un objectif prioritaire, options propres :

1. **Fork et publier un mamba-ssm-lythea** sur PyPI avec le patch
   intégré. Coût : maintenance perpétuelle.

2. **Wrapper local** dans `lythea/model.py` qui détecte l'import cassé
   au boot et patche en mémoire (sys.modules). Hack mais auto-réparable.

3. **Attendre** une version officielle compatible. mamba-ssm est en
   développement actif, le souci sera probablement résolu côté
   upstream. Vérifier périodiquement.

Recommandation : option 3 (attendre) sauf si Jamba devient critique.

---

## 📊 Méta — Dette de tests

Le fix #11 (profils sampling) est testé en isolation (catalogue
invariants + routes mockées). Les **vrais tests d'intégration**
(profil → applique au stream_generate → bonne température observée)
sont validés **manuellement** sur le pod, pas automatisés.

Pour automatiser, il faudrait :
- Un fixture pytest qui mock `model.generate()` et inspecte les kwargs
- Un test qui charge le modèle (smallest possible), envoie un message,
  vérifie que la température utilisée match le profil

Effort : ~1 jour. À faire si on veut un score de confiance plus haut.

---

## 🎨 UI — Settings éparpillés sur plusieurs onglets

Aujourd'hui les controls sont dans 6 onglets : Modèle, Vision, Mémoire,
Web, Génération, Système. Certains sont liés conceptuellement mais
séparés (ex: cross_encoder_min_score est dans Système, top_p est dans
Génération — alors que les deux affectent le RAG indirectement).

Refonte possible : regrouper par flux d'usage plutôt que par catégorie
technique. Par exemple :
- "Conversation" (température, top_p, max_tokens, reasoning toggle)
- "Mémoire" (entropy, cross_encoder_min_score, retrieval_top_n)
- "Système" (auth, rate limit, debug)

Pas urgent, à faire si l'UI devient confuse à l'usage.

---

## 📝 Notes de session

Items observés mais sans impact direct :

- LFM2-2.6B confabule abondamment (compense par sa vitesse)
- Mistral-7B est sycophant à T=0.7 (corrigé avec profil T=0.5 dans v5)
- Qwen3-Thinking : `<think>` de qualité variable selon la complexité
- BLIP captioner toujours problématique (invente une abeille sur fraise) —
  Qwen2VL est bien meilleur, l'utiliser par défaut quand dispo

---

## 🌀 SmolLM3-3B — mode dual partiellement exploitable

Statut empirique observé pendant le sprint v9 :

**Ce qui marche** :
- Mode standard (no-think) : SmolLM3 fonctionne parfaitement en mode
  conversationnel. Validé 5/5 en prod (factuel, concis, respecte les
  règles 9+10 du SYSTEM_PROMPT).
- Toggle Réflexion de l'UI : active correctement le mode raisonnement
  via le REASONING_PROMPT injecté par Lythéa. Confirmé en prod.

**Ce qui ne marche pas** :
- Balises `/think` et `/no_think` placées dans le message utilisateur :
  ne sont pas relayées correctement par `tokenizer.apply_chat_template`
  vers le modèle (probablement strippées ou mal positionnées dans le
  format de chat attendu par SmolLM3).
- Comportement émergent observé : SmolLM3 émet du `<think>` au tout
  premier message d'une nouvelle session puis bascule implicitement
  en mode rapide pour les tours suivants (mean entropy chute brutalement
  de 0.075 → 0.004 entre T1 et T2). Documenté mais non maîtrisé.

**À investiguer en v10** :
1. Inspecter le chat template SmolLM3 pour comprendre où placer la
   balise `/think` (system prompt ? user message ? tokens spéciaux ?).
2. Si on veut exploiter le dual via balises in-message, ajouter une
   couche dans `hippocampe.py` qui détecte le modèle SmolLM3 et insère
   la balise au bon endroit selon l'état du toggle Réflexion.
3. Alternative simple : laisser le toggle UI gérer (qui marche déjà)
   et ne pas chercher à exploiter les balises in-message.

**Verdict pour v9** : SmolLM3 catalogué comme modèle compact validé
sans la mention "Dual-mode" qui s'est avérée trompeuse à l'usage.
Le label final est juste "SmolLM3-3B".

---

## 🔀 Qwen1.5-MoE — non testé en prod

Ajouté au CATALOG v9 mais non validé empiriquement. Piège connu :
nécessite `transformers ≥ 4.39` sinon `KeyError: 'qwen2_moe'`. Sur
le pod RunPod actuel transformers est récent, mais à valider au
premier chargement.

À tester en priorité au prochain sprint pour valider la 5e famille
d'archi (MoE classique) sur le pipeline cognition.


---

## 🌀 V3.9 Cascade — items différés à V3.10

Issus du sprint étape 9 livré le 2026-05-03. La cascade Gemini→local
est fonctionnelle mais quelques améliorations sont notées pour plus
tard.

### UI hybride pour la clé API

Aujourd'hui la clé Google se met uniquement dans `.env`. Acceptable
pour un workflow solo SSH+pod, mais limite la portabilité (changement
de pod = re-SSH + édition fichier).

Plan V3.10 :
- Module `lythea/secrets.py` (~80 lignes) avec chiffrement Fernet
- Clé dérivée par PBKDF2 depuis `LYTHEA_AUTH_TOKEN` (déjà sécurisé)
- Persistance dans `data/secrets.enc`
- 4 endpoints : POST/GET-status/DELETE/POST-test sur `/api/secrets/google_api_key`
- Section UI dans Settings (champ password masqué + bouton "Tester")
- Priorité : `.env` gagne sur UI si les deux sont présents (philosophie 12-factor)

### Métriques cascade dans le panneau debug

Aujourd'hui le bandeau cascade signale "synthétisé" / "fallback X" mais
sans agréger sur la session. Pour valider empiriquement la qualité :

- Compteurs : nb_cascade_ok, nb_fallback_unauthorized, nb_fallback_quota,
  nb_fallback_network, nb_synthesis_skipped, nb_synthesis_failed
- Latences : moy/p50/p95 pour Gemini call, synthesis call, total cascade
- Tokens consommés : input/output Gemini cumulés sur la session
- Endpoint `/api/config/cascade/metrics` pour exposer

Utile pour comparer empiriquement V3 pur vs V3.9 cascade sur la même
séquence de 4 messages standards.

### test_connection() exposé via endpoint

`GeminiClient.test_connection()` existe déjà mais n'est pas câblée à un
endpoint. Pour un bouton "Tester la connexion" dans l'UI :

- POST `/api/config/cascade/test` qui appelle `test_connection()`
- Retourne `{"ok": true|false, "message": "...", "latency_ms": int}`
- Consomme 1 quota unit (à mentionner dans la doc UI)

### Support multi-fournisseur

Aujourd'hui `gemini_client.py` est spécifique à Google. Pour la même
qualité avec d'autres providers, abstraction nécessaire :

```
lythea/external/
├── llm_client.py        # interface abstraite (generate, test_connection)
├── gemini_client.py     # impl Google (déjà fait V3.9)
├── anthropic_client.py  # impl Claude API
├── mistral_client.py    # impl Mistral La Plateforme (FR-friendly)
└── openai_client.py     # impl OpenAI / compatible
```

Auto-détection du provider depuis le format de la clé (`AIzaSy...` =
Google, `sk-ant-...` = Anthropic, etc.). Permet à l'utilisateur de
basculer sans modifier le code.

### Streaming Gemini (optionnel)

Gemini supporte un endpoint streaming `streamGenerateContent`. Aujourd'hui
on utilise le bloc-call `generateContent` car la cascade fait deux passes
(Gemini draft + synthèse locale) et streamer la première casserait la
synthèse. Mais quand la synthèse est skippée (draft court), on pourrait
streamer Gemini directement à l'UI.

Optimisation latence : ~1s gagnée sur les réponses courtes. Pas critique
pour conversation perso mais utile si Lythéa est utilisé par plusieurs
utilisateurs simultanés.

### Affect-aware cascade routing

Idée plus exotique : ne déclencher la cascade Gemini que sur les
messages où le pipeline cognition détecte une surprise élevée ou un
besoin de raisonnement complexe. Pour les chitchats simples (genre
"D'accord", "merci"), le modèle local seul suffit.

Économie de quota : ~30-50% des tours en moins selon le profil
conversationnel. Plan V3.10 ou plus tard.

---

## 🪞 V4.4 Métacognition — Calibration des seuils sur petits modèles

**Statut** : identifié pendant les tests manuels V4.0.1 in vivo (10 mai 2026).

**Constat** :

Les seuils par défaut de la métacognition (`low: 0.15, high: 0.35,
very_high: 0.55`) sont calibrés pour des modèles qui *savent qu'ils
ne savent pas* (typiquement 7B+).

Sur Qwen2.5-3B, le `doubt_index` reste systématiquement < 0.15 même
quand le modèle hallucine avec aplomb (testé : "Décris-moi la cuisine
des Sentinelles d'Andamane" → `confidence_label: très_certaine`,
`doubt_index ≈ 0.10`).

Conséquence : les 4 bandes de certitude ne discriminent pas les
questions faciles vs impossibles → l'utilité de la métacognition
est limitée au tracking de calibration cumulative (Brier score) sans
impact sur le comportement.

**Pistes** :

1. **Seuils par modèle** — détecter le modèle actif au boot et charger
   un préset adapté. Pour Qwen 3B :
   ```
   metacog_low_doubt=0.05
   metacog_high_doubt=0.10
   metacog_very_high_doubt=0.18
   ```
   Pour Qwen 7B+ : garder les défauts. Stockage : dict dans
   `lythea/cognition/metacognition_presets.py`.

2. **Enrichir la décision avec d'autres signaux** — le `doubt_index`
   du LLM est un proxy faible sur petits modèles. Combiner avec :
   - **KG hits** : si l'entité de la question est absente du KG
     ET que le RAG n'a rien pertinent → +0.3 au doute
   - **Web search déjà tenté sans résultat** → +0.2
   - **Distance sémantique de la question** vs SDM : si embedding
     orthogonal à tout ce qu'on a en mémoire → +0.2
   Le `doubt_index` final devient une combinaison pondérée plutôt
   qu'une lecture brute du LLM.

3. **Auto-calibration** — après ~200 échanges, si `calibration_score`
   > 0.9 et que `confidence_label` est toujours "très_certaine",
   abaisser automatiquement `low_doubt` de 0.02. Module qui
   apprend ses propres seuils.

**Priorité** : moyenne. La métacognition fonctionne (tracking OK,
hooks OK) mais sa valeur opérationnelle dépend de cette calibration.

---

## 🎯 Planning — Goal advance via commande utilisateur

**Statut** : identifié pendant les tests manuels V4.0.1 in vivo.

**Constat** :

`GoalStack.advance_step()` existe et est testé unitairement, mais
aucun chemin du code ne l'appelle depuis le chat. Quand l'utilisateur
dit "j'ai fini la première étape", `current_step` reste à 0.

**Pistes** :

1. **Commande explicite `/done`** — Lythéa intercepte `/done` (au
   début ou à la fin du message), appelle `advance_step()`, et inclut
   dans sa réponse "Étape 1/3 marquée comme faite, étape 2 active :
   …".

2. **Classifier d'avancement** — détecter dans `IntentClassifier` un
   nouveau intent `step_completion` sur des phrases comme "j'ai fait X",
   "next", "ok pour la 1", "passons à la suite". Plus naturel mais
   risque de faux positifs.

3. **Bouton UI** — à côté du diagnostic "But actif" dans l'onglet
   Cognition, un bouton "✓ Marquer fait" qui POST sur un endpoint
   `/api/v4/planning/advance_step`. Le moins ambigu.

**Priorité** : haute si le module Planning est utilisé en pratique
(sinon les goals s'accumulent sans avancer et le module devient un
journal sans dynamique).

---

## 🔮 V4.2 Predictive coding — Seuils inadaptés à Qwen 3B

**Statut** : identifié pendant les tests manuels V4.0.1 in vivo (10 mai 2026).

**Constat** :

Les seuils par défaut (`low: 0.15, high: 0.65`) sont inadaptés aux
embeddings produits par Qwen2.5-3B. Sur des messages testés in vivo
(de "qu'est-ce que la photosynthèse" à "j'ai cassé mon vélo dans la
descente du Galibier"), l'`error` cosine reste systématiquement entre
0.06 et 0.10 — bien sous le seuil low. Le module est toujours en mode
`low_power`, ne discrimine jamais les sauts sémantiques nets.

**Trois causes probables (à investiguer)** :

1. **Embedding `mean_latent` peu discriminant sur messages courts** —
   la moyenne des hidden states est dominée par les tokens fonctionnels
   (articles, ponctuation) sur les messages < 50 tokens. Tous les
   messages courts FR se ressemblent dans cet espace.

2. **EMA decay trop conservateur** — `pc_ema_decay=0.7` lisse trop la
   prédiction, qui finit toujours proche de l'observation. Tester
   avec 0.4-0.5.

3. **Embeddings sur sphère unitaire** — si Lythéa normalise
   `mean_latent`, les distances cosine sont compressées. Solution :
   utiliser la distance L2 brute plutôt que cosine, OU recalibrer
   les seuils empiriquement.

**Pistes de fix** :

- Collecter empiriquement la distribution de `pc_error` sur ~200
  échanges réels avec Qwen 3B → ajuster les seuils aux quantiles
  observés (P30 pour `low`, P85 pour `high`).
- Au lieu de `mean_latent`, utiliser le hidden state du **dernier**
  token (le plus chargé sémantiquement, après attention) comme
  embedding.
- Ajouter un préset par modèle (cf. backlog métacognition) :
  - Qwen 3B : `low: 0.05, high: 0.20`
  - Qwen 7B+ : `low: 0.15, high: 0.65` (défauts actuels)

**Convergence avec V4.4 Métacognition** : c'est le **3ème point**
qui converge sur la même conclusion — les seuils V4 par défaut ont
été calibrés théoriquement, pas mesurés sur le LLM réel. Une
session de tuning empirique sur Qwen 3B (~2-3 h) résoudrait
métacog + predictive_coding d'un coup.

**Priorité** : moyenne. Le module ne crashe pas, il est juste
silencieux. Bloque uniquement la sub-feature `pc_apply_gating` qui
ne supprimerait jamais de web search dans cet état.

---

## ⚠️ V4 — Convention warnings actionnables (helper générique)

**Statut** : identifié pendant les tests manuels V4.0.2 in vivo
(10 mai 2026, Qwen3-4B Thinking).

**Constat** :

Le module Timeline V4.0.2 produit correctement un warning d'incohérence
temporelle ("hier on a soutenu la réunion du 12 mai 2026"). Le warning
est lu par le LLM dans son raisonnement `<think>` :

> "the system's memory has a note about a temporal inconsistency where
> 'hier' (May 9) doesn't match May 12"

**Mais la réponse finale visible à l'utilisateur reste vague** :
> "Tu as probablement confondu les dates, car nous sommes actuellement
> le 10 mai 2026."

Le LLM mentionne qu'il y a confusion mais ne caractérise pas
**précisément** les deux dates en conflit, et il détourne sur les
résultats web search au lieu de demander une clarification ciblée.

**Cause** :

Le warning donne **les faits** mais aucune **directive d'action** au
LLM. Sans consigne explicite, le modèle rend le warning de façon
édulcorée pour rester poli.

**Solution recommandée — Option A générique** :

Créer un helper réutilisable par tous les modules V4 :

```python
# lythea/cognition/warnings_v4.py (nouveau module)

def format_warning(
    issue: str,           # "Incohérence temporelle"
    details: str,         # "« hier » résolu au 09/05/2026..."
    directive: str,       # "Demande à l'utilisateur quelle date..."
    icon: str = "⚠️",
) -> str:
    """Convention V4 : tout warning d'un module V4 inclut une directive
    d'action commençant par '→' qui indique au LLM comment réagir.
    """
    return f"{icon} {issue} : {details}\n   → {directive}"
```

Modifier `timeline.detect_inconsistencies` pour utiliser ce helper
avec des directives par type d'incohérence :

- Cas "hier + future date" → directive : "Demande à l'utilisateur
  quelle date correspond à l'événement : X ou Y. Ne paraphrase pas
  le message comme s'il était cohérent."
- Cas "discordance" → directive : "Pointe la discordance précise
  avant de répondre sur le fond."
- Cas "deux absolutes" → directive : "Demande laquelle des deux
  dates est correcte."

**Génericité** :

Tout futur module V4 qui produira un warning (ex: métacognition →
"haute confiance sans source KG/web → demande clarification ou cherche
web") devra passer par `format_warning()` pour rester cohérent. Une
note de doc dans le helper le rappelle.

**Pourquoi pas Option B** (consigne système globale) :

Un système qui détecte les ⚠️ dans tous les blocs introduit un
coupling implicite (tous les modules doivent assumer le même format).
La convention par helper est plus propre — chaque module reste maître
de ses warnings ET de leurs directives.

**Estimation** : ~40 lignes (helper + 3 directives Timeline + tests).

**Priorité** : moyenne. Le module fait techniquement son travail
(warning produit, lu par le LLM). C'est l'expressivité de la réponse
finale qui est sub-optimale, surtout sur les modèles thinking.
