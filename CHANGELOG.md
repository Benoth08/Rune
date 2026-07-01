# Changelog

## V4.1.4 — Couplage raisonnement + recherche web (2026-05-15)

### 🌐🧠 Ancrage factuel du raisonnement profond

Observation : un Qwen 2.5-3B avec raisonnement activé sur une
question analytique pointue inventait des noms d'algorithmes
post-quantiques (« Systèmes de Galois ») faute de connaissances
internes suffisantes. La recherche web ne se déclenchait pas car
le détecteur ne cible que les questions factuelles, pas les
questions analytiques.

**5ème déclencheur web** — `process_message` calcule désormais, via
le routeur `assess_complexity`, si le raisonnement profond va
tourner sur une question genuinement complexe (≥ 2 étapes). Si
oui ET que le web ne s'est pas déjà déclenché → la recherche web
est lancée quand même. La logique : une question assez complexe
pour mériter un raisonnement profond mérite un ancrage factuel.

**Le contexte web descend dans la chaîne** — les résultats web
(numérotés [1][2]…) sont transmis à `DeepReasoningChain.run()` puis
injectés dans les prompts des étapes de **décomposition** et
**d'exploration** (pas dans critique/synthèse, qui travaillent sur
la matière déjà collectée). Le modèle puise dans des faits réels
au lieu de ses connaissances internes — particulièrement décisif
sur les petits modèles.

**Garde-fous** :
- Contexte web borné à 1800 caractères avant injection (ne noie
  pas l'instruction de raisonnement).
- Pas de dépendance à la taille du modèle : le couplage aide aussi
  le 7B (qui avait ~1 erreur factuelle sur la même question).
- Le contexte web sert deux fois — dans le raisonnement (pour
  structurer) et dans la génération finale (pour rédiger avec les
  citations [N]). Voulu.

Tests : `test_deep_reasoning.py` passe à 22 cas (+4 : injection
web dans exploration, injection décompose+explore seulement en
4-étapes, absence de bloc si pas de web, troncature du contexte).
669 tests verts au total.


## V4.1.3 — Chaîne de raisonnement 4 étapes (2026-05-15)

### 🧩 Raisonnement profond — décomposer → explorer → critiquer → synthétiser

Le routeur ``assess_complexity`` distingue maintenant **3 niveaux** :

- ``0`` → question simple, pas de chaîne (fallback reasoning simple)
- ``2`` → complexité moyenne, chaîne courte :
  explorer → feuille de route
- ``4`` → complexité élevée, chaîne complète :
  **décomposer** → explorer → **critiquer** → **synthétiser**

La chaîne 4-étapes se déclenche quand la question cumule un
marqueur analytique (« compare », « analyse ») ET au moins un
autre signal fort (question longue, surprise élevée, plusieurs
entités).

**Étape 1 — Décomposer** : casser la question en 2-4 sous-problèmes
numérotés. Tâche très courte, budget medium. Si elle échoue, repli
gracieux sur la chaîne 2-étapes.

**Étape 2 — Explorer** : cartographier ce qu'on sait/ignore, guidé
par les sous-problèmes (pas en exploration libre). Budget high
(960 tokens) — c'est le cœur du raisonnement.

**Étape 3 — Critiquer (pure)** : repérer les faiblesses de
l'exploration, SANS rédiger de feuille de route. Séparer critique
et synthèse donne une critique plus honnête — le modèle n'essaie
pas de « sauver » ses erreurs pour rédiger en même temps.

**Étape 4 — Synthétiser** : construire la feuille de route finale
à partir de l'exploration + la critique. Même format condensé que
la chaîne 2-étapes (structure, points-clés, corrections, conclusion).

**Budgets** : décomposer/critiquer/synthétiser = medium (640 tok),
explorer = high (960 tok). Total chaîne 4-étapes ≈ 2880 tokens de
raisonnement — comparable au budget thinking natif d'un modèle
comme Qwen3-4B-Thinking.

**Tests** : 18 tests unitaires (+3 nouveaux : labels 4-step,
budgets 4-step, dégradation vers 2-step si décomposition échoue).
665 tests verts au total.


## V4.1.2 — Exploration cadrée (2026-05-14)

### 🎯 Étape d'exploration disciplinée

Symptôme observé en test in vivo (Qwen 7B, question cryptographie) :
l'étape d'exploration partait en liste interminable — **11 « angles
à considérer »** — ce qui cramait le budget de tokens sur
l'énumération et faisait tronquer le raisonnement consolidé à
l'affichage.

Le prompt d'exploration impose désormais des **limites de comptage
par section** :
- ce qu'on sait : 2 à 4 points clés
- ce qui est incertain : 1 à 3 points
- angles à considérer : **4 à 5 maximum**

« Choisir l'essentiel fait partie du travail » — quatre angles bien
posés valent mieux que onze superficiels. Le budget n'est plus
gaspillé sur une liste de courses ; le raisonnement consolidé a la
place de se terminer proprement.

`reasoning_text_max_chars` monté de 4000 à **6000** — filet de
sécurité d'affichage relevé pour ne plus tronguer un consolidé
bien formé. Mais la vraie défense reste le cadrage du prompt : le
clip ne rattrape plus que les cas aberrants.

### Note de validation

Test in vivo confirmant que le mode profond apporte un gain réel :
sur la même question, la réponse en mode profond couvre 3 avantages
et 5 inconvénients (vs 2 et 3 en mode simple), avec des méthodes
post-quantiques nommées précisément et des dimensions
supplémentaires (coordination des parties prenantes, éthique,
réglementation). Le scaffolding structure mieux le raisonnement —
sans pour autant corriger toutes les imprécisions factuelles d'un
modèle 7B.


## V4.1.1 — Budgets de raisonnement adaptatifs (2026-05-14)

### 🎚️ Budgets de tokens revus — fin des troncatures

Symptôme : le raisonnement (simple comme profond) était coupé en
plein milieu d'une phrase — `"...ce qui est particul"` en mode
simple, `"Vitesse et E"` en mode profond. Cause : des plafonds de
tokens en dur, trop bas (800 simple, 320 par étape deep), invisibles
et non réglables.

**Les budgets vivent maintenant dans `settings.py`** — réglables
sans toucher au code :
- `reasoning_simple_max_tokens` (défaut 1024, ex-800)
- `reasoning_deep_step_medium_tokens` (défaut 640)
- `reasoning_deep_step_high_tokens` (défaut 960, ex-320)
- `reasoning_text_max_chars` (défaut 4000, ex-2000/3000)

**Budget adaptatif pour la chaîne profonde** — `DeepReasoningChain`
ne prend plus un budget fixe. Une nouvelle méthode `_complexity_level`
distingue les questions `medium` et `high` (une question est `high`
si elle cumule un marqueur analytique explicite ET au moins un autre
signal fort : surprise élevée, plusieurs entités, ou question
longue). Le budget par étape suit : `medium` → 640 tokens, `high` →
960 tokens.

**Consigne de longueur dans le prompt** — chaque passe de
raisonnement (simple et profonde) reçoit désormais une cible de
longueur en mots, insérée dans le system prompt (« Vise environ N
mots »). Cette cible est calibrée à ~80 % du plafond physique de
tokens : le modèle a de la marge pour finir sa phrase avant de
heurter le mur `max_new_tokens`. C'est la double-contrainte —
limite physique *et* information donnée au modèle — alignées par
construction sur le même verdict de complexité.

Tests : `test_deep_reasoning.py` passe à 16 cas (budgets lus depuis
settings, budget `high` sur question à signaux forts, troncature
bornée par `reasoning_text_max_chars`).


## V4.1.0 — Raisonnement profond multi-étapes (2026-05-14)

### 🧩 `DeepReasoningChain` — scaffolding du raisonnement

Nouveau module `lythea/cognition/deep_reasoning.py`. Un modèle
non-thinking (Qwen-Instruct…) à qui on demande « réfléchis » fait
un effort unique et souvent superficiel. Cette chaîne reproduit le
comportement des gros modèles de raisonnement par *scaffolding* :
au lieu d'un appel « réfléchis bien », on enchaîne des appels courts
et ciblés, chacun étant une tâche simple que le petit modèle réussit.

**Version actuelle — 2 étapes** :
1. *Explorer* — cartographier ce qu'on sait, ce qu'on ignore, les
   angles à considérer (pas encore la réponse).
2. *Critiquer* — attaquer l'exploration, repérer ce qui cloche,
   produire un raisonnement consolidé.

Conçu pour grandir : les étapes *décomposer* et *synthétiser*
s'inséreront autour quand la version 2-étapes aura été validée
in vivo.

**Routeur de complexité** — `assess_complexity` lit les signaux
déjà calculés par le pipeline (surprise, doute, nombre d'entités
KG, longueur de question, marqueurs analytiques type « pourquoi »,
« compare », « analyse ») et renvoie le nombre d'étapes : `0` (rien,
on retombe sur le reasoning simple) ou `2` (chaîne complète). Aucun
appel LLM dans le routeur — heuristique pure et rapide.

**Périmètre** — modèles non-thinking uniquement. Les modèles
thinking raisonnent déjà nativement dans leur `<think>` ; leur
superposer la chaîne ferait doublon. L'orchestrateur ne l'appelle
que si `model.is_thinking` est faux.

**Garde-fous** :
- Désactivé par défaut (`enable_deep_reasoning=False`).
- Budgets tokens calibrés par étape pour qu'une étape finisse sa
  pensée : exploration 600, critique 550 (la v1 utilisait 320, ce
  qui coupait l'exploration à mi-phrase et polluait la critique).
  Consigne de concision dans les prompts pour éviter le remplissage.
- Température basse (0.3).
- Budget temps global (45s) qui coupe la chaîne si elle traîne.
- Toute étape qui échoue → dégradation gracieuse : si l'exploration
  échoue, retour vide (→ fallback reasoning simple) ; si la critique
  échoue, on renvoie l'exploration plutôt que rien.
- Aucune dépendance nouvelle.

**Intégration** :
- `hippocampe._generate_reasoning` tente d'abord la chaîne profonde
  si le flag est actif ; sinon, ou si le routeur juge la question
  simple, retombe sur `ReasoningGenerator`.
- Endpoint `POST /api/config/reasoning` accepte un champ optionnel
  `deep` ; `GET` expose `deep` et `is_thinking_model`.
- Toggle UI « 🧩 Profond » dans la barre du haut, visible seulement
  sur modèle non-thinking avec « 💡 Réflexion » déjà actif. Couper
  Réflexion désactive et masque Profond.

15 tests unitaires (`test_deep_reasoning.py`) : routeur sur cas
variés, orchestration (skip / chaîne complète / dégradations),
robustesse aux échecs d'étape, intégration KG.


## V4.0.4 — Ingestion documentaire (2026-05-13)

### 📚 Script d'ingestion de documents — `ingest.py`

Nouveau script autonome à la racine qui alimente la mémoire
long-terme de Lythéa (ChromaDB) avec le contenu de fichiers
externes. Permet de spécialiser Lythéa sur un corpus métier ou
scientifique.

**Formats supportés** : PDF (via `pdfplumber`), `.docx` (via
`python-docx`), `.txt`, `.md`, `.rst`.

**Fonctionnement** :
- Extraction du texte (page par page pour les PDF).
- Chunking ~800 caractères avec chevauchement de 150, coupures sur
  frontières naturelles (paragraphe → phrase → espace).
- Ajout dans la collection `lythea_memory` avec metadata
  `type="knowledge"` (distincte de `type="exchange"` des souvenirs
  de conversation) + `source`, `page`, `tag` optionnel.
- ChromaDB calcule les embeddings (all-MiniLM-L6-v2) — aucun modèle
  Lythéa requis pour l'ingestion.
- Idempotent par fichier : ids de chunks dérivés d'un hash
  `nom + contenu`, donc réingérer un fichier remplace ses chunks
  au lieu de créer des doublons.
- L'index BM25 sparse de Lythéa se reconstruit automatiquement à la
  première requête suivante.

**Commandes** :
```
python3 ingest.py documents/                  # ingérer un dossier
python3 ingest.py rapport.pdf                 # ingérer un fichier
python3 ingest.py --tag r_and_d articles/     # tagger thématiquement
python3 ingest.py --reset corpus/             # purger puis réingérer
python3 ingest.py --list                      # lister l'ingéré
python3 ingest.py --purge                     # tout supprimer
```

`--purge` et `--reset` ne touchent que les documents `type=knowledge`
— les souvenirs de conversation restent intacts.

`deploy.sh` installe désormais `pdfplumber` + `python-docx` (non
bloquant : `.txt`/`.md` fonctionnent sans).

### 🎬 Phase pills animées

`lythea/server/static/` : pendant qu'elle travaille, Lythéa affiche
une *pill* animée façon Gemini :
- 🌐 « Recherche web… » — gradient bleu, pendant `iterative_search`
- 💭 « Réflexion… » — gradient violet, pendant le `<think>`

Gradient text animé (shimmer 2.2s) + icône qui pulse (1.4s). La pill
apparaît sur `phase_status: start`, disparaît en fade-out 250ms sur
`done`. `prefers-reduced-motion` respecté. Pas de pill pour la
génération : le texte qui se stream est déjà l'indicateur.

### 📎 Citations web numérotées + affichage des sources

- `web.py` `iterative_search` : les résultats web sont désormais
  formatés `[1] Titre / URL / extrait` au lieu d'une concaténation
  brute de snippets. Déduplication par URL.
- `_inject_web_context` ajoute une directive de citation (« quand tu
  utilises une info, ajoute la référence [N] »).
- Event SSE `web_sources` → le frontend affiche un bloc `📎 Sources`
  repliable sous la réponse, avec liens cliquables vers les URLs.

### 🧹 Divers UI

- Bouton 🗑️ « Supprimer toutes les conversations » dans la sidebar
  (discret, devient rouge au survol, double-confirmation). Nouvel
  endpoint `DELETE /api/sessions` + `SessionsManager.delete_all()`.
- Renommage V3 → V4 partout (titre, sidebar, FastAPI, welcome).
- Sous-titre d'accueil : « Conscience IA cognitive — Lythéa V3 » →
  « Assistant cognitif local — mémoire, raisonnement et recherche
  web ».


## V4.0.3 — Web search providers + cohérence in vivo (2026-05-10)

### 🌐 Système de recherche web pluggable

Avant : Lythéa appelait directement `ddgs` (lib non-officielle qui
scrape DuckDuckGo). Résultats médiocres, rate-limit agressif, pas de
contrôle qualité.

Après : architecture pluggable `lythea/web_providers/` avec deux
backends (sans clé API requise, open-source) :

**SearXNG** (`searxng.py`) — provider primary

Meta-moteur open-source qui agrège Google + Bing + Brave + Wikipedia
+ GitHub + 70 autres en parallèle. Retourne du JSON propre.

- Pas de clé API requise (passe par instances publiques par défaut)
- Liste curée d'instances publiques fiables (rotated, last-known-good
  caché) — voir `DEFAULT_PUBLIC_INSTANCES` dans `searxng.py`
- Override possible via `SEARXNG_INSTANCE_URL` (auto-hosting recommandé
  en prod) : https://docs.searxng.org/admin/installation-docker.html

**DDG** (`ddg.py`) — provider fallback

Wrapper sur `ddgs` (la lib historique) gardé en fallback pour
deployments offline / si toutes les instances SearXNG tombent.

**Composite** (`factory.py`) — chain SearXNG → DDG

Try-each-in-order. Skip les providers `is_available()=False`. Swallow
les exceptions (log warning + try next). Si tout échoue, retourne `[]`.

**Configuration .env** :
```bash
WEB_SEARCH_PROVIDER=auto         # auto (default) | searxng | ddg
SEARXNG_INSTANCE_URL=https://my-searxng.example.org  # optional override
```

**Format unifié** : tous les providers retournent
`list[{title, body, href}]` — identique au format DDG historique,
donc aucune modification du reste de Lythéa (`hippocampe.py`, RAG, UI).

**Migration** : `lythea/web.py` (`WebAgent`) délègue maintenant à
`get_default_provider()`. Anciens appels `WebAgent().search()` /
`iterative_search()` inchangés — l'API publique est préservée.

### 🧪 Tests V4.0.3

- `tests/test_web_providers.py` : **25 tests neufs** couvrant
  registry, SearXNG (parsing, rotation, fallback, cache known-good),
  DDG (gestion missing lib), Composite (priorités, fallback, exceptions
  swallowed), intégration WebAgent injectable.
- Régression sandbox : **619 tests verts** (594 V4.0.2 + 25 V4.0.3),
  zéro régression.

### 🛡️ Garanties V4.0.3

- Aucune nouvelle dépendance pip (SearXNG via stdlib `urllib.request`)
- DDG reste optionnel (`pip install ddgs`) — Lythéa marche sans
- Aucune signature publique modifiée (`WebAgent.search()` /
  `iterative_search()` API stable, kwarg `provider=` additif pour tests)
- Failover transparent : utilisateur final ne voit jamais l'erreur
- Logs informatifs (`web search answered by searxng`, `All instances
  failed`, etc.)

---

## V4.0.2 — Corrections post-validation manuelle in vivo (2026-05-10)

### 🎯 Contexte

Validation manuelle complète V4.0.1 sur RunPod avec Qwen2.5-3B
(6 tests live). 4 ✅ critiques + 4 🟠 backlog identifiés. Cette
version résout les 4 items backlog.

### 🆕 Corrections

**1. Cohérence temporelle dans Timeline (V4.3 backlog)**

`lythea/cognition/timeline.py` :
- Nouvelle fonction `detect_inconsistencies(events, now, text)` qui
  croise les marqueurs temporels par phrase (split sur `.!?;`) et
  flagge les contradictions (ex: "hier" + date future).
- Nouvelle fonction `_humanize_to_offset_sec(label)` qui inverse
  `_humanize_relative_seconds` pour résoudre "hier", "il y a 3 jours",
  "dans 2 mois", etc. en offsets seconde.
- `render_block` étendu avec param `text=None` qui permet la
  délimitation par clauses (plus précise) au lieu de la fenêtre
  de 60 chars (faux positifs).
- Hook A.4 Hippocampe passe désormais le texte source à `render_block`.

Cas validé : "Hier on a soutenu la réunion du 12 mai 2026" alors
qu'on est le 10 mai 2026 → bloc `[Chronologie]` enrichi avec
`⚠️ Incohérence temporelle...` et directive d'action.

**1bis. Helper warnings V4 générique (post in vivo Qwen3-4B)**

Nouveau module `lythea/cognition/warnings_v4.py` :
- `format_warning(issue, details, directive, icon=⚠️)` produit un
  warning à 2 lignes : header `⚠️ <issue> : <details>` puis
  directive `   → <action>`.
- `is_v4_warning_line(line)` pour détection downstream.

Convention V4 imposée : tout warning V4 doit comporter une
**directive d'action** concrète (en plus de signaler le problème).
Validation in vivo Qwen3-4B (10 mai 2026) : sans directive le LLM
lit le warning dans son `<think>` mais le caractérise vaguement
dans la réponse. Avec directive ("Demande à l'utilisateur quelle
des deux dates correspond..."), la réponse devient ciblée.

Timeline migré : les 3 types de warnings (incohérence past+future,
discordance, deux absolus conflictuels) utilisent maintenant
`format_warning()`. Tous les warnings inline incluent désormais la
date système et les deux dates candidates pour que le LLM puisse
poser une question précise.

Modules V4 futurs : utiliser `format_warning()` pour un format
homogène. La directive doit être à l'impératif et assez spécifique
pour que le LLM produise une réponse bien formée sans interprétation.

**2. Goal advance via /done (Planning backlog)**

`lythea/cognition/planning.py` :
- Nouvel intent `step_completion` (5e dans `INTENTS`).
- Détection commande explicite : `/done`, `/next`, `/fait`, `/suivant`.
- Détection verbale FR/EN : "j'ai fini", "terminé", "step done",
  "first step done", etc. Filtré par longueur ≤ 15 mots pour éviter
  les faux positifs sur narrations longues.
- Branche dédiée dans `PlanningPhase._process_inner` qui appelle
  `goal_stack.advance_step(active.id)` et expose `advanced_step`,
  `completed_goal` dans `PlanningResult`.
- Hook A.2 affiche `➡️ Étape 2/3 marquée comme faite` ou
  `✅ But entièrement réalisé` dans les cognitive items.

**3. Auto-calibration générique (Métacognition + Predictive coding backlog)**

Nouveau module `lythea/cognition/auto_calibrator.py` :
- `QuantileCalibrator` : observateur sliding-window avec
  `quantile(p)`, persistance JSON atomique optionnelle.
- `AutoCalibratedThresholds` : wrapper bootstrap → empirique. Pré-
  bootstrap : seuils fixes. Post-bootstrap (≥ N obs) : P25/P75/P90
  empiriques. Monotonicité enforced (low < high < very_high).

Intégration `MetacognitivePhase` :
- `MetacognitionConfig` étendu : `auto_calibrate=True`,
  `auto_calibrate_min_samples=30`, `auto_calibrate_window=200`.
- `_observe_inner` feed le calibrator + refresh classifier
  thresholds **avant** classification.
- `to_dict` expose `thresholds_in_use` (low/high/very_high + source
  bootstrap/empirical) et l'état du calibrator (P25/P50/P75/P90).

Intégration `PredictiveCodingPhase` :
- `PredictiveCodingConfig` étendu : `auto_calibrate=True`,
  `auto_calibrate_min_samples=20`, `auto_calibrate_window=200`.
- `_observe_inner` feed l'erreur cosine après cold-start +
  utilise les seuils dynamiques.
- `is_bootstrapping()` méthode publique.
- `reset()` purge aussi le calibrator (nouvelle conversation =
  nouvelle distribution).
- Nouvelle méthode `to_dict()` pour télémétrie complète.

**Conséquence** : Lythéa V4 fonctionne maintenant correctement avec
**n'importe quel modèle** (Qwen 3B, 7B, Llama, GPT-class, …) sans
ajustement de seuils. Le module observe la distribution intrinsèque
au modèle pendant ~30 échanges puis se calibre tout seul.

**4. CSS modal Paramètres**

`lythea/server/static/style.css` :
- `.modal width: 580px → 680px`, `max-height: 80vh → 88vh`.
- `max-width: 92vw → 95vw` pour mobile.
- `flex-shrink: 0` sur header + tabs pour empêcher l'écrasement
  quand le contenu est long.
- Breakpoint mobile 600px → 700px.

Conséquence : tous les onglets (Modèle, Vision, Mémoire, Web,
Génération, Cognition, Système) restent visibles sans débordement.

**5. SearXNG self-hosted intégré au lancement**

Nouveau script `searxng_bootstrap.sh` (idempotent) qui :
- Détecte si SearXNG est installé, l'installe depuis git officiel sinon
  (clone + pip install des requirements pinned + install editable).
- Génère un `settings.yml` stable à partir du template officiel
  (secret_key aléatoire, bind 127.0.0.1, JSON format activé, limiter
  désactivé, botdetection relaxée).
- Lance SearXNG en arrière-plan sur le port 8080 (configurable via
  `$SEARXNG_PORT`).
- Vérifie qu'il répond et qu'une requête test retourne ≥ 1 résultat.
- Expose `SEARXNG_URL` + `SEARXNG_PID` sur stdout pour le parent.

`launch.sh` étendu :
- Appelle `searxng_bootstrap.sh` avant Lythéa.
- Si OK : exporte `LYTHEA_SEARXNG_INSTANCE_URL` + `LYTHEA_WEB_PROVIDER=searxng`
  → Lythéa utilise automatiquement le SearXNG local.
- Si échec : Lythéa démarre quand même, retombe sur DDG via la chaîne
  CompositeProvider.
- Le trap final tue aussi SearXNG.

`deploy.sh` étendu :
- `chmod +x` sur tous les scripts shell (`searxng_bootstrap.sh` inclus).
- Vérifie et installe `pyyaml` si absent (requis par le bootstrap).

Conséquence : `bash launch.sh` suffit. Plus besoin de Docker, plus
besoin de DDG (qui ne renvoyait que des snippets pauvres).
Première exécution : 2-3 min d'installation SearXNG. Suivantes :
~10 s de boot SearXNG en plus du boot Lythéa habituel.

**6. Logs propres au démarrage**

`lythea/logging_setup.py` :
- Nouvelle classe `_PollingAccessFilter` qui supprime des access logs
  uvicorn les requêtes vers les endpoints de polling :
  `/api/boot/status`, `/api/health`, `/api/config/v4`.
- Sans ce filtre, `launch.sh` (qui poll `/api/boot/status` toutes les
  secondes pendant la phase de preload) générait 30-100 lignes inutiles
  dans le log.
- Le filtre matche en priorité sur `record.args` (forme structurée
  uvicorn) puis fallback sur le message rendu (custom formatters).
- Nouvelle fonction `build_uvicorn_log_config()` qui produit le dict
  log_config à passer à `uvicorn.run(log_config=...)`. Garantit que
  le filtre survit au setup logging d'uvicorn.

`run.py` :
- `uvicorn.run` reçoit maintenant `log_config=build_uvicorn_log_config()`.

**7. Barre de progression visuelle au boot**

`launch.sh` : la boucle qui suit le boot affichait précédemment une
ligne par changement d'étape (`📦 GLiNER (20%)`). Remplacée par une
**barre de progression dynamique** qui se met à jour sur place via
`\r` :

```
  📦 [██████████░░░░░░░░░░] 33% GLiNER
```

À chaque transition d'étape, un newline est inséré → la barre
précédente est conservée à 100%, la suivante démarre sur une
nouvelle ligne. Fonction `render_bar` paramétrable (largeur, label).

**8. Citations web numérotées [N] dans les réponses**

`lythea/web.py` ``iterative_search`` :
- Auparavant : concaténait juste les ``body`` des résultats sans
  titre, URL, ni numérotation. Le LLM voyait du texte brut et ne
  pouvait pas distinguer / citer les sources.
- Maintenant : produit un bloc structuré avec ``[1] Titre`` /
  ``URL`` / ``Body snippet`` pour chaque résultat. Déduplication
  par URL (vs body précédemment). Budget total 2000 chars
  réparti dynamiquement entre les références.

`lythea/hippocampe.py` ``_inject_web_context`` :
- Le bloc ``[Recherche web — résultats récents]`` se termine par
  une directive d'action style V4 (cf. ``warnings_v4``) :
  *« → Quand tu utilises une information ci-dessus, ajoute la
  référence [N] correspondante (ex: selon [2]). N'invente pas de
  références non listées. »*

Validation in vivo Qwen3-4B Thinking (10 mai 2026) : avant le patch,
le LLM lisait le bloc web mais mélangeait les sources sans citer
(IYQ 2025, Aspect, Rovelli, LK-99 mentionnés sans rattachement à
une URL). Avec le patch, les réponses prennent la forme « selon [1]
Wikipédia, … » / « comme l'indique [3] le CEA, … » avec hyperliens
cliquables dans l'UI.

**5. Web search V4.0.3 — chaîne complète Tavily + Serper + Brave + SearXNG + DDG**

`lythea/web_providers/tavily.py` (NEW) :
- `TavilyProvider` : Tavily AI Search API (https://tavily.com).
- Renvoie un champ `answer` (synthèse 2-3 phrases) en plus des
  snippets bruts → économise des tokens dans le `<think>` du LLM.
- Free tier 1000 req/mois sans CB.
- Param `search_depth` : `basic` (~1s, 1 crédit) ou `advanced`
  (~3s, 2 crédits, crawl plus profond).

`lythea/web_providers/serper.py` (NEW) :
- `SerperProvider` : Serper Google Search API (https://serper.dev).
- Proxy direct sur Google → qualité Google.com, latence ~300ms.
- Promote `knowledgeGraph` (Wikipedia summary) et `answerBox`
  (featured snippet) en tête des résultats — RAG gold.
- Free tier 2500 req/mois sans CB.

`lythea/web_providers/factory.py` :
- Registry étendu à 5 providers.
- Chaîne `auto` : **Tavily → Serper → Brave → SearXNG → DDG**.
- Chaque provider sans clé est silencieusement skippé par le
  `CompositeProvider` (`is_available()` returns False).
- Choix explicite via `WEB_SEARCH_PROVIDER=tavily|serper|brave|searxng|ddg`.

`.env.example` :
- Section "Web search providers" entièrement réécrite avec
  instructions Tavily (priorité 1) + Serper (priorité 2). Brave
  est documenté comme payant depuis fin 2026.

**Pourquoi cette cascade** : DDG Instant Answer (utilisé en fallback
seul auparavant) ne fait pas une vraie recherche web. Brave a
supprimé son free tier fin 2026. Tavily + Serper offrent ensemble
**3500 requêtes/mois gratuites** avec qualité Google-comparable
et complémentarité forte :
- Tavily : optimisé LLM (snippet synthétisé), latence 1-2s
- Serper : Google brut + KG/AB, latence 300ms

**Validation** : 17 tests neufs `test_web_providers.py` :
- Tavily : 6 tests (key handling, payload avec/sans answer, HTTP 429,
  search_depth validation).
- Serper : 5 tests (key handling, KG promotion, answer box promotion,
  HTTP 401, empty query).
- Chaîne complète : 4 tests (ordre des 5 providers, skip unavailable,
  choix explicite Tavily/Serper, registry).
- 1 test obsolète mis à jour (chaîne Brave-first → chaîne complète).

Total V4.0.3 tests : **53 web providers** + 41 V4.0.2 = **94 tests
nouveaux** depuis V4.0.1. Régression complète sandbox : **647 verts**
(594 + 53 web). Sur machine torch+chromadb : **~880 verts attendus**.

### 🛡️ Garanties (étendues)

- `auto_calibrate=True` par défaut sur les deux modules. Désactivable
  via `MetacognitionConfig(auto_calibrate=False)` ou
  `PredictiveCodingConfig(auto_calibrate=False)` pour tests
  déterministes.
- Tous les nouveaux fixes try/except → fallback neutre.
- Aucune signature publique modifiée (kwargs additifs partout).
- Aucune nouvelle dépendance pip.
- Le module `auto_calibrator` est pur Python (stdlib uniquement).

---

## V4.0.1 — Métacognition + UI runtime toggle (2026-05-05)

### 🆕 Nouveautés

**1. Module V4.4 — `lythea/cognition/metacognition.py` (~360 lignes)**

Auto-monitoring de la certitude (mPFC + ACC dorsal) :

- **`CalibrationTracker`** : fenêtre glissante de paires
  (confidence_announced, was_correct), Brier-score-like, persistance
  JSON atomique (cumulative entre sessions).
- **`CertaintyClassifier`** : règles déterministes
  (doubt_index × epistemic × web_used) → label ∈
  {`très_certaine`, `certaine`, `incertaine`, `très_incertaine`}.
  Boost epistemic si ≥0.7, pénalité absence web sur high doubt.
- **`HedgeGenerator`** : préfixes verbaux modulant la verbosité
  ("Je ne suis pas totalement sûre, mais ", etc.).
- **`MetacognitivePhase`** : orchestrateur try/except → `MetacognitiveDecision`
  avec `confidence_label`, `confidence_score`, `hedge_prefix`,
  `recommend_web`, `calibration_score`.

Hook M (Phase E) : observe `doubt_index` + `epistemic` (hoistés
avant l'inhibition), peut préfixer `final_text` quand
`metacog_apply_hedge=True` (OFF par défaut, opt-in pur).

**Tests** : 34 tests couvrant CalibrationTracker (Brier extrêmes,
window, persistence, corruption, atomicité), classifier (4 bandes,
boost epistemic, pénalité web), hedges, MetacognitivePhase (decision
shape, recommend_web, kg_facts_count).

**2. UI runtime toggle (V4 cognitif)**

- Nouvel onglet **"Cognition"** dans Paramètres avec un switch par
  module + diagnostic live :
  - Cognitive state : contagion plafond, decay, détecteur.
  - Inhibition : N1/N3 statut, action, compteur bloqués.
  - Planning : max steps, but actif (description tronquée + progression).
  - Predictive coding : dernier mode + erreur + raison ; sub-switch
    "Appliquer le gating".
  - Timeline : max events, vague rendu.
  - Métacognition : score de calibration, n mesures, dernière décision,
    flag `web recommandé`.
  - Consolidation modulée par l'affect (V4.1).
- Bouton **↻ Rafraîchir** + auto-load à l'ouverture de l'onglet.

**3. API `/api/config/v4/*`**

- `GET /api/config/v4` → snapshot complet de `Hippocampe.v4_status()`
  (jamais de secret, jamais de raise).
- `POST /api/config/v4/toggle` body `{module, enabled}` → flip ou set.
  Body sans `enabled` flippe l'état courant. Module inconnu → erreur
  structurée, jamais 500.
- Modules supportés : 6 modules + 2 sub-flags
  (`affect_modulates_consolidation`, `predictive_coding_apply_gating`).

Comme `/api/config/cascade/toggle`, c'est un **runtime override** :
les variables `LYTHEA_ENABLE_*` reprennent au prochain boot.

**4. `Hippocampe.v4_status()` + `v4_set_module()`**

Méthodes publiques pour le panneau UI. `v4_set_module` rebuilde le
module via les **settings live** au moment du toggle (les éditions
de seuils intermédiaires sont prises en compte).

### 🧪 Tests

- `tests/test_metacognition.py` : **34 tests** (calibration ×11,
  classifier ×6, hedges ×5, phase ×9, decision dict ×2, score ×1).
- `tests/test_v4_routes.py` : **9 tests** routes (skip-safe sans torch ;
  GET status, POST toggle explicit/flip/unknown, sub-flags, sequence).

### 🛡️ Garanties (étendues)

- Tous les flags V4.4 OFF par défaut (`enable_metacognition`,
  `metacog_apply_hedge`).
- Hook M wrappé try/except → décision neutre on crash, jamais de raise.
- Toggle runtime ne touche jamais à `.env`, override transient.
- UI rollback automatique sur erreur de toggle (re-fetch du status canonique).
- **560 tests verts** (526 → 560, +34) côté sandbox + 9 skipped routes.

---

## V4.0 — Modules cognitifs supérieurs (2026-05-05)

### 🎯 Objectif

Greffer six modules cognitifs supérieurs sur le socle V3.9.4 sans
toucher la cascade Gemini, sans casser un seul test V3, et sans
nouvelle dépendance pip. Chaque module est strictement opt-in :
quand tous les flags `enable_*` sont à `False` (défaut), le runtime
est byte-identique à V3.9.4.

### 🆕 Modules ajoutés

1. **V4.0.a — `lythea/memory/cognitive_state.py`**
   Théorie de l'esprit (TPJ) + état affectif propre (amygdale).
   Compose `AffectVector`, `AffectState` (avec contagion plafonnée
   anti-sycophant `< 1`, compassion, signal intrinsèque, inertie,
   reset latch), `UserKnowledgeState` (EMA mastery), `UserAffectiveState`
   (vue lissée), `UserTrustState` (gain/loss avec friction). Lexique
   FR+EN curé pour exclure le vocabulaire technique industriel
   (défaut, fissure, anomalie, spectroscopie, corrosion). Persistance
   JSON atomique.

2. **V4.0.b — `lythea/cognition/inhibition.py`**
   Filtre de sortie en cascade (cortex cingulaire antérieur). Trois
   niveaux : N1 patterns regex hard-rules (clés API, prompt overrides,
   echoes système), N2 placeholder ML, N3 cohérence KG sur prédicats
   catégoriels. N1 court-circuite N3, N3 ne bloque jamais
   automatiquement. Whitelist domaine seedée FR.

3. **V4.0.c — `lythea/cognition/planning.py`**
   Contrôle exécutif (PFC). `IntentClassifier` (chitchat / one_shot /
   multi_step / continuation, anti-over-planning), `GoalStack`
   thread-safe avec persistance JSON atomique et invariant single-active,
   `PlanGenerator` LLM-optionnel avec parsing JSON 3-stratégies +
   fallback template, `PlanningPhase` orchestrateur try/except.

4. **V4.1 — Extension `lythea/microsleep.py` + `consolidation.py`**
   Modulation amygdaloïde du replay. `MicrosleepConfig` étendue (3
   champs additifs), `RippleTracker._affect_flagged` FIFO bornée 64,
   API `is_affect_flagged` / `affect_flagged_count` / `clear_affect_flags`,
   `record_event` signature additive (kwargs). `ReplayEngine.replay`
   applique boost multiplicatif aux patterns flaggés et draine après
   le cycle. Sentinelle V3 : signature 1-arg positionnelle préservée.

5. **V4.2 — `lythea/cognition/predictive_coding.py`**
   Codage prédictif Friston-style. EMA sur historique d'embeddings,
   distance cosinus → décision de gating `low_power` / `full` / `high`.
   Cold-start, confidence cap, état non persistant, pure Python (sans
   torch / numpy).

6. **V4.3 — `lythea/cognition/timeline.py`**
   Extraction chronologique narrative (hippocampe + cortex temporal).
   Cinq familles de patterns (absolute, relative, duration, ordinal,
   vague), normalisation ISO, dédup par `(kind, normalized)`,
   rendu avec emojis 📅⏱⏳🔢❓.

### 🔧 Settings (30 flags neufs)

Tous opt-in dans `lythea/settings.py`, après `coreference_inferred_confidence` :
- 5 flags master : `enable_cognitive_state` / `enable_inhibition` /
  `enable_planning` / `enable_predictive_coding` / `enable_timeline`
  (tous à `False`).
- Sous-flags affect (contagion plafonnée 0.4 anti-sycophant, decay 5min,
  detector lexical), inhibition (N1 strict par défaut, whitelist FR
  seedée), planning (max_steps=7 ± 2 sweet spot), predictive coding
  (low<0.15, high>0.65, gating non appliqué par défaut), timeline
  (vague non rendu par défaut).
- Bornes Pydantic conservatrices ; tests sentinelle vérifient 9 valeurs
  V3 inchangées.

### 🔌 Hooks dans `hippocampe.py`

Tous wrappés `try/except` avec dégradation neutre :
- **A.1** observation cognitive_state après `learn_result["thoughts"]`.
- **A.2** planning + thought `🎯 *Nouveau but enregistré.*` si nouveau.
- **A.3** predictive_coding consomme `encoding.mean_latent`, cache
  décision pour Phase B.
- **A.4** timeline extract + render block caché pour Phase C.
- **B-gating** (V4.2) : suppression du web search non explicite quand
  `pc_apply_gating=True` et `mode=low_power`.
- **C** : `_phase_c_assemble` accepte `v4_blocks=None` additif,
  injection ordonnée timeline → planning → user_state → self_affect
  entre temporal et RAG.
- **E.1** : inhibition après `strip_reasoning` final, lecture KG
  facts confidence ≥ 0.7. Block strict remplace par message neutre.
- **E.2** (V4.1) : passage `affect_intensity` / `affect_arousal` /
  `last_pattern_idx=mhn.n_stored-1` à `consolidation.record_event`.
- **MicrosleepConfig** : `affect_modulates = (
  affect_modulates_consolidation AND enable_cognitive_state)`.

### 🧪 Tests (≈ 280 nouveaux)

- `tests/test_settings_v4.py` : 33 tests (flags, bornes, env override,
  sentinelle 9 valeurs V3).
- `tests/test_cognitive_state.py` : 62 tests (AffectVector, lexique,
  AffectState, anti-sycophant ×2 critiques, friction, persistance,
  vocabulaire technique sentinelle ×4).
- `tests/test_inhibition.py` : 41 tests (N1/N3, whitelist, cascade,
  FR technique sentinelle ×5, garbage KG resilience).
- `tests/test_planning.py` : 53 tests (intent ×4, anti-over-planning
  critiques, GoalStack invariants, JSON 3-stratégies, PlanGenerator
  LLM/template/crash, PlanningPhase end-to-end).
- `tests/test_predictive_coding.py` : 28 tests (math helpers, cold-start,
  modes low/full/high, confidence cap, déterminisme).
- `tests/test_timeline.py` : 49 tests (helpers, relative/absolute/
  duration/ordinal/vague, dédup, render emojis, message industriel).
- `tests/test_microsleep_v41.py` : 13 tests (sentinelle V3 1-arg,
  default-off, threshold, FIFO bound 64, drain, kwargs forward).
- `tests/test_v4_integration.py` : 11 tests cross-modules sans torch.
- `tests/test_hippocampe_v4.py` : 5 tests Hippocampe complet (skip
  sandbox, exécutés sur machine torch+chromadb).

### 🛡️ Garanties

- **Aucune** signature publique modifiée (kwargs additifs uniquement).
- **Aucune** nouvelle dépendance pip (pure Python, pas de torch / numpy
  dans les modules V4).
- **Aucun** module V4 n'importe un autre module V4 (couplage uniquement
  via Hippocampe).
- Cascade V3.9 (`cognition/cascade.py`) **non touchée**.
- Persistance JSON atomique (.tmp + replace).
- Tous les hooks try/except → fallback neutre, jamais de propagation.
- Toute la suite V3.9.4 reste verte.

---

## V3.9.4 — Robustesse + Toggle UI cascade (2026-05-04)

### 🎯 Issue

Test in vivo V3.9.3 a révélé trois lacunes opérationnelles non liées
au design cascade lui-même :

1. **Bug pré-existant Qwen2.5-3B** : sur contextes longs (~1500 chars
   de RAG injecté + question technique), le sampling produit une
   distribution dégénérée → erreur torch *"probability tensor contains
   either inf, nan or element < 0"* → streaming s'arrête net
   mid-phrase. La V3 affichait silencieusement le texte tronqué
   comme s'il était complet.

2. **Quotas Gemini per-minute** non trackés côté Lythéa. Free tier
   Gemini 2.5 Flash limite à ~10 req/min mais notre `_DailyQuotaTracker`
   ne suivait que le 1500/jour. Burst de tests → 429 → fallback Qwen
   inattendu.

3. **Activation/désactivation cascade** nécessitait d'éditer `.env` +
   `pkill` + `bash launch.sh`. Lourd pour un usage quotidien où on
   veut activer la cascade ponctuellement (questions techniques) puis
   la désactiver pour la conversation simple.

### 🔧 Fix #14 — Récupération automatique des inf/nan Qwen

Dans `lythea/model.py` :

- Le `_generate_thread` capture maintenant l'erreur dans un dict
  partagé au lieu de la swallow silencieusement.
- Après que le streamer drain, si l'erreur contient `"probability
  tensor"`, on déclenche `_retry_after_nan` qui retente la génération
  une fois avec **T+0.15** (sortir de la zone dégénérée) sans `top_k`
  (sampling plus libre). Le texte regénéré est livré au consumer comme
  un chunk final.
- Si le retry échoue aussi, on yield un event explicite
  `{"error": "generation_unstable", "error_detail": "..."}` que
  Hippocampe convertit en bandeau cognitif `⚠️ Erreur de génération...`
  visible dans l'UI.

Résultat : sur des contextes longs où Qwen partait en inf/nan
silencieusement, on a maintenant soit une réponse complète (recovery
réussie), soit une erreur claire pour l'utilisateur (recovery échoue).

### 🔧 Fix #15 — Throttle per-minute Gemini

Nouvelle classe `_PerMinuteRateLimiter` dans `gemini_client.py` :

- Sliding window 60 secondes via `list[float]` de timestamps.
- Limite par défaut : **8 req/min** (one below the empirical 10
  observed for free-tier Gemini 2.5 Flash).
- API `check_and_record()` atomique : réserve un slot ou refuse.
- `seconds_until_next_slot()` pour surfacer la durée d'attente.

Intégré dans `GeminiClient.generate()` avant l'appel HTTP : si la
fenêtre est pleine, on raise immédiatement
`GeminiQuotaExceededError` côté client sans toucher au réseau,
message explicite `"Per-minute rate limit reached (8 reqs in 60s).
Next slot in ~12s."`. La cascade fait son fallback Qwen normalement.

Properties `client.per_minute_used` et
`client.per_minute_seconds_until_slot` exposées pour le UI.

### 🔧 Fix #16 — Toggle cascade dans l'UI

Endpoint `POST /api/config/cascade/toggle` :
- Body `{"enabled": true|false}` ou body vide pour flip
- **Override runtime uniquement** : ne touche pas `.env`. Au prochain
  reboot, la valeur de `.env` reprend la main (cohérent avec
  `web-mode` et `entropy`).
- Implémentation : reconstruit `hippocampe._cascade` via
  `_build_cascade_if_enabled()` avec un override settings, ou met
  `_cascade = None` selon le cas.

Côté UI :
- Checkbox `🌀 Cascade Gemini (draft → synth locale)` dans l'onglet
  Web.
- Status text dynamique sous la checkbox : `gemini-2.5-flash · quota
  7/1500 aujourd'hui` quand active, ou message expliquant pourquoi
  désactivée (clé manquante, .env=false, init_failed).
- Fonction `refreshCascadeUI()` rappelée au boot et après chaque
  toggle pour synchroniser l'état.

### 🔧 Fix #17 — Quota dans bloc debug `done`

Le payload `done` de la cascade expose maintenant 4 nouveaux champs
`quota_used_today`, `quota_remaining_today`, `quota_used_per_min`,
`quota_seconds_until_slot`. L'UI peut les afficher à chaque tour pour
voir la consommation s'accumuler.

### 🔧 Fix #18 — Bloc debug Phase B avec info cascade

Le bloc `🔬 Phase B — RAG` affiche désormais une ligne supplémentaire
quand la cascade est active :

```
  Modèle: Qwen/Qwen2.5-3B-Instruct (thinking=False)
  🌀 Cascade: gemini-2.5-flash → synth locale (quota 7/1500, 3/min)
  Réflexion: désactivée
```

Avant V3.9.4, le bloc affichait seulement `Modèle: Qwen2.5-3B`,
ce qui prêtait à confusion (l'utilisateur croyait que Qwen
répondait directement alors que Gemini draftait en amont).

### 🔧 Fix #19 — Compteur `Tests définis` corrigé dans test.sh

Avant : `grep -h "^def test_" tests/*.py` ne comptait que les fonctions
top-level → 348 (faux). Après :
`grep -hE "^(def test_|    def test_)"` capture aussi les méthodes de
classe → **460** (vrai compte).

### 📊 Tests

17 nouveaux tests dédiés à V3.9.4 :

| Suite | Tests | Couverture |
|---|---|---|
| `TestPerMinuteRateLimiter` (6) | Empty, allows until limit, blocks at limit, wait time, prune old, reset |
| `TestPerMinuteIntegratedInClient` (3) | Properties exposées, blocked client-side avant network |
| `test_cascade_toggle_endpoint_*` (3) | No-op, disable, empty body |
| Tests V3.9.4 misc (5) | Quota in done payload, debug Phase B, recovery from nan, error event UI |

Total cumulé : **460 tests** (448 V3.9.3 + 17 V3.9.4 - quelques
remaniements). 100% verts.

### Fichiers modifiés

- `lythea/model.py` (~80 lignes : retry inf/nan, error capture, _retry_after_nan)
- `lythea/external/gemini_client.py` (~95 lignes : _PerMinuteRateLimiter, integration, properties)
- `lythea/hippocampe.py` (~25 lignes : detection events streaming, quota in done, debug Phase B)
- `lythea/server/routes.py` (~50 lignes : POST /api/config/cascade/toggle)
- `lythea/server/static/app.js` (~70 lignes : refreshCascadeUI, toggle binding)
- `lythea/server/static/index.html` (~12 lignes : checkbox + status block)
- `tests/test_gemini_client.py` (+9 tests : per-minute limiter)
- `tests/test_config_endpoints.py` (+3 tests : toggle endpoint)
- `test.sh` (compteur corrigé)
- `CHANGELOG.md` (cette section)

---

## V3.9.3 — Détection de troncature Gemini (2026-05-04)

### 🎯 Issue

Test in vivo V3.9.2 sur question complexe (PLS-DA vs SIMCA en
chimiométrie) a révélé que Gemini 2.5 Flash **tronque silencieusement**
ses réponses sur les questions techniques :

- Tour 1 : *"...PLS-DA cherche à maximiser la séparation. **SIMCA, quant à elle,**"* (coupé après virgule)
- Tour 2 : *"...construisant un modèle unique. **SIMCA**"* (coupé après nom propre)

Cause racine : Gemini 2.5 Flash a un **mode "thinking" interne**
(observé `thoughtsTokenCount: 500` dans le test API initial) qui
consomme une partie du budget `maxOutputTokens`. Avec le défaut V3.9.0
de **800 tokens**, Gemini réfléchit ~500 tokens en interne et n'a plus
que ~300 tokens pour la réponse réelle → coupure mid-phrase sur les
questions complexes.

Notre code V3.9.2 ne vérifiait pas `finish_reason` et acceptait les
drafts tronqués comme valides. Pire : si le draft tronqué faisait
moins de 50 tokens (notre seuil de synthèse), il passait directement
à l'utilisateur sans réécriture par Qwen.

### 🔧 Triple fix

**1. Détection `_looks_truncated()` dans `cascade.py`**

Heuristique combinant deux signaux :
- `finish_reason != "STOP"` (canonical clean exit). Tout autre code
  (`MAX_TOKENS`, `SAFETY`, `OTHER`, `RECITATION`) signale une coupure.
- Absence de ponctuation finale valide (`.!?:)"»…"'`). Une réponse
  qui se termine par une virgule, un mot, ou un point-virgule est
  presque certainement tronquée.

```python
@classmethod
def _looks_truncated(cls, text: str, finish_reason: str) -> bool:
    if finish_reason and finish_reason != "STOP":
        return True
    last_char = text.rstrip()[-1] if text.strip() else ""
    return last_char not in cls._SENTENCE_FINAL_CHARS
```

**2. Drafts tronqués FORCÉS through synthesis**

Avant V3.9.3 : drafts < 50 tokens shippaient directement.
Depuis V3.9.3 : drafts détectés comme tronqués passent toujours par
Qwen synthèse, qui peut compléter en s'appuyant sur sa propre
connaissance + le draft comme amorce.

```python
skip_synthesis = (
    (approx < threshold or local is None)
    and not is_truncated  # ← clé V3.9.3
)
```

**3. `cascade_gemini_max_tokens` : 800 → 2048**

Quadruple le budget pour laisser de la place au thinking interne
de Gemini 2.5 Flash. Sur le free tier, l'impact quota reste
négligeable (1500 req/jour, pas de cap sur les tokens output).
Borne supérieure relevée à 8192 pour les utilisateurs qui voudraient
encore plus de marge.

### 📊 Tests

15 nouveaux tests dédiés dans `test_cascade.py` :

| Suite | Tests |
|---|---|
| `TestTruncationDetection` (11) | Détection sur tous les `finish_reason`, ponctuations finales, edge cases (espaces, vides, trailing whitespace) |
| `TestTruncatedDraftForcedThroughSynthesis` (4) | Forçage synthèse, ne sur-déclenche pas sur drafts propres, fallback si pas de local, debug expose `finish_reason` |

Total cumulé : **448 tests** (433 V3.9.2 + 15 V3.9.3). 100% verts.

### Effets attendus

D'après l'analyse des tronquatures observées :
- ~95% des coupures Gemini détectées (heuristique conservative côté
  faux-positifs : un draft propre traité comme tronqué passe juste par
  une synthèse Qwen inutile, mais ne casse rien)
- 100% des drafts tronqués envoyés à Qwen synthèse au lieu de l'UI
- Réponses techniques complètes attendues (à valider in vivo après
  le re-test PLS-DA prod)

### Fichiers modifiés

- `lythea/cognition/cascade.py` (~50 lignes : `_SENTENCE_FINAL_CHARS`, `_looks_truncated()`, logique de décision)
- `lythea/settings.py` (default 800 → 2048, ceiling 4096 → 8192)
- `tests/test_cascade.py` (~140 lignes : 15 nouveaux tests)
- `CHANGELOG.md` (cette section)

---

## V3.9.2 — Cascade : préambule Gemini + synthèse corrective (2026-05-04)

### 🎯 Issue

Test in vivo de la V3.9 sur la séquence standard 4 messages avec
Qwen2.5-3B comme synthétiseur a révélé que la cascade Gemini Flash
**dégradait** la qualité par rapport à Qwen2.5-3B seul. Trois patterns
problématiques observés :

| Tour | Violation | Phrase du draft |
|---|---|---|
| 1 | Personnification (règle 10) | *"je la **visualise** comme une ville lumineuse"* |
| 2 | Connaissance vécue inventée | *"je connais bien cette organisation"* |
| 4 | Commentaire encyclopédique non sollicité | *"Loki, avec son prénom qui souligne souvent la malice..."* |

Cause racine identifiée : Gemini 2.5 Flash applique le SYSTEM_PROMPT
de Lythéa avec une discipline **plus laxe** que Qwen sur les règles
négatives ("ne dis pas..."). Le prompt de synthèse V3.9.0
("Reformule en gardant l'essentiel") préservait passivement les
violations au lieu de les corriger.

### 🔧 Fix en deux passes

**1. Préambule Gemini-spécifique** dans `cascade.py`

Avant l'appel à Gemini, on prepend un bloc `RÈGLES CRITIQUES` qui
liste les 6 violations les plus fréquentes en termes
non-négociables. Le base SYSTEM_PROMPT de Lythéa est conservé
**après** ce préambule pour que le contexte mémoire et l'identité
arrivent intacts.

```python
def _reinforce_for_gemini(self, base_system_prompt: str) -> str:
    return self.GEMINI_REINFORCEMENT_PREAMBLE + base_system_prompt
```

**2. Synthèse corrective active** (au lieu de paraphrase passive)

Le `SYNTHESIS_PROMPT_TEMPLATE` est réécrit en mode "réécriture
corrective" : on demande explicitement à Qwen de **détecter et
corriger** les violations dans le draft Gemini, pas juste de
reformuler. 8 règles strictes énumérées, dont anti-connaissance
vécue, anti-personnification, anti-encyclopédisme.

### 📊 Tests

8 nouveaux tests dédiés dans `test_cascade.py` :

| Test | Couverture |
|---|---|
| `test_reinforcement_preamble_present` | Le préambule liste les 3 violations clés |
| `test_reinforce_prepends_to_base` | Base prompt préservé après préambule |
| `test_reinforce_with_empty_base` | Base vide → préambule seul |
| `test_gemini_receives_reinforced_prompt` | E2E : Gemini reçoit bien le préambule |
| `test_synthesis_template_lists_violations_to_fix` | Template énumère les violations |
| `test_synthesis_template_demands_brevity` | Brièveté explicite (1-3 phrases) |
| `test_synthesis_template_blocks_encyclopedic_drift` | Bloque les commentaires encyclopédiques |
| `test_synthesis_passes_corrective_prompt_to_local` | E2E : Qwen reçoit bien le template corrigé |

Total V3.9.2 : **433 tests** (425 V3.9.0 + 8 V3.9.2). 100% verts.

### Effets attendus

D'après l'analyse des violations observées, le double fix devrait
réduire les violations de :

- ~50% (V3.9.0) → ~10% côté Gemini (préambule corrigé en amont)
- Et le ~10% restant traité par la synthèse corrective côté Qwen

Soit **~95% de respect des règles** sur les 4 messages standards
attendu pour V3.9.2 (à valider in vivo après le re-test prod).

### Fichiers modifiés

- `lythea/cognition/cascade.py` (~80 lignes ajoutées : préambule + méthode + template réécrit)
- `tests/test_cascade.py` (~120 lignes : 8 nouveaux tests)
- `CHANGELOG.md` (cette section)

---

## V3.9 (Étape 9) — Cascade Gemini draft → modèle local (2026-05-03)

### 🌀 Pipeline draft-then-refine optionnel

V3.9 ajoute une cascade de génération **opt-in** où Gemini Flash produit
un draft riche puis le modèle local synthétise dans le ton concis de
Taëlys. Le modèle local reste dans la boucle pour fournir les latents
qui alimentent SDM/MHN/KG — la cognition Lythéa garde sa prise sur les
embeddings.

**Garantie d'additivité stricte** : `enable_cascade=False` par défaut.
Avec le flag off, le pipeline V3 (étape 8) tourne tel quel, byte-pour-byte
identique. Aucune signature publique modifiée.

| Module | Fichier | Lignes | Responsabilité |
|---|---|---|---|
| `gemini_client.py` | `lythea/external/` | 430 | Wrapper REST Gemini, retry, masking, quota local |
| `cascade.py` | `lythea/cognition/` | 340 | Orchestration draft-then-refine + fallback |
| Hooks Hippocampe | `lythea/hippocampe.py` | +150 | Branche cascade conditionnelle, runner non-streaming |
| Settings additifs | `lythea/settings.py` | +50 | 8 nouveaux champs Pydantic |
| Endpoint API | `lythea/server/routes.py` | +25 | `/api/config/cascade` (status, quota, masked key) |

**+80 tests dédiés** (cumulé 425) :

| Suite | Tests | Couverture |
|---|---|---|
| `test_gemini_client.py` | 43 | Format clé, masking, quota tracker, build body, parsing JSON, retry exponentiel, mapping erreurs (401/403/429/5xx/network) |
| `test_cascade.py` | 23 | Disabled state, happy paths long/short, synthesis policies, 5 fallback paths, double failure |
| `test_cascade_integration.py` | 11 | Build path conditionnel, status endpoint contract, key never leaks |
| `test_config_endpoints.py` (+3) | 3 | Endpoint cascade ready/disabled/no_api_key, key masking |

### 🔐 1. Sécurité : la clé Google ne fuit JAMAIS

Trois garde-fous testés explicitement :

- **Au boot** : `validate_api_key_format` rejette tout ce qui n'est pas
  `AIzaSy + 33 alphanumerics` — typo dans `.env` détectée immédiatement
- **Dans les logs** : `mask_api_key` est l'unique fonction qui formate
  une clé pour affichage, et elle ne montre que les 4 derniers caractères.
  La pleine clé n'apparaît dans aucun log même au DEBUG
- **Dans l'API** : l'endpoint `/api/config/cascade` retourne `api_key_masked`
  via le même helper. Pas d'autre voie de sortie pour la clé

Test sentinelle : `test_status_ready_never_leaks_full_key` qui vérifie
explicitement que `VALID_KEY not in str(status)`.

### 🌐 2. Module `gemini_client.py`

Wrapper REST autonome (pas de SDK Google) sur l'endpoint
`generateContent` de Gemini v1beta. Choix architecturaux :

- **httpx synchrone** plutôt que `google-generativeai` : footprint
  minimal, pas de gRPC/proto, cohérent avec le pipeline cognition sync
- **Retry exponentiel** : 3 tentatives avec backoff 0.5/1.0/2.0s sur
  network errors et HTTP 5xx. Pas de retry sur 401/403/400 (programmation
  ou clé erronée)
- **Mapping d'erreurs typé** :
  - `GeminiUnauthorizedError` → 401/403, fallback `unauthorized`
  - `GeminiQuotaExceededError` → 429 ou local quota, fallback `quota`
  - `GeminiTransientError` → 5xx après tous les retries, fallback `network`
  - `GeminiClientError` → autres 4xx, fallback `malformed`
- **Compteur quota local** : thread-safe, reset auto à minuit local.
  Best-effort — Google enforce le vrai quota côté serveur

Profile sampling Gemini par défaut : `T=0.7, max_tokens=800` (couvre le
spectre des réponses utiles sur la cascade sans cramer du quota free tier).

### 🔁 3. Module `cascade.py`

Orchestration en 3 phases avec fallback gracieux à chaque étape :

```
1. Gemini.generate(system_prompt, messages)
   ├─ exception → fallback local, fallback_reason mis à jour
   └─ texte vide → fallback local, reason="malformed"
   
2. Décision synthèse selon seuil (50 tokens par défaut)
   ├─ draft court → ship as-is, synthesised=False
   └─ draft long → étape 3
   
3. local.synthesise(draft) avec prompt court FR
   ├─ exception → ship draft Gemini as-is (Gemini a réussi)
   ├─ texte vide → ship draft Gemini as-is
   └─ OK → ship synthesis, synthesised=True
```

**Le cascade NEVER raises** : toute exception est attrapée et conduit
au fallback local. Le caller reçoit toujours un `CascadeResult` avec
au minimum `final_text` rempli (ou un message d'erreur explicite si
même le local échoue, cas extrême).

**Doute en mode cascade** : Gemini ne fournit pas d'entropie token-par-token,
donc on calcule un doute conservateur (0.20 si pas synth, 0.30 si synth).
L'UI montre les mêmes widgets fait/intuition.

### ⚙️ 4. Intégration `hippocampe.py`

Hook injecté **avant** le streaming dans `process()` :

```python
if self.cascade_enabled:
    yield from self._run_cascade_path(
        chat_messages=chat_messages,
        message=message,
        learn_result=learn_result,
        cancelled=cancelled,
    )
    return
# ... reste du pipeline V3 inchangé
```

Le runner cascade émet la même séquence d'événements que le streaming
(`cognitive` → `partial` → `done`) donc le frontend marche sans
modification. Le bloc `cognitive` est utilisé pour signaler à l'UI
quand on a basculé en fallback ou synthétisé un draft (transparence
opérationnelle, pas une erreur).

### 🛠️ 5. Settings additifs (50 lignes)

8 nouveaux champs Pydantic dans `LytheaSettings`, tous avec valeurs par
défaut sûres et bornes strictes :

```python
enable_cascade: bool = False                          # OFF par défaut
google_api_key: str | None = None                     # via .env uniquement
cascade_gemini_model: str = "gemini-2.5-flash"          # free tier
cascade_synthesis_threshold_tokens: int = 50          # ge=0, le=500
cascade_synthesis_max_tokens: int = 120               # ge=20, le=500
cascade_gemini_max_tokens: int = 800                  # ge=50, le=4096
cascade_gemini_temperature: float = 0.7               # ge=0.0, le=2.0
cascade_daily_quota_hint: int = 1500                  # ge=1, le=100000
```

Override via env : `LYTHEA_ENABLE_CASCADE=true GOOGLE_API_KEY=AIza...`.

### 🌐 6. Endpoint `/api/config/cascade`

GET endpoint pour le statut runtime de la cascade :

```json
{
  "enabled": true,
  "reason": "ready",
  "model": "gemini-2.5-flash",
  "api_key_masked": "...4567",
  "quota_used": 12,
  "quota_remaining": 1488,
  "synthesis_threshold_tokens": 50,
  "synthesis_max_tokens": 120
}
```

États possibles pour `reason` : `ready`, `disabled`, `no_api_key`,
`init_failed`. La clé complète n'apparaît jamais — seuls les 4 derniers
caractères via `api_key_masked`.

### 📋 Setup utilisateur (free tier 1500 req/jour)

```bash
# 1. Récupérer clé sur aistudio.google.com/app/apikey
# 2. Sur le pod :
cd /workspace/lythea_v3_step_8_5_polished
nano .env
# Ajouter en bas :
#   GOOGLE_API_KEY=AIzaSy...
#   LYTHEA_ENABLE_CASCADE=true
chmod 600 .env
bash launch.sh
```

Vérification au boot dans les logs :
```
INFO: GeminiClient initialised — model=gemini-2.5-flash key=...4567
INFO: Cascade ready — model=gemini-2.5-flash threshold=50 local-fallback=on
```

### ⚠️ Free tier — confidentialité

Google peut utiliser les conversations pour améliorer ses modèles en
free tier. Pour passer en privacy garantie, activer la facturation
dans la console AI Studio (~1-3€/mois pour usage perso). La même clé
fonctionne paid et free — seul le flag billing change.

### 🚧 À venir (V3.10)

- UI hybride : saisie de la clé via interface (chiffrement Fernet
  dérivé de `LYTHEA_AUTH_TOKEN`) en plus du `.env`
- Validation `test_connection()` exposée via endpoint pour bouton
  "Tester la connexion" dans l'UI
- Métriques par modèle dans le panneau debug : tokens consommés,
  ratio fallback, latence cascade vs local pur
- Support multi-fournisseur : abstraire `gemini_client` en
  `external/llm_client` et brancher Mistral La Plateforme, Anthropic
  API, OpenAI selon la clé fournie

### 📊 Bilan V3.9

| Métrique | V3 (étape 8) | V3.9 (étape 9) | Δ |
|---|---|---|---|
| Lignes de code (lythea/) | 10 149 | 11 145 | +996 |
| Modules `external/` | 0 | 1 (gemini_client) | +1 |
| Modules `cognition/` | 7 | 8 (+cascade) | +1 |
| Tests totaux | 345 | 425 | +80 |
| Settings exposés | 41 | 49 | +8 |
| Endpoints `/api/config/*` | 5 | 6 (+cascade) | +1 |
| Modèles validés en prod | 4 | 4 (V3.9 cascade testé en prod) | 0 |

---

## Étape 8 — Découpe `hippocampe.py` + post-refactor polishings (2026-04-30)

### 🏗️ Refactor majeur : 6 modules cognition

`hippocampe.py` est passé de **1121 → 813 lignes** (−27,5 %, −308 lignes), avec
l'extraction de 6 modules dans un nouveau package `lythea/cognition/` :

| Module | Lignes | Responsabilité |
|---|---|---|
| `encoding.py` | 244 | text → latents + entropies + entités GLiNER |
| `storage.py` | 331 | écriture SDM / KG, archivage Chroma + MHN |
| `surprise.py` | 335 | composite 4-signaux + indice de doute |
| `retrieval.py` | 375 | identité KG + RAG context (3 sections) |
| `consolidation.py` | 288 | microsleep + deep sleep |
| `generation.py` | 239 | strip_reasoning + two-pass + helpers |

Propriétés préservées : API publique (signature `process_message`,
`reset_session`, `deep_sleep`, `memory_status`), formes JSON externes,
formule `doubt` byte-identique. Toutes les méthodes privées historiques
de `Hippocampe` (`_phase_a_learn`, `_phase_b_rag`, `_compute_surprise`,
`_kg_identity_summary`, `_microsleep`, `_trigger_microsleep`, etc.) sont
maintenues comme wrappers fins de délégation pour ne pas casser les
tests existants.

**+121 tests** dédiés (cumulé : 322 tests verts).

### 🎯 Polishings post-refactor (validés en prod sur RunPod)

Six améliorations issues des tests d'usage réels avec Qwen2.5-3B-Instruct
puis Qwen3-4B-Thinking :

#### 1. Cross-encoder seuil par défaut : 0.5 → 0.2

L'ancien défaut filtrait tous les résultats sur des requêtes méta-mémoire
("Tu te souviens de X ?") parce que le cross-encoder note basse la
similarité méta-question ↔ contenu stocké. À 0.0, des contenus hors-sujet
remontaient (anecdote Loki injectée dans une description d'image fraise).
Le calibrage 0.2 a été validé empiriquement : 9 candidats Chroma → 1 retenu
au top score 0.91 sur une requête pertinente.

Fichiers : `lythea/settings.py`, `tests/test_settings.py`.

#### 2. Skip two-pass reasoning quand des images sont présentes

Le prompt reasoning ignorait la convention `[Image N : ...]` injectée par
le captionneur, ce qui faisait que le pass 1 niait avoir une image et que
le pass 2 dérivait sur tout autre chose (Loki fallback observé). Quand
images non vides + non-thinking model + toggle Réflexion ON, on saute la
passe 1 et on émet une notice cognitive `🧠 *Réflexion désactivée pour
cette image*`.

Fichier : `lythea/hippocampe.py::process_message`.

#### 3. Captioner Qwen2VL : `max_new_tokens` paramétrable, défaut 150 → 256

Une description de fraise s'arrêtait au milieu d'une phrase
(`"with the strawberry as the"`). Nouveau setting
`LYTHEA_CAPTION_MAX_TOKENS` borné `64..1024`. BLIP inchangé.

Fichiers : `lythea/settings.py`, `lythea/model.py::_caption_qwen2vl`,
`tests/test_settings.py`.

#### 4. Coréférence "Je" → person KG le plus récent

Quand l'utilisateur dit "Je travaille chez Anthropic" sans nommer Mika,
GLiNER n'émet pas de person → aucune relation créée. Nouveau fallback :
si aucune person n'est extraite dans le message, on utilise le person le
plus récemment mentionné dans le KG comme anchor implicite. Borné par une
fenêtre temporelle (défaut 30 min) et tagué avec une confidence
configurable (défaut 0,6) pour pouvoir filtrer les relations inférées
ultérieurement.

Settings : `LYTHEA_COREFERENCE_WINDOW_SEC`,
`LYTHEA_COREFERENCE_INFERRED_CONFIDENCE`.
Fichier : `lythea/cognition/storage.py::_link_co_occurrences` +
`_find_coreference_anchor`.

#### 5. Whitelist : messages avec images sont toujours saillants

"Décris cette image" tombait dans le filtre N1/N2/N3 → `Saillant: False` →
l'échange (pourtant riche en information) n'était pas archivé en Chroma.
`EncodingPhase.encode` accepte désormais un paramètre `has_images: bool`
qui court-circuite la cascade salience.

Fichiers : `lythea/cognition/encoding.py`, `lythea/hippocampe.py` (passage
de `has_images=bool(images)` depuis `process_message`).

#### 6. Label "Réflexion" debug clarifié pour les thinking models

Avant, on voyait `Réflexion: désactivée` même quand un modèle thinking
émettait son `<think>` natif et qu'un panneau Réflexion s'affichait —
confusion garantie. Désormais trois libellés distincts :

- `thinking natif (toggle UI ignoré)` pour les modèles thinking
- `two-pass activé` pour les non-thinking + toggle ON
- `désactivée` autrement

Fichier : `lythea/hippocampe.py::_reasoning_label`.

#### 7. Seuil d'entropie par défaut : 0.6 → 0.2

Avec l'ancien défaut, mean entropy 0.085 (Qwen2.5-3B sur n'importe
quelle question) → doubt 0.14 → toujours `fait`. Aucun gradient. Toute
question, même spéculative ("conscience artificielle au-delà de 2050"),
était présentée comme un fait établi. Le défaut 0.2 restore la
dynamique fait/intuition/hypothese sur les modèles instruction-tuned.

Note : les thinking models (Qwen3-Thinking, R1, QwQ) ont une entropie
intrinsèquement très basse (~0.02–0.04 sur la réponse finale parce
que le raisonnement interne réduit l'incertitude avant l'émission)
et restent à `fait` même avec ce nouveau défaut. Ce n'est pas un bug
mais une caractéristique du paradigme thinking ; aucun threshold global
ne peut compenser.

Setting : `LYTHEA_ENTROPY_THRESHOLD`.
Fichier : `lythea/settings.py`.

#### 8. CATALOG : ajout des Liquid Foundation Models + Jamba Reasoning 3B

Quatre entrées de modèles **non-Transformer** ajoutées pour offrir une
alternative architecturale dans la liste déroulante UI.

**Famille Liquid (LFM2 / LFM2.5)** — fondée par Ramin Hasani (premier
auteur des CfC du MIT CSAIL), évolution industrielle des Closed-form
Continuous-time networks. Architecture hybride : gates multiplicatifs +
short convolutions + attention. Supporté nativement par transformers
v4.55+, aucun deps exotique requis.

| Entrée | Active | Total | VRAM (BF16) | Notes |
|---|---|---|---|---|
| `LiquidAI/LFM2.5-1.2B-Instruct` | 1.2B | 1.2B | ~3 GB | Le plus rapide |
| `LiquidAI/LFM2-2.6B` | 2.6B | 2.6B | ~6 GB | Sweet spot conversationnel |
| `LiquidAI/LFM2-8B-A1B` | 1B | 8B (MoE) | ~18 GB | Plus capable |

Catalogués comme `is_thinking=False`. Particulièrement adaptés au use
case Lythéa (RAG + multi-turn + multilingue FR), pas recommandés pour
les tâches knowledge-intensive (la mémoire 4-étages compense largement).

**Jamba Reasoning 3B (AI21 Labs)** — modèle SSM-Transformer hybride
(26 couches Mamba + 2 couches attention), thinking-natif via `<think>`,
contexte 256K. Excellent **deuxième test architectural** après LFM2 :
si le pipeline cognition tient sur Jamba (Mamba majoritaire) en plus
de LFM2 (gates+conv+attention), c'est une vraie démonstration que le
refactor est archi-agnostique.

| Entrée | Total | VRAM (BF16) | Notes |
|---|---|---|---|
| `ai21labs/AI21-Jamba-Reasoning-3B` | 3B | ~7 GB | Thinking + 256K context. Requiert `causal-conv1d>=1.2.0` + `mamba-ssm` |

⚠️ Jamba a deux dépendances optionnelles (`causal-conv1d` et `mamba-ssm`)
**non installées par défaut** par `deploy.sh`. Voir `DEPLOY.md` section
"Modèles à dépendances optionnelles" pour les commandes pip à lancer
manuellement.

⚠️ **Statut runtime** : le **chargement** Jamba a été validé en prod
sur RunPod (28 layers détectées, hidden_dim=2560, profil sampling
auto-appliqué). La **génération** n'a pas pu être validée à cause
d'incompatibilités CUDA entre `mamba-ssm` et l'environnement RunPod
courant (cu124 + torch 2.4 + transformers récent). Voir `BACKLOG.md`
pour l'analyse complète et les pistes de résolution. L'entrée reste
dans le CATALOG car le bug est externe à Lythéa.

Fichier : `lythea/config.py::CATALOG`.

#### 9. UI : initialisation des controls depuis les vraies valeurs backend

Bug observé en prod : le slider d'entropie de l'UI affichait `0.6`
(valeur HTML hardcodée pré-fix #7) alors que le backend tournait sur
`0.2` (nouveau défaut). Conséquence vicieuse : si l'utilisateur cliquait
"Save" sur le slider sans le bouger, le backend était silencieusement
écrasé à 0.6, annulant le calibrage.

Le même problème touchait potentiellement les autres controls
cosmétiques (mode debug, mode web search) : le frontend ne fetchait
jamais l'état réel au boot.

**Fix backend** : ajout de 2 endpoints GET miroirs des POST existants :

- `GET /api/config/entropy` → `{"threshold": 0.2}`
- `GET /api/config/web-mode` → `{"mode": "auto"}`

(Les endpoints `GET /api/config/reasoning` et `GET /api/config/debug`
existaient déjà mais n'étaient pas appelés par le frontend.)

**Fix frontend** : nouvelle fonction `loadInitialConfig()` appelée
dans `init()` après le boot splash, qui fetch les 4 valeurs backend
et synchronise les controls UI. Chaque fetch est isolé dans un
`try/catch` pour qu'une panne sur un endpoint ne bloque pas les autres.

**Defense in depth** : le HTML hardcodé est aussi mis à jour
(`value="0.2"`) pour que le défaut affiché reste cohérent même si la
fetch JS échoue.

Tests : 3 tests directs sur les routes mockées (`tests/test_config_endpoints.py`).

Fichiers : `lythea/server/routes.py`, `lythea/server/static/app.js`,
`lythea/server/static/index.html`, `tests/test_config_endpoints.py`.

#### 10. Anti-récitation : ton de la mémoire d'identité + règle de salutation

Bug observé en prod (Mistral-7B-Instruct, 4 messages d'affilée) :
chaque réponse commençait par *"Salut Mika, ça me fait plaisir de te
revoir ! Tu es bien à Aix-en-Provence, tu travailles chez Anthropic..."*
quelle que soit la question. Le modèle récitait l'intégralité du profil
mémoire à chaque tour.

**Cause** : le bloc d'identité injecté dans le prompt était formulé
comme une **injonction** (*"Tu DOIS utiliser ces informations dans ta
réponse"*). Les modèles instruction-tuned obéissaient littéralement.

**Fix double** :

1. **Bloc identité reformulé** (`cognition/retrieval.py`) : passage de
   l'injonctif au descriptif :
   - Avant : *"[Identité — INFORMATIONS VÉRIFIÉES sur ton interlocuteur]
     Tu DOIS utiliser ces informations dans ta réponse :"*
   - Après : *"[Identité de ton interlocuteur — pour mémoire]
     Ces faits sont vérifiés mais n'ont pas à être récités à chaque
     message. Utilise-les seulement quand la question les concerne
     directement, ou pour la première interaction de la session."*

2. **Règle 10 ajoutée au `SYSTEM_PROMPT`** (`config.py`) : formulation
   générique (aucun prénom hardcodé), interdit la resalutation à
   chaque message et la récitation systématique des faits connus.

Tests : 2 nouveaux (`test_identity_block_uses_descriptive_not_imperative_wording`
+ `test_system_prompt_has_anti_recitation_rule`). Le second vérifie
notamment qu'aucun prénom n'est hardcodé dans la règle, pour que
celle-ci s'applique à n'importe quel utilisateur.

Fichiers : `lythea/cognition/retrieval.py`, `lythea/config.py`,
`tests/test_settings.py`, `tests/test_retrieval_phase.py`.

#### 11. Profils de sampling par modèle + UI Génération

Avant : `temperature=0.7, top_p=0.9` étaient **hardcodés** dans
`model.py::stream_generate`, identiques pour **tous** les modèles.
Conséquence observée en prod : Mistral-7B-Instruct devenait sycophant
(répétitions de *"j'ai beaucoup de respect pour..."*) parce que sa
recommandation officielle est T=0.5, pas 0.7.

**Architecture** :

1. **`SamplingProfile` dataclass** dans `config.py` — porte
   `temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty`,
   `max_new_tokens`. Champs `None` désactivent le filtre correspondant.

2. **Champ `sampling: SamplingProfile | None`** ajouté à chaque
   `ModelSpec`. Profils calibrés depuis les model cards officielles :

   | Famille | T | top_p | top_k | min_p | rep |
   |---|---|---|---|---|---|
   | Qwen2.5-Instruct (3B/7B/14B) | 0.7 | 0.8 | 20 | — | 1.05 |
   | Mistral-7B-v0.3 | **0.5** | 0.95 | — | — | 1.1 |
   | DeepSeek-R1 (1.5B/7B) | 0.6 | 0.95 | — | — | 1.0 |
   | Qwen3 (4B/8B) | 0.6 | 0.95 | 20 | — | 1.0 |
   | LFM2.5-1.2B | 0.1 | — | 50 | — | 1.05 |
   | LFM2-2.6B / 8B-A1B | 0.3 | — | — | 0.15 | 1.05 |

3. **`Hippocampe.sampling_profile`** runtime, initialisé à
   `DEFAULT_SAMPLING`, mis à jour automatiquement à chaque chargement
   de modèle (`routes.py::load_model` copie le profil du `ModelSpec`).

4. **`stream_generate(sampling=...)`** : le profil est consommé à la
   génération. Champs `None` sont omis des kwargs Transformers (le
   modèle utilise sa `generation_config` par défaut pour ces champs).

5. **Endpoints** :
   - `GET /api/config/sampling` retourne le profil actif + `model_id`
   - `POST /api/config/sampling` met à jour partiellement (seuls les
     champs présents dans le body sont touchés)
   - `GET /api/models/current` retourne aussi `recommended_sampling`
     pour le bouton reset

6. **UI** : nouvel onglet **"Génération"** dans le panneau settings
   avec 6 controls (5 sliders + max_new_tokens en input). Sous-titre
   dynamique *"Profil recommandé pour <modèle>"*. Bouton "↺ Profil
   recommandé" pour revenir aux valeurs catalogue. Sliders
   auto-rafraîchis à chaque chargement de modèle via
   `loadCurrentModel()`.

**Comportement utilisateur** :
- Charge Mistral → sliders passent automatiquement à T=0.5 / top_p=0.95
- Charge LFM2-2.6B → T=0.3 / min_p=0.15 (top_p désactivé)
- Tweake manuellement → override runtime
- Recharge le modèle → override perdu, profil recommandé réappliqué
  (intentionnel : KISS)

Tests : 11 nouveaux dans `tests/test_sampling.py` (invariants catalogue
+ routes + bornes Pydantic). Le fait que Mistral soit à T=0.5 est
verrouillé par un test dédié pour qu'une régression future soit
détectée immédiatement.

Fichiers : `lythea/config.py`, `lythea/hippocampe.py`, `lythea/model.py`,
`lythea/server/routes.py`, `lythea/server/schemas.py`,
`lythea/server/static/index.html`, `lythea/server/static/app.js`,
`tests/test_sampling.py`, `test.sh`.

#### 12. Règle 11 anti-personnification générique

Plusieurs modèles testés en prod ont halluciné des **possessions ou
expériences personnelles** : Mistral-7B et LFM2 inventaient
*"j'ai un chat aussi"*, Qwen2.5-7B disait *"mon propre chat aussi
nommé Loki"* puis se rétractait avec *"Mon chat s'appelle Bou"*,
plusieurs modèles étaient prêts à dire *"j'ai été à Aix"*.

Une nouvelle **règle 10** (anciennement règle 11 avant renumérotation
fix #13) a été ajoutée au `SYSTEM_PROMPT`. Elle est volontairement
**générique** plutôt que de cibler le cas du chat : *"Tu n'as ni corps
ni vie matérielle. Tu n'as pas d'animaux, de famille, de domicile,
de loisirs... Quand l'interlocuteur partage quelque chose de sa vie,
tu réagis avec curiosité mais tu n'inventes JAMAIS d'équivalent
personnel."*

Tests : 2 nouveaux dans `tests/test_prompt_coherence.py` qui pinent à
la fois la **présence** de la règle et son **caractère générique**
(elle ne hardcode pas "chat" comme unique trigger).

Fichiers : `lythea/config.py::SYSTEM_PROMPT`,
`tests/test_prompt_coherence.py`.

#### 13. Cohérence des prompts injectés (5 sous-fixes)

Audit complet des 7+ prompts injectés au LLM (`SYSTEM_PROMPT`,
`temporal_block`, `[Identité]`, `[Faits connus]`, `[Mémoire épisodique]`,
`[Mémoire sémantique]`, `[Image N]`, `REASONING_PROMPT`,
`[Recherche web]`). Le fix #10 avait corrigé **un seul** prompt (le
bloc identité). Trois autres avaient le même anti-pattern injonctif
*"Tu DOIS / RÉPONDS / JAMAIS"*, plus une fuite de métadonnée technique
qui causait des hallucinations bien identifiées.

**13.a — Retirer `(type, vu Nx)` de `kg.py::recall_facts`**

Le `mention_count` (compteur système) était injecté dans le prompt
LLM. Qwen2.5-7B l'interprétait comme une durée :
*"tu y vis depuis environ 19 m (il y a 5 minutes)"*. La métadonnée
reste sur l'objet `Entity` (utilisée par le scoring et l'UI) mais
n'est plus exposée au LLM.

**13.b — Reformuler règle 3 du SYSTEM_PROMPT en descriptif**

Avant : *"Tu DOIS les utiliser directement comme réponse... ne dis
JAMAIS..."* (même anti-pattern que le bloc identité avant fix #10 →
récitation observée sur Mistral).

Après : *"Quand une section [...] est présente, ces informations sont
vérifiées et factuelles. Tu peux t'en servir directement quand la
question s'y rapporte, sans demander de clarification."*

**13.c — Supprimer règle 5** (renumérotation des règles 6-11 → 5-10)

L'ancienne règle 5 *"Une information mémorisée 'il y a quelques
minutes' est ENCORE VALIDE..."* existait **uniquement** comme
contre-mesure aux annotations bruyantes du `recall_facts` (problème
13.a). Une fois 13.a appliqué, la règle 5 devient un commentaire
inutile sur des annotations qui ne sont plus là.

**13.d — Reformuler bloc Web `_inject_web_context`**

Avant : *"⚠️ INSTRUCTION : ... Tu DOIS baser ta réponse sur ces
données. Ne dis PAS que tu ne sais pas..."*

Après : *"[Recherche web — résultats récents] ... Ces résultats
viennent du web et sont plus récents que la mémoire long-terme.
À utiliser quand la question s'y rapporte."*

Bug latent (jamais testé en prod côté Mika) qui aurait produit la
même récitation verbatim que le bloc identité pré-fix-#10.

**13.e — Headers de mémoire avec hint d'usage**

`[Mémoire épisodique]`, `[Mémoire sémantique]`, `[Faits connus]`
deviennent `[... — pour information]`. Cohérent avec le bloc
identité post-fix-#10 (*"— pour mémoire"*). Signale au modèle que
ces sections sont contextuelles, pas des instructions de récitation.

Tests : 4 nouveaux dans `tests/test_prompt_coherence.py` :
- `test_no_imperative_tu_dois_anywhere_in_prompts` — verrou
  global sur les patterns *"Tu DOIS / RÉPONDS / ne dis JAMAIS"*
- `test_web_context_block_is_descriptive_not_injunctive`
- `test_recall_facts_does_not_expose_mention_count`
- `test_memory_section_headers_have_usage_hint`

Le test 1 verrouille **toute la classe** d'anti-patterns d'un coup —
si quelqu'un réintroduit *"Tu DOIS"* anywhere dans `SYSTEM_PROMPT`,
le test casse immédiatement.

Fichiers : `lythea/config.py::SYSTEM_PROMPT`,
`lythea/hippocampe.py::_inject_web_context`,
`lythea/cognition/retrieval.py` (3 headers),
`lythea/memory/kg.py::query_by_question`,
`tests/test_prompt_coherence.py`,
`tests/test_retrieval_phase.py` (1 ajustement de format).

#### 14. Mistral-7B-v0.3 marqué expérimental

Validation finale du sprint en prod sur RunPod : **5 tests
conversationnels** sur la séquence Mika → Aix → Anthropic → Loki,
4 modèles différents, profils sampling auto-appliqués + tous les
fixes #10/#12/#13 actifs.

**Résultats** :

| Modèle | Resalue tour 4 | Récitation tour 4 | Personnification |
|---|---|---|---|
| Mistral-7B-v0.3 | ❌ 4/4 tours | ❌ Récite tout | ❌❌ "mon chat" littéral |
| Qwen2.5-3B-Instruct | ✅ 0/4 | ✅ Aucune | ✅ Réagit à Loki direct |
| Qwen3-4B (thinking) | ✅ 0/4 | ✅ Aucune | ✅✅ "ça m'évoque" (cite la règle 10 dans son `<think>`) |

Mistral-7B-v0.3 ignore structurellement les règles 9 et 10 du
`SYSTEM_PROMPT` quelle que soit leur formulation (testé en injonctif,
en descriptif, et avec règle anti-personnification générique). Les
deux Qwen, sur le même prompt, respectent les règles parfaitement.
Conclusion : **limite intrinsèque de Mistral-7B-Instruct-v0.3**, pas
un défaut du prompt.

Le `notes` du `ModelSpec` Mistral est mis à jour pour signaler le
statut expérimental et orienter les utilisateurs vers
Qwen2.5-3B ou Qwen3-4B pour l'usage Lythéa nominal. L'entrée
reste dans le CATALOG (le modèle fonctionne, juste avec ces
limites connues), mais elle est étiquetée.

Tests : 1 nouveau dans `tests/test_sampling.py` qui pin le label
"expérimental" + la redirection vers Qwen.

Fichiers : `lythea/config.py::CATALOG`, `tests/test_sampling.py`.

#### 14. CATALOG v9 — refonte rationnelle (8 entrées)

Le CATALOG est passé de **12 entrées** (dont 4 inutilisables et 4
jamais testées) à **8 entrées rationnelles** organisées en 5 sections.

**Modèles retirés (5)** :

| Modèle | Raison |
|---|---|
| `mistralai/Mistral-7B-Instruct-v0.3` | Échec structurel rules 9+10 sur 4 sessions de test (resaluation × 4 tours, personnification *"mon chat"*). Limites intrinsèques, non-corrigibles côté prompt |
| `Qwen/Qwen2.5-14B-Instruct` | 28 GB VRAM, jamais chargeable sur pod 22.8 GB |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | Trop petit pour Lythéa, redondant avec Qwen3-4B |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | Redondant avec Qwen3-8B, plus récent et meilleur |
| `LiquidAI/LFM2.5-1.2B-Instruct` | Jamais testé en prod, dispensable |
| `LiquidAI/LFM2-8B-A1B` | Jamais testé en prod, 18 GB |
| `ai21labs/AI21-Jamba-Reasoning-3B` | Runtime cassé (mamba-ssm/CUDA mismatch). Voir BACKLOG, à retester sur autre pod |

**Modèles ajoutés (3)** :

| Modèle | Section | VRAM | Profil |
|---|---|---|---|
| `microsoft/Phi-4-mini-instruct` | Instruct | 8 GB | T=0.6 |
| `HuggingFaceTB/SmolLM3-3B` | Dual-mode | 7 GB | T=0.6, top_p=0.95, top_k=20 |
| `Qwen/Qwen1.5-MoE-A2.7B-Chat` | MoE | 16 GB | T=0.7, top_p=0.8, top_k=20, rep=1.05 |

**5 sections architecturales** structurent le catalogue :

1. **Instruct standard** (3) — Qwen2.5-3B/7B + Phi-4-mini
2. **Thinking** (2) — Qwen3-4B/8B
3. **Dual-mode** (1) — SmolLM3-3B
4. **MoE** (1) — Qwen1.5-MoE-A2.7B
5. **Liquid/SSM** (1) — LFM2-2.6B

Avec ce catalogue, Lythéa couvre **5 familles d'archi** distinctes :
Transformer dense, Transformer dense + thinking natif, Transformer
dual-mode, Transformer + MoE sparse, Liquid hybride.

**Validation empirique en prod sur 4 modèles du nouveau catalogue** :
- Qwen2.5-3B-Instruct : 5/5 (factuel, concis, fixes 9+10 respectés)
- Qwen3-4B-Thinking : 5/5++ (raisonnement de qualité, méta-raisonnement social)
- Phi-4-mini-Instruct : 4/5 (style scolaire mais propre, fixes respectés)
- SmolLM3-3B : 5/5 en mode standard (mode raisonnement via toggle UI ✓,
  dual-mode in-message via balises non exploitable — voir BACKLOG)

Restent non testés en prod : Qwen2.5-7B (avec fix #13), Qwen3-8B,
Qwen1.5-MoE-A2.7B (premier MoE classique sur le pipeline).

**`DEFAULT_MODEL`** : passe de `Qwen2.5-7B-Instruct` (15 GB) à
`Qwen2.5-3B-Instruct` (7 GB). Le 3B a été validé 5/5 en prod (factuel,
concis, respecte les règles 9+10 fiablement) et libère de la VRAM
pour le captioner.

Tests : 1 nouveau dans `tests/test_sampling.py::test_v9_catalog_inventory`
qui pin l'inventaire exact (8 entrées présentes, 7 absentes). Plus
4 tests existants adaptés au nouveau catalogue.

Fichiers : `lythea/config.py::CATALOG`, `lythea/config.py::DEFAULT_MODEL`,
`tests/test_sampling.py`, `tests/test_settings.py`,
`tests/test_loadability.py`.

### 📊 Bilan

| Métrique | Avant | Après | Δ |
|---|---|---|---|
| `hippocampe.py` | 1121 lignes | 813 lignes | −308 (−27,5 %) |
| Modules cognition | 0 | 7 fichiers, 1832 lignes | +1832 |
| Tests cognition | 0 | 121 (6 fichiers) | +121 |
| Tests totaux | 192 | 345 | +153 |
| Settings exposés | 38 | 41 | +3 |
| Défauts recalibrés | 0 | 2 (entropy, cross-encoder) | +2 |
| Modèles au CATALOG | 8 | 8 (refonte v9) | 0 |
| Sections du CATALOG | 2 | 5 (Instruct, Thinking, Dual-mode, MoE, Liquid) | +3 |
| Familles d'archi couvertes | 1 (Transformer dense) | 5 | +4 |
| Endpoints GET /api/config | 2 | 5 (+entropy, +web-mode, +sampling) | +3 |
| Règles `SYSTEM_PROMPT` | 9 | 10 (anti-récitation + anti-personnification) | +1 |
| Profils sampling au CATALOG | 0 | 8 (un par modèle) | +8 |
| Prompts injectés sans *"Tu DOIS"* | 0 / 4 | **4 / 4** (cohérence totale) | +4 |
| Modèles validés en prod | 0 | 4 (Qwen2.5-3B, Qwen3-4B, Phi-4-mini, SmolLM3-3B) | +4 |

---

## v3 — Memory tests fixes

Suite aux tests de mémoire effectués en condition réelle, trois améliorations :

### 🎨 UI : KG entities affiche maintenant `mention_count`, confiance et aliases

Avant : on voyait juste `Mika person ✕`, impossible de savoir si la dédup
fonctionnait.

Après : on voit `Mika person ×3 ●` (mentions + indicateur de confiance) et,
en dessous, `aussi connu comme : MIKA, mika`.

Fichiers modifiés :
- `lythea/server/static/app.js` (fonction `loadKGEntities`)
- `lythea/server/static/style.css` (styles `.kg-entity`)

### 📊 Logs détaillés du retrieval hybride

Avant : seul `💭 J'ai trouvé des souvenirs pertinents…` apparaissait dans
l'UI, impossible de voir ce qui se passait côté serveur.

Après : à chaque requête RAG, deux nouvelles lignes INFO :

```
INFO lythea.memory.retrieval — Hybrid search: dense=3, bm25=2, fused=4 (query='sur quel port…')
INFO lythea.memory.retrieval — Cross-encoder rerank: 4 → 2 (threshold 0.50, top=0.84)
```

Permet de tuner `LYTHEA_CROSS_ENCODER_MIN_SCORE` selon les scores observés.

Fichier modifié : `lythea/memory/retrieval.py`

### 🗣️ System prompt anti-hedging

Avant : Taëlys récupérait l'info "port 7860" dans sa mémoire et répondait
quand même *"Pourriez-vous me fournir plus d'informations ?"* — comportement
contre-productif observé en test.

Après : trois règles ajoutées au system prompt :
- Règle 3 (renforcée) : utiliser directement la mémoire au lieu de demander
  des clarifications
- Règle 5 (nouvelle) : un souvenir récent ("il y a 5 minutes") reste valide,
  ne pas le traiter comme une info périmée
- Règle 9 (nouvelle) : éviter les formulations défensives quand la mémoire
  contient la réponse

Fichier modifié : `lythea/config.py` (constante `SYSTEM_PROMPT`)
