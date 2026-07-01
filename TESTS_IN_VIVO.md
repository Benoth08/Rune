# Plan de tests in vivo — Lythéa V5.5

**Objectif** : valider sur le pod RunPod (vrais modèles, vraie mémoire, vrai web)
toutes les briques V5 / V5.1 / V5.2 / V5.3 / V5.4 / V5.5.

**Nouveau V5.3** : Memory Health Dashboard (clic sur badge version).
**Nouveau V5.4** : Procedural Memory — patterns trigger→approach extraits au microsleep.
**Nouveau V5.5** : Reflection Loop — self-critique sélective sur cas à risque.

---

## Bloc I — Memory Health Dashboard V5.3

### I.1 — Ouverture du dashboard
**Action** : clic sur le badge "Lythéa V5.5" en bas de la sidebar
- **UI attendue** : overlay avec card centrée affichant :
  - Score géant (0-100) en vert/jaune/rouge selon ≥70/40
  - Texte court (cognitive_hint) — ex *"Mémoire saine (78/100)."*
  - 5 barres de progression : Fraîcheur / Couverture / Cohérence / Efficience / Connectivité
  - Stats brutes en bas (entités KG / relations / communautés / chroma / pending)
  - Bouton ✕ et fermeture au clic dehors
- **Log attendu** : `Memory health: score=X freshness=Y coverage=Z ...`

### I.2 — Score évolue après microsleep
**Action** : ouvrir le dashboard, noter le score → faire 10 exchanges → attendre microsleep → réouvrir
- **Attendu** : la cohérence et reachability auront changé (relations + communautés ajoutées au KG).
- **Validation** : différence visible dans les barres.

### I.3 — Score quand Chroma est vide
**Action** : test sur un nouveau pod fraîchement démarré
- **Attendu** : score bas (< 30), hint *"Mémoire encore jeune..."*

---

## Bloc J — Procedural Memory V5.4

### J.1 — Extraction au microsleep
**Action** : faire 10 exchanges thématiques (par ex. 3 calculs Python d'affilée, puis 3 recommandations tech, puis 3 questions sur fichier joint) → attendre microsleep
- **Log attendu** :
  - `Procedural extraction: N new patterns, M total active`
  - Ou `Procedural extraction: no new patterns this cycle` si le LLM n'a rien identifié
- **Validation** : ouvrir `data/procedural/skills.json` — patterns extraits, format `{trigger, approach, confidence, applied_count}`.

### J.2 — Playbook injecté dans le prompt
**Action** : après J.1, faire un exchange normal et regarder les logs DEBUG du prompt système
- **Log attendu** : `Procedural playbook injected: N procedures, X chars`
- **Validation** : le bloc `[Habitudes apprises — à appliquer si pertinent]` est présent dans le prompt envoyé au LLM.

### J.3 — Dédup au microsleep suivant
**Action** : refaire les mêmes types de questions que J.1 → microsleep
- **Log attendu** : `Procedural extraction: 0 new patterns, M total active` (les patterns existants ont juste vu leur applied_count incrémenté)
- **Validation** : `skills.json` n'a pas grossi mais les `applied_count` ont augmenté.

### J.4 — Refus pattern interdit
**Test artificiel** : il faudrait forcer le LLM à proposer un pattern interdit, ce qui est rare en pratique. Plutôt, vérifier les logs sur 50 exchanges : aucun `Procedure refused (forbidden pattern)` n'apparaît pour des conversations normales.

### J.5 — Archivage forgetting curve
**Action** : laisser le système tourner plusieurs jours sans réutiliser certains patterns
- **Log attendu** après microsleep : `Procedural memory: archived N stale procedures`
- **Validation** : dans `skills.json`, les procédures vieilles + peu utilisées ont `archived: true`.

### J.6 — Effet observable sur le comportement
**Action** : après plusieurs microsleeps, comparer le comportement de Taëlys sur une question récurrente
- **Attendu subjectif** : Taëlys devrait répondre plus directement / cohéremment grâce à l'injection du playbook. Pas de métrique automatique ici, c'est ressenti.

---

## Bloc K — Reflection Loop V5.5

### K.1 — Trigger tech_reco → réflexion
**Question** : `Recommande-moi un modèle NER en français`
- **UI attendue** :
  - Pill 🌐 (web tech_reco déclenché)
  - **Puis** cognitive item 🪞 *"J'ai relu ma réponse, elle me paraît correcte."* OU 🔧 *"J'ai relu ma réponse et corrigé N points."*
- **Log attendu** : `Reflection: trigger=tech_reco needs_revision=... issues=... duration=XXms`

### K.2 — Trigger CRAG INCORRECT → réflexion
**Question** : sur un sujet absent de la mémoire, par ex. `Tu te souviens de notre échange sur la pizza ?`
- **UI attendue** :
  - ⚠️ CRAG INCORRECT
  - **Puis** cognitive item de réflexion (Taëlys s'auto-vérifie sur sa réponse "depuis mémoire interne")
- **Log attendu** : `Reflection: trigger=crag_incorrect ...`

### K.3 — Skip réponse trop courte
**Question** : `Bonjour !`
- **Attendu** : pas de cognitive item de réflexion (réponse courte, skip silencieux)
- **Log attendu** : pas de ligne `Reflection:` (skip silencieux fait que la branche n'appelle pas le LLM)

### K.4 — Skip Python tool
**Question** : `Combien font 17 × 23 ?`
- **Attendu** : 🐍 pill, **pas** de cognitive item de réflexion (résultat déterministe)
- **Log attendu** : pas de ligne `Reflection:` car SKIP_PYTHON_RESULT

### K.5 — Skip raisonnement actif
**Action** : activer le toggle "Raisonnement" en haut → poser une question complexe tech_reco
- **Attendu** : DeepReasoningChain prend le relais, **pas** de réflexion V5.5 (skip car reasoning_active)
- **Validation** : DeepReasoning a sa propre phase critique intégrée, donc on évite la double-critique.

### K.6 — Révision appliquée visible
**Question** : forcer un cas où le modèle invente — par ex. `Cite-moi 3 papers récents sur l'EEG predictive coding`
- **Attendu** :
  - Première réponse partielle s'affiche (peut-être avec papers inventés)
  - Cognitive item 🔧 *"J'ai relu ma réponse et corrigé N points."*
  - Le bloc texte se met à jour avec la version révisée (issue de l'event `partial_revised`)
- **Validation** : Taëlys retire ou marque comme incertaines les références qu'elle ne peut pas vérifier.

### K.7 — Doute accumulé déclenche
**Question** : volontairement floue, par ex. `C'est quoi déjà la lib dont on a parlé pour les trucs ?`
- **Attendu** : si Taëlys utilise 3+ marqueurs de doute ("je crois", "peut-être", "il me semble"), réflexion s'active
- **Log attendu** : `Reflection: trigger=doubt_markers ...`

---

## Bloc L — Performance et coûts

### L.1 — Surcoût Reflection sur tech_reco
**Mesure** : sur 5 questions tech_reco, comparer le temps total avec et sans `reflection_enabled`
- **Attendu** : +1-3s sur les cas où la réflexion s'active (timeout configuré à 6s max)
- **Trade-off** : surcoût mais correction de confabulations → ROI dépend de ton usage

### L.2 — Surcoût Microsleep avec V5.4
**Mesure** : temps entre `🛏️ Microsleep started` et `🛏️ Microsleep completed`
- **V5.2** (sans procedural) : ~3-5s avec communities
- **V5.4** : +5-10s pour l'extraction procédurale (un appel LLM)
- **Total V5.4 attendu** : ~10-15s par microsleep — acceptable car asynchrone

### L.3 — Health Dashboard latence
**Mesure** : temps de réponse `GET /api/memory/health`
- **Attendu** : < 100ms sur KG ~100 entités, < 300ms sur ~500 entités
- **Si > 1s** : networkx connected_components ou Chroma get(limit=200) ralentit → réduire l'échantillon

---

## Checklist finale V5.5

- [ ] Badge "Lythéa V5.5" cliquable → modal s'ouvre
- [ ] 5 dimensions affichées avec couleurs cohérentes
- [ ] Au moins 1 microsleep avec `Procedural extraction: X new patterns`
- [ ] `skills.json` contient au moins 1 procédure après usage
- [ ] Au moins 3 cognitive items de réflexion observés (🪞 ou 🔧)
- [ ] Pas de crash sur cas extrêmes (KG vide, Chroma vide, LLM down)
- [ ] Suite tests : `pytest tests/test_memory_health.py tests/test_procedural_memory.py tests/test_reflection.py` → 88 passants

---

## Bloc A — Routage hybride V5.0 (web classifier)

### A.1 — Question d'actualité claire → fast-path tech_reco
**Question** : `Recommande-moi un modèle NER en français`
- **UI attendue** : pill 🌐 *"Recommandation technique demandée, je vérifie les références exactes en ligne…"*
- **Log attendu** : `WEB_DECISION decided=True via=fast_path reason='tech_reco: Recommande-moi'`
- **Validation** : pas de modèles `fr_core_news_md/lg` inventés sans `[N]`.

### A.2 — Faux positif tech_reco doit être bloqué (V4.4 fix 2-passes)
**Question** : `Recommande-moi un mug rigolo`
- **UI attendue** : pas de pill 🌐 (réponse directe sans web)
- **Log attendu** : `WEB_DECISION decided=False via=fast_path` (pas de tech_reco)
- **Validation** : Taëlys répond sur des marques de mugs depuis sa mémoire,
  pas de bullshit Hugging Face.

### A.3 — Casual chat → pas d'appel LLM classifier
**Question** : `Merci beaucoup !`
- **UI attendue** : réponse instantanée, pas de pill 🌐 ni 🐍
- **Log attendu** : `WEB_DECISION decided=False via=fast_path reason='none'`
  (pas d'appel slow-path car `looks_like_question = False`)

### A.4 — Question ambiguë → slow-path LLM tranche
**Question** : `Tu connais une bonne lib pour les graphes en JavaScript ?`
- **UI attendue** : pill 🌐 *"Je pense qu'une vérification en ligne ferait du bien…"*
- **Log attendu** : `WEB_DECISION decided=True via=slow_path reason='llm_classifier: ...'`
- **Validation** : le classifier LLM a bien tranché malgré l'ambiguïté.

### A.5 — Tag /noweb force le silence web
**Question** : `/noweb Recommande-moi un modèle NER en français`
- **UI attendue** : PAS de pill web (alors que sans `/noweb` ça déclencherait)
- **Log attendu** : `manual /noweb tag` dans la decision
- **Validation** : Taëlys répond de mémoire (peut citer flair/spaCy depuis ses connaissances).

### A.6 — Tag /web force le web
**Question** : `/web Quelle est ta couleur préférée ?`
- **UI attendue** : pill 🌐 *"Tu as demandé explicitement une recherche web…"*
- **Log attendu** : `manual /web tag` dans la decision
- **Validation** : recherche faite même sur une question subjective.

---

## Bloc B — Multi-tool router V5.1

### B.1 — Question de calcul → python executor
**Question** : `Combien font 17 × 23 + 89 ÷ 11 ?`
- **UI attendue** : pill 🐍 *"J'exécute du Python pour répondre…"*
- **Log attendu** :
  - `Router (semantic): python (conf=0.6X)` OU
  - `Router (LLM dispatcher): python — ...`
  - Puis `Python executor: ok=True duration=...ms`
- **Validation** : réponse finale contient `399.09` (ou `399.0909...`) avec
  une mention que le calcul a été fait par Python.

### B.2 — Question d'actualité → web tool
**Question** : `Quel temps fait-il à Aix-en-Provence demain ?`
- **UI attendue** : pill 🌐 *"Question d'actualité détectée, je vérifie en ligne."*
- **Log attendu** : `Router (semantic): web (conf=0.6X)` puis SearXNG query.

### B.3 — Concept stable → none (réponse directe)
**Question** : `Explique-moi comment fonctionne la backpropagation`
- **UI attendue** : pas de pill 🌐 ni 🐍
- **Log attendu** : `Router (semantic): none (conf=0.5X)` ou ambiguous puis
  dispatcher = "none"
- **Validation** : réponse pédagogique depuis la connaissance du modèle.

### B.4 — Génération + exécution Python avec plot
**Question** : `Trace le graphe de la fonction sin(x) entre 0 et 2*pi`
- **UI attendue** : pill 🐍, puis dans la réponse Taëlys décrit le graphe
- **Log attendu** : `Python executor: ok=True ... plots=1`
- **Validation** : `_lythea_plot_0.png` généré dans le cwd tempo, base64 dans `_last_python_plots`.

### B.5 — Python avec erreur (gracieux)
**Question** : `Calcule la racine carrée de -1` (la requête déclenche python,
mais le code peut générer une erreur si le LLM oublie d'importer cmath)
- **UI attendue** : pill 🐍, réponse qui explique l'erreur ou retombe en réponse mathématique
- **Log attendu** : si erreur Python → `Python executor: ok=False stderr=ValueError`
- **Validation** : pas de crash serveur, message utilisateur cohérent.

---

## Bloc C — CRAG V5.2

**Pré-requis** : avoir une mémoire long-terme alimentée (sinon CRAG sera
EMPTY systématiquement). Lance d'abord quelques questions pour peupler.

### C.1 — Retrieval CORRECT (silencieux)
**Question** : sur un sujet où Taëlys a beaucoup discuté (par exemple `Comment va Lythéa ?`)
- **UI attendue** : pas de cognitive item CRAG (cas silencieux)
- **Log attendu** : `CRAG verdict: status=correct top=0.XX` avec top ≥ 0.7

### C.2 — Retrieval AMBIGUOUS → rewrite déclenché
**Question** : volontairement vague, par ex. `Le truc qu'on a fait hier`
- **UI attendue** : 🟡 *"Souvenirs partiellement pertinents…"* ou 🔄 *"Première recherche imprécise, j'ai reformulé et trouvé mieux."*
- **Log attendu** :
  - `CRAG rewrite: 'Le truc qu'on a fait hier' → 'X'`
  - `CRAG verdict: status=correct top=0.XX (rewritten=True)`  (si rescue OK)
  - ou `CRAG verdict: status=ambiguous` (si rescue n'améliore pas)
- **Validation** : la query reformulée est sensée et plus spécifique.

### C.3 — Retrieval INCORRECT → signalisation honnête
**Question** : sur un sujet totalement absent de la mémoire, par ex.
`Tu te souviens de mon voyage en Antarctique ?`
- **UI attendue** : ⚠️ *"Rien de vraiment pertinent en mémoire long-terme — je vais répondre depuis ce que je sais."*
- **Log attendu** : `CRAG verdict: status=incorrect top=0.0X`
- **Validation** : Taëlys ne fait PAS de souvenir inventé. Elle dit qu'elle
  n'a pas de trace.

---

## Bloc D — GraphRAG communities V5.2

**Pré-requis** : avoir un KG avec plusieurs entités et relations (idéalement
plusieurs microsleeps déjà passés). Vérifier avant : `len(self.kg.entities) >= 6`.

### D.1 — Détection après microsleep
**Action** : déclencher un microsleep (10 exchanges, ou attendre l'interval)
- **Log attendu** : `Community detection (louvain|components): N entities, M edges → K communities (sizes: [...])`
- **Validation** : `K >= 1`, fichier `data/kg/communities.json` créé/mis à jour.

### D.2 — Summaries générés
- **Log attendu** : pour chaque communauté top-10, soit `Community X summary failed` (cas d'erreur)
  soit pas de log d'échec → summary OK
- **Validation** : ouvrir `communities.json`, vérifier que les top communautés ont un champ `summary` non vide
  de 1-2 phrases qui décrit un thème (genre "Projets techniques autour de Lythéa et Taëlys").

### D.3 — Communauté injectée au retrieval (focus entité)
**Question** : `Parle-moi de Lythéa` (ou d'une entité connue par le KG)
- **Log attendu** : `KG communities injected: N total, focus=1 entities, block_size=XXX chars`
- **Validation** : dans le prompt système (vérifier en logs DEBUG) il y a un
  bloc `[Thématiques de mémoire]` listant la communauté contenant Lythéa.

### D.4 — Communauté top-N si pas de focus
**Question** : générale sans entité KG, par ex. `Qu'est-ce que tu trouves intéressant ?`
- **Log attendu** : `KG communities injected: ... focus=0 entities ...`
- **Validation** : block contient les 3 plus grandes communautés.

### D.5 — Persistence cross-session
**Action** : redémarrer le pod, vérifier les logs au démarrage
- **Log attendu** : `Loaded N persisted communities`
- **Validation** : pas besoin de re-microsleep pour avoir des communautés disponibles.

---

## Bloc E — Tests de régression V5 (anti-confabulation, mémoire)

### E.1 — Mémoire interne ne déclenche pas web
**Question** : `Tu te souviens de notre dernière discussion ?`
- **UI attendue** : pas de pill 🌐
- **Log attendu** : `WEB_DECISION decided=False ... reason='none'`
- **Validation** : Taëlys consulte sa mémoire MHN+Chroma, répond sans web.

### E.2 — Self-answer pattern bloque web
**Question** : `Explique-moi récemment comment fonctionne X` (le mot "récemment" 
ne doit PAS déclencher du temporel car la structure est une demande
d'explication)
- **UI attendue** : pas de pill 🌐
- **Log attendu** : `WEB_DECISION decided=False` (self_answer pattern matché)

### E.3 — Citation web honnête
**Question** : `Recommande-moi un package Python pour la séries temporelles`
- **UI attendue** : pill 🌐 + sources [1], [2], ...
- **Validation** : chaque mention de package (statsmodels, prophet, etc.)
  a un `[N]` qui pointe vers une vraie source dans la liste. Pas de
  citation `[3]` sur un package qui n'apparaît dans aucune source.

---

## Bloc F — Tests d'interaction (combinaisons)

### F.1 — Python + mémoire RAG
**Question** : `Calcule la moyenne de mes notes de la semaine` (en supposant
que les notes ont été stockées dans la mémoire)
- **UI attendue** : 🐍 ou ⚠️ selon ce que CRAG décide sur la mémoire
- **Validation** : 2 scénarios possibles selon données :
  - Si notes en mémoire CORRECT : python exécuté avec les notes
  - Si pas de notes : ⚠️ CRAG signale, Taëlys demande les notes

### F.2 — Web + Python (pas de chaînage encore, mais à valider)
**Question** : `Trouve le prix actuel de l'or et convertis-le en EUR par gramme`
- **Comportement actuel attendu** : un seul outil choisi (web). Pas de chaînage en V5.2.
- **Limitation documentée** : chaînage = V6.

### F.3 — Toggle raisonnement + multi-tool
**Action** : activer le toggle "Raisonnement" en haut + poser une question python
**Question** : `Calcule la corrélation entre 12, 18, 23, 41, 7 et 10, 15, 19, 30, 6`
- **Comportement attendu** : raisonnement déclenché, **mais** comme la question
  matche python sémantique → conflit potentiel. À observer : qui gagne ?
- **Validation** : pas de crash. Documenter le comportement réel pour
  ajustement futur.

---

## Bloc G — Tests d'erreur (gracieux)

### G.1 — Modèle pas chargé, slow-path → fallback safe
**Action** : faire un /api/chat juste après le démarrage avant que le modèle
ne finisse de charger (race condition)
- **Log attendu** : `Router: model not loaded, defaulting to 'none'`
- **Validation** : pas de 500, juste pas d'outil utilisé.

### G.2 — Embeddings router fail → cascade
**Action** : couper temporairement le download de paraphrase-multilingual
(par exemple : `pip uninstall sentence-transformers` puis test, puis réinstaller)
- **Log attendu** : `Both embedding models failed: ...` puis fallback dispatcher
- **Validation** : le pipeline continue de fonctionner même sans le router sémantique.

### G.3 — Python executor timeout
**Question** : `Exécute while True: pass`
- **UI attendue** : 🐍 puis message d'erreur informant du timeout
- **Log attendu** : `Python executor timeout after 5.0s` puis cleanup du subprocess
- **Validation** : le serveur Lythéa reste responsive.

---

## Bloc H — Performance et observation

### H.1 — Coût latence du slow-path
**Mesure** : sur 10 questions ambiguës, mesurer le temps entre
"WEB_DECISION ... via=slow_path" et la réponse finale.
- **Attendu** : ~300-500ms ajoutés vs fast-path. Si > 1s → investiguer.

### H.2 — Hit rate du cache classifier
**Mesure** : après 20 questions variées, vérifier les stats du cache :
```python
# Dans la console serveur
from rune.cognition.web_classifier import get_cache_stats
print(get_cache_stats())
```
- **Attendu** : si tu poses 2-3 fois la même question, hit_rate > 0.

### H.3 — Coût microsleep avec communities
**Mesure** : timer entre `🛏️ Microsleep started` et `🛏️ Microsleep completed`
- **Attendu V5.0** : ~3-5s
- **Attendu V5.2 avec communities** : +1-3s pour la détection et les
  summaries top-10. Si > 10s → considérer `max_communities=5`.

---

## Checklist finale avant prod

- [ ] Tous les blocs A-G donnent le comportement attendu
- [ ] Aucun crash serveur pendant 30+ exchanges
- [ ] Microsleep s'exécute correctement avec communities
- [ ] `communities.json` est créé et contient des summaries
- [ ] Au moins 3 cognitive items CRAG observés (🟡, 🔄, ⚠️)
- [ ] Au moins 3 outils différents déclenchés (web, python, none) sur des questions adaptées
- [ ] `/web` et `/noweb` fonctionnent en override
- [ ] Pas de fuite de secrets (vérifier que `OPENAI_API_KEY` etc. ne sortent pas du subprocess Python)

---

## Si quelque chose ne marche pas

**Symptôme : le router sémantique ne se déclenche jamais**
→ Vérifier que `paraphrase-multilingual-MiniLM-L12-v2` se télécharge bien
au premier appel. Sinon `all-MiniLM-L6-v2` (déjà sur disque via KG) doit
être chargé en fallback. Voir logs `Semantic router loaded: ...`.

**Symptôme : CRAG est toujours EMPTY**
→ Vérifier que Chroma a bien des documents : `self.chroma.count()`.
Si vide, c'est juste que la mémoire long-terme n'a pas encore été
peuplée (besoin de microsleeps).

**Symptôme : pas de communities détectées même avec 10+ entités**
→ Vérifier qu'il y a bien des **relations** entre elles. Sans relations,
pas de graphe → pas de communautés. Le log dit
`Community detection: 0 relations, skipping`.

**Symptôme : Python executor crash systématique**
→ Vérifier que `matplotlib` est installé (`pip install matplotlib`).
Le préambule essaie de l'importer ; en échec il continue sans, mais
les plots ne marcheront pas.
