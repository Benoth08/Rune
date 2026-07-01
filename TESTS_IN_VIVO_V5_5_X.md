# Plan de tests in vivo — Lythéa V5.5.x

**Objectif** : valider sur ton pod RunPod (vrai modèle 7B + vraie
mémoire + vrai web) tous les fixes ajoutés entre V5.5.1 et V5.5.8.

**Préparation** :
1. Déployer le tarball `lythea_v5_5_combo.tar.gz`
2. Vérifier le badge en bas à gauche : doit afficher **Lythéa V5.5.8**
3. Logs ouverts dans un terminal séparé : `tail -f /workspace/lythea.log`
4. Modèle conseillé : `Qwen/Qwen2.5-7B-Instruct`

**Convention** : chaque test indique (1) ce que tu tapes,
(2) ce que la UI doit afficher, (3) ce que tu dois voir dans les logs.

---

## Préambule : nettoyer la pollution historique

Si tu as déjà testé avant V5.5.4 / V5.5.5 et que ta mémoire contient
des entités "Je suis" / "Emilien" ou des docs Chroma pollués, fais
les 2 cleanups **avant** de commencer les tests.

```bash
# 1. KG cleanup (preview)
curl -X POST http://localhost:7860/api/memory/cleanup_noise \
  -H "Content-Type: application/json" -d '{"dry_run": true}' | jq

# Si liste OK, exécute :
curl -X POST http://localhost:7860/api/memory/cleanup_noise \
  -H "Content-Type: application/json" -d '{}' | jq

# 2. Chroma cleanup (preview)
curl -X POST http://localhost:7860/api/memory/cleanup_chroma \
  -H "Content-Type: application/json" -d '{"dry_run": true}' | jq

# Si liste OK, exécute :
curl -X POST http://localhost:7860/api/memory/cleanup_chroma \
  -H "Content-Type: application/json" -d '{}' | jq
```

**Attendu** :
- `cleanup_noise` retourne `{"removed": [...], "removed_count": N}`
- `cleanup_chroma` retourne `{"removed_ids": [...], "matched_substrings": {...}}`
- Une fois exécutés, le KG et Chroma sont propres pour les tests qui suivent

---

## Bloc M — Extraction d'entités (V5.5.1 à V5.5.3)

### M.1 — Prénom en minuscule
**Tape** : `Bonjour, je m'appelle cédric`
- **UI attendue** : entité `Cédric` dans les "Entités connues" (label person)
- **Log attendu** : `Self-intro fallback: 'Cédric' → person`
- **Validation** : la casse est normalisée (Cédric et pas cédric)

### M.2 — Métier multi-mots
**Tape** : `Je suis data scientist`
- **UI attendue** : entité `data scientist` (label role)
- **Log attendu** : `Self-role fallback: 'data scientist' → role`
- **Validation** : capture en multi-mots, en minuscule

### M.3 — Métier avec accent et minuscule
**Tape** : `je suis chimiométricien`
- **UI attendue** : entité `chimiométricien` (label role)
- **Validation** : accent préservé, minuscule conservée pour les rôles

### M.4 — Employeur en minuscule, multi-mots
**Tape** : `Je travaille chez topnir systems`
- **UI attendue** : entité `Topnir Systems` (label organization)
- **Log attendu** : `Self-employer fallback: 'Topnir Systems' → organization`

### M.5 — Lieu de résidence
**Tape** : `j'habite à aix-en-provence`
- **UI attendue** : entité `Aix-En-Provence` (label location)
- **Validation** : title-case appliqué sur chaque segment du tiret

### M.6 — Âge direct
**Tape** : `j'ai 36 ans`
- **UI attendue** : entité `36` (label age)
- **Log attendu** : `Self-age fallback: 36 → age`

### M.7 — Année de naissance (V5.5.3)
**Tape** : `Je suis né en 1985`
- **UI attendue** : deux entités
  - `année:1985` (label date)
  - `41` (label age, dérivé automatiquement de 2026 - 1985)
- **Log attendu** :
  - `Self-birth-year fallback: 1985 → date`
  - `Self-birth-year derived age: 41 → age`
- **Validation critique** : il NE doit PAS y avoir une entité `1985`
  brute (label date) en plus — c'est le bug V5.5.4 fixé

### M.8 — Année future rejetée
**Tape** : `Je suis né en 2099`
- **UI attendue** : aucune entité date créée
- **Validation** : sanity check qui empêche le bug "j'ai été clonée dans le futur"

### M.9 — Tierce personne ne match pas
**Tape** : `il s'appelle Pierre`
- **UI attendue** : aucune entité person ajoutée par notre fallback
  (GLiNER peut quand même trouver Pierre — c'est OK)
- **Validation** : le pattern "il s'appelle" n'est pas auto-référent

### M.10 — Déplacement ≠ résidence
**Tape** : `je vais à Paris demain`
- **UI attendue** : pas d'entité location créée par notre fallback
- **Validation** : `je vais` ≠ `j'habite`

---

## Bloc N — Filtres anti-pollution (V5.5.3)

### N.1 — "Je suis" filtré
**Tape** : `Je suis vraiment content aujourd'hui`
- **UI attendue** : pas d'entité `Je suis` dans les "Entités connues"
- **Validation** : même si GLiNER hallucine "Je suis" comme person,
  il est filtré par ENTITY_NOISE étendu

### N.2 — Apostrophe typographique normalisée
**Tape** : `J'ai un nouveau projet` (avec apostrophe Mac/iPhone si possible : `J’ai`)
- **UI attendue** : pas d'entité `J'ai` créée
- **Validation** : la normalisation gère les deux types d'apostrophe

---

## Bloc O — Endpoints maintenance (V5.5.4 + V5.5.5)

### O.1 — Cleanup noise dry-run
```bash
curl -X POST http://localhost:7860/api/memory/cleanup_noise \
  -H "Content-Type: application/json" -d '{"dry_run": true}' | jq
```
- **Attendu** : JSON avec `"dry_run": true`, `"removed": []` ou liste de candidats, `"kept_count"` égal au nombre actuel d'entités
- **Validation** : aucune modification du KG (dry-run respecté)

### O.2 — Cleanup chroma avec filtre custom
```bash
curl -X POST http://localhost:7860/api/memory/cleanup_chroma \
  -H "Content-Type: application/json" \
  -d '{"contains": ["XYZ_jamais_vu"], "dry_run": true}' | jq
```
- **Attendu** : `"removed_count": 0` (aucun doc ne matche ce filtre exotique)
- **Validation** : le filtre custom fonctionne

### O.3 — Cleanup chroma last_n
```bash
curl -X POST http://localhost:7860/api/memory/cleanup_chroma \
  -H "Content-Type: application/json" \
  -d '{"last_n": 3, "dry_run": true}' | jq
```
- **Attendu** : `"removed_count": 3` (3 derniers docs ciblés)
- **Validation** : sélection par récence fonctionne

---

## Bloc P — Anti-salutation et anti-relance (V5.5.6 à V5.5.8)

**Pré-requis** : avoir au moins 5 documents dans Chroma (le strip
n'active qu'à partir de 3 docs).

### P.1 — Strip salutation streaming
**Tape** : n'importe quelle question simple, par exemple `Je suis né en 1985`
- **UI pendant le streaming** : la réponse commence DIRECTEMENT par le contenu utile
  (par exemple "Tu as 41 ans..." ou "Noté, je retiens 1985...")
- **PAS de flash visible** "Salut Cédric, comment ça va ?" puis disparition
- **Log attendu** : `Streaming greeting stripped: N chars (at first emit)`
- **Validation critique** : c'est le bug initial — il ne doit plus jamais y avoir
  de salutation visible quand Chroma ≥ 3 docs

### P.2 — Pas de strip si pas de salutation
**Tape** : `Quel est le calcul de 17 × 23 + 89 ÷ 11 ?`
- **UI attendue** : streaming normal, sans pause initiale, commence par "Pour calculer..."
  ou "Le résultat est..."
- **Log attendu** : pas de ligne `Streaming greeting stripped`
- **Validation** : le mécanisme n'ajoute pas de latence sur les réponses normales

### P.3 — Pas de relance artificielle sur self-disclosure
**Tape** : `J'ai 36 ans`
- **UI attendue** : réponse courte, accuse réception, **pas de question** type
  "Que penses-tu de ton âge ?" ou "Comment tu vis ça ?"
- **Validation** : le SYSTEM_PROMPT V5.5.6 a son effet

### P.4 — Salutation préservée si conversation neuve
**Action** : créer une nouvelle session (badge "+ Nouveau" dans la sidebar)
**Tape** : `Bonjour`
- **UI attendue** : Lythéa peut saluer en retour (logique sur première rencontre)
- **Validation** : le strip ne se déclenche QUE si Chroma a déjà du contenu
- **Note** : si tu as nettoyé Chroma à zéro avec cleanup_chroma, le seuil ≥ 3
  n'est pas atteint → salutation normale autorisée

### P.5 — Streaming sur préfixe ambigu
**Tape** : `Calcule 2+2`
- **UI attendue** : le modèle peut commencer par "Selon mes calculs..." ou
  "Le résultat..." — pas de blocage du streaming
- **Validation** : les préfixes qui commencent par "Sa"/"Bon"/etc. mais qui
  ne sont PAS des salutations ne bloquent pas le probe

### P.6 — Pas d'appel web sur self-disclosure
**Tape** : `Je suis né en 1985`
- **UI attendue** : pas de pill 🌐, juste le calcul direct
- **Log attendu** : `Skipping reasoning_grounding: user_intent is self-disclosure`
- **Validation** : le garde-fou V5.5.3 contre les déclencheurs web inutiles

---

## Bloc Q — Memory Health Dashboard (V5.3 + fix V5.5.2)

### Q.1 — Ouverture dashboard
**Action** : clic sur le badge "Lythéa V5.5.8" en bas de la sidebar
- **UI attendue** : modal avec score géant (couleur selon valeur), 5 barres
  de progression, stats brutes (entités KG / relations / communautés / chroma)
- **Validation critique** : les valeurs ne doivent PAS être toutes à zéro
  (bug V5.5.2 fixé)

### Q.2 — Évolution après extraction
**Action** : taper plusieurs messages avec self-disclosure (M.1 à M.7),
puis ouvrir à nouveau le dashboard
- **UI attendue** : `n_entities` et `coverage` augmentent
- **Validation** : le dashboard reflète l'état temps réel

---

## Bloc R — Régression intégrale

Refaire les tests des blocs A à H du `TESTS_IN_VIVO.md` original pour
vérifier que les fixes V5.5.x n'ont rien cassé sur les V5.0-V5.2.

Points critiques à re-tester :
- Tag `/web` force la recherche → toujours fonctionnel
- Calcul Python via le router → toujours fonctionnel
- Microsleep + community detection → toujours fonctionnel
- Tests automatisés (cf. section suivante)

---

## Checklist finale V5.5.x

- [ ] Badge "Lythéa V5.5.8" visible
- [ ] M.7 (Je suis né en 1985) → `année:1985` + `41` sans doublon
- [ ] N.1 (Je suis) → entité filtrée
- [ ] O.1-O.3 (endpoints cleanup) → JSON correct
- [ ] P.1 (streaming) → AUCUN flash "Salut Cédric"
- [ ] P.3 (anti-relance) → pas de question artificielle
- [ ] Q.1 (dashboard) → scores non-nuls

---

## Lancer les tests automatisés en parallèle

Sur ton pod, avant ou après les tests in vivo :

```bash
cd /workspace/lythea_v5_5_combo/lythea_v5_5

# Suite complète V5.5.x (les nouveaux tests créés)
python3 -m pytest tests/test_v5_5_x_features.py -v

# Suite V5 originale (V5.0 à V5.2)
python3 -m pytest tests/test_v5_features.py -v

# Tests cognition (V5.3 + V5.4 + V5.5)
python3 -m pytest tests/test_memory_health.py tests/test_procedural_memory.py tests/test_reflection.py -v

# Tests d'extraction (V5.5.1 + V5.5.2)
python3 -m pytest tests/test_self_intro_fallback.py tests/test_self_disclosure_extended.py -v

# Tout d'un coup (ignore les tests torch-only qui passent ailleurs)
python3 -m pytest tests/ -q
```

Sur RunPod (avec torch installé), les chiffres attendus :
- `test_v5_5_x_features.py` : ~40 passants
- `test_self_intro_fallback.py` : ~27 passants
- `test_self_disclosure_extended.py` : ~36 passants
- Suite complète : 900+ passants, < 50 skip

---

## Si quelque chose ne marche pas

**Symptôme : le streaming flashe encore "Salut Cédric"**
- Vérifier que Chroma a bien ≥ 3 docs : `curl http://localhost:7860/api/memory/health`
- Si `n_chroma < 3` → le strip n'est pas activé, c'est normal
- Si `n_chroma ≥ 3` mais le flash apparaît → vérifier les logs pour
  `Streaming greeting stripped` ; absence = bug dans le probe

**Symptôme : entité "1985" brute persiste**
- Vérifier la version du fichier `encoding.py` (la dédup V5.5.4 doit y être)
- Faire un `cleanup_noise` peut aider si l'entité a été archivée avant le fix
- Si le bug réapparaît à chaque envoi → vérifier que `_apply_self_birth_year`
  est bien appelé après les autres fallbacks dans `_extract_entities`

**Symptôme : "Je suis" extrait comme person**
- Vérifier que ENTITY_NOISE étendu contient "je suis" :
  ```bash
  python3 -c "from rune.cognition.encoding import ENTITY_NOISE; print('je suis' in ENTITY_NOISE)"
  ```
- Si False → le fichier `encoding.py` n'a pas le fix V5.5.3

**Symptôme : memory health affiche 0 partout**
- Bug V5.5.2 réapparu : vérifier la version `app.chroma_collection` vs `app.chroma`
- Stack trace dans les logs après `Memory health computation failed`
