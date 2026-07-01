# 📘 Guide de déploiement détaillé

## ⚡ Méthode rapide

```bash
# Sur ta machine RunPod / serveur
cd /workspace                    # ou le répertoire qui contient ton lythea actuel

# 1. Sauvegarde de l'ancienne version
mv lythea lythea_backup_$(date +%Y%m%d) 2>/dev/null || true

# 2. Extraire la nouvelle version
# Le flag --no-same-owner évite les warnings bénins "Cannot change ownership"
# quand le tarball a été créé par un autre uid (typiquement uid 999 sandbox).
tar --no-same-owner -xzf lythea_v5_9_torch28.tar.gz
cd lythea_v5_8_combo/lythea_v5_8

# 3. Rapatrier les données existantes (si tu en as)
cp -r ../lythea_backup_*/data ./ 2>/dev/null || true
cp ../lythea_backup_*/.env ./ 2>/dev/null || true

# 4. Installer les nouvelles dépendances
bash deploy.sh

# 5. Vérifier que tout marche
bash test.sh

# 6. Démarrer
bash launch.sh
```

## 🔧 Ce que `deploy.sh` fait

1. Vérifie Python ≥ 3.10
2. Installe : `pydantic`, `pydantic-settings`, `slowapi`, `rapidfuzz`
3. Crée `.env` à partir de `.env.example` si absent
4. Génère un `LYTHEA_AUTH_TOKEN` aléatoire si absent
5. Crée les répertoires `data/`

## 🧪 Ce que `test.sh` fait

1. Vérifie la syntaxe Python
2. Lance les tests pytest (322 tests : 192 historiques + 121 cognition + 9 polishings)
3. Affiche les statistiques

> **Note** : `test.sh` a une whitelist hardcodée des fichiers de tests pour
> esquiver les dépendances optionnelles. Pour tout lancer en force :
> `python3 -m pytest tests/ -q`

## 🛠️ Déploiement manuel

### Dépendances Python supplémentaires

Le projet requiert ces nouvelles deps (en plus de tes deps Lythéa actuelles) :

```bash
pip install --upgrade \
    pydantic>=2.10 \
    pydantic-settings>=2.0 \
    slowapi>=0.1.9 \
    rapidfuzz>=3.0
```

Si certaines deps de base de Lythéa ne sont pas dans ton env (fastapi, chromadb, etc.), tu peux les installer ainsi :

```bash
pip install fastapi chromadb sentence-transformers gliner uvicorn rank-bm25 pillow ddgs
```

Ou laisser `python3 run.py` les auto-installer au démarrage (logique présente depuis la version originale).

### Configuration

```bash
cp .env.example .env
echo "LYTHEA_AUTH_TOKEN=$(openssl rand -hex 32)" >> .env
```

### Variables d'env importantes

```bash
LYTHEA_AUTH_TOKEN=...               # token d'accès web
LYTHEA_AUTH_STRICT=0                # 1 = exiger token même en local
LYTHEA_HOST=0.0.0.0
LYTHEA_PORT=7860

# Performance (laisser par défaut sauf cas spécifique)
LYTHEA_EMBED_CACHE_SIZE=1024
LYTHEA_ANALYZE_CACHE_SIZE=64

# Cross-encoder — défaut 0.2 calibré empiriquement (étape 8 polishings).
# Voir CHANGELOG.md pour l'historique des calibrations.
LYTHEA_CROSS_ENCODER_MODEL=BAAI/bge-reranker-v2-m3
LYTHEA_CROSS_ENCODER_MIN_SCORE=0.2

# Image captioner — augmente si les descriptions Qwen2VL sont tronquées.
LYTHEA_CAPTION_MAX_TOKENS=256

# Coréférence "Je → person KG récent" (étape 8 polishings)
LYTHEA_COREFERENCE_WINDOW_SEC=1800           # 30 min de fenêtre
LYTHEA_COREFERENCE_INFERRED_CONFIDENCE=0.6   # baisse-le si tu veux filtrer

# Microsleep (consolidation enrichie)
LYTHEA_RIPPLE_TRIGGER_COUNT=5
LYTHEA_REPLAY_SEQUENCE_LENGTH=4

# Soft memory (opt-in, désactivé par défaut)
LYTHEA_ENABLE_SOFT_MEMORY=0
```

## 🧩 Modèles à dépendances optionnelles

Certains modèles du `CATALOG` ont des dépendances spécifiques qui **ne sont pas
installées par défaut** par `deploy.sh` parce qu'elles rallongent
significativement le déploiement (compilation CUDA native, ~2-5 min) et
peuvent échouer sur certains pods. Si tu ne comptes pas utiliser ces modèles,
tu n'as rien à faire.

### Jamba Reasoning 3B (AI21) — Architecture SSM-Mamba hybride

Modèle SSM-Transformer hybride (26 couches Mamba + 2 attention), thinking
natif, contexte 256K. C'est un excellent **deuxième test architectural**
après LFM2 pour valider que le pipeline cognition de Lythéa est
archi-agnostique.

> ⚠️ **Statut runtime** : la **chargement** de Jamba via transformers
> fonctionne (modèle bien détecté, 28 layers, profil sampling appliqué),
> mais la **génération** nécessite que les kernels Mamba CUDA tournent
> correctement, ce qui dépend d'un alignement précis entre `mamba-ssm`,
> `causal-conv1d`, `transformers` et la version CUDA du pod. Validé en
> prod sur RunPod : la combinaison cu124 + torch 2.4 + transformers
> récent est **problématique**. Voir le BACKLOG pour les détails.

**Tentative d'installation** (à tes risques) :

```bash
# Étape 1 : causal-conv1d
# Le wheel précompilé pour ta combinaison cu/torch/python n'existe peut-être
# pas. Si l'install plante avec "CUDA mismatch", essaie une version pinnée
# qui a plus de wheels précompilés disponibles :
pip install --break-system-packages "causal-conv1d==1.5.0.post8"
# (validé sur RunPod cu124 + torch 2.4.1 + py3.11)

# Étape 2 : mamba-ssm
# La dernière version (2.2.x) référence des symboles transformers obsolètes
# (GreedySearchDecoderOnlyOutput) → ImportError. Idem pour les versions très
# anciennes (2.0.x) qui peuvent avoir d'autres soucis.
pip install --break-system-packages "mamba-ssm==2.2.2"

# Étape 3 (optionnel) : flash-attn pour les perfs nominales
pip install --break-system-packages flash-attn --no-build-isolation

# Vérification
python3 -c "import causal_conv1d; print('causal_conv1d:', causal_conv1d.__version__)"
python3 -c "import mamba_ssm; print('mamba_ssm: OK')"
```

**Pièges connus** (chronologie de notre session de débug) :

1. **`pip install causal-conv1d` plante avec `CUDA mismatch`**
   → Pip tire torch 2.11+cu130 dans son env de build, qui ne matche pas ta
     CUDA 12.4 système. Solution : version pinnée avec wheel précompilé
     (`==1.5.0.post8`).

2. **`import mamba_ssm` plante avec `ImportError: cannot import name
   'GreedySearchDecoderOnlyOutput'`**
   → Le code mamba-ssm 2.2 référence une classe qui a été renommée dans
     transformers récent. Ce code est dans un module utility non utilisé
     par Jamba lui-même, donc on peut le neutraliser (mais c'est sale —
     voir BACKLOG pour solution propre).

3. **`Fast Mamba kernels are not available`** au runtime, alors que
   `import mamba_ssm` marche
   → Les kernels CUDA selective_scan ne sont pas exécutables sur le GPU,
     malgré l'import OK. Tentative ultime : recompiler depuis les sources
     (`pip install git+https://github.com/state-spaces/mamba`). Pas
     toujours suffisant.

**Recommandation pratique** : si tu rencontres l'un de ces problèmes, il
est plus efficace de **passer au modèle suivant** que d'insister sur
Jamba aujourd'hui. La dette d'environnement est résolvable, mais ça
prend potentiellement plusieurs heures et un changement de pod. Voir
BACKLOG.md.

`flash-attn` est optionnel — sans lui, les modèles tournent ~30% plus
lentement (l'attention dense remplace `flash_attn`). À installer si tu
veux les perfs nominales sur Qwen / Mistral / Jamba.

### Recharger Lythéa après installation

```bash
# Pas besoin de rerun deploy.sh, juste relancer le serveur :
bash launch.sh
```

Le modèle apparaîtra dans la liste déroulante de l'UI dès le prochain boot.

## 🔥 Vérification end-to-end

### 1. Boot status

```bash
curl http://localhost:7860/api/boot/status
# Doit retourner : {"ready": true, ...}
```

### 2. Auth bearer

```bash
# Avec token Cloudflare-tunneled →  401 sans token
curl -H "Authorization: Bearer $LYTHEA_AUTH_TOKEN" \
    https://your-tunnel.trycloudflare.com/api/health
```

### 3. KG avec accents

Dis "Je m'appelle François" puis dans une autre session "Je m'appelle francois". Vérifie l'onglet Mémoire — c'est la même entité (mention_count=2).

### 4. Microsleep ripples

Dans les logs après ~10 échanges :

```
🛏️ Microsleep started (exchange #10)
  ⚡ ripple x6 events  •  replay: 3 sequences, 4 patterns  •  compressed: 2
🛏️ Microsleep completed
```

## 🆘 Troubleshooting

### "ModuleNotFoundError: fastapi"

Ces deps ne sont pas dans le `pyproject.toml` (elles étaient auto-installées par `run.py` dans la version originale). Pour les installer manuellement :

```bash
pip install fastapi chromadb sentence-transformers gliner uvicorn rank-bm25 pillow ddgs
```

### Le serveur ne démarre pas

```bash
python3 -c "from rune.server.app import create_app; print('OK')"
```

Si erreur, vérifier les deps avec `pip list`.

### Le splash reste bloqué

Vérifier les logs serveur pour voir où le boot a planté. Le boot continue toujours jusqu'à `ready=True` même en cas d'échec d'une étape.

### Cross-encoder ne se charge pas

Vérifier l'espace disque (~570 MB pour BGE-v2-m3). Sinon utiliser :

```bash
LYTHEA_CROSS_ENCODER_MODEL=BAAI/bge-reranker-base   # ~280 MB
```

### Le RAG ne remonte plus rien (Cross-encoder rerank: N → 0)

Le seuil par défaut a été baissé de 0.5 à 0.2 en étape 8. Si tu repars
d'une vieille version de `.env` qui a `LYTHEA_CROSS_ENCODER_MIN_SCORE=0.5`,
tu vas observer des cas où aucun doc Chroma ne survit au rerank
(typiquement sur les questions méta-mémoire "Tu te souviens de X ?"). 
Solution : passer à `0.2` ou retirer la ligne du `.env` pour utiliser le
nouveau défaut. Pour tester sans toucher à `.env` :

```bash
LYTHEA_CROSS_ENCODER_MIN_SCORE=0.2 bash launch.sh
```

### Caption d'image tronquée mid-phrase

Le captionneur Qwen2VL s'arrête au milieu d'une phrase (`"with the
strawberry as the"`). Augmenter le budget tokens :

```bash
LYTHEA_CAPTION_MAX_TOKENS=400 bash launch.sh
```

### Toutes les réponses sont étiquetées "fait" même les spéculations

Le défaut `LYTHEA_ENTROPY_THRESHOLD` a été baissé de 0.6 à 0.2 en étape 8.
Si tu repars d'une vieille version de `.env` qui a `LYTHEA_ENTROPY_THRESHOLD=0.6`
ou similaire, tu vas observer que toutes les réponses sont étiquetées
`fait` même quand le modèle spécule, parce que les modèles instruction-
tuned modernes produisent des entropies trop basses pour franchir un
seuil aussi élevé. Solution : passer à `0.2` ou retirer la ligne pour
utiliser le nouveau défaut.

Note pour les thinking models (Qwen3-Thinking, R1, QwQ) : leurs entropies
sont si basses (~0.02–0.04) qu'ils restent à `fait` même avec threshold=0.2.
C'est intrinsèque au paradigme thinking, aucun threshold global ne peut
compenser ce comportement — le raisonnement interne réduit l'incertitude
avant l'émission des tokens.

### Réflexion semble activée alors que je l'ai désactivée

Tu utilises probablement un modèle thinking (Qwen3-*, QwQ, R1, o1).
Ces modèles émettent leur `<think>` natif que le toggle UI ne peut pas
désactiver — c'est dans les poids du modèle. Le panneau debug affiche
`Réflexion: thinking natif (toggle UI ignoré)` pour clarifier. Si tu
veux vraiment couper le thinking, charge un modèle non-thinking
(Qwen2.5-Instruct, Llama-Instruct, etc.).

### Token oublié

Voir dans `.env` :

```bash
grep LYTHEA_AUTH_TOKEN .env
```

Ou réinitialiser via Paramètres → Système → "Réinitialiser le token" dans l'UI.

## 🔄 Migration des données

Toutes les nouvelles versions sont **rétro-compatibles** :

- **KG** : `_rebuild_index()` migre les anciens formats au load (accents)
- **Sessions** : format JSON inchangé
- **MHN/SDM** : aucun changement de format
- **Chroma** : aucun changement

Pas de script de migration à exécuter.
