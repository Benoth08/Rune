#!/bin/bash
# Lythéa — Script de déploiement automatique
#
# Usage : bash deploy.sh
#
# Ce script :
# 1. Vérifie la version Python
# 2. Installe les nouvelles dépendances du refactoring
# 3. Vérifie/installe les deps de base si absentes (fastapi, chromadb…)
# 4. Crée un .env si absent
# 5. Génère un LYTHEA_AUTH_TOKEN aléatoire si absent
# 6. Crée les répertoires de données nécessaires

set -e

# Aller dans le répertoire du script (robuste au chemin d'extraction)
cd "$(cd "$(dirname "$(readlink -f "$0" 2>/dev/null || echo "$0")")" && pwd)"

# Couleurs
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
BLUE="\033[0;34m"
NC="\033[0m"

ok()    { echo -e "${GREEN}  ✓${NC} $1"; }
warn()  { echo -e "${YELLOW}  ⚠${NC} $1"; }
err()   { echo -e "${RED}  ✗${NC} $1"; }
info()  { echo -e "${BLUE}  →${NC} $1"; }

echo ""
echo "============================================================"
echo "  Rune — Déploiement"
echo "============================================================"
echo ""

# ── 1. Vérification Python ───────────────────────────────────────────

echo "1️⃣  Vérification Python…"
if ! command -v python3 &> /dev/null; then
    err "Python 3 introuvable"
    exit 1
fi
PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$(echo $PY_VERSION | cut -d. -f1)
PY_MINOR=$(echo $PY_VERSION | cut -d. -f2)

if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]); then
    err "Python $PY_VERSION détecté — requis : ≥ 3.10"
    exit 1
fi
ok "Python $PY_VERSION"
echo ""

# ── 2. Détection du flag pip ─────────────────────────────────────────

PIP_FLAGS=""
if pip3 install --dry-run --quiet pydantic 2>&1 | grep -q "externally-managed-environment"; then
    warn "Système externally-managed détecté → utilisation de --break-system-packages"
    PIP_FLAGS="--break-system-packages"
fi

# ── 3. Dépendances du refactoring ────────────────────────────────────

echo "2️⃣  Installation des dépendances du refactoring…"

declare -A REFACTOR_DEPS=(
    ["pydantic"]="2.10"
    ["pydantic_settings"]="2.0"
    ["slowapi"]="0.1.9"
    ["rapidfuzz"]="3.0"
    ["httpx"]="0.27"
)

NEW_INSTALLS=0
for pkg in "${!REFACTOR_DEPS[@]}"; do
    min_ver="${REFACTOR_DEPS[$pkg]}"
    if python3 -c "import ${pkg}" 2>/dev/null; then
        installed=$(python3 -c "import ${pkg}; print(getattr(${pkg}, '__version__', '?'))" 2>/dev/null || echo "?")
        ok "$pkg installé (version $installed)"
    else
        info "Installation de ${pkg/_/-}>=$min_ver…"
        pip3 install $PIP_FLAGS --quiet "${pkg/_/-}>=$min_ver" 2>&1 | grep -v "^WARNING\|^\[notice" || true
        NEW_INSTALLS=$((NEW_INSTALLS + 1))
        ok "${pkg/_/-} installé"
    fi
done

echo ""

# ── 4. Dépendances de base de Lythéa ─────────────────────────────────
#
# Ces deps étaient autrefois auto-installées par run.py. On les
# vérifie explicitement et on les installe si absentes pour éviter
# l'erreur "ModuleNotFoundError: fastapi" / "chromadb" au premier run.

echo "3️⃣  Vérification des dépendances de base de Lythéa…"

declare -A BASE_DEPS=(
    ["fastapi"]=""
    ["uvicorn"]=""
    ["chromadb"]=""
    ["pillow"]="PIL"
)

BASE_INSTALLS=0
for pkg in "${!BASE_DEPS[@]}"; do
    import_name="${BASE_DEPS[$pkg]:-$pkg}"
    if python3 -c "import ${import_name}" 2>/dev/null; then
        ok "$pkg déjà disponible"
    else
        info "Installation de $pkg…"
        pip3 install $PIP_FLAGS --quiet "$pkg" 2>&1 | grep -v "^WARNING\|^\[notice" || true
        BASE_INSTALLS=$((BASE_INSTALLS + 1))
        ok "$pkg installé"
    fi
done

# ── Cœur ML : transformers & co, installés EXPLICITEMENT (pas par run.py) ──
# run.py installait ces deps au 1er boot, ce qui (a) ralentissait le boot,
# (b) cassait les tests de deploy.sh lancés AVANT, (c) en lot multiple tirait
# torch et écrasait la build CUDA. On les pose ICI, en --no-deps, pour ne
# JAMAIS toucher torch/torchvision/torchaudio (pinés par l'image). Les
# compagnons non-torch sont posés normalement.
echo "    → Cœur ML (transformers, accelerate, gliner…) en --no-deps (torch préservé)…"
# Compagnons SANS torch (sûrs en deps complètes)
# --ignore-installed sur typer : certaines images de pod embarquent un typer
# sans fichier RECORD, ce qui fait échouer toute tentative de mise à jour
# ("Cannot uninstall typer None: no RECORD file"). --ignore-installed pose la
# nouvelle version par-dessus sans désinstaller l'ancienne (bénin ici : typer
# est une petite lib CLI sans état partagé).
pip3 install $PIP_FLAGS --quiet --ignore-installed typer 2>&1 | grep -v "^WARNING\|^\[notice" || true
pip3 install $PIP_FLAGS --quiet \
    "huggingface_hub>=0.24" "tokenizers>=0.22.0,<=0.23.0" regex safetensors einops timm \
    "sse-starlette>=2.0" "rank-bm25>=0.2" ddgs \
    tiktoken sentencepiece "apscheduler>=3.10" pandas "structlog>=24.1" \
    pytest \
    networkx python-igraph python-louvain leidenalg \
    2>&1 | grep -v "^WARNING\|^\[notice" || true
# Paquets ML lourds : --no-deps STRICT (sinon ils tirent torch)
for spec in "transformers>=5.5.0" "accelerate>=0.30"; do
    pip3 install $PIP_FLAGS --no-deps --quiet "$spec" 2>&1 | grep -v "^WARNING\|^\[notice" || true
done
# GLiNER : extraction d'entités du Knowledge Graph. --no-deps STRICT
# (sinon il retélécharge torch et écrase la build CUDA figée). Ses deps
# non-torch (huggingface_hub, tokenizers, transformers) sont déjà posées
# ci-dessus. Sans gliner, le KG ne se remplit pas et la CLI logge
# "GLiNER indisponible" à répétition.
pip3 install $PIP_FLAGS --no-deps --quiet gliner 2>&1 | grep -v "^WARNING\|^\[notice" || true
if python3 -c "import gliner" 2>/dev/null; then
    ok "gliner installé (extraction d'entités KG active)"
else
    warn "gliner non importable — le KG restera vide (non bloquant). Détail :"
    python3 -c "import gliner" 2>&1 | sed 's/^/        /' | tail -3
fi
if python3 -c "import transformers" 2>/dev/null; then
    ok "transformers $(python3 -c 'import transformers;print(transformers.__version__)' 2>/dev/null) + cœur ML installés (run.py n'aura rien à faire au boot)"
else
    warn "transformers non importable à cette étape — cause réelle ci-dessous :"
    python3 -c "import transformers" 2>&1 | sed 's/^/        /' | tail -6
    info "→ en général sans gravité : run.py réinstalle les deps au boot et se ré-exécute (Protection 7)."
fi


# --no-deps que s'il y a UN seul paquet à poser ; en lot multiple il
# tirerait torch comme dépendance et écraserait la build CUDA figée.
# On force donc --no-deps + les compagnons non-torch (scikit-learn,
# scipy) dont l'import de sentence-transformers a besoin.
if python3 -c "import sentence_transformers" 2>/dev/null; then
    ok "sentence-transformers déjà importable (re-rank cross-encoder actif)"
else
    info "Installation de sentence-transformers (--no-deps, torch préservé)…"
    # Compagnons non-torch — n'affectent jamais la version de torch.
    pip3 install $PIP_FLAGS --quiet scikit-learn scipy 2>&1 | grep -v "^WARNING\|^\[notice" || true
    # Le paquet lui-même, SANS ses dépendances (torch reste celui de l'image).
    pip3 install $PIP_FLAGS --no-deps --quiet "sentence-transformers>=3.0" 2>&1 | grep -v "^WARNING\|^\[notice" || true
    if python3 -c "import sentence_transformers" 2>/dev/null; then
        ok "sentence-transformers installé (re-rank cross-encoder actif)"
        info "  Le modèle de re-rank se télécharge au 1ᵉʳ usage (BAAI/bge-reranker-v2-m3, ~600 Mo ;"
        info "  pour un modèle léger : export LYTHEA_CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2)."
    else
        warn "sentence-transformers non importable — re-rank cosine en repli (sans incidence). Détail :"
        python3 -c "import sentence_transformers" 2>&1 | sed 's/^/        /' | tail -4
    fi
fi

# ── bitsandbytes (voie 4-bit à la volée, modèles fp16) ───────────────
# --no-deps pour ne PAS toucher torch (piné par l'image). S'il casse, seuls
# les modèles bnb-4bit sont touchés (ils retombent en bf16).
if python3 -c "import bitsandbytes" 2>/dev/null; then
    ok "bitsandbytes déjà importable (chargement 4-bit NF4 disponible)"
else
    info "Installation de bitsandbytes (--no-deps, torch préservé)…"
    pip3 install $PIP_FLAGS --no-deps --quiet bitsandbytes 2>&1 | grep -v "^WARNING\|^\[notice" || true
    if python3 -c "import bitsandbytes" 2>/dev/null; then
        ok "bitsandbytes installé (modèles fp16 chargeables 4-bit sur 24 Go)"
    else
        warn "bitsandbytes non importable — modèles bnb-4bit en bf16. Détail :"
        python3 -c "import bitsandbytes" 2>&1 | sed 's/^/        /' | tail -4
    fi
fi

# GLiNER + transformers : laissés à run.py (déjà présents sur l'image CUDA).
info "Autres deps ML (gliner, transformers…) : vérifiées par run.py au boot."

echo ""

# ── Installation du package Rune (crée la commande `rune`) ────────────
# Sans ça, `rune chat` donne "command not found" : l'entry point défini
# dans pyproject.toml ([project.scripts] rune = ...) n'existe pas tant
# que le package n'est pas installé. --no-deps pour ne PAS toucher torch
# ni retélécharger les deps déjà posées ci-dessus.
echo "🔧 Installation de la commande 'rune' (pip install -e . --no-deps)…"
if pip3 install $PIP_FLAGS --no-deps --quiet -e . 2>&1 | grep -v "^WARNING\|^\[notice"; then
    :
fi
if command -v rune &>/dev/null; then
    ok "Commande 'rune' disponible ($(command -v rune))"
else
    warn "Commande 'rune' non trouvée dans le PATH après install."
    info "  Repli : utilise [cyan]python3 -m rune.cli <commande>[/] (ex: python3 -m rune.cli chat)"
fi

echo ""

# ── 5. Configuration .env ─────────────────────────────────────────────

echo "4️⃣  Configuration de l'environnement…"

if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        ok ".env créé à partir de .env.example"
    else
        warn ".env.example introuvable, création d'un .env minimal"
        echo "# Lythéa environment" > .env
    fi
else
    ok ".env existe déjà (préservé)"
fi

# Token d'auth
if ! grep -q "^LYTHEA_AUTH_TOKEN=." .env; then
    if command -v openssl &> /dev/null; then
        TOKEN=$(openssl rand -hex 32)
        grep -v "^#*\s*LYTHEA_AUTH_TOKEN=" .env > .env.tmp || true
        mv .env.tmp .env
        echo "" >> .env
        echo "# Token généré le $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> .env
        echo "LYTHEA_AUTH_TOKEN=$TOKEN" >> .env
        ok "LYTHEA_AUTH_TOKEN généré (32 octets)"
        info "Token : $TOKEN"
    else
        warn "openssl non disponible — génère ton token manuellement :"
        warn "  python3 -c 'import secrets; print(secrets.token_hex(32))'"
    fi
else
    ok "LYTHEA_AUTH_TOKEN déjà configuré"
fi

echo ""

# ── 6. Répertoires de données ─────────────────────────────────────────

echo "5️⃣  Préparation des répertoires…"
mkdir -p data/sessions data/chroma data/sdm data/mhn data/kg data/soft_memory
ok "data/{sessions,chroma,sdm,mhn,kg,soft_memory}/ prêts"
echo ""

# ── 7. Vérification structure ─────────────────────────────────────────

echo "6️⃣  Vérification de la structure…"

REQUIRED_FILES=(
    "rune/settings.py"
    "rune/boot.py"
    "rune/cache.py"
    "rune/temporal.py"
    "rune/microsleep.py"
    "rune/soft_memory.py"
    "rune/server/auth.py"
    "rune/server/rate_limit.py"
    "rune/server/app.py"
    "run.py"
)

MISSING=0
for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$f" ]; then
        err "Manquant : $f"
        MISSING=$((MISSING + 1))
    fi
done

if [ $MISSING -gt 0 ]; then
    err "$MISSING fichier(s) manquant(s)"
    exit 1
fi
ok "Tous les fichiers attendus sont présents"
echo ""

# ── 8. Test syntaxe ──────────────────────────────────────────────────

echo "7️⃣  Vérification syntaxe Python…"
SYNTAX_ERRORS=0
for f in $(find rune/ -name "*.py" -not -path "*/__pycache__/*"); do
    # encoding='utf-8' explicite : sur un pod dont la locale n'est pas
    # UTF-8, open() par défaut pourrait choisir ASCII et lever une
    # UnicodeDecodeError sur les accents français, faussement comptée
    # comme erreur de syntaxe.
    if ! python3 -c "import ast; ast.parse(open('$f', encoding='utf-8').read())" 2>/dev/null; then
        err "Erreur de syntaxe dans $f"
        SYNTAX_ERRORS=$((SYNTAX_ERRORS + 1))
    fi
done

if [ $SYNTAX_ERRORS -gt 0 ]; then
    err "$SYNTAX_ERRORS fichier(s) avec erreurs de syntaxe"
    exit 1
fi
ok "Tous les fichiers Python sont syntaxiquement valides"

# ── 9. Permissions exécutables sur les scripts shell ─────────────────

echo ""
echo "8️⃣  Permissions exécutables…"
for script in launch.sh test.sh deploy.sh searxng_bootstrap.sh; do
    if [ -f "$script" ]; then
        chmod +x "$script"
        ok "chmod +x $script"
    fi
done

# ── 10. Dépendance pyyaml (pour bootstrap SearXNG) ────────────────────

echo ""
echo "9️⃣  Vérification PyYAML (bootstrap SearXNG)…"
if python3 -c "import yaml" 2>/dev/null; then
    ok "PyYAML déjà installé"
else
    info "Installation de PyYAML…"
    pip install --break-system-packages --quiet pyyaml 2>/dev/null && ok "PyYAML installé" \
        || warn "Échec install PyYAML — SearXNG bootstrap pourrait échouer"
fi

echo ""
echo "🔟 Dépendances d'ingestion documentaire (ingest.py + upload UI)…"
# V6.0.0-rc rev9 — Liste exhaustive des paquets non-ML nécessaires.
# Tout est aussi déclaré dans pyproject.toml ; ce script est un
# filet de sécurité pour les setups qui n'ont pas fait pip install -e .
#
# Mapping pkg → module Python pour le check d'import :
#   - pdfplumber → pdfplumber       (extract_pdf)
#   - python-docx → docx            (extract_docx)
#   - openpyxl → openpyxl           (extract_xlsx, V6.0.0-rc rev5)
#   - python-multipart → multipart  (REQUIS par FastAPI UploadFile)
#   - beautifulsoup4 → bs4          (extract_html/xml propre)
#   - lxml → lxml                   (parser HTML/XML rapide pour bs4)
#   - networkx → networkx           (base graphe pour KG communities)
#   - python-igraph → igraph        (clustering Leiden)
#   - leidenalg → leidenalg         (algo Leiden)
#   - python-louvain → community    (fallback Louvain)
#   - httpx → httpx                 (clients HTTP externes)
#   - packaging → packaging         (version checks dans run.py)
declare -A INGEST_DEPS=(
    [pdfplumber]=pdfplumber
    [python-docx]=docx
    [openpyxl]=openpyxl
    [python-multipart]=multipart
    [beautifulsoup4]=bs4
    [lxml]=lxml
    [networkx]=networkx
    [python-igraph]=igraph
    [leidenalg]=leidenalg
    [python-louvain]=community
    [httpx]=httpx
    [packaging]=packaging
)

for pkg in "${!INGEST_DEPS[@]}"; do
    mod="${INGEST_DEPS[$pkg]}"
    if python3 -c "import $mod" 2>/dev/null; then
        ok "$pkg déjà installé"
    else
        info "Installation de $pkg…"
        pip install --break-system-packages --quiet "$pkg" 2>/dev/null \
            && ok "$pkg installé" \
            || warn "Échec install $pkg — fonctionnalité associée indisponible"
    fi
done

echo ""
echo "1️⃣1️⃣  Node.js (serveurs MCP : filesystem, GitHub, YouTube)…"
# Les outils MCP de l'agent sont lancés via `npx`. Sur un pod neuf, Node
# n'est pas présent → on l'installe automatiquement (NodeSource v20, sinon
# binaire officiel en repli). Non bloquant : sans Node, l'agent tourne quand
# même avec ses outils internes (write_file/run_tests…), seuls les MCP
# externes manquent.
NODE_MIN_MAJOR=18
node_major() { node --version 2>/dev/null | sed -E 's/^v([0-9]+).*/\1/'; }
CUR_NODE=$(node_major)
if [ -n "$CUR_NODE" ] && [ "$CUR_NODE" -ge "$NODE_MIN_MAJOR" ] 2>/dev/null; then
    ok "Node.js $(node --version) déjà présent"
else
    info "Node.js absent ou trop ancien — installation (v20)…"
    NODE_OK=""
    # Voie 1 : NodeSource + apt (Debian/Ubuntu)
    if command -v apt-get &>/dev/null; then
        if curl -fsSL https://deb.nodesource.com/setup_20.x 2>/dev/null | bash - &>/dev/null \
                && apt-get install -y nodejs &>/dev/null; then
            NODE_OK="1"
        fi
    fi
    # Voie 2 (repli) : binaire officiel décompressé dans /usr/local
    if [ -z "$NODE_OK" ]; then
        warn "apt indisponible/bloqué — repli binaire officiel…"
        NODE_VER="v20.18.0"
        ARCH=$(uname -m)
        case "$ARCH" in
            x86_64)  NODE_ARCH="linux-x64" ;;
            aarch64) NODE_ARCH="linux-arm64" ;;
            *)       NODE_ARCH="" ;;
        esac
        if [ -n "$NODE_ARCH" ]; then
            TARBALL="node-${NODE_VER}-${NODE_ARCH}.tar.xz"
            if curl -fsSL "https://nodejs.org/dist/${NODE_VER}/${TARBALL}" \
                    -o "/tmp/${TARBALL}" 2>/dev/null \
                    && tar -xf "/tmp/${TARBALL}" -C /tmp 2>/dev/null \
                    && cp -r /tmp/node-${NODE_VER}-${NODE_ARCH}/* /usr/local/ 2>/dev/null; then
                rm -f "/tmp/${TARBALL}"
                NODE_OK="1"
            fi
        fi
    fi
    if [ -n "$NODE_OK" ] && [ -n "$(node_major)" ]; then
        ok "Node.js $(node --version) installé"
    else
        warn "Node.js non installé — les serveurs MCP (filesystem/GitHub/"
        warn "YouTube) seront indisponibles. L'agent fonctionne sans eux."
    fi
fi

echo ""
echo "============================================================"
echo -e "  ${GREEN}✅ Déploiement terminé${NC}"
echo "============================================================"
echo ""
echo "Prochaines étapes :"
echo "  1. Lancer les tests       : bash test.sh"
echo "  2. Démarrer Lythéa        : bash launch.sh"
echo "  3. Ou démarrer directement: python3 run.py"
echo ""
echo "  📚 Spécialiser Lythéa avec tes documents :"
echo "     python3 ingest.py mes_documents/        # PDF, txt, md, docx"
echo "     python3 ingest.py --list                # voir ce qui est ingéré"
echo "     python3 ingest.py --help                # toutes les options"
echo ""

if [ $NEW_INSTALLS -gt 0 ] || [ $BASE_INSTALLS -gt 0 ]; then
    echo "  📦 $((NEW_INSTALLS + BASE_INSTALLS)) nouvelle(s) dépendance(s) installée(s)"
fi

if grep -q "^LYTHEA_AUTH_TOKEN=." .env; then
    TOKEN_FROM_ENV=$(grep "^LYTHEA_AUTH_TOKEN=" .env | cut -d= -f2)
    echo "  🔐 Token d'auth (à saisir au 1er accès web) :"
    echo "     $TOKEN_FROM_ENV"
fi

echo ""
