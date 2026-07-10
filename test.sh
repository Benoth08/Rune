#!/bin/bash
# Lythéa — Script de validation
#
# Usage : bash test.sh

set -e

# Aller dans le répertoire du script
cd "$(cd "$(dirname "$(readlink -f "$0" 2>/dev/null || echo "$0")")" && pwd)"

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
echo "  Rune — Tests"
echo "============================================================"
echo ""

# ── 1. Syntaxe ────────────────────────────────────────────────────────

echo "1️⃣  Vérification syntaxe…"
ERRORS=0
for f in $(find rune/ tests/ -name "*.py" -not -path "*/__pycache__/*"); do
    if ! python3 -c "import ast; ast.parse(open('$f', encoding='utf-8').read())" 2>/dev/null; then
        err "$f"
        ERRORS=$((ERRORS + 1))
    fi
done
if [ $ERRORS -eq 0 ]; then
    ok "Tous les fichiers OK"
else
    err "$ERRORS erreur(s) de syntaxe"
    exit 1
fi

if command -v node &> /dev/null; then
    : # JS check skipped — Rune est headless (pas de static/app.js)
fi
echo ""

# ── 2. Import des modules principaux ──────────────────────────────────

echo "2️⃣  Vérification des imports principaux…"

PYTHON_IMPORT_TEST="
import sys
sys.path.insert(0, '.')
errors = []
for mod in [
    'rune.settings',
    'rune.boot',
    'rune.cache',
    'rune.temporal',
    'rune.microsleep',
    'rune.soft_memory',
    'rune.git_sync',
    'rune.sessions',
    'rune.server.auth',
    'rune.server.rate_limit',
]:
    try:
        __import__(mod)
        print(f'  \u2713 {mod}')
    except ImportError as e:
        errors.append((mod, str(e)))
        print(f'  \u2717 {mod}: {e}')
sys.exit(len(errors))
"

if python3 -c "$PYTHON_IMPORT_TEST"; then
    ok "Tous les modules s'importent"
else
    warn "Certains modules ne s'importent pas — vérifier les dépendances"
    warn "  Si erreur 'fastapi' : pip install fastapi uvicorn"
    warn "  Si erreur 'chromadb' : pip install chromadb"
fi
echo ""

# ── 3. pytest ─────────────────────────────────────────────────────────

echo "3️⃣  Lancement de la suite de tests…"

PIP_FLAGS=""
if pip3 install --dry-run --quiet pytest 2>&1 | grep -q "externally-managed-environment"; then
    PIP_FLAGS="--break-system-packages"
fi

if ! python3 -m pytest --version &> /dev/null 2>&1; then
    info "Installation de pytest…"
    pip3 install $PIP_FLAGS --quiet pytest pytest-asyncio 2>&1 | grep -v "^WARNING\|^\[notice" || true
fi

# Détection torch et chromadb
HAS_TORCH=0
HAS_CHROMA=0
if python3 -c "import torch" 2>/dev/null; then HAS_TORCH=1; fi
if python3 -c "import chromadb" 2>/dev/null; then HAS_CHROMA=1; fi

if [ $HAS_TORCH -eq 1 ]; then
    info "torch disponible"
fi
if [ $HAS_CHROMA -eq 1 ]; then
    info "chromadb disponible"
fi

# Construire la liste de tests à exécuter selon les modules présents
TESTS_TO_RUN=()
ALWAYS_TESTS=(
    "tests/test_settings.py"
    "tests/test_boot.py"
    "tests/test_cache.py"
    "tests/test_kg.py"
    "tests/test_kg_improved.py"
    "tests/test_api.py"
    "tests/test_auth.py"
    "tests/test_config_endpoints.py"
    "tests/test_codegen.py"
    "tests/test_agentic.py"
    "tests/test_sandbox.py"
    "tests/test_tools.py"
    "tests/test_steering.py"
    "tests/test_sampling.py"
    "tests/test_prompt_coherence.py"
    "tests/test_git_sync.py"
    "tests/test_sessions_concurrency.py"
    "tests/test_temporal.py"
    "tests/test_loadability.py"
    "tests/test_cross_encoder.py"
    "tests/test_microsleep.py"
    "tests/test_soft_memory.py"
    "tests/test_gemini_client.py"
    "tests/test_cascade.py"
    "tests/test_cascade_integration.py"
    # ── V4 cognitive modules (pure Python, no torch needed) ────────────
    "tests/test_settings_v4.py"
    "tests/test_cognitive_state.py"
    "tests/test_inhibition.py"
    "tests/test_planning.py"
    "tests/test_predictive_coding.py"
    "tests/test_timeline.py"
    "tests/test_microsleep_v41.py"
    "tests/test_metacognition.py"
    "tests/test_v4_integration.py"
    "tests/test_v4_routes.py"
)
for t in "${ALWAYS_TESTS[@]}"; do
    [ -f "$t" ] && TESTS_TO_RUN+=("$t")
done

if [ $HAS_TORCH -eq 1 ] && [ $HAS_CHROMA -eq 1 ]; then
    # Ces tests dépendent de torch ET chromadb
    [ -f tests/test_hippocampe.py ] && TESTS_TO_RUN+=("tests/test_hippocampe.py")
    [ -f tests/test_hippocampe_v4.py ] && TESTS_TO_RUN+=("tests/test_hippocampe_v4.py")
fi
if [ $HAS_TORCH -eq 1 ]; then
    [ -f tests/test_mhn.py ] && TESTS_TO_RUN+=("tests/test_mhn.py")
    [ -f tests/test_salience.py ] && TESTS_TO_RUN+=("tests/test_salience.py")
    [ -f tests/test_sdm.py ] && TESTS_TO_RUN+=("tests/test_sdm.py")

    # Cognition phase tests (étape 8 refactor — need torch).
    # These were missing from the whitelist in earlier versions, which
    # silently skipped 121 tests when running ``bash test.sh``.
    for t in tests/test_encoding.py tests/test_storage.py \
             tests/test_surprise.py tests/test_retrieval_phase.py \
             tests/test_consolidation.py tests/test_generation.py; do
        [ -f "$t" ] && TESTS_TO_RUN+=("$t")
    done
fi

echo ""
# Les warnings tiers (Swig à l'import via extensions C igraph/leidenalg,
# httpx/starlette dans le TestClient, resume_download de huggingface_hub)
# sont émis soit AVANT que pytest n'installe ses filtres ini (import très
# tôt → filtre raté + dédupliqué), soit à l'ARRÊT de l'interpréteur (hors
# capture pytest). On les neutralise donc au niveau process via
# PYTHONWARNINGS, posé au démarrage de l'interpréteur, AVANT tout import.
# Filtres ciblés (pas d'ignore global) — une vraie régression resterait
# visible. Note : entrées séparées par des virgules, champs par des deux-
# points ; quotes simples obligatoires pour ne pas exécuter les backticks.
export PYTHONWARNINGS='ignore:builtin type:DeprecationWarning,ignore:Using `httpx` with:,ignore:The `resume_download` argument is deprecated:UserWarning'
# En plus de PYTHONWARNINGS (pré-import + arrêt), on passe les mêmes filtres
# en -W directement à pytest : il les applique dans SON contexte de capture,
# ce qui couvre le cas où pytest réinitialise les filtres au démarrage de
# session. Quotes simples pour ne pas exécuter les backticks.
python3 -m pytest "${TESTS_TO_RUN[@]}" --tb=short \
    -W 'ignore:builtin type:DeprecationWarning' \
    -W 'ignore:Using `httpx` with:' \
    -W 'ignore:The `resume_download` argument is deprecated:UserWarning' \
    2>&1 | tail -30
PYTEST_EXIT=${PIPESTATUS[0]}
echo ""

if [ $PYTEST_EXIT -eq 0 ]; then
    ok "Tests OK"
else
    err "Tests en échec — code $PYTEST_EXIT"
fi
echo ""

# ── 4. Statistiques ───────────────────────────────────────────────────

echo "4️⃣  Statistiques…"
# Count both top-level functions AND indented methods inside classes.
# The previous grep missed all class-based tests (~70% of V3.9 tests).
TOTAL_TESTS=$(grep -rhE "^(def test_|    def test_)" tests/*.py 2>/dev/null | wc -l)
PY_FILES=$(find rune/ -name "*.py" -not -path "*/__pycache__/*" | wc -l)
PY_LINES=$(find rune/ -name "*.py" -not -path "*/__pycache__/*" -exec cat {} + | wc -l)
ok "Fichiers Python   : $PY_FILES"
ok "Lignes de code    : $PY_LINES"
ok "Tests définis     : $TOTAL_TESTS"

echo ""
echo "============================================================"
if [ $PYTEST_EXIT -eq 0 ]; then
    echo -e "  ${GREEN}✅ Validation réussie${NC}"
    echo "============================================================"
    echo ""
    echo "Rune est prêt à être lancée :"
    echo "  bash launch.sh        # avec tunnel Cloudflare"
    echo "  python3 run.py        # local seulement"
    echo ""
    exit 0
else
    echo -e "  ${RED}❌ Validation en échec${NC}"
    echo "============================================================"
    echo ""
    echo "Pour plus de détails : python3 -m pytest tests/ -v"
    echo ""
    exit 1
fi
