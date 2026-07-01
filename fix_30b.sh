#!/bin/bash
# fix_30b.sh — Débloque le chargement du Qwen3-30B-A3B en 4-bit sur le pod.
#
# Le bug : transformers 5.x (core_model_loading.py, issue #43032) matérialise
# les poids en PLEINE PRÉCISION sur le GPU avant de quantiser → un 30B tente
# d'allouer ~22 GB et OOM sur 24 GB, alors qu'en 4-bit il ne fait que ~18 GB.
#
# Ce script applique la correction DÉTERMINISTE : downgrade de transformers
# vers la dernière 4.x (qui n'a pas le bug), tout en préservant torch et
# bitsandbytes. Le downgrade est vérifié (échoue bruyamment s'il ne prend pas).
#
# ⚠️ Effet de bord : les modèles Gemma 4 exigent transformers>=5.5 et ne
#    chargeront plus en 4.57. C'est un compromis assumé pour débloquer le 30B
#    MAINTENANT. Pour revenir à la 5.x : pip install "transformers>=5.5".
#
# Usage : bash fix_30b.sh

set -e
GREEN="\033[0;32m"; YELLOW="\033[1;33m"; RED="\033[0;31m"; NC="\033[0m"
ok()   { echo -e "${GREEN}  ✓${NC} $1"; }
warn() { echo -e "${YELLOW}  ⚠${NC} $1"; }
err()  { echo -e "${RED}  ✗${NC} $1"; }

echo ""
echo "============================================================"
echo "  Fix Qwen3-30B-A3B — contournement bug transformers v5"
echo "============================================================"
echo ""

CUR=$(python3 -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "absent")
echo "  Version transformers actuelle : $CUR"

case "$CUR" in
    4.*)
        ok "Déjà en 4.x — pas de bug v5. Rien à faire."
        echo "  Tu peux charger le 30B directement."
        exit 0
        ;;
esac

echo ""
warn "transformers $CUR a le bug v5 (matérialisation pleine précision)."
echo "  → Downgrade vers 4.57.1 (préserve torch + bitsandbytes)…"
echo ""

# --no-deps pour NE PAS toucher à torch ; on force la version.
pip3 install --break-system-packages --no-deps --force-reinstall \
    "transformers==4.57.1" 2>&1 | tail -5

echo ""
NEW=$(python3 -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "absent")
if [[ "$NEW" == 4.57.* ]]; then
    ok "transformers $NEW installé — le bug v5 est contourné."
    echo ""
    # Vérif que torch et bitsandbytes répondent toujours
    python3 -c "import torch, bitsandbytes" 2>/dev/null \
        && ok "torch + bitsandbytes toujours fonctionnels" \
        || warn "vérifie torch/bitsandbytes (import KO)"
    echo ""
    echo "  Prochaine étape :"
    echo "    1. redémarre le serveur : pkill -9 -f run.py; sleep 3; bash launch.sh"
    echo "    2. captioner off + charge Qwen/Qwen3-30B-A3B-Thinking-2507"
    echo "    → le log doit montrer un chargement à ~18 GB, plus d'OOM à 22."
else
    err "Le downgrade n'a pas pris (version actuelle : $NEW)."
    err "Cause probable : un autre paquet a ré-imposé la 5.x, ou conflit pip."
    echo "  Essaie manuellement :"
    echo "    pip3 install --break-system-packages --force-reinstall 'transformers==4.57.1'"
    echo "    python3 -c 'import transformers; print(transformers.__version__)'"
    exit 1
fi
