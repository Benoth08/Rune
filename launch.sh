#!/bin/bash
# Lythéa — Launch with Cloudflare tunnel and full preload
# Usage: bash launch.sh [PORT]
#
# Démarre dans l'ordre :
#   1. SearXNG self-hosted (port 8080) — bootstrap automatique
#   2. Lythéa (port $PORT, default 7860)
#   3. Tunnel Cloudflare
#
# SearXNG est démarré idempotemment via ``searxng_bootstrap.sh`` qui :
#   - Installe SearXNG si absent
#   - Génère un settings.yml stable + secret_key
#   - Lance le daemon en arrière-plan
#   - Vérifie qu'il répond aux requêtes JSON
# Si SearXNG échoue, Lythéa démarre quand même mais retombe sur DDG via
# la chaîne CompositeProvider — pas de blocage.

set -e

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0" 2>/dev/null || echo "$0")")" && pwd)"
cd "$SCRIPT_DIR"

PORT=${1:-7860}
SEARXNG_PORT=${SEARXNG_PORT:-8080}

# ── Install cloudflared if needed ──────────────────────────────────────
if ! command -v cloudflared &>/dev/null; then
    echo "📦 Installing cloudflared..."
    wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
        -O /usr/local/bin/cloudflared
    chmod +x /usr/local/bin/cloudflared
fi

# ── Kill previous instances ────────────────────────────────────────────
pkill -f "python.*run.py" 2>/dev/null || true
pkill -f cloudflared 2>/dev/null || true
sleep 1

# ── Bootstrap SearXNG (idempotent) ─────────────────────────────────────
echo "🔍 Initialisation de SearXNG (port $SEARXNG_PORT)..."
SEARXNG_RESULT=""
SEARXNG_URL=""
SEARXNG_PID=""
if [ -x "$SCRIPT_DIR/searxng_bootstrap.sh" ]; then
    if SEARXNG_RESULT=$(bash "$SCRIPT_DIR/searxng_bootstrap.sh" "$SEARXNG_PORT"); then
        SEARXNG_URL=$(echo "$SEARXNG_RESULT" | grep '^SEARXNG_URL=' | cut -d= -f2-)
        SEARXNG_PID=$(echo "$SEARXNG_RESULT" | grep '^SEARXNG_PID=' | cut -d= -f2)
        if [ -n "$SEARXNG_URL" ]; then
            echo "  ✅ SearXNG actif : $SEARXNG_URL (PID $SEARXNG_PID)"
            export LYTHEA_SEARXNG_INSTANCE_URL="$SEARXNG_URL"
            # On ne force PAS le provider sur "searxng" : on garde le mode
            # composite (auto) qui essaie SearXNG en premier PUIS retombe
            # sur DDG si les requêtes SearXNG échouent (rate-limit Google,
            # instances bloquées…). Forcer "searxng" seul supprimerait ce
            # fallback — c'est ce qui donnait "All SearXNG instances failed"
            # sans réponse. Le mode auto est le défaut, donc rien à exporter.
            export LYTHEA_WEB_PROVIDER="auto"
        fi
    else
        echo "  ⚠️  SearXNG bootstrap a échoué — fallback sur DDG."
    fi
else
    echo "  ⚠️  searxng_bootstrap.sh introuvable — fallback DDG."
fi

# ── Start Lythéa ───────────────────────────────────────────────────────
LYTHEA_LOG=$(mktemp)
echo "🚀 Démarrage de Lythéa sur le port $PORT..."
python3 run.py --port "$PORT" 2>&1 | tee "$LYTHEA_LOG" &
LYTHEA_PID=$!

# ── Wait for HTTP server liveness ──────────────────────────────────────
echo "⏳ Attente du serveur HTTP…"
for i in $(seq 1 30); do
    if curl -s "http://localhost:$PORT/api/boot/status" > /dev/null 2>&1; then
        break
    fi
    sleep 1
done

# ── Start Cloudflare tunnel in parallel ────────────────────────────────
echo "🌐 Création du tunnel Cloudflare…"
TUNNEL_LOG=$(mktemp)
cloudflared tunnel --url "http://localhost:$PORT" > "$TUNNEL_LOG" 2>&1 &
TUNNEL_PID=$!

for i in $(seq 1 15); do
    URL=$(grep -o 'https://[^ ]*trycloudflare.com' "$TUNNEL_LOG" 2>/dev/null | head -1)
    if [ -n "$URL" ]; then break; fi
    sleep 1
done

# ── Poll boot/status until ready ───────────────────────────────────────
echo "⏳ Préchargement des modèles auxiliaires…"
echo ""
LAST_STAGE=""
LAST_PCT=-1
# Render a 30-char progress bar : [██████████░░░░░░░░░░░░░░░░░░░░]  33%
render_bar() {
    local pct="$1"
    local label="$2"
    local width=30
    local filled=$(( pct * width / 100 ))
    [ "$filled" -gt "$width" ] && filled="$width"
    local empty=$(( width - filled ))
    local bar=""
    for ((i=0; i<filled; i++)); do bar="${bar}█"; done
    for ((i=0; i<empty; i++)); do bar="${bar}░"; done
    # \r returns cursor to start of line; trailing spaces clear old text
    printf "\r  📦 [%s] %3d%%  %-40s" "$bar" "$pct" "$label"
}

for i in $(seq 1 600); do
    STATUS=$(curl -s "http://localhost:$PORT/api/boot/status" 2>/dev/null || echo "{}")
    READY=$(echo "$STATUS" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('ready', False))" 2>/dev/null || echo "False")
    STAGE=$(echo "$STATUS" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('current_step', ''))" 2>/dev/null || echo "")
    PCT=$(echo "$STATUS" | python3 -c "import sys, json; d=json.load(sys.stdin); print(int(d.get('progress_pct', 0)))" 2>/dev/null || echo "0")

    if [ -n "$STAGE" ]; then
        LABEL=$(echo "$STATUS" | python3 -c "
import sys, json
d = json.load(sys.stdin)
labels = d.get('stage_labels', {})
key = d.get('current_step', '').replace('loading_', '')
print(labels.get(key, key))
" 2>/dev/null || echo "$STAGE")

        # New stage: newline + redraw
        if [ "$STAGE" != "$LAST_STAGE" ]; then
            [ -n "$LAST_STAGE" ] && echo ""  # close previous bar with newline
            LAST_STAGE="$STAGE"
            LAST_PCT=-1
        fi

        # Refresh bar only if % changed (avoid flicker / log spam)
        if [ "$PCT" != "$LAST_PCT" ]; then
            render_bar "$PCT" "$LABEL"
            LAST_PCT="$PCT"
        fi
    fi

    if [ "$READY" = "True" ]; then
        # Finish current line with 100% snapshot
        [ -n "$LAST_STAGE" ] && render_bar 100 "Prêt"
        echo ""
        break
    fi
    sleep 1
done

ELAPSED=$(curl -s "http://localhost:$PORT/api/boot/status" 2>/dev/null | python3 -c "import sys, json; print(json.load(sys.stdin).get('elapsed_s', '?'))" 2>/dev/null)

# ── Display URLs ───────────────────────────────────────────────────────
echo ""
echo "🌟 Rune est prêt ! (preload total: ${ELAPSED}s)"
if [ -n "$URL" ]; then
    echo "🔗 Tunnel : $URL"
else
    echo "⚠️  Tunnel non trouvé — voir : $TUNNEL_LOG"
fi
echo "🔗 Local  : http://localhost:$PORT"
if [ -n "$SEARXNG_URL" ]; then
    echo "🔍 Search : $SEARXNG_URL (SearXNG self-hosted)"
fi
echo ""
echo "Ctrl+C pour arrêter"
echo ""

# Forward signals — kill children including SearXNG
cleanup() {
    kill "$LYTHEA_PID" "$TUNNEL_PID" 2>/dev/null || true
    if [ -n "$SEARXNG_PID" ]; then
        kill "$SEARXNG_PID" 2>/dev/null || true
    fi
    pkill -f "searx.webapp" 2>/dev/null || true
    exit 0
}
trap cleanup INT TERM
wait $LYTHEA_PID
