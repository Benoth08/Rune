#!/bin/bash
# Lythéa — SearXNG self-hosted bootstrap
# Usage: bash searxng_bootstrap.sh [PORT]
#
# Idempotent : safe to re-run. Detects what's already in place and
# only does what's missing.
#
# Outputs to stdout:
#   SEARXNG_URL=http://127.0.0.1:<port>      (on success)
#   SEARXNG_PID=<pid>                         (on success)
# Sets exit code:
#   0  → SearXNG ready and responding
#   1  → installation or boot failure
#
# All progress logging goes to stderr so the parent script can capture
# the URL via stdout.

set -e

PORT=${1:-8080}
LOG_FILE="${SEARXNG_LOG_FILE:-/tmp/lythea_searxng.log}"
SRC_DIR="${SEARXNG_SRC_DIR:-/workspace/searxng-src}"
CONFIG_DIR="${SEARXNG_CONFIG_DIR:-/workspace/searxng-config}"
SETTINGS_FILE="$CONFIG_DIR/settings.yml"

log() { echo -e "  🔍 $*" >&2; }
warn() { echo -e "  ⚠️  $*" >&2; }
err() { echo -e "  ❌ $*" >&2; }

# ── 1. Vérifier si SearXNG est déjà importable ──────────────────────────
if ! python3 -c "from searx import webapp" 2>/dev/null; then
    log "SearXNG pas encore installé"

    # Clone si nécessaire — ou RE-clone si un dossier incomplet traîne
    # (clone précédent interrompu sur un pod neuf → requirements.txt absent,
    #  ce qui faisait échouer l'install ensuite).
    if [ ! -f "$SRC_DIR/requirements.txt" ] && [ ! -f "$SRC_DIR/pyproject.toml" ]; then
        if [ -d "$SRC_DIR" ]; then
            warn "Dossier SearXNG incomplet — nettoyage et re-clone…"
            rm -rf "$SRC_DIR"
        fi
        log "Clone du dépôt SearXNG..."
        if ! git clone --depth 1 https://github.com/searxng/searxng.git "$SRC_DIR" >&2; then
            err "Échec du clone SearXNG (réseau ?) — fallback DDG"
            exit 1
        fi
    fi

    # Install requirements + paquet en editable. SearXNG récent peut n'avoir
    # qu'un pyproject.toml (plus de requirements.txt) → on gère les deux.
    log "Installation des dépendances Python..."
    if [ -f "$SRC_DIR/requirements.txt" ]; then
        pip install --break-system-packages --quiet --ignore-installed blinker \
            -r "$SRC_DIR/requirements.txt" >&2 || {
                err "Échec install requirements"
                exit 1
            }
    else
        log "Pas de requirements.txt — install via pyproject (editable)…"
    fi

    log "Installation de SearXNG..."
    pip install --break-system-packages --quiet --no-build-isolation -e "$SRC_DIR" >&2 || {
            err "Échec install SearXNG"
            exit 1
        }
fi

# ── 2. Générer settings.yml si absent ───────────────────────────────────
if [ ! -f "$SETTINGS_FILE" ]; then
    log "Génération de settings.yml..."
    mkdir -p "$CONFIG_DIR"
    SRC_SETTINGS="$(python3 -c 'import searx, os; print(os.path.join(os.path.dirname(searx.__file__), "settings.yml"))')"
    if [ ! -f "$SRC_SETTINGS" ]; then
        # Fallback : chercher dans le clone
        SRC_SETTINGS="$SRC_DIR/searx/settings.yml"
    fi
    if [ ! -f "$SRC_SETTINGS" ]; then
        err "settings.yml de référence introuvable"
        exit 1
    fi

    SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_hex(32))')
    python3 << PYEOF
import yaml, sys
with open("$SRC_SETTINGS") as f:
    cfg = yaml.safe_load(f)

# ── Server patches Lythéa ─────────────────────────────────────────
cfg['server']['secret_key'] = "$SECRET_KEY"
cfg['server']['bind_address'] = '127.0.0.1'
cfg['server']['port'] = $PORT
cfg['server']['base_url'] = 'http://127.0.0.1:$PORT/'
cfg['server']['limiter'] = False
cfg['server']['public_instance'] = False
cfg['server']['image_proxy'] = False

# Bot detection relaxée (Lythéa UA n'est pas Mozilla par défaut)
cfg['botdetection'] = {
    'ip_limit': {
        'filter_link_local': False,
        'link_token': False,
    },
}

# JSON format obligatoire pour l'API
if 'json' not in cfg['search']['formats']:
    cfg['search']['formats'].append('json')

# ── Outgoing tuning anti-rate-limit ───────────────────────────────
# Augmenter timeouts pour laisser le temps aux engines lents
# (Mojeek, Brave) sans précipiter Google.
if 'outgoing' not in cfg:
    cfg['outgoing'] = {}
cfg['outgoing']['request_timeout'] = 8.0
cfg['outgoing']['max_request_timeout'] = 15.0
cfg['outgoing']['pool_connections'] = 100
cfg['outgoing']['pool_maxsize'] = 20
cfg['outgoing']['enable_http2'] = True
# Suspend penalty plus court (default = 60s, on baisse à 30s)
# pour qu'un engine ban temporaire ne bloque pas 3 minutes.
cfg['outgoing']['retries'] = 1

# ── Engines : favoriser ceux qui ne rate-limit pas ───────────────
# Google rate-limite vite les IPs serveur. On désactive Google et on
# privilégie : Brave (généreux), DuckDuckGo, Mojeek, Wikipedia, Qwant,
# Wikidata. C'est largement suffisant pour la couverture, et bien plus
# stable depuis un VPS / pod RunPod.
ENGINES_PRIORITY = {
    # Engines à GARDER actifs (résistants au rate-limit)
    'brave', 'duckduckgo', 'mojeek', 'qwant',
    'wikipedia', 'wikidata', 'startpage',
    'bing',  # supporte mieux les serveurs que Google
    'searx',  # méta-search interne
    # Engines à DÉSACTIVER (rate-limit agressif)
    # google, google_news, google_scholar : bannissent IPs serveur en quelques requêtes
}
ENGINES_TO_DISABLE = {
    'google', 'google news', 'google scholar', 'google images',
    'google videos', 'google play movies', 'google play apps',
}

# Parcourir les engines et ajuster
disabled_count = 0
enabled_count = 0
for engine in cfg.get('engines', []):
    name = engine.get('name', '').lower()
    # Désactiver Google explicitement
    if name in ENGINES_TO_DISABLE or name.startswith('google'):
        engine['disabled'] = True
        disabled_count += 1
    # Activer les engines de la priority list
    elif name in ENGINES_PRIORITY and engine.get('disabled', False):
        engine['disabled'] = False
        enabled_count += 1

print(f"Engines: {disabled_count} désactivés (Google/famille), "
      f"{enabled_count} réactivés (anti-rate-limit)", file=sys.stderr)

with open("$SETTINGS_FILE", 'w') as f:
    yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
print("settings.yml généré", file=sys.stderr)
PYEOF
else
    log "settings.yml déjà présent"
fi

# ── 3. Tuer ancienne instance + démarrer SearXNG ────────────────────────
pkill -f "searx.webapp" 2>/dev/null || true
sleep 2

log "Démarrage de SearXNG sur le port $PORT..."
SEARXNG_SETTINGS_PATH="$SETTINGS_FILE" nohup python3 -m searx.webapp \
    > "$LOG_FILE" 2>&1 &
SEARXNG_PID=$!

# ── 4. Attendre que SearXNG réponde sur /healthz ────────────────────────
log "Attente du démarrage..."
READY=0
for i in $(seq 1 30); do
    if curl -s -m 2 "http://127.0.0.1:$PORT/healthz" > /dev/null 2>&1 \
            || curl -s -m 2 "http://127.0.0.1:$PORT/" > /dev/null 2>&1; then
        READY=1
        break
    fi
    sleep 1
done

if [ "$READY" -ne 1 ]; then
    err "SearXNG n'a pas répondu après 30s"
    err "Voir le log : $LOG_FILE"
    tail -20 "$LOG_FILE" >&2
    exit 1
fi

# ── 5. Test d'une vraie requête ─────────────────────────────────────────
log "Test d'une recherche..."
TEST_OUT=$(curl -s -m 15 "http://127.0.0.1:$PORT/search?q=python&format=json&categories=general" 2>&1)
N_RESULTS=$(echo "$TEST_OUT" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(len(d.get("results",[])))' 2>/dev/null || echo "0")

if [ "$N_RESULTS" -lt 1 ]; then
    err "SearXNG répond mais ne retourne aucun résultat"
    err "Voir le log : $LOG_FILE"
    exit 1
fi

log "SearXNG OK ($N_RESULTS résultats sur 'python', PID $SEARXNG_PID)"

# Sortie machine-readable pour le script parent
echo "SEARXNG_URL=http://127.0.0.1:$PORT"
echo "SEARXNG_PID=$SEARXNG_PID"
exit 0
