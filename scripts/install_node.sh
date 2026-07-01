#!/usr/bin/env bash
# install_node.sh — Installe Node.js via fnm (Fast Node Manager).
#
# fnm est portable Linux/macOS, sans sudo, et ne pollue pas le système.
# Il est requis pour les serveurs MCP de Lythéa (filesystem, GitHub,
# YouTube) qui sont distribués comme packages npm.
#
# Usage :
#   bash scripts/install_node.sh        # installe Node 20 LTS
#   bash scripts/install_node.sh 22     # installe Node 22

set -e

NODE_VERSION="${1:-20}"

echo "🚀 Installation de Node.js $NODE_VERSION via fnm..."
echo

# Détection
if command -v node &> /dev/null; then
    CURRENT_VERSION=$(node --version)
    echo "ℹ️  Node.js déjà installé : $CURRENT_VERSION"
    read -p "Réinstaller / changer de version ? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Abandon."
        exit 0
    fi
fi

# Install fnm si absent
if ! command -v fnm &> /dev/null; then
    echo "📥 Installation de fnm..."
    curl -fsSL https://fnm.vercel.app/install | bash

    # Source le shell config pour avoir fnm dans le PATH
    # On essaye plusieurs fichiers selon le shell
    for rc in ~/.bashrc ~/.zshrc ~/.profile; do
        if [ -f "$rc" ]; then
            # shellcheck disable=SC1090
            source "$rc" 2>/dev/null || true
        fi
    done

    # Si toujours pas trouvé, on l'ajoute manuellement au PATH
    export PATH="$HOME/.local/share/fnm:$PATH"
    eval "$(fnm env)" 2>/dev/null || true
fi

# Install Node
echo
echo "📦 Installation de Node.js $NODE_VERSION..."
fnm install "$NODE_VERSION"
fnm use "$NODE_VERSION"
fnm default "$NODE_VERSION"

# Vérification
echo
echo "✅ Installation terminée"
node --version
npm --version

echo
echo "ℹ️  Pour les nouvelles sessions terminal, fnm doit être chargé."
echo "    Si 'node' n'est pas trouvé après reconnexion SSH :"
echo "    echo 'eval \"\$(fnm env)\"' >> ~/.bashrc"
echo
echo "🌙 Lythéa peut maintenant démarrer ses serveurs MCP."
