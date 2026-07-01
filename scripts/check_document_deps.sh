#!/usr/bin/env bash
# check_document_deps.sh — Vérifie et installe les déps Python pour
# l'ingestion mémoire de documents (PDF, DOCX) dans Lythéa.
#
# Usage :
#   bash check_document_deps.sh         # diagnostic seul
#   bash check_document_deps.sh --install   # installe les manquantes
#
# Ce que ça checke :
#   - pdfplumber (lecture PDF, dépendance "extract_pdf")
#   - python-docx (lecture DOCX, dépendance "extract_docx")
#
# Sans ces libs, l'ingestion fonctionne pour .txt/.md/.csv/.json mais
# pas pour PDF/Word. Si tu veux nourrir la mémoire de Lythéa avec des
# papiers scientifiques, des rapports, ou des docs Word, il faut les
# avoir.

set -e

INSTALL=false
if [[ "$1" == "--install" ]]; then
    INSTALL=true
fi

echo "🔍 Diagnostic des dépendances d'ingestion de documents..."
echo

check_lib() {
    local lib_name="$1"
    local pip_name="$2"
    local for_what="$3"

    if python3 -c "import $lib_name" 2>/dev/null; then
        local version
        version=$(python3 -c "import $lib_name; print(getattr($lib_name, '__version__', 'unknown'))" 2>/dev/null || echo "?")
        echo "  ✅ $lib_name ($version) — OK — $for_what"
        return 0
    else
        echo "  ❌ $lib_name MANQUANT — $for_what"
        if [[ "$INSTALL" == "true" ]]; then
            echo "     → pip install $pip_name"
            pip install "$pip_name" --break-system-packages --quiet 2>/dev/null \
                || pip install "$pip_name" --user --quiet 2>/dev/null \
                || pip install "$pip_name" --quiet
            echo "     ✅ $pip_name installé"
        else
            echo "     → installer avec : bash $0 --install"
        fi
        return 1
    fi
}

check_lib "pdfplumber" "pdfplumber" "Ingestion mémoire des PDF"
check_lib "docx" "python-docx" "Ingestion mémoire des .docx (Word)"

echo
echo "ℹ️  Pour ingérer un document dans la mémoire long-terme de Lythéa :"
echo "    1. Ouvre l'UI Lythéa dans ton navigateur"
echo "    2. Clique sur le bouton d'upload de document (📎 ou similaire)"
echo "    3. Sélectionne ton PDF / DOCX / TXT"
echo "    4. Bascule le badge sur '📚 En mémoire' avant d'envoyer"
echo "    5. Lythéa l'ingérera dans ChromaDB + extraira les entités dans le KG"
echo
echo "   Le document sera ensuite retrievable dans toutes tes futures"
echo "   conversations via le RAG, et ses entités principales seront"
echo "   connues par Lythéa."
echo
