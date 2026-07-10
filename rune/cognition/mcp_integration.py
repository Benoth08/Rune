"""MCP cognitive integration — V6.0.0-rc.

When the router picks the ``mcp`` route, this module decides which
MCP tool to invoke (read_file, write_file, list_directory, ...) and
calls it via the MCPServerManager. The result is then injected into
the LLM context as a tool result.

Workflow
--------
1. Router picks "mcp" (from semantic_router or LLM dispatcher).
2. ``plan_mcp_call(query, workspace_files)`` analyses the query and
   the current workspace state, then returns a structured plan :
   ``{action, target, content}``. Where action ∈ {read, list, write,
   delete, rename} and target is a relative path inside sandbox.
3. ``execute_mcp_call(plan, manager, mcp_loop)`` actually runs the
   MCP tool and returns the result as a string ready to be appended
   to the user message context.

The planner uses an LLM call when ambiguous (e.g. "lis mon fichier"
without naming which), and falls back on heuristics for clear cases
("lis sales.csv"). For now we focus on **read** and **list** which
are the most useful immediate operations.

Write is delegated to a different mechanism : when the LLM generates
a substantial artefact in its response, the post-generation phase
extracts it and offers to write it to the workspace.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("rune.cognition.mcp_integration")


# ── Plan structures ────────────────────────────────────────────────────


@dataclass
class MCPPlan:
    """A structured plan for a single MCP call.

    Attributes
    ----------
    action : str
        ``"read"`` | ``"list"`` | ``"write"`` | ``"delete"`` | ``"rename"``
    target : str
        Relative path inside the sandbox. For ``list``, can be ``""``
        (workspace root).
    content : str | None
        For ``write`` only : the content to write. Ignored otherwise.
    reason : str
        Short human-readable explanation for logs/UI.
    """
    action: str
    target: str
    content: str | None = None
    reason: str = ""


# ── Heuristic planner ──────────────────────────────────────────────────


# Match an explicit filename in the query. We look for a single word
# (no internal spaces) ending with a recognised extension. We anchor
# on word boundary AND require a non-space character before the
# extension to avoid capturing "Lis sales.csv" as "Lis sales.csv".
_FILENAME_RE = re.compile(
    r"(?:^|[\s'\"\(\[])"             # début de chaîne ou délimiteur
    r"([A-Za-z0-9_\-]+\.(?:"           # juste un mot + extension
    r"csv|tsv|json|yaml|yml|toml|md|markdown|txt|log|py|js|ts|"
    r"html|css|xml|pdf|docx|xlsx|png|jpg|jpeg|webp|svg|"
    r"sh|sql|rs|go|java|cpp|c|h|hpp"
    r"))\b",
    re.IGNORECASE,
)

# Keywords that indicate a list/explore intent without naming any file
_LIST_KEYWORDS = (
    "liste les fichiers", "list files", "list workspace",
    "qu'est-ce qu'il y a dans", "what's in the workspace",
    "what files do i have", "quels fichiers", "quels documents",
    "contenu du workspace", "workspace content", "show my files",
    "montre-moi les fichiers", "show me the files",
)


def plan_mcp_call(
    query: str,
    workspace_files: list[str],
) -> MCPPlan:
    """Decide which MCP call to make based on the query.

    Parameters
    ----------
    query : str
        The user's question, lowercased internally.
    workspace_files : list[str]
        Names (basenames) of files currently in the workspace. Used
        to resolve implicit references ("le fichier que je viens
        d'ajouter" → most recent file).

    Returns
    -------
    MCPPlan
        With ``action`` ∈ {"read", "list"} for V6.0.0-rc. Write/delete/
        rename are deferred to direct UI actions (sidebar) for now.

    Heuristics (in priority order)
    ------------------------------
    1. Explicit filename in query → ``read``
    2. Listing keyword (« liste mes fichiers ») → ``list``
    3. Implicit reference (« le fichier que je viens d'ajouter ») +
       workspace has files → ``read`` on most recent
    4. Otherwise → ``list`` (safest default : show what's available)
    """
    q_lower = query.lower()

    # 1. Explicit filename match
    m = _FILENAME_RE.search(query)
    if m:
        filename = m.group(1)
        log.info(
            "MCP planner: filename détecté=%r ; workspace_files=%s",
            filename, workspace_files[:10],
        )
        # V6.0.0-rc rev4 : route intelligente selon le format.
        # - Formats texte brut (CSV, TXT, MD, JSON, code, ...) → MCP
        #   filesystem read_text_file (rapide).
        # - Formats binaires structurés (PDF, DOCX) → pipeline
        #   d'extraction (document_ingest.extract_uploaded_document)
        #   qui sait extraire le texte de ces formats.
        # - Formats vraiment binaires (images, archives, exécutables)
        #   → refus avec proposition d'autres mécanismes.
        ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
        # Formats avec extraction texte disponible (cf document_ingest.SUPPORTED_EXTENSIONS)
        STRUCTURED_BINARY_EXTS = {"pdf", "docx", "xlsx"}
        # Formats vraiment non lisibles
        UNREADABLE_EXTS = {"png", "jpg", "jpeg", "webp", "gif",
                           "zip", "tar", "gz", "7z", "rar",
                           "exe", "dll", "so", "dylib",
                           "mp3", "mp4", "wav", "avi", "mkv"}

        # Cherche le fichier (peu importe son format) avant tout
        match = _find_file_case_insensitive(filename, workspace_files)
        target = match or filename

        if ext in UNREADABLE_EXTS:
            return MCPPlan(
                action="unreadable",
                target=filename,
                reason=f"format non textuel {ext}",
            )
        if ext in STRUCTURED_BINARY_EXTS:
            log.info("MCP planner: format structuré %s → extraction texte", ext)
            return MCPPlan(
                action="read_extracted",
                target=target,
                reason=f"extraction texte depuis {ext}",
            )

        if match:
            log.info("MCP planner: match trouvé=%r", match)
            return MCPPlan(
                action="read",
                target=match,
                reason=f"filename explicite : {filename}",
            )
        log.info("MCP planner: AUCUN match — tentative directe %r", filename)
        # Filename mentioned but not found in workspace — still try
        # to read at that path (the MCP will return a clear error
        # that Lythéa can relay to the user).
        return MCPPlan(
            action="read",
            target=filename,
            reason=f"filename inconnu, tentative : {filename}",
        )

    # 2. Listing keyword
    if any(kw in q_lower for kw in _LIST_KEYWORDS):
        return MCPPlan(
            action="list",
            target="",
            reason="mot-clé listage",
        )

    # 3. Implicit reference + at least one file present
    implicit_refs = (
        "le fichier que",  "le document que",  "le csv que",
        "le pdf que",       "le rapport que",   "le fichier déposé",
        "que je viens d",   "que je t'ai donné", "que j'ai uploadé",
        "que j'ai mis dans", "the file i",       "the document i",
        "my file",          "my dataset",        "my csv",
    )
    if any(ref in q_lower for ref in implicit_refs) and workspace_files:
        # Pick the workspace file with the most recent mtime — caller
        # gave us names sorted by recency (most recent first).
        return MCPPlan(
            action="read",
            target=workspace_files[0],
            reason=f"référence implicite → fichier le plus récent : {workspace_files[0]}",
        )

    # 4. Fallback : list the workspace so the user can see what's there
    return MCPPlan(
        action="list",
        target="",
        reason="ambigu, listage par défaut",
    )


def _find_file_case_insensitive(name: str, files: list[str]) -> str | None:
    """Find a file in the workspace, case-insensitive on basename."""
    name_lower = name.lower()
    for f in files:
        if Path(f).name.lower() == name_lower:
            return f
    # Partial match (filename without extension)
    name_stem = Path(name_lower).stem
    for f in files:
        if Path(f).name.lower().startswith(name_stem):
            return f
    return None


# ── Executor ───────────────────────────────────────────────────────────


def execute_mcp_call(
    plan: MCPPlan,
    mcp_manager: Any,  # MCPServerManager
    mcp_loop: asyncio.AbstractEventLoop,
    timeout: float = 10.0,
) -> tuple[bool, str]:
    """Run the MCP call described by ``plan`` and return its result.

    Parameters
    ----------
    plan : MCPPlan
        Built by :func:`plan_mcp_call`.
    mcp_manager : MCPServerManager
        The shared manager instance attached to the FastAPI app.
    mcp_loop : asyncio.AbstractEventLoop
        The dedicated MCP event loop (where the MCP clients live).
    timeout : float
        Hard cap per call. MCP filesystem ops are usually <100ms.

    Returns
    -------
    tuple[bool, str]
        ``(ok, content)`` where content is either the file content
        (read), the directory listing (list), or an error message.
    """
    # Map MCPPlan.action → (server, tool, arguments)
    action_map = {
        "read": ("filesystem", "read_text_file"),
        "list": ("filesystem", "list_directory"),
    }

    # V6.0.0-rc rev4 : lecture de PDF/DOCX via document_ingest. On lit
    # les bytes via MCP filesystem (read_media_file pour les binaires
    # ou Python direct), puis on extrait le texte avec les libs
    # spécialisées (pdfplumber, mammoth, etc.) déjà installées.
    if plan.action == "read_extracted":
        return _execute_read_extracted(plan)

    # V6.0.0-rc rev4 : refus pour les formats vraiment non lisibles
    # (images, archives, exécutables). On suggère le 📎 ou clic-droit.
    if plan.action == "unreadable":
        ext = plan.target.lower().rsplit(".", 1)[-1] if "." in plan.target else "?"
        return False, (
            f"Le fichier '{plan.target}' est un format non textuel ({ext}). "
            f"Je peux le lire si :\n"
            f"  - C'est une image → joins-la au chat via 📎, je verrai le contenu visuellement.\n"
            f"  - C'est une archive ou un exécutable → désolée, je ne sais pas l'inspecter."
        )

    if plan.action not in action_map:
        return False, f"Action MCP non supportée pour l'instant : {plan.action}"

    server, tool = action_map[plan.action]

    # Resolve absolute path inside the sandbox. The MCP filesystem
    # server is scoped to the sandbox root, so we need to pass an
    # absolute path that's inside that root.
    from rune.settings import get_settings
    s = get_settings()
    sandbox_root = (
        getattr(s, "mcp_sandbox_dir", "")
        or str(Path.home() / ".lythea" / "sandbox")
    )

    if plan.action == "list":
        target_abs = sandbox_root if not plan.target else f"{sandbox_root}/{plan.target}"
        arguments = {"path": target_abs}
    else:  # read
        target_abs = f"{sandbox_root}/{plan.target}"
        arguments = {"path": target_abs}

    log.info("MCP call : %s.%s(%s)", server, tool, arguments)

    # Schedule on the MCP loop and wait
    try:
        future = asyncio.run_coroutine_threadsafe(
            mcp_manager.call_tool(server, tool, arguments, timeout=timeout),
            mcp_loop,
        )
        result = future.result(timeout=timeout + 2.0)
    except Exception as exc:
        log.warning("MCP call crashed: %s", exc)
        return False, f"L'appel MCP a échoué : {exc}"

    # MCP result shape : {"content": [{"type": "text", "text": "..."}], "isError": bool}
    if not isinstance(result, dict):
        return False, f"Réponse MCP malformée : {result!r}"

    if result.get("isError"):
        # Filesystem error : file not found, access denied, etc.
        content_blocks = result.get("content", [])
        msg = content_blocks[0].get("text", "Erreur inconnue") if content_blocks else "Erreur inconnue"
        return False, msg

    # Extract the text content
    content_blocks = result.get("content", [])
    if not content_blocks:
        return True, ""  # Empty result (e.g. empty file or empty dir)
    text = content_blocks[0].get("text", "")
    return True, text


def _execute_read_extracted(plan: MCPPlan) -> tuple[bool, str]:
    """V6.0.0-rc rev4 — Lit un fichier binaire structuré (PDF, DOCX) en
    passant par le pipeline d'extraction texte de ``document_ingest``.

    On lit les bytes directement depuis le disque (le sandbox est local)
    plutôt que via MCP filesystem.read_media_file → décodage base64,
    parce que c'est plus simple et qu'on est déjà process-local.

    Returns
    -------
    tuple[bool, str]
        ``(ok, text)``. Texte extrait du document, ou message d'erreur.
    """
    try:
        from pathlib import Path
        from rune.settings import get_settings
        s = get_settings()
        sandbox_root = (
            getattr(s, "mcp_sandbox_dir", "")
            or str(Path.home() / ".lythea" / "sandbox")
        )
        abs_path = Path(sandbox_root) / plan.target

        # Sécurité : confirmer que le chemin résolu est bien dans le sandbox
        try:
            abs_path.resolve().relative_to(Path(sandbox_root).resolve())
        except ValueError:
            return False, f"Chemin hors sandbox : {plan.target!r}"

        if not abs_path.exists():
            return False, (
                f"ENOENT : le fichier '{plan.target}' n'existe pas dans "
                f"le workspace."
            )
        if abs_path.is_dir():
            return False, f"'{plan.target}' est un dossier, pas un fichier."

        # Lecture binaire + extraction texte
        file_bytes = abs_path.read_bytes()
        log.info(
            "Read_extracted : %s (%d bytes)", abs_path.name, len(file_bytes),
        )

        try:
            from rune.server.document_ingest import (
                extract_uploaded_document, is_supported,
            )
        except ImportError as exc:
            return False, f"Pipeline d'extraction indisponible : {exc}"

        if not is_supported(abs_path.name):
            return False, (
                f"Le format de '{plan.target}' n'est pas supporté par le "
                f"pipeline d'extraction texte."
            )

        try:
            text, n_chars = extract_uploaded_document(file_bytes, abs_path.name)
        except Exception as exc:
            log.exception("Extraction texte échouée pour %s", abs_path.name)
            return False, f"Extraction texte échouée : {exc}"

        if not text.strip():
            return True, (
                f"[Fichier '{plan.target}' lu, mais aucun texte n'a pu être "
                f"extrait. Le fichier est peut-être vide, scanné sans OCR, "
                f"ou mal formé.]"
            )

        log.info(
            "Read_extracted : %s → %d chars extraits",
            abs_path.name, n_chars,
        )
        return True, text
    except Exception as exc:
        log.exception("_execute_read_extracted crashed")
        return False, f"Erreur interne : {exc}"


def format_mcp_result_for_context(plan: MCPPlan, ok: bool, content: str) -> str:
    """Format the MCP result as text that will be injected in the LLM prompt.

    Returned text is appended to the user's message before generation,
    framed clearly so the LLM knows it's MCP output (not user text).
    """
    if not ok:
        # V6.0.0-rc rev3 : cas spécial binary_not_supported — message
        # V6.0.0-rc rev4 : cas spécial "unreadable" (image, archive,
        # exécutable) — content explique déjà à l'utilisateur quoi faire.
        if plan.action == "unreadable":
            return (
                f"\n\n[Information système pour ta réponse finale]\n"
                f"L'utilisateur a demandé à lire un fichier que je ne peux "
                f"pas traiter (image, archive ou exécutable). Voici le "
                f"message à transmettre :\n\n{content}\n\n"
                f"[Reformule en français de façon naturelle et brève. "
                f"Ne mentionne pas le terme 'MCP'.]"
            )
        # Anti-rétro-compat : si quelqu'un envoie encore binary_not_supported
        if plan.action == "binary_not_supported":
            return (
                f"\n\n[Information système]\n{content}\n\n"
                f"[Reformule pour l'utilisateur. Réponds en français.]"
            )
        return (
            f"\n\n[Appel MCP échoué — action={plan.action}, target={plan.target!r}]\n"
            f"Erreur : {content}\n\n"
            f"[NOTE SYSTÈME IMPÉRATIVE pour ta réponse finale]\n"
            f"Le fichier '{plan.target}' n'a PAS pu être lu (il n'existe "
            f"probablement pas dans le workspace, ou son nom est différent). "
            f"Tu DOIS le dire CLAIREMENT à l'utilisateur :\n"
            f"  1. NE FAIS PAS semblant d'avoir lu le fichier.\n"
            f"  2. N'INVENTE PAS son contenu (colonnes, valeurs, structure).\n"
            f"  3. NE DÉCRIS PAS de Date/Valeur1/Valeur2/Catégorie ou autre "
            f"contenu imaginaire.\n"
            f"  4. Réponse attendue : explique brièvement l'erreur, propose "
            f"à l'utilisateur de vérifier le nom du fichier dans le workspace "
            f"(sidebar droite), ou d'uploader le fichier s'il n'y est pas. "
            f"Tu peux aussi proposer de lister les fichiers présents.\n"
            f"Cette note est invisible pour l'utilisateur. Réponds en français."
        )

    if plan.action == "list":
        return (
            f"\n\n[Contenu du workspace — `{plan.target or 'racine'}`]\n"
            f"{content}\n"
            f"[Fin du listing]"
        )

    if plan.action in ("read", "read_extracted"):
        # Truncate very large file contents to avoid context blow-up.
        # The LLM can ask for more details if needed (V6.0.x : chunked read).
        max_chars = 12000
        if len(content) > max_chars:
            truncated_note = (
                f"\n\n[Note : fichier tronqué à {max_chars} caractères sur "
                f"{len(content)} au total. Lythéa peut demander la suite "
                f"si nécessaire.]"
            )
            content = content[:max_chars] + truncated_note

        source_label = (
            "lu via extraction texte (PDF/DOCX/XLSX)"
            if plan.action == "read_extracted"
            else "lu via MCP filesystem"
        )
        # V6.0.0-rc rev7 : conseils de format quand on a un contenu
        # tabulaire (Excel, CSV) avec beaucoup de lignes/colonnes.
        # Sans ça, le LLM essaie de tout dumper en tableau markdown
        # qui (a) explose la limite de tokens, (b) est illisible.
        is_tabular_heavy = (
            content.count("\t") > 50  # XLSX rendu en TSV
            or content.count(",") > 100  # CSV
        ) and len(content) > 1500
        format_advice = ""
        if is_tabular_heavy:
            format_advice = (
                "\n\nCONSEIL DE FORMAT : le contenu est volumineux et "
                "tabulaire. NE recopie PAS le tableau brut dans ta réponse "
                "(ça dépasse la limite de tokens et c'est illisible). À la "
                "place :\n"
                "  - Décris la structure (nombre de feuilles, colonnes, "
                "lignes).\n"
                "  - Calcule mentalement quelques agrégats utiles "
                "(moyennes, totaux, min/max).\n"
                "  - Identifie les variations notables (un mois "
                "différent des autres, une rupture de tendance).\n"
                "  - Réponds en texte conversationnel structuré, pas en "
                "tableau exhaustif.\n"
                "  - Si l'utilisateur demande un rapport markdown, "
                "produis un rapport SYNTHÉTIQUE (~500-1000 mots) avec "
                "agrégats et observations clés, pas un dump des données."
            )
        return (
            f"\n\n[Contenu du fichier — `{plan.target}` ({source_label})]\n"
            f"{content}\n"
            f"[Fin du contenu]\n\n"
            f"[NOTE SYSTÈME pour ta réponse finale]\n"
            f"Le contenu ci-dessus est le VRAI contenu du fichier. Réponds en "
            f"te basant UNIQUEMENT sur ce contenu réel — n'invente pas de "
            f"colonnes, valeurs ou structure qui ne seraient pas présentes. "
            f"Si le contenu est court ou vide, dis-le honnêtement à "
            f"l'utilisateur. Si tu n'es pas sûr, propose de re-lire ou de "
            f"vérifier. Cette note est invisible pour l'utilisateur."
            f"{format_advice}"
        )

    return f"\n\n[Résultat MCP — {plan.action}]\n{content}"


# ── Artefact detection (Phase 2 — Lythéa writes back) ─────────────────


@dataclass
class DetectedArtefact:
    """An artefact extracted from Lythéa's response, ready to write.

    Lythéa's responses sometimes contain substantial blocks of content
    that the user would benefit from having as a standalone file :
    Python scripts, CSV outputs, long markdown documents, etc. This
    captures those and tags them for workspace writing.

    Attributes
    ----------
    content : str
        The raw content (code or markdown).
    suggested_filename : str
        A reasonable filename, including extension.
    kind : str
        ``"python"`` | ``"markdown"`` | ``"json"`` | ``"csv"`` | ``"text"``
    """
    content: str
    suggested_filename: str
    kind: str


# Python fenced code block, captures the body
_PYTHON_BLOCK_RE = re.compile(
    r"```(?:python|py)\n(.*?)\n```",
    re.DOTALL | re.IGNORECASE,
)

# Generic code block with explicit language
_CODE_BLOCK_RE = re.compile(
    r"```(?P<lang>[a-zA-Z0-9_+-]+)\n(?P<body>.*?)\n```",
    re.DOTALL,
)


# Minimum sizes : below these, the block isn't worth a separate file
MIN_PYTHON_LINES = 10
# V6.0.0-rc rev6 — Seuil markdown remonté de 800 à 2500 chars. Avant,
# une réponse conversationnelle bien structurée (avec quelques titres ##)
# déclenchait l'écriture d'un faux "rapport.md" dans le workspace.
# 2500 chars correspond à ~400 mots — clairement au-delà d'une réponse
# normale, vraiment un livrable.
MIN_MARKDOWN_CHARS = 2500
# Nombre minimum de titres ##/### pour considérer comme document structuré
MIN_MARKDOWN_HEADERS = 4
# Mots-clés qui indiquent une intention EXPLICITE de Lythéa de générer
# un livrable (et non une simple réponse). Si présents, on baisse le
# seuil de longueur pour permettre la détection d'un livrable court.
_EXPLICIT_DOCUMENT_HINTS = (
    "voici le rapport",
    "voici le document",
    "j'ai préparé le rapport",
    "j'ai rédigé le document",
    "je sauvegarde dans",
    "je l'écris dans",
    "je crée le fichier",
    "here is the report",
    "i'll save this as",
)


# ── User intent for artefact writing ───────────────────────────────────
# V6.0.0-rc rev6 — On n'écrit dans le workspace que si l'utilisateur le
# demande explicitement. Ces patterns matchent la query utilisateur, pas
# la réponse de Lythéa. Sans cette intention claire, même un long
# markdown structuré ne crée plus de rapport.md (évite le faux positif
# "résumé d'Excel" qu'on a observé en rev5).

_USER_WRITE_REQUEST_PATTERNS = (
    # Français — verbes d'écriture/création
    "écris", "écrit", "écrire", "redige", "rédige", "rédiger",
    "génère ", "génere ", "generes ", "génère-moi", "genere-moi",
    "génère un", "génère le", "génère-moi un", "génère-moi le",
    "génère-moi le rapport", "génère-moi un rapport",
    "crée ", "créer", "crée un", "crée le", "crée-moi", "creer un",
    "crée-moi un", "crée-moi le",
    "fais ", "fais-moi", "fais un", "fais-moi un",
    "donne-moi un fichier", "donne-moi le fichier",
    "rédige-moi", "écris-moi",
    # Français — destinations explicites
    "sauvegarde", "sauve dans", "stocke dans", "enregistre dans",
    "exporte", "exporter",
    "mets-le dans un fichier", "mets ça dans un fichier",
    "dans un fichier", "dans un .py", "dans un .md", "dans un .csv",
    "comme un fichier", "sous forme de fichier",
    # English
    "write a", "write me a", "write me the", "write the",
    "save it to", "save this as", "save to a file", "save as",
    "create a", "create me a", "create the",
    "generate a", "generate me a", "generate the",
    "export to", "output to file", "make me a",
)


def _user_wants_file(user_intent: str) -> bool:
    """Did the user explicitly ask for a deliverable file ?

    Returns True if the query contains any of the patterns above.
    Case-insensitive match.
    """
    if not user_intent:
        return False
    q = user_intent.lower()
    return any(p in q for p in _USER_WRITE_REQUEST_PATTERNS)


def detect_artefacts(
    response_text: str, user_intent: str = "",
) -> list[DetectedArtefact]:
    """Scan an LLM response for substantial artefacts worth saving.

    V6.0.0-rc rev6 — La règle a changé : on ne crée des fichiers QUE si
    l'utilisateur l'a explicitement demandé dans sa query, à une
    exception près : un script Python complet (≥10 lignes) est
    toujours sauvé car il est clairement un livrable.

    Le markdown long n'est PLUS auto-détecté comme "rapport" sans
    demande explicite. Sinon Lythéa créait des rapport.md à chaque
    résumé structuré (bug rev5 sur les résumés Excel).

    Parameters
    ----------
    response_text : str
        Le texte de la réponse de Lythéa.
    user_intent : str
        La query originale de l'utilisateur (utilisée comme filtre
        d'intention pour les artefacts non-Python).

    Returns at most 3 artefacts per response to avoid spam.
    """
    artefacts: list[DetectedArtefact] = []
    user_asked_for_file = _user_wants_file(user_intent)

    # ── Python blocks ──
    # Toujours détectés : un script Python ≥10 lignes EST un livrable,
    # même sans demande explicite (la conversation contient rarement
    # 10+ lignes de Python par hasard).
    for i, m in enumerate(_PYTHON_BLOCK_RE.finditer(response_text)):
        if len(artefacts) >= 3:
            break
        body = m.group(1).strip()
        n_lines = body.count("\n") + 1
        if n_lines < MIN_PYTHON_LINES:
            continue
        suggested = _guess_python_filename(body, idx=i)
        artefacts.append(DetectedArtefact(
            content=body,
            suggested_filename=suggested,
            kind="python",
        ))

    # ── CSV / JSON blocks ──
    # Seulement si l'utilisateur a demandé un fichier (sinon on garde
    # le code formaté dans la réponse, pas dans un .csv séparé).
    if user_asked_for_file:
        for m in _CODE_BLOCK_RE.finditer(response_text):
            if len(artefacts) >= 3:
                break
            lang = m.group("lang").lower()
            body = m.group("body").strip()
            if lang == "csv" and "," in body and "\n" in body:
                artefacts.append(DetectedArtefact(
                    content=body, suggested_filename="data.csv", kind="csv",
                ))
            elif lang == "json" and len(body) > 100:
                artefacts.append(DetectedArtefact(
                    content=body, suggested_filename="data.json", kind="json",
                ))

    # ── Long markdown ──
    # SUPPRIMÉ en rev6 sans demande explicite. Seulement si l'utilisateur
    # a clairement demandé un rapport/document.
    if user_asked_for_file and len(artefacts) < 3:
        text_no_blocks = _CODE_BLOCK_RE.sub("", response_text).strip()
        if len(text_no_blocks) >= 200:  # seuil minimal pour éviter du vide
            artefacts.append(DetectedArtefact(
                content=response_text.strip(),
                suggested_filename="rapport.md",
                kind="markdown",
            ))

    return artefacts


def _guess_python_filename(code: str, idx: int = 0) -> str:
    """Pick a reasonable .py filename from a code block.

    Looks for the first ``def`` or ``class`` definition. Falls back to
    ``script_N.py`` when nothing identifies.
    """
    m = re.search(r"^(?:def|class)\s+([a-zA-Z_][a-zA-Z0-9_]*)", code, re.MULTILINE)
    if m:
        name = m.group(1)
        # Strip leading underscores for clean filename
        name = name.lstrip("_") or name
        return f"{name}.py"
    # Look for a docstring/comment that names something
    m = re.search(r'^"""(.+?)"""', code, re.DOTALL)
    if m:
        first_line = m.group(1).strip().split("\n")[0]
        # Take first 3 words, sanitize
        words = re.findall(r"[a-zA-Z0-9]+", first_line.lower())[:3]
        if words:
            return f"{'_'.join(words)}.py"
    return f"script_{idx + 1}.py" if idx else "script.py"


def write_artefact_to_workspace(
    artefact: DetectedArtefact,
    workspace_manager: Any,
) -> dict | None:
    """Write a detected artefact to the workspace via WorkspaceManager.

    Returns the FileEntry dict (suitable for ``workspace_file_offer``
    SSE event) on success, or ``None`` on failure.
    """
    try:
        # Use the manager's save_upload which handles dedup + atomic write
        entry = workspace_manager.save_upload(
            filename=artefact.suggested_filename,
            data=artefact.content.encode("utf-8"),
            target_dir="",
        )
        log.info(
            "Workspace write : %s (kind=%s, %d chars)",
            entry.name, artefact.kind, len(artefact.content),
        )
        return entry.to_dict()
    except Exception as exc:
        log.warning("Workspace write failed for %s : %s", artefact.suggested_filename, exc)
        return None
