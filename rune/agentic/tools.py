"""Bounded, model-driven tool-calling for the agent — the "Hermès loop".

The model may emit ``<tool_call>{json}</tool_call>``; the orchestrator
intercepts it, runs ONE vetted, *parameterized* operation, and feeds back a
structured ``<tool_response>``. There is deliberately **no open ``python`` /
shell tool**: the model chooses *which safe tool* and *its arguments*, never a
raw command line. Dependency installs stay reactive inside ``run_tests`` (never
a tool the model can spam).

This module is pure: the orchestrator injects the concrete operations
(``ops``) as callables bound to the mission's sandbox + workspace, so file
paths are jailed and writes go through the same idempotent path as the rest.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex

log = logging.getLogger(__name__)

# Accepte DEUX balises de tool-call JSON :
#  • <tool_call> : format rune, natif des modèles thinking (Qwen3, R1…)
#  • <tools>     : format que les Qwen-Coder suivent à ~100% avec un exemple
#                  few-shot (les Coder IGNORENT rune — cf. issue vLLM #32926).
# Le parseur étant partagé, supporter les deux couvre TOUS les modèles.
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>|<tools>(.*?)</tools>",
                           re.S)

# Format MARKDOWN ENRICHI (anti-échappement) : le code est pris BRUT entre les
# balises, SANS passer par un parsing JSON (qui casse sur les guillemets,
# backslashes et sauts de ligne du code). Parade racine aux tool_calls de code
# malformés des petits modèles. path accepté en attribut, code brut au milieu.
_MD_WRITE_RE = re.compile(
    r"<write_file\s+path\s*=\s*[\"']?([^\"'>\n]+)[\"']?\s*>\n?(.*?)</write_file>",
    re.S)
_MD_EDIT_RE = re.compile(
    r"<edit_file\s+path\s*=\s*[\"']?([^\"'>\n]+)[\"']?\s*>\n?"
    r"<<<FIND\n?(.*?)\n?>>>REPLACE\n?(.*?)\n?</edit_file>",
    re.S)


def _strip_code_fence(s: str) -> str:
    """Retire un fence ```python … ``` que le modèle aurait ajouté autour du
    code dans une balise markdown (fréquent, inoffensif à enlever)."""
    s = re.sub(r"^\s*```[a-zA-Z0-9_+-]*\n", "", s)
    s = re.sub(r"\n```\s*$", "", s)
    return s


# Fence de code NOMMÉE : les modèles de code (Qwen-Coder…) « affichent »
# naturellement le code dans un bloc ```python dont la 1re ligne est un
# commentaire « # nom.py ». On exploite CE réflexe : un tel bloc est traité
# comme un write_file de ce fichier. Couvre les modèles qui n'adoptent pas
# spontanément les balises <write_file>. Le langage du fence est libre
# (python, py, ou absent) ; seul le commentaire « # <fichier>.<ext> » compte.
# Fence de code générique : tout bloc ```lang … ``` (le nom de fichier est
# résolu par _extract_named_fences via une cascade, pas exigé dans le fence).
_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)```", re.S)
# Nom de fichier en 1re ligne de commentaire DANS le bloc (« # nom.py »).
_FENCE_INNER_NAME_RE = re.compile(r"^\s*[#/]+\s*([\w./-]+\.\w+)\s*\n")
# Nom de fichier mentionné dans le TEXTE précédant le fence : entre backticks
# (`calculatrice.py`) ou après module/fichier/file. On lit les ~200 derniers
# caractères avant le ```.
_NAME_BACKTICK_RE = re.compile(r"`([\w./-]+\.\w+)`")
_NAME_WORD_RE = re.compile(
    r"(?:module|fichier|file|fichier\s+de\s+tests?|script)\s+[`\"']?"
    r"([\w./-]+\.\w+)", re.I)


def _infer_name_from_content(code: str, used: set) -> str | None:
    """Devine le nom de fichier depuis le CODE quand il n'est indiqué nulle
    part : un bloc qui IMPORTE un module local (``from X import``) est son
    fichier de tests (``test_X.py``) ; un bloc qui contient des ``def test_``
    est un test ; sinon, si une def « principale » existe, on nomme d'après le
    1er module importé localement par les autres blocs (résolu par l'appelant)."""
    m = re.search(r"^\s*from\s+([\w.]+)\s+import\b", code, re.M)
    if m:
        mod = m.group(1).split(".")[-1]
        cand = f"test_{mod}.py"
        if cand not in used:
            return cand
    if re.search(r"^\s*def\s+test_", code, re.M):
        # test sans import explicite : nom générique non collidant
        i = 1
        while f"test_module_{i}.py" in used:
            i += 1
        return f"test_module_{i}.py"
    return None


def _extract_named_fences(text: str) -> list[dict]:
    """Cascade pour récupérer les fichiers que les modèles de code « affichent »
    dans des blocs ```python sans appeler l'outil. Pour CHAQUE fence, le nom est
    résolu par : (a) commentaire « # nom.py » en 1re ligne du bloc, sinon
    (b) nom de fichier mentionné dans le texte juste AVANT le fence (backticks /
    « module X »), sinon (c) inférence depuis le contenu (un bloc qui importe un
    module local → test_<module>.py). Les blocs non nommables sont ignorés.

    ⚠️ Un fence ``json``/``yaml`` ou dont le contenu est un tool_call JSON
    (« {"name": ..., "arguments": ...} ») n'est PAS un fichier de code : on le
    SAUTE (il sera traité par le parsing JSON), sinon on écrirait le tool_call
    brut comme contenu de fichier."""
    out: list[dict] = []
    used: set = set()
    spans = []
    for m in _FENCE_RE.finditer(text):
        # langue du fence (```python, ```json…) pour filtrer les non-code
        lang_m = re.match(r"```([a-zA-Z0-9_+-]*)\n", text[m.start():])
        lang = (lang_m.group(1).lower() if lang_m else "")
        if lang in ("json", "yaml", "yml"):
            continue
        code = m.group(1)
        # un contenu qui « ressemble » à un tool_call JSON → pas un fichier
        _stripped = code.strip()
        if (_stripped.startswith("{") and '"name"' in _stripped
                and '"arguments"' in _stripped):
            continue
        before = text[max(0, m.start() - 200):m.start()]
        name = None
        mi = _FENCE_INNER_NAME_RE.match(code)
        if mi:
            name = mi.group(1)
            code = code[mi.end():]  # retire la ligne de commentaire-nom
        if not name:
            cands = _NAME_BACKTICK_RE.findall(before) or \
                _NAME_WORD_RE.findall(before)
            if cands:
                name = cands[-1]
        spans.append([name, code])
    for sp in spans:
        if sp[0] is None:
            sp[0] = _infer_name_from_content(sp[1], used)
        if sp[0]:
            used.add(sp[0])
    for name, code in spans:
        if name and code.strip():
            out.append({"name": "write_file",
                        "arguments": {"path": name.strip(), "content": code}})
    return out


_MAX_CONTENT = 60_000  # cap on write_file content (chars)

# Executables the agent may run via run_command/serve_and_probe. The
# orchestrator owns this allowlist — the model only picks from it, never a raw
# shell. ``None`` = any sub-command; a set = only those sub-commands.
_CMD_ALLOW: dict[str, set[str] | None] = {
    "python": None, "python3": None, "pytest": None,
    "pip": {"install", "list", "show", "freeze"},
    "pip3": {"install", "list", "show", "freeze"},
    "node": None, "npx": None,
    "npm": {"install", "ci", "run", "test", "build", "exec", "version"},
    "pnpm": {"install", "run", "test", "build"},
    "ruff": None, "black": None, "isort": None, "prettier": None,
    "eslint": None, "tsc": None,
    # Server launchers (for serve_and_probe). They bind a local port; the
    # sandbox tears them down after probing.
    "uvicorn": None, "gunicorn": None, "hypercorn": None, "daphne": None,
    "flask": None, "streamlit": None,
}
_NET_SUB = {"install", "ci", "add"}


def _first_subcommand(argv: list[str]) -> str | None:
    for tok in argv[1:]:
        if not tok.startswith("-"):
            return tok
    return None


def validate_command(argv: list[str]) -> tuple[bool, str]:
    """Allow only whitelisted executables (+ sub-commands). Never a shell."""
    if not argv:
        return False, "commande vide"
    exe = os.path.basename(argv[0])
    if exe not in _CMD_ALLOW:
        return False, f"exécutable non autorisé: {exe!r}"
    allowed = _CMD_ALLOW[exe]
    if allowed is not None:
        sub = _first_subcommand(argv)
        if sub not in allowed:
            return False, f"sous-commande non autorisée pour {exe}: {sub!r}"
    return True, ""


def command_needs_net(argv: list[str]) -> bool:
    if not argv:
        return False
    if os.path.basename(argv[0]) == "npx":
        return True
    return any(tok in _NET_SUB for tok in argv)

# The menu injected (as JSON) into the system prompt.
TOOLS_SCHEMA = [
    {"name": "list_files", "description": "Liste les fichiers de la mission.",
     "parameters": {}},
    {"name": "read_file", "description": "Lit un fichier de la mission.",
     "parameters": {"path": "chemin relatif dans la mission"}},
    {"name": "write_file",
     "description": "Crée ou remplace un fichier de la mission (contenu complet).",
     "parameters": {"path": "chemin relatif", "content": "contenu complet du fichier"}},
    {"name": "run_tests",
     "description": ("Lance pytest dans la mission (installe automatiquement les "
                     "dépendances manquantes) et renvoie le résultat réel."),
     "parameters": {}},
    {"name": "run_command",
     "description": ("Exécute UNE commande autorisée dans la mission "
                     "(python, pytest, pip install, node, npm install/run/build/test, "
                     "ruff, prettier…). Pour builder/installer/linter."),
     "parameters": {"command": "ligne de commande, ex. 'npm install' ou 'npm run build'"}},
    {"name": "edit_file",
     "description": "Remplace une portion EXACTE d'un fichier (find → replace), sans tout réécrire.",
     "parameters": {"path": "chemin relatif", "find": "texte exact à remplacer",
                    "replace": "nouveau texte"}},
    {"name": "serve_and_probe",
     "description": ("Démarre un serveur (ex. 'uvicorn app:app --port 8000' ou "
                     "'npm run dev'), teste des endpoints HTTP en local puis l'arrête."),
     "parameters": {"command": "commande de démarrage du serveur",
                    "paths": "liste de chemins à tester, ex. ['/', '/health']",
                    "port": "port du serveur (défaut 8000)"}},
    {"name": "delete_file",
     "description": "Supprime un fichier de la mission.",
     "parameters": {"path": "chemin relatif dans la mission"}},
    {"name": "search_files",
     "description": ("Cherche un motif (texte ou regex) dans les fichiers de la "
                     "mission et renvoie les lignes correspondantes (fichier:ligne)."),
     "parameters": {"query": "motif à chercher",
                    "glob": "filtre optionnel, ex. '*.py' (défaut : tous)"}},
    {"name": "run_python",
     "description": ("Exécute un court extrait de code Python dans la mission et "
                     "renvoie sa sortie (stdout/stderr). Pratique pour un calcul "
                     "rapide sans créer de fichier."),
     "parameters": {"code": "code Python à exécuter"}},
    {"name": "web_search",
     "description": ("Recherche sur le web (mêmes fournisseurs que le chat) et "
                     "renvoie une liste {title, url, extrait}. Pour toute info "
                     "externe ou à jour que tu n'as pas."),
     "parameters": {"query": "requête de recherche",
                    "max_results": "nombre de résultats, défaut 5"}},
    {"name": "web_fetch",
     "description": ("Récupère le contenu texte d'une URL (http/https), tronqué. "
                     "À utiliser après web_search pour lire une page précise."),
     "parameters": {"url": "URL complète à lire"}},
    {"name": "recall",
     "description": ("Cherche dans la mémoire de Lythéa : souvenirs de "
                     "conversations/missions passées et connaissances déjà "
                     "acquises. À utiliser si un élément de contexte personnel "
                     "ou une info déjà rencontrée pourrait éclairer la tâche."),
     "parameters": {"query": "ce que tu cherches à retrouver"}},
    {"name": "query_kg",
     "description": ("Interroge le graphe de connaissances (faits structurés) "
                     "sur une entité ou une question précise. Renvoie les faits "
                     "connus, ou rien s'il n'y en a pas."),
     "parameters": {"question": "entité ou question (ex: « projet X », « qui est Y »)"}},
    {"name": "finish",
     "description": ("Termine la mission en fournissant ta réponse/synthèse "
                     "finale. À n'appeler que lorsque la tâche est réellement "
                     "terminée (en mode build : seulement après un run_tests vert)."),
     "parameters": {"answer": "réponse finale en texte clair"}},
]


def tools_prompt(names=None) -> str:
    """The tool menu as pretty JSON, for the system prompt. Optionally limited
    to ``names`` (so irrelevant tools aren't dangled in front of a weak model)."""
    schema = (TOOLS_SCHEMA if names is None
              else [t for t in TOOLS_SCHEMA if t["name"] in set(names)])
    return json.dumps(schema, ensure_ascii=False, indent=2)


# ── parsing (tolerant, with a repair pass) ────────────────────────────
def _escape_control_in_strings(s: str) -> str:
    """Échappe les caractères de contrôle LITTÉRAUX (saut de ligne, retour
    chariot, tabulation) à l'intérieur des chaînes JSON. Les petits modèles
    mettent souvent de VRAIS sauts de ligne dans l'argument ``content`` d'un
    write_file (du code multi-lignes) au lieu de ``\\n`` échappés ; ``json.loads``
    rejette alors avec « Invalid control character ». On parcourt le texte en
    suivant l'état « dans une chaîne / hors chaîne » (en respectant les
    échappements) et on n'échappe ces caractères QUE dans les chaînes — sans
    toucher aux ``\\n`` déjà corrects ni aux sauts hors chaîne."""
    out: list[str] = []
    in_str = False
    escaped = False
    for ch in s:
        if in_str:
            if escaped:
                out.append(ch)
                escaped = False
            elif ch == "\\":
                out.append(ch)
                escaped = True
            elif ch == '"':
                out.append(ch)
                in_str = False
            elif ch == "\n":
                out.append("\\n")
            elif ch == "\r":
                out.append("\\r")
            elif ch == "\t":
                out.append("\\t")
            else:
                out.append(ch)
        else:
            out.append(ch)
            if ch == '"':
                in_str = True
    return "".join(out)


def _repair_json(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
    s = re.sub(r",\s*([}\]])", r"\1", s)  # trailing commas
    # Échappe les control chars bruts dans les strings (code multi-lignes mal
    # sérialisé par les petits modèles → sinon « Invalid control character »).
    s = _escape_control_in_strings(s)
    return s


def has_tool_call(text: str) -> bool:
    return bool(parse_tool_calls(text or ""))


_KNOWN_TOOLS = {t["name"] for t in TOOLS_SCHEMA}


def _coerce_call(obj) -> dict | None:
    """Normalise a parsed object into ``{name, arguments}`` or ``None``."""
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or obj.get("tool")
    if not name:
        return None
    name = str(name)
    args = obj.get("arguments")
    if args is None:
        args = obj.get("args")
    if args is None:
        args = obj.get("parameters")
    if args is None:
        # Flat form: {"name": "write_file", "path": "...", "content": "..."}.
        args = {k: v for k, v in obj.items()
                if k not in ("name", "tool", "arguments", "args", "parameters")}
    if isinstance(args, str):
        try:
            args = json.loads(_repair_json(args))
        except Exception:  # noqa: BLE001
            args = {"_raw": args}
    return {"name": name, "arguments": args if isinstance(args, dict) else {}}


def _balanced_json_objects(text: str) -> list[str]:
    """All balanced ``{…}`` substrings, respecting string literals/escapes so
    braces inside content (e.g. a regex ``{2,}``) don't break the matching."""
    out: list[str] = []
    i, n = 0, len(text or "")
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth, j, in_str, esc = 0, i, False, False
        while j < n:
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    out.append(text[i:j + 1])
                    break
            j += 1
        i = j + 1
    return out


def parse_tool_calls(text: str) -> list[dict]:
    """Extract ``[{name, arguments}]``. Accepts ``<tool_call>`` blocks first;
    if none, falls back to BARE JSON objects (smaller models often omit the
    tags and just print ``{"name": "write_file", "arguments": {…}}``)."""
    text = text or ""
    out: list[dict] = []
    # 0) Format MARKDOWN ENRICHI : code BRUT entre balises, sans JSON. Parsé en
    #    PREMIER car immunisé à l'échappement (la plaie des petits modèles sur
    #    du code multi-lignes). Profite aux DEUX modes (thinking/non-thinking) :
    #    c'est l'unique point de parsing, partagé par toute la boucle agent.
    for m in _MD_WRITE_RE.finditer(text):
        path = m.group(1).strip()
        content = _strip_code_fence(m.group(2))
        if path:
            out.append({"name": "write_file",
                        "arguments": {"path": path, "content": content}})
    for m in _MD_EDIT_RE.finditer(text):
        path = m.group(1).strip()
        if path:
            out.append({"name": "edit_file",
                        "arguments": {"path": path, "find": m.group(2),
                                      "replace": m.group(3)}})
    if out:
        return out
    # 0b) Fences de code « affichées » (réflexe des modèles de code qui montrent
    #     le code dans ```python au lieu d'appeler l'outil). Le nom de fichier
    #     est résolu en cascade (commentaire interne → texte précédent →
    #     inférence). Activé SEULEMENT si AUCUNE balise <write_file>/<tool_call>
    #     n'est présente, pour ne pas doubler ni capter du code d'explication.
    if ("<write_file" not in text and "<tool_call>" not in text
            and "<tools>" not in text):
        out = _extract_named_fences(text)
        if out:
            return out
    # 1) Tagged blocks (the documented format). Le regex a 2 groupes :
    #    group(1) = contenu de <tool_call>, group(2) = contenu de <tools>.
    for m in _TOOL_CALL_RE.finditer(text):
        raw = m.group(1) if m.group(1) is not None else m.group(2)
        for cand in (raw, _repair_json(raw)):
            try:
                call = _coerce_call(json.loads(cand))
            except Exception:  # noqa: BLE001
                call = None
            if call:
                out.append(call)
                break
    if out:
        return out
    # 2) Fallback: bare JSON tool calls anywhere in the text. Restricted to
    #    known tool names so arbitrary JSON isn't mistaken for a call.
    for blob in _balanced_json_objects(text):
        for cand in (blob, _repair_json(blob)):
            try:
                obj = json.loads(cand)
            except Exception:  # noqa: BLE001
                continue
            call = _coerce_call(obj)
            if call and call["name"] in _KNOWN_TOOLS:
                out.append(call)
            break
    # OBSERVATION : un <tool_call> était présent mais RIEN n'a été extrait →
    # échec de parsing silencieux (cause connue : control chars non échappés
    # dans le content d'un write_file de code multi-lignes). On le rend VISIBLE
    # et on capture le tool_call brut pour diagnostic définitif, au lieu de
    # l'abandon muet qui faisait boucler l'agent (faux « ✓ », 0 fichier écrit).
    if not out and ("<tool_call>" in (text or "") or "<tools>" in (text or "")):
        _m = _TOOL_CALL_RE.search(text)
        _raw = ""
        if _m:
            _raw = _m.group(1) if _m.group(1) is not None else (_m.group(2) or "")
        log.warning(
            "tool_call présent mais NON PARSÉ (probable control-char/échappement "
            "dans le content). Brut (500 1ers car.): %r", _raw[:500])
    return out


# ── dispatch (the interceptor's executor) ─────────────────────────────
def _safe_rel(path) -> str:
    p = str(path or "").strip().replace("\\", "/")
    if not p or p.startswith("/") or ".." in p.split("/"):
        return ""
    return p


def _err(name: str, msg: str) -> dict:
    return {"tool": name, "ok": False, "error": msg}


def dispatch(call: dict, ops: dict) -> dict:
    """Run one tool call via the injected ``ops`` callables. Always returns a
    structured dict (errors included) so the model can read & recover."""
    name = (call or {}).get("name", "")
    args = (call or {}).get("arguments", {}) or {}
    try:
        if name == "list_files":
            return {"tool": name, "ok": True, "result": ops["list_files"]()}

        if name == "read_file":
            path = _safe_rel(args.get("path", ""))
            if not path:
                return _err(name, "argument 'path' manquant ou invalide")
            return {"tool": name, "ok": True, "result": ops["read_file"](path)}

        if name == "write_file":
            path = _safe_rel(args.get("path", ""))
            content = args.get("content", "")
            if not path:
                return _err(name, "argument 'path' manquant ou invalide")
            if not isinstance(content, str):
                content = str(content)
            if len(content) > _MAX_CONTENT:
                return _err(name, "contenu trop volumineux")
            return {"tool": name, "ok": True, "result": ops["write_file"](path, content)}

        if name == "run_tests":
            return {"tool": name, "ok": True, "result": ops["run_tests"]()}

        if name == "run_command":
            argv_arg = args.get("argv")
            if isinstance(argv_arg, list) and argv_arg:
                argv = [str(a) for a in argv_arg]
            else:
                cmd = args.get("command") or args.get("cmd") or ""
                if not isinstance(cmd, str) or not cmd.strip():
                    return _err(name, "argument 'command' manquant")
                try:
                    argv = shlex.split(cmd)
                except Exception as exc:  # noqa: BLE001
                    return _err(name, f"commande illisible: {exc}")
            ok, reason = validate_command(argv)
            if not ok:
                return _err(name, reason)
            return {"tool": name, "ok": True,
                    "result": ops["run_command"](argv, command_needs_net(argv))}

        if name == "edit_file":
            path = _safe_rel(args.get("path", ""))
            find = args.get("find", "")
            replace = args.get("replace", "")
            if not path:
                return _err(name, "argument 'path' manquant ou invalide")
            if not isinstance(find, str) or find == "":
                return _err(name, "argument 'find' manquant")
            if not isinstance(replace, str):
                replace = str(replace)
            return {"tool": name, "ok": True,
                    "result": ops["edit_file"](path, find, replace)}

        if name == "serve_and_probe":
            cmd = args.get("command") or ""
            if not isinstance(cmd, str) or not cmd.strip():
                return _err(name, "argument 'command' manquant")
            try:
                argv = shlex.split(cmd)
            except Exception as exc:  # noqa: BLE001
                return _err(name, f"commande illisible: {exc}")
            ok, reason = validate_command(argv)
            if not ok:
                return _err(name, reason)
            paths = args.get("paths") or ["/"]
            if isinstance(paths, str):
                paths = [paths]
            try:
                port = int(args.get("port", 8000))
            except Exception:  # noqa: BLE001
                port = 8000
            return {"tool": name, "ok": True,
                    "result": ops["serve_and_probe"](argv, list(paths)[:8], port)}

        if name == "delete_file":
            path = _safe_rel(args.get("path", ""))
            if not path:
                return _err(name, "argument 'path' manquant ou invalide")
            return {"tool": name, "ok": True, "result": ops["delete_file"](path)}

        if name == "search_files":
            query = args.get("query") or args.get("pattern") or ""
            if not isinstance(query, str) or not query.strip():
                return _err(name, "argument 'query' manquant")
            glob = args.get("glob") or args.get("pattern_glob") or "*"
            if not isinstance(glob, str) or not glob.strip():
                glob = "*"
            return {"tool": name, "ok": True,
                    "result": ops["search_files"](query, glob)}

        if name == "run_python":
            code = args.get("code") or args.get("source") or ""
            if not isinstance(code, str) or not code.strip():
                return _err(name, "argument 'code' manquant")
            if len(code) > _MAX_CONTENT:
                return _err(name, "code trop volumineux")
            return {"tool": name, "ok": True, "result": ops["run_python"](code)}

        if name == "web_search":
            query = args.get("query") or args.get("q") or ""
            if not isinstance(query, str) or not query.strip():
                return _err(name, "argument 'query' manquant")
            try:
                n = int(args.get("max_results", 5))
            except Exception:  # noqa: BLE001
                n = 5
            n = max(1, min(n, 10))
            return {"tool": name, "ok": True,
                    "result": ops["web_search"](query, n)}

        if name == "web_fetch":
            url = args.get("url") or args.get("href") or ""
            if not isinstance(url, str) or not url.strip():
                return _err(name, "argument 'url' manquant")
            return {"tool": name, "ok": True,
                    "result": ops["web_fetch"](url.strip())}

        if name == "recall":
            query = args.get("query") or args.get("q") or ""
            if not isinstance(query, str) or not query.strip():
                return _err(name, "argument 'query' manquant")
            return {"tool": name, "ok": True,
                    "result": ops["recall"](query.strip())}

        if name == "query_kg":
            q = (args.get("question") or args.get("query")
                 or args.get("entity") or "")
            if not isinstance(q, str) or not q.strip():
                return _err(name, "argument 'question' manquant")
            return {"tool": name, "ok": True,
                    "result": ops["query_kg"](q.strip())}

        # NOTE: 'finish' is intercepted by the ReAct loop (it signals
        # completion), so it never reaches dispatch.
        return _err(name, f"outil inconnu: {name!r}")
    except KeyError:
        return _err(name, f"outil indisponible: {name!r}")
    except Exception as exc:  # noqa: BLE001
        log.exception("tool dispatch failed: %s", name)
        return _err(name, str(exc))
