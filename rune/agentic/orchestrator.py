"""Agent orchestrator — the "hippocampe agentique", sibling of the chat.

V6 Phase 1. This class does NOT subclass or modify the chat
:class:`~rune.hippocampe.Hippocampe`. It *composes* it: it borrows the
shared model and memory (by reference, so the agent writes into the same
consolidating memory as the chat) and adds a bounded plan → act →
critique loop on top, routed over a :class:`WorkerPool`.

Design constraints honoured here:
* **Self-contained loop.** Planning/critique are prompt-based via the
  worker pool, so the loop depends only on ``worker.generate(prompt)``
  plus the codegen extractor + workspace writer (both already shipped &
  tested). It does not call private phase methods with unknown
  signatures — robust against the chat internals evolving.
* **Safe by default.** The executor's concrete action is *writing code
  files into the workspace sandbox* (reusing ``codegen`` + ``workspace``).
  No autonomous shell/exec in Phase 1 — that surface needs the approval
  UI and is deliberately out of scope.
* **Interruptible.** An inbox per run lets the user inject new
  instructions; they are absorbed between steps. A stop flag ends a run.

Events are plain dicts; the route serialises them as SSE.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
from collections import deque
import threading
import os
import re
import socket
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from rune.agentic.skills import SkillLibrary

log = logging.getLogger(__name__)

_PLAN_LINE = re.compile(r"^\s*\d+[.)\]]\s+(.+?)\s*$")
_CONF_RE = re.compile(r"(\d+(?:\.\d+)?)")

_MAX_STEPS = 8
# 5→8 : un cycle TDD complet (écrire le test + le module + lancer + plusieurs
# corrections) dépassait 5 étapes et finissait coupé prématurément. L'arrêt
# anti-boucle reste géré par ProgressTracker (progress.should_stop, palier
# STOP) — généraliste, tous types de tâche — donc plus d'étapes n'autorise
# PAS une boucle infinie.

# Raisonnement <think> sous chaque étape. Les modèles thinking (Qwen3…)
# produisent un long monologue, souvent en anglais. Par cohérence avec le
# chat — qui ne l'affiche pas — on le MASQUE : 0 = caché (le frontend teste
# `if (data.thought)`, donc aucun bloc n'est rendu quand c'est vide). Mettre
# > 0 pour réafficher un aperçu de N caractères (transparence du raisonnement).
_REASONING_PREVIEW = 0

# Anti-répétition pour les générations de TEXTE pur de l'agent (synthèse,
# critique), passé via _gen(prose=True). repetition_penalty décourage la
# dérive ; no_repeat_ngram_size interdit mécaniquement toute séquence de N
# tokens de se répéter → fin des synthèses qui bouclent, traité À LA SOURCE.
# Jamais appliqué au code (où ces pénalités casseraient des répétitions
# légitimes). _collapse_runaway ne reste plus qu'en ultime filet.
_TEXT_REPETITION_PENALTY = 1.2
_TEXT_NO_REPEAT_NGRAM = 4
_CTX_CLIP = 1200  # chars of prior-step context carried forward

# Plan steps that are pure *process* (no file produced) and that the agent
# cannot actually perform (it only writes files, never runs a shell). We drop
# them so the plan stays concrete and the ⚠ noise disappears.
_NONACTIONABLE = re.compile(
    r"\b(commit|push|pousse[rz]?|git\b|d[ée]p[ôo]t|venv|environnement\s+virtuel|"
    r"virtualenv|pip\s+install|installe[rz]?\s+(les\s+)?d[ée]pendances|"
    r"ex[ée]cut\w*\s+les\s+tests|lance[rz]?\s+les\s+tests|run\s+the\s+tests|"
    r"r[ée]p[éè]t\w*|repeat)\b",
    re.I,
)

# Scaffolding files we never write unless the task explicitly asks for a
# package / CI (otherwise the model spontaneously litters the mission).
_SCAFFOLD = {
    "readme.md", "readme", "readme.txt", "license", "license.md", "licence",
    ".gitignore", ".gitmodules", ".gitattributes", ".dockerignore",
    "setup.py", "setup.cfg", "pyproject.toml", "manifest.in", "tox.ini",
    "requirements.txt", "requirements-dev.txt", "pytest.ini", "conftest.py",
}

# The 7B sometimes interprets "donne ta synthèse finale" as "write a synthesis
# FILE" and loops writing synthese_finale.txt instead of fixing the code. A
# narrative name + a doc/text extension = that deflection → refuse it.
_NARRATIVE_FILE_RE = re.compile(
    r"(synth[eè]se|synthesis|r[eé]sum[eé]|resume|rapport|report|conclusion|"
    r"r[eé]ponse|reponse|answer|r[eé]sultat|resultat|output|sortie)", re.I)
_NARRATIVE_EXT = {".txt", ".md", ".rst", ".log", ".out", ".text"}


def _host_is_public(host: str) -> bool:
    """True only if every address ``host`` resolves to is a public, routable
    IP. Rejects loopback / private / link-local / reserved / multicast /
    unspecified — the SSRF targets (localhost, 10/172.16/192.168, 169.254
    cloud-metadata, …)."""
    try:
        infos = socket.getaddrinfo(host, None)
    except Exception:  # noqa: BLE001 — DNS failure → treat as non-public
        return False
    if not infos:
        return False
    for info in infos:
        sockaddr = info[4]
        try:
            ip = ipaddress.ip_address(sockaddr[0])
        except ValueError:
            return False
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            return False
    return True


def _assert_public_url(url: str) -> None:
    """Raise ``ValueError`` unless ``url`` is http(s) to a public host.

    The anti-SSRF gate for ``web_fetch``: blocks the model (or an instruction
    injected via a fetched page / skill) from reaching internal services,
    localhost, or the cloud-metadata endpoint. Validates the host at call
    time; the redirect handler re-validates every hop, which is where a
    redirect would otherwise smuggle an internal target past this check."""
    from urllib.parse import urlparse

    p = urlparse(url)
    if p.scheme.lower() not in ("http", "https"):
        raise ValueError("schéma non autorisé (http/https requis)")
    if not p.hostname:
        raise ValueError("URL sans hôte")
    if not _host_is_public(p.hostname):
        raise ValueError("hôte interne/privé bloqué (anti-SSRF)")


def _extract_py_block(text: str) -> str:
    """First fenced code block from a model answer (```python … ``` or ``` … ```).
    Salvages code a weak model put in PROSE instead of calling write_file."""
    if not text:
        return ""
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S | re.I)
    return m.group(1).strip() if m else ""


def _filename_from_code(code: str) -> str:
    """Sensible module filename from the first top-level def/class."""
    m = re.search(r"^\s*(?:def|class)\s+([a-zA-Z_]\w*)", code, flags=re.M)
    return (m.group(1) if m else "solution") + ".py"


def _jail_relpath(path: str, subdir: str) -> str:
    """Model-supplied path → MISSION-relative path, or '' if it escapes.

    Strips an echoed ``missions/<slug>/`` prefix, then normalises and rejects
    any ``..`` / absolute component that would leave the mission directory —
    so an agent working in mission A cannot read/write mission B via ``../``.
    Nested sub-paths inside the mission (``src/x.py``) are preserved; callers
    treat ``''`` as an invalid path.
    """
    p = str(path or "").strip().replace("\\", "/").lstrip("/")
    sd = (subdir or "").strip("/")
    if sd and p == sd:
        return ""
    if sd and p.startswith(sd + "/"):
        p = p[len(sd) + 1:]
    if not p:
        return ""
    norm = os.path.normpath(p).replace("\\", "/")
    if (norm in ("..", ".") or norm.startswith("../")
            or norm.startswith("/") or (len(norm) > 1 and norm[1] == ":")):
        return ""                       # escapes the mission → reject
    return norm


def _py_syntax_error(content: str) -> str:
    """SyntaxError description for Python source, or '' if it parses.
    Pre-write gate: refusing unparsable code costs zero run_tests cycles."""
    import ast
    try:
        ast.parse(content)
        return ""
    except SyntaxError as exc:
        return f"SyntaxError ligne {exc.lineno}: {exc.msg}"


def _unified_diff(old: str, new: str, name: str, cap: int = 900) -> str:
    """Short unified diff of an edit — shows the model what ACTUALLY changed
    (a no-op 'fix' shows up as an empty diff, instantly)."""
    import difflib
    lines = list(difflib.unified_diff(
        old.splitlines(), new.splitlines(),
        fromfile=f"{name} (avant)", tofile=f"{name} (après)", lineterm="", n=1))
    return "\n".join(lines)[:cap]


def _parse_test_counts(summary: str) -> tuple[int | None, int | None]:
    """(failed, passed) from a pytest summary like '2 failed, 3 passed in 0.1s'.
    None when absent — drives the graduated progress metric."""
    if not summary:
        return None, None
    mf = re.search(r"(\d+)\s+failed", summary)
    mp = re.search(r"(\d+)\s+passed", summary)
    me = re.search(r"(\d+)\s+error", summary)
    failed = (int(mf.group(1)) if mf else 0) + (int(me.group(1)) if me else 0)
    passed = int(mp.group(1)) if mp else 0
    if mf is None and mp is None and me is None:
        return None, None
    return failed, passed


def _strip_think(text: str) -> str:
    """Retire le raisonnement <think>…</think> d'un modèle thinking, en gérant
    les cas pathologiques qu'un simple regex non-greedy rate :

    - bloc fermé normal → retiré ;
    - bloc NON fermé (génération coupée par le budget de tokens au milieu du
      raisonnement) → tout ce qui suit le <think> non clos est du raisonnement,
      on le coupe ;
    - </think> orpheline (le <think> a été mangé par un stop/troncature amont)
      → on garde ce qui suit la fermeture ;
    - balises multiples → toutes traitées.

    Sans ça, un R1 coupé en plein <think> laisse une balise ouverte, le regex
    standard ne matche pas, et l'extracteur de tool_call reçoit du raisonnement
    brut → action introuvable, étape « vide ». C'est LE bug n°1 des modèles
    thinking dans une boucle ReAct."""
    if not text:
        return ""
    s = text
    # 1) blocs complets
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S | re.I)
    # 2) <think> ouvert mais jamais fermé → couper du marqueur à la fin
    m = re.search(r"<think>", s, flags=re.I)
    if m:
        s = s[:m.start()]
    # 3) </think> orpheline (ouverture perdue en amont) → garder l'après
    m = re.search(r"</think>", s, flags=re.I)
    if m:
        s = s[m.end():]
    return s.strip()


def _capture_think(text: str) -> str:
    """Extrait le texte de raisonnement pour l'afficher dans l'UI, blocs
    fermés ET bloc non fermé final (le plus utile : c'est le raisonnement en
    cours quand le budget coupe)."""
    if not text:
        return ""
    parts = re.findall(r"<think>(.*?)</think>", text, flags=re.S | re.I)
    tail = re.search(r"<think>(?!.*</think>)(.*)$", text, flags=re.S | re.I)
    if tail:
        parts.append(tail.group(1))
    return "\n".join(p.strip() for p in parts).strip()


# Balises de protocole d'appel d'outil. Le parsing (tools.py) les consomme en
# amont ; ces regex servent UNIQUEMENT à les retirer du texte AFFICHÉ — sinon un
# modèle qui émet <tools>/<write_file> (ex. Qwen) les laisse fuiter dans la
# « pensée », alors que <tool_call> seul était masqué.
_TOOL_TAGS_RE = re.compile(
    r"<tool_call>.*?</tool_call>|<tools>.*?</tools>|"
    r"<write_file\b[^>]*>.*?</write_file>|<edit_file\b[^>]*>.*?</edit_file>",
    re.S | re.I,
)
_TOOL_TAG_FRAGMENT_RE = re.compile(
    r"</?(?:tool_call|tools|write_file|edit_file)\b[^>]*>", re.I)


def _strip_tool_syntax(text: str) -> str:
    """Retire les balises d'appel d'outil du texte AFFICHÉ (<tools>,
    <tool_call>, <write_file>, <edit_file>) — blocs complets puis fragments
    orphelins. N'affecte QUE l'affichage ; le parsing se fait sur le texte
    brut en amont. No-op si le texte n'en contient pas."""
    if not text:
        return ""
    s = _TOOL_TAGS_RE.sub("", text)
    s = _TOOL_TAG_FRAGMENT_RE.sub("", s)
    return s.strip()


def _focus_failing_case(stdout_tail: str) -> dict:
    """Isole UN cas d'échec précis depuis la sortie pytest, pour une
    correction ciblée (niveau 4 de l'escalier de déblocage) plutôt qu'une
    énième régénération globale. Retourne {test, assertion, raw} où :
    - ``test``       : nom du test qui échoue (ex. 'test_subdomain')
    - ``assertion``  : la ligne ``E   assert …`` la plus parlante
    - ``raw``        : les quelques lignes brutes autour, pour le contexte.

    Pourquoi : quand le best-of-N a échoué, régénérer encore tout le module
    ne sert à rien (le modèle rate le même cas K fois). Mieux vaut pointer le
    SEUL cas qui résiste et demander de le traiter spécifiquement."""
    if not stdout_tail:
        return {}
    test = ""
    assertion = ""
    raw: list[str] = []
    for ln in stdout_tail.splitlines():
        s = ln.strip()
        if not test and s.startswith("FAILED "):
            # 'FAILED test_x.py::test_subdomain - AssertionError: ...'
            seg = s[len("FAILED "):]
            test = seg.split("::", 1)[-1].split(" - ", 1)[0].strip() or seg
            raw.append(s)
        elif (s.startswith("E   assert") or s.startswith("E   AssertionError")
              or s.startswith("E   ")) and not assertion:
            assertion = s[2:].strip()
            raw.append(s)
        elif s.startswith("E   ") and len(raw) < 6:
            raw.append(s)
    return {"test": test, "assertion": assertion,
            "raw": "\n".join(raw[:6])} if (test or assertion) else {}


def _pytest_failure_digest(stdout_tail: str, limit: int = 700) -> str:
    """Extract the model-actionable lines from pytest output: the
    ``FAILED test::name - AssertionError: …`` summary and the ``E   …``
    assertion lines. Far more useful to a weak model than a raw 600-char
    JSON dump that often truncates the assertion away."""
    if not stdout_tail:
        return ""
    keep: list[str] = []
    for ln in stdout_tail.splitlines():
        s = ln.strip()
        if s.startswith("FAILED ") or s.startswith("E   ") or s.startswith("E "):
            keep.append(s)
        elif s.startswith("ERROR ") or "Error:" in s[:40]:
            keep.append(s)
    out = "\n".join(keep[:14]).strip()
    return out[:limit]


def _largest_code_block(text: str) -> str:
    """Best-effort extraction of source code from a model reply that has no
    proper ``# file:`` marker — used to salvage a recovery module write.
    Prefers fenced blocks; falls back to the whole reply."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text or "", flags=re.S)
    if not blocks:
        blocks = re.findall(r"```\s*\n?(.*?)```", text or "", flags=re.S)
    cand = max(blocks, key=len) if blocks else (text or "")
    cand = re.sub(r"(?im)^\s*#\s*file:.*$\n?", "", cand)  # drop a marker line
    return cand.strip()

# Task keywords that justify a real package layout (src/ + tests/ + scaffold).
_PACKAGE_HINT = re.compile(
    r"\b(package|packaging|mlops|ci\b|github\s+actions|pyproject|setuptools|"
    r"structure\s+de\s+projet|arborescence|installable|pip\s+installable)\b",
    re.I,
)

# ── Task-type routing (polyvalent agent) ──────────────────────────────
# The ReAct agent handles three kinds of work, each with its own workflow
# and completion criterion. Default is "build" (the validated test-driven
# code path) — only a clear analyze/answer signal switches mode, so code
# generation is never weakened. Future capabilities (web, email) plug in as
# new modes/tools in the same frame.
_RE_ANSWER = re.compile(
    r"\b(synth[èe]se|synth[ée]tis\w+|r[ée]sum\w+|explique\w*|expliqu\w+|"
    r"d[ée]cri\w+|liste\w*|[ée]num[èe]r\w+|quels?\s+sont|qu['’]?est[- ]ce|"
    r"que\s+(?:contient|dit|raconte)|de\s+quoi|pourquoi|aper[çc]u|"
    r"passe\s+en\s+revue|revue\s+de|dis[- ]moi)\b",
    re.I,
)
_RE_ANALYZE = re.compile(
    r"\b(analyse\w*|analys\w+|calcul\w+|moyenne\w*|m[ée]dian\w*|somme\w*|"
    r"[ée]cart[- ]type|variance|statisti\w+|compte\w*|compter|combien|"
    r"extrai\w+|extrair\w+|agr[èée]g\w+|trie\w*|trier|filtr\w+|"
    r"distribution|corr[ée]lation|histogramm\w*|graphi\w+|trace\w*|tracer|"
    r"visualis\w+|nombre\s+de|fr[ée]quence)\b",
    re.I,
)
_RE_BUILD = re.compile(
    r"\b(module|fonction\w*|function|classe\w*|class\b|m[ée]thode\w*|"
    r"impl[ée]ment\w+|programme\w*|api\b|endpoint\w*|serveur|server|"
    r"librairie|biblioth[èe]que|package|paquet|test\w*|refactor\w*|"
    r"corrig\w+|debug\w*|d[ée]bog\w+|fixe\b|patch\w*|"
    r"[ée]cri\w*\s+(?:un|une|le|la|du)\s+(?:module|code|script|fonction|"
    r"classe|programme|api))\b",
    re.I,
)


_RE_RESEARCH = re.compile(
    r"\b(recherch\w+|cherch\w+|trouve\w*|web|internet|en\s+ligne|"
    r"actualit\w+|r[ée]cent\w*|derni[èe]r\w*|nouveaut\w+|2024|2025|2026|"
    r"[ée]tat\s+de\s+l['’]art|state\s+of\s+the\s+art|source\w*)\b",
    re.I,
)


def _classify_task(task: str) -> str:
    """Return the agent workflow for a task:
    'build' | 'analyze' | 'answer' | 'research'.

    Conservative on purpose: defaults to ``build`` (the validated test-driven
    path) unless there is a clear non-build signal AND no build signal, so code
    generation is never accidentally weakened. Ordering matters: a CALCULATION
    signal (analyze) wins over a mere « résume/synthèse » (answer) so a compute
    task that also asks for a summary isn't sent to the answer mode (which has
    no compute tools). A WEB/research signal routes to ``research``.
    """
    t = task or ""
    build = bool(_RE_BUILD.search(t))
    if build:
        return "build"
    analyze = bool(_RE_ANALYZE.search(t))
    research = bool(_RE_RESEARCH.search(t))
    # Recherche web : prioritaire si signal web ET pas un simple calcul local.
    if research and not analyze:
        return "research"
    # Calcul : prioritaire sur « answer » (un calcul qui demande aussi un
    # résumé reste un calcul — il a besoin des outils de calcul).
    if analyze:
        return "analyze"
    if _RE_ANSWER.search(t):
        return "answer"
    if research:
        return "research"
    return "build"

# Meta/non-code plan steps dropped on simple tasks (the model keeps adding
# documentation/packaging/UI steps that dilute the actual code production).
_PLAN_DROP = re.compile(
    r"\b(document\w*|docstring|readme|commentaire\w*|interface\s+utilisateur|"
    r"\bUI\b|exigenc\w*|requirements?|packag\w*|déploie\w*|deploie\w*|"
    r"publi\w*|structure\s+du\s+projet)\b",
    re.I,
)

# Conversational solicitations the chatty 7B appends; stripped from the
# synthesis (e.g. "Votre avis est apprécié.", "Est-ce correct ?").
_SOLICIT_RE = re.compile(
    r"(votre avis[^.?\n]*[.?]|n['’]h[ée]sitez[^.?\n]*[.?]|"
    r"faites-?moi savoir[^.?\n]*[.?]|je suis pr[êe]t[^.?\n]*[.?]?|"
    r"est-ce correct[^?\n]*\?|ai-je manqu[ée][^?\n]*\?|"
    r"qu['’]en pensez-vous[^?\n]*\?|dites-moi[^.?\n]*[.?])",
    re.I,
)


# Certains modèles (souvent « coder », non-thinking) terminent leur synthèse
# puis REDÉMARRENT un raisonnement en clair (chain-of-thought), parfois dans une
# autre langue : « Okay, let me try to figure out… », « The user wants… ». Ce
# n'est ni du <think> balisé ni du code : on tronque dès la première amorce de
# méta-raisonnement. Ciblé sur des tournures de CoT (jamais du contenu d'une
# synthèse) → no-op sur une synthèse normale, même rédigée en anglais.
_REASONING_RESTART_RE = re.compile(
    r"(?is)\b(?:"
    r"okay,?\s+(?:so\s+)?let'?s?\s+(?:me\s+)?(?:try|think|figure|start|see|break|begin)"
    r"|alright,?\s+(?:so\s+)?let'?s?\s+(?:me\s+)?(?:try|think|figure|start|see|begin)"
    r"|let'?s?\s+me\s+(?:try\s+to\s+figure|think\s+about|start\s+by|break\s+this|figure\s+out)"
    r"|the\s+user\s+(?:wants?|is\s+asking|needs?|wrote|said|provided)"
    r"|first,?\s+i\s+(?:need|should|have|must|'ll|'d)\b"
    r"|i\s+need\s+to\s+(?:remember|figure|think|recall|start|first|understand)"
    r"|so\s+the\s+steps?\s+(?:would|are|is|here)"
    r"|wait,?\s+but\s+(?:in|the|if|how|what|maybe)"
    r"|hmm,?\s+(?:first|let|so|the\s|i\s|wait|maybe|okay)"
    r"|how\s+to\s+approach\s+this"
    r").*$"
)


def _collapse_runaway(text: str, min_words: int = 4, max_repeats: int = 1) -> str:
    """Coupe une répétition dégénérée (le modèle boucle sur les mêmes
    phrases). Conservateur : ne traite QUE les segments « substantiels »
    (≥ min_words mots) qui réapparaissent au-delà de max_repeats — une
    synthèse saine (phrases uniques) passe intacte (no-op)."""
    if not text:
        return ""
    segs = re.split(r"(?<=[.!?])\s+|\n+", text)
    seen: dict[str, int] = {}
    out: list[str] = []
    for seg in segs:
        norm = re.sub(r"\s+", " ", seg.strip().lower())
        if len(norm.split()) < min_words:
            out.append(seg)            # garder les transitions courtes
            continue
        seen[norm] = seen.get(norm, 0) + 1
        if seen[norm] > max_repeats:
            break                      # boucle détectée → on coupe ici
        out.append(seg)
    return " ".join(s.strip() for s in out if s.strip()).strip()


def _clean_synthesis(text: str) -> str:
    """Nettoie la synthèse finale : leak <think>, déballage de code, redémarrage
    de raisonnement (CoT, parfois en anglais), glitch CJK isolé, répétition en
    boucle, sollicitations. No-op sur une synthèse saine (texte cohérent, sans
    <think>, sans code, sans CoT, sans répétition → rien retiré)."""
    text = _strip_think(text or "")          # raisonnement <think> (modèles thinking)
    # La synthèse est du TEXTE (le code vit dans les fichiers, jamais ici). Les
    # modèles « coder » tendent à déballer du code + de l'auto-revue après leur
    # réponse : on tronque dès le premier bloc ``` → jette le code ET les
    # commentaires qui suivent. Une synthèse propre n'a pas de ``` (no-op).
    text = text.split("```", 1)[0]
    # Puis on tronque un éventuel REDÉMARRAGE de raisonnement en clair après la
    # synthèse (« Okay, let me try… », « The user wants… ») — fréquent sur les
    # modèles non-thinking, indétectable par <think> (non balisé).
    text = _REASONING_RESTART_RE.sub("", text)
    # Glitch multilingue : caractères CJK isolés AU MILIEU de mots latins
    # (« cas测试és »). Retirés seulement s'ils sont entourés de latin → un texte
    # réellement en CJK (caractères entre eux) n'est jamais touché.
    text = re.sub(r"(?<=[A-Za-zÀ-ÿ])[\u4e00-\u9fff]+(?=[A-Za-zÀ-ÿ])", "", text)
    text = _SOLICIT_RE.sub("", text)
    text = _collapse_runaway(text)           # boucle de phrases répétées
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# A leading imperative verb makes a poor mission name ("Écris en python");
# strip it, and if nothing solid remains we derive the name from the task.
_NAME_VERB_PREFIX = re.compile(
    r"^(écri\w*|ecri\w*|cré\w*|cre\w*|fai\w*|gén\w*|gen\w*|impl\w*|"
    r"cod\w*|développ\w*|developp\w*|ajout\w*|construi\w*|réalis\w*|"
    r"realis\w*|fabriqu\w*|prépar\w*|prepar\w*|mets?|met\w*)\b[\s:,–-]*",
    re.I,
)

# Filler words dropped when deriving a name from the task itself.
_NAME_STOP = {
    "écris", "ecris", "écrire", "crée", "creer", "cree", "fais", "faire",
    "génère", "genere", "code", "coder", "développe", "developpe", "moi",
    "un", "une", "le", "la", "les", "de", "des", "du", "en", "avec", "et",
    "pour", "stp", "svp", "python", "script", "module", "fichier", "petit",
    "petite", "simple", "programme", "fonction", "ses", "son", "sa", "qui",
    "test", "tests", "unittest", "pytest", "unitaire", "unitaires",
}

# Generic words the model sometimes echoes as a "title" (e.g. it parrots the
# instruction "nom de mission" or picks the test framework). A name made only
# of these is meaningless → derive the name from the task domain instead.
_NAME_PLACEHOLDER = {
    "nom", "titre", "mission", "tache", "tâche", "sans",
    "test", "tests", "unittest", "pytest", "unitaire", "unitaires",
}


def _slugify(text: str, maxlen: int = 32) -> str:
    """Folder-safe slug from a mission name (accents folded: é→e)."""
    import unicodedata

    s = unicodedata.normalize("NFKD", (text or "").strip().lower())
    s = s.encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:maxlen].strip("-") or "mission"


def _sandbox_mode() -> str:
    try:
        from rune.agentic.sandbox import isolation

        return isolation()["mode"]
    except Exception:  # noqa: BLE001
        return "subprocess"


@dataclass
class _Run:
    task: str
    inbox: list[str] = field(default_factory=list)
    stop: bool = False
    # Levé par stop() : consulté par le StoppingCriteria du modèle à chaque
    # token → interrompt la génération EN COURS (stop réactif, pas seulement
    # entre les étapes).
    cancel: threading.Event = field(default_factory=threading.Event)
    # V0.1.1 — buffer d'événements récents pour le monitoring live
    # (dashboard). Alimenté au point unique où run() yield ses events.
    # deque bornée : ne grossit jamais, garde les N derniers pas de l'agent.
    events: deque = field(default_factory=lambda: deque(maxlen=40))
    started_at: float = field(default_factory=time.time)
    name: str = ""
    slug: str = ""
    done: bool = False


def _record_event(run: "_Run", ev: dict) -> None:
    """Résume un event de l'agent et l'ajoute au buffer du run.

    Défensif : ne lève JAMAIS (le monitoring ne doit pas casser une
    mission). On extrait juste de quoi afficher une ligne lisible dans
    le dashboard — type, outil, ok/ko, court résumé — sans stocker les
    gros payloads (contenus de fichiers, etc.).
    """
    try:
        etype = ev.get("type", "?")
        entry: dict = {"t": round(time.time() - run.started_at, 1), "type": etype}
        if etype == "tool_call":
            entry["tool"] = ev.get("name", "?")
            args = ev.get("arguments", {}) or {}
            # Résumé compact de l'argument principal (path ou command).
            hint = args.get("path") or args.get("command") or args.get("test_file") or ""
            if hint:
                entry["hint"] = str(hint)[:60]
        elif etype == "tool_result":
            entry["tool"] = ev.get("name", "?")
            entry["ok"] = bool(ev.get("ok", False))
            prev = ev.get("preview", "")
            if prev:
                entry["hint"] = str(prev)[:80]
        elif etype in ("agent_warning", "critique", "deliberation"):
            entry["hint"] = str(ev.get("message") or ev.get("text") or "")[:80]
        elif etype == "plan":
            steps = ev.get("steps", [])
            entry["hint"] = f"{len(steps)} étapes"
        elif etype == "synthesis":
            entry["hint"] = "synthèse produite"
        elif etype in ("run_done", "run_stopped"):
            run.done = True
        elif etype == "run_start":
            run.name = ev.get("name", "") or run.name
            run.slug = ev.get("slug", "") or run.slug
        run.events.append(entry)
    except Exception:  # noqa: BLE001 — le monitoring ne casse jamais un run
        pass


class AgentOrchestrator:
    """Bounded agentic loop over a shared model + memory."""

    def __init__(
        self,
        hippocampe,
        worker_pool,
        workspace_manager=None,
        *,
        settings=None,
        execution_enabled: bool = False,
        sandbox_factory=None,
        react_enabled: bool = False,
    ):
        self.hippocampe = hippocampe
        self.pool = worker_pool
        self.workspace = workspace_manager
        self.settings = settings
        # Execution defaults OFF so direct construction (tests) never spawns a
        # venv; the route turns it on from settings (default True there).
        self.execution_enabled = execution_enabled
        self.react_enabled = react_enabled
        self._sandbox_factory = sandbox_factory
        self._runs: dict[str, _Run] = {}
        # V0.1.1 — dernière mission terminée, conservée pour l'affichage
        # dashboard APRÈS la purge de _runs (sinon le dashboard se vide
        # dès la fin d'une mission). Une seule référence : pas de fuite.
        self._last_run: tuple[str, _Run] | None = None
        self._trigger_emb_cache: dict[str, object] = {}  # trigger text → emb
        self._skill_lib: SkillLibrary | None = None       # lazy

    # Minimum cosine for a past lesson to be considered relevant (semantic
    # recall). Below it, a lesson is unrelated and is not injected.
    _RECALL_MIN_COS = 0.35

    # ── sandbox plumbing ─────────────────────────────────────────────
    def _make_sandbox(self, mission_dir):
        if self._sandbox_factory is not None:
            return self._sandbox_factory(mission_dir)
        from rune.agentic.sandbox import make_mission_sandbox

        return make_mission_sandbox(mission_dir, self.settings)

    def _mission_abs(self, subdir: str):
        ws = self.workspace
        if ws is None:
            return None
        try:
            if hasattr(ws, "_resolve_safe"):
                return ws._resolve_safe(subdir)
            return Path(ws.root) / subdir
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _existing_files(abs_dir, limit: int = 40) -> list[str]:
        if abs_dir is None or not Path(abs_dir).exists():
            return []
        out: list[str] = []
        base = Path(abs_dir)
        for p in sorted(base.rglob("*")):
            if ".venv" in p.parts or p.is_dir():
                continue
            if p.name.startswith(".lythea-"):
                continue
            out.append(str(p.relative_to(base)).replace("\\", "/"))
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _missing_modules_under_test(abs_dir) -> list[str]:
        """Local modules a ``test_*.py`` imports but that were never written.

        Convention-driven and high-precision: for ``test_<stem>.py`` (or
        ``<stem>_test.py``) whose ``<stem>`` is imported in the test yet has no
        local ``<stem>.py`` / ``<stem>/__init__.py``, ``<stem>`` is the module
        under test and is MISSING. We generate it rather than let the reactive
        installer fetch a PyPI homonym (e.g. ``email_validator``).
        """
        if abs_dir is None or not Path(abs_dir).exists():
            return []
        base = Path(abs_dir)
        present, tests = set(), []
        for p in base.rglob("*.py"):
            if ".venv" in p.parts:
                continue
            present.add(p.stem)
            if p.stem.startswith("test_") or p.stem.endswith("_test"):
                tests.append(p)
        missing: set[str] = set()
        for p in tests:
            s = p.stem
            stem = s[5:] if s.startswith("test_") else (s[:-5] if s.endswith("_test") else "")
            if not stem or stem in present:
                continue
            if (base / f"{stem}.py").exists() or (base / stem / "__init__.py").exists():
                continue
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:  # noqa: BLE001
                continue
            if re.search(rf"(?:from|import)\s+{re.escape(stem)}\b", txt):
                missing.add(stem)
        return sorted(missing)

    @staticmethod
    def _public_symbols(abs_dir, rel: str) -> list[str]:
        """Top-level class/function names of a module (for accurate imports)."""
        try:
            txt = (Path(abs_dir) / rel).read_text(encoding="utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            return []
        names = re.findall(r"^(?:class|def)\s+([A-Za-z_]\w*)", txt, re.M)
        seen, out = set(), []
        for n in names:
            if n.startswith("_") or n in seen:
                continue
            seen.add(n)
            out.append(n)
        return out[:12]

    @staticmethod
    def _modules_needing_impl(abs_dir) -> list[tuple[str, list[str]]]:
        """Local modules that a test imports but that don't (fully) exist.

        Returns ``[(stem, [required_symbols]), …]`` where a ``test_*.py``
        does ``from <stem> import a, b`` but ``<stem>.py`` is either absent OR
        present yet missing some of ``a, b``. Covers the "stub module" case
        (file exists, function not implemented → ImportError at collection).
        Restricted to local-intended modules (a sibling ``test_<stem>.py`` or
        an existing ``<stem>.py``) so third-party imports are never flagged.
        """
        if abs_dir is None or not Path(abs_dir).exists():
            return []
        base = Path(abs_dir)
        py = [p for p in base.rglob("*.py") if ".venv" not in p.parts]
        present = {p.stem for p in py}
        local_intended = set(present)
        for s in present:
            if s.startswith("test_"):
                local_intended.add(s[5:])
            elif s.endswith("_test"):
                local_intended.add(s[:-5])
        req: dict[str, set] = {}
        from_re = re.compile(r"^\s*from\s+([A-Za-z_]\w*)\s+import\s+(.+)$", re.M)
        for p in py:
            if not (p.stem.startswith("test_") or p.stem.endswith("_test")):
                continue
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:  # noqa: BLE001
                continue
            for m in from_re.finditer(txt):
                mod, seg = m.group(1), m.group(2)
                if mod not in local_intended:
                    continue
                seg = seg.split("#")[0].replace("(", "").replace(")", "")
                for name in seg.split(","):
                    name = name.strip().split(" as ")[0].strip()
                    if re.fullmatch(r"[A-Za-z_]\w*", name) and name != "*":
                        req.setdefault(mod, set()).add(name)
        out: list[tuple[str, list[str]]] = []
        for mod, syms in req.items():
            f = base / f"{mod}.py"
            if not f.exists() and not (base / mod / "__init__.py").exists():
                out.append((mod, sorted(syms)))
                continue
            try:
                txt = f.read_text(encoding="utf-8", errors="ignore")
            except Exception:  # noqa: BLE001
                txt = ""
            defined = set(re.findall(r"^(?:class|def)\s+([A-Za-z_]\w*)", txt, re.M))
            defined |= set(re.findall(r"^([A-Za-z_]\w*)\s*=", txt, re.M))
            missing = [s for s in sorted(syms) if s not in defined]
            if missing:
                out.append((mod, missing))
        return out

    @staticmethod
    def _circular_modules(abs_dir) -> list[tuple[str, str]]:
        """Local modules that import THEMSELVES → circular/partial-import error
        at collection (the file exists, the symbols exist, but importing it
        raises). The 7B hits this when it mirrors a real PyPI package's API
        (e.g. ``email_validator``) and writes ``from email_validator import …``
        inside ``email_validator.py``. Returns ``[(stem, source_excerpt), …]``.
        """
        if abs_dir is None or not Path(abs_dir).exists():
            return []
        base = Path(abs_dir)
        out: list[tuple[str, str]] = []
        for p in base.rglob("*.py"):
            if ".venv" in p.parts or p.stem.startswith("test_") or p.stem.endswith("_test"):
                continue
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:  # noqa: BLE001
                continue
            stem = re.escape(p.stem)
            if re.search(rf"^\s*(from\s+{stem}\s+import|import\s+{stem})\b", txt, re.M):
                out.append((p.stem, txt[:1500]))
        return out

    def _flatten_imports(self, abs_dir, subdir=None) -> None:
        """Rewrite ``from pkg.mod import`` / ``import pkg.mod`` to drop a
        package prefix that flat layout collapsed away.

        When the model emits ``utils/email_validator.py`` (flattened to
        ``email_validator.py``) and a test does ``from utils.email_validator
        import …``, the ``utils`` prefix no longer resolves. We rewrite to
        ``from email_validator import …`` whenever ``email_validator`` is a flat
        local module and ``utils`` is not a real directory in the mission.
        """
        if abs_dir is None:
            return
        base = Path(abs_dir)
        pyfiles = [p for p in base.rglob("*.py") if ".venv" not in p.parts]
        stems = {p.stem for p in pyfiles}
        dirs = {p.name for p in base.rglob("*") if p.is_dir()}
        from_re = re.compile(r"from\s+([A-Za-z_]\w*)\.([A-Za-z_]\w*)\s+import\s+")
        imp_re = re.compile(r"import\s+([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b")

        def _rf(m):
            prefix, stem = m.group(1), m.group(2)
            return (f"from {stem} import "
                    if (stem in stems and prefix not in dirs) else m.group(0))

        def _ri(m):
            prefix, stem = m.group(1), m.group(2)
            return (f"import {stem}"
                    if (stem in stems and prefix not in dirs) else m.group(0))

        for p in pyfiles:
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:  # noqa: BLE001
                continue
            new = imp_re.sub(_ri, from_re.sub(_rf, txt))
            if new != txt:
                try:
                    p.write_text(new, encoding="utf-8")
                except Exception:  # noqa: BLE001
                    log.debug("flatten_imports write failed: %s", p, exc_info=True)

    @staticmethod
    def _is_local_pkg_prefix(abs_dir, mod: str) -> bool:
        """True if ``mod`` is used only as a dotted import prefix (``mod.x``)
        with no real ``mod/`` directory — a flatten artifact, never a PyPI dep
        to install (guards against installing squatted names like ``utils``)."""
        if abs_dir is None or not mod:
            return False
        base = Path(abs_dir)
        if (base / mod).is_dir():
            return False
        pat = re.compile(rf"(?:from|import)\s+{re.escape(mod)}\.")
        for p in base.rglob("*.py"):
            if ".venv" in p.parts:
                continue
            try:
                if pat.search(p.read_text(encoding="utf-8", errors="ignore")):
                    return True
            except Exception:  # noqa: BLE001
                continue
        return False

    def _write_manifest(self, abs_dir, data: dict) -> None:
        if abs_dir is None:
            return
        try:
            Path(abs_dir).mkdir(parents=True, exist_ok=True)
            (Path(abs_dir) / ".lythea-mission.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:  # noqa: BLE001
            log.debug("manifest write failed", exc_info=True)

    @staticmethod
    def _read_manifest(abs_dir) -> dict:
        try:
            return json.loads(
                (Path(abs_dir) / ".lythea-mission.json").read_text(encoding="utf-8")
            )
        except Exception:  # noqa: BLE001
            return {}

    async def _to_thread(self, fn):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, fn)

    # ── public control surface ───────────────────────────────────────
    def interject(self, run_id: str, text: str) -> bool:
        run = self._runs.get(run_id)
        if run is None or not text.strip():
            return False
        run.inbox.append(text.strip())
        return True

    def stop(self, run_id: str) -> bool:
        run = self._runs.get(run_id)
        if run is None:
            return False
        run.stop = True
        run.cancel.set()   # interrompt aussi la génération EN COURS (réactif)
        return True

    # ── helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _parse_plan(text: str) -> list[str]:
        steps: list[str] = []
        for line in (text or "").splitlines():
            m = _PLAN_LINE.match(line)
            if m:
                title = m.group(1).strip()
                # Drop pure-process steps the agent can't perform (commit,
                # venv, "run the tests", "repeat steps 2-4"…).
                if _NONACTIONABLE.search(title):
                    continue
                steps.append(title)
        return steps[:_MAX_STEPS]

    async def _gen(self, worker, prompt: str, *, prose: bool = False, **kw) -> str:
        # Agent generations default to a LOW temperature (near-deterministic —
        # code/agentic work benefits from consistency, not creativity) and a
        # larger token budget (so a module/test isn't truncated mid-file). Both
        # are overridable per-call and via settings.
        s = getattr(self, "settings", None)
        kw.setdefault("temperature", float(getattr(s, "agent_temperature", 0.3)))
        kw.setdefault("max_new_tokens",
                      int(getattr(s, "agent_max_new_tokens", 1024)))
        # prose=True : génération de TEXTE pur (synthèse, critique). On bride la
        # répétition À LA SOURCE (cf. _TEXT_*), au lieu de la rattraper après
        # coup. Jamais pour les générations qui portent du code (write_file…),
        # où répéter un mot-clé ou une indentation est légitime. Restent
        # overridables par appel.
        if prose:
            kw.setdefault("repetition_penalty", _TEXT_REPETITION_PENALTY)
            kw.setdefault("no_repeat_ngram_size", _TEXT_NO_REPEAT_NGRAM)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: worker.generate(prompt, **kw))

    async def _gen_batch(self, worker, prompts: list[str], **kw) -> list[str]:
        """Batched generation (one GPU pass) with graceful fallback to a
        sequential loop when the worker doesn't support batching. Batches are
        CHUNKED to the hardware profile's batch_max (24 GB card ≠ H100 ≠
        multi-GPU — same code, adaptive width)."""
        s = getattr(self, "settings", None)
        kw.setdefault("temperature", float(getattr(s, "agent_temperature", 0.3)))
        kw.setdefault("max_new_tokens",
                      int(getattr(s, "agent_max_new_tokens", 1024)))
        loop = asyncio.get_event_loop()
        if hasattr(worker, "generate_batch"):
            try:
                bmax = max(1, int(self._hw_knobs().get("batch_max", 4)))
                out: list[str] = []
                for i in range(0, len(prompts), bmax):
                    chunk = list(prompts[i:i + bmax])
                    out.extend(await loop.run_in_executor(
                        None, lambda c=chunk: worker.generate_batch(c, **kw)))
                return out
            except Exception:  # noqa: BLE001 — OOM/odd tokenizer → fallback
                log.warning("generate_batch failed, falling back to sequential",
                            exc_info=True)
        out = []
        for p in prompts:
            out.append(await loop.run_in_executor(
                None, lambda p=p: worker.generate(p, **kw)))
        return out

    def _hw_knobs(self) -> dict:
        """Hardware-adaptive knobs, computed once (24 GB / H100 / multi-GPU /
        CPU-only give different batch, best-of-N and parallelism widths)."""
        if getattr(self, "_hwk", None) is None:
            try:
                from rune import hwprofile
                self._hwk = hwprofile.knobs(hwprofile.detect())
            except Exception:  # noqa: BLE001
                self._hwk = {"batch_max": 2, "bestofn": 2,
                             "subagents": 1, "parallel_tests": 2}
        return self._hwk

    def _thinking_mode(self, worker) -> bool:
        """Le profil thinking s'applique-t-il ? « auto » = oui si le modèle
        chargé raisonne en interne (is_thinking) ; « on »/« off » forcent.

        Quand actif, l'échafaudage que le modèle fait NATIVEMENT est allégé
        (plan initial, micro-checks, best-of-N) — mais blackboard, routeur,
        gardes de sécurité, batching restent ON. Le bench tranche au final."""
        mode = str(getattr(self.settings, "agent_thinking_profile", "auto"))
        if mode == "on":
            return True
        if mode == "off":
            return False
        return bool(getattr(getattr(worker, "model", None), "is_thinking", False)
                    or getattr(worker, "is_thinking", False))

    @staticmethod
    def _norm_path(path: str, flat: bool) -> str:
        """Orchestrator owns the layout, not the model.

        Flat mode (default): collapse everything to the mission root
        (``src/fibo.py`` → ``fibo.py``) so two runs of the same task give
        the same tree and ``pytest`` at the root resolves imports. Package
        mode keeps whatever structure the model emitted.
        """
        p = (path or "").strip().lstrip("/").replace("\\", "/")
        if flat:
            p = os.path.basename(p)
        return p

    @staticmethod
    def _is_scaffold(path: str) -> bool:
        low = path.lower()
        base = os.path.basename(low)
        return (
            base in _SCAFFOLD
            or low.startswith(".github")
            or "/.github/" in low
            or base.startswith(".git")
        )

    def _write_files(
        self,
        text: str,
        subdir: str,
        *,
        flat: bool = True,
        seen: dict | None = None,
        allow_scaffold: bool = False,
    ) -> list[dict]:
        """Extract ``# file:`` blocks and write them into the sandbox.

        ``seen`` (run-scoped) deduplicates: a path re-emitted with identical
        content is skipped, and collapsing to the mission root merges
        ``src/x.py``/``x.py`` into a single file (latest content wins).
        """
        if self.workspace is None:
            return []
        try:
            from rune.server.codegen import extract_code_files
            from rune.server.workspace import WorkspaceError
        except Exception:  # noqa: BLE001
            return []
        if seen is None:
            seen = {}
        written: list[dict] = []
        sub = (subdir or "").strip().strip("/")
        for cf in extract_code_files(text):
            rel0 = self._norm_path(cf.path, flat)
            if not rel0:
                continue
            if not allow_scaffold and self._is_scaffold(rel0):
                continue  # no spontaneous README/.gitignore/CI/setup.py
            rel = f"{sub}/{rel0}" if sub else rel0
            digest = hash(cf.content)
            if seen.get(rel) == digest:
                continue  # identical content already written this run
            try:
                entry = self.workspace.write_text_file(rel, cf.content)
                seen[rel] = digest
                written.append({"path": entry.path, "size": entry.size})
            except WorkspaceError as exc:
                written.append({"path": rel, "error": str(exc)})
            except Exception as exc:  # noqa: BLE001
                log.exception("agent file write failed: %s", rel)
                written.append({"path": rel, "error": str(exc)})
        return written

    @staticmethod
    def _confidence(text: str) -> float:
        m = _CONF_RE.search(text or "")
        if not m:
            return 0.5
        try:
            v = float(m.group(1))
        except ValueError:
            return 0.5
        if v > 1.0:  # model answered on a 0-100 scale
            v /= 100.0
        return max(0.0, min(1.0, v))

    async def _gen_name(self, core, task: str) -> str:
        """Ask the core for a short mission title; fall back to the task."""
        prompt = (
            "Donne un titre très court (2 à 4 mots) pour cette tâche, pour "
            "l'afficher comme nom de mission. Pas de ponctuation finale, pas "
            "de guillemets — juste le titre.\n\nTâche : " + task
        )
        raw = ""
        try:
            raw = await self._gen(core, prompt)
        except Exception:  # noqa: BLE001
            pass
        lines = [ln.strip() for ln in (raw or "").splitlines() if ln.strip()]
        name = lines[0] if lines else ""
        # The model often prefixes a label/preamble ("Titre :", "Voici un
        # titre : …"): keep only what follows the first colon.
        if ":" in name:
            name = name.split(":", 1)[1]
        name = name.strip("\"'`*#–-. ").strip()
        name = _NAME_VERB_PREFIX.sub("", name).strip()  # drop "Écris…", "Crée…"
        words = name.split()
        if len(words) > 4:                      # keep it genuinely short
            name = " ".join(words[:4])
        name = re.sub(r"[\s&\-–:.]+$", "", name)  # no trailing junk like " &"
        # Reject generic placeholders the model echoes back ("Nom de la
        # mission", "Titre", "Mission"…): if nothing meaningful remains, derive
        # the name from the task instead.
        meaningful = [
            w for w in re.findall(r"[a-zA-Zàâäéèêëîïôöùûüç]+", name.lower())
            if w not in _NAME_STOP and w not in _NAME_PLACEHOLDER
        ]
        if not name or len(name) > 40 or not meaningful:
            tw = [
                w for w in re.findall(r"\w+", task)
                if len(w) > 1 and w.lower() not in _NAME_STOP
            ]
            name = " ".join(tw[:4]).title() if tw else "Mission"
        return name[:40]

    def _unique_slug(self, slug: str) -> str:
        """Suffix the slug (-2, -3, …) until ``missions/<slug>`` is free."""
        ws = self.workspace
        if ws is None or not hasattr(ws, "exists"):
            return slug
        candidate, n = slug, 1
        while ws.exists(f"missions/{candidate}"):
            n += 1
            candidate = f"{slug}-{n}"
        return candidate

    # ── the model-driven tool-calling loop (Hermès, bounded) ─────────
    _BINARY_DOC_EXT = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx",
                       ".ppt", ".odt", ".rtf"}

    def _seed_attachments(self, subdir, attachments):
        """Write user-provided files into the mission dir so the agent's
        read_file / list_files (scoped to ``missions/<slug>/``) can see them.

        Uses the SHARED ingestion (``rune.agentic.ingest``) — the same
        captioner-vs-native-pixels decision as the chat: docs → extracted
        text; images → caption text if the brain is text-only, or kept as
        native pixels if the loaded model is multimodal (Gemma 4…).

        Returns ``(names, note)`` and stores any native images on
        ``self._pending_images`` for the step prompt to pass through.
        """
        self._pending_images = []
        if not attachments or self.workspace is None:
            return [], ""
        from rune.agentic.ingest import ingest_attachment
        # Le cerveau voit-il les pixels ? (Gemma 4 & co.) Sinon captioner.
        worker = None
        try:
            worker = self.pool.pick(needs_prefix=True)
        except Exception:  # noqa: BLE001
            worker = None
        native_mm = bool(getattr(getattr(worker, "model", None),
                                 "is_natively_multimodal", False))
        cap = getattr(self.hippocampe, "captioner", None)
        caption_fn = None
        if cap is not None and not native_mm:
            def caption_fn(img):  # noqa: ANN001
                try:
                    return cap.caption(img) if cap.ensure_loaded() else ""
                except Exception:  # noqa: BLE001
                    return ""
        written: list[str] = []
        notes: list[str] = []
        for att in attachments:
            try:
                res = ingest_attachment(att, native_multimodal=native_mm,
                                        caption_fn=caption_fn)
                if res is None:
                    continue
                if res.text is not None:
                    self.workspace.write_text_file(
                        f"{subdir}/{res.filename}", res.text)
                    written.append(res.filename)
                if res.pil_image is not None:
                    self._pending_images.append(res.pil_image)
                if res.note:
                    notes.append(res.note)
            except Exception:  # noqa: BLE001
                log.debug("attachment seed failed", exc_info=True)
        if not written and not self._pending_images:
            return [], ""
        bits = []
        if written:
            bits.append("Fichiers fournis, DÉJÀ présents dans la mission "
                        "(lis-les avec read_file avant d'agir) : "
                        + ", ".join(written) + ".")
        if self._pending_images:
            bits.append(f"{len(self._pending_images)} image(s) jointe(s) que "
                        "tu PERÇOIS directement (vision native).")
        return written, " ".join(bits)

    def _recall_lessons(self, task: str, limit: int = 3) -> str:
        """Pull relevant past lessons (trigger→approach) from the SHARED
        procedural memory — the same store the chat uses.

        Prefers SEMANTIC recall: cosine on trigger embeddings, reusing the
        chat retriever's embedder, so a lesson matches by *meaning* even when
        it shares no words with the task. Falls back to keyword overlap when
        no embedder is available (tests, embedder off)."""
        store = getattr(self.hippocampe, "procedural_store", None)
        if store is None:
            return ""
        try:
            procs = store.active()[:80]
            if not procs:
                return ""
            top = self._recall_semantic(task, procs, limit)
            if top is None:                     # no embedder → lexical fallback
                top = self._recall_keyword(task, procs, limit)
            if not top:
                return ""
            return "\n".join(f"- {p.trigger} → {p.approach}" for p in top)
        except Exception:  # noqa: BLE001
            log.debug("recall lessons failed", exc_info=True)
            return ""

    def _embedder(self):
        """The chat retriever's embedding callable (text → tensor), or None."""
        r = getattr(self.hippocampe, "retriever", None)
        return getattr(r, "embedder", None) if r is not None else None

    def _embed(self, text: str):
        """Cached embedding tensor for ``text`` (deterministic per text), or
        None when no embedder / failure."""
        emb_fn = self._embedder()
        if emb_fn is None or not text:
            return None
        cached = self._trigger_emb_cache.get(text)
        if cached is not None:
            return cached
        try:
            v = emb_fn(text)
        except Exception:  # noqa: BLE001
            return None
        if v is None:
            return None
        if len(self._trigger_emb_cache) > 600:  # bounded; triggers are short
            self._trigger_emb_cache.clear()
        self._trigger_emb_cache[text] = v
        return v

    def _recall_semantic(self, task, procs, limit):
        """Rank procedures by cosine of their trigger to the task. Returns a
        list of procedures, or None when no embedder is available so the
        caller can fall back to keyword recall."""
        if self._embedder() is None:
            return None
        try:
            import torch
        except Exception:  # noqa: BLE001
            return None
        q = self._embed(task)
        if q is None:
            return None
        q = q.view(1, -1)
        scored = []
        for p in procs:
            d = self._embed(getattr(p, "trigger", "") or "")
            if d is None:
                continue
            try:
                cos = torch.nn.functional.cosine_similarity(q, d.view(1, -1)).item()
            except Exception:  # noqa: BLE001
                continue
            if cos >= self._RECALL_MIN_COS:
                scored.append((cos, getattr(p, "utility_score", 0.0), p))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        return [p for _c, _u, p in scored[:limit]]

    def _recall_keyword(self, task, procs, limit):
        """Lexical fallback: keyword overlap on the trigger."""
        toks = set(re.findall(r"\w{4,}", (task or "").lower()))
        if not toks:
            return []
        scored = []
        for p in procs:
            pt = set(re.findall(r"\w{4,}", (getattr(p, "trigger", "") or "").lower()))
            ov = len(toks & pt)
            if ov:
                scored.append((ov, getattr(p, "utility_score", 0.0), p))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        return [p for _o, _u, p in scored[:limit]]

    def _trigger_similarity(self, a: str, b: str) -> float:
        """Cosine similarity 0–1 between two trigger texts (for store dedup).
        Returns 0.0 when no embedder, so dedup degrades to exact-match."""
        try:
            import torch
        except Exception:  # noqa: BLE001
            return 0.0
        ea, eb = self._embed(a), self._embed(b)
        if ea is None or eb is None:
            return 0.0
        try:
            return float(torch.nn.functional.cosine_similarity(
                ea.view(1, -1), eb.view(1, -1)).item())
        except Exception:  # noqa: BLE001
            return 0.0

    def _skills(self) -> SkillLibrary | None:
        """Lazily build the skill library from configured + bundled dirs.

        Dirs default to the package's ``lythea/skills`` plus ``~/.lythea/skills``
        (user-dropped / vetted imports). Built once; ``None`` if it fails."""
        if self._skill_lib is not None:
            return self._skill_lib
        try:
            dirs = getattr(self.settings, "agent_skills_dirs", None)
            if not dirs:
                pkg_skills = Path(__file__).resolve().parent.parent / "skills"
                dirs = [str(pkg_skills), str(Path.home() / ".lythea" / "skills")]
            self._skill_lib = SkillLibrary(
                dirs, min_score=self._RECALL_MIN_COS,
            )
        except Exception:  # noqa: BLE001
            log.debug("skill library init failed", exc_info=True)
            self._skill_lib = SkillLibrary([])  # empty, harmless
        return self._skill_lib

    # ── Proven-snippet library ───────────────────────────────────────
    # Every GREEN mission archives its module(s); a new similar task gets the
    # proven snippet injected as a vetted starting point. Self-improvement
    # without fine-tuning: the agent's past successes become its few-shots.

    def _snippets_file(self) -> Path:
        d = Path.home() / ".lythea"
        d.mkdir(parents=True, exist_ok=True)
        return d / "agent_snippets.jsonl"

    def _archive_snippet(self, task: str, files: list, read) -> None:
        if not bool(getattr(self.settings, "agent_snippets_enabled", True)):
            return
        try:
            mods = [f["path"] for f in files
                    if f["path"].endswith(".py")
                    and not os.path.basename(f["path"]).startswith("test_")][:2]
            if not mods:
                return
            entry = {"task": task[:400], "ts": time.time(), "files": []}
            for p in mods:
                c = read(p)
                if isinstance(c, str) and c and not c.startswith("[introuvable"):
                    entry["files"].append(
                        {"name": os.path.basename(p), "content": c[:4000]})
            if not entry["files"]:
                return
            fp = self._snippets_file()
            lines = []
            if fp.exists():
                lines = fp.read_text(encoding="utf-8").splitlines()[-199:]
            lines.append(json.dumps(entry, ensure_ascii=False))
            fp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception:  # noqa: BLE001
            log.debug("snippet archive failed", exc_info=True)

    def _recall_snippet(self, task: str) -> str:
        """Best past GREEN module for a similar task (cos ≥ 0.5), as a prompt
        block — or '' when nothing relevant."""
        if not bool(getattr(self.settings, "agent_snippets_enabled", True)):
            return ""
        try:
            fp = self._snippets_file()
            if not fp.exists():
                return ""
            entries = []
            for ln in fp.read_text(encoding="utf-8").splitlines():
                try:
                    entries.append(json.loads(ln))
                except Exception:  # noqa: BLE001
                    continue
            if not entries:
                return ""
            qv = self._embed(task)
            best, best_s = None, 0.0
            for e in entries:
                s = 0.0
                if qv is not None:
                    ev = self._embed(e.get("task", ""))
                    if ev is not None:
                        import torch as _t
                        s = float(_t.nn.functional.cosine_similarity(
                            qv, ev, dim=-1))
                else:  # keyword fallback (no embedder available)
                    a = set(re.findall(r"\w{4,}", task.lower()))
                    b = set(re.findall(r"\w{4,}", e.get("task", "").lower()))
                    s = len(a & b) / max(1, len(a | b))
                if s > best_s:
                    best, best_s = e, s
            if best is None or best_s < 0.5:
                return ""
            f0 = (best.get("files") or [{}])[0]
            return (
                "\n[Système] Mission similaire DÉJÀ RÉUSSIE "
                f"(« {best.get('task', '')[:120]} », similarité {best_s:.2f}). "
                f"Module éprouvé « {f0.get('name', '')} » — adapte-le à la "
                "tâche actuelle, ne le copie pas aveuglément :\n"
                f"```python\n{(f0.get('content', '') or '')[:1000]}\n```\n")
        except Exception:  # noqa: BLE001
            return ""

    # ── Autonomous skill authoring ───────────────────────────────────
    # After a GREEN mission, write a structured SKILL.md (procedure, pitfalls,
    # verification) into the skills dir — the loader picks it up on the next
    # similar task. Closes the learning loop: lessons (1 line) → snippets
    # (code) → skills (full reusable procedure).

    @staticmethod
    def _render_skill_md(name: str, description: str, body: str) -> str:
        """Assemble a SKILL.md (frontmatter + body). Pure → unit-testable."""
        name = re.sub(r"[^a-z0-9-]", "-", name.lower().strip())[:64] or "skill"
        description = (description or "").replace("\n", " ").strip()[:300]
        return (f"---\nname: {name}\ndescription: {description}\n---\n\n"
                + (body or "").strip() + "\n")

    async def _author_skill(self, worker, task: str, history: str,
                            slug: str, files: list, read) -> str:
        """Write the SKILL.md for a solved mission; returns its path or ''."""
        if not bool(getattr(self.settings, "agent_skill_writing", True)):
            return ""
        try:
            dirs = list(getattr(self.settings, "agent_skills_dirs", []) or [])
            base = Path(dirs[0]) if dirs else (Path.home() / ".lythea" / "skills")
            sdir = base / re.sub(r"[^a-z0-9-]", "-", slug.lower())[:48]
            if (sdir / "SKILL.md").exists():
                return ""                       # don't overwrite a proven skill
            mods = [f["path"] for f in files
                    if f["path"].endswith(".py")
                    and not os.path.basename(f["path"]).startswith("test_")][:1]
            _src = read(mods[0]) if mods else ""
            prompt = (
                f"Mission RÉUSSIE : {task}\n\n"
                f"Trace (extraits) :\n{history[-3000:]}\n\n"
                "Rédige une fiche de compétence RÉUTILISABLE pour ce TYPE de "
                "tâche (pas cette instance) :\n"
                "1) une ligne de description (quand utiliser cette skill),\n"
                "2) la procédure pas-à-pas (5-8 étapes),\n"
                "3) les pièges rencontrés et comment les éviter,\n"
                "4) comment vérifier que c'est réussi.\n"
                "Markdown sobre, en français, max 400 mots. Pas de code "
                "complet, des principes.\nFiche :")
            body = await self._gen(worker, prompt, max_new_tokens=600,
                                   temperature=0.3)
            body = _strip_think(body)
            if len(body) < 80:
                return ""                       # too thin to be a skill
            if _src:
                body += ("\n\n## Exemple éprouvé (extrait)\n```python\n"
                         + str(_src)[:800] + "\n```\n")
            desc_m = re.search(r"^[^\n#].{20,200}", body, flags=re.M)
            desc = desc_m.group(0) if desc_m else task[:200]
            sdir.mkdir(parents=True, exist_ok=True)
            (sdir / "SKILL.md").write_text(
                self._render_skill_md(slug, desc, body), encoding="utf-8")
            return str(sdir / "SKILL.md")
        except Exception:  # noqa: BLE001
            log.debug("skill authoring failed", exc_info=True)
            return ""

    def _recall_skills(self, task: str, limit: int = 2) -> str:
        """Surface the most relevant SKILL.md capabilities for the task,
        ranked semantically (same embedder as lessons) with keyword fallback."""
        lib = self._skills()
        if lib is None or not lib.all():
            return ""
        sim = self._trigger_similarity if self._embedder() is not None else None
        try:
            hits = lib.retrieve(task, limit=limit, similarity_fn=sim)
            return lib.render(hits)
        except Exception:  # noqa: BLE001
            log.debug("recall skills failed", exc_info=True)
            return ""

    async def _learn_lesson(self, worker, task, mode, history, exec_last, slug):
        """Reflexion: distill ONE reusable lesson (trigger→approach) from a run
        that hit errors, and persist it to the SHARED procedural store, tagged
        with provenance so it stays auditable/prunable."""
        store = getattr(self.hippocampe, "procedural_store", None)
        if store is None:
            return None
        verdict = ("" if exec_last is None else
                   ("résolu (tests verts)" if exec_last.get("ok")
                    else f"NON résolu ({exec_last.get('summary')})"))
        prompt = (
            "À partir de cette mission qui a rencontré des erreurs, formule UNE "
            "leçon réutilisable pour l'avenir, au format STRICT :\n"
            "TRIGGER: <quand cette situation se présente, ≤ 160 caractères>\n"
            "APPROACH: <quoi faire / quoi éviter concrètement, ≤ 260 caractères>\n\n"
            f"Tâche ({mode}) : {task}\n"
            f"Issue : {verdict}\n"
            f"Déroulé (erreurs rencontrées et corrections) :\n{history[-3500:]}\n\n"
            "Donne SEULEMENT les deux lignes TRIGGER/APPROACH, sans autre texte."
        )
        try:
            out = await self._gen(worker, prompt, max_new_tokens=220)
            out = _strip_think(out)
            out = re.sub(r"<tool_call>.*?</tool_call>", "", out, flags=re.S)
            mt = re.search(r"(?im)^\s*TRIGGER\s*:\s*(.+)$", out)
            ma = re.search(r"(?im)^\s*APPROACH\s*:\s*(.+)$", out)
            if not mt or not ma:
                return None
            trigger = mt.group(1).strip()[:200]
            approach = ma.group(1).strip()[:300]
            if not trigger or not approach:
                return None
            proc = store.add(
                trigger=trigger, approach=approach,
                confidence=0.45,
                source_episodes=[f"agent:{slug}"],
                similarity_check=self._trigger_similarity,
            )
            try:
                store.save()
            except Exception:  # noqa: BLE001
                log.debug("procedural store save failed", exc_info=True)
            return {"trigger": trigger, "approach": approach} if proc else None
        except Exception:  # noqa: BLE001
            log.exception("learn lesson failed")
            return None

    async def _critic_review(self, worker, task, mode, files_total,
                             exec_last, answer, abs_dir):
        """A separate, demanding review pass over a candidate solution.

        Returns ``(approved, feedback)``. ``approved`` True ⇒ let the run
        finish; otherwise ``feedback`` is concrete, actionable correction
        text to feed back into the loop. Cheap: one bounded generation.
        """
        file_list = ", ".join(f.get("path", "") for f in files_total) or "aucun"
        snippets = ""
        if mode == "build" and abs_dir:
            parts = []
            for f in files_total[:6]:
                rel = str(f.get("path", "")).split("/", )[-1]
                ap = self._mission_abs(f.get("path", "")) if f.get("path") else None
                try:
                    if ap and Path(ap).is_file():
                        parts.append(f"### {rel}\n"
                                     + Path(ap).read_text(encoding="utf-8",
                                                          errors="replace")[:1500])
                except Exception:  # noqa: BLE001
                    pass
            snippets = "\n\n".join(parts)
        verdict = ("" if exec_last is None else
                   ("Tests : OK." if exec_last.get("ok")
                    else f"Tests EN ÉCHEC : {exec_last.get('summary')}."))
        if mode == "build":
            crit = ("Les tests couvrent-ils les cas limites PERTINENTS "
                    "(vides/nuls, bornes, erreurs attendues) ou sont-ils "
                    "triviaux ? Le module est-il correct et complet ?")
        elif mode == "analyze":
            crit = ("Le résultat répond-il vraiment à la question, est-il "
                    "plausible et chiffré ? La méthode de calcul est-elle juste ?")
        elif mode == "research":
            crit = ("La synthèse répond-elle à la question, est-elle "
                    "structurée et APPUYÉE SUR DES SOURCES (URLs) ? "
                    "(Les sources/URLs sont REQUISES et BIENVENUES — ne "
                    "demande jamais de les retirer.)")
        else:
            crit = ("La réponse est-elle FONDÉE sur les fichiers fournis, "
                    "complète et exacte (pas d'invention) ?")
        # Le rappel « Fichiers produits » n'a de sens qu'en build/analyze (où
        # l'agent écrit des fichiers). En research/answer le livrable est du
        # TEXTE : mentionner « fichiers produits : aucun » fait délirer le
        # relecteur (il croit qu'il faut supprimer les sources). On l'omet.
        _files_line = (f"Fichiers produits : {file_list}\n"
                       if mode in ("build", "analyze") else "")
        # L'instruction de correction parle de fichiers/tests en build/analyze,
        # de rédaction/sources en research/answer.
        if mode in ("build", "analyze"):
            _fix_hint = ("quel fichier, quel test à ajouter, quoi corriger")
        else:
            _fix_hint = ("quoi préciser, quelle source/URL à citer, quelle "
                         "section à compléter — SANS écrire de fichier")
        prompt = (
            "Tu es un relecteur critique et exigeant d'un travail d'agent.\n"
            f"Tâche confiée : {task}\n"
            f"{_files_line}"
            f"{verdict}\n"
            + (f"\n{snippets}\n" if snippets else "")
            + f"\nRéponse/synthèse proposée : {answer[:1200]}\n\n"
            f"Évalue : {crit}\n"
            "Si le travail répond vraiment et complètement à la tâche, réponds "
            "EXACTEMENT « OK ». Sinon, réponds « CORRIGER : » suivi d'1 à 3 "
            f"actions CONCRÈTES et précises ({_fix_hint}). "
            "Sois bref, pas de bla-bla.\nRelecteur :"
        )
        try:
            # Budget serré : un verdict « OK » ou « CORRIGER : 1-3 actions »
            # tient largement. 400 laissait un modèle hybride (Qwen3…) remplir
            # l'espace en répétant « Réponse : OK » jusqu'à la troncature.
            out = await self._gen(worker, prompt, max_new_tokens=200, prose=True)
        except Exception:  # noqa: BLE001
            log.exception("critic review failed")
            return True, ""   # fail-open: never block a finish on critic error
        out = _strip_think(out)
        out = re.sub(r"<tool_call>.*?</tool_call>", "", out, flags=re.S).strip()
        # Garde-fou anti-boucle : un modèle peut dégénérer en répétant la même
        # ligne (« Réponse : OK\nRéponse : OK\n… ») au lieu de répondre « OK »
        # une fois. On détecte cette répétition dégénérée et on la traite comme
        # une approbation molle (fail-open) plutôt que de la propager comme
        # feedback de correction — ce qui relançait l'agent à tort.
        # Si les tests sont CONNUS en échec, aucune approbation (molle ou
        # explicite) ne doit clore la mission : un critic qui dégénère en
        # répétition ou répond « OK » à tort ne peut pas valider du rouge.
        # On renvoie alors une correction concrète → l'agent persévère.
        # (En research/answer, exec_last vaut None → comportement inchangé.)
        _tests_failed = exec_last is not None and not exec_last.get("ok")
        _force_fix = ("Les tests échouent encore — lis le message d'erreur "
                      "et corrige le module pour qu'ils passent.")
        _lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        _uniq = set(_lines)
        if len(_lines) >= 4 and len(_uniq) <= 2:
            if not any("CORRIGER" in ln.upper() for ln in _uniq):
                return (False, _force_fix) if _tests_failed else (True, "")
            out = next(iter(_uniq))  # dé-duplique avant de critiquer
        up = out.upper()
        # Approbation tolérante : pas de « CORRIGER » et un marqueur OK dans
        # l'en-tête. Gère « Réponse : OK », « La réponse est OK », etc. — que
        # le ^OK strict ratait (→ faux négatif → boucle propagée).
        if "CORRIGER" not in up and re.search(
                r"\b(OK|VALID|APPROUV|RIEN|CORRECT|COMPLET)", up[:80]):
            return (False, _force_fix) if _tests_failed else (True, "")
        # Strip a leading "CORRIGER :" (ou « Réponse : CORRIGER ») label.
        fb = re.sub(r"(?is)^\s*(r[ée]ponse\s*:?\s*)?corriger\s*:?\s*", "",
                    out).strip()
        fb = re.sub(r"\n{3,}", "\n\n", fb)[:800]
        return (False, fb) if fb else (True, "")

    async def _decompose(self, worker, task: str) -> list[dict]:
        """Decomposition judge (inspired by DeerFlow's planner → subtasks, but
        adapted to a SINGLE local model): decide whether the build task splits
        into TRULY INDEPENDENT subtasks (disjoint files, no cross-import at
        build time). Returns a list of subtask dicts, or [] when the task is
        atomic / the judge is unsure → the caller falls back to the normal
        single-agent ReAct loop. Conservative by design: a 14B is mediocre at
        module integration, so we only split on clear-cut cases."""
        if not bool(getattr(self.settings, "agent_multiagent", True)):
            return []
        prompt = (
            "Tu es un planificateur. Décide si cette tâche de DÉVELOPPEMENT se "
            "découpe en sous-tâches RÉELLEMENT INDÉPENDANTES — fichiers "
            "DISJOINTS, aucune dépendance d'import entre elles au moment de "
            "l'écriture (l'intégration vient après). Ne découpe QUE si c'est "
            "franc et qu'il y a au moins 2 modules distincts.\n"
            "Réponds en JSON STRICT, rien d'autre :\n"
            '{"decompose": true|false, "subtasks": ['
            '{"title": "...", "files": ["x.py", "test_x.py"], '
            '"task": "consigne précise et autonome"}]}\n'
            "Si tâche atomique (un seul module) ou si tu hésites : "
            '{"decompose": false, "subtasks": []}.\n\n'
            f"Tâche : {task}\n\nJSON :")
        try:
            raw = await self._gen(worker, prompt, temperature=0.1,
                                  max_new_tokens=400)
        except Exception:  # noqa: BLE001
            return []
        m = re.search(r"\{.*\}", raw or "", flags=re.S)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except Exception:  # noqa: BLE001
            return []
        if not data.get("decompose"):
            return []
        subs = data.get("subtasks") or []
        clean, seen_files = [], set()
        for s in subs:
            t = (s.get("task") or "").strip()
            files = [os.path.basename(str(f)) for f in (s.get("files") or [])]
            if not t or not files:
                continue
            if seen_files & set(files):        # overlap → not independent
                return []
            seen_files.update(files)
            clean.append({"title": (s.get("title") or t)[:60],
                          "task": t, "files": files})
        # Cap to the hardware's subagent width; need ≥2 to be worth it.
        cap = max(1, int(self._hw_knobs().get("subagents", 2)))
        return clean[:cap] if len(clean) >= 2 else []

    @staticmethod
    def _compose_subtask_plan(task: str, subs: list[dict]) -> str:
        """Render the decomposition as a guiding block prepended to the build
        prompt — the model tackles ONE subtask's files at a time, then a final
        integration pass runs the whole test suite."""
        lines = [f"  {i+1}. {s['title']} → {', '.join(s['files'])} : {s['task']}"
                 for i, s in enumerate(subs)]
        return (
            "\n[Plan multi-modules] Cette mission a été découpée en sous-parties "
            "INDÉPENDANTES. Traite-les UNE PAR UNE (écris ses fichiers + ses "
            "tests, fais-les passer au vert) avant de passer à la suivante :\n"
            + "\n".join(lines)
            + "\nQuand toutes sont vertes, lance run_tests une dernière fois "
            "sur l'ENSEMBLE (phase d'intégration) et corrige les éventuels "
            "conflits d'interface entre modules.\n")

    async def _react(self, task: str, run_id: str, run, core, attachments=None):
        from rune.agentic import tools as T

        name = await self._gen_name(core, task)
        slug = self._unique_slug(_slugify(name))
        subdir = f"missions/{slug}"
        abs_dir = self._mission_abs(subdir)
        # Tableau noir de mission : carnet de bord persistant et anti-amnésie.
        # En mono-agent, une seule section ("agent") = ce qui a marché / ce qui
        # a échoué et POURQUOI, relu à chaque étape pour ne pas rabâcher.
        from rune.agentic.blackboard import MissionBlackboard
        bb = MissionBlackboard.load(Path(abs_dir) / "blackboard.json")
        bb.ensure("agent", goal=task[:240])
        # Routage généraliste (heuristique pure, zéro génération) : type de
        # tâche, niveau de complexité, forme de livrable. Pilote la profondeur
        # du plan initial et le vérificateur (code = pytest ; sinon texte).
        from rune.agentic.router import route as _route_task
        route = _route_task(task)
        bb.note("agent", f"route: {route.kind}/L{route.level}/{route.deliverable}")
        self._write_manifest(abs_dir, {
            "name": name, "slug": slug, "task": task, "status": "running",
            "mode": "react",
        })
        # User-provided files → dropped into the mission dir so read_file sees
        # them (the agent is scoped to missions/<slug>/, not the workspace root).
        seed_names, seed_note = self._seed_attachments(subdir, attachments)
        yield {
            "type": "run_start", "run_id": run_id, "task": task, "name": name,
            "slug": slug, "subdir": subdir, "mode": "react",
            "attachments": seed_names,
            "workers": self.pool.available_names(),
        }

        files_total: list[dict] = []
        exec_holder: dict = {"last": None}
        seen: dict = {}

        def _rel(path):
            """Mission-relative path, JAILED to the mission dir. Strips an
            echoed 'missions/<slug>/…' prefix and rejects any '..'/absolute
            escape so an agent in one mission can't reach another (returns ''
            → callers treat it as an invalid path)."""
            return _jail_relpath(path, subdir)

        def op_list():
            return self._existing_files(abs_dir)

        def op_read(path):
            rel = _rel(path)
            ap = self._mission_abs(f"{subdir}/{rel}") if rel else None
            try:
                if ap and Path(ap).exists() and Path(ap).is_file():
                    return Path(ap).read_text(encoding="utf-8", errors="replace")[:6000]
            except Exception:  # noqa: BLE001
                pass
            return f"[introuvable: {rel or path}]"

        def op_write(path, content):
            relp = _rel(path)
            if not relp:
                return {"ok": False, "error": "chemin de fichier manquant"}
            _bn = os.path.basename(relp).lower()
            # Garde système : ne JAMAIS écraser les fichiers internes de mission
            # (le modèle confond parfois « lire l'état » et « écrire dedans »).
            if _bn in ("blackboard.json", ".lythea-mission.json") \
                    or _bn.startswith(".lythea"):
                return {"ok": False, "error": (
                    "fichier système protégé — n'écris pas dedans. Écris ton "
                    "module/script avec un autre nom.")}
            # Garde anti-tool_call : si le « contenu » est en fait un appel
            # d'outil JSON (le modèle a collé son tool_call comme contenu), on
            # REFUSE — sinon on écrirait du JSON brut à la place du code.
            _cs = (content or "").strip()
            if (_cs.startswith("{") and '"name"' in _cs
                    and '"arguments"' in _cs
                    and re.search(r'"name"\s*:\s*"(write_file|read_file|'
                                  r'edit_file|run_tests|run_command|run_python|'
                                  r'list_files|web_search|web_fetch|finish)"',
                                  _cs)):
                return {"ok": False, "error": (
                    "le contenu ressemble à un appel d'outil JSON, pas à du "
                    "code. Renvoie le CODE du fichier (pas le tool_call) dans "
                    "« content ».")}
            # Deterministically refuse scaffold (setup.py/README/pyproject/…)
            # on simple tasks — the 7B ignores the prompt and spams these.
            if (os.path.basename(relp).lower() in _SCAFFOLD
                    and not _PACKAGE_HINT.search(task)):
                _only = ("le module et son fichier de tests « test_<module>.py »"
                         if mode == "build"
                         else "le fichier de code utile à la tâche")
                return {"ok": False, "error": (
                    f"fichier scaffold refusé ({os.path.basename(relp)}). "
                    f"Écris seulement {_only}.")}
            # Refuse a synthesis/answer text-file dump — the final synthesis is
            # prose in the reply (no tool_call), never a file.
            _base, _ext = os.path.splitext(os.path.basename(relp))
            if _ext.lower() in _NARRATIVE_EXT and _NARRATIVE_FILE_RE.search(_base):
                return {"ok": False, "error": (
                    "n'écris pas de fichier de synthèse. Ta synthèse finale est "
                    "du TEXTE dans ta réponse, SANS tool_call. Si les tests "
                    "échouent, corrige plutôt le module ou test_<module>.py.")}
            key = f"{subdir}/{relp}"
            if seen.get(key) == hash(content):
                return {"path": relp, "size": len(content), "unchanged": True}
            # Syntax gate: refuse unparsable Python BEFORE writing — saves a
            # full run_tests cycle per syntax slip (e.g. IndentationError).
            if relp.endswith(".py"):
                err = _py_syntax_error(content)
                if err:
                    return {"ok": False, "error": (
                        f"code Python invalide, écriture refusée — {err}. "
                        "Corrige la syntaxe et renvoie le fichier complet.")}
            _old = ""
            _ap0 = self._mission_abs(key)
            if _ap0 is not None and Path(_ap0).is_file():
                _old = Path(_ap0).read_text(encoding="utf-8", errors="replace")
            entry = self.workspace.write_text_file(key, content)
            seen[key] = hash(content)
            files_total.append({"path": entry.path, "size": entry.size})
            out = {"path": relp, "size": entry.size}
            if _old:
                _d = _unified_diff(_old, content, relp)
                out["diff"] = _d if _d else "(aucun changement réel)"
            # Report the MISSION-RELATIVE path so the model reuses it as-is.
            return out

        sandbox = None
        if self.execution_enabled and abs_dir is not None:
            sandbox = self._make_sandbox(abs_dir)

        def op_tests():
            if sandbox is None:
                return {"ok": False, "summary": "exécution désactivée"}
            # No test file → don't run pytest (it would just say "no tests
            # ran" and the model loops). Tell it plainly to write one first.
            files_here = self._existing_files(abs_dir)
            _py = [f for f in files_here if f.endswith(".py")]
            has_test = any(
                os.path.basename(f).startswith("test_")
                or os.path.basename(f).endswith("_test.py")
                for f in _py
            )
            has_module = any(
                not os.path.basename(f).startswith("test_")
                and not os.path.basename(f).endswith("_test.py")
                for f in _py
            )
            if not has_test or not has_module:
                # Message CIBLÉ sur ce qui manque réellement — sinon le modèle
                # boucle : on lui réclamait « un fichier de test » alors qu'il
                # en avait écrit un mais PAS le module (observé sur fibo, 8B :
                # 4× write_file(test_fibonacci.py), jamais fibonacci.py).
                if has_test and not has_module:
                    _msg = ("Tu as écrit le fichier de test mais PAS le module "
                            "qu'il importe. Écris le module correspondant "
                            "avec write_file MAINTENANT, puis run_tests.")
                elif has_module and not has_test:
                    _msg = ("Tu as écrit le module mais aucun fichier de test. "
                            "Écris « test_<module>.py » avec write_file, puis "
                            "run_tests.")
                else:
                    _msg = ("Aucun fichier Python. Écris le module ET "
                            "« test_<module>.py » avec write_file, puis "
                            "run_tests.")
                return {"ok": False, "summary": _msg, "no_test_file": True}
            from rune.agentic.sandbox import (  # noqa: PLC0415
                _STDLIB, module_to_pypi, parse_missing_module,
            )
            res = sandbox.run_pytest()
            installs: list[dict] = []
            guard = 0
            while not res.ok and guard < sandbox.max_installs:
                guard += 1
                mod = parse_missing_module(res.stdout + "\n" + res.stderr)
                if not mod or mod in _STDLIB:
                    break
                ok_i, _ir = sandbox.pip_install(mod)
                installs.append({"package": module_to_pypi(mod), "ok": bool(ok_i)})
                if not ok_i:
                    break
                res = sandbox.run_pytest()
            d = res.to_dict()
            d["installs"] = installs
            d["isolation"] = _sandbox_mode()
            exec_holder["last"] = d
            return d

        def op_command(argv, net):
            if sandbox is None:
                return {"ok": False, "summary": "exécution désactivée"}
            argv = list(argv or [])
            # En mode NON-build, le modèle (surtout non-thinking) lance souvent
            # « pytest <script> » par réflexe alors qu'il n'y a aucun test et que
            # le livrable est un résultat. On le réécrit en « python <script> »
            # pour qu'il obtienne réellement la sortie au lieu d'un échec pytest.
            if not build_mode and argv and "pytest" in argv[0]:
                rest = [a for a in argv[1:] if a.endswith(".py")]
                if rest:
                    argv = ["python", rest[0]]
                else:
                    return {"ok": False, "summary": (
                        "Pas de tests ici : EXÉCUTE ton script avec "
                        "« python <fichier>.py » (pas pytest) pour obtenir le "
                        "résultat, puis donne ta synthèse.")}
            res = sandbox.run_argv(
                argv, net=net,
                timeout=int(getattr(self.settings, "agent_cmd_timeout_s", 120)),
            )
            return res.to_dict()

        def op_edit(path, find, replace):
            relp = _rel(path)
            ap = self._mission_abs(f"{subdir}/{relp}") if relp else None
            if not ap or not Path(ap).exists() or not Path(ap).is_file():
                return {"ok": False, "error": f"introuvable: {relp or path}"}
            txt = Path(ap).read_text(encoding="utf-8", errors="replace")
            if find not in txt:
                return {"ok": False, "error": "motif 'find' introuvable"}
            n = txt.count(find)
            new = txt.replace(find, replace)
            if relp.endswith(".py"):
                err = _py_syntax_error(new)
                if err:
                    return {"ok": False, "error": (
                        f"l'édition produirait du Python invalide — {err}. "
                        "Édition refusée ; réécris le fichier complet corrigé "
                        "avec write_file.")}
            entry = self.workspace.write_text_file(f"{subdir}/{relp}", new)
            seen[f"{subdir}/{relp}"] = hash(new)
            _d = _unified_diff(txt, new, relp)
            return {"ok": True, "replaced": n, "path": relp, "size": entry.size,
                    "diff": _d if _d else "(aucun changement réel)"}

        def op_serve(argv, paths, port):
            if sandbox is None:
                return {"ok": False, "summary": "exécution désactivée"}
            return sandbox.serve_and_probe(
                argv, paths, port=port,
                boot_timeout=int(getattr(self.settings, "agent_serve_timeout_s", 12)),
            )

        def op_delete(path):
            relp = _rel(path)
            ap = self._mission_abs(f"{subdir}/{relp}") if relp else None
            if not ap or not Path(ap).exists() or not Path(ap).is_file():
                return {"ok": False, "error": f"introuvable: {relp or path}"}
            try:
                Path(ap).unlink()
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "error": str(exc)}
            key = f"{subdir}/{relp}"
            seen.pop(key, None)
            files_total[:] = [f for f in files_total if f.get("path") != key]
            return {"ok": True, "deleted": relp}

        _SEARCH_IGNORE = {".venv", "venv", "__pycache__", ".pytest_cache",
                          ".git", ".mypy_cache", ".ruff_cache", "node_modules"}

        def op_search(query, glob="*"):
            import fnmatch as _fn
            base = Path(abs_dir) if abs_dir else None
            if base is None or not base.exists():
                return {"ok": True, "matches": [], "count": 0}
            try:
                rx = re.compile(query)
            except re.error:
                rx = None
            matches: list[str] = []
            for p in sorted(base.rglob("*")):
                if not p.is_file():
                    continue
                rel = p.relative_to(base).as_posix()
                if any(seg in _SEARCH_IGNORE for seg in rel.split("/")):
                    continue
                if glob and glob != "*" and not _fn.fnmatch(p.name, glob):
                    continue
                try:
                    txt = p.read_text(encoding="utf-8", errors="ignore")
                except Exception:  # noqa: BLE001
                    continue
                for i, line in enumerate(txt.splitlines(), 1):
                    hit = rx.search(line) if rx else (query in line)
                    if hit:
                        matches.append(f"{rel}:{i}: {line.strip()[:200]}")
                        if len(matches) >= 100:
                            return {"ok": True, "matches": matches,
                                    "count": len(matches), "truncated": True}
            return {"ok": True, "matches": matches, "count": len(matches)}

        def op_python(code):
            if sandbox is None:
                return {"ok": False, "error": "exécution désactivée"}
            res = sandbox.run_argv(
                ["python", "-c", code], net=False,
                timeout=int(getattr(self.settings, "agent_cmd_timeout_s", 120)))
            return res.to_dict() if hasattr(res, "to_dict") else res

        def op_websearch(query, max_results=5):
            # Reuse the chat's web backend (Tavily→Serper→Brave→SearXNG→DDG
            # auto-chain, configured by the same env keys).
            try:
                from rune.web_providers.factory import get_default_provider
                prov = get_default_provider()
                if not prov.is_available():
                    return {"ok": False, "error": (
                        "aucun fournisseur de recherche configuré "
                        "(TAVILY_API_KEY / SERPER_API_KEY / BRAVE_API_KEY…).")}
                hits = prov.search(query, max_results=max_results) or []
                results = [{"title": h.get("title", ""),
                            "url": h.get("href", ""),
                            "extrait": (h.get("body", "") or "")[:500]}
                           for h in hits]
                return {"ok": True, "query": query,
                        "results": results, "count": len(results)}
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "error": str(exc)}

        def op_webfetch(url):
            try:
                _assert_public_url(url)        # anti-SSRF (scheme + host)
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
            try:
                import html as _html
                from urllib.request import (
                    HTTPRedirectHandler, Request, build_opener)

                class _SafeRedirect(HTTPRedirectHandler):
                    def redirect_request(self, req, fp, code, msg, hdrs, newurl):
                        _assert_public_url(newurl)   # re-validate every hop
                        return super().redirect_request(
                            req, fp, code, msg, hdrs, newurl)

                opener = build_opener(_SafeRedirect())
                req = Request(url, headers={
                    "User-Agent": "Lythea-Agent/1.0 (+local)"})
                with opener.open(req, timeout=15) as resp:  # noqa: S310
                    raw = resp.read(2_000_000)
                    ctype = (resp.headers.get("Content-Type") or "").lower()
                    enc = resp.headers.get_content_charset() or "utf-8"
                    final_url = resp.geturl()
                txt = raw.decode(enc, errors="replace")
                if "html" in ctype or "<html" in txt[:2000].lower():
                    txt = re.sub(r"(?is)<(script|style|noscript|template)\b.*?</\1>",
                                 " ", txt)
                    txt = re.sub(r"(?s)<[^>]+>", " ", txt)
                    txt = _html.unescape(txt)
                    txt = re.sub(r"[ \t\r\f]+", " ", txt)
                    txt = re.sub(r"\n\s*\n\s*\n+", "\n\n", txt).strip()
                return {"ok": True, "url": final_url,
                        "content": txt[:6000], "truncated": len(txt) > 6000}
            except ValueError as exc:       # redirect to a blocked host
                return {"ok": False, "error": str(exc)}
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "error": str(exc)}

        def op_recall(query):
            """PULL mémoire (lecture seule) : réutilise le gather() du chat,
            qui assemble KG + MHN + Chroma. Plafonné par budget. Best-effort :
            dégrade en message neutre si la mémoire est indisponible."""
            try:
                rp = getattr(self.hippocampe, "retrieval_phase", None)
                if rp is None or not hasattr(rp, "gather"):
                    return "[mémoire indisponible]"
                ctx = rp.gather(str(query))
                sections = getattr(ctx, "sections", None) or []
                text = "\n\n".join(s for s in sections if s).strip()
                if not text:
                    return "[aucun souvenir pertinent]"
                budget = int(getattr(self.settings,
                                     "agent_memory_recall_budget_chars", 1200))
                return text[:budget]
            except Exception:  # noqa: BLE001
                log.debug("op_recall failed", exc_info=True)
                return "[mémoire indisponible]"

        def op_querykg(question):
            """PULL KG (lecture seule) : faits structurés via
            query_by_question. Best-effort."""
            try:
                kg = getattr(self.hippocampe, "kg", None)
                if kg is None or not hasattr(kg, "query_by_question"):
                    return "[KG indisponible]"
                facts = kg.query_by_question(str(question), [], max_facts=5) or []
                if not facts:
                    return "[aucun fait connu sur ce sujet]"
                return "\n".join(f"- {f}" for f in facts)
            except Exception:  # noqa: BLE001
                log.debug("op_querykg failed", exc_info=True)
                return "[KG indisponible]"

        ops = {"list_files": op_list, "read_file": op_read,
               "write_file": op_write, "run_tests": op_tests,
               "run_command": op_command, "edit_file": op_edit,
               "serve_and_probe": op_serve, "delete_file": op_delete,
               "search_files": op_search, "run_python": op_python,
               "web_search": op_websearch, "web_fetch": op_webfetch}
        # Don't dangle web tooling in front of the model on a non-web task —
        # a weak model wanders (e.g. starting an http.server for an email
        # validator). Expose serve_and_probe only when the task is web-ish.
        web_task = bool(re.search(
            r"\b(web|serveur|server|http|https|flask|fastapi|django|api|"
            r"endpoint|route|html|site|uvicorn|asgi|wsgi|starlette)\b",
            task, re.I))
        if not web_task:
            ops.pop("serve_and_probe", None)

        # Task-type routing: pick the workflow, prune the tools offered, and
        # tailor the method + completion criterion. 'build' is the validated
        # test-driven path (unchanged); 'analyze'/'answer' are lighter modes.
        # Future capabilities (web, email) plug in as new modes here.
        mode = _classify_task(task)
        if mode == "answer":
            for _k in ("write_file", "edit_file", "run_tests", "run_command",
                       "serve_and_probe", "delete_file", "run_python"):
                ops.pop(_k, None)
        elif mode == "research":
            # Recherche web : on garde web_search/web_fetch + read_file, on
            # retire tout l'outillage de code (le livrable est une synthèse).
            for _k in ("write_file", "edit_file", "run_tests", "run_command",
                       "serve_and_probe", "delete_file", "run_python"):
                ops.pop(_k, None)
        elif mode == "analyze":
            # V0.1.1 — on NE retire plus run_tests : un calcul peut
            # légitimement vouloir se vérifier par un test, et le prompt
            # (exemples codés en dur) mentionne run_tests → le retirer
            # faisait appeler un outil « indisponible » et tâtonner
            # l'agent. On ne retire que serve_and_probe (inutile hors
            # service web).
            for _k in ("serve_and_probe",):
                ops.pop(_k, None)

        # Morceau 1 — outils mémoire PULL (lecture seule). Ajoutés APRÈS le
        # pruning par mode pour rester disponibles dans tous les modes (utiles
        # en build comme en research). Derrière deux flags, OFF par défaut.
        _mem_tools_on = (
            bool(getattr(self.settings, "agent_memory_v2_enabled", False))
            and bool(getattr(self.settings, "agent_memory_tools_enabled", False)))
        if _mem_tools_on:
            ops["recall"] = op_recall
            ops["query_kg"] = op_querykg

        _tools_block = (
            "Tu es Rune, ingénieure agentique. Tu disposes de ces outils "
            "(JSON) :\n" + T.tools_prompt(sorted(ops)) + "\n\n"
            "⚠️ POUR ÉCRIRE DU CODE — utilise le format BALISE (PAS le JSON) : "
            "le code va BRUT entre les balises, sans guillemets ni échappement.\n"
            "Créer un fichier :\n"
            "<write_file path=\"module.py\">\n"
            "def exemple():\n"
            "    return 42\n"
            "</write_file>\n"
            "Modifier un fichier existant :\n"
            "<edit_file path=\"module.py\">\n"
            "<<<FIND\n"
            "ancien_texte_exact\n"
            ">>>REPLACE\n"
            "nouveau_texte\n"
            "</edit_file>\n"
            "Ce format évite toute erreur d'échappement.\n"
            "Pour les AUTRES outils (run_tests, run_command, read_file, "
            "list_files…), émets le JSON dans une balise <tools> (ou "
            "<tool_call>, les deux marchent), par exemple :\n"
            "<tools>{\"name\": \"run_tests\", \"arguments\": {}}</tools>\n"
            "<tools>{\"name\": \"read_file\", \"arguments\": {\"path\": "
            "\"module.py\"}}</tools>\n"
            "Un seul outil par tour.\n\n"
            "Méthode (raisonne PUIS agis) :\n"
            "- À chaque tour, écris d'abord « Pensée : … » : analyse le dernier "
            "résultat d'outil (l'observation) — ce qu'il signifie, ce qui reste "
            "à faire — puis décide la prochaine action. ENSUITE une seule balise "
            "<tools>{\"name\": \"...\", \"arguments\": {...}}</tools> "
            "(ou <write_file>/<edit_file> pour le code). "
            "Un seul outil par tour.\n"
            "- Quand la tâche est terminée, appelle l'outil finish avec ta "
            "réponse finale (ou réponds simplement en texte sans balise).\n"
        )
        if mode == "build":
            system = (
                _tools_block
                + "- Ordre normal : write_file le module → write_file "
                "« test_<module>.py » → run_tests → si échec, lis l'erreur, "
                "corrige, relance run_tests.\n"
                "- Avant d'écrire une logique NON TRIVIALE (regex, parsing, "
                "calcul délicat), VÉRIFIE-la d'abord avec run_python : essaie "
                "l'expression sur 2-3 cas (valide + invalide), regarde le "
                "résultat réel, PUIS écris le module avec la version vérifiée.\n"
                "- COMPARAISON DE NOMBRES À VIRGULE : ne teste JAMAIS l'égalité "
                "exacte de flottants (ex. assert calc() == 659.95 échoue pour "
                "659.96). Utilise pytest.approx — « assert calc() == "
                "pytest.approx(659.95, abs=1e-2) » — ou décide d'un arrondi "
                "explicite cohérent entre le module ET le test.\n"
                "- Les tests doivent être SÉRIEUX et au FORMAT pytest : des "
                "fonctions « def test_xxx(): » avec des assert dedans (pytest "
                "collecte QUE les fonctions test_*) — JAMAIS des assert au "
                "niveau module ni un script. Couvre les cas limites PERTINENTS "
                "au problème — entrées vides/nulles, bornes min/max, valeurs "
                "invalides/erreurs attendues, cas particuliers, allers-retours "
                "quand c'est applicable. Plusieurs fonctions de test, pas un "
                "seul cas trivial.\n"
                "- N'écris QUE le module et ses tests. PAS de setup.py, "
                "README, pyproject.toml, requirements ni config (sauf demande "
                "explicite).\n"
                "- « no tests ran » = il manque le fichier test_<module>.py ; "
                "écris-le, ne touche pas à la config. Une erreur de collecte = "
                "un import faux : aligne les noms entre test et module.\n"
                "- Quand la tâche est faite ET que run_tests réussit, donne ta "
                "synthèse finale en texte clair (3-5 phrases) DIRECTEMENT dans "
                "ta réponse, SANS tool_call et SANS écrire de fichier de "
                "synthèse. N'annonce ce succès qu'APRÈS un run_tests vert."
            )
        elif mode == "analyze":
            system = (
                _tools_block
                + "- C'est une tâche d'ANALYSE / de CALCUL : le livrable est le "
                "RÉSULTAT, pas un module testé.\n"
                "- Si des fichiers de données sont fournis, lis-les d'abord "
                "avec read_file pour connaître leur format (colonnes, "
                "séparateur, en-têtes).\n"
                "- Écris un script Python qui calcule et IMPRIME le résultat "
                "(print), puis EXÉCUTE-le avec run_command, ex. : "
                "<tool_call>{\"name\": \"run_command\", \"arguments\": {\"argv\": "
                "[\"python\", \"analyse.py\"]}}</tool_call>. Lis la sortie.\n"
                "- N'écris PAS de tests pytest et n'utilise PAS run_tests : ce "
                "n'est pas demandé.\n"
                "- Une fois le résultat obtenu, donne ta synthèse finale en "
                "texte clair (chiffres à l'appui) DIRECTEMENT dans ta réponse, "
                "SANS tool_call."
            )
        elif mode == "research":
            system = (
                _tools_block
                + "- C'est une tâche de RECHERCHE WEB : le livrable est une "
                "SYNTHÈSE sourcée. Aucun fichier, aucun code, aucun test.\n"
                "- Utilise web_search pour trouver des sources, puis web_fetch "
                "pour lire les pages pertinentes en détail. Croise plusieurs "
                "sources.\n"
                "- Cite TES SOURCES (URLs) pour chaque point important.\n"
                "- Une fois assez d'informations réunies (2-3 sources "
                "suffisent), rédige ta synthèse finale structurée DIRECTEMENT "
                "en texte clair, SANS tool_call. Écris-la UNE SEULE FOIS, de "
                "façon concise — ne te répète pas. N'écris NI fichier NI code "
                "NI test."
            )
        else:  # answer
            system = (
                _tools_block
                + "- C'est une tâche de SYNTHÈSE / RÉPONSE : le livrable est du "
                "TEXTE, aucun fichier ni test.\n"
                "- Si des fichiers sont fournis, lis-les avec read_file avant "
                "de répondre (ne devine pas leur contenu).\n"
                "- Puis réponds DIRECTEMENT en texte clair et structuré, SANS "
                "tool_call et SANS écrire de fichier. N'écris ni code ni tests."
            )
        history = f"Tâche : {task}\n"
        if seed_note:
            history += seed_note + "\n"
        learn_enabled = bool(getattr(self.settings, "agent_learn_from_errors", True))
        had_failures = False     # ≥1 real failure (→ worth learning a lesson)
        if learn_enabled:
            _lessons = self._recall_lessons(task)
            if _lessons:
                history += ("Leçons de missions passées (tiens-en compte, "
                            "évite ces erreurs) :\n" + _lessons + "\n")
        if bool(getattr(self.settings, "agent_skills_enabled", True)):
            _skills = self._recall_skills(task)
            if _skills:
                history += ("Compétences pertinentes (procédures de référence, "
                            "applique-les si utile) :\n" + _skills + "\n")
            _snip = self._recall_snippet(task)
            if _snip:
                history += _snip
        worker = self.pool.pick(needs_prefix=True)  # core reasoning
        max_iters = int(getattr(self.settings, "agent_react_max_iters", 24))
        no_call_streak = 0
        tests_no_write = 0   # run_tests calls since the last file write
        writes_since_test = 0  # write/edit calls since the last run_tests
        forced_runs = 0      # auto run_tests triggered by write-spam
        red_streak = 0       # tests rouges CONSÉCUTIFS (spécifique code) :
                             # alimente la délibération code (@2 rouges) et le
                             # statut blackboard. COMPLÉMENTAIRE de progress :
                             # progress.stalled = tours sans progrès TOUS types
                             # de tâche (pilote l'escalade généraliste et l'arrêt
                             # anti-boucle) ; red_streak = signal code-spécifique
                             # plus fin. Ne PAS fusionner : granularités distinctes.
        # Suivi de progression GÉNÉRALISTE (remplace progressivement le maquis
        # red_streak code-centré) : compte les tours SANS PROGRÈS, agnostique
        # au type de tâche. Branché en parallèle pour l'instant ; l'escalade
        # lira ``progress.stalled`` au lieu de chaînes de rouges consécutifs.
        from rune.agentic.progress import ProgressTracker
        progress = ProgressTracker(kind=getattr(route, "kind", "code"))
        last_failed = None   # failed-test count of the previous red (progress metric)
        did_websearch = False  # web-on-block fired (once per run)
        web_calls = 0          # model-initiated web_search/fetch count (capped)
        _web_capped = False    # research/answer : plafond atteint → ne plus re-bloquer la synthèse
        _WEB_CALL_MAX = 3      # build : au-delà, on renvoie le modèle au code
        _last_call_sig = ""    # signature du dernier appel (anti-répétition)
        _repeat_calls = 0      # répétitions identiques consécutives
        _repeat_blocks = 0     # nb de fois où la redirection a été ignorée
        _REPEAT_CALL_MAX = 3   # même appel ≥3× → bloqué et redirigé
        sandbox_calls = 0      # run_command/python/serve cumulés (coûteux)
        _SANDBOX_CALL_MAX = 12  # borne le coût total d'exécution par run
        did_bestofn = False    # best-of-N repair fired (once per run)
        did_decompose = False  # niveau 4 : décomposition ciblée (once per run)
        second_chance = False  # forced re-grounding after premature conclusion
        verify_retries = 0     # non-code deliverable re-tries (generalist verify)
        test_edits_red = 0   # times the model edited the TEST while it was red
        content_hist: dict[str, list] = {}  # per-file content-hash history (flip-flop)
        # Deliberation = the "reason harder when stuck" pass. Gated by the
        # config default AND the live UI "raisonnement" toggle, so turning that
        # button OFF disables the 🧠 pass (the inline Pensée always stays on).
        deliberate = (
            bool(getattr(self.settings, "agent_deliberate_when_stuck", True))
            and bool(getattr(self.hippocampe, "reasoning_enabled", True))
        )
        pending_delib = False  # a deliberation pass is queued (set when stuck)
        dirty = False        # a file was written since the last run_tests
        synth = ""
        build_mode = (mode == "build")
        # En recherche/analyse pure, chercher EST la tâche : un plafond de 3
        # (pensé pour empêcher un build de fuir vers le web) est trop bas et
        # fait dérailler la synthèse. On l'élargit hors build.
        if not build_mode:
            _WEB_CALL_MAX = 8
        had_attachments = bool(seed_names)
        did_read = False     # read_file called at least once (grounding)
        ran_command = False  # run_command succeeded (analyze grounding)
        did_search = False   # web_search/web_fetch succeeded (research grounding)
        critic_enabled = bool(getattr(self.settings, "agent_critic_enabled", True))
        critic_max = int(getattr(self.settings, "agent_critic_rounds", 1))
        critic_rounds = 0    # critic reviews already spent this run

        hard_iters = max_iters + max_iters // 2   # adaptive-budget ceiling
        # Plan initial à profondeur adaptative (L0 = aucun ; L1 = court ;
        # L2 = profond). Généraliste : on raisonne en livrables et critères de
        # réussite, pas en termes de code. Le plan alimente le blackboard, relu
        # à chaque étape — « bien clair, bien propre dès le départ ».
        thinking_mode = self._thinking_mode(worker)  # allège l'échafaudage natif
        if (route.level >= 1
                and (build_mode or route.deliverable == "fichier")
                and not thinking_mode      # un modèle thinking planifie déjà
                and bool(getattr(self.settings, "agent_initial_plan", True))):
            _budget = 200 if route.level == 1 else 380
            _plan_prompt = (
                f"Tâche ({route.kind}) : {task}\n\n"
                "Avant d'agir, planifie BRIÈVEMENT (pas de réalisation ici) :\n"
                "1) Reformule l'objectif en une phrase.\n"
                "2) Liste les 3-5 étapes nécessaires.\n"
                "3) Donne les critères qui prouveront que c'est RÉUSSI.\n"
                + ("4) Pour les cas délicats, anticipe les pièges.\n"
                   if route.level == 2 else "")
                + "Réponds en quelques lignes, sans réaliser la tâche.")
            try:
                _plan = await self._gen(worker, _plan_prompt,
                                        max_new_tokens=_budget, temperature=0.3)
                _plan = _strip_think(_plan)
                if _plan:
                    bb.note("agent", "PLAN — " + _plan[:600])
                    history += (f"\n[Plan initial]\n{_plan[:800]}\n")
                    yield {"type": "deliberation", "run_id": run_id,
                           "text": "Plan initial :\n" + _plan[:600]}
            except Exception:  # noqa: BLE001
                log.debug("initial plan failed", exc_info=True)
        it = 0
        while it < max_iters:                     # extensible on real progress
            it += 1
            _retried_budget = False               # think-truncation reroll (1×/tour)
            if run.stop:
                yield {"type": "run_stopped", "run_id": run_id, "files": files_total}
                return
            if run.inbox:
                injected = run.inbox[:]
                run.inbox.clear()
                history += "\n[Nouvelle consigne] " + " ".join(injected) + "\n"
                yield {"type": "interjection_applied", "run_id": run_id,
                       "messages": injected}

            files_now = ", ".join(op_list()) or "aucun"
            # « Fichiers actuels » n'a de sens qu'en build/analyze (l'agent y
            # écrit des fichiers). En research/answer, le livrable est du TEXTE :
            # injecter « Fichiers actuels : aucun » fait délirer le modèle (il
            # croit devoir « supprimer les sources » faute de fichiers). On
            # l'omet pour ces modes.
            if mode in ("build", "analyze"):
                ctx = (f"Tâche : {task}\nFichiers actuels : {files_now}\n\n"
                       + bb.render_for("agent", peers=False) + "\n"
                       + history[-8000:])
            else:
                ctx = (f"Tâche : {task}\n\n"
                       + bb.render_for("agent", peers=False) + "\n"
                       + history[-8000:])
            prompt = system + "\n\n" + ctx + "\nRune :"
            # Images natives (cerveau multimodal) : passées au PREMIER tour
            # seulement — ensuite l'agent travaille sur ce qu'il en a tiré.
            _imgs = getattr(self, "_pending_images", None) if it == 1 else None
            # Stopword ReAct : on coupe à </tool_call> pour NE PAS gâcher des
            # tokens après l'appel… SAUF pour les modèles thinking. La doc Qwen
            # déconseille les stopwords avec un modèle qui raisonne : il peut
            # écrire « </tool_call> » DANS son <think>, ce qui couperait la
            # génération en plein raisonnement, avant le vrai appel. Pour eux,
            # on laisse générer jusqu'au bout (le budget élargi ci-dessous
            # absorbe la longueur) et on parse l'appel a posteriori.
            if thinking_mode:
                _gkw = {"max_new_tokens": int(getattr(
                    self.settings, "agent_thinking_max_tokens", 2048))}
            else:
                _gkw = {"stop_strings": ["</tool_call>"]}
                # research/answer produisent une SYNTHÈSE (texte long) : 1024
                # tokens la coupent en plein milieu. On élargit pour ces modes
                # (le code, lui, tient en 1024 par étape).
                if mode in ("research", "answer"):
                    _gkw["max_new_tokens"] = int(getattr(
                        self.settings, "agent_synthesis_max_tokens", 2048))
            # STOP réactif : la génération de réflexion (la plus longue) est
            # interruptible au token près via l'event du run.
            _gkw["cancel_event"] = run.cancel
            if _imgs:
                _gkw["pil_images"] = _imgs
            out_raw = await self._gen(worker, prompt, **_gkw)
            # Génération interrompue par le bouton stop (cancel event levé) :
            # on sort AVANT de parser/exécuter le texte partiel. C'est ce qui
            # rend l'arrêt propre ET immédiat, même en pleine réflexion.
            if run.stop:
                yield {"type": "run_stopped", "run_id": run_id,
                       "files": files_total}
                return
            # A reasoning model wraps its scratchpad in <think>…</think>. We
            # CAPTURE it (UI trace) then strip it ROBUSTLY (handles a <think>
            # left unclosed when the token budget cuts mid-reasoning — the
            # n°1 failure mode of thinking models in a ReAct loop).
            think_txt = _capture_think(out_raw)
            # Génération coupée en plein raisonnement (think ouvert, jamais
            # fermé, et aucun tool_call) → on REJOUE l'étape avec plus de budget
            # au lieu de gâcher un tour. Une seule fois par étape.
            _think_truncated = (
                bool(re.search(r"<think>", out_raw, flags=re.I))
                and not re.search(r"</think>", out_raw, flags=re.I)
                and "<tool_call>" not in out_raw)
            if (_think_truncated and thinking_mode
                    and not _retried_budget):
                _retried_budget = True
                _gkw2 = dict(_gkw)
                _gkw2["max_new_tokens"] = int(getattr(
                    self.settings, "agent_thinking_max_tokens", 2048))
                out_raw = await self._gen(worker, prompt, **_gkw2)
                think_txt = _capture_think(out_raw)
            out = _strip_think(out_raw)
            calls = T.parse_tool_calls(out)

            if not calls:
                text = re.sub(r"<tool_call>.*?</tool_call>", "", out, flags=re.S).strip()
                no_call_streak += 1
                ex = exec_holder["last"]
                tests_ok = bool(ex and ex.get("ok"))
                if build_mode:
                    # A clean finish requires VERIFIED work: files written,
                    # nothing changed since the last run_tests, and it passed.
                    final_ok = bool(files_total) and (not dirty) and tests_ok
                else:
                    # analyze/answer: a substantive prose reply IS the
                    # deliverable. Require grounding so it isn't hallucinated —
                    # analyze needs a run_command (or a read); answer needs a
                    # read when files were attached.
                    if mode == "analyze":
                        grounded = ran_command or did_read
                    elif mode == "research":
                        # une synthèse de recherche n'est fondée que si une
                        # recherche/fetch web a réellement abouti (sinon
                        # l'agent pourrait inventer sans chercher).
                        grounded = did_search or did_read
                    else:  # answer
                        grounded = did_read or not had_attachments
                    final_ok = bool(text) and len(text) > 40 and grounded
                    # Vérificateur généraliste (non-code) : avant d'accepter la
                    # conclusion, l'heuristique contrôle longueur/esquive/
                    # sources/structure selon le type routé. L'auto-critique
                    # LLM (déjà présente plus bas) forme le second signal — les
                    # deux combinés. Le livrable n'est vert que si les deux OK.
                    if (final_ok and not _web_capped
                            and bool(getattr(self.settings,
                                             "agent_generalist_verify", True))):
                        try:
                            from rune.agentic.verifier import check_deliverable
                            _vd = check_deliverable(text, route.kind, route.level)
                            if not _vd.ok and verify_retries < 1:
                                verify_retries += 1
                                final_ok = False
                                _why = " ; ".join(_vd.reasons)
                                bb.record_fail("agent", "livrable insuffisant",
                                               why=_why)
                                history += (
                                    f"\n[Système] Livrable incomplet : {_why}. "
                                    "Améliore-le concrètement (ne conclus pas "
                                    "en supposant), puis termine.\n")
                                yield {"type": "critique", "run_id": run_id,
                                       "text": f"Livrable à compléter : {_why}"}
                                no_call_streak = 0
                                continue
                        except Exception:  # noqa: BLE001
                            log.debug("generalist verify failed", exc_info=True)
                if text and len(text) > 40 and final_ok:
                    if (critic_enabled and critic_rounds < critic_max
                            and not _web_capped):
                        critic_rounds += 1
                        approved, fb = await self._critic_review(
                            worker, task, mode, files_total,
                            exec_holder["last"], text, abs_dir)
                        if not approved and fb:
                            had_failures = True
                            history += (f"\n[Critique] {fb}\n[Système] Prends "
                                        "cette critique en compte : corrige, "
                                        "puis termine.\n")
                            yield {"type": "critique", "run_id": run_id, "text": fb}
                            no_call_streak = 0
                            continue
                    synth = text
                    break
                # Seuil adaptatif : un modèle thinking produit légitimement
                # des tours de pur <think> avant d'agir → le couper à 4 le
                # stoppe en plein raisonnement (observé sur email, 8B : 11 min,
                # arrêt à 9 étapes sans que l'escalade ait pu tirer). On lui
                # laisse 6 tours ; un modèle non-thinking garde 4.
                _no_call_max = 6 if thinking_mode else 4
                if no_call_streak >= _no_call_max:  # stuck → ground, then stop
                    # Second chance (once): in build mode with RED tests and
                    # budget left, the model repeating prose conclusions gets
                    # one forced re-grounding instead of a silent give-up —
                    # the observed "supposons que les tests ont été réalisés"
                    # failure: it concluded by ASSUMING the work done.
                    if (build_mode and files_total and not second_chance
                            and not tests_ok and it < max_iters):
                        second_chance = True
                        no_call_streak = 0
                        _ex = exec_holder.get("last") or {}
                        history += (
                            "\n[Système] ⛔ Tu essaies de CONCLURE alors que "
                            f"les tests ÉCHOUENT ({_ex.get('summary', 'rouges')}). "
                            "Interdit de supposer le travail fait. Reprends : "
                            "corrige le MODULE avec write_file d'après le "
                            "dernier détail d'échec ci-dessus, puis run_tests. "
                            "Tu ne peux conclure qu'avec des tests VERTS.\n")
                        continue
                    # Salvage: a 7B sometimes answers a trivial build task in
                    # PROSE (code in a ```python block) instead of calling
                    # write_file, ending with 0 files. Rescue that code block so
                    # the mission delivers something.
                    if build_mode and not files_total:
                        _code = _extract_py_block(text)
                        if _code:
                            _fn = _filename_from_code(_code)
                            _rw = op_write(_fn, _code)
                            if isinstance(_rw, dict) and _rw.get("ok") is not False:
                                yield {"type": "tool_result", "run_id": run_id,
                                       "name": "write_file", "ok": True,
                                       "path": _fn,
                                       "summary": f"code récupéré → {_fn}"}
                                log.info("salvaged code block from prose → %s", _fn)
                    if build_mode and dirty:       # verify before concluding
                        res = await self._to_thread(op_tests)
                        dirty = False
                        if isinstance(res, dict):
                            yield {"type": "exec_result", "run_id": run_id,
                                   "ok": res.get("ok"),
                                   "summary": res.get("summary"),
                                   "isolation": res.get("isolation"),
                                   "installs": res.get("installs", [])}
                    # Non-build: keep the model's prose rather than discard the
                    # only deliverable. (build) Don't trust prose — the grounded
                    # fallback synthesis (built from the real verdict) speaks.
                    if not build_mode and text and len(text) > 20:
                        synth = text
                    break
                # Nudges, mode-specific.
                if build_mode:
                    if not files_total:
                        nudge = ("Tu n'as encore écrit aucun fichier. N'explique "
                                 "pas : agis avec un <tool_call> write_file.")
                    elif dirty:
                        nudge = ("Tu as écrit/modifié des fichiers depuis le "
                                 "dernier run_tests. Relance run_tests pour "
                                 "VÉRIFIER avant de conclure — n'affirme pas que "
                                 "les tests passent sans les avoir relancés.")
                    elif ex is not None and not tests_ok:
                        nudge = ("Les tests échouent encore. Lis l'erreur, corrige "
                                 "le fichier fautif (write_file/edit_file) puis "
                                 "relance run_tests — uniquement par des <tool_call>.")
                    else:
                        nudge = ("Réponds par un bloc <tool_call> valide, ou par ta "
                                 "synthèse finale si la tâche est terminée.")
                elif mode == "analyze":
                    if had_attachments and not did_read:
                        nudge = ("Lis d'abord le fichier fourni avec read_file "
                                 "pour connaître son format, puis écris et "
                                 "exécute un script de calcul.")
                    elif not ran_command:
                        nudge = ("Écris un script qui calcule et IMPRIME le "
                                 "résultat, exécute-le avec run_command, puis "
                                 "donne le résultat. Réponds par un <tool_call> "
                                 "ou par ta synthèse finale.")
                    else:
                        nudge = ("Tu as le résultat : donne ta synthèse finale "
                                 "en texte clair (avec les chiffres), SANS "
                                 "tool_call.")
                else:  # answer
                    if had_attachments and not did_read:
                        nudge = ("Lis d'abord le(s) fichier(s) fourni(s) avec "
                                 "read_file, puis réponds en texte.")
                    else:
                        nudge = ("Donne ta réponse / synthèse finale en texte "
                                 "clair, SANS tool_call et sans écrire de fichier.")
                history += f"\n[Système] {nudge}\n"
                continue

            no_call_streak = 0
            history += f"\nRune (action {it}) : {out.strip()[:1500]}\n"
            # Reasoning surfaced in the UI: prefer the model's native <think>
            # trace (rich), else the prose "Pensée : …" that preceded the call.
            pensee = re.split(
                r"<tool_call>|<tools>|<write_file|<edit_file|\{\s*[\"']name[\"']",
                out)[0]
            pensee = re.sub(r"(?i)^\s*pens[eé]+e?\s*:?\s*", "", pensee.strip())
            reasoning = (think_txt or pensee).strip()
            reasoning = _strip_tool_syntax(reasoning)   # masque les balises résiduelles
            reasoning = _collapse_runaway(reasoning)    # coupe un raisonnement qui boucle
            reasoning = re.sub(r"\n{3,}", "\n\n", reasoning).strip()
            # Aperçu court (cf. _REASONING_PREVIEW) : coupe nette au mot + ellipse.
            if _REASONING_PREVIEW <= 0:
                reasoning = ""
            elif len(reasoning) > _REASONING_PREVIEW:
                reasoning = (reasoning[:_REASONING_PREVIEW].rsplit(" ", 1)[0]
                             .rstrip(" .,;:") + " …")

            # Explicit completion via the 'finish' tool. Validated with the
            # SAME grounding as a prose finish: build needs green tests,
            # analyze needs a run/read, answer needs a read (if files joined).
            if calls and calls[0].get("name") == "finish":
                fargs = calls[0].get("arguments", {}) or {}
                answer = str(fargs.get("answer") or fargs.get("text") or "").strip()
                ex = exec_holder["last"]
                tests_ok = bool(ex and ex.get("ok"))
                if build_mode:
                    ok_finish = bool(files_total) and (not dirty) and tests_ok
                elif mode == "analyze":
                    ok_finish = ran_command or did_read
                else:
                    ok_finish = did_read or not had_attachments
                if answer and ok_finish:
                    if critic_enabled and critic_rounds < critic_max:
                        critic_rounds += 1
                        approved, fb = await self._critic_review(
                            worker, task, mode, files_total,
                            exec_holder["last"], answer, abs_dir)
                        if not approved and fb:
                            had_failures = True
                            history += (f"\n[Critique] {fb}\n[Système] Prends "
                                        "cette critique en compte : corrige, "
                                        "puis rappelle finish.\n")
                            yield {"type": "critique", "run_id": run_id, "text": fb}
                            no_call_streak = 0
                            continue
                    yield {"type": "tool_call", "run_id": run_id, "index": it,
                           "name": "finish", "arguments": fargs,
                           "thought": reasoning}
                    synth = answer
                    break
                if not answer:
                    history += ("\n[Système] 'finish' exige un argument 'answer' "
                                "(ta réponse finale en texte).\n")
                elif build_mode and not tests_ok:
                    history += ("\n[Système] Tu ne peux pas conclure : lance "
                                "run_tests et obtiens un vert AVANT 'finish'.\n")
                else:
                    history += ("\n[Système] Termine d'abord le travail "
                                "(lis/calcule) avant d'appeler 'finish'.\n")
                continue

            # Sélection des appels à exécuter ce tour. Règle : UN SEUL outil à
            # effet de bord par tour (run_tests, run_command, finish… : on doit
            # voir leur résultat avant d'enchaîner). MAIS les écritures de
            # fichiers (write_file/edit_file) sont sûres et indépendantes — les
            # modèles de code émettent souvent module + test ENSEMBLE ; les
            # exécuter toutes évite de perdre le test (sinon « aucun fichier de
            # test »). On prend donc la série de write/edit en tête, et on
            # s'arrête au 1er outil à effet de bord (inclus).
            _WRITE_OPS = {"write_file", "edit_file"}
            _to_run = []
            for _c in calls:
                if _c.get("name") in _WRITE_OPS:
                    _to_run.append(_c)
                else:
                    _to_run.append(_c)   # 1er non-write → inclus puis stop
                    break
            if not _to_run:
                _to_run = calls[:1]

            for call in _to_run:
                # ── Garde générique anti-emballement (TOUS outils) ──
                # Un modèle bloqué répète le MÊME appel (mêmes args) en boucle
                # — read_file du même fichier, list_files, run_python identique,
                # delete/serve… On détecte la répétition exacte et on bloque
                # au-delà d'un seuil, en redirigeant vers une action utile.
                # Couvre d'un coup tous les outils non plafonnés
                # individuellement (read/list/search/python/command/serve/
                # delete), pas seulement le web.
                _sig = call["name"] + "|" + json.dumps(
                    call.get("arguments", {}), sort_keys=True, ensure_ascii=False)
                if _sig == _last_call_sig:
                    _repeat_calls += 1
                else:
                    _repeat_calls = 0
                    _last_call_sig = _sig
                # run_tests/write/edit légitimement répétables (gérés par leurs
                # propres compteurs) ; on cible les outils NON mutateurs.
                if (_repeat_calls >= _REPEAT_CALL_MAX
                        and call["name"] not in ("run_tests", "write_file",
                                                 "edit_file")):
                    _repeat_blocks += 1
                    # Message généraliste : l'agent ne fait pas QUE du code.
                    # On nomme l'impasse et on demande une action différente,
                    # sans présumer du type de tâche (module/test).
                    history += (
                        f"\n[Système] Tu répètes le même appel "
                        f"({call['name']}) sans que ça fasse avancer la "
                        "mission. Change de stratégie : une action DIFFÉRENTE "
                        "qui te rapproche du but (produis/corrige le livrable, "
                        "ou conclus si c'est prêt).\n")
                    yield {"type": "agent_warning", "run_id": run_id,
                           "message": f"{call['name']} répété — redirigé"}
                    # NE PAS réinitialiser à 0 : sinon le modèle re-répète par
                    # cycles de 3 à l'infini (observé : ⚠ redirigé puis même
                    # appel au tour suivant). On laisse le compteur monter ; au
                    # 2e blocage ignoré, on casse vraiment la boucle.
                    if _repeat_blocks >= 2:
                        history += (
                            "\n[Système] Tu n'as pas changé d'approche malgré "
                            "l'avertissement. Arrêt de cette piste.\n")
                        yield {"type": "agent_warning", "run_id": run_id,
                               "message": "boucle non résolue — arrêt de piste"}
                        break
                    _repeat_calls = 0
                    continue
                # Anti-emballement : un modèle bloqué (vu sur loan, 8B) peut
                # spammer web_search en boucle au lieu de coder. On plafonne
                # les recherches web DU MODÈLE par run ; au-delà, on refuse et
                # on le renvoie au code/tests.
                if call["name"] in ("web_search", "web_fetch"):
                    web_calls += 1
                    if web_calls > _WEB_CALL_MAX:
                        if build_mode:
                            history += (
                                "\n[Système] Trop de recherches web. STOP la "
                                "recherche : reviens au MODULE, corrige-le "
                                "d'après le dernier échec de test, puis "
                                "run_tests.\n")
                            _msg = "web_search plafonné — retour au code"
                        else:
                            history += (
                                "\n[Système] Tu as assez cherché. STOP les "
                                "recherches : rédige MAINTENANT ta synthèse "
                                "finale en citant les sources (URLs) déjà "
                                "trouvées. Réponds en texte clair, sans "
                                "tool_call.\n")
                            _msg = "recherches plafonnées — rédige la synthèse"
                            # Plafond atteint + recherche déjà faite : on ne
                            # RE-bloque plus la synthèse en boucle (critique/
                            # vérif), sinon l'agent tourne sans jamais conclure.
                            _web_capped = True
                        yield {"type": "agent_warning", "run_id": run_id,
                               "message": _msg}
                        continue
                # Cap global des appels sandbox COÛTEUX (run_command, run_python,
                # serve_and_probe) : même avec des args différents (donc hors
                # repeat-guard), un modèle peut les enchaîner sans fin. Chacun
                # peut durer jusqu'au timeout → borne le coût total d'un run.
                if call["name"] in ("run_command", "run_python",
                                    "serve_and_probe"):
                    sandbox_calls += 1
                    if sandbox_calls > _SANDBOX_CALL_MAX:
                        history += (
                            "\n[Système] Trop d'exécutions. Arrête d'exécuter "
                            "à répétition : écris/corrige le module et lance "
                            "run_tests pour valider.\n")
                        yield {"type": "agent_warning", "run_id": run_id,
                               "message": "exécutions sandbox plafonnées"}
                        continue
                yield {"type": "tool_call", "run_id": run_id, "index": it,
                       "name": call["name"], "arguments": call.get("arguments", {}),
                       "thought": reasoning}
                result = await self._to_thread(lambda c=call: T.dispatch(c, ops))
                preview = json.dumps(result.get("result", result.get("error", "")),
                                     ensure_ascii=False)[:600]
                history += f"<tool_response>{preview}</tool_response>\n"
                yield {"type": "tool_result", "run_id": run_id, "index": it,
                       "name": call["name"], "ok": bool(result.get("ok")),
                       "preview": preview}
                # Grounding flags for analyze/answer completion criteria.
                if call["name"] == "read_file" and result.get("ok") is not False:
                    did_read = True
                if call["name"] in ("web_search", "web_fetch") and \
                        result.get("ok") is not False:
                    did_search = True
                if call["name"] == "run_command" and result.get("ok"):
                    ran_command = True
                if call["name"] == "run_tests" and isinstance(result.get("result"), dict):
                    r = result["result"]
                    yield {"type": "exec_result", "run_id": run_id,
                           "ok": r.get("ok"), "summary": r.get("summary"),
                           "isolation": r.get("isolation"),
                           "installs": r.get("installs", [])}
                    if r.get("ok"):
                        red_streak = 0
                        last_failed = None
                        progress.update_code(0, passed=True)  # tracker généraliste
                        bb.record_win("agent",
                                      f"tests verts : {r.get('summary', '')}")
                        bb.set_status("agent", "vert")
                        bb.set_files("agent", [f["path"] for f in files_total])
                        history += (
                            f"\n[Système] Les tests passent ✓ ({r.get('summary')}). "
                            "Si la tâche est terminée, réponds par ta synthèse "
                            "finale (3-5 phrases) SANS tool_call ; n'édite plus "
                            "inutilement les fichiers.\n"
                        )
                    elif not r.get("no_test_file"):
                        had_failures = True
                        # FIX: red_streak (escalade) monte TOUJOURS sur un
                        # rouge. Auparavant un progrès (5→2 failed) le faisait
                        # REDESCENDRE → sur un run qui « progresse puis stagne »
                        # (vu sur email : 3→1 failed puis bloqué), il n'atteint
                        # jamais 4 et le best-of-N ne tire JAMAIS. Le progrès ne
                        # sert plus qu'à ÉTENDRE le budget, pas à freiner
                        # l'escalade. Les deux signaux sont découplés.
                        _f, _p = _parse_test_counts(r.get("summary", ""))
                        red_streak += 1
                        progress.update_code(_f, passed=False)  # tracker généraliste
                        if (_f is not None and last_failed is not None
                                and _f < last_failed):
                            if max_iters < hard_iters:
                                max_iters = min(max_iters + 2, hard_iters)
                            history += (
                                f"\n[Système] Progrès : {last_failed} → {_f} "
                                "échec(s). Continue exactement dans cette "
                                "direction.\n")
                        if _f is not None:
                            last_failed = _f
                        # Real failure → show the model WHICH test failed and the
                        # assertion, so it can target the fix instead of guessing.
                        digest = _pytest_failure_digest(r.get("stdout_tail", ""))
                        # Publish to the blackboard: what was tried + why it
                        # failed. Next step re-reads this → no more re-treading
                        # dead ends (the amnesia the 14B showed in traces).
                        _last_act = "édition/écriture du module"
                        for _h in reversed(history.split("\n")):
                            if "write_file(" in _h or "edit_file(" in _h:
                                _last_act = _h.strip()[:120]
                                break
                        bb.record_fail("agent",
                                       f"{_last_act} → {r.get('summary', 'échec')}",
                                       why=(digest or "").split("\n")[0][:200])
                        # Classer la NATURE du blocage pour router la bonne
                        # stratégie : un échec de test (FAILURE) se corrige,
                        # mais une impasse (BLOCKED), une info manquante
                        # (UNKNOWN) ou une ambiguïté (WAITING_INPUT) appellent
                        # une réaction différente — régénérer du code ne sert
                        # à rien si le problème n'est pas un bug.
                        from rune.agentic.blockage import (
                            classify_blockage, strategy_for, FAILURE as _FAIL)
                        _btype = classify_blockage(
                            has_test_failure=True,
                            no_call_streak=no_call_streak,
                            repeat_calls=_repeat_calls,
                            last_error=(digest or r.get("summary", "")),
                        )
                        bb.set_status("agent", "bloque" if red_streak >= 3
                                      else "en_cours")
                        if _btype != _FAIL:
                            bb.note("agent", f"nature du blocage : {_btype}")
                        if digest:
                            history += (
                                f"\n[Système] Détail de l'échec :\n{digest}\n"
                                f"{strategy_for(_btype)}\n"
                            )
                        # Stuck (2nd red in a row) → ONE separate deliberation
                        # pass: reason about the root cause with no tool_call,
                        # fed back so the next action is grounded. Only fires
                        # when cheap inline reasoning isn't converging, so we
                        # pay the extra generation rarely, not every step.
                        # Stuck (2nd red in a row) → ground the debugging in
                        # REALITY before deliberating: inject the actual module
                        # + test sources. Without this the model debugs from
                        # its (often wrong) memory of what it wrote — e.g. a
                        # 42-byte `return True` stub it believes is a real
                        # implementation.
                        if red_streak == 2:
                            _srcs = []
                            _mods = [f["path"] for f in files_total
                                     if f["path"].endswith(".py")
                                     and not os.path.basename(
                                         f["path"]).startswith("test_")][:2]
                            _tsts = [f["path"] for f in files_total
                                     if os.path.basename(
                                         f["path"]).startswith("test_")][:1]
                            for _p in _mods + _tsts:
                                _c = op_read(_p)
                                if isinstance(_c, str) and _c and \
                                        not _c.startswith("[introuvable"):
                                    _srcs.append(
                                        f"--- {os.path.basename(_p)} ---\n"
                                        f"{_c[:1200]}")
                            if _srcs:
                                history += (
                                    "\n[Système] Contenu RÉEL des fichiers "
                                    "(débogue d'après CECI, pas de mémoire) :\n"
                                    + "\n".join(_srcs) + "\n")
                        if deliberate and red_streak == 2:
                            pending_delib = True
                        # 3rd red → ONE web search on the failure (doc/error
                        # lookup), injected as grounding. Once per run.
                        if (progress.stalled >= progress.ALTERNATE
                                and not did_websearch
                                and bool(getattr(self.settings,
                                                 "agent_web_on_block", True))):
                            did_websearch = True
                            _q = (digest or r.get("summary", "")).splitlines()
                            _q = (_q[0] if _q else "")[:120]
                            if _q:
                                try:
                                    _wr = await self._to_thread(
                                        lambda: op_websearch(f"python {_q}", 3))
                                except Exception:  # noqa: BLE001
                                    _wr = None
                                _hits = (_wr or {}).get("results") or []
                                if _hits:
                                    _blk = "\n".join(
                                        f"- {h.get('title', '')} : "
                                        f"{(h.get('snippet', '') or '')[:200]}"
                                        for h in _hits[:2])
                                    history += (
                                        "\n[Système] Recherche web sur "
                                        f"l'erreur :\n{_blk}\nSers-t'en si "
                                        "pertinent, sinon ignore.\n")
                        # 4th red → best-of-N repair: K full-module rewrites,
                        # each TESTED in the sandbox, keep the best. The
                        # sandbox is the oracle — use it as a SELECTOR, not
                        # just a failure detector. Once per run.
                        # FIX: ``>=`` (was ``==``) — red_streak can jump 3→5
                        # when two reds land back-to-back, skipping 4 exactly
                        # and NEVER firing (observed: email/loan stuck at "3
                        # failed" to the 6-red hard stop without best-of-N).
                        # Best-of-N stays ON even in thinking mode: it's a
                        # deblocking net useful to ANY model, unlike the
                        # initial plan (which a thinking model does natively).
                        if (build_mode and progress.stalled >= progress.ALTERNATE
                                and not did_bestofn):
                            did_bestofn = True
                            _k_set = int(getattr(self.settings,
                                                 "agent_bestofn", 0))
                            if _k_set >= 2:
                                _k = _k_set
                            else:
                                _k = max(2, int(self._hw_knobs().get("bestofn", 2)))
                            _mods = [f["path"] for f in files_total
                                     if f["path"].endswith(".py")
                                     and not os.path.basename(
                                         f["path"]).startswith("test_")]
                            _tsts2 = [f["path"] for f in files_total
                                      if os.path.basename(
                                          f["path"]).startswith("test_")]
                            if _mods and _tsts2:
                                _mp = _mods[0]
                                _msrc = op_read(_mp)
                                _tsrc = op_read(_tsts2[0])
                                _base = (
                                    f"Tâche : {task}\n\nTest (vérité, ne PAS "
                                    "le changer) :\n```python\n"
                                    f"{str(_tsrc)[:1400]}\n```\n"
                                    "Module actuel (ÉCHOUE) :\n```python\n"
                                    f"{str(_msrc)[:1200]}\n```\n"
                                    f"Échec : {digest[:400]}\n\n"
                                    "Réécris le module COMPLET corrigé pour "
                                    "satisfaire TOUS les tests. Réponds "
                                    "UNIQUEMENT avec le code Python du module, "
                                    "sans balises ```.\n")
                                _best, _best_f = None, 10**9
                                # ONE batched GPU pass for all K candidates
                                # (sampling is per-row independent → diverse).
                                _cands = await self._gen_batch(
                                    worker, [_base] * _k,
                                    temperature=0.5, max_new_tokens=900)
                                for _cand in _cands:
                                    _code = (_extract_py_block(_cand)
                                             or _cand.strip())
                                    if not _code or _py_syntax_error(_code):
                                        continue
                                    _w = op_write(_mp, _code)
                                    if (not isinstance(_w, dict)
                                            or _w.get("ok") is False):
                                        continue
                                    _tr = await self._to_thread(op_tests)
                                    _tf, _tp = _parse_test_counts(
                                        (_tr or {}).get("summary", ""))
                                    _tf = 10**9 if _tf is None else _tf
                                    if (_tr or {}).get("ok"):
                                        _best, _best_f = _code, 0
                                        break
                                    if _tf < _best_f:
                                        _best, _best_f = _code, _tf
                                if _best is not None:
                                    op_write(_mp, _best)
                                    _vr = await self._to_thread(op_tests)
                                    _note = (
                                        f"Best-of-{_k} : meilleur candidat "
                                        f"retenu ({_vr.get('summary', '?')}).")
                                    history += f"\n[Système] {_note}\n"
                                    yield {"type": "deliberation",
                                           "run_id": run_id, "text": _note}
                                    if (_vr or {}).get("ok"):
                                        red_streak = 0
                                        last_failed = None

                        # 5th red → NIVEAU 4 (décomposition ciblée) : le
                        # best-of-N a échoué, régénérer encore tout le module
                        # ne sert à rien (même cas raté K fois). On ISOLE le
                        # SEUL test qui résiste + son assertion + le code réel,
                        # et on demande une correction CIBLÉE sur ce cas précis
                        # — dernière stratégie qualitativement différente avant
                        # l'arrêt honnête (red 6). Une fois par run.
                        if (build_mode and progress.stalled >= progress.DECOMPOSE
                                and not did_decompose):
                            did_decompose = True
                            _focus = _focus_failing_case(
                                r.get("stdout_tail", ""))
                            if _focus:
                                _mods5 = [f["path"] for f in files_total
                                          if f["path"].endswith(".py")
                                          and not os.path.basename(
                                              f["path"]).startswith("test_")][:1]
                                _src5 = ""
                                if _mods5:
                                    _c5 = op_read(_mods5[0])
                                    if isinstance(_c5, str) and _c5 and \
                                            not _c5.startswith("[introuvable"):
                                        _src5 = (f"\n--- {os.path.basename(_mods5[0])}"
                                                 f" (actuel) ---\n{_c5[:1400]}")
                                history += (
                                    "\n[Système] Un SEUL cas résiste. Oublie le "
                                    "reste (il passe) et concentre-toi EXACTEMENT "
                                    f"sur ce cas :\n"
                                    f"• test : {_focus.get('test', '?')}\n"
                                    f"• échec : {_focus.get('assertion', '?')}\n"
                                    f"{_focus.get('raw', '')}\n"
                                    f"{_src5}\n"
                                    "Demande-toi : que produit le code AUJOURD'HUI "
                                    "pour ce cas, et que DEVRAIT-il produire ? "
                                    "Corrige UNIQUEMENT ce qui fait diverger les "
                                    "deux (pas de réécriture globale), puis "
                                    "run_tests.\n")
                                bb.note("agent", f"décomposition ciblée sur "
                                        f"{_focus.get('test', 'cas résistant')}")
                                yield {"type": "deliberation", "run_id": run_id,
                                       "text": "Décomposition : focalisation sur "
                                       f"le cas {_focus.get('test', '?')}"}

                # Steer away from the "no tests ran" rabbit hole: the 7B tends
                # to fiddle with pyproject/pytest.ini instead of writing tests.
                blob = json.dumps(result, ensure_ascii=False).lower()
                if ("no tests ran" in blob or "no tests collected" in blob
                        or "collected 0 items" in blob):
                    has_test = any(
                        os.path.basename(f["path"]).startswith("test_")
                        or os.path.basename(f["path"]).endswith("_test.py")
                        for f in files_total
                    )
                    if not has_test:
                        history += (
                            "\n[Système] Aucun test collecté : tu n'as pas "
                            "encore écrit de fichier de tests. NE touche PAS à "
                            "la config (pyproject.toml / pytest.ini). Écris un "
                            "fichier « test_<module>.py » avec write_file (des "
                            "fonctions « def test_xxx(): » avec des assert), "
                            "puis relance run_tests.\n"
                        )
                    else:
                        # File exists but pytest collected nothing → almost
                        # always because the model wrote module-level asserts
                        # instead of test functions. pytest only collects
                        # functions named test_* (or unittest.TestCase classes).
                        history += (
                            "\n[Système] Ton fichier de tests existe mais "
                            "pytest n'a collecté AUCUN test. pytest ne "
                            "reconnaît QUE les fonctions nommées « def "
                            "test_xxx(): » (ou les classes unittest.TestCase). "
                            "Réécris test_<module>.py avec PLUSIEURS « def "
                            "test_xxx(): » contenant des assert — PAS des "
                            "assert au niveau module, PAS de script. Puis "
                            "relance run_tests.\n"
                        )

                # Collection error = a test imports a module that doesn't (yet)
                # exist or lacks the symbol. Name it so the 7B writes it instead
                # of looping (it tends to scaffold setup.py/README and forget
                # the actual module).
                if ("error during collection" in blob or "importerror" in blob
                        or "modulenotfounderror" in blob):
                    circ = self._circular_modules(abs_dir)
                    if circ or "circular import" in blob or "partially initialized" in blob:
                        # Self-import / circular import: the model can't see its
                        # own bug by guessing. Feed it the actual source + the
                        # full traceback and name the fix explicitly.
                        if circ:
                            stem, src = circ[0]
                            history += (
                                f"\n[Système] Import CIRCULAIRE détecté : "
                                f"« {stem}.py » s'importe lui-même (une ligne "
                                f"`from {stem} import …` ou `import {stem}` "
                                f"dans {stem}.py). Voici son contenu actuel :\n"
                                f"```python\n{src}\n```\n"
                                "SUPPRIME cette ligne d'auto-import et définis "
                                "le symbole directement dans le fichier. Ne "
                                "réécris que ça, puis relance run_tests.\n"
                            )
                        tail = (result.get("result", {}) or {}).get("stdout_tail", "")
                        if tail:
                            history += f"\n[Système] Trace complète :\n{tail[-1200:]}\n"
                    else:
                        needs = self._modules_needing_impl(abs_dir)
                        if needs:
                            spec = "; ".join(
                                f"{stem}.py doit définir {', '.join(syms)}"
                                for stem, syms in needs
                            )
                            history += (
                                "\n[Système] Erreur de collecte : un test importe "
                                f"un module absent ou incomplet. {spec}. Écris ce(s) "
                                "fichier(s) avec write_file (vraie implémentation, "
                                "pas de stub), puis relance run_tests.\n"
                            )

                # Sterile-loop guard + verification tracking.
                if call["name"] in ("write_file", "edit_file"):
                    # edit_file whose `find` pattern didn't match: the model
                    # invented a pattern absent from the file and burns turns
                    # re-trying (observed email/loan loops). Feed back the
                    # ACTUAL content + tell it to rewrite the whole file with
                    # write_file instead of guessing a find pattern.
                    if (call["name"] == "edit_file"
                            and result.get("ok") is False
                            and "motif" in str(result.get("error", ""))):
                        _ep = (call.get("arguments") or {}).get("path", "") or ""
                        _cur = op_read(_ep) if _ep else ""
                        if (isinstance(_cur, str) and _cur
                                and not _cur.startswith("[introuvable")):
                            history += (
                                "\n[Système] Ton edit_file a échoué : le motif "
                                "`find` n'existe pas dans le fichier. Contenu "
                                f"ACTUEL :\n```python\n{_cur}\n```\n"
                                "Réécris le fichier ENTIER avec write_file en y "
                                "intégrant ta correction — n'utilise pas "
                                "edit_file avec un motif deviné.\n")
                    tests_no_write = 0
                    writes_since_test += 1
                    # TDD anti-cheat: editing the TEST while it's red (and a
                    # real module exists) is "change the test to make it pass".
                    # The 7B falls back to this when it can't debug the module
                    # (exactly the observed step-8 failure). Warn hard — but
                    # don't hard-block: a genuinely buggy test is legitimate.
                    _tgt = (call.get("arguments") or {}).get("path", "") or ""
                    _stem = os.path.splitext(os.path.basename(_tgt))[0]
                    _is_test = _stem.startswith("test_") or _stem.endswith("_test")
                    _has_mod = bool(abs_dir) and any(
                        p.suffix == ".py"
                        and not (p.stem.startswith("test_")
                                 or p.stem.endswith("_test"))
                        for p in Path(abs_dir).glob("*.py")
                    )
                    if _is_test and red_streak >= 1 and _has_mod:
                        test_edits_red += 1
                        history += (
                            "\n[Système] ⚠ Tu modifies le TEST alors qu'il "
                            "échoue. Le test définit le comportement ATTENDU — "
                            "le changer pour qu'il passe, c'est tricher. Corrige "
                            "le MODULE pour satisfaire le test existant. Si (et "
                            "seulement si) le test est lui-même faux, explique "
                            "pourquoi AVANT de le toucher.\n")
                    if result.get("ok") is not False and not result.get("unchanged"):
                        dirty = True       # unverified change pending
                        # Flip-flop detector: the model re-writing a content it
                        # ALREADY tried (A→B→A — e.g. toggling return True ↔
                        # return False in a stub) burns turns without progress.
                        _cur2 = op_read(_tgt) if _tgt else ""
                        if isinstance(_cur2, str) and _cur2 and \
                                not _cur2.startswith("[introuvable"):
                            _h = hash(_cur2)
                            _hist = content_hist.setdefault(_tgt, [])
                            if (len(_hist) >= 2 and _h == _hist[-2]
                                    and _h != _hist[-1]):
                                history += (
                                    "\n[Système] ⚠ Tu viens de RÉÉCRIRE un "
                                    "contenu déjà essayé qui ÉCHOUAIT (aller-"
                                    "retour entre deux versions). Aucune des "
                                    "deux n'est la solution. Écris une "
                                    "implémentation RÉELLE et complète qui "
                                    "satisfait TOUS les cas du test :\n"
                                    f"```python\n{_cur2[:800]}\n```\n")
                            _hist.append(_h)
                            del _hist[:-6]
                    if writes_since_test == 1 and had_failures:
                        # Déjà en rouge : une seule réécriture suffit avant de
                        # devoir re-tester. Éditer à l'aveugle plusieurs fois
                        # de suite (vu sur email/loan : write→write→write) ne
                        # fait que brûler des tours sans vérifier l'effet.
                        history += (
                            "\n[Système] Tu es en ÉCHEC et tu viens de "
                            "modifier le module. Lance run_tests MAINTENANT "
                            "(un seul tool_call) pour voir l'effet AVANT toute "
                            "autre édition.\n")
                    elif writes_since_test == 2:
                        history += (
                            "\n[Système] Tu écris sans vérifier. Lance "
                            "run_tests MAINTENANT (un seul tool_call run_tests) "
                            "avant de continuer à éditer.\n")
                elif call["name"] == "run_tests":
                    dirty = False          # verified (whatever the verdict)
                    writes_since_test = 0
                    tests_no_write += 1
                    if tests_no_write == 1 and not files_total:
                        # First run_tests before writing anything: give the
                        # 7B the exact first move (a template unsticks it).
                        history += (
                            "\n[Système] Tu n'as RIEN écrit. Ta première action "
                            "DOIT être write_file du module, ex. :\n"
                            "<tool_call>{\"name\": \"write_file\", \"arguments\": "
                            "{\"path\": \"module.py\", \"content\": \"<code>\"}}"
                            "</tool_call>\nPuis write_file de « test_module.py », "
                            "puis run_tests.\n")
                    elif tests_no_write >= 2:
                        history += (
                            "\n[Système] Tu relances run_tests sans rien "
                            "écrire. Écris le module ET « test_<module>.py » "
                            "avec write_file MAINTENANT.\n")

            # Model keeps writing without testing → force a verification so it
            # gets the real verdict (even if it rewrote identical content, which
            # leaves `dirty` False). Forced after just 2 edits (was 3) so the
            # model can't burn turns editing blind — this also makes red_streak
            # (hence the best-of-N / deliberation triggers) advance on time.
            # Two forced runs still failing → stop: better an honest ⚠ than an
            # endless spin.
            if build_mode and writes_since_test >= 2:
                res = await self._to_thread(op_tests)
                writes_since_test = 0
                dirty = False
                forced_runs += 1
                if isinstance(res, dict):
                    yield {"type": "exec_result", "run_id": run_id,
                           "ok": res.get("ok"), "summary": res.get("summary"),
                           "isolation": res.get("isolation"),
                           "installs": res.get("installs", [])}
                    if res.get("ok"):
                        red_streak = 0
                    else:
                        red_streak += 1
                        had_failures = True
                        if deliberate and red_streak == 2:
                            pending_delib = True
                    history += (
                        f"\n[Système] J'ai lancé run_tests : "
                        f"{res.get('summary')}. Corrige d'après CE verdict, "
                        "puis relance run_tests toi-même.\n"
                    )
                if forced_runs >= 2:
                    break
            if tests_no_write >= 3:        # model won't write — stop wasting
                break
            # Hard cap on consecutive failing tests: a perfectly alternating
            # write→test→write→test loop that never goes green slips past the
            # write/test guards above. Stop honestly rather than spin to the
            # iteration cap (or the user's manual abort).
            # Arrêt honnête généralisé : N tours SANS PROGRÈS (tous types de
            # tâche confondus), plus seulement des "tests rouges consécutifs".
            # Le compteur progress.stalled s'accumule même quand des écritures
            # ou lectures intercalent les tests — ce qui faisait que l'ancien
            # red_streak consécutif n'atteignait jamais le seuil.
            if progress.should_stop:
                _le = exec_holder.get("last")
                digest = _pytest_failure_digest(_le.get("stdout_tail", "")) \
                    if isinstance(_le, dict) else ""
                history += (
                    f"\n[Système] {progress.stalled} tours sans progrès — "
                    "j'arrête pour ne pas boucler. Donne ta meilleure "
                    "synthèse de l'état actuel et du blocage, SANS tool_call.\n"
                )
                yield {"type": "agent_warning", "run_id": run_id,
                       "message": f"Arrêt anti-boucle : {red_streak} échecs "
                                  "consécutifs. " + (digest[:200] if digest else "")}
                break

            # Deliberation (separate reasoning pass) when stuck — triggered from
            # either a model run_tests or a forced one. Done here, once, so the
            # reasoning lands in history before the next action generation.
            if pending_delib:
                pending_delib = False
                files_n = ", ".join(op_list()) or "aucun"
                delib = ""
                # Modèles NON-thinking : raisonnement multi-angles dédié
                # (decompose → explore → critique croisée → synthèse). Un
                # Instruct ne raisonne pas seul ; le simple « réfléchis » inline
                # ci-dessous lui donne du raisonnement plat. DeliberativeReasoner
                # compense en le forçant à changer d'angle. Coûteux (plusieurs
                # appels) → réservé au cas bloqué, et SEULEMENT non-thinking.
                if not thinking_mode:
                    try:
                        from rune.cognition.deliberation import (
                            DeliberativeReasoner)
                        import asyncio as _aio
                        _reasoner = DeliberativeReasoner(
                            model=getattr(worker, "model", worker),
                            kg=getattr(self.hippocampe, "kg", None),
                        )
                        _delib_msg = (f"Tâche : {task}\nFichiers : {files_n}\n"
                                      "Les tests échouent toujours. Analyse la "
                                      "cause racine et le plan de correction.")
                        delib = await _aio.to_thread(
                            _reasoner.deliberate, _delib_msg, 4)
                        delib = _strip_think(delib or "")
                    except Exception:  # noqa: BLE001 — repli sur l'inline
                        delib = ""
                # Modèles thinking (ou repli si le multi-angles a échoué) :
                # réflexion profonde inline, un seul appel ciblé cause-racine.
                if not delib:
                    dprompt = (
                        system + "\n\n"
                        + f"Tâche : {task}\nFichiers : {files_n}\n\n"
                        + history[-6000:]
                        + "\n[Réflexion approfondie] Les tests échouent toujours. "
                        "NE produis AUCUN tool_call. Analyse la cause RACINE : que "
                        "dit précisément l'erreur, le module ou le test a-t-il "
                        "tort, et quel est le plan de correction exact (quel "
                        "fichier, quel changement) ? Réponds en 3-6 phrases. "
                        "IMPORTANT : tu n'exécutes RIEN pendant cette réflexion — "
                        "ne décris JAMAIS une action comme déjà effectuée "
                        "(pas de « correction effectuée », « j'ai relancé les "
                        "tests ») ; uniquement le diagnostic et le plan.\n"
                        "Rune :")
                    delib = await self._gen(worker, dprompt, max_new_tokens=512)
                    delib = _strip_think(delib)
                delib = re.sub(r"<tool_call>.*?</tool_call>", "", delib,
                               flags=re.S).strip()
                delib = re.sub(r"\n{3,}", "\n\n", delib)[:1000]
                if delib:
                    history += f"\n[Réflexion] {delib}\n"
                    history += (
                        "\n[Système] Applique CE plan MAINTENANT : écris la "
                        "correction avec write_file/edit_file, puis relance "
                        "run_tests. N'enchaîne pas les run_tests sans corriger.\n"
                    )
                    yield {"type": "deliberation", "run_id": run_id, "text": delib}
                # A deliberation is a fresh start on the fix — clear the
                # sterile-loop counter so the safety break can't kill the run
                # before the model has acted on the plan it was just given.
                tests_no_write = 0

        # Synthesis (use the model's final text, else ask for one).
        if not synth.strip():
            try:
                if build_mode:
                    paths = ", ".join(f["path"] for f in files_total) or "aucun"
                    ex = exec_holder["last"]
                    verdict = ("" if ex is None else
                               "Tests : " + ("OK. " if ex.get("ok")
                                             else f"échec ({ex.get('summary')}). "))
                    synth = await self._gen(core, (
                        f"Tâche : {task}\nFichiers : {paths}\n{verdict}\n"
                        "Rédige UNE synthèse brève (3-5 phrases) de ce qui a été "
                        "accompli et de l'état des tests. Pas de code."
                    ), prose=True, max_new_tokens=384)
                else:
                    # analyze/answer: ground the fallback on what was actually
                    # read/run, never on test verdicts (there are none).
                    synth = await self._gen(core, (
                        system + "\n\n" + history[-6000:]
                        + "\n[Système] Donne maintenant ta réponse finale en "
                        "texte clair, d'après ce qui précède. SANS tool_call, "
                        "sans code. Si l'information manque, dis-le."
                        "\nRune :"
                    ), prose=True, max_new_tokens=768)
                    synth = _strip_think(synth)
                    synth = re.sub(r"<tool_call>.*?</tool_call>", "", synth,
                                   flags=re.S).strip()
            except Exception:  # noqa: BLE001
                log.exception("react synthesis failed")
        synth = _clean_synthesis(synth)   # même nettoyage que le chemin `run`
        if synth.strip():
            yield {"type": "synthesis", "run_id": run_id, "text": synth}
            # Mode RESEARCH : une synthèse CONSÉQUENTE est un livrable qu'on
            # archive → on l'écrit aussi en rapport .md dans la mission. Une
            # réponse courte reste en chat seulement (un .md serait surdimensionné).
            _SYNTH_FILE_MIN = int(getattr(
                self.settings, "agent_research_report_min_chars", 1200))
            if mode == "research" and len(synth.strip()) >= _SYNTH_FILE_MIN:
                try:
                    _title = (name or "rapport").strip()
                    _md = f"# {_title}\n\n*Synthèse de recherche — {task}*\n\n{synth.strip()}\n"
                    # Écriture DIRECTE (acte délibéré de Lythéa en fin de run,
                    # pas une écriture du modèle) : on contourne la garde
                    # anti-déflection « narrative file » qui ne vise que les
                    # write_file du modèle pendant la boucle.
                    _fp = os.path.join(abs_dir, "rapport.md")
                    with open(_fp, "w", encoding="utf-8") as _fh:
                        _fh.write(_md)
                    if not any(f.get("path") == "rapport.md"
                               for f in files_total):
                        files_total.append({"path": "rapport.md"})
                    yield {"type": "file_written", "run_id": run_id,
                           "path": "rapport.md",
                           "message": "Rapport de recherche enregistré : rapport.md"}
                except Exception:  # noqa: BLE001 — le rapport ne doit jamais casser un run
                    log.debug("research report write failed", exc_info=True)

        # Learn from errors: if the run hit real failures, distill ONE reusable
        # lesson into the SHARED procedural memory (tagged with provenance).
        if learn_enabled and had_failures:
            lesson = await self._learn_lesson(
                worker, task, mode, history, exec_holder["last"], slug)
            if lesson:
                yield {"type": "lesson_learned", "run_id": run_id,
                       "trigger": lesson["trigger"], "approach": lesson["approach"]}
        _le2 = exec_holder.get("last")
        if build_mode and isinstance(_le2, dict) and _le2.get("ok"):
            self._archive_snippet(task, files_total, op_read)  # proven snippet
            _sp = await self._author_skill(worker, task, history, slug,
                                           files_total, op_read)
            if _sp:
                yield {"type": "lesson_learned", "run_id": run_id,
                       "trigger": "skill rédigée",
                       "approach": f"SKILL.md créée : {os.path.basename(os.path.dirname(_sp))}"}

        self._write_manifest(abs_dir, {
            "name": name, "slug": slug, "task": task, "status": "done",
            "mode": "react", "task_mode": mode,
            "files": [f.get("path") for f in files_total],
            "exec": exec_holder["last"], "synthesis": synth.strip()[:2000],
        })
        yield {
            "type": "run_done", "run_id": run_id,
            "steps": 0, "files": files_total, "exec": exec_holder["last"],
        }

    # ── the loop ─────────────────────────────────────────────────────
    async def run(self, task: str, *, run_id: str | None = None, subdir: str = "",
                  react: bool | None = None, attachments=None):
        run_id = run_id or uuid.uuid4().hex[:12]
        run = _Run(task=task)
        self._runs[run_id] = run
        core = self.pool.core
        flat = not bool(_PACKAGE_HINT.search(task))   # package layout on demand
        seen: dict = {}                               # cross-step idempotence
        # Per-run mode override (UI toggle); fall back to the configured default.
        use_react = self.react_enabled if react is None else bool(react)
        try:
            if use_react:
                async for ev in self._react(task, run_id, run, core,
                                            attachments=attachments):
                    _record_event(run, ev)
                    yield ev
                return
            # Mission identity: a short name + a dedicated folder, so files
            # don't scatter and several missions stay distinguishable. If the
            # caller passes an existing folder, we *resume* it.
            ws = self.workspace
            resumed = bool(
                subdir and ws is not None
                and getattr(ws, "exists", None) and ws.exists(subdir)
            )
            abs_dir = self._mission_abs(subdir) if subdir else None
            if resumed:
                manifest = self._read_manifest(abs_dir)
                name = manifest.get("name") or await self._gen_name(core, task)
                slug = subdir.rsplit("/", 1)[-1]
            else:
                name = await self._gen_name(core, task)
                slug = self._unique_slug(_slugify(name))
                if not subdir:
                    subdir = f"missions/{slug}"
                abs_dir = self._mission_abs(subdir)
            self._write_manifest(abs_dir, {
                "name": name, "slug": slug, "task": task,
                "status": "running", "resumed": resumed,
            })
            # Drop user-provided files into the mission dir (linear fallback).
            _seed_names, _seed_note = self._seed_attachments(subdir, attachments)
            if _seed_note:
                task = task + "\n\n" + _seed_note
            yield {
                "type": "run_start",
                "run_id": run_id,
                "task": task,
                "name": name,
                "slug": slug,
                "subdir": subdir,
                "resumed": resumed,
                "attachments": _seed_names,
                "workers": self.pool.available_names(),
            }

            # 1. PLAN (steered core — the plan is Rune reasoning).
            plan_prompt = (
                "Tu planifies une tâche de développement logiciel. Découpe-la "
                "en 2 à 4 étapes concrètes et ordonnées, une par ligne, "
                "numérotées « 1. … ». CHAQUE étape doit produire ou modifier du "
                "code (fichiers). N'ajoute PAS d'étape « exigences », "
                "« interface utilisateur », « documentation » ni "
                "« requirements » sauf si la tâche le demande explicitement. "
                "Pour « un module X avec ses tests », deux étapes suffisent : "
                "(1) écrire le module, (2) écrire les tests. Sois concise, pas "
                "de sous-points.\n\nTâche : " + task
            )
            plan_text = await self._gen(core, plan_prompt)
            steps = self._parse_plan(plan_text) or [task]
            # On simple tasks, drop meta steps (doc/requirements/UI/packaging)
            # the model adds despite the prompt — keep only code-producing ones.
            if not _PACKAGE_HINT.search(task):
                kept = [s for s in steps if not _PLAN_DROP.search(s)]
                steps = kept or steps
            yield {"type": "plan", "run_id": run_id, "steps": steps}

            context = ""
            files_total: list[dict] = []
            for i, step_title in enumerate(steps, start=1):
                if run.stop:
                    yield {"type": "run_stopped", "run_id": run_id, "at_step": i}
                    return

                # Absorb interjections between steps.
                if run.inbox:
                    injected = run.inbox[:]
                    run.inbox.clear()
                    context += "\n[Nouvelle consigne de l'utilisateur] " + " ".join(injected)
                    yield {
                        "type": "interjection_applied",
                        "run_id": run_id,
                        "messages": injected,
                    }

                yield {"type": "step_start", "run_id": run_id, "index": i, "title": step_title}

                # Execution can run on an auxiliary worker (no prefix needed).
                worker = self.pool.pick(needs_prefix=False)
                existing = self._existing_files(abs_dir)
                exist_line = (
                    "Fichiers déjà présents dans la mission : "
                    f"{', '.join(existing)}\nComplète/modifie ces fichiers, "
                    "ne les recrée pas sans raison.\n"
                    if existing else ""
                )
                step_prompt = (
                    f"Tâche globale : {task}\n"
                    f"Plan : {'; '.join(steps)}\n"
                    f"{exist_line}"
                    f"Contexte des étapes précédentes :\n{context[-_CTX_CLIP:]}\n\n"
                    f"Réalise UNIQUEMENT l'étape {i} : « {step_title} », "
                    "MAINTENANT et entièrement. Ne diffère rien à une étape ou "
                    "une demande ultérieure ; produis le ou les fichiers de "
                    "cette étape dès maintenant.\n"
                    "Si l'étape produit un ou plusieurs fichiers, donne un bloc "
                    "de code par fichier avec le chemin en 1ʳᵉ ligne en "
                    "commentaire, ex. « # file: app.py ». Un seul fichier par "
                    "rôle, pas de fichiers superflus. Réponds SANS phrase "
                    "d'introduction ni de conclusion, SANS question, SANS "
                    "formule de politesse ni demande d'avis : uniquement le ou "
                    "les blocs de code."
                )
                output = await self._gen(worker, step_prompt)

                written = self._write_files(
                    output, subdir, flat=flat, seen=seen, allow_scaffold=not flat
                )
                files_total.extend(w for w in written if "error" not in w)

                # Responsive stop: honour it right after the (long) execution
                # generation — before spending another generate on critique or
                # moving to the next step. Files already produced are kept.
                if run.stop:
                    yield {"type": "run_stopped", "run_id": run_id,
                           "at_step": i, "files": files_total}
                    return

                # Step health = did it actually produce a file? This is a real
                # signal (and one generate cheaper than asking the model to
                # rate itself, which was noisy — it flagged ⚠ even on steps
                # that wrote a valid file). The authoritative quality verdict
                # is the pytest run below.
                produced = bool([w for w in written if "error" not in w])
                doubt = 0.0 if produced else 0.85

                context += f"\n[Étape {i}] {output[:_CTX_CLIP]}"
                yield {
                    "type": "step_done",
                    "run_id": run_id,
                    "index": i,
                    "title": step_title,
                    "worker": worker.name,
                    "doubt": doubt,
                    "content": output[:4000],
                    "files": written,
                    "needs_attention": not produced,
                }

            # ── Verification: actually run the tests (sandboxed) ─────────
            exec_dict = None
            if (self.execution_enabled and self.workspace is not None
                    and abs_dir is not None and files_total and not run.stop):
                try:
                    # Pre-verify recovery #1: the task asked for tests, a module
                    # exists, but no test file was produced (the model deferred
                    # or skipped the test step). Generate the tests now.
                    wants_tests = bool(re.search(
                        r"\btests?\b|unittest|pytest", task, re.I))
                    existing_py = [
                        f for f in self._existing_files(abs_dir)
                        if f.endswith(".py")
                    ]
                    def _is_test(f):
                        b = os.path.basename(f)
                        return b.startswith("test_") or b.endswith("_test.py")
                    mods = [f for f in existing_py if not _is_test(f)]
                    if (wants_tests and mods and not any(_is_test(f) for f in existing_py)
                            and not run.stop):
                        stem = os.path.basename(mods[0])[:-3]
                        syms = self._public_symbols(abs_dir, mods[0])
                        if syms:
                            imp = f"from {stem} import " + ", ".join(syms)
                            sym_line = (
                                f"Le module « {stem}.py » définit : "
                                f"{', '.join(syms)}. Importe-les exactement "
                                f"ainsi : « {imp} » (n'invente aucun autre nom).\n"
                            )
                        else:
                            sym_line = ""
                        yield {"type": "step_start", "run_id": run_id,
                               "index": len(steps) + 1,
                               "title": f"Tests manquants : test_{stem}.py"}
                        gen_prompt = (
                            f"Tâche : {task}\n"
                            f"Le module « {stem}.py » existe mais aucun test "
                            "n'a été écrit. Écris-les.\n"
                            f"{sym_line}"
                            f"Donne UNIQUEMENT « test_{stem}.py » : un seul bloc "
                            f"de code avec « # file: test_{stem}.py » en 1ʳᵉ "
                            "ligne, des tests pytest qui couvrent les cas "
                            "valides ET invalides. Pas de blabla ni autre "
                            "fichier."
                        )
                        genout = await self._gen(core, gen_prompt)
                        genw = self._write_files(
                            genout, subdir, flat=flat, seen=seen,
                            allow_scaffold=not flat,
                        )
                        files_total.extend(w for w in genw if "error" not in w)
                        if genw:
                            context += ("\n[Récupération] Fichier de tests créé : "
                                        + ", ".join(w["path"] for w in genw))
                        yield {"type": "step_done", "run_id": run_id,
                               "index": len(steps) + 1,
                               "title": f"Tests manquants : test_{stem}.py",
                               "worker": core.name,
                               "doubt": 0.0 if genw else 1.0,
                               "content": genout[:2000], "files": genw,
                               "needs_attention": not genw}

                    # Pre-verify recovery #2: a test imports a local module
                    # that is absent OR present-but-incomplete (stub without
                    # the imported symbols → ImportError at collection). (Re)write
                    # it targeting the exact symbols the tests need. Remember
                    # the names so the installer never fetches a PyPI homonym.
                    needs = self._modules_needing_impl(abs_dir)
                    local_missing = {stem for stem, _ in needs}
                    for f in self._existing_files(abs_dir):
                        if f.endswith(".py"):
                            local_missing.add(os.path.basename(f)[:-3])
                    for stem, req_syms in needs:
                        if run.stop:
                            break
                        need_line = (
                            f"Le module DOIT définir (les tests les importent) : "
                            f"{', '.join(req_syms)}. Implémente-les réellement "
                            "(pas de stub, pas de « pass »).\n"
                            if req_syms else ""
                        )
                        yield {"type": "step_start", "run_id": run_id,
                               "index": len(steps) + 1,
                               "title": f"Module à implémenter : {stem}.py"}
                        gen_prompt = (
                            f"Tâche : {task}\n"
                            f"Le fichier de tests importe « "
                            f"{', '.join(req_syms) or stem} » depuis le module "
                            f"local « {stem} », mais « {stem}.py » est absent ou "
                            "n'implémente pas ces symboles. Écris-le.\n"
                            f"{need_line}"
                            f"Donne UNIQUEMENT « {stem}.py » : un seul bloc de "
                            f"code avec « # file: {stem}.py » en 1ʳᵉ ligne, "
                            "l'implémentation complète et fonctionnelle, sans "
                            "blabla ni autre fichier."
                        )
                        genout = await self._gen(core, gen_prompt)
                        genw = self._write_files(
                            genout, subdir, flat=flat, seen=seen,
                            allow_scaffold=not flat,
                        )
                        # Salvage: model returned code but without a proper
                        # « # file: » marker → force-write it as {stem}.py so the
                        # module exists and pytest can collect (no more silent
                        # "module not written → collection error").
                        if not genw and self.workspace is not None:
                            code = _largest_code_block(genout)
                            if code and ("def " in code or "class " in code):
                                rel0 = self._norm_path(f"{stem}.py", flat)
                                sub = (subdir or "").strip().strip("/")
                                rel = (f"{sub}/{rel0}" if (sub and rel0)
                                       else (rel0 or f"{stem}.py"))
                                try:
                                    entry = self.workspace.write_text_file(rel, code)
                                    genw = [{"path": entry.path, "size": entry.size}]
                                    if seen is not None:
                                        seen[rel] = hash(code)
                                except Exception:  # noqa: BLE001
                                    log.exception("module salvage failed: %s", stem)
                        files_total.extend(w for w in genw if "error" not in w)
                        if genw:
                            context += ("\n[Récupération] Module (ré)écrit : "
                                        + ", ".join(w["path"] for w in genw))
                        yield {"type": "step_done", "run_id": run_id,
                               "index": len(steps) + 1,
                               "title": f"Module à implémenter : {stem}.py",
                               "worker": core.name,
                               "doubt": 0.0 if genw else 1.0,
                               "content": genout[:2000], "files": genw,
                               "needs_attention": not genw}

                    # Flat layout collapsed files to the root, but the model's
                    # imports may still carry the original package prefix
                    # ("from utils.email_validator import …"). Rewrite those to
                    # match the flat files so pytest imports the LOCAL module
                    # instead of triggering a bogus install of "utils".
                    if flat:
                        self._flatten_imports(abs_dir, subdir)

                    sb = self._make_sandbox(abs_dir)
                    if await self._to_thread(sb.discover_tests):
                        yield {"type": "exec_start", "run_id": run_id}
                        res = await self._to_thread(sb.run_pytest)
                        # Reactive, validated installs from a real
                        # ModuleNotFoundError — never dictated by the model.
                        from rune.agentic.sandbox import (  # noqa: PLC0415
                            _STDLIB, module_to_pypi, parse_missing_module,
                        )
                        guard = 0
                        tried_mods: set[str] = set()
                        while not res.ok and guard < sb.max_installs:
                            guard += 1
                            mod = parse_missing_module(res.stdout + "\n" + res.stderr)
                            if not mod or mod in _STDLIB or mod in local_missing:
                                break
                            if mod in tried_mods:
                                break  # already tried — don't loop on the same one
                            # A dotted local import (e.g. flattened "from utils.x")
                            # leaves a bogus top-level "utils" — never a real dep.
                            if self._is_local_pkg_prefix(abs_dir, mod):
                                break
                            tried_mods.add(mod)
                            ok_i, _ir = await self._to_thread(
                                lambda m=mod: sb.pip_install(m)
                            )
                            yield {
                                "type": "exec_install", "run_id": run_id,
                                "package": module_to_pypi(mod), "ok": bool(ok_i),
                            }
                            if not ok_i:
                                break
                            res = await self._to_thread(sb.run_pytest)
                        # Up to 2 model-driven fix attempts if tests fail.
                        # Give the model the modules' REAL public symbols so it
                        # repairs bad imports (the usual "error during
                        # collection" cause) instead of inventing names.
                        sig_lines = []
                        for f in self._existing_files(abs_dir):
                            b = os.path.basename(f)
                            if not f.endswith(".py"):
                                continue
                            if b.startswith("test_") or b.endswith("_test.py"):
                                continue
                            syms = self._public_symbols(abs_dir, f)
                            if syms:
                                sig_lines.append(f"{b} définit : {', '.join(syms)}")
                        sig_block = (
                            "Symboles publics réels (importe EXACTEMENT "
                            "ceux-ci dans les tests, n'en invente aucun) :\n- "
                            + "\n- ".join(sig_lines) + "\n"
                        ) if sig_lines else ""
                        fix_tries = 0
                        while (not res.ok and not res.timed_out
                               and fix_tries < 2):
                            fix_tries += 1
                            collect_hint = (
                                "Une « error during collection » = un mauvais "
                                "import dans le test : corrige l'import pour "
                                "utiliser les symboles ci-dessus.\n"
                                if "collection" in (res.stdout + res.stderr).lower()
                                else ""
                            )
                            fix_prompt = (
                                f"Tâche : {task}\n"
                                f"Fichiers : {', '.join(self._existing_files(abs_dir))}\n"
                                f"{sig_block}"
                                "Les tests échouent. Sortie de pytest :\n"
                                f"{(res.stdout + res.stderr)[-1500:]}\n\n"
                                f"{collect_hint}"
                                "Corrige le ou les fichiers fautifs. Redonne "
                                "UNIQUEMENT les fichiers à modifier, un bloc par "
                                "fichier avec « # file: <chemin> » en 1ʳᵉ ligne. "
                                "Pas de blabla."
                            )
                            fixout = await self._gen(core, fix_prompt)
                            fixw = self._write_files(
                                fixout, subdir, flat=flat, seen=seen,
                                allow_scaffold=not flat,
                            )
                            files_total.extend(w for w in fixw if "error" not in w)
                            if not fixw:
                                break
                            res = await self._to_thread(sb.run_pytest)
                        exec_dict = res.to_dict()
                        exec_dict["installs"] = sb.installs
                        exec_dict["isolation"] = _sandbox_mode()
                        yield {"type": "exec_result", "run_id": run_id, **exec_dict}
                except Exception:  # noqa: BLE001
                    log.exception("agent execution phase failed")

            # A file rewritten across steps/fix attempts must count ONCE:
            # deduplicate by path (last write wins) so the reported count and
            # the synthesis match what is actually on disk.
            _uniq: dict[str, dict] = {}
            for _f in files_total:
                _uniq[_f["path"]] = _f
            files_total = list(_uniq.values())

            # ── Final synthesis (truthful: folds in the test verdict) ────
            synth = ""
            try:
                paths = ", ".join(f["path"] for f in files_total) or "aucun"
                verdict = ""
                if exec_dict is not None:
                    verdict = "Tests exécutés : " + (
                        "réussis. " if exec_dict["ok"]
                        else f"échec ({exec_dict['summary']}). "
                    )
                synth_prompt = (
                    f"Tâche demandée : {task}\n"
                    f"Fichiers produits (EXISTENT sur le disque) : {paths}\n"
                    f"VERDICT FINAL DES TESTS : {verdict or 'non exécutés.'}\n\n"
                    "Rédige UNE synthèse brève (3 à 5 phrases) de l'ÉTAT FINAL : "
                    "ce qui a été produit, comment l'utiliser, et l'état des "
                    "tests TEL QU'INDIQUÉ PAR LE VERDICT FINAL ci-dessus. "
                    "IMPORTANT : décris l'état final, PAS les étapes "
                    "intermédiaires. Si le verdict final est « réussis », dis "
                    "que les tests passent — ne mentionne aucun échec passé ni "
                    "« étape suivante » de correction. Si le verdict est "
                    "« échec », dis-le honnêtement. Ne mentionne QUE les "
                    "fichiers listés ; n'en invente aucun ; ne propose pas de "
                    "les créer (ils existent). Pas de code, une seule synthèse, "
                    "sans question ni formule de politesse."
                )
                synth = await self._gen(core, synth_prompt, max_new_tokens=384,
                                        prose=True)
                synth = _clean_synthesis(synth)
            except Exception:  # noqa: BLE001
                log.exception("agent synthesis failed")
            if synth.strip():
                yield {"type": "synthesis", "run_id": run_id, "text": synth}

            # Persist the mission manifest (resume / audit).
            self._write_manifest(abs_dir, {
                "name": name, "slug": slug, "task": task, "status": "done",
                "files": [f.get("path") for f in files_total],
                "exec": exec_dict, "synthesis": synth.strip()[:2000],
            })

            yield {
                "type": "run_done",
                "run_id": run_id,
                "steps": len(steps),
                "files": files_total,
                "exec": exec_dict,
            }
        except asyncio.CancelledError:
            yield {"type": "run_stopped", "run_id": run_id, "reason": "cancelled"}
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("Agent run failed")
            yield {"type": "run_error", "run_id": run_id, "error": str(exc)}
        finally:
            # Conserve la dernière mission pour l'affichage dashboard
            # AVANT de la retirer du registre actif (évite que le
            # dashboard se vide dès la fin). marque done au cas où le
            # dernier event n'aurait pas été capturé.
            _r = self._runs.get(run_id)
            if _r is not None:
                _r.done = True
                self._last_run = (run_id, _r)
            self._runs.pop(run_id, None)
