"""Multi-file code extraction from a Markdown answer.

V6 — When Rune produces a project, she emits **one fenced code block
per file**, each declaring its relative path either:

* on the first content line, as a language comment marker::

      # file: src/app.py
      // file: src/index.js
      <!-- file: index.html -->

* or in the closing-fence info string::

      ```python path=src/app.py

This module is the single source of truth for turning that Markdown into
a list of ``(path, lang, content)`` triples — reused by the HTTP routes
that offer per-file download, a ``.zip`` of the whole project, and
direct writing into the workspace sandbox. Keeping the parser server-side
(rather than scraping the rendered DOM) avoids JS/Python drift and
survives Markdown rendering quirks (``marked`` drops the info-string
tail, so the in-content marker is the robust path).

Only blocks that declare a path are returned. Anonymous snippets are
left as-is (they are not "files").
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass

__all__ = ["CodeFile", "extract_code_files", "build_zip"]


# Fence openers: 3+ backticks or tildes, optional info string after.
_FENCE_RE = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>`{3,}|~{3,})(?P<info>[^\n]*)$")

# Path declared in the fence info string: ```python path=src/app.py
# Accept path= / file= , quoted or not.
_INFO_PATH_RE = re.compile(
    r"""(?:^|\s)(?:path|file)\s*=\s*(?P<q>['"]?)(?P<path>[^'"\s]+)(?P=q)""",
    re.IGNORECASE,
)

# Path declared on the first content line, as a comment marker.
# Handles #, //, --, ;, <!-- -->, % comment leaders. Keyword: file|path.
_LINE_PATH_RE = re.compile(
    r"""^\s*
        (?:\#|//|--|;{1,2}|%|<!--)?      # optional comment leader
        \s*
        (?:file|path)\s*[:=]\s*
        (?P<path>[^\s*][^\s]*?)           # the path (no leading space/star)
        \s*
        (?:-->)?                          # optional HTML comment closer
        \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


@dataclass
class CodeFile:
    """One extracted file: its relative path, language, and content."""

    path: str
    lang: str
    content: str

    def to_dict(self) -> dict:
        return {"path": self.path, "lang": self.lang, "content": self.content}


def _normalize_path(raw: str) -> str | None:
    """Best-effort normalisation for display/zip. Returns None if unsafe.

    The authoritative safety check happens in WorkspaceManager on write;
    here we only reject the obviously broken so the zip/UI stay clean.
    """
    p = raw.strip().strip("`").replace("\\", "/")
    p = re.sub(r"^\./", "", p)
    p = p.lstrip("/")  # never absolute in a project
    if not p or "\x00" in p:
        return None
    parts = [seg for seg in p.split("/") if seg not in ("", ".")]
    if any(seg == ".." for seg in parts):
        return None
    return "/".join(parts) or None


def _lang_from_info(info: str) -> str:
    """First token of the info string is the language (marked convention)."""
    info = info.strip()
    if not info:
        return ""
    first = info.split()[0]
    # Strip a path=... if it accidentally is the first token.
    if "=" in first:
        return ""
    return first.lower()


_CODE_EXT = {
    "py", "pyi", "js", "ts", "jsx", "tsx", "mjs", "cjs", "json", "txt", "md",
    "rst", "html", "htm", "css", "scss", "sass", "less", "sh", "bash", "zsh",
    "yaml", "yml", "toml", "cfg", "ini", "env", "java", "kt", "go", "rs", "c",
    "cc", "cpp", "cxx", "h", "hpp", "rb", "php", "sql", "xml", "csv", "tsv",
    "vue", "svelte", "lua", "pl", "r", "scala", "swift", "dart", "proto",
}


def _preceding_label_path(line: str) -> str | None:
    """Path declared on the line *just before* a fence (no in-block marker).

    Handles models that write the filename above the fence, e.g.::

        file: app.py
        ```python
        ...
        ```

    Accepts ``file:``/``path:`` markers and decorated bare filenames
    (``**app.py**``, `` `app.py` ``, ``## src/app.py``). The bare form is
    gated on a known code extension to avoid matching prose like a domain.
    """
    s = (line or "").strip()
    if not s:
        return None
    lm = _LINE_PATH_RE.match(s)
    if lm:
        return lm.group("path")
    t = s.strip("*`#").strip().rstrip(":").strip().strip("*`").strip()
    m2 = re.fullmatch(r"[\w./+-]+\.([A-Za-z0-9]{1,8})", t)
    if m2 and m2.group(1).lower() in _CODE_EXT:
        return t
    return None


def extract_code_files(markdown: str) -> list[CodeFile]:
    """Parse fenced code blocks and return those that declare a file path.

    Precedence for the path: info-string ``path=`` first, then a first-line
    comment marker. When the path comes from a first-line marker, that line
    is stripped from the returned content.
    """
    if not markdown:
        return []

    lines = markdown.splitlines()
    out: list[CodeFile] = []
    i = 0
    n = len(lines)
    prev_nonempty = ""               # last non-blank line before a fence
    while i < n:
        m = _FENCE_RE.match(lines[i])
        if not m:
            if lines[i].strip():
                prev_nonempty = lines[i]
            i += 1
            continue

        fence = m.group("fence")
        fence_char = fence[0]
        fence_len = len(fence)
        info = m.group("info") or ""
        lang = _lang_from_info(info)

        # Collect body until a matching closing fence (same char, >= length).
        body: list[str] = []
        j = i + 1
        closed = False
        while j < n:
            cm = _FENCE_RE.match(lines[j])
            if (
                cm
                and cm.group("fence")[0] == fence_char
                and len(cm.group("fence")) >= fence_len
                and not cm.group("info").strip()
            ):
                closed = True
                break
            body.append(lines[j])
            j += 1

        # Determine the path(s). A single fenced block may declare MULTIPLE
        # files when the model emits several "# file: X" markers inside one
        # fence (common with smaller models). Split the body on every marker
        # line; the text before the first marker belongs to the info-string
        # path when one was given, otherwise it is discarded.
        info_match = _INFO_PATH_RE.search(info)
        info_path = info_match.group("path") if info_match else None

        segments: list[tuple[str, list[str]]] = []
        cur_path: str | None = info_path
        cur_lines: list[str] = []
        for bl in body:
            lm = _LINE_PATH_RE.match(bl)
            if lm:
                if cur_path is not None and any(s.strip() for s in cur_lines):
                    segments.append((cur_path, cur_lines))
                cur_path = lm.group("path")
                cur_lines = []
            else:
                cur_lines.append(bl)
        if cur_path is not None and any(s.strip() for s in cur_lines):
            segments.append((cur_path, cur_lines))

        # Fallback: no path anywhere in/at the fence → use the line just
        # before the fence as a label (the model wrote the name above it).
        if not segments and any(s.strip() for s in body):
            lbl = _preceding_label_path(prev_nonempty)
            if lbl:
                segments = [(lbl, body)]

        for p_raw, p_lines in segments:
            path = _normalize_path(p_raw)
            if not path:
                continue
            content = "\n".join(p_lines)
            if content and not content.endswith("\n"):
                content += "\n"
            out.append(CodeFile(path=path, lang=lang, content=content))

        # Advance past the closing fence (or to EOF if unterminated).
        prev_nonempty = ""           # consumed; don't reuse for next block
        i = (j + 1) if closed else j

    # De-duplicate by path, last write wins (matches "iterate on same file").
    dedup: dict[str, CodeFile] = {}
    for cf in out:
        dedup[cf.path] = cf
    return list(dedup.values())


def build_zip(files: list[CodeFile], *, mtime: tuple | None = None) -> bytes:
    """Build an in-memory ``.zip`` from extracted files. Deterministic.

    ``mtime`` is an optional 6-tuple (Y, M, D, h, m, s) for reproducible
    archives; defaults to a fixed epoch so re-zipping identical content
    yields identical bytes.
    """
    date_time = mtime or (1980, 1, 1, 0, 0, 0)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for cf in files:
            zi = zipfile.ZipInfo(cf.path, date_time=date_time)
            zi.external_attr = 0o644 << 16
            zf.writestr(zi, cf.content)
    return buf.getvalue()
