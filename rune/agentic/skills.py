"""Agent skill library — the *import* side of the same template/recipe
mechanism as procedural memory.

A skill is a ``SKILL.md`` file (the open Agent-Skills convention): YAML-ish
frontmatter (``name``, ``description``, optional ``tags``) plus a body of
procedural instructions. The library loads a directory of them and surfaces
the most relevant one(s) for a task, reusing the SAME semantic-recall the
procedural memory uses (the orchestrator passes its cosine ``similarity_fn``;
without one, retrieval degrades to keyword overlap).

SECURITY: skills are third-party *text*. The library only ever surfaces the
frontmatter + a length-capped body as context — it NEVER executes bundled
scripts. The agent may choose to read/run files via its normal sandboxed
tools. A light forbidden-pattern guard drops obviously unsafe entries.

This module is torch-free and dependency-free so it stays unit-testable.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

_MAX_BODY = 2500          # chars of body surfaced per skill (context budget)
_MAX_DESC = 400
# Obvious prompt-injection / exfiltration markers → skill is dropped, not run.
_UNSAFE = re.compile(
    r"(ignore (all |previous )?instructions|exfiltrat|reverse shell|"
    r"curl\s+[^\n]*\|\s*(sh|bash)|rm\s+-rf\s+/|base64\s+-d\s*\|)",
    re.I,
)


@dataclass
class Skill:
    name: str
    description: str
    body: str
    source: str = "bundled"          # provenance: "bundled" | repo slug | path
    tags: list[str] = field(default_factory=list)
    path: str = ""

    def haystack(self) -> str:
        """Text used for keyword matching."""
        return f"{self.name} {self.description} {' '.join(self.tags)}".lower()


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split a ``---``-fenced YAML-ish frontmatter from the body.

    Minimal flat ``key: value`` parser (no external YAML dep). ``tags`` may be
    ``[a, b]`` or ``a, b``. Returns ``(meta, body)``; meta is ``{}`` when no
    frontmatter is present.
    """
    if not text:
        return {}, ""
    t = text.lstrip("\ufeff").lstrip()
    if not t.startswith("---"):
        return {}, text.strip()
    # Find the closing fence on its own line.
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", t, re.S)
    if not m:
        return {}, text.strip()
    raw_meta, body = m.group(1), m.group(2)
    meta: dict = {}
    for line in raw_meta.splitlines():
        line = line.rstrip()
        if not line or line.lstrip().startswith("#") or ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip().lower()
        val = val.strip().strip('"').strip("'")
        if key == "tags":
            val = val.strip("[]")
            meta[key] = [v.strip().strip('"').strip("'")
                         for v in val.split(",") if v.strip()]
        else:
            meta[key] = val
    return meta, body.strip()


def load_skill_file(path: Path, source: str = "bundled") -> Skill | None:
    """Parse one ``SKILL.md`` into a :class:`Skill`, or None if invalid/unsafe."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return None
    meta, body = parse_frontmatter(text)
    name = (meta.get("name") or Path(path).parent.name or "").strip()
    desc = (meta.get("description") or "").strip()[:_MAX_DESC]
    if not name or not desc:
        log.debug("skill %s missing name/description; skipped", path)
        return None
    if _UNSAFE.search(text):
        log.warning("skill %s dropped: matched unsafe pattern", path)
        return None
    tags = meta.get("tags") if isinstance(meta.get("tags"), list) else []
    return Skill(
        name=name, description=desc, body=body[:_MAX_BODY],
        source=source, tags=tags, path=str(path),
    )


class SkillLibrary:
    """Loads ``*/SKILL.md`` from one or more directories and retrieves the
    most relevant skills for a task."""

    def __init__(self, dirs, *, min_score: float = 0.35):
        self.dirs = [Path(d) for d in (dirs or [])]
        self.min_score = float(min_score)
        self._skills: list[Skill] = []
        self.reload()

    def reload(self) -> None:
        found: list[Skill] = []
        for d in self.dirs:
            try:
                if not d.exists():
                    continue
                for sk_file in sorted(d.glob("**/SKILL.md")):
                    sk = load_skill_file(sk_file, source=d.name or "bundled")
                    if sk is not None:
                        found.append(sk)
            except Exception:  # noqa: BLE001
                log.debug("skill dir scan failed: %s", d, exc_info=True)
        self._skills = found
        if found:
            log.info("skill library: %d skills from %s",
                     len(found), [str(d) for d in self.dirs])

    def all(self) -> list[Skill]:
        return list(self._skills)

    def retrieve(self, task: str, limit: int = 2, similarity_fn=None) -> list[Skill]:
        """Top-``limit`` skills for ``task``.

        With ``similarity_fn`` (``(a, b) -> cosine 0-1``) ranking is semantic on
        the description, floored at ``min_score``. Without it, ranking is
        keyword overlap on name/description/tags.
        """
        if not self._skills or not (task or "").strip():
            return []
        if similarity_fn is not None:
            scored = []
            for sk in self._skills:
                try:
                    s = float(similarity_fn(task, sk.description))
                except Exception:  # noqa: BLE001
                    s = 0.0
                if s >= self.min_score:
                    scored.append((s, sk))
            if scored:
                scored.sort(key=lambda x: -x[0])
                return [sk for _s, sk in scored[:limit]]
            return []  # semantic available but nothing relevant → inject nothing
        # Keyword fallback.
        toks = set(re.findall(r"\w{4,}", (task or "").lower()))
        if not toks:
            return []
        scored = []
        for sk in self._skills:
            ov = len(toks & set(re.findall(r"\w{4,}", sk.haystack())))
            if ov:
                scored.append((ov, sk))
        scored.sort(key=lambda x: -x[0])
        return [sk for _o, sk in scored[:limit]]

    @staticmethod
    def render(skills: list[Skill]) -> str:
        """Format matched skills as an injectable context block."""
        if not skills:
            return ""
        parts = []
        for sk in skills:
            block = f"### Compétence : {sk.name}\n{sk.description}"
            if sk.body:
                block += f"\n{sk.body}"
            parts.append(block)
        return "\n\n".join(parts)
