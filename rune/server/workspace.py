"""Workspace operations — file management for the Lythéa sandbox.

V6.0.0-α2 — The workspace is a shared directory between the user
(via the browser UI) and Lythéa (via the filesystem MCP). This
module provides the HTTP layer that the UI uses to list, upload,
download, delete and rename files.

Lythéa accesses the same files via her filesystem MCP server, which
is scoped to the same sandbox directory. So a file you drop in the
sidebar becomes immediately readable by Lythéa via natural language
("analyse le CSV que je viens d'uploader").

Security model
--------------
- All paths are normalised and validated to stay inside the sandbox
  root. Any attempt at path traversal (``../``) is rejected.
- Upload size limits enforced from settings.
- Executable file extensions blocked (.exe, .so, .dll, .dylib).
- No symlinks created or followed outside the sandbox.

Public API
----------
The module exposes a single class :class:`WorkspaceManager` that
the HTTP routes use. It's bound to one sandbox directory at init
and exposes :meth:`list_tree`, :meth:`save_upload`, etc.
"""

from __future__ import annotations

import logging
import mimetypes
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# Tooling / sandbox dirs that must not count toward the workspace quota nor
# appear in the file tree — a per-mission .venv alone is thousands of files.
_IGNORE_DIRS = {
    ".venv", "venv", "__pycache__", ".pytest_cache", ".git", ".mypy_cache",
    ".ruff_cache", "node_modules", ".tox", ".eggs",
}

log = logging.getLogger("rune.server.workspace")


# ── Configuration ─────────────────────────────────────────────────────

# Extensions blocked for upload — executable / library files.
# Goal : reduce attack surface if someone uploads something malicious.
# Not exhaustive — true sandbox isolation is the layer below (Lythéa
# doesn't *execute* files from the workspace, she only reads them).
BLOCKED_EXTENSIONS: frozenset[str] = frozenset({
    ".exe", ".com", ".bat", ".cmd", ".sh", ".ps1",
    ".so", ".dll", ".dylib", ".o",
    ".msi", ".dmg", ".pkg", ".deb", ".rpm",
})

# Filename character set : alphanumeric, common punctuation, accents.
# We strip anything else from filenames at upload time to keep the
# filesystem tidy.
_FILENAME_SAFE_RE = re.compile(r"[^\w\s\-\.\(\)\[\]àâäçéèêëîïôöùûüÿñÀÂÄÇÉÈÊËÎÏÔÖÙÛÜŸÑ]")


# ── Data shapes ───────────────────────────────────────────────────────


@dataclass
class FileEntry:
    """One entry in the workspace tree (file or directory).

    Serialised as JSON for the UI. Sizes in bytes, mtime as Unix ts.
    """

    name: str
    path: str       # relative to sandbox root, POSIX separators
    is_dir: bool
    size: int = 0    # 0 for directories
    mtime: float = 0.0
    mime: str = ""   # empty for directories
    children: list["FileEntry"] = field(default_factory=list)

    def to_dict(self) -> dict:
        d: dict = {
            "name": self.name,
            "path": self.path,
            "is_dir": self.is_dir,
            "size": self.size,
            "mtime": self.mtime,
        }
        if not self.is_dir:
            d["mime"] = self.mime
        else:
            d["children"] = [c.to_dict() for c in self.children]
        return d


@dataclass
class WorkspaceStats:
    """Summary statistics for UI display."""

    total_files: int
    total_size_bytes: int
    max_size_bytes: int   # from settings

    def to_dict(self) -> dict:
        used_pct = (
            self.total_size_bytes / self.max_size_bytes * 100
            if self.max_size_bytes > 0 else 0.0
        )
        return {
            "total_files": self.total_files,
            "total_size_bytes": self.total_size_bytes,
            "max_size_bytes": self.max_size_bytes,
            "used_pct": round(used_pct, 1),
        }


# ── Exceptions ────────────────────────────────────────────────────────


class WorkspaceError(Exception):
    """Base class for workspace operation errors."""


class WorkspacePathError(WorkspaceError):
    """Invalid path (traversal attempt, outside sandbox, malformed)."""


class WorkspaceSizeError(WorkspaceError):
    """File too large or workspace would exceed total quota."""


class WorkspaceTypeError(WorkspaceError):
    """File extension blocked."""


# ── Manager ───────────────────────────────────────────────────────────


class WorkspaceManager:
    """Operations on the workspace sandbox directory.

    Bound at init to one absolute path. All operations take paths
    *relative to the sandbox root* — the manager resolves and
    validates them internally. Callers can never pass an absolute
    path or escape via ``../``.

    Parameters
    ----------
    sandbox_dir : Path
        Absolute path of the workspace root. Created if missing.
    max_file_bytes : int
        Per-file upload limit.
    max_total_bytes : int
        Cumulative limit across the whole workspace.
    """

    def __init__(
        self,
        sandbox_dir: Path,
        max_file_bytes: int,
        max_total_bytes: int,
    ) -> None:
        self.root = Path(sandbox_dir).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_file_bytes = max_file_bytes
        self.max_total_bytes = max_total_bytes

    # ── Path validation ──────────────────────────────────────────────

    def _resolve_safe(self, rel_path: str) -> Path:
        """Resolve a relative path inside the sandbox, with safety checks.

        Raises :class:`WorkspacePathError` if :
        - path is absolute
        - path tries to escape via ``..``
        - resolved path is outside the sandbox

        Returns an absolute :class:`Path`.
        """
        # Reject absolute paths up-front
        if rel_path.startswith("/") or (
            len(rel_path) > 1 and rel_path[1] == ":"
        ):
            raise WorkspacePathError(f"Absolute paths not allowed: {rel_path!r}")
        # Reject NULL bytes (some filesystems explode)
        if "\x00" in rel_path:
            raise WorkspacePathError("Path contains NULL byte")
        # Normalise and resolve
        candidate = (self.root / rel_path).resolve()
        # Final check : the resolved path must be inside root
        try:
            candidate.relative_to(self.root)
        except ValueError:
            raise WorkspacePathError(
                f"Path escapes sandbox: {rel_path!r}"
            ) from None
        return candidate

    def _sanitize_filename(self, name: str) -> str:
        """Clean a filename : strip path separators and weird chars."""
        # Remove any directory component
        name = Path(name).name
        # Strip control chars and weirdness
        name = _FILENAME_SAFE_RE.sub("_", name)
        # Trim and collapse whitespace
        name = " ".join(name.split())
        # Avoid empty
        if not name or name in (".", ".."):
            name = "file"
        # Cap length (filesystem limits + URLs)
        if len(name) > 200:
            stem, _, ext = name.rpartition(".")
            if ext and len(ext) < 10:
                name = stem[:200 - len(ext) - 1] + "." + ext
            else:
                name = name[:200]
        return name

    def _check_extension(self, filename: str) -> None:
        """Raise :class:`WorkspaceTypeError` if extension is blocked."""
        ext = Path(filename).suffix.lower()
        if ext in BLOCKED_EXTENSIONS:
            raise WorkspaceTypeError(
                f"Extension {ext!r} blocked for security. "
                f"Blocked: {sorted(BLOCKED_EXTENSIONS)}"
            )

    # ── Listing ──────────────────────────────────────────────────────

    def list_tree(self, max_depth: int = 5) -> FileEntry:
        """Return the workspace as a tree rooted at the sandbox.

        Parameters
        ----------
        max_depth : int
            Cap traversal to avoid huge responses on accidental
            deep nesting. 5 is plenty for normal use.

        Returns
        -------
        FileEntry
            The root entry, ``is_dir=True``, with nested ``children``.
        """
        return self._walk(self.root, "", max_depth)

    def _walk(self, abs_path: Path, rel_path: str, depth: int) -> FileEntry:
        """Recursively walk a directory and build the tree."""
        try:
            stat = abs_path.stat()
        except OSError:
            stat = None

        is_dir = abs_path.is_dir()
        entry = FileEntry(
            name=abs_path.name or "sandbox",
            path=rel_path,
            is_dir=is_dir,
            size=stat.st_size if (stat and not is_dir) else 0,
            mtime=stat.st_mtime if stat else 0.0,
        )

        if is_dir and depth > 0:
            try:
                children = sorted(
                    abs_path.iterdir(),
                    # Dirs first, then files, both alphabetically
                    key=lambda p: (not p.is_dir(), p.name.lower()),
                )
            except OSError:
                children = []
            for child in children:
                # Skip hidden files (dotfiles) + tooling dirs (.venv shown as
                # dotfile already; __pycache__ etc. are not dotfiles).
                if child.name.startswith(".") and child.name not in (".gitkeep",):
                    continue
                if child.name in _IGNORE_DIRS:
                    continue
                child_rel = (
                    f"{rel_path}/{child.name}" if rel_path else child.name
                )
                entry.children.append(
                    self._walk(child, child_rel, depth - 1)
                )
        elif not is_dir:
            mime, _ = mimetypes.guess_type(abs_path.name)
            entry.mime = mime or "application/octet-stream"

        return entry

    def stats(self) -> WorkspaceStats:
        """Compute total file count and cumulative size, EXCLUDING tooling
        dirs (per-mission .venv, __pycache__, caches). Those are sandbox
        internals — a single .venv is thousands of files / tens of MB and must
        not count against the user's workspace quota or inflate the meter."""
        total_files = 0
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if d not in _IGNORE_DIRS]
            for fn in filenames:
                try:
                    total_size += (Path(dirpath) / fn).stat().st_size
                    total_files += 1
                except OSError:
                    pass
        return WorkspaceStats(
            total_files=total_files,
            total_size_bytes=total_size,
            max_size_bytes=self.max_total_bytes,
        )

    # ── Upload ───────────────────────────────────────────────────────

    def save_upload(
        self,
        filename: str,
        data: bytes,
        target_dir: str = "",
    ) -> FileEntry:
        """Save uploaded bytes to the workspace.

        Validates extension, file size, and total quota before writing.

        Parameters
        ----------
        filename : str
            Original filename from the upload. Will be sanitised.
        data : bytes
            Raw file contents.
        target_dir : str
            Relative subdirectory to save into. Empty = root.

        Returns
        -------
        FileEntry
            Descriptor of the saved file.

        Raises
        ------
        WorkspaceTypeError, WorkspaceSizeError, WorkspacePathError
        """
        # Validate size before extension (cheaper to fail-fast)
        size = len(data)
        if size > self.max_file_bytes:
            raise WorkspaceSizeError(
                f"File too large: {size} bytes > "
                f"{self.max_file_bytes} byte limit"
            )

        # Validate quota
        current = self.stats()
        if current.total_size_bytes + size > self.max_total_bytes:
            raise WorkspaceSizeError(
                f"Workspace quota would be exceeded: "
                f"{current.total_size_bytes} + {size} > "
                f"{self.max_total_bytes}"
            )

        # Validate extension
        clean_name = self._sanitize_filename(filename)
        self._check_extension(clean_name)

        # Resolve target dir, ensure it exists
        target = self._resolve_safe(target_dir) if target_dir else self.root
        if not target.exists():
            target.mkdir(parents=True, exist_ok=True)
        elif not target.is_dir():
            raise WorkspacePathError(
                f"Target is not a directory: {target_dir!r}"
            )

        # Avoid overwriting : if file exists, append a counter
        dest = target / clean_name
        if dest.exists():
            stem = dest.stem
            ext = dest.suffix
            i = 1
            while dest.exists() and i < 1000:
                dest = target / f"{stem} ({i}){ext}"
                i += 1

        # Write atomically (write to temp then rename)
        tmp = dest.with_suffix(dest.suffix + ".uploading")
        try:
            tmp.write_bytes(data)
            tmp.rename(dest)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

        # Build the FileEntry
        stat = dest.stat()
        rel = str(dest.relative_to(self.root)).replace("\\", "/")
        mime, _ = mimetypes.guess_type(dest.name)
        log.info(
            "Workspace upload: %s (%d bytes, mime=%s)",
            rel, stat.st_size, mime or "?",
        )
        return FileEntry(
            name=dest.name,
            path=rel,
            is_dir=False,
            size=stat.st_size,
            mtime=stat.st_mtime,
            mime=mime or "application/octet-stream",
        )

    # ── Download ─────────────────────────────────────────────────────

    def open_for_download(self, rel_path: str) -> tuple[Path, str]:
        """Validate and return the absolute path + mime for download.

        The caller is responsible for streaming the file.
        """
        abs_path = self._resolve_safe(rel_path)
        if not abs_path.exists():
            raise WorkspacePathError(f"File not found: {rel_path!r}")
        if abs_path.is_dir():
            raise WorkspacePathError(f"Cannot download a directory: {rel_path!r}")
        mime, _ = mimetypes.guess_type(abs_path.name)
        return abs_path, mime or "application/octet-stream"

    def iter_chunks(self, abs_path: Path, chunk_size: int = 64 * 1024) -> Iterator[bytes]:
        """Stream a file in chunks. Used by FastAPI's StreamingResponse."""
        with abs_path.open("rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    # ── Delete / rename ──────────────────────────────────────────────

    def exists(self, rel_path: str) -> bool:
        """True if a file or directory exists at ``rel_path`` in the sandbox."""
        try:
            return self._resolve_safe(rel_path).exists()
        except Exception:  # noqa: BLE001 — traversal / bad path → treat as absent
            return False

    def delete(self, rel_path: str) -> None:
        """Remove a file or empty directory.

        Non-empty directories are deleted recursively (intentional —
        the UI confirms with the user before calling).
        """
        abs_path = self._resolve_safe(rel_path)
        if not abs_path.exists():
            raise WorkspacePathError(f"Not found: {rel_path!r}")
        if abs_path == self.root:
            raise WorkspacePathError("Cannot delete the workspace root")
        if abs_path.is_dir():
            shutil.rmtree(abs_path)
        else:
            abs_path.unlink()
        log.info("Workspace delete: %s", rel_path)

    def rename(self, old_rel: str, new_name: str) -> FileEntry:
        """Rename a file or directory in place (keeps the parent).

        ``new_name`` is just the new basename — not a path. To move
        to a different directory, use a different endpoint (not
        implemented in V6.0.0-α2, may come later if needed).
        """
        clean_new = self._sanitize_filename(new_name)
        self._check_extension(clean_new)
        abs_path = self._resolve_safe(old_rel)
        if not abs_path.exists():
            raise WorkspacePathError(f"Not found: {old_rel!r}")
        dest = abs_path.parent / clean_new
        if dest.exists():
            raise WorkspacePathError(
                f"Destination already exists: {clean_new!r}"
            )
        abs_path.rename(dest)
        stat = dest.stat()
        rel = str(dest.relative_to(self.root)).replace("\\", "/")
        mime, _ = mimetypes.guess_type(dest.name) if not dest.is_dir() else (None, None)
        log.info("Workspace rename: %s → %s", old_rel, rel)
        return FileEntry(
            name=dest.name,
            path=rel,
            is_dir=dest.is_dir(),
            size=stat.st_size if not dest.is_dir() else 0,
            mtime=stat.st_mtime,
            mime=mime or "" if dest.is_dir() else mime or "application/octet-stream",
        )

    # ── Lythéa side : write generated code into the sandbox ──────────

    def write_text_file(
        self,
        rel_path: str,
        content: str,
        *,
        overwrite: bool = True,
    ) -> FileEntry:
        """Write a (sub-path) text file into the sandbox, creating dirs.

        Used by the codegen commit route to materialise a multi-file
        answer into the workspace. Unlike :meth:`save_upload`, this takes
        a full *relative path* (``src/app.py``) and overwrites by default
        — generated projects are iterated on, not accumulated.

        The upload extension blocklist is intentionally NOT applied: the
        payload is UTF-8 text authored by Lythéa (the same content she
        could already write via her filesystem MCP), so binary-executable
        concerns don't apply. Path-traversal safety is enforced by
        :meth:`_resolve_safe`.

        Raises
        ------
        WorkspaceSizeError, WorkspacePathError
        """
        data = content.encode("utf-8")
        size = len(data)
        if size > self.max_file_bytes:
            raise WorkspaceSizeError(
                f"File too large: {size} bytes > {self.max_file_bytes} limit"
            )

        abs_path = self._resolve_safe(rel_path)
        if abs_path.is_dir():
            raise WorkspacePathError(f"Target is a directory: {rel_path!r}")

        # Quota check on the net delta (overwriting reclaims old bytes).
        existing = abs_path.stat().st_size if abs_path.exists() else 0
        projected = self.stats().total_size_bytes - existing + size
        if projected > self.max_total_bytes:
            raise WorkspaceSizeError(
                f"Workspace quota would be exceeded: {projected} > "
                f"{self.max_total_bytes}"
            )

        if abs_path.exists() and not overwrite:
            raise WorkspacePathError(f"File exists: {rel_path!r}")

        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(content, encoding="utf-8")
        log.info("workspace write: %s (%d bytes)", rel_path, size)
        return self.file_metadata(
            str(abs_path.relative_to(self.root)).replace("\\", "/")
        )

    # ── Lythéa side : offer a file in chat ───────────────────────────

    def file_metadata(self, rel_path: str) -> FileEntry:
        """Build a FileEntry for a path Lythéa wants to offer for download.

        Called when Lythéa creates a file via MCP filesystem and the
        hippocampe wants to emit a ``workspace_file_offer`` SSE event.
        """
        abs_path = self._resolve_safe(rel_path)
        if not abs_path.exists() or abs_path.is_dir():
            raise WorkspacePathError(
                f"Cannot offer download: {rel_path!r}"
            )
        stat = abs_path.stat()
        mime, _ = mimetypes.guess_type(abs_path.name)
        return FileEntry(
            name=abs_path.name,
            path=str(abs_path.relative_to(self.root)).replace("\\", "/"),
            is_dir=False,
            size=stat.st_size,
            mtime=stat.st_mtime,
            mime=mime or "application/octet-stream",
        )
