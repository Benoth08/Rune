"""Tests for lythea.server.codegen — multi-file extraction + zip.

Pure-Python (no torch / chromadb), so part of the always-run whitelist.
"""

from __future__ import annotations

import io
import zipfile

from rune.server.codegen import CodeFile, build_zip, extract_code_files


SAMPLE = """Voici le projet.

```python
# file: src/app.py
def main():
    print("hi")
```

Le test :

```python path=tests/test_app.py
from src.app import main
def test_main():
    main()
```

Le front :

```html
<!-- file: web/index.html -->
<!doctype html><title>X</title>
```

Un snippet anonyme (ne doit PAS devenir un fichier) :

```bash
ls -la
```

Readme via marqueur sans commentaire de langage :

```
file: README.md
# Mon projet
```
"""


def test_extracts_only_path_tagged_blocks():
    files = extract_code_files(SAMPLE)
    paths = {f.path for f in files}
    assert paths == {
        "src/app.py",
        "tests/test_app.py",
        "web/index.html",
        "README.md",
    }


def test_anonymous_snippet_is_ignored():
    files = extract_code_files(SAMPLE)
    joined = "\n".join(f.content for f in files)
    assert "ls -la" not in joined


def test_marker_line_is_stripped_from_content():
    files = {f.path: f for f in extract_code_files(SAMPLE)}
    assert "file:" not in files["src/app.py"].content.lower()
    assert "file:" not in files["web/index.html"].content.lower()
    assert files["src/app.py"].content.startswith("def main")


def test_lang_detected_from_info_string():
    files = {f.path: f for f in extract_code_files(SAMPLE)}
    assert files["src/app.py"].lang == "python"
    assert files["web/index.html"].lang == "html"


def test_path_traversal_rejected():
    md = "```python\n# file: ../../etc/passwd\nx = 1\n```"
    assert extract_code_files(md) == []


def test_absolute_path_is_neutralised():
    md = "```python\n# file: /etc/passwd\nx = 1\n```"
    files = extract_code_files(md)
    assert len(files) == 1
    assert files[0].path == "etc/passwd"


def test_dedup_last_write_wins():
    md = (
        "```python\n# file: a.py\nx = 1\n```\n"
        "```python\n# file: a.py\nx = 2\n```\n"
    )
    files = extract_code_files(md)
    assert len(files) == 1
    assert "x = 2" in files[0].content


def test_empty_input_returns_empty():
    assert extract_code_files("") == []
    assert extract_code_files("just prose, no fences") == []


def test_build_zip_roundtrip():
    files = extract_code_files(SAMPLE)
    blob = build_zip(files)
    zf = zipfile.ZipFile(io.BytesIO(blob))
    assert set(zf.namelist()) == {f.path for f in files}
    # content of one file survives the round-trip
    assert b"def main" in zf.read("src/app.py")


def test_build_zip_is_deterministic():
    files = [CodeFile(path="a.py", lang="python", content="x = 1\n")]
    assert build_zip(files) == build_zip(files)


def test_multiple_files_in_one_block():
    """A single fenced block with several '# file:' markers yields one
    CodeFile per marker (common small-model output shape)."""
    md = (
        "Voici le module et ses tests :\n\n"
        "```python\n"
        "# file: email_validator.py\n"
        "def validate_email(addr):\n"
        "    return '@' in addr\n"
        "\n"
        "# file: test_email_validator.py\n"
        "from email_validator import validate_email\n"
        "def test_ok():\n"
        "    assert validate_email('a@b.co')\n"
        "```\n"
    )
    files = extract_code_files(md)
    paths = {f.path for f in files}
    assert paths == {"email_validator.py", "test_email_validator.py"}
    mod = next(f for f in files if f.path == "email_validator.py")
    tst = next(f for f in files if f.path == "test_email_validator.py")
    assert "def validate_email" in mod.content
    # the second marker must NOT leak into the first file
    assert "test_email_validator" not in mod.content
    assert "def test_ok" in tst.content
    assert "def validate_email" not in tst.content


def test_single_info_path_block_unchanged():
    """Backward-compat: one block, path in the info string, no inner markers."""
    md = "```python path=solver.py\nx = 1\n```\n"
    files = extract_code_files(md)
    assert len(files) == 1
    assert files[0].path == "solver.py"
    assert files[0].content == "x = 1\n"


def test_path_label_on_line_before_fence():
    """Model writes 'file: x.py' ABOVE the fence (fence info is just lang)."""
    md = (
        "file: email_validator.py\n"
        "```python\n"
        "import re\n"
        "def validate_email(e):\n"
        "    return '@' in e\n"
        "```\n"
    )
    files = extract_code_files(md)
    assert len(files) == 1
    assert files[0].path == "email_validator.py"
    assert "def validate_email" in files[0].content
    assert "file:" not in files[0].content  # label line not in content


def test_decorated_filename_label_before_fence():
    md = "**solver.py**\n```python\nx = 1\n```\n"
    files = extract_code_files(md)
    assert [f.path for f in files] == ["solver.py"]
    assert files[0].content == "x = 1\n"


def test_prose_before_fence_is_not_a_path():
    """A non-filename line (e.g. a domain) must not be taken as a path."""
    md = "Visitez example.com pour la doc\n```python\nx = 1\n```\n"
    assert extract_code_files(md) == []
