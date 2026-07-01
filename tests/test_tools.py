"""Tests for the bounded tool-calling layer (parser + dispatch). Pure."""

from __future__ import annotations

from rune.agentic.tools import (
    dispatch,
    has_tool_call,
    parse_tool_calls,
    tools_prompt,
)


def test_parse_basic_tool_call():
    txt = 'blah <tool_call>{"name": "run_tests", "arguments": {}}</tool_call> end'
    calls = parse_tool_calls(txt)
    assert calls == [{"name": "run_tests", "arguments": {}}]
    assert has_tool_call(txt)


def test_parse_repairs_trailing_comma_and_fences():
    txt = '<tool_call>```json\n{"name": "read_file", "arguments": {"path": "a.py",},}\n```</tool_call>'
    calls = parse_tool_calls(txt)
    assert calls[0]["name"] == "read_file"
    assert calls[0]["arguments"]["path"] == "a.py"


def test_parse_accepts_args_alias_and_string_args():
    txt = '<tool_call>{"tool": "write_file", "parameters": "{\\"path\\": \\"x.py\\", \\"content\\": \\"y\\"}"}</tool_call>'
    calls = parse_tool_calls(txt)
    assert calls[0]["name"] == "write_file"
    assert calls[0]["arguments"]["path"] == "x.py"


def test_no_tool_call():
    assert parse_tool_calls("just prose, done") == []
    assert has_tool_call("nope") is False


def test_tools_prompt_is_json_menu():
    p = tools_prompt()
    assert "run_tests" in p and "write_file" in p


# ── dispatch ───────────────────────────────────────────────────────────
def _ops(log):
    return {
        "list_files": lambda: ["a.py"],
        "read_file": lambda path: f"content-of-{path}",
        "write_file": lambda path, content: log.append((path, content)) or {"path": path, "size": len(content)},
        "run_tests": lambda: {"ok": True, "summary": "1 passed"},
    }


def test_dispatch_write_and_read_and_tests():
    log = []
    ops = _ops(log)
    assert dispatch({"name": "list_files", "arguments": {}}, ops)["result"] == ["a.py"]
    r = dispatch({"name": "read_file", "arguments": {"path": "a.py"}}, ops)
    assert r["ok"] and r["result"] == "content-of-a.py"
    w = dispatch({"name": "write_file", "arguments": {"path": "b.py", "content": "x"}}, ops)
    assert w["ok"] and log == [("b.py", "x")]
    t = dispatch({"name": "run_tests", "arguments": {}}, ops)
    assert t["ok"] and t["result"]["ok"] is True


def test_dispatch_rejects_traversal_and_unknown():
    ops = _ops([])
    assert dispatch({"name": "read_file", "arguments": {"path": "../../etc/passwd"}}, ops)["ok"] is False
    assert dispatch({"name": "write_file", "arguments": {"path": "/abs", "content": "x"}}, ops)["ok"] is False
    assert dispatch({"name": "python", "arguments": {"code": "x"}}, ops)["ok"] is False


def test_dispatch_missing_args():
    ops = _ops([])
    assert dispatch({"name": "read_file", "arguments": {}}, ops)["ok"] is False


# ── run_command allowlist + new tools (A/B/C/E) ──────────────────────
from rune.agentic.tools import command_needs_net, validate_command  # noqa: E402


def test_validate_command_allowlist():
    assert validate_command(["npm", "install"])[0] is True
    assert validate_command(["npm", "run", "build"])[0] is True
    assert validate_command(["npm", "publish"])[0] is False        # sub not allowed
    assert validate_command(["rm", "-rf", "/"])[0] is False         # exe not allowed
    assert validate_command(["python", "app.py"])[0] is True
    assert validate_command(["pip", "install", "flask"])[0] is True
    assert validate_command(["pip", "uninstall", "x"])[0] is False
    assert validate_command([])[0] is False


def test_command_needs_net():
    assert command_needs_net(["npm", "install"]) is True
    assert command_needs_net(["python", "-m", "pip", "install", "flask"]) is True
    assert command_needs_net(["npx", "create-vite"]) is True
    assert command_needs_net(["npm", "run", "build"]) is False
    assert command_needs_net(["pytest"]) is False


def _ops_ext():
    log = {"cmd": [], "edit": [], "serve": []}
    ops = {
        "list_files": lambda: [],
        "read_file": lambda p: "",
        "write_file": lambda p, c: {"path": p},
        "run_tests": lambda: {"ok": True},
        "run_command": lambda argv, net: log["cmd"].append((argv, net)) or {"ok": True},
        "edit_file": lambda p, f, r: log["edit"].append((p, f, r)) or {"ok": True, "replaced": 1},
        "serve_and_probe": lambda argv, paths, port: log["serve"].append((argv, paths, port)) or {"started": True},
    }
    return ops, log


def test_dispatch_run_command_validated_and_net():
    ops, log = _ops_ext()
    r = dispatch({"name": "run_command", "arguments": {"command": "npm install"}}, ops)
    assert r["ok"] and log["cmd"][0][1] is True          # net=True for install
    b = dispatch({"name": "run_command", "arguments": {"command": "curl http://x"}}, ops)
    assert b["ok"] is False                              # curl not allowlisted
    rb = dispatch({"name": "run_command", "arguments": {"command": "npm run build"}}, ops)
    assert rb["ok"] and log["cmd"][1][1] is False        # build = no net


def test_dispatch_edit_file():
    ops, log = _ops_ext()
    e = dispatch({"name": "edit_file", "arguments": {"path": "a.py", "find": "x", "replace": "y"}}, ops)
    assert e["ok"] and log["edit"] == [("a.py", "x", "y")]
    miss = dispatch({"name": "edit_file", "arguments": {"path": "a.py", "find": ""}}, ops)
    assert miss["ok"] is False


def test_dispatch_new_tools_delete_search_run_python():
    ops = {
        "delete_file": lambda path: {"ok": True, "deleted": path},
        "search_files": lambda q, g="*": {"ok": True,
                                          "matches": [f"a.py:1: {q}"], "count": 1},
        "run_python": lambda code: {"ok": True, "stdout": "42"},
    }
    d = dispatch({"name": "delete_file", "arguments": {"path": "a.py"}}, ops)
    assert d["ok"] and d["result"]["deleted"] == "a.py"
    s = dispatch({"name": "search_files",
                  "arguments": {"query": "foo", "glob": "*.py"}}, ops)
    assert s["ok"] and s["result"]["count"] == 1
    p = dispatch({"name": "run_python", "arguments": {"code": "print(42)"}}, ops)
    assert p["ok"] and p["result"]["stdout"] == "42"
    # missing required args → structured error, not a crash
    assert dispatch({"name": "delete_file", "arguments": {}}, ops)["ok"] is False
    assert dispatch({"name": "search_files", "arguments": {}}, ops)["ok"] is False
    assert dispatch({"name": "run_python", "arguments": {}}, ops)["ok"] is False


def test_dispatch_run_command_accepts_argv_list():
    seen = {}
    ops = {"run_command": lambda argv, net: seen.setdefault("argv", argv) or {"ok": True}}
    r = dispatch({"name": "run_command",
                  "arguments": {"argv": ["python", "x.py"]}}, ops)
    assert r["ok"] and seen["argv"] == ["python", "x.py"]


def test_dispatch_web_search():
    ops = {"web_search": lambda q, n=5: {"ok": True, "query": q, "count": 1,
                                         "results": [{"title": "t", "url": "u",
                                                      "extrait": "e"}]}}
    r = dispatch({"name": "web_search",
                  "arguments": {"query": "qwen3 coder", "max_results": 3}}, ops)
    assert r["ok"] and r["result"]["count"] == 1
    assert dispatch({"name": "web_search", "arguments": {}}, ops)["ok"] is False


def test_dispatch_web_fetch():
    ops = {"web_fetch": lambda url: {"ok": True, "url": url, "content": "hello"}}
    r = dispatch({"name": "web_fetch", "arguments": {"url": "https://x.io"}}, ops)
    assert r["ok"] and r["result"]["content"] == "hello"
    assert dispatch({"name": "web_fetch", "arguments": {}}, ops)["ok"] is False


def test_dispatch_serve_and_probe():
    ops, log = _ops_ext()
    s = dispatch({"name": "serve_and_probe", "arguments": {
        "command": "uvicorn app:app --port 8000", "paths": ["/", "/health"], "port": 8000}}, ops)
    assert s["ok"] and log["serve"][0][2] == 8000
    bad = dispatch({"name": "serve_and_probe", "arguments": {"command": "./evil"}}, ops)
    assert bad["ok"] is False


def test_parse_bare_json_tool_call_no_tags():
    """7B often omits <tool_call> tags and prints raw JSON; with braces inside
    the content (regex {2,}) that must not break brace matching."""
    out = (
        '{"name": "write_file", "arguments": {"path": "email_validator.py", '
        '"content": "import re\\ndef v(e):\\n return re.match(r\'a{2,}\', e)"}} ```'
    )
    calls = parse_tool_calls(out)
    assert len(calls) == 1
    assert calls[0]["name"] == "write_file"
    assert calls[0]["arguments"]["path"] == "email_validator.py"
    assert "{2,}" in calls[0]["arguments"]["content"]


def test_parse_bare_json_in_fence():
    out = '```json\n{"name": "run_tests", "arguments": {}}\n```'
    calls = parse_tool_calls(out)
    assert len(calls) == 1 and calls[0]["name"] == "run_tests"


def test_parse_flat_args_form():
    out = '{"name": "read_file", "path": "app.py"}'
    calls = parse_tool_calls(out)
    assert calls and calls[0]["name"] == "read_file"
    assert calls[0]["arguments"].get("path") == "app.py"


def test_bare_json_unknown_name_ignored():
    out = '{"name": "definitely_not_a_tool", "arguments": {"x": 1}}'
    assert parse_tool_calls(out) == []


def test_tagged_still_takes_precedence_and_bare_ignored_when_tagged():
    out = ('<tool_call>{"name":"list_files","arguments":{}}</tool_call> '
           'plus {"name":"run_tests","arguments":{}}')
    calls = parse_tool_calls(out)
    assert len(calls) == 1 and calls[0]["name"] == "list_files"


def test_has_tool_call_detects_bare():
    assert has_tool_call('{"name": "list_files", "arguments": {}}') is True
    assert has_tool_call("juste de la prose") is False
