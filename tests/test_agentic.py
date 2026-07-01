"""Tests for the V6 agentic layer (workers + orchestrator).

Pure-Python: a fake worker + a temp workspace, no torch/chromadb, so part
of the always-run whitelist.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from rune.agentic.orchestrator import AgentOrchestrator, _MAX_STEPS
from rune.agentic.workers import InProcessWorker, WorkerPool
from rune.server.workspace import WorkspaceManager


class _FakeModel:
    is_loaded = True

    def generate(self, prompt: str, **kw) -> str:
        if "Découpe" in prompt:
            return "1. Créer l'app\n2. Écrire les tests\n3. Ajouter le README"
        if "échelle de 0 à 1" in prompt:
            return "0.9"
        if "étape 1" in prompt:
            return "```python\n# file: src/app.py\nprint('hi')\n```"
        if "étape 2" in prompt:
            return "```python\n# file: tests/test_app.py\ndef test_x(): assert True\n```"
        if "synthèse" in prompt or "réponse finale" in prompt:
            return "Travail terminé : app.py et test_app.py écrits, tests au vert."
        return "```\nfile: README.md\n# Projet\n```"


def _pool():
    return WorkerPool(core=InProcessWorker(model=_FakeModel()))


# ── Worker pool routing ───────────────────────────────────────────────
def test_pool_needs_prefix_routes_to_core():
    assert _pool().pick(needs_prefix=True).name == "taelys-core"


def test_pool_prefers_available_auxiliary():
    class Aux:
        name = "ollama:x"
        needs_prefix = False
        def available(self):
            return True
        def generate(self, p, **k):
            return "aux"

    pool = WorkerPool(core=InProcessWorker(model=_FakeModel()), auxiliaries=[Aux()])
    assert pool.pick(needs_prefix=False).name == "ollama:x"


def test_pool_falls_back_to_core_when_aux_unavailable():
    class Aux:
        name = "ollama:x"
        needs_prefix = False
        def available(self):
            return False
        def generate(self, p, **k):
            return "aux"

    pool = WorkerPool(core=InProcessWorker(model=_FakeModel()), auxiliaries=[Aux()])
    assert pool.pick(needs_prefix=False).name == "taelys-core"


# ── Parsing helpers ───────────────────────────────────────────────────
def test_parse_plan_numbered_lines():
    steps = AgentOrchestrator._parse_plan("1. A\n2) B\nprose\n3. C")
    assert steps == ["A", "B", "C"]


def test_parse_plan_caps_at_max():
    text = "\n".join(f"{i}. step{i}" for i in range(1, 20))
    assert len(AgentOrchestrator._parse_plan(text)) == _MAX_STEPS


def test_confidence_scales_and_clamps():
    assert AgentOrchestrator._confidence("0.8") == 0.8
    assert AgentOrchestrator._confidence("90") == 0.9   # 0-100 scale
    assert AgentOrchestrator._confidence("nope") == 0.5  # default


# ── Full loop ─────────────────────────────────────────────────────────
def test_run_full_loop_writes_files(tmp_path: Path | None = None):
    async def go():
        with tempfile.TemporaryDirectory() as d:
            mgr = WorkspaceManager(
                sandbox_dir=Path(d),
                max_file_bytes=1_000_000,
                max_total_bytes=9_000_000,
            )
            ao = AgentOrchestrator(
                hippocampe=SimpleNamespace(),
                worker_pool=_pool(),
                workspace_manager=mgr,
            )
            types = []
            async for ev in ao.run("API", run_id="r1", subdir="projets/demo"):
                types.append(ev["type"])
            assert types[0] == "run_start" and types[-1] == "run_done"
            # The README step is dropped by the plan filter (meta step), so
            # only the two code steps run.
            assert types.count("step_done") == 2
            assert "synthesis" in types and types.index("synthesis") < types.index("run_done")
            base = Path(d) / "projets/demo"
            # Flat layout (orchestrator owns it): src/ collapsed to the root.
            assert (base / "app.py").exists()
            assert (base / "test_app.py").exists()
            assert not (base / "src/app.py").exists()
            # Spontaneous scaffolding (README) is dropped.
            assert not (base / "README.md").exists()
            # marker line stripped from the written file
            assert "file:" not in (base / "app.py").read_text()

    asyncio.run(go())


def test_seed_attachments_into_mission_dir():
    """User-provided files land in the mission dir; binary extensions are
    rewritten to .txt (the content is extracted text); junk is skipped."""
    with tempfile.TemporaryDirectory() as d:
        mgr = WorkspaceManager(
            sandbox_dir=Path(d),
            max_file_bytes=1_000_000,
            max_total_bytes=9_000_000,
        )
        ao = AgentOrchestrator(
            hippocampe=SimpleNamespace(),
            worker_pool=_pool(),
            workspace_manager=mgr,
        )
        names, note = ao._seed_attachments("missions/demo", [
            {"filename": "data.csv", "content": "a,b\n1,2\n"},
            {"filename": "rapport.pdf", "content": "texte extrait"},  # → .txt
            {"filename": "", "content": "x"},                          # skip
            {"filename": "vide.txt", "content": None},                 # skip
        ])
        assert names == ["data.csv", "rapport.txt"]
        assert "data.csv" in note and "rapport.txt" in note
        base = Path(d) / "missions/demo"
        assert (base / "data.csv").read_text() == "a,b\n1,2\n"
        assert (base / "rapport.txt").read_text() == "texte extrait"
        assert not (base / "rapport.pdf").exists()
        # Empty / None inputs are no-ops.
        assert ao._seed_attachments("missions/demo", []) == ([], "")
        assert ao._seed_attachments("missions/demo", None) == ([], "")
    async def go():
        ao = AgentOrchestrator(hippocampe=SimpleNamespace(), worker_pool=_pool())
        gen = ao.run("t", run_id="r2")
        await gen.__anext__()  # run_start
        ao.interject("r2", "ajoute l'auth")
        seen_interject = seen_stop = False
        async for ev in gen:
            if ev["type"] == "interjection_applied":
                seen_interject = True
            if ev["type"] == "step_done":
                ao.stop("r2")
            if ev["type"] == "run_stopped":
                seen_stop = True
        assert seen_interject and seen_stop

    asyncio.run(go())


def test_interject_unknown_run_returns_false():
    ao = AgentOrchestrator(hippocampe=SimpleNamespace(), worker_pool=_pool())
    assert ao.interject("nope", "x") is False
    assert ao.stop("nope") is False


def test_mission_name_slug_and_folder():
    class _M:
        is_loaded = True
        def generate(self, p, **k):
            if "titre très court" in p:
                return "Fibo Démo"
            if "Découpe" in p:
                return "1. Créer a.py"
            if "échelle de 0 à 1" in p:
                return "0.9"
            if "Réalise" in p:
                return "```python\n# file: a.py\nx = 1\n```"
            return "Synthèse OK."

    async def go():
        with tempfile.TemporaryDirectory() as d:
            mgr = WorkspaceManager(
                sandbox_dir=Path(d), max_file_bytes=1_000_000, max_total_bytes=9_000_000
            )
            ao = AgentOrchestrator(
                hippocampe=SimpleNamespace(),
                worker_pool=WorkerPool(core=InProcessWorker(model=_M())),
                workspace_manager=mgr,
            )
            start = None
            async for ev in ao.run("fais un fibo"):
                if ev["type"] == "run_start":
                    start = ev
            assert start["name"] == "Fibo Démo"
            assert start["slug"] == "fibo-demo"            # accents folded
            assert (Path(d) / "missions/fibo-demo" / "a.py").exists()

            # Second mission, same name → suffixed slug.
            start2 = None
            async for ev in ao.run("fais un fibo"):
                if ev["type"] == "run_start":
                    start2 = ev
            assert start2["slug"] == "fibo-demo-2"

    asyncio.run(go())


# ── Execution loop (V6.6) — hermetic, fake sandbox injected ───────────
class _FakeSandbox:
    max_installs = 5

    def __init__(self):
        self.installs = 0
        self._runs = 0

    def discover_tests(self):
        return True

    def run_pytest(self):
        from rune.agentic.sandbox import SandboxResult
        self._runs += 1
        if self._runs == 1:  # first run: a missing dependency
            return SandboxResult(False, 1, "ModuleNotFoundError: No module named 'foo'", "", 0.1)
        return SandboxResult(True, 0, "1 passed in 0.01s", "", 0.1)

    def pip_install(self, mod):
        from rune.agentic.sandbox import SandboxResult
        self.installs += 1
        return True, SandboxResult(True, 0, f"Successfully installed {mod}", "", 0.5)


def test_execution_loop_installs_then_passes():
    async def go():
        with tempfile.TemporaryDirectory() as d:
            mgr = WorkspaceManager(
                sandbox_dir=Path(d), max_file_bytes=1_000_000, max_total_bytes=9_000_000,
            )
            fake = _FakeSandbox()
            ao = AgentOrchestrator(
                hippocampe=SimpleNamespace(), worker_pool=_pool(),
                workspace_manager=mgr, execution_enabled=True,
                sandbox_factory=lambda _d: fake,
            )
            evs = [ev async for ev in ao.run("API", run_id="rx", subdir="projets/exec")]
            types = [e["type"] for e in evs]
            assert "exec_start" in types
            install = [e for e in evs if e["type"] == "exec_install"][0]
            assert install["package"] == "foo" and install["ok"] is True
            result = [e for e in evs if e["type"] == "exec_result"][0]
            assert result["ok"] is True
            assert fake.installs == 1
            # manifest persisted for resume
            assert (Path(d) / "projets/exec/.lythea-mission.json").exists()

    asyncio.run(go())


def test_execution_disabled_emits_no_exec_events():
    async def go():
        with tempfile.TemporaryDirectory() as d:
            mgr = WorkspaceManager(
                sandbox_dir=Path(d), max_file_bytes=1_000_000, max_total_bytes=9_000_000,
            )
            ao = AgentOrchestrator(
                hippocampe=SimpleNamespace(), worker_pool=_pool(),
                workspace_manager=mgr,  # execution_enabled defaults False
            )
            types = [ev["type"] async for ev in ao.run("API", run_id="rz", subdir="p/d")]
            assert not any(t.startswith("exec_") for t in types)

    asyncio.run(go())


# ── ReAct (model-driven tool-calling) — hermetic ─────────────────────
class _ReactFakeModel:
    is_loaded = True

    def generate(self, prompt: str, **kw) -> str:
        if "titre très court" in prompt:
            return "Calc TVA"
        if '"summary"' in prompt:            # run_tests already returned
            return "Le module calc.py et ses tests sont en place et passent tous."
        if "test_calc.py" in prompt:         # both files exist → verify
            return '<tool_call>{"name": "run_tests", "arguments": {}}</tool_call>'
        if "<tool_response>" in prompt:      # module written → write the test
            return ('<tool_call>{"name": "write_file", "arguments": '
                    '{"path": "test_calc.py", "content": '
                    '"from calc import tva\\ndef test_tva():\\n    assert tva(100)==20\\n"}}'
                    '</tool_call>')
        return ('<tool_call>{"name": "write_file", "arguments": '
                '{"path": "calc.py", "content": "def tva(x):\\n    return x*0.2\\n"}}'
                '</tool_call>')


class _PassSandbox:
    max_installs = 5

    def __init__(self):
        self.installs = 0

    def discover_tests(self):
        return True

    def run_pytest(self):
        from rune.agentic.sandbox import SandboxResult
        return SandboxResult(True, 0, "1 passed in 0.01s", "", 0.1)

    def pip_install(self, m):
        from rune.agentic.sandbox import SandboxResult
        self.installs += 1
        return True, SandboxResult(True, 0, "", "", 0.1)


def test_react_loop_writes_runs_and_synthesizes():
    async def go():
        with tempfile.TemporaryDirectory() as d:
            mgr = WorkspaceManager(
                sandbox_dir=Path(d), max_file_bytes=1_000_000, max_total_bytes=9_000_000,
            )
            ao = AgentOrchestrator(
                hippocampe=SimpleNamespace(),
                worker_pool=WorkerPool(core=InProcessWorker(model=_ReactFakeModel())),
                workspace_manager=mgr,
                execution_enabled=True,
                react_enabled=True,
                sandbox_factory=lambda _d: _PassSandbox(),
            )
            evs = [ev async for ev in ao.run(
                "crée un module de calcul de TVA avec ses tests", run_id="rr")]
            types = [e["type"] for e in evs]
            assert evs[0]["type"] == "run_start" and evs[0]["mode"] == "react"
            tool_names = [e["name"] for e in evs if e["type"] == "tool_call"]
            assert "write_file" in tool_names and "run_tests" in tool_names
            assert any(e["type"] == "exec_result" and e["ok"] for e in evs)
            assert "synthesis" in types
            assert types[-1] == "run_done"
            # the files the model wrote exist under the mission folder
            assert (Path(d) / "missions/calc-tva/calc.py").exists()
            assert (Path(d) / "missions/calc-tva/test_calc.py").exists()

    asyncio.run(go())


class _AnswerFakeModel:
    """answer mode: read the file once, then reply in prose."""
    is_loaded = True

    def __init__(self):
        self.n = 0

    def generate(self, prompt, **kw):
        if "titre très court" in prompt:
            return "Synthese fichier"
        self.n += 1
        if self.n == 1:
            return ('<tool_call>{"name": "read_file", "arguments": '
                    '{"path": "data.csv"}}</tool_call>')
        return ("Le fichier liste des produits financiers : colonnes titre, "
                "quantite et prix, sur plusieurs lignes.")


class _AnalyzeFakeModel:
    """analyze mode: read → write a script → run_command → prose result."""
    is_loaded = True

    def __init__(self):
        self.n = 0

    def generate(self, prompt, **kw):
        if "titre très court" in prompt:
            return "Analyse CSV"
        self.n += 1
        if self.n == 1:
            return ('<tool_call>{"name": "read_file", "arguments": '
                    '{"path": "data.csv"}}</tool_call>')
        if self.n == 2:
            return ('<tool_call>{"name": "write_file", "arguments": '
                    '{"path": "analyse.py", "content": "print(42.0)\\n"}}</tool_call>')
        if self.n == 3:
            return ('<tool_call>{"name": "run_command", "arguments": '
                    '{"argv": ["python", "analyse.py"]}}</tool_call>')
        return "La moyenne de la colonne prix vaut 42.0 d'après le calcul."


class _AnalyzeSandbox(_PassSandbox):
    def run_argv(self, argv, net=False, timeout=120):
        from rune.agentic.sandbox import SandboxResult
        return SandboxResult(True, 0, "42.0", "", 0.1)


def test_react_answer_mode_reads_then_responds():
    """A synthesis task reads the attached file, then answers in prose — no
    files written, no tests run."""
    async def go():
        with tempfile.TemporaryDirectory() as d:
            mgr = WorkspaceManager(
                sandbox_dir=Path(d), max_file_bytes=1_000_000, max_total_bytes=9_000_000,
            )
            ao = AgentOrchestrator(
                hippocampe=SimpleNamespace(),
                worker_pool=WorkerPool(core=InProcessWorker(model=_AnswerFakeModel())),
                workspace_manager=mgr,
                execution_enabled=True,
                react_enabled=True,
                sandbox_factory=lambda _d: _PassSandbox(),
            )
            evs = [ev async for ev in ao.run(
                "fais une synthèse de ce fichier", run_id="ra",
                attachments=[{"filename": "data.csv",
                              "content": "titre,qte,prix\nA,1,2\n"}])]
            types = [e["type"] for e in evs]
            tool_names = [e["name"] for e in evs if e["type"] == "tool_call"]
            assert "read_file" in tool_names
            assert "write_file" not in tool_names
            assert "run_tests" not in tool_names
            assert "synthesis" in types and types[-1] == "run_done"

    asyncio.run(go())


def test_react_analyze_mode_runs_script_no_pytest():
    """An analysis task reads, writes a script, runs it with run_command, and
    reports the result — never invoking run_tests."""
    async def go():
        with tempfile.TemporaryDirectory() as d:
            mgr = WorkspaceManager(
                sandbox_dir=Path(d), max_file_bytes=1_000_000, max_total_bytes=9_000_000,
            )
            ao = AgentOrchestrator(
                hippocampe=SimpleNamespace(),
                worker_pool=WorkerPool(core=InProcessWorker(model=_AnalyzeFakeModel())),
                workspace_manager=mgr,
                execution_enabled=True,
                react_enabled=True,
                sandbox_factory=lambda _d: _AnalyzeSandbox(),
            )
            evs = [ev async for ev in ao.run(
                "analyse data.csv et calcule la moyenne par colonne", run_id="rz",
                attachments=[{"filename": "data.csv",
                              "content": "prix\n40\n44\n"}])]
            types = [e["type"] for e in evs]
            tool_names = [e["name"] for e in evs if e["type"] == "tool_call"]
            assert "read_file" in tool_names
            assert "run_command" in tool_names
            assert "run_tests" not in tool_names
            assert "synthesis" in types and types[-1] == "run_done"

    asyncio.run(go())


class _FinishFakeModel:
    """build mode: write module → write test → run_tests (green) → finish."""
    is_loaded = True

    def __init__(self):
        self.n = 0

    def generate(self, prompt, **kw):
        if "titre très court" in prompt:
            return "Calc TVA"
        self.n += 1
        if self.n == 1:
            return ('<tool_call>{"name": "write_file", "arguments": '
                    '{"path": "calc.py", "content": "def tva(x):\\n    return x*0.2\\n"}}'
                    '</tool_call>')
        if self.n == 2:
            return ('<tool_call>{"name": "write_file", "arguments": '
                    '{"path": "test_calc.py", "content": '
                    '"from calc import tva\\ndef test_tva():\\n    assert tva(100)==20\\n"}}'
                    '</tool_call>')
        if self.n == 3:
            return '<tool_call>{"name": "run_tests", "arguments": {}}</tool_call>'
        return ('<tool_call>{"name": "finish", "arguments": '
                '{"answer": "Le module calc.py et ses tests passent ; TVA à 20%."}}'
                '</tool_call>')


def test_react_finish_tool_completes_after_green():
    """The finish tool ends the run once tests are green, surfacing its
    answer as the synthesis."""
    async def go():
        with tempfile.TemporaryDirectory() as d:
            mgr = WorkspaceManager(
                sandbox_dir=Path(d), max_file_bytes=1_000_000, max_total_bytes=9_000_000,
            )
            ao = AgentOrchestrator(
                hippocampe=SimpleNamespace(),
                worker_pool=WorkerPool(core=InProcessWorker(model=_FinishFakeModel())),
                workspace_manager=mgr,
                execution_enabled=True,
                react_enabled=True,
                sandbox_factory=lambda _d: _PassSandbox(),
            )
            evs = [ev async for ev in ao.run(
                "crée un module de calcul de TVA avec ses tests", run_id="rf")]
            types = [e["type"] for e in evs]
            tool_names = [e["name"] for e in evs if e["type"] == "tool_call"]
            assert "finish" in tool_names
            synth = [e for e in evs if e["type"] == "synthesis"]
            assert synth and "20%" in synth[0]["text"]
            assert types[-1] == "run_done"

    asyncio.run(go())


class _CriticFakeModel:
    """build mode: write → test → run (green) → finish; the critic rejects the
    first finish once, the model addresses it, then finishes for good."""
    is_loaded = True

    def __init__(self):
        self.n = 0

    def generate(self, prompt, **kw):
        if "titre très court" in prompt:
            return "Calc"
        if "relecteur" in prompt.lower():        # critic pass — don't touch n
            return "CORRIGER : ajoute un test pour l'entrée 0."
        if "leçon réutilisable" in prompt.lower():  # reflexion distill — keep n
            return ("TRIGGER: créer un module avec tests\n"
                    "APPROACH: couvrir l'entrée 0 et les cas limites")
        self.n += 1
        if self.n == 1:
            return ('<tool_call>{"name": "write_file", "arguments": '
                    '{"path": "calc.py", "content": "def f(x):\\n    return x*2\\n"}}'
                    '</tool_call>')
        if self.n == 2:
            return ('<tool_call>{"name": "write_file", "arguments": '
                    '{"path": "test_calc.py", "content": '
                    '"from calc import f\\ndef test_f():\\n    assert f(2)==4\\n"}}'
                    '</tool_call>')
        if self.n in (3, 6):
            return '<tool_call>{"name": "run_tests", "arguments": {}}</tool_call>'
        if self.n == 5:
            return ('<tool_call>{"name": "write_file", "arguments": '
                    '{"path": "test_calc.py", "content": '
                    '"from calc import f\\ndef test_f():\\n    assert f(2)==4\\n'
                    'def test_zero():\\n    assert f(0)==0\\n"}}'
                    '</tool_call>')
        return ('<tool_call>{"name": "finish", "arguments": '
                '{"answer": "Module f et tests (dont l\'entrée 0) au vert."}}'
                '</tool_call>')


def test_react_critic_rejects_then_accepts():
    """The critic bounces the first finish with concrete feedback; after the
    model addresses it, the run completes."""
    async def go():
        with tempfile.TemporaryDirectory() as d:
            mgr = WorkspaceManager(
                sandbox_dir=Path(d), max_file_bytes=1_000_000, max_total_bytes=9_000_000,
            )
            ao = AgentOrchestrator(
                hippocampe=SimpleNamespace(),
                worker_pool=WorkerPool(core=InProcessWorker(model=_CriticFakeModel())),
                workspace_manager=mgr,
                execution_enabled=True,
                react_enabled=True,
                sandbox_factory=lambda _d: _PassSandbox(),
            )
            evs = [ev async for ev in ao.run(
                "crée un module f avec ses tests", run_id="rc")]
            types = [e["type"] for e in evs]
            assert "critique" in types          # critic fired and rejected once
            crit = [e for e in evs if e["type"] == "critique"]
            assert "0" in crit[0]["text"]        # actionable feedback surfaced
            assert "finish" in [e["name"] for e in evs if e["type"] == "tool_call"]
            assert "synthesis" in types and types[-1] == "run_done"

    asyncio.run(go())


def test_react_learns_lesson_from_errors():
    """A run that hits a failure (critic rejection) distills and persists a
    lesson into the shared procedural store, and emits a lesson_learned event."""
    class _FakeProcStore:
        def __init__(self):
            self.added = []

        def active(self):
            return []

        def add(self, trigger, approach, **kw):
            self.added.append((trigger, approach, kw.get("source_episodes")))
            return object()

        def save(self):
            pass

    async def go():
        with tempfile.TemporaryDirectory() as d:
            mgr = WorkspaceManager(
                sandbox_dir=Path(d), max_file_bytes=1_000_000, max_total_bytes=9_000_000,
            )
            store = _FakeProcStore()
            ao = AgentOrchestrator(
                hippocampe=SimpleNamespace(procedural_store=store),
                worker_pool=WorkerPool(core=InProcessWorker(model=_CriticFakeModel())),
                workspace_manager=mgr,
                execution_enabled=True,
                react_enabled=True,
                sandbox_factory=lambda _d: _PassSandbox(),
            )
            evs = [ev async for ev in ao.run(
                "crée un module f avec ses tests", run_id="rl")]
            types = [e["type"] for e in evs]
            assert "lesson_learned" in types
            assert store.added                      # persisted to shared memory
            trig, appr, prov = store.added[0]
            assert trig and appr
            assert prov and prov[0].startswith("agent:")   # provenance tagged

    asyncio.run(go())


def test_missing_modules_under_test_detects_local_target():
    """test_<stem>.py importing an absent <stem> flags <stem> as missing."""
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        (base / "test_email_validator.py").write_text(
            "from email_validator import validate_email\n"
            "def test_ok():\n    assert validate_email('a@b.co')\n",
            encoding="utf-8",
        )
        # Only the test + an unrelated requirements file exist.
        (base / "requirements.txt").write_text("email-validator==2.0.0\n", encoding="utf-8")
        missing = AgentOrchestrator._missing_modules_under_test(base)
        assert missing == ["email_validator"]


def test_missing_modules_under_test_satisfied_when_module_present():
    """When <stem>.py exists, nothing is reported missing (no homonym install)."""
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        (base / "test_email_validator.py").write_text(
            "from email_validator import validate_email\n", encoding="utf-8")
        (base / "email_validator.py").write_text(
            "def validate_email(a):\n    return '@' in a\n", encoding="utf-8")
        assert AgentOrchestrator._missing_modules_under_test(base) == []


# ── Plan filtering + naming robustness ────────────────────────────────
def test_plan_drop_removes_meta_steps():
    from rune.agentic.orchestrator import _PLAN_DROP
    keep = "Créer un fichier email_validator.py avec validate_email"
    drops = [
        "Ajoutez une documentation docstring à la fonction",
        "Écrire le README du projet",
        "Définir les requirements du module",
        "Concevoir l'interface utilisateur",
    ]
    assert not _PLAN_DROP.search(keep)
    for d in drops:
        assert _PLAN_DROP.search(d), d


def test_gen_name_rejects_placeholder_echo():
    """Model echoes 'Nom de la mission' → fall back to a task-derived name."""
    class _NameEcho:
        is_loaded = True
        def generate(self, prompt, **kw):
            return "Nom de la mission"
    ao = AgentOrchestrator(hippocampe=SimpleNamespace(),
                           worker_pool=WorkerPool(core=InProcessWorker(model=_NameEcho())))
    async def go():
        return await ao._gen_name(
            ao.pool.core,
            "crée un module de validation d'email avec ses tests",
        )
    name = asyncio.run(go())
    low = name.lower()
    assert "validation" in low or "email" in low
    assert low not in ("nom de la mission", "mission", "titre")


def test_gen_name_prefers_domain_over_framework():
    """Model answers 'Unittest' → derive a domain name, not the framework."""
    class _FwEcho:
        is_loaded = True
        def generate(self, prompt, **kw):
            return "Unittest"
    ao = AgentOrchestrator(hippocampe=SimpleNamespace(),
                           worker_pool=WorkerPool(core=InProcessWorker(model=_FwEcho())))
    async def go():
        return await ao._gen_name(
            ao.pool.core,
            "crée un module Python de validation d'email avec ses tests",
        )
    name = asyncio.run(go()).lower()
    assert "unittest" not in name and "test" not in name
    assert "validation" in name or "email" in name


def test_clean_synthesis_strips_solicitations_and_empty_fences():
    from rune.agentic.orchestrator import _clean_synthesis
    raw = (
        "Le module a été créé.\n\n```undefined\n\n```\n"
        "Votre avis est apprécié. Est-ce correct ? "
        "N'hésitez pas à demander."
    )
    out = _clean_synthesis(raw)
    assert "Le module a été créé." in out
    assert "```" not in out
    low = out.lower()
    assert "votre avis" not in low
    assert "est-ce correct" not in low
    assert "n'hésitez" not in low and "n’hésitez" not in low


def test_public_symbols_extracts_class_and_def():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        (base / "parser_ini.py").write_text(
            "import re\n\nclass IniParser:\n    def read(self):\n        pass\n"
            "\ndef helper():\n    pass\n\ndef _private():\n    pass\n",
            encoding="utf-8",
        )
        syms = AgentOrchestrator._public_symbols(base, "parser_ini.py")
        assert "IniParser" in syms and "helper" in syms
        assert "_private" not in syms
        # method 'read' is indented, not top-level → excluded
        assert "read" not in syms


def test_modules_needing_impl_flags_missing_and_stub():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        (base / "test_email_validator.py").write_text(
            "from email_validator import validate_email\n"
            "def test_ok():\n    assert validate_email('a@b.co')\n",
            encoding="utf-8",
        )
        # Case 1: module entirely absent.
        need = dict(AgentOrchestrator._modules_needing_impl(base))
        assert need.get("email_validator") == ["validate_email"]
        # Case 2: module present but a stub without the imported symbol.
        (base / "email_validator.py").write_text(
            '"""stub"""\nVERSION = "1"\n', encoding="utf-8")
        need = dict(AgentOrchestrator._modules_needing_impl(base))
        assert need.get("email_validator") == ["validate_email"]
        # Case 3: module implements the symbol → nothing needed.
        (base / "email_validator.py").write_text(
            "def validate_email(e):\n    return '@' in e\n", encoding="utf-8")
        assert AgentOrchestrator._modules_needing_impl(base) == []


def test_modules_needing_impl_ignores_third_party():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        # A test importing a third-party lib with no local sibling → ignored.
        (base / "test_app.py").write_text(
            "from app import main\nimport requests\n", encoding="utf-8")
        need = dict(AgentOrchestrator._modules_needing_impl(base))
        assert "requests" not in need
        assert need.get("app") == ["main"]


def test_flatten_imports_rewrites_collapsed_prefix():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        (base / "email_validator.py").write_text(
            "def validate_email(e):\n    return '@' in e\n", encoding="utf-8")
        (base / "test_email_validator.py").write_text(
            "from utils.email_validator import validate_email\n"
            "import utils.email_validator\n"
            "def test_ok():\n    assert validate_email('a@b.co')\n",
            encoding="utf-8",
        )
        ao = AgentOrchestrator(hippocampe=SimpleNamespace(), worker_pool=_pool())
        ao._flatten_imports(base)
        txt = (base / "test_email_validator.py").read_text()
        assert "from email_validator import validate_email" in txt
        assert "import email_validator" in txt
        assert "utils." not in txt


def test_flatten_imports_keeps_real_package_dir():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        (base / "pkg").mkdir()
        (base / "pkg" / "mod.py").write_text("X = 1\n", encoding="utf-8")
        (base / "test_x.py").write_text("from pkg.mod import X\n", encoding="utf-8")
        ao = AgentOrchestrator(hippocampe=SimpleNamespace(), worker_pool=_pool())
        ao._flatten_imports(base)
        # pkg/ is a real dir → import left intact
        assert "from pkg.mod import X" in (base / "test_x.py").read_text()


def test_is_local_pkg_prefix():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        (base / "t.py").write_text("from utils.x import y\n", encoding="utf-8")
        assert AgentOrchestrator._is_local_pkg_prefix(base, "utils") is True
        assert AgentOrchestrator._is_local_pkg_prefix(base, "requests") is False
        (base / "utils").mkdir()
        # real dir → not a flatten artifact
        assert AgentOrchestrator._is_local_pkg_prefix(base, "utils") is False


def test_pytest_failure_digest_extracts_assertion():
    from rune.agentic.orchestrator import _pytest_failure_digest
    sample = (
        "test_x.py:14: in test_rejects\n"
        "    assert validate('bad') is False\n"
        "E   AssertionError: assert True is False\n"
        "=== short test summary info ===\n"
        "FAILED test_x.py::test_rejects - AssertionError: assert True is False\n"
        "1 failed, 1 passed in 0.02s"
    )
    out = _pytest_failure_digest(sample)
    assert "FAILED test_x.py::test_rejects" in out
    assert "AssertionError" in out
    # passing-only output yields no digest
    assert _pytest_failure_digest("2 passed in 0.01s") == ""


def test_recall_lessons_keyword_fallback_without_embedder():
    """With no retriever/embedder, recall degrades to keyword overlap on the
    trigger and only surfaces lessons that share words with the task."""
    class _Store:
        def active(self):
            return [
                SimpleNamespace(
                    trigger="extraire tableau pdf nettoyer colonnes",
                    approach="lire pdf, repérer grille, caster colonnes, vérifier totaux",
                    utility_score=1.0),
                SimpleNamespace(
                    trigger="déployer service fastapi docker",
                    approach="dockerfile, uvicorn, healthcheck",
                    utility_score=1.0),
            ]
    ao = AgentOrchestrator(
        hippocampe=SimpleNamespace(procedural_store=_Store()),
        worker_pool=_pool(),
    )
    assert ao._embedder() is None                  # no retriever → lexical path
    hit = ao._recall_lessons("comment extraire un tableau depuis ce document tableau")
    assert "grille" in hit                         # the extraction lesson surfaced
    assert "uvicorn" not in hit                    # the unrelated one did not
    assert ao._recall_lessons("xyzzy quux foobar") == ""   # no shared words → nothing


def test_trigger_similarity_zero_without_embedder():
    ao = AgentOrchestrator(hippocampe=SimpleNamespace(), worker_pool=_pool())
    assert ao._trigger_similarity("a", "b") == 0.0  # no embedder → exact-match dedup


def test_circular_modules_detects_self_import():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        Path(d, "email_validator.py").write_text(
            "from email_validator import EmailNotValidError\n"
            "def validate_email(x): return '@' in x\n", encoding="utf-8")
        Path(d, "test_email_validator.py").write_text(
            "from email_validator import validate_email\n"
            "def test_ok(): assert validate_email('a@b.c')\n", encoding="utf-8")
        Path(d, "clean.py").write_text("def f(): return 1\n", encoding="utf-8")
        circ = AgentOrchestrator._circular_modules(d)
        stems = [s for s, _ in circ]
        assert stems == ["email_validator"]          # self-import flagged
        assert "clean" not in stems                  # clean module not flagged
        # source excerpt is returned for force-feeding
        assert "from email_validator import" in circ[0][1]


def test_circular_modules_empty_dir():
    assert AgentOrchestrator._circular_modules("/no/such/dir") == []


def test_ssrf_guard_blocks_internal_hosts():
    import pytest
    from rune.agentic.orchestrator import _assert_public_url, _host_is_public
    # internal / private / link-local / metadata → blocked
    for ip in ("127.0.0.1", "10.0.0.5", "192.168.1.1", "172.16.0.1",
               "169.254.169.254", "0.0.0.0"):
        assert _host_is_public(ip) is False, ip
        with pytest.raises(ValueError):
            _assert_public_url(f"http://{ip}/")
    # public IP literal → allowed
    assert _host_is_public("8.8.8.8") is True
    # non-http scheme → blocked
    with pytest.raises(ValueError):
        _assert_public_url("file:///etc/passwd")
    with pytest.raises(ValueError):
        _assert_public_url("gopher://8.8.8.8/")


def test_extract_py_block_and_filename():
    from rune.agentic.orchestrator import _extract_py_block, _filename_from_code
    txt = ("Voici le code :\n```python\n"
           "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n"
           "        a, b = b, a + b\n    return a\n```\nVoilà.")
    code = _extract_py_block(txt)
    assert "def fibonacci" in code and "for _ in range" in code
    assert _filename_from_code(code) == "fibonacci.py"
    # no fence → empty; no def → solution.py
    assert _extract_py_block("pas de code ici") == ""
    assert _filename_from_code("x = 1") == "solution.py"


def test_extract_py_block_unlabelled_fence():
    from rune.agentic.orchestrator import _extract_py_block
    code = _extract_py_block("```\nprint('hi')\n```")
    assert code == "print('hi')"


def test_jail_relpath_blocks_cross_mission_and_escapes():
    from rune.agentic.orchestrator import _jail_relpath
    sd = "missions/validation-email"
    # legitimate paths inside the mission are preserved
    assert _jail_relpath("email_validator.py", sd) == "email_validator.py"
    assert _jail_relpath("src/utils.py", sd) == "src/utils.py"
    # echoed mission prefix is stripped
    assert _jail_relpath(f"{sd}/email_validator.py", sd) == "email_validator.py"
    # internal .. that stays inside is collapsed, not rejected
    assert _jail_relpath("a/../b.py", sd) == "b.py"
    # cross-mission / escapes → '' (rejected)
    assert _jail_relpath("../fibo-mission/secret.py", sd) == ""
    assert _jail_relpath(f"{sd}/../fibo-mission/secret.py", sd) == ""
    assert _jail_relpath("../../etc/passwd", sd) == ""
    assert _jail_relpath("..", sd) == ""
    # absolute is relativised into the mission (stays inside), never escapes
    assert not _jail_relpath("/etc/passwd", sd).startswith("/")


def test_py_syntax_error_gate():
    from rune.agentic.orchestrator import _py_syntax_error
    assert _py_syntax_error("def f():\n    return 1\n") == ""
    err = _py_syntax_error("def f():\nreturn 1\n")     # bad indent
    assert "ligne" in err
    assert _py_syntax_error("x = (") != ""             # unclosed


def test_unified_diff_shows_noop():
    from rune.agentic.orchestrator import _unified_diff
    d = _unified_diff("return x is not None\n", "return bool(x)\n", "m.py")
    assert "-return x is not None" in d and "+return bool(x)" in d
    assert _unified_diff("same\n", "same\n", "m.py") == ""   # no-op → empty


def test_parse_test_counts():
    from rune.agentic.orchestrator import _parse_test_counts
    assert _parse_test_counts("2 failed, 3 passed in 0.1s") == (2, 3)
    assert _parse_test_counts("5 passed in 0.01s") == (0, 5)
    assert _parse_test_counts("1 error during collection") == (1, 0)
    assert _parse_test_counts("") == (None, None)
    assert _parse_test_counts("rien à voir") == (None, None)


def test_gen_batch_uses_batch_then_falls_back():
    import asyncio
    from rune.agentic.orchestrator import AgentOrchestrator

    orch = AgentOrchestrator.__new__(AgentOrchestrator)   # no __init__ deps

    class _Batched:
        def generate_batch(self, prompts, **kw):
            return [f"B:{p}" for p in prompts]

        def generate(self, p, **kw):
            raise AssertionError("ne doit pas être appelé")

    class _Legacy:
        def generate(self, p, **kw):
            return f"S:{p}"

    class _Broken:
        def generate_batch(self, prompts, **kw):
            raise RuntimeError("OOM simulé")

        def generate(self, p, **kw):
            return f"F:{p}"

    out = asyncio.run(orch._gen_batch(_Batched(), ["a", "b"]))
    assert out == ["B:a", "B:b"]                      # chemin batch
    out = asyncio.run(orch._gen_batch(_Legacy(), ["a", "b"]))
    assert out == ["S:a", "S:b"]                      # pas de batch → séquentiel
    out = asyncio.run(orch._gen_batch(_Broken(), ["x"]))
    assert out == ["F:x"]                             # batch casse → fallback


def test_thinking_mode_resolution():
    from rune.agentic.orchestrator import AgentOrchestrator

    class _S:
        agent_thinking_profile = "auto"

    orch = AgentOrchestrator.__new__(AgentOrchestrator)
    orch.settings = _S()

    class _ModelThinking:
        is_thinking = True

    class _ModelPlain:
        is_thinking = False

    class _W:
        def __init__(self, m): self.model = m

    # auto → suit le modèle
    assert orch._thinking_mode(_W(_ModelThinking())) is True
    assert orch._thinking_mode(_W(_ModelPlain())) is False
    # on/off forcent quel que soit le modèle
    orch.settings.agent_thinking_profile = "on"
    assert orch._thinking_mode(_W(_ModelPlain())) is True
    orch.settings.agent_thinking_profile = "off"
    assert orch._thinking_mode(_W(_ModelThinking())) is False


def test_strip_think_handles_all_cases():
    from rune.agentic.orchestrator import _strip_think
    # bloc fermé normal
    assert _strip_think("<think>raisonnement</think>action") == "action"
    # NON fermé (coupé par budget) → tout le raisonnement disparaît
    assert _strip_think("avant<think>je réfléchis sans fin") == "avant"
    # </think> orpheline → garder l'après
    assert _strip_think("bla bla</think>la vraie réponse") == "la vraie réponse"
    # multiples blocs
    assert _strip_think("<think>a</think>X<think>b</think>Y") == "XY"
    # rien
    assert _strip_think("") == ""
    assert _strip_think("juste du texte") == "juste du texte"
    # casse-insensible
    assert _strip_think("<THINK>x</THINK>ok") == "ok"


def test_capture_think_includes_unclosed_tail():
    from rune.agentic.orchestrator import _capture_think
    assert _capture_think("<think>fermé</think>z") == "fermé"
    # bloc non fermé → on capture quand même le raisonnement en cours
    cap = _capture_think("<think>en cours de réflexion coupée")
    assert "en cours de réflexion" in cap
    # combinaison fermé + tail ouvert
    cap2 = _capture_think("<think>un</think>milieu<think>deux non fermé")
    assert "un" in cap2 and "deux non fermé" in cap2


def test_tool_call_survives_unclosed_think():
    # un tool_call précédé d'un think NON fermé doit rester parsable après strip
    from rune.agentic.orchestrator import _strip_think
    raw = ('<think>je dois écrire le fichier mais je continue de réfléchir '
           'très longtemps</think>'
           '<tool_call>{"name": "write_file", "arguments": {}}</tool_call>')
    out = _strip_think(raw)
    assert "<tool_call>" in out and "write_file" in out


def test_runaway_guards_constants_present():
    """Audit anti-emballement : les bornes clés existent dans l'orchestrateur."""
    src = open("lythea/agentic/orchestrator.py", encoding="utf-8").read()
    # caps d'appels d'outils
    assert "_WEB_CALL_MAX" in src           # web spam
    assert "_REPEAT_CALL_MAX" in src        # même appel répété (tous outils)
    assert "_SANDBOX_CALL_MAX" in src       # run_command/python/serve cumulés
    assert "progress.should_stop" in src    # arrêt anti-boucle (généraliste)
    # la signature d'appel est bien calculée pour le repeat-guard
    assert "_last_call_sig" in src
    # bornes de boucles internes
    assert "max_installs" in src            # install deps bornée
    assert "fix_tries < 2" in src           # fix collection borné
    assert "verify_retries < 1" in src      # re-vérif livrable bornée


def test_focus_failing_case_isolates_one_test():
    from rune.agentic.orchestrator import _focus_failing_case
    out = """test_validate_email.py::test_subdomain FAILED
FAILED test_validate_email.py::test_subdomain - AssertionError: assert False == True
E   assert False == True
E    +  where False = validate('a@b.co.uk')"""
    f = _focus_failing_case(out)
    assert f["test"] == "test_subdomain"
    assert "assert False == True" in f["assertion"]
    assert "validate('a@b.co.uk')" in f["raw"]


def test_focus_failing_case_empty_on_green():
    from rune.agentic.orchestrator import _focus_failing_case
    assert _focus_failing_case("") == {}
    assert _focus_failing_case("3 passed in 0.02s") == {}


def test_decompose_escalation_wired():
    src = open("lythea/agentic/orchestrator.py", encoding="utf-8").read()
    assert "did_decompose" in src
    assert "progress.stalled >= progress.DECOMPOSE" in src  # palier généraliste
    assert "_focus_failing_case" in src
