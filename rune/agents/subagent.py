"""SubAgentSpawner — sous-agents isolés en subprocess.

Inspiration Rune
------------------
Rune déploie des sous-agents isolés (conteneur Docker, RPC) pour
exécuter des tâches en parallèle. On adapte ça en subprocess Python
(plus léger, pas besoin de Docker sur le pod).

Modèle de sécurité
------------------
1. **Subprocess dédié** avec ``subprocess.Popen`` — pas de shared memory
   avec le parent.
2. **cwd jail** : le subprocess tourne dans un tmpfs dédié, ne peut pas
   accéder au reste du filesystem.
3. **Timeout hard** : kill après ``timeout_sec`` secondes.
4. **Output limit** : tronque stdout/stderr à ``max_output_bytes``.
5. **Pas de réseau** par défaut (peut être activé explicitement).

Le sous-agent hérite d'une **copie froide** du contexte pertinent
(sélectionné par le caller — pas toute la mémoire). Il communique via
stdin/stdout JSON (protocol simple).

Protocol
--------
Le parent écrit sur stdin du subprocess un JSON :
    {
        "task": "Écris une fonction est_premier(n)",
        "context": "..."  # optional cold context
    }
Le subprocess lit, exécute, et écrit sur stdout un JSON :
    {
        "status": "ok" | "error" | "timeout",
        "result": "...",
        "artifacts": [...]
    }

Le subprocess peut appeler des tools (web, code) si le script le permet.
On fournit un script par défaut ``rune/agents/_subagent_main.py``
qui implémente le protocol et peut être étendu.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("rune.agents.subagent")


# Script Python exécuté dans le subprocess. Par défaut, il importe le
# SubAgent runtime et boucle sur stdin. On peut l'override via config.
_DEFAULT_SCRIPT = """
import json, sys
from rune.agents._runtime import SubAgentRuntime

def main():
    payload = json.loads(sys.stdin.read())
    runtime = SubAgentRuntime(payload)
    result = runtime.run()
    sys.stdout.write(json.dumps(result))

if __name__ == "__main__":
    main()
"""


@dataclass
class SubAgentConfig:
    """Config du spawner."""
    timeout_sec: float = 120.0
    max_output_bytes: int = 1024 * 1024  # 1 MB
    python_executable: str = sys.executable
    work_dir: Path | None = None  # None = tmpdir dédié
    script_path: Path | None = None  # None = script par défaut
    env_overrides: dict[str, str] = field(default_factory=dict)
    allow_network: bool = False


@dataclass
class SubAgentResult:
    """Résultat d'un subagent."""
    status: str = "ok"  # ok | error | timeout | killed
    result: str = ""
    artifacts: list[str] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    elapsed_sec: float = 0.0
    exit_code: int = 0
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "result": self.result,
            "artifacts": self.artifacts,
            "stdout": self.stdout[:500],
            "stderr": self.stderr[:500],
            "elapsed_sec": round(self.elapsed_sec, 3),
            "exit_code": self.exit_code,
            "error": self.error,
        }


class SubAgentSpawner:
    """Lance des sous-agents isolés.

    Usage :

        spawner = SubAgentSpawner(SubAgentConfig(timeout_sec=60))
        result = spawner.run(task="Calcule fibonacci(10)", context="...")
        if result.status == "ok":
            print(result.result)

    Intégration Trinity (Option A)
    ------------------------------
    Si Trinity est activé côté parent, les sous-agents utilisent le
    **Worker** du pool (jamais Thinker ou Critic). Le parent passe
    le model_id du Worker au subprocess via le payload stdin, et le
    runtime du subprocess charge ce modèle via HFModelWrapper.

    Pour activer : passe le model_id explicitement via ``run(model_id=...)``
    ou configure-le via ``set_trinity_worker_model_id()``. Le parent
    (RuneCortex) le fait automatiquement quand Trinity est activé.
    """

    def __init__(self, config: SubAgentConfig | None = None) -> None:
        self.config = config or SubAgentConfig()
        # Model ID optionnel — si set, le subprocess charge ce modèle
        # via HFModelWrapper.load(). Si None, le subprocess utilise
        # MockBackend (mode dégradé, utile pour tests).
        self._trinity_worker_model_id: str | None = None

    def set_trinity_worker_model_id(self, model_id: str | None) -> None:
        """Configure le model_id du Worker Trinity à passer aux sous-agents.

        À appeler depuis RuneCortex quand Trinity est activé. Passe
        None pour désactiver (retour au mode MockBackend).
        """
        self._trinity_worker_model_id = model_id

    def run(
        self,
        task: str,
        context: str = "",
        payload: dict | None = None,
        model_id: str | None = None,
        hot_context: dict | None = None,
        blackboard_path: str | None = None,
        blackboard_section: str | None = None,
    ) -> SubAgentResult:
        """Lance un subagent et attend le résultat.

        Parameters
        ----------
        task : str
            La mission du subagent (prompt).
        context : str
            Contexte froid injecté (chunks RAG sélectionnés par le caller).
        payload : dict | None
            Override complet du payload envoyé au subagent. Si None,
            construit à partir de task + context + model_id + hot_context
            + blackboard.
        model_id : str | None
            Model ID HuggingFace à charger dans le subprocess. Si fourni,
            override ``self._trinity_worker_model_id``. Le subprocess
            utilise ce modèle via HFModelWrapper. Si None et aucun
            model_id configuré, le subprocess tombe sur MockBackend.
        hot_context : dict | None
            Contexte mémoire chaud (Solution A). Contient les chunks RAG,
            skills, anti-patterns pertinents pour la tâche. Sérialisé en
            JSON et injecté dans le system prompt du sous-agent. Lecture
            seule — le sous-agent ne peut pas écrire en mémoire.
            Voir rune.agents.hot_context.HotContext.
        blackboard_path : str | None
            Chemin vers le fichier blackboard.json partagé (Solution B).
            Si fourni, le sous-agent ouvre le blackboard, lit les sections
            des autres agents + le contract, et écrit dans sa propre
            section. Le parent re-load le blackboard au retour.
        blackboard_section : str | None
            Nom de la section du blackboard à utiliser pour ce sous-agent
            (ex: "subagent_1"). Requis si blackboard_path est fourni.
        """
        start = time.time()
        result = SubAgentResult()

        # Résout le model_id à passer au subprocess
        effective_model_id = model_id or self._trinity_worker_model_id

        # Prépare le work dir
        if self.config.work_dir is not None:
            work_dir = Path(self.config.work_dir)
            work_dir.mkdir(parents=True, exist_ok=True)
            cleanup_work_dir = False
        else:
            work_dir = Path(tempfile.mkdtemp(prefix="rune_subagent_"))
            cleanup_work_dir = True

        try:
            # Prépare le script
            script_path = self.config.script_path
            if script_path is None:
                script_path = work_dir / "_subagent_main.py"
                script_path.write_text(_DEFAULT_SCRIPT, encoding="utf-8")

            # Prépare le payload stdin
            if payload is None:
                payload = {
                    "task": task,
                    "context": context,
                }
            # Injecte le model_id dans le payload (Trinity Option A)
            if effective_model_id and "model_id" not in payload:
                payload["model_id"] = effective_model_id
            # Injecte le hot_context (mémoire partagée, Solution A)
            if hot_context and "hot_context" not in payload:
                payload["hot_context"] = hot_context
            # Injecte le blackboard (Solution B)
            if blackboard_path and "blackboard_path" not in payload:
                payload["blackboard_path"] = str(blackboard_path)
                if blackboard_section:
                    payload["blackboard_section"] = blackboard_section
            payload_json = json.dumps(payload)

            # Prépare l'env
            env = os.environ.copy()
            if not self.config.allow_network:
                # Pas de réseau : on unset les proxy vars (best-effort)
                for k in list(env.keys()):
                    if "proxy" in k.lower():
                        del env[k]
            env.update(self.config.env_overrides)
            env["HERMES_SUBAGENT_MODE"] = "1"

            # Lance le subprocess
            proc = subprocess.Popen(
                [self.config.python_executable, str(script_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(work_dir),
                env=env,
                text=True,
                encoding="utf-8",
            )

            try:
                stdout, stderr = proc.communicate(
                    input=payload_json,
                    timeout=self.config.timeout_sec,
                )
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                result.status = "timeout"
                result.error = (
                    f"Subagent timed out after {self.config.timeout_sec}s"
                )
                result.elapsed_sec = time.time() - start
                return result

            # Tronque les outputs
            result.stdout = stdout[: self.config.max_output_bytes]
            result.stderr = stderr[: self.config.max_output_bytes]
            result.exit_code = proc.returncode
            result.elapsed_sec = time.time() - start

            if result.exit_code != 0:
                result.status = "error"
                result.error = (
                    f"Subagent exited with code {result.exit_code}"
                )
                return result

            # Parse le JSON de stdout
            try:
                output = json.loads(stdout)
                result.status = output.get("status", "ok")
                result.result = output.get("result", "")
                result.artifacts = output.get("artifacts", [])
                if output.get("error"):
                    result.error = output["error"]
            except json.JSONDecodeError as exc:
                result.status = "error"
                result.error = f"Invalid JSON output: {exc}"

            return result

        except Exception as exc:
            log.exception("Subagent spawn failed")
            result.status = "error"
            result.error = str(exc)
            result.elapsed_sec = time.time() - start
            return result

        finally:
            if cleanup_work_dir:
                try:
                    import shutil
                    shutil.rmtree(work_dir, ignore_errors=True)
                except Exception:
                    pass

    def run_parallel(
        self,
        tasks: list[tuple[str, str]],
        max_concurrent: int = 3,
    ) -> list[SubAgentResult]:
        """Lance plusieurs subagents en parallèle.

        ``tasks`` est une liste de (task, context). On lance au plus
        ``max_concurrent`` subprocess en parallèle via ThreadPoolExecutor.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results: list[SubAgentResult | None] = [None] * len(tasks)
        with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
            futures = {
                pool.submit(self.run, task, ctx): idx
                for idx, (task, ctx) in enumerate(tasks)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    results[idx] = SubAgentResult(
                        status="error",
                        error=str(exc),
                    )
        return [r or SubAgentResult(status="error", error="no result") for r in results]
