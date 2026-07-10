"""Bootstrap environment variables before any HuggingFace import.

This module MUST be imported before `transformers`, `torch`, or any HF
library. It redirects all caches to a platform-aware root directory to
avoid filling up the system disk on RunPod / Colab.
"""
from __future__ import annotations

import os
from pathlib import Path

PLATFORMS = ("runpod", "colab", "kaggle", "local")


def detect_platform() -> str:
    """Detect the execution platform from filesystem markers."""
    if Path("/workspace").is_dir():
        return "runpod"
    if Path("/content").is_dir() and "COLAB_GPU" in os.environ:
        return "colab"
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        return "kaggle"
    return "local"


def _load_dotenv_into_environ() -> None:
    """Charge le fichier .env (racine du projet) dans ``os.environ``.

    POURQUOI c'est nécessaire
    -------------------------
    pydantic-settings lit .env pour peupler les objets Settings
    (LytheaSettings, RuneSettings). MAIS beaucoup de code lit des
    variables directement via ``os.getenv(...)`` — par exemple le choix
    du provider web (``LYTHEA_WEB_PROVIDER``) dans web_providers/factory.
    Or ``os.getenv`` ne voit QUE l'environnement shell, jamais le fichier
    .env. Résultat : mettre ``LYTHEA_WEB_PROVIDER=auto`` dans .env n'avait
    aucun effet tant qu'on ne l'exportait pas manuellement dans le shell.

    En chargeant .env ici (très tôt, avant tout le reste), on rend le
    fichier visible à la fois pour pydantic ET pour tous les os.getenv().

    Règles
    ------
    - Ne SURCHARGE PAS une variable déjà présente dans l'environnement
      (``os.environ`` gagne sur .env) : un export shell explicite reste
      prioritaire, ce qui est le comportement attendu.
    - Parsing minimal maison (pas de dépendance à python-dotenv, qui
      n'est pas toujours installé) : lignes ``CLE=VALEUR``, ignore les
      commentaires (#) et les lignes vides, retire les guillemets
      entourant la valeur.
    """
    # Cherche .env dans le répertoire courant puis à la racine du package.
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]
    env_path = next((p for p in candidates if p.is_file()), None)
    if env_path is None:
        return
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip()
            # Retire les guillemets entourants éventuels
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
                val = val[1:-1]
            if not key:
                continue
            # os.environ (shell) est prioritaire — on ne surcharge pas.
            os.environ.setdefault(key, val)
    except Exception:
        # Un .env malformé ne doit jamais empêcher le démarrage.
        pass


def bootstrap_env() -> Path:
    """Set environment variables and return the cache root.

    Returns
    -------
    Path
        The root directory for all Lythéa caches and data.
    """
    # Charge .env AVANT tout : rend les variables visibles pour pydantic
    # ET pour les os.getenv() directs (provider web, etc.).
    _load_dotenv_into_environ()

    platform = detect_platform()

    roots = {
        "runpod": Path("/workspace/.lythea"),
        "colab": Path("/content/.lythea"),
        "kaggle": Path("/kaggle/working/.lythea"),
        "local": Path.home() / ".lythea",
    }
    root = roots[platform]
    root.mkdir(parents=True, exist_ok=True)

    defaults = {
        "HF_HOME": str(root / "hf"),
        "HF_HUB_CACHE": str(root / "hf" / "hub"),
        "TORCH_HOME": str(root / "torch"),
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TRANSFORMERS_VERBOSITY": "error",
        "TOKENIZERS_PARALLELISM": "false",
        # Silence les barres de progression et avertissements bruyants
        # qui polluent la CLI à chaque message (chargement paresseux des
        # modèles auxiliaires : GLiNER, cross-encoder, ONNX Chroma…).
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",   # Fetching/Downloading/Reconstruction
        "HF_HUB_VERBOSITY": "error",
        "HF_HUB_DISABLE_XET": "0",
        "SAFETENSORS_FAST_GPU": "1",
        "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
    }
    for key, val in defaults.items():
        os.environ.setdefault(key, val)

    return root


def ensure_local_searxng(
    port: int = 8080, *, autostart: bool = False, quiet: bool = False
) -> str | None:
    """Détecte (et optionnellement lance) une instance SearXNG locale.

    Pourquoi : les instances SearXNG *publiques* sont massivement
    rate-limitées / bloquées par Google, d'où les « All SearXNG
    instances failed » quand on n'a pas de SearXNG local. ``launch.sh``
    lance un SearXNG self-hosted sur 127.0.0.1:8080 — mais ``rune chat``
    ne passe pas par ``launch.sh``. Cette fonction reproduit le même
    résultat pour la CLI.

    Comportement :
    1. Si ``LYTHEA_SEARXNG_INSTANCE_URL`` est déjà défini → ne fait rien,
       retourne cette URL.
    2. Sinon, sonde ``http://127.0.0.1:<port>/`` : si un SearXNG local
       répond (lancé par un ``launch.sh`` précédent), on exporte son URL
       dans l'environnement et on la retourne.
    3. Sinon, si ``autostart=True`` et que ``searxng_bootstrap.sh`` est
       présent, on le lance (clone + install au 1er run, ~1-2 min) et on
       exporte l'URL obtenue.

    Retourne l'URL SearXNG utilisable, ou ``None`` si aucune (le provider
    composite retombera alors sur DDG).
    """
    import os

    # 1. Déjà configuré (ex: via .env ou launch.sh) → rien à faire.
    existing = (
        os.getenv("LYTHEA_SEARXNG_INSTANCE_URL")
        or os.getenv("SEARXNG_INSTANCE_URL")
    )
    if existing:
        return existing.rstrip("/")

    def _probe(url: str) -> bool:
        import urllib.request
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "rune-probe"})
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    local_url = f"http://127.0.0.1:{port}"

    # 2. Un SearXNG local tourne-t-il déjà ?
    if _probe(local_url + "/healthz") or _probe(local_url + "/"):
        os.environ["LYTHEA_SEARXNG_INSTANCE_URL"] = local_url
        if not quiet:
            print(f"SearXNG local détecté sur {local_url} — utilisé pour la recherche web.")
        return local_url

    # 3. Auto-start optionnel via le script de bootstrap.
    if autostart:
        from pathlib import Path
        import subprocess

        # Cherche searxng_bootstrap.sh à côté du package ou dans le CWD.
        candidates = [
            Path.cwd() / "searxng_bootstrap.sh",
            Path(__file__).resolve().parent.parent / "searxng_bootstrap.sh",
        ]
        script = next((p for p in candidates if p.exists()), None)
        if script is None:
            if not quiet:
                print(
                    "searxng_bootstrap.sh introuvable — recherche web via DDG "
                    "en repli (moins fiable)."
                )
            return None
        if not quiet:
            print(
                f"Lancement de SearXNG local (1re fois : clone + install, ~1-2 min)…"
            )
        try:
            out = subprocess.check_output(
                ["bash", str(script), str(port)],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=600,
            )
        except Exception as exc:
            if not quiet:
                print(f"Échec du bootstrap SearXNG ({exc}) — repli DDG.")
            return None
        # Le script émet SEARXNG_URL=... sur stdout.
        url = None
        for line in out.splitlines():
            if line.startswith("SEARXNG_URL="):
                url = line.split("=", 1)[1].strip()
                break
        if url:
            os.environ["LYTHEA_SEARXNG_INSTANCE_URL"] = url
            if not quiet:
                print(f"SearXNG local actif sur {url}.")
            return url

    return None


PLATFORM = detect_platform()
CACHE_ROOT = bootstrap_env()
