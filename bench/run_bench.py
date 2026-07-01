#!/usr/bin/env python3
"""Benchmark interne de l'agent Lythéa.

Rejoue une suite de tâches types contre le serveur live (SSE) et mesure le
taux de réussite, la durée et le nombre d'étapes par tâche. C'est le juge de
paix de TOUT changement d'orchestrateur/modèle : sans lui on pilote à
l'anecdote, avec lui chaque réglage (température, best-of-N, garde-fou) est
mesuré.

Usage (sur le pod, serveur lancé, modèle chargé) :
    python bench/run_bench.py --base-url http://127.0.0.1:7860 \
        [--tasks bench/tasks.json] [--only email,fibo] [--timeout 900]

Sortie : tableau par tâche + taux global, et un JSON horodaté dans bench/out/.
Dépendance : requests (déjà dans l'environnement Lythéa).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests


def run_task(base_url: str, task: str, timeout: int, kind: str = "code") -> dict:
    """POST /api/agent/run et consomme le flux SSE jusqu'à run_done.

    Verdict adapté au type : code → les tests passent (oracle binaire) ;
    non-code → livrable substantiel produit sans formule d'esquive."""
    out = {"ok": False, "steps": 0, "tests_ok": False, "duration_s": 0.0,
           "warnings": 0, "synthesis": "", "kind": kind}
    t0 = time.time()
    try:
        resp = requests.post(
            f"{base_url}/api/agent/run",
            json={"task": task, "react": True},
            stream=True, timeout=timeout,
        )
        resp.raise_for_status()
        ev_type = ""
        for raw in resp.iter_lines(decode_unicode=True):
            if time.time() - t0 > timeout:
                break
            if not raw:
                continue
            if raw.startswith("event:"):
                ev_type = raw.split(":", 1)[1].strip()
                continue
            if not raw.startswith("data:"):
                continue
            try:
                data = json.loads(raw.split(":", 1)[1].strip())
            except Exception:  # noqa: BLE001
                continue
            if ev_type == "tool_call":
                out["steps"] += 1
                print(f"\n    [{out['steps']:>2}] {data.get('name', '?')}"
                      f"({(data.get('arguments') or {}).get('path', '')})",
                      end="", flush=True)
            elif ev_type == "exec_result":
                out["tests_ok"] = bool(data.get("ok"))
                print(" → " + ("✓ tests verts"
                               if data.get("ok")
                               else f"✗ {data.get('summary', '')}"),
                      end="", flush=True)
            elif ev_type == "agent_warning":
                out["warnings"] += 1
                print(f"\n    ⚠ {(data.get('message') or '')[:90]}",
                      end="", flush=True)
            elif ev_type == "synthesis":
                out["synthesis"] = (data.get("text") or "")[:400]
            elif ev_type == "run_done":
                out["ok"] = bool(data.get("ok", out["tests_ok"]))
                break
            elif ev_type == "run_error":
                out["synthesis"] = f"run_error: {data.get('error', '?')}"
                break
    except Exception as exc:  # noqa: BLE001
        out["synthesis"] = f"harness error: {exc}"
    out["duration_s"] = round(time.time() - t0, 1)
    if kind == "code":
        out["ok"] = out["tests_ok"] or out["ok"]
    else:
        syn = (out["synthesis"] or "").lower()
        evasive = any(e in syn for e in (
            "je suppose", "supposons", "vous pouvez supposer", "à compléter"))
        out["ok"] = bool(out["ok"]) and len(out["synthesis"]) > 80 and not evasive
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:7860")
    ap.add_argument("--tasks", default=str(Path(__file__).parent / "tasks.json"))
    ap.add_argument("--only", default="", help="ids séparés par des virgules")
    ap.add_argument("--timeout", type=int, default=900, help="par tâche (s)")
    args = ap.parse_args()

    tasks = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
    if args.only:
        keep = {t.strip() for t in args.only.split(",") if t.strip()}
        tasks = [t for t in tasks if t["id"] in keep]

    results = []
    for i, t in enumerate(tasks, 1):
        print(f"[{i}/{len(tasks)}] {t['id']} (~5-10 min/tâche sur 14B) …",
              flush=True)
        r = run_task(args.base_url, t["task"], args.timeout,
                     kind=t.get("kind", "code"))
        r["id"] = t["id"]
        results.append(r)
        print("\n    " + ("✓ RÉUSSI" if r["ok"] else "✗ ÉCHEC")
              + f"  {r['duration_s']}s, {r['steps']} étapes"
              + (f", {r['warnings']} ⚠" if r["warnings"] else ""), flush=True)

    n_ok = sum(1 for r in results if r["ok"])
    print("\n── Bilan ─────────────────────────────")
    for r in results:
        print(f"  {'✓' if r['ok'] else '✗'} {r['id']:<12} [{r.get('kind','code'):<9}] "
              f"{r['duration_s']:>7.1f}s  {r['steps']:>2} étapes")
    # ventilation code / non-code (le bench reste centré code mais généraliste)
    code = [r for r in results if r.get("kind", "code") == "code"]
    noncode = [r for r in results if r.get("kind", "code") != "code"]
    if code:
        c_ok = sum(1 for r in code if r["ok"])
        print(f"\n  code     : {c_ok}/{len(code)}")
    if noncode:
        nc_ok = sum(1 for r in noncode if r["ok"])
        print(f"  non-code : {nc_ok}/{len(noncode)}")
    print(f"\nTaux de réussite global : {n_ok}/{len(results)} "
          f"({100.0 * n_ok / max(1, len(results)):.0f}%)")

    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"bench-{stamp}.json"
    out_path.write_text(
        json.dumps({"when": stamp, "results": results,
                    "success_rate": n_ok / max(1, len(results))},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"Détail : {out_path}")


if __name__ == "__main__":
    main()
