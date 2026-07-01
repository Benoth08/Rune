"""Profil matériel et curseurs adaptatifs.

Principe : Lythéa doit tourner aussi bien sur une carte 24 Go (RTX A5000),
une H100 80 Go, plusieurs GPU, ou un CPU seul — sans recompilation ni
réglage manuel. Ce module détecte la machine au boot et en dérive des
curseurs (taille de batch de génération, K du best-of-N, largeur multi-agent,
parallélisme des tests sandbox). Les règles sont pures et testables ;
la détection torch est isolée dans ``detect()``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

log = logging.getLogger("lythea.hwprofile")


@dataclass(frozen=True)
class HwProfile:
    gpu_count: int
    total_vram_gb: float      # somme sur tous les GPU
    cpu_count: int

    @property
    def tier(self) -> str:
        if self.gpu_count == 0:
            return "cpu"
        if self.gpu_count > 1:
            return "multi-gpu"
        if self.total_vram_gb >= 60:
            return "datacenter"          # H100/A100 80 Go et +
        return "workstation"             # 24-48 Go (A5000, 4090, L40S…)


def detect() -> HwProfile:
    """Inspecte la machine réelle (torch optionnel → CPU-only sinon)."""
    gpus, vram = 0, 0.0
    try:
        import torch
        if torch.cuda.is_available():
            gpus = torch.cuda.device_count()
            for i in range(gpus):
                vram += torch.cuda.get_device_properties(i).total_memory / 1e9
    except Exception:  # noqa: BLE001 — pas de torch / pas de CUDA
        pass
    prof = HwProfile(gpu_count=gpus, total_vram_gb=round(vram, 1),
                     cpu_count=os.cpu_count() or 4)
    log.info("hardware profile: %s (%d GPU, %.0f GB VRAM, %d CPU)",
             prof.tier, prof.gpu_count, prof.total_vram_gb, prof.cpu_count)
    return prof


def knobs(profile: HwProfile, model_size_gb: float = 10.0) -> dict:
    """Curseurs recommandés pour un profil et une empreinte modèle donnés.

    ``batch_max``      : lignes max par passage generate_batch (borné par la
                         marge VRAM restante — le KV cache croît avec le batch).
    ``bestofn``        : K candidats de réparation.
    ``subagents``      : largeur max du futur mode multi-agent.
    ``parallel_tests`` : pytest sandbox concurrents (CPU).
    """
    headroom = max(0.0, profile.total_vram_gb - model_size_gb)
    if profile.tier == "cpu":
        b, k, sub = 1, 2, 1
    elif profile.tier == "workstation":      # ex. A5000 24 Go
        b = 4 if headroom >= 8 else 2
        k, sub = (3, 2) if headroom >= 8 else (2, 2)
    elif profile.tier == "datacenter":       # ex. H100 80 Go
        b, k, sub = 8, 4, 4
    else:                                    # multi-gpu
        b = min(16, 8 * profile.gpu_count)
        k, sub = 4, min(6, 2 * profile.gpu_count)
    return {
        "batch_max": b,
        "bestofn": k,
        "subagents": sub,
        "parallel_tests": max(1, min(profile.cpu_count // 2, 2 * sub)),
    }
