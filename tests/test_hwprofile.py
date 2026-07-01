"""Curseurs adaptatifs : mêmes règles du CPU seul au multi-H100."""

from rune.hwprofile import HwProfile, knobs


def test_tiers():
    assert HwProfile(0, 0.0, 8).tier == "cpu"
    assert HwProfile(1, 24.0, 16).tier == "workstation"     # A5000
    assert HwProfile(1, 80.0, 32).tier == "datacenter"      # H100
    assert HwProfile(4, 320.0, 64).tier == "multi-gpu"


def test_knobs_scale_with_hardware():
    cpu = knobs(HwProfile(0, 0.0, 8))
    a5000 = knobs(HwProfile(1, 24.0, 16), model_size_gb=10.0)   # 14B NF4
    a5000_30b = knobs(HwProfile(1, 24.0, 16), model_size_gb=18.0)  # 30B serré
    h100 = knobs(HwProfile(1, 80.0, 32))
    multi = knobs(HwProfile(4, 320.0, 64))

    assert cpu["batch_max"] == 1 and cpu["subagents"] == 1
    assert a5000["batch_max"] == 4 and a5000["bestofn"] == 3
    # marge réduite (gros modèle sur 24 Go) → curseurs resserrés, pas d'OOM
    assert a5000_30b["batch_max"] == 2 and a5000_30b["bestofn"] == 2
    assert h100["batch_max"] == 8 and h100["subagents"] == 4
    assert multi["batch_max"] == 16 and multi["subagents"] == 6
    # le parallélisme CPU reste borné par les cœurs
    assert cpu["parallel_tests"] >= 1
    assert multi["parallel_tests"] <= 64 // 2
