"""SDM persistence: addresses must survive save/load or reads corrupt."""

from __future__ import annotations

import pytest

try:
    import torch
except (ImportError, OSError):
    pytest.skip("torch not available or broken CUDA", allow_module_level=True)


def test_sdm_addresses_persist_across_reload(tmp_path):
    from rune.memory.sdm import SDM

    a = SDM(dim=32, rows=64, k=4, device="cpu")
    # Mutate addresses so they differ from a fresh (seeded) instance — this is
    # what makes the test actually prove RESTORATION, not just seed-equality.
    a.addresses = a.addresses * -1.0
    a.write(torch.randn(32), torch.randn(32))
    path = tmp_path / "sdm.pt"
    a.save(path)

    b = SDM(dim=32, rows=64, k=4, device="cpu")
    assert not torch.allclose(b.addresses, a.addresses)   # differ before load
    b.load_state(path)
    assert torch.allclose(b.addresses, a.addresses), "addresses not restored"
    assert torch.allclose(b.contents, a.contents), "contents not restored"


def test_sdm_addresses_seeded_deterministic():
    from rune.memory.sdm import SDM
    a = SDM(dim=16, rows=32, k=2, device="cpu")
    b = SDM(dim=16, rows=32, k=2, device="cpu")
    assert torch.allclose(a.addresses, b.addresses), "addresses not deterministic"
