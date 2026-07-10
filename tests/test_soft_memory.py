"""Tests for soft memory (prefix-tuning) — opt-in feature."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

torch_available = True
try:
    import torch
except (ImportError, OSError):
    torch_available = False

from rune.soft_memory import SoftMemoryConfig


# ── Pure logic that doesn't need torch ───────────────────────────────

def test_config_defaults():
    cfg = SoftMemoryConfig()
    assert cfg.prefix_length == 32
    assert cfg.learning_rate == 1e-3
    assert cfg.epochs == 1
    assert cfg.batch_size == 4


def test_config_overrides():
    cfg = SoftMemoryConfig(
        prefix_length=16, learning_rate=5e-4, epochs=3,
    )
    assert cfg.prefix_length == 16
    assert cfg.learning_rate == 5e-4
    assert cfg.epochs == 3


# ── Tests requiring torch ────────────────────────────────────────────

@pytest.mark.skipif(not torch_available, reason="torch not installed")
def test_soft_prefix_initialised_small():
    """Prefix params start near zero so they don't disturb the model."""
    from rune.soft_memory import SoftPrefix
    cfg = SoftMemoryConfig(prefix_length=8)
    prefix = SoftPrefix(num_layers=4, hidden_dim=64, config=cfg)

    # Shape: [num_layers, 2, prefix_length, hidden_dim]
    assert prefix.params.shape == (4, 2, 8, 64)
    # Values should be small (≈ N(0, 0.01))
    assert prefix.params.abs().max().item() < 0.5


@pytest.mark.skipif(not torch_available, reason="torch not installed")
def test_soft_prefix_save_and_load_roundtrip():
    from rune.soft_memory import SoftPrefix
    with tempfile.TemporaryDirectory() as tmp:
        cfg = SoftMemoryConfig(prefix_length=4, storage_dir=Path(tmp))
        p1 = SoftPrefix(num_layers=2, hidden_dim=8, config=cfg)
        ckpt = Path(tmp) / "test.pt"
        p1.save(ckpt, metadata={"label": "test"})

        p2 = SoftPrefix.load(ckpt, config=cfg)
        # Values are bit-identical
        assert torch.allclose(p1.params.detach().cpu(),
                              p2.params.detach().cpu())
        assert p2.num_layers == p1.num_layers
        assert p2.hidden_dim == p1.hidden_dim


@pytest.mark.skipif(not torch_available, reason="torch not installed")
def test_trainer_collect_dataset_parses_chroma_docs():
    from rune.soft_memory import SoftMemoryTrainer, SoftPrefix
    with tempfile.TemporaryDirectory() as tmp:
        cfg = SoftMemoryConfig(storage_dir=Path(tmp))
        prefix = SoftPrefix(num_layers=1, hidden_dim=4, config=cfg)
        trainer = SoftMemoryTrainer(prefix, cfg)

        # Fake chroma collection with two valid Q/R docs and one invalid
        fake_chroma = MagicMock()
        fake_chroma.get.return_value = {
            "documents": [
                "Q: Comment ça va ?\nR: Très bien, merci !\n[Atoms: x]",
                "Q: Quel temps fait-il ?\nR: Il pleut.\n",
                "totally unrelated text",  # ignored
            ],
            "metadatas": [{}, {}, {}],
        }
        examples = trainer.collect_dataset(fake_chroma)
        assert len(examples) == 2
        assert examples[0]["input_text"] == "Comment ça va ?"
        assert examples[0]["target_text"] == "Très bien, merci !"


@pytest.mark.skipif(not torch_available, reason="torch not installed")
def test_trainer_save_and_list_checkpoints():
    from rune.soft_memory import SoftMemoryTrainer, SoftPrefix
    with tempfile.TemporaryDirectory() as tmp:
        cfg = SoftMemoryConfig(storage_dir=Path(tmp))
        prefix = SoftPrefix(num_layers=1, hidden_dim=4, config=cfg)
        trainer = SoftMemoryTrainer(prefix, cfg)

        # Two consecutive saves: even within the same second, the
        # checkpoint IDs must be distinct (the trainer auto-appends a
        # millisecond suffix on collision).
        ck1 = trainer.save_checkpoint("first")
        ck2 = trainer.save_checkpoint("second")
        assert ck1 != ck2

        ckpts = trainer.list_checkpoints()
        assert len(ckpts) == 2
        # Newest first
        assert ckpts[0]["id"] == ck2
        assert ckpts[0]["label"] == "second"


@pytest.mark.skipif(not torch_available, reason="torch not installed")
def test_trainer_rollback_restores_prefix_values():
    from rune.soft_memory import SoftMemoryTrainer, SoftPrefix
    with tempfile.TemporaryDirectory() as tmp:
        cfg = SoftMemoryConfig(storage_dir=Path(tmp))
        prefix = SoftPrefix(num_layers=1, hidden_dim=4, config=cfg)
        trainer = SoftMemoryTrainer(prefix, cfg)

        # Save initial state
        ck1 = trainer.save_checkpoint("initial")
        initial = prefix.params.detach().clone()

        # Mutate the prefix
        with torch.no_grad():
            prefix.params.data.fill_(0.5)
        assert not torch.allclose(prefix.params.detach(), initial)

        # Rollback
        ok = trainer.rollback_to(ck1)
        assert ok
        assert torch.allclose(prefix.params.detach(), initial)


@pytest.mark.skipif(not torch_available, reason="torch not installed")
def test_trainer_rollback_unknown_returns_false():
    from rune.soft_memory import SoftMemoryTrainer, SoftPrefix
    with tempfile.TemporaryDirectory() as tmp:
        cfg = SoftMemoryConfig(storage_dir=Path(tmp))
        prefix = SoftPrefix(num_layers=1, hidden_dim=4, config=cfg)
        trainer = SoftMemoryTrainer(prefix, cfg)
        assert trainer.rollback_to("nonexistent") is False


@pytest.mark.skipif(not torch_available, reason="torch not installed")
def test_trainer_reset_clears_everything():
    from rune.soft_memory import SoftMemoryTrainer, SoftPrefix
    with tempfile.TemporaryDirectory() as tmp:
        cfg = SoftMemoryConfig(storage_dir=Path(tmp))
        prefix = SoftPrefix(num_layers=1, hidden_dim=4, config=cfg)
        trainer = SoftMemoryTrainer(prefix, cfg)

        # Mutate + save
        with torch.no_grad():
            prefix.params.data.fill_(0.7)
        trainer.save_checkpoint("a")
        trainer.save_checkpoint("b")
        assert len(trainer.list_checkpoints()) == 2

        # Reset
        trainer.reset()
        assert len(trainer.list_checkpoints()) == 0
        # New values are small (re-init with std=0.01)
        assert prefix.params.abs().max().item() < 0.5


@pytest.mark.skipif(not torch_available, reason="torch not installed")
def test_train_no_examples_returns_error():
    from rune.soft_memory import SoftMemoryTrainer, SoftPrefix
    with tempfile.TemporaryDirectory() as tmp:
        cfg = SoftMemoryConfig(storage_dir=Path(tmp))
        prefix = SoftPrefix(num_layers=1, hidden_dim=4, config=cfg)
        trainer = SoftMemoryTrainer(prefix, cfg)
        result = trainer.train([], MagicMock(), MagicMock())
        assert result.get("error") == "no_training_data"


@pytest.mark.skipif(not torch_available, reason="torch not installed")
def test_train_assertion_when_base_model_not_frozen():
    """Training MUST refuse to run if base model has trainable params.

    This is the safety guarantee that prevents accidental fine-tuning
    of the base LLM (which is what makes prefix-tuning safe).
    """
    from rune.soft_memory import SoftMemoryTrainer, SoftPrefix
    with tempfile.TemporaryDirectory() as tmp:
        cfg = SoftMemoryConfig(storage_dir=Path(tmp))
        prefix = SoftPrefix(num_layers=1, hidden_dim=4, config=cfg)
        trainer = SoftMemoryTrainer(prefix, cfg)

        # Fake model with a trainable parameter — this MUST raise
        fake_model = MagicMock()
        bad_param = torch.zeros(2, requires_grad=True)
        fake_model.parameters = MagicMock(return_value=[bad_param])
        fake_tokenizer = MagicMock()

        with pytest.raises(AssertionError):
            trainer.train(
                [{"input_text": "hi", "target_text": "hello"}],
                fake_model, fake_tokenizer,
            )
