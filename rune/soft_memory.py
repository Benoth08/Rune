"""Soft memory via prefix-tuning — opt-in long-term parametric adaptation.

Why prefix-tuning over LoRA?
============================
Both prefix-tuning and LoRA are parameter-efficient ways to adapt a
pretrained LLM to new content. We chose **prefix-tuning** for Lythéa
for the following reasons specific to this project:

1. **Frozen base model.** Prefix-tuning never modifies the base model
   weights — it prepends learned key/value vectors to each attention
   layer's KV cache. This makes catastrophic forgetting structurally
   impossible: the model's general capabilities are preserved exactly,
   and the only thing that can degrade is the prefix itself (which is
   versioned and rollbackable).

2. **Smaller parameter count.** For Qwen2.5-7B, a prefix of length 32
   across all 28 layers totals ~24 MB (32 × 28 × 2 × 3584 × 2 bytes
   in bf16). LoRA on the same model with rank 16 across q/k/v/o would
   weigh ~120 MB. Smaller parameters mean faster training, less I/O,
   and easier checkpointing — important for a "memory" feature that
   should consolidate frequently.

3. **Localised effect.** Prefix vectors only influence the attention
   layers; they cannot rewrite token embeddings or feedforward weights.
   This keeps the soft memory's behaviour predictable and limits the
   blast radius of a bad training run.

4. **Trivial composition.** Multiple prefixes can be averaged or
   stacked at inference time — useful for combining "domain prefix"
   (e.g. one trained on technical conversations) with "user prefix"
   (one trained on this user's history). LoRA composition exists too
   but requires careful weight merging to avoid drift.

When LoRA might be a future upgrade
-----------------------------------
If, after extensive use, the prefix capacity becomes a bottleneck
(measurable as: prefix training plateaus before reaching target
perplexity on the user's own history), LoRA could be added as a
secondary mechanism — kept narrow (rank 4-8) and applied only to
the q_proj layers to retain the no-forgetting guarantee. This module
is structured so that adding a LoRAAdapter sibling class would be
an additive change.

Opt-in design
-------------
Soft memory is **off by default** (env: ``LYTHEA_ENABLE_SOFT_MEMORY=1``).
The reasons:
- It requires extra GPU memory (training step uses ~2× inference VRAM).
- It introduces non-determinism into the model's outputs.
- It needs a meaningful corpus to train on (the user's own history)
  before it produces benefits, so enabling it on a brand-new install
  with an empty memory would just slow things down.

When enabled, training is triggered manually by the operator via
``POST /api/soft-memory/train``. A typical workflow is to let Lythéa
accumulate a few hundred exchanges of memory, then run a training
round, then evaluate, then run another round, etc. Versioned
checkpoints under ``data/soft_memory/<timestamp>.pt`` allow rollback
to any prior state if a training round damages the prefix.

Files
-----
- :class:`SoftPrefix` — the learnable parameter container.
- :class:`SoftMemoryTrainer` — collects training data, runs gradient
  steps, and persists checkpoints.
- The HFModelWrapper exposes :meth:`attach_soft_prefix` /
  :meth:`detach_soft_prefix` to install or remove the prefix from the
  active model's KV cache pathway.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# torch is required for the actual training; we import lazily so the
# module can be inspected (docstrings, settings) without GPU stack.
try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    _TORCH = False

log = logging.getLogger("rune.soft_memory")


# ── Config ────────────────────────────────────────────────────────────

@dataclass
class SoftMemoryConfig:
    """Hyperparameters for prefix-tuning training and runtime.

    Defaults are tuned for Qwen2.5-7B-class models. For smaller
    models (3B), reduce ``prefix_length`` to ~16 to keep training fast.
    """

    # Length of the learnable prefix in tokens (per layer, per head).
    prefix_length: int = 32
    # Training learning rate for the prefix params.
    learning_rate: float = 1e-3
    # Number of full passes over the training set per ``train()`` call.
    epochs: int = 1
    # Mini-batch size during training.
    batch_size: int = 4
    # Maximum input sequence length (for both training and at inference).
    max_seq_length: int = 512
    # Storage root for checkpoints and metadata.
    storage_dir: Path = field(
        default_factory=lambda: Path("data/soft_memory")
    )


# ── Soft prefix module ────────────────────────────────────────────────

class SoftPrefix:
    """Learnable key/value prefix prepended to each attention layer.

    The prefix is a tensor of shape::

        [num_layers, 2, prefix_length, hidden_dim]

    where the second dimension is ``[keys, values]``. At inference,
    these are concatenated with the existing KV cache before attention,
    so the prefix tokens influence every subsequent generation step.

    This class wraps a :class:`torch.nn.Parameter` so it can be
    optimised by a standard optimizer, and provides save/load helpers
    that handle versioning with timestamps.
    """

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        config: SoftMemoryConfig,
        dtype: Any = None,
        device: str = "cpu",
    ) -> None:
        if not _TORCH:
            raise RuntimeError(
                "SoftPrefix requires torch. Install lythea with the "
                "[ml] extras."
            )
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.config = config
        self.dtype = dtype if dtype is not None else torch.bfloat16
        self.device = device

        # Initialise small to avoid disrupting the base model on first use.
        # Scale 0.01 is a common starting value for prefix-tuning.
        shape = (num_layers, 2, config.prefix_length, hidden_dim)
        init = torch.randn(shape, dtype=self.dtype, device=device) * 0.01
        self.params = nn.Parameter(init, requires_grad=True)

    # ── Persistence ────────────────────────────────────────────────────

    def save(self, path: Path | str, metadata: dict | None = None) -> None:
        """Save the prefix to ``path`` along with optional metadata."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "params": self.params.detach().cpu(),
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "prefix_length": self.config.prefix_length,
            "metadata": metadata or {},
            "saved_at": time.time(),
        }
        torch.save(payload, str(path))
        log.info(
            "Soft prefix saved → %s (%.2f MB)",
            path, path.stat().st_size / 1e6,
        )

    @classmethod
    def load(
        cls,
        path: Path | str,
        config: SoftMemoryConfig,
        device: str = "cpu",
    ) -> "SoftPrefix":
        """Reload a prefix from disk and return a new SoftPrefix."""
        if not _TORCH:
            raise RuntimeError("torch required to load SoftPrefix")
        payload = torch.load(str(path), map_location=device, weights_only=False)
        prefix = cls(
            num_layers=payload["num_layers"],
            hidden_dim=payload["hidden_dim"],
            config=config,
            device=device,
        )
        # Replace the parameter with the saved values.
        with torch.no_grad():
            prefix.params.data.copy_(payload["params"].to(device))
        log.info("Soft prefix loaded ← %s", path)
        return prefix


# ── Trainer ───────────────────────────────────────────────────────────

class SoftMemoryTrainer:
    """Train a SoftPrefix on the user's exchange history.

    Workflow
    --------
    1. ``collect_dataset()`` extracts (input, target) pairs from recent
       MHN/Chroma content.
    2. ``train_step()`` runs gradient descent on the prefix params,
       keeping the base model frozen.
    3. ``save()`` writes a versioned checkpoint with metadata.
    4. ``rollback_to(timestamp)`` reloads any prior checkpoint.

    The trainer NEVER touches the base model — only the prefix params
    receive gradients, and we run the optimizer with ``params=[prefix.params]``.
    """

    def __init__(
        self,
        prefix: SoftPrefix,
        config: SoftMemoryConfig,
    ) -> None:
        self.prefix = prefix
        self.config = config
        self.config.storage_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.config.storage_dir / "index.json"
        self._index = self._load_index()

    # ── Dataset ────────────────────────────────────────────────────────

    def collect_dataset(
        self,
        chroma_collection: Any,
        max_examples: int = 200,
    ) -> list[dict]:
        """Pull recent ``"exchange"``-tagged docs from Chroma as training data.

        Returns a list of ``{"input_text": ..., "target_text": ...}``
        pairs. A plain language-model objective is used: predict the
        response given the query.
        """
        try:
            data = chroma_collection.get(include=["documents", "metadatas"])
        except Exception as exc:
            log.warning("collect_dataset: chroma read failed: %s", exc)
            return []

        examples: list[dict] = []
        for doc, meta in zip(
            data.get("documents", []),
            data.get("metadatas", []),
        ):
            if not isinstance(doc, str) or "Q:" not in doc or "R:" not in doc:
                continue
            try:
                _, rest = doc.split("Q:", 1)
                q, r = rest.split("R:", 1)
                q = q.strip()
                r = r.split("[Atoms:")[0].strip()
                if q and r:
                    examples.append({"input_text": q, "target_text": r})
            except Exception:
                continue
            if len(examples) >= max_examples:
                break
        log.info("Collected %d training examples for soft memory", len(examples))
        return examples

    # ── Training step ──────────────────────────────────────────────────

    def train(
        self,
        examples: list[dict],
        model: Any,
        tokenizer: Any,
    ) -> dict:
        """Run ``epochs`` passes over ``examples``.

        The base ``model`` MUST be in eval mode and have all its
        parameters frozen (``requires_grad=False``). We attach the
        prefix to its forward hook (caller's responsibility — see
        ``HFModelWrapper.attach_soft_prefix``), then run a standard
        cross-entropy LM loss with the optimizer touching only
        ``self.prefix.params``.

        Returns
        -------
        dict
            Stats: ``loss_initial``, ``loss_final``, ``n_steps``,
            ``examples_seen``.
        """
        if not _TORCH:
            return {"error": "torch unavailable"}
        if not examples:
            return {"error": "no_training_data"}

        # Sanity check: base model frozen
        for p in model.parameters():
            assert not p.requires_grad, (
                "Base model must be frozen during prefix training. "
                "Call .requires_grad_(False) on all model params first."
            )

        optimizer = torch.optim.AdamW(
            [self.prefix.params],
            lr=self.config.learning_rate,
        )
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

        stats = {
            "loss_initial": None, "loss_final": None,
            "n_steps": 0, "examples_seen": 0,
        }
        running_losses: list[float] = []

        for epoch in range(self.config.epochs):
            for batch_start in range(0, len(examples), self.config.batch_size):
                batch = examples[batch_start:batch_start + self.config.batch_size]
                loss = self._batch_step(batch, model, tokenizer, loss_fn)
                if loss is None:
                    continue

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                stats["n_steps"] += 1
                stats["examples_seen"] += len(batch)
                running_losses.append(loss.item())

                if stats["loss_initial"] is None:
                    stats["loss_initial"] = loss.item()
                stats["loss_final"] = loss.item()

        if running_losses:
            log.info(
                "Soft memory train: %d steps, loss %.3f → %.3f",
                stats["n_steps"], running_losses[0], running_losses[-1],
            )
        return stats

    def _batch_step(
        self, batch: list[dict], model: Any, tokenizer: Any, loss_fn: Any,
    ) -> Any | None:
        """One forward + loss pass over a batch."""
        try:
            # Concatenate input + target with a separator the model will see
            texts = [
                f"{ex['input_text']}\n{ex['target_text']}"
                for ex in batch
            ]
            enc = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_length,
            )
            enc = {k: v.to(model.device) for k, v in enc.items()}

            # Standard LM loss; the prefix is injected via the wrapper's
            # forward hook (see HFModelWrapper.attach_soft_prefix).
            outputs = model(
                **enc, labels=enc["input_ids"],
            )
            return outputs.loss
        except Exception as exc:
            log.warning("Soft memory batch step failed: %s", exc)
            return None

    # ── Versioning ─────────────────────────────────────────────────────

    def save_checkpoint(self, label: str = "") -> str:
        """Save the current prefix as a timestamped checkpoint.

        Returns the checkpoint identifier. The default format is
        ``"YYYYmmdd_HHMMSS"``; if two saves happen within the same
        second, a millisecond suffix (``"_NNN"``) is appended to the
        second one to prevent collision in the index.
        """
        ts = time.strftime("%Y%m%d_%H%M%S")
        # Disambiguate sub-second saves so the index never overwrites
        # a previous entry. We append milliseconds-mod-1000 which gives
        # a 1000-fold safety margin within any given second.
        if ts in self._index:
            ts = f"{ts}_{int(time.time() * 1000) % 1000:03d}"
        ckpt_path = self.config.storage_dir / f"{ts}.pt"
        self.prefix.save(ckpt_path, metadata={"label": label})

        self._index[ts] = {
            "path": str(ckpt_path),
            "label": label,
            "saved_at": time.time(),
        }
        self._save_index()
        return ts

    def rollback_to(self, checkpoint_id: str) -> bool:
        """Reload the prefix from a previous checkpoint.

        Returns False if the checkpoint id is unknown.
        """
        entry = self._index.get(checkpoint_id)
        if entry is None:
            return False
        loaded = SoftPrefix.load(
            entry["path"],
            config=self.config,
            device=self.prefix.device,
        )
        with torch.no_grad():
            self.prefix.params.data.copy_(loaded.params.data)
        log.info("Soft prefix rolled back to %s", checkpoint_id)
        return True

    def list_checkpoints(self) -> list[dict]:
        """Return all checkpoints sorted newest first."""
        items = list(self._index.items())
        items.sort(key=lambda kv: kv[1]["saved_at"], reverse=True)
        return [{"id": k, **v} for k, v in items]

    def reset(self) -> None:
        """Reset the prefix to small random values, deleting all checkpoints."""
        # Re-init in place
        with torch.no_grad():
            self.prefix.params.data.normal_(mean=0.0, std=0.01)
        # Remove checkpoints
        for entry in self._index.values():
            try:
                Path(entry["path"]).unlink(missing_ok=True)
            except Exception:
                pass
        self._index.clear()
        self._save_index()
        log.info("Soft memory reset (prefix re-initialised, checkpoints cleared)")

    # ── Index helpers ──────────────────────────────────────────────────

    def _load_index(self) -> dict:
        if self._index_path.exists():
            try:
                return json.loads(self._index_path.read_text("utf-8"))
            except Exception:
                pass
        return {}

    def _save_index(self) -> None:
        tmp = self._index_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._index, f, ensure_ascii=False, indent=2)
        os.replace(str(tmp), str(self._index_path))
