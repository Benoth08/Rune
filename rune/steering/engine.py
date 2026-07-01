"""Activation-steering engine — torch bridge (V6 steering BETA).

Computes CAA steering vectors on the *currently loaded* model, picks the
layers automatically, caches per model, and injects ``alpha *
_STEER_FRACTION * ||h|| * v̂_l`` (a bounded fraction of the live activation
norm) into the residual stream of the selected layers via forward hooks
(same mechanism family as the existing latent/prefix hooks).

torch is imported lazily inside methods so this module — and the pure
``vectors``/``axes`` — stay importable (and testable) without torch.

BETA / honesty: the numerical core (``vectors.py``) is unit-tested, but
the torch path (capture, hidden-state indexing, decoder-layer output
shape) can vary slightly by architecture and is validated on a live model
on the pod, not here. Everything is defensive: a failure degrades to
"steering unavailable" without breaking generation. Steering never
bypasses the inhibition guardrails — it only nudges style/tone.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from rune.steering import vectors as V
from rune.steering.axes import AXES

log = logging.getLogger(__name__)

_SEARCH_PATHS = [
    "model.layers", "model.model.layers", "model.model.model.layers",
    "language_model.model.layers", "transformer.h", "gpt_neox.layers",
]
_ALPHA_MAX = 4.0          # hard clamp on the user coefficient (coherence)
_STEER_FRACTION = 0.10    # per-|alpha| fraction of the LIVE activation norm
_COMBINED_MAX = 0.15      # GLOBAL cap on the summed per-layer nudge (multi-axis)
_COLLINEAR_WARN = 0.7     # |cos| above which two combined axes are redundant
#   added per selected layer (split across layers). Measured on Qwen2.5-7B:
#   ~25 %/layer corrupts words & repeats, so we cap the whole range BELOW
#   that — at |alpha|=4 (max) ≈ 13 %, |alpha|=2 ≈ 7 %. Clean across the
#   range; the concision axis is fragile, so this is the honest ceiling
#   for gain tuning (stronger needs a cleaner axis, not more gain).
_MAX_TOKENS = 64          # contrastive prompts are short
_CAUSAL_SHORTLIST = 6     # most-separable band layers to causally probe
_CAUSAL_WEIGHT = 0.6      # weight of causal efficacy vs separability in select


def _decoder_layers(model):
    """Return the decoder layer ModuleList across architectures, or None."""
    best = None
    for attr in _SEARCH_PATHS:
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, "__len__") and len(obj) > 0:
                if best is None or len(obj) > len(best):
                    best = obj
        except (AttributeError, TypeError):
            continue
    return best


class SteeringEngine:
    """One engine per app, bound to the in-process model wrapper."""

    def __init__(self, wrapper, cache_dir: str | Path | None = None):
        self.wrapper = wrapper
        self.cache_dir = Path(cache_dir or (Path.home() / ".lythea" / "steering"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._handles: list = []
        self._mix: dict[str, float] = {}   # axis -> alpha (the active mix)
        self._calib: dict[str, dict] = {}  # (model_id|axis) -> result

    # ── introspection ────────────────────────────────────────────────
    def _model_id(self) -> str:
        return getattr(self.wrapper, "model_id", "") or "unknown"

    def state(self) -> dict:
        mix = self._mix
        primary = next(iter(mix), None)
        return {
            "active_axis": primary,                       # back-compat
            "alpha": mix.get(primary, 0.0) if primary else 0.0,
            "active": dict(mix),                          # full mix {axis: alpha}
            "alpha_max": _ALPHA_MAX,
            "combined_max": _COMBINED_MAX,
            "axes": [
                {"name": k, "label": v["label"], "description": v["description"],
                 "calibrated": self._cache_path(k).exists()}
                for k, v in AXES.items()
            ],
        }

    def _cache_path(self, axis: str) -> Path:
        safe = self._model_id().replace("/", "_")
        return self.cache_dir / f"{safe}__{axis}.json"

    # ── calibration ──────────────────────────────────────────────────
    def _capture(self, texts: list[str], pool: str = "mean"):
        """Return per-layer pooled activations for each text.

        ``pool="mean"`` averages over the content tokens — a cleaner
        contrastive signal than a single position, since the last token is
        often shared punctuation/template across prompts and dilutes the
        difference. ``pool="last"`` keeps the original final-token behaviour.

        Output: list over texts of {layer_index: numpy vector}.
        """
        import torch

        model = self.wrapper.model
        tok = self.wrapper.tokenizer
        per_text: list[dict] = []
        for text in texts:
            ids = tok(text, return_tensors="pt", truncation=True, max_length=_MAX_TOKENS)
            ids = {k: v.to(model.device) for k, v in ids.items()}
            with torch.no_grad():
                out = model(**ids, output_hidden_states=True)
            hs = out.hidden_states  # tuple len L+1; hs[0]=embeddings
            vecs = {}
            for li in range(len(hs) - 1):           # 0-indexed decoder layer
                layer_hs = hs[li + 1][0]            # (T, H); batch size 1, no pad
                if pool == "last":
                    v = layer_hs[-1, :].float().cpu().numpy()
                else:
                    v = layer_hs.mean(dim=0).float().cpu().numpy()
                vecs[li] = v
            per_text.append(vecs)
            del out
        return per_text

    def calibrate(self, axis: str) -> dict:
        import numpy as np

        if axis not in AXES:
            raise ValueError(f"Axe inconnu : {axis}")
        if not getattr(self.wrapper, "is_loaded", False):
            raise RuntimeError("Modèle non chargé : calibration impossible.")

        pairs = AXES[axis]["pairs"]
        pos_texts = [p for p, _ in pairs]
        neg_texts = [n for _, n in pairs]
        # Mean-pooled extraction → cleaner contrastive signal than a single token.
        pos_caps = self._capture(pos_texts, pool="mean")
        neg_caps = self._capture(neg_texts, pool="mean")

        num_layers = len(pos_caps[0]) if pos_caps else 0
        layers_vec: dict[int, list[float]] = {}   # unit direction per layer
        scales: dict[int, float] = {}
        sep: dict[int, float] = {}                # separability (cohens_d)
        for li in range(num_layers):
            pos = np.stack([c[li] for c in pos_caps])
            neg = np.stack([c[li] for c in neg_caps])
            # Robust contrastive direction (mean diff; see vectors.diff_of_means).
            direction = V.diff_of_means(pos, neg)
            sep[li] = V.cohens_d(pos, neg, direction)
            layers_vec[li] = V.unit(direction).tolist()
            allv = np.concatenate([pos, neg], axis=0)
            scales[li] = float(np.linalg.norm(allv, axis=1).mean())

        # ── Causal layer selection ──────────────────────────────────────
        # Separability tells us where the axis is *readable*; it is not where
        # injecting it best *moves the output*. Shortlist the most separable
        # band layers, then measure each one's real causal effect (how far
        # injecting it shifts the final hidden along the axis) and select on a
        # blend that favours causal efficacy. Degrades to separability-only.
        shortlist = V.select_layers(sep, num_layers, band=(0.4, 0.7),
                                    top_k=_CAUSAL_SHORTLIST)
        causal: dict[int, float] = {}
        try:
            d_final = np.asarray(layers_vec[num_layers - 1], dtype=np.float64)
            d_final = d_final / (np.linalg.norm(d_final) or 1.0)
            for li in shortlist:
                causal[li] = self._causal_effect(
                    layers_vec[li], li, pos_texts + neg_texts, d_final,
                )
        except Exception:  # noqa: BLE001
            log.exception("steering causal scoring failed; separability only")
            causal = {}

        if causal:
            ranking = V.blend_scores(
                {l: sep[l] for l in shortlist}, causal, w_causal=_CAUSAL_WEIGHT,
            )
            selected = sorted(
                sorted(ranking, key=lambda l: ranking[l], reverse=True)[:3]
            )
        else:
            selected = V.select_layers(sep, num_layers, band=(0.4, 0.7), top_k=3)
        if not selected:  # ultra-defensive
            selected = V.select_layers(sep, num_layers, band=(0.4, 0.7), top_k=3)

        result = {
            "axis": axis,
            "model_id": self._model_id(),
            "num_layers": num_layers,
            "layers": selected,
            "vectors": {str(l): layers_vec[l] for l in selected},
            "scales": {str(l): scales[l] for l in selected},
            "scores": {str(l): sep[l] for l in selected},
            "causal": {str(l): causal[l] for l in selected if l in causal},
            "selection": "causal_blend" if causal else "separability",
            "alpha_max": _ALPHA_MAX,
        }
        try:
            self._cache_path(axis).write_text(json.dumps(result), encoding="utf-8")
        except Exception:  # noqa: BLE001
            log.exception("steering cache write failed")
        self._calib[f"{self._model_id()}|{axis}"] = result
        log.info("steering calibrated: %s on %s → layers %s (%s)",
                 axis, self._model_id(), selected, result["selection"])
        return result

    def _causal_effect(self, unit_vec, layer_idx, texts, d_final) -> float:
        """How strongly injecting ``unit_vec`` at ``layer_idx`` moves the final
        hidden state along the axis direction ``d_final``.

        For each text, compare a clean forward to one with a temporary hook that
        adds ``(probe_alpha * frac) * ||h|| * v̂`` at ``layer_idx`` (the same
        injection form as inference), measuring the change in the projection of
        the mean-pooled final hidden onto ``d_final``. Returns the mean signed
        shift — larger ⇒ this layer steers the output more. Pod path; any
        failure propagates to the caller's separability fallback.
        """
        import numpy as np
        import torch

        model = self.wrapper.model
        tok = self.wrapper.tokenizer
        layers = _decoder_layers(model)
        if layers is None:
            return 0.0
        try:
            dtype = next(model.parameters()).dtype
            device = next(model.parameters()).device
        except StopIteration:
            dtype, device = torch.float32, "cpu"

        vhat = torch.tensor(unit_vec, dtype=dtype, device=device)
        d_final_np = np.asarray(d_final, dtype=np.float64)
        probe_alpha = 2.0
        frac = _STEER_FRACTION    # single-layer probe → full per-step fraction

        def _final_proj(text: str, inject: bool) -> float:
            ids = tok(text, return_tensors="pt", truncation=True, max_length=_MAX_TOKENS)
            ids = {k: v.to(device) for k, v in ids.items()}
            handle = None
            if inject:
                def hook(_m, _i, output):
                    h = output[0] if isinstance(output, tuple) else output
                    hn = h.norm(dim=-1, keepdim=True)
                    delta = (probe_alpha * frac) * hn * vhat
                    if isinstance(output, tuple):
                        return (h + delta,) + tuple(output[1:])
                    return h + delta
                handle = layers[layer_idx].register_forward_hook(hook)
            try:
                with torch.no_grad():
                    out = model(**ids, output_hidden_states=True)
                fin = out.hidden_states[-1][0].mean(dim=0).float().cpu().numpy()
            finally:
                if handle is not None:
                    handle.remove()
            return float(fin.astype(np.float64) @ d_final_np)

        shifts = [(_final_proj(t, True) - _final_proj(t, False)) for t in texts]
        return float(np.mean(shifts)) if shifts else 0.0

    def _ensure(self, axis: str) -> dict:
        key = f"{self._model_id()}|{axis}"
        if key in self._calib:
            return self._calib[key]
        path = self._cache_path(axis)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("model_id") == self._model_id():
                    self._calib[key] = data
                    return data
            except Exception:  # noqa: BLE001
                log.exception("steering cache read failed; recalibrating")
        return self.calibrate(axis)

    # ── apply / remove ───────────────────────────────────────────────
    @staticmethod
    def _clamp(alpha) -> float:
        try:
            a = float(alpha)
        except (TypeError, ValueError):
            a = 0.0
        return max(-_ALPHA_MAX, min(_ALPHA_MAX, a))

    def set_mix(self, mix: dict) -> dict:
        """Replace the active steering mix. ``{axis: alpha}``; alpha 0 / absent
        means off. Several axes inject simultaneously, their per-layer nudges
        summed and clamped to a global budget (``_COMBINED_MAX``)."""
        clean: dict[str, float] = {}
        for axis, alpha in (mix or {}).items():
            a = self._clamp(alpha)
            if a == 0.0:
                continue
            if axis not in AXES:
                raise ValueError(f"Axe inconnu : {axis}")
            clean[axis] = a
        self._mix = clean
        self._rebuild()
        st = self.state()
        warns = self._collinear_pairs(list(clean))
        if warns:
            st["collinear"] = warns
            log.warning("steering mix has collinear axes: %s", warns)
        return st

    def _collinear_pairs(self, axes: list) -> list:
        """Pairs of active axes whose steering vectors are near-collinear
        (|mean cosine| over shared layers ≥ ``_COLLINEAR_WARN``). The global
        magnitude cap bounds size, not direction — stacking redundant axes
        pushes the same direction thrice. This surfaces that."""
        out = []
        for i in range(len(axes)):
            for j in range(i + 1, len(axes)):
                try:
                    va = self._ensure(axes[i])["vectors"]
                    vb = self._ensure(axes[j])["vectors"]
                    c = V.axis_cosine(va, vb)
                except Exception:  # noqa: BLE001
                    c = None
                if c is not None and abs(c) >= _COLLINEAR_WARN:
                    out.append({"axes": [axes[i], axes[j]], "cosine": round(c, 3)})
        return out

    def engage(self, axis: str, alpha: float) -> dict:
        """Add/update/remove ONE axis in the current mix (additive)."""
        m = dict(self._mix)
        if self._clamp(alpha) == 0.0:
            m.pop(axis, None)
        else:
            m[axis] = self._clamp(alpha)
        return self.set_mix(m)

    def attach(self, axis: str, alpha: float) -> dict:
        """Back-compat single-axis API: replaces the whole mix with one axis
        (alpha 0 clears). The UI's one-axis-at-a-time behaviour is preserved."""
        if self._clamp(alpha) == 0.0:
            return self.set_mix({})
        return self.set_mix({axis: alpha})

    def set_alpha(self, alpha: float) -> None:
        """Back-compat: adjust the alpha when exactly one axis is active."""
        if len(self._mix) == 1:
            self.set_mix({next(iter(self._mix)): alpha})

    def _rebuild(self) -> None:
        """(Re)register one MERGED forward hook per layer touched by the mix.

        At each layer, the hook sums every active axis's contribution
        ``(alpha_axis * frac_axis) * v̂_axis`` (alphas read live from the mix),
        clamps the summed coefficient to ``_COMBINED_MAX``, then nudges by
        ``||h_token|| * coef`` — so the total perturbation stays a bounded
        fraction of the live signal no matter how many axes stack."""
        import torch

        for h in self._handles:
            try:
                h.remove()
            except Exception:  # noqa: BLE001
                pass
        self._handles = []
        if not self._mix:
            return

        model = self.wrapper.model
        layers = _decoder_layers(model)
        if layers is None:
            raise RuntimeError("Couches du décodeur introuvables pour le steering.")
        try:
            dtype = next(model.parameters()).dtype
            device = next(model.parameters()).device
        except StopIteration:
            dtype, device = torch.float32, "cpu"

        # layer index -> list of (axis, frac, unit_vector_tensor)
        per_layer: dict[int, list] = {}
        for axis in list(self._mix):
            data = self._ensure(axis)
            sel = data["layers"]
            frac = _STEER_FRACTION / max(1, len(sel))
            for li in sel:
                vhat = torch.tensor(data["vectors"][str(li)],
                                    dtype=dtype, device=device)
                per_layer.setdefault(int(li), []).append((axis, frac, vhat))

        cap = _COMBINED_MAX
        for li, contribs in per_layer.items():
            def make_hook(contribs):
                def hook(_module, _inp, output):
                    h = output[0] if isinstance(output, tuple) else output
                    coef = None
                    for axis, frac, vhat in contribs:
                        a = self._mix.get(axis, 0.0)     # live alpha
                        if a == 0.0:
                            continue
                        term = (a * frac) * vhat
                        coef = term if coef is None else coef + term
                    if coef is None:
                        return output
                    cn = coef.norm()
                    if cn > cap:                          # global budget
                        coef = coef * (cap / cn)
                    out = h + h.norm(dim=-1, keepdim=True) * coef
                    if torch.isnan(out).any() or torch.isinf(out).any():
                        return output                     # never emit garbage
                    if isinstance(output, tuple):
                        return (out,) + tuple(output[1:])
                    return out
                return hook
            self._handles.append(layers[li].register_forward_hook(make_hook(contribs)))

    def detach(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:  # noqa: BLE001
                pass
        self._handles = []
        self._mix = {}
