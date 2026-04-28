"""Curated allowlist of "consequential" config leaves tracked in the HPO graph.

We deliberately do *not* record every leaf in `SandboxConfig` — most are logging
windows, file paths, or downstream derivatives that don't change what the model
learns. Recording only the consequential axes keeps the graph small, fast to
diff, and meaningful as an HPO history.

Add new axes here whenever a new knob is exposed via CLI/YAML and is expected
to influence training outcome. Removing an axis is fine for new runs but
historical nodes keep whatever was there at write time.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

# Dotted-path leaves of `SandboxConfig` we record on every node.
CONSEQUENTIAL_AXES: Tuple[str, ...] = (
    # ── Data / regime ────────────────────────────────────────────────────────
    "data.regime",
    "data.region_stratified_loss",
    "data.context_length",
    # ── Model / heads ────────────────────────────────────────────────────────
    "model.signal_dist",
    "model.encode_input_transform",
    "model.n_cnn_layers",
    "model.expansion_factor",
    "model.n_transformer_layers",
    "model.nhead",
    "model.dropout",
    "model.separate_decoders",
    "model.mask_stem",
    # ── Training schedule / optimizer ────────────────────────────────────────
    "training.epochs",
    "training.batch_size",
    "training.optimizer.name",
    "training.optimizer.adam.lr",
    "training.optimizer.adamw.lr",
    "training.optimizer.adamax.lr",
    "training.optimizer.sgd.lr",
    "training.optimizer.adam.eps",
    "training.optimizer.adamw.eps",
    "training.optimizer.adamax.eps",
    "training.optimizer.adamw.weight_decay",
    "training.schedule.name",
    "training.schedule.warmup_frac",
    "training.schedule.min_lr_ratio",
    "training.grad.clip_norm",
    "training.grad.clip_mode",
    # ── Masking / DSF / augment ──────────────────────────────────────────────
    "training.masking.p_full_assay",
    "training.masking.p_full_loci",
    "training.masking.p_chunks",
    "training.masking.mask_fraction",
    "training.dsf.sampling",
    "training.augment_reverse_complement_prob",
    # ── Loss weights (head-isolation experiments) ────────────────────────────
    "training.loss_weights.count_weight",
    "training.loss_weights.pval_weight",
    "training.loss_weights.peak_weight",
    "training.loss_weights.obs_weight",
    "training.loss_weights.imp_weight",
)


def _get_dotted(d: Any, path: str) -> Any:
    """Walk a dict-of-dicts by dotted path; return None if any segment is missing."""
    cur: Any = d
    for seg in path.split("."):
        if isinstance(cur, dict) and seg in cur:
            cur = cur[seg]
        else:
            return None
    return cur


def extract_axes(resolved_cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the allowlisted axes from a fully-resolved config dict.

    Missing leaves are dropped (not stored as None) so older nodes stay readable
    when new axes are added.
    """
    out: Dict[str, Any] = {}
    for path in CONSEQUENTIAL_AXES:
        v = _get_dotted(resolved_cfg_dict, path)
        if v is not None:
            out[path] = v
    return out


def diff(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, List[Any]]:
    """Return ``{axis: [a_value, b_value]}`` for every leaf where a and b differ."""
    keys = set(a) | set(b)
    out: Dict[str, List[Any]] = {}
    for k in sorted(keys):
        va = a.get(k)
        vb = b.get(k)
        if va != vb:
            out[k] = [va, vb]
    return out


def axes_distance(a: Dict[str, Any], b: Dict[str, Any]) -> int:
    """Number of allowlisted leaves that differ between two axis dicts."""
    return len(diff(a, b))


__all__ = [
    "CONSEQUENTIAL_AXES",
    "extract_axes",
    "diff",
    "axes_distance",
]
