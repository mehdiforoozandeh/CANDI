"""Sandbox JEPA trainer (encoder-only, E19 Stage 1).

Mirrors sandbox/train.py structure but trains a CANDIJepa instead of a full
CANDI model.  No decoders, no NLL heads — only:
  pred_loss + lambda_sigreg * sigreg_loss

Usage:
    python -m sandbox.train_jepa \\
        --config sandbox/configs/jepa_default.yaml \\
        --h5 sandbox/data/sandbox.h5 \\
        --run-name e19_jepa_stage1

All W&B metrics live under the ``lejepa/`` prefix.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import deque
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from sandbox import SANDBOX_ASSAYS
from sandbox.batch import CLOZE, make_masker
from sandbox.config import deep_merge, dump_yaml, load_yaml
from sandbox.config_types import (
    AdamHParams, AdamWHParams, AdamaxHParams,
    config_from_dict,
)
from sandbox.data import SandboxH5Dataset
from sandbox.chr21_umap import run_chr21_umap
from sandbox.jepa import CANDIJepa, compute_latent_geometry, compute_metadata_sensitivity
from sandbox.jepa_config import JEPAConfig, JEPATrainingConfig
from sandbox.jepa_model import JEPAModel, JEPAModelConfig as FreshJEPAModelConfig, JEPATransformerPredictor, MetadataEmbedding as FreshMetadataEmbedding
from sandbox.model import build_sandbox_candi
from _utils import reverse_complement_dna, reverse_signal


_DEFAULT_YAML = Path(__file__).resolve().parent / "configs" / "jepa_default.yaml"

T = TypeVar("T")


# ──────────────────────────────────────────────────────────────────────────────
# Batch preparation
# ──────────────────────────────────────────────────────────────────────────────

def prepare_jepa_batch(
    batch: Dict[str, torch.Tensor],
    masker,
    device: torch.device,
    *,
    target_dsf: str = "dsf1",
    mask_cond_type: str = "assay",
    preserve_assay_id: bool = False,
) -> Optional[Dict[str, torch.Tensor]]:
    """Build context and target views for one JEPA training step.

    Returns ``None`` if masking leaves no usable assays.

    mask_cond_type controls how the predictor's AdaLN is conditioned:
      "assay"       — assay-level mask; mask_cond: [B, F] binary (1=masked).
      "loci"        — loci-level mask; mask_cond: [B, L] binary (1=masked position).
                      Requires masker configured with p_full_loci=1, p_full_assay=0.
      "meta_concat" — no data masking; context=x_data(DSF≥1), target=y_data(DSF=1);
                      mask_cond: [B, 8*(F+1)] = flatten(meta_ctx ++ meta_tgt).
                      Requires dsf_list=[4] so context is always downsampled.
      "meta_tgt"    — task specification: applies assay masking as normal but conditions
                      predictor on flattened y_meta (DSF=1 target metadata) [B, 4*(F+1)].
                      Works for both assay imputation (masking) and DSF corruption (depth
                      difference between meta_ctx and meta_tgt drives has_corruption).
      "none"        — same as "assay" for batch construction; AdaLN ignored by model.

    has_corruption: True if the batch has a meaningful context-target difference.
    """
    x_data = batch["x_data"].to(device)         # [B, L, F]
    x_meta = batch["x_meta"].to(device)         # [B, 4, F]
    x_avail = batch["x_avail"].to(device)       # [B, F]
    x_dna = batch["x_dna"].to(device)
    y_data = batch["y_data"].to(device)         # [B, L, F]  DSF=1 target
    control_data = batch["control_data"].to(device)   # [B, L, 1]
    control_meta = batch["control_meta"].to(device)   # [B, 4, 1]

    B, L, F = x_data.shape

    # ── apply masking (skipped for meta_concat — corruption is DSF-only) ───
    x_data_m = x_data.clone()
    x_meta_m = x_meta.clone()
    x_avail_m = x_avail.clone()

    if mask_cond_type != "meta_concat":
        x_data_m, x_meta_m, x_avail_m = masker.apply_mask(x_data_m, x_meta_m, x_avail_m)
        # Skip if ALL assays are masked in any sample.
        if (x_avail_m == CLOZE).all(dim=1).any():
            return None

    # ── context ─────────────────────────────────────────────────────────────
    x_ctx = torch.cat([x_data_m, control_data], dim=2)     # [B, L, F+1]
    meta_ctx = torch.cat([x_meta_m, control_meta], dim=2)  # [B, 4, F+1]

    # ── target ──────────────────────────────────────────────────────────────
    if target_dsf == "dsf1" or mask_cond_type == "meta_concat":
        x_tgt = torch.cat([y_data, control_data], dim=2)   # [B, L, F+1]
    else:  # "same" — pre-masking, same DSF as context
        x_tgt = torch.cat([x_data, control_data], dim=2)   # [B, L, F+1]

    # meta_tgt: meta_concat and meta_tgt both use y_meta (DSF=1); others use unmasked x_meta.
    if mask_cond_type in ("meta_concat", "meta_tgt") and "y_meta" in batch:
        y_meta = batch["y_meta"].to(device)   # [B, 4, F]
        meta_tgt = torch.cat([y_meta, control_meta], dim=2)  # [B, 4, F+1]
    else:
        meta_tgt = torch.cat([x_meta, control_meta], dim=2)  # [B, 4, F+1]  unmasked

    if preserve_assay_id:
        if bool((meta_ctx[:, 1, :] < 0).any().item()):
            raise ValueError("assay_id must be present for all slots in meta_ctx")
        if bool((meta_tgt[:, 1, :] < 0).any().item()):
            raise ValueError("assay_id must be present for all slots in meta_tgt")

    # ── mask_cond and has_corruption ─────────────────────────────────────────
    if mask_cond_type == "loci":
        # Per-position binary mask at full L resolution (any assay masked at position).
        loci_mask = (x_data_m == CLOZE).any(dim=2).float()  # [B, L]
        mask_cond: torch.Tensor = loci_mask
        mask_indicator = torch.zeros(B, F, device=device)   # assay avail unchanged
        has_corruption = bool(loci_mask.any().item())
    elif mask_cond_type == "meta_concat":
        # Flatten and concatenate context and target metadata.
        mask_cond = torch.cat(
            [meta_ctx.reshape(B, -1), meta_tgt.reshape(B, -1)], dim=-1
        )  # [B, 8*(F+1)]
        mask_indicator = torch.zeros(B, F, device=device)
        has_corruption = True   # DSF difference is always present
    elif mask_cond_type == "meta_tgt":
        # Task specification: condition predictor on target metadata only (y_meta, DSF=1).
        # Corruption = assay masking (x_avail_m has CLOZE values) OR DSF difference
        # (x_meta_m depth_log2 < y_meta depth_log2 when context is downsampled).
        mask_indicator = (x_avail_m == CLOZE).float()   # [B, F]
        mask_cond = meta_tgt.reshape(B, -1)              # [B, 4*(F+1)]
        y_meta_for_check = batch["y_meta"].to(device) if "y_meta" in batch else x_meta
        # has_corruption: masked assays OR DSF difference between context and target metadata
        has_corruption = (
            bool(mask_indicator.any().item())
            or bool((x_meta_m != y_meta_for_check).any().item())
        )
    else:  # "assay" or "none"
        mask_indicator = (x_avail_m == CLOZE).float()   # [B, F]
        mask_cond = mask_indicator
        has_corruption = bool(mask_indicator.any().item())

    return {
        "x_ctx": x_ctx,
        "x_tgt": x_tgt,
        "x_dna": x_dna,
        "meta_ctx": meta_ctx,
        "meta_tgt": meta_tgt,
        "mask_cond": mask_cond,
        "mask_indicator": mask_indicator,  # always [B, F] for mask_frac logging
        "has_corruption": has_corruption,
    }


def _maybe_augment_rc(batch: Dict[str, Any], prob: float) -> Dict[str, Any]:
    """Reverse-complement augmentation (mirrors train.py)."""
    if prob <= 0.0 or float(torch.rand(1).item()) >= float(prob):
        return batch
    out = dict(batch)
    out["x_dna"] = reverse_complement_dna(out["x_dna"].clone())
    for k in ("x_data", "y_data", "control_data"):
        if k in out and isinstance(out[k], torch.Tensor):
            out[k] = reverse_signal(out[k].clone())
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Config helpers
# ──────────────────────────────────────────────────────────────────────────────

def _warn_inactive_optimizer_overrides(cfg: JEPAConfig) -> None:
    """Warn when a non-active optimizer's LR was overridden — a silent no-op.

    This catches the common mistake of setting e.g.
        training.optimizer.adamax.lr=5e-4
    when the active optimizer is adamw (jepa_default.yaml default).
    The override writes to a dead field and has zero effect on training.
    """
    active = cfg.training.optimizer.name
    inactive_defaults = {
        "adam":   AdamHParams().lr,
        "adamw":  AdamWHParams().lr,
        "adamax": AdamaxHParams().lr,
    }
    for name, default_lr in inactive_defaults.items():
        if name == active:
            continue
        actual_lr = getattr(cfg.training.optimizer, name).lr
        if actual_lr != default_lr:
            print(
                f"\n[WARN] training.optimizer.{name}.lr={actual_lr} was set, "
                f"but the active optimizer is '{active}' — this override has NO EFFECT.\n"
                f"       To change the learning rate use: "
                f"training.optimizer.{active}.lr={actual_lr}\n",
                file=sys.stderr,
            )


def _active_lr(cfg: JEPAConfig) -> float:
    on = cfg.training.optimizer.name
    m = {"adam": "adam", "adamw": "adamw", "adamax": "adamax", "sgd": "sgd"}
    return float(getattr(cfg.training.optimizer, m[on]).lr)


def build_optimizer(model: torch.nn.Module, cfg: JEPAConfig) -> torch.optim.Optimizer:
    name = cfg.training.optimizer.name
    if name == "adamw":
        h = cfg.training.optimizer.adamw
        return torch.optim.AdamW(
            model.parameters(), lr=h.lr, betas=(h.beta1, h.beta2),
            eps=h.eps, weight_decay=h.weight_decay,
        )
    if name == "adam":
        h = cfg.training.optimizer.adam
        return torch.optim.Adam(
            model.parameters(), lr=h.lr, betas=(h.beta1, h.beta2),
            eps=h.eps, weight_decay=h.weight_decay,
        )
    if name == "adamax":
        h = cfg.training.optimizer.adamax
        return torch.optim.Adamax(
            model.parameters(), lr=h.lr, betas=(h.beta1, h.beta2),
            eps=h.eps, weight_decay=h.weight_decay,
        )
    raise ValueError(f"unsupported optimizer: {name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: JEPAConfig,
    *,
    total_steps: int,
) -> Optional[Any]:
    sch = cfg.training.schedule
    base_lr = _active_lr(cfg)
    eta_min = base_lr * float(sch.min_lr_ratio)
    if sch.name == "constant" or total_steps <= 0:
        return None
    if sch.name == "cosine":
        wf = float(sch.warmup_frac)
        if wf <= 0.0 or total_steps <= 1:
            return CosineAnnealingLR(optimizer, T_max=max(1, total_steps), eta_min=eta_min)
        warmup_steps = min(max(1, int(round(total_steps * wf))), max(1, total_steps - 1))
        cosine_steps = max(1, total_steps - warmup_steps)
        return SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.2, end_factor=1.0, total_iters=warmup_steps),
                CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=eta_min),
            ],
            milestones=[warmup_steps],
        )
    return None


def load_jepa_config(
    *yaml_paths: Path,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> JEPAConfig:
    defaults = asdict(JEPAConfig())
    merged = defaults
    for p in yaml_paths:
        merged = deep_merge(merged, load_yaml(Path(p)))
    if cli_overrides:
        merged = deep_merge(merged, cli_overrides)
    return config_from_dict(JEPAConfig, merged)


def _maybe_wandb(cfg: JEPAConfig):
    if cfg.wandb.mode == "disabled":
        return None
    try:
        import wandb  # type: ignore
    except ImportError:
        print("wandb not installed; logging disabled", file=sys.stderr)
        return None
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.run_name,
        mode=cfg.wandb.mode,
        tags=list(cfg.wandb.tags),
        config=asdict(cfg),
    )
    return wandb


# ──────────────────────────────────────────────────────────────────────────────
# Gradient utilities
# ──────────────────────────────────────────────────────────────────────────────

def _module_grad_norm(module: torch.nn.Module) -> float:
    """L2 norm of all .grad tensors in a module (0.0 if none exist)."""
    sq = sum(
        p.grad.detach().float().pow(2).sum().item()
        for p in module.parameters()
        if p.grad is not None
    )
    return float(sq ** 0.5)


def _loss_grad_norm(
    loss_tensor: torch.Tensor,
    params: List[torch.nn.Parameter],
) -> float:
    """Grad norm of *loss_tensor* w.r.t. *params* via autograd.grad (retain_graph=True)."""
    try:
        grads = torch.autograd.grad(
            loss_tensor, params, retain_graph=True, allow_unused=True
        )
        sq = sum(
            g.detach().float().pow(2).sum()
            for g in grads if g is not None
        )
        return float(sq.sqrt().item()) if isinstance(sq, torch.Tensor) else 0.0
    except Exception:
        return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    dataset: SandboxH5Dataset,
    device: torch.device,
    masker,
    optimizer: torch.optim.Optimizer,
    cfg: JEPAConfig,
    *,
    scheduler=None,
    use_amp: bool = False,
    global_step: int = 0,
    wandb_run=None,
    max_batches: Optional[int] = None,
    geometry_log_every: int = 50,
    training_stats_jsonl_every: int = 200,
    on_snapshot=None,
    epoch_index: int = 0,
    state: Optional[Dict] = None,
) -> Tuple[int, Dict]:
    model.train()
    amp_on = bool(use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_on)
    clip_cap = float(cfg.training.grad.clip_norm)
    clip_mode = str(cfg.training.grad.clip_mode)
    clip_win = max(1, int(cfg.training.grad.clip_log_window))
    clipped_hist: deque[int] = deque(maxlen=clip_win)
    all_clipped: list[int] = []
    batches = 0
    target_dsf = str(cfg.jepa.target_dsf)
    # Batch-construction semantics (especially meta_tgt/y_meta routing) should
    # remain user-configurable and consistent across model types.
    mask_cond_type = str(cfg.jepa.pred_mask_cond_type)
    preserve_assay_id = bool(cfg.training.masking.preserve_assay_id)

    # ── cross-epoch state (initialised on first call, mutated in-place) ─────
    # Must match MASK_FRAC_THRESHOLD in extract_jepa_metrics.py (both = 0.05).
    _SPIKE_THRESH: float = 0.05
    if state is None:
        state = {}
    state.setdefault("cos_sim_best_filtered", float("inf"))
    state.setdefault("pred_loss_best_filtered", float("inf"))
    state.setdefault("enc_er_history", deque(maxlen=5))
    state.setdefault("pred_loss_history", deque(maxlen=8))
    state.setdefault("sigreg_history", deque(maxlen=8))
    state.setdefault("best_combined_loss", float("inf"))
    state.setdefault("best_combined_epoch", -1)
    state.setdefault("best_combined_step", -1)
    state.setdefault("best_state_dict", None)
    state.setdefault("best_state_dirty", False)

    # ── per-epoch accumulators ───────────────────────────────────────────────
    hi_mask_losses: List[float] = []
    lo_mask_losses: List[float] = []
    last_enc_er: Optional[float] = None
    last_cos_sim: Optional[float] = None
    last_adaLN_gamma: Optional[float] = None
    last_runtype_sens: Optional[float] = None
    has_spike_step: bool = False

    for raw_batch in dataset:
        raw_batch = _maybe_augment_rc(dict(raw_batch), 0.0)
        prep = prepare_jepa_batch(
            raw_batch, masker, device,
            target_dsf=target_dsf,
            mask_cond_type=mask_cond_type,
            preserve_assay_id=preserve_assay_id,
        )
        if prep is None:
            continue

        # Skip degenerate batches where context = target (no corruption occurred).
        # For assay/loci modes: skip if nothing was masked (FJ2 partial fix).
        # For meta_concat: never skip — DSF difference is always the corruption.
        if not prep["has_corruption"]:
            global_step += 1
            batches += 1
            continue

        with torch.amp.autocast("cuda", enabled=amp_on, dtype=torch.bfloat16 if amp_on else torch.float32):
            out = model(
                prep["x_ctx"],
                prep["x_tgt"],
                prep["x_dna"],
                prep["meta_ctx"],
                prep["meta_tgt"],
                prep["mask_cond"],
            )

        loss = out["loss"]
        if not torch.isfinite(loss):
            print(f"[jepa] non-finite loss at step {global_step}: {loss.item()}", file=sys.stderr)
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            batches += 1
            continue

        # ── per-loss grad norms (2 extra backward passes, geometry steps only) ──
        # Use (global_step + 1) so do_geo aligns with the post-increment value used by
        # the jsonl snapshot condition — both will fire at the same step numbers.
        do_geo = geometry_log_every <= 0 or ((global_step + 1) % geometry_log_every == 0)
        grad_pred_norm: float = 0.0
        grad_sig_norm: float = 0.0
        if do_geo:
            all_params = [p for p in model.parameters() if p.requires_grad]
            grad_pred_norm = _loss_grad_norm(out["pred_loss"], all_params)
            grad_sig_norm = _loss_grad_norm(out["sigreg_loss"], all_params)

        optimizer.zero_grad(set_to_none=True)
        if amp_on:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        # Gradient clipping (required: LeWM uses clip_val=1.0)
        if clip_mode == "norm" and clip_cap > 0.0:
            pre_clip_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_cap).item()
            )
            clipped = int(pre_clip_norm > clip_cap + 1e-12)
        else:
            pre_clip_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")).item()
            )
            clipped = 0

        clipped_hist.append(clipped)
        all_clipped.append(clipped)

        # ── per-module grad norms (cheap — read .grad after backward) ──────────
        enc_gnorm: float = 0.0
        proj_gnorm: float = 0.0
        pred_gnorm: float = 0.0
        if do_geo:
            enc_gnorm = _module_grad_norm(model.candi)
            proj_gnorm = _module_grad_norm(model.jepa_projector)
            pred_gnorm = _module_grad_norm(model.jepa_predictor)

        if amp_on:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        global_step += 1
        batches += 1

        # ── logging ─────────────────────────────────────────────────────────
        pred_loss_val = float(out["pred_loss"].detach().item())
        sigreg_val = float(out["sigreg_loss"].detach().item())
        total_val = float(out["loss"].detach().item())
        clipped_running = float(sum(all_clipped)) / float(len(all_clipped))
        mask_frac = float(prep["mask_indicator"].mean().item())
        cos_sim_val = float(out["cos_sim"].item())

        logd: Dict[str, float] = {
            "lejepa/pred_loss": pred_loss_val,
            "lejepa/sigreg_loss": sigreg_val,
            "lejepa/total_loss": total_val,
            "lejepa/combined_loss_scaled": float(out["combined_loss_scaled"].detach().item()),
            "lejepa/grad_norm_pre_clip": pre_clip_norm,
            "lejepa/grad_clipped": float(clipped),
            "lejepa/grad_clipped_frac_running": clipped_running,
            "lejepa/lr": float(optimizer.param_groups[0]["lr"]),
            "lejepa/step": float(global_step),
            # Metric 2: cosine similarity between ctx and tgt projections
            "lejepa/cos_sim_ctx_tgt": cos_sim_val,
            # Metric 6: fraction of assays masked per batch
            "lejepa/mask_frac": mask_frac,
        }

        # Split pred_loss by mask_frac; accumulate for epoch-level hi/lo ratio
        if mask_frac > 0.5:
            logd["lejepa/pred_loss_hi_mask"] = pred_loss_val
            hi_mask_losses.append(pred_loss_val)
        else:
            logd["lejepa/pred_loss_lo_mask"] = pred_loss_val

        # SIGReg / pred LOSS ratio in linear space.
        # Distinct from lejepa/sigreg_to_pred_ratio which is log(grad_sig/grad_pred).
        # Starts ~25–35 at init; converges to 1–5 in healthy runs.
        if pred_loss_val > 1e-8:
            logd["lejepa/sigreg_loss_pred_loss_ratio"] = sigreg_val / pred_loss_val

        # ── spike-filtered running best (FJ2 mitigation) ─────────────────────
        # Steps where mask_frac ≈ 0 are zero-mask artifact batches that inflate
        # cos_sim and deflate pred_loss; exclude them from the running best.
        _is_spike = mask_frac < _SPIKE_THRESH
        if _is_spike:
            has_spike_step = True
        else:
            if mask_frac <= 0.5:
                lo_mask_losses.append(pred_loss_val)
            if cos_sim_val < state["cos_sim_best_filtered"]:
                state["cos_sim_best_filtered"] = cos_sim_val
            if pred_loss_val < state["pred_loss_best_filtered"]:
                state["pred_loss_best_filtered"] = pred_loss_val
            last_cos_sim = cos_sim_val
        # Always log the running filtered best so W&B plots are spike-free.
        if state["cos_sim_best_filtered"] < float("inf"):
            logd["lejepa/cos_sim_best_filtered"] = state["cos_sim_best_filtered"]
        if state["pred_loss_best_filtered"] < float("inf"):
            logd["lejepa/pred_loss_best_filtered"] = state["pred_loss_best_filtered"]

        # ── best checkpoint tracking (spike-filtered) ─────────────────────
        if not _is_spike:
            _cls = logd["lejepa/combined_loss_scaled"]
            if _cls < state["best_combined_loss"]:
                state["best_combined_loss"] = _cls
                state["best_combined_epoch"] = epoch_index
                state["best_combined_step"] = global_step
                state["best_state_dict"] = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                state["best_state_dirty"] = True

        # Geometry metrics (every geometry_log_every steps)
        if do_geo:
            state["pred_loss_history"].append(pred_loss_val)
            state["sigreg_history"].append(sigreg_val)
            # Existing: per-loss gradient norms
            logd["lejepa/grad_pred"] = grad_pred_norm
            logd["lejepa/grad_sig"] = grad_sig_norm
            # New metric 6: SIGReg-to-pred gradient ratio (log scale)
            if grad_pred_norm > 1e-8:
                import math as _math
                logd["lejepa/sigreg_to_pred_ratio"] = _math.log(
                    grad_sig_norm / (grad_pred_norm + 1e-8)
                )
            # AdaLN modulation norms (zero if mask cond disabled)
            logd["lejepa/adaLN_gamma_norm"] = float(out["adaLN_gamma_norm"])
            logd["lejepa/adaLN_beta_norm"] = float(out["adaLN_beta_norm"])
            logd["lejepa/sigreg_projection_std"] = float(out["sigreg_projection_std"])
            # Per-module gradient norms
            logd["lejepa/enc_gnorm"] = enc_gnorm
            logd["lejepa/proj_gnorm"] = proj_gnorm
            logd["lejepa/pred_gnorm"] = pred_gnorm
            # Projector-output geometry (latent_eff_rank, latent_std_mean/min/max, n_dead)
            geo = compute_latent_geometry(out["proj_tgt"])
            logd.update(geo)
            # Keep embedding mean norm anchored to raw encoder output.
            logd["lejepa/embedding_mean_norm"] = float(out["embedding_mean_norm"].detach().item())
            # Eff-rank of raw encoder output (before projector)
            z_raw = out["z_tgt_raw"].detach().float().reshape(-1, out["z_tgt_raw"].shape[-1])
            Nr, Dr = z_raw.shape
            if Nr >= Dr:
                try:
                    _, Sr, _ = torch.linalg.svd(z_raw, full_matrices=False)
                    pr = Sr / (Sr.sum() + 1e-12)
                    logd["lejepa/encoder_eff_rank"] = float(
                        torch.exp(-(pr * (pr + 1e-12).log()).sum()).item()
                    )
                except Exception:
                    pass

            # Enc-ER rate of change: rolling slope over the last N geometry steps.
            # Negative values signal the onset of dimensional collapse.
            _er_val = logd.get("lejepa/encoder_eff_rank")
            if _er_val is not None:
                _er_f = float(_er_val)
                state["enc_er_history"].append(_er_f)
                last_enc_er = _er_f
                _er_hist = list(state["enc_er_history"])
                if len(_er_hist) >= 2:
                    logd["lejepa/enc_er_delta"] = (
                        (_er_hist[-1] - _er_hist[0]) / max(1, len(_er_hist) - 1)
                    )
            _pred_hist = list(state["pred_loss_history"])
            if len(_pred_hist) >= 2:
                deltas = [_pred_hist[i] - _pred_hist[i - 1] for i in range(1, len(_pred_hist))]
                inc = sum(1 for d in deltas if d > 0.0)
                dec = sum(1 for d in deltas if d < 0.0)
                logd["lejepa/pred_loss_slope"] = float((inc - dec) / max(1, len(deltas)))
            _sig_hist = list(state["sigreg_history"])
            if len(_sig_hist) >= 3:
                sig_std = float(torch.tensor(_sig_hist, dtype=torch.float32).std(unbiased=False).item())
                logd["lejepa/sigreg_converged"] = 1.0 if sig_std < 0.05 else 0.0

            # Metadata sensitivity: does the encoder respond to metadata field changes?
            # Measures 1 - cos_sim between embeddings when depth / read_length / run_type vary.
            try:
                meta_sens = compute_metadata_sensitivity(
                    model.candi,
                    prep["x_tgt"].detach(),
                    prep["x_dna"].detach(),
                    prep["meta_tgt"].detach(),
                )
                logd.update(meta_sens)
            except Exception:
                pass

            # ── epoch-level geometry trackers (for epoch summary record) ─────
            _ag = logd.get("lejepa/adaLN_gamma_norm")
            if _ag is not None:
                last_adaLN_gamma = float(_ag)
            _rt = logd.get("lejepa/meta_sens_runtype")
            if _rt is not None:
                last_runtype_sens = float(_rt)

        if wandb_run is not None:
            wandb_run.log(logd, step=global_step)

        # Snapshot to metrics.jsonl
        if (
            on_snapshot is not None
            and training_stats_jsonl_every > 0
            and (global_step % training_stats_jsonl_every == 0)
        ):
            on_snapshot(epoch_index, global_step, logd)

        if max_batches is not None and batches >= max_batches:
            break

    # ── epoch summary ────────────────────────────────────────────────────────
    hi_lo_ratio: Optional[float] = None
    if hi_mask_losses and lo_mask_losses:
        hi_mean = sum(hi_mask_losses) / len(hi_mask_losses)
        lo_mean = sum(lo_mask_losses) / len(lo_mask_losses)
        if lo_mean > 1e-8:
            hi_lo_ratio = hi_mean / lo_mean

    epoch_summary: Dict[str, Any] = {"is_spike_epoch": has_spike_step}
    if last_enc_er is not None:
        epoch_summary["lejepa/enc_er_last"] = last_enc_er
    if last_cos_sim is not None:
        epoch_summary["lejepa/cos_sim_last"] = last_cos_sim
    if last_adaLN_gamma is not None:
        epoch_summary["lejepa/adaLN_gamma_last"] = last_adaLN_gamma
    if last_runtype_sens is not None:
        epoch_summary["lejepa/runtype_sens_last"] = last_runtype_sens
    if hi_lo_ratio is not None:
        epoch_summary["lejepa/pred_loss_hi_lo_ratio"] = hi_lo_ratio
    if state["cos_sim_best_filtered"] < float("inf"):
        epoch_summary["lejepa/cos_sim_best_filtered"] = state["cos_sim_best_filtered"]
    if state["pred_loss_best_filtered"] < float("inf"):
        epoch_summary["lejepa/pred_loss_best_filtered"] = state["pred_loss_best_filtered"]

    return global_step, epoch_summary


# ──────────────────────────────────────────────────────────────────────────────
# CLI argument registration (mirrors train.py add_sandbox_cli_args pattern)
# ──────────────────────────────────────────────────────────────────────────────

def _dotted_overrides_from_argv(argv: List[str]) -> Dict[str, Any]:
    """Parse ``key=value`` positional dotted overrides (e.g. ``jepa.lambda_sigreg=0.05``)."""
    out: Dict[str, Any] = {}
    for tok in argv:
        if "=" not in tok or tok.startswith("-"):
            continue
        k, v = tok.split("=", 1)
        parts = k.split(".")
        node = out
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        # best-effort type coercion
        for conv in (int, float):
            try:
                v = conv(v)  # type: ignore[assignment]
                break
            except (ValueError, TypeError):
                pass
        if v == "true":
            v = True  # type: ignore[assignment]
        elif v == "false":
            v = False  # type: ignore[assignment]
        node[parts[-1]] = v
    return out


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="Sandbox CANDI JEPA trainer (E19 Stage 1)")
    p.add_argument("--config", type=Path, action="append", default=[],
                   help="YAML overlays (merged after jepa_default.yaml).")
    p.add_argument("--h5", type=Path, default=None, help="Override data.h5_path.")
    p.add_argument("--run-name", type=str, default=None, help="W&B run name.")
    p.add_argument("--no-wandb", action="store_true", help="Disable W&B logging.")
    p.add_argument("--wandb", action="store_true", help="Force W&B online mode.")
    p.add_argument("--dry-run", action="store_true", help="Validate config and exit.")
    p.add_argument("--print-config", action="store_true", help="Print resolved config and exit.")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--save-checkpoint", action="store_true")
    # Accept dotted overrides as positional args (e.g. jepa.lambda_sigreg=0.05)
    p.add_argument("dotted_overrides", nargs="*", default=[])
    args = p.parse_args(argv)

    # ── config resolution ───────────────────────────────────────────────────
    cfg_dict: Dict[str, Any] = asdict(JEPAConfig())
    if _DEFAULT_YAML.exists():
        cfg_dict = deep_merge(cfg_dict, load_yaml(_DEFAULT_YAML))
    for cpath in args.config:
        cfg_dict = deep_merge(cfg_dict, load_yaml(Path(cpath)))

    short: Dict[str, Any] = {}
    if args.h5 is not None:
        short.setdefault("data", {})["h5_path"] = str(args.h5)
    if args.epochs is not None:
        short.setdefault("training", {})["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        short.setdefault("training", {})["batch_size"] = int(args.batch_size)
    if args.device is not None:
        short.setdefault("training", {})["device"] = str(args.device)
    if args.seed is not None:
        short.setdefault("training", {})["seed"] = int(args.seed)
    if args.save_checkpoint:
        short.setdefault("training", {})["save_checkpoint"] = True
    if short:
        cfg_dict = deep_merge(cfg_dict, short)

    if args.no_wandb:
        cfg_dict.setdefault("wandb", {})["mode"] = "disabled"
    elif args.wandb:
        cfg_dict.setdefault("wandb", {})["mode"] = "online"
    if args.run_name is not None:
        cfg_dict.setdefault("wandb", {})["run_name"] = str(args.run_name)

    # Dotted overrides last
    dot_ov = _dotted_overrides_from_argv(args.dotted_overrides)
    if dot_ov:
        cfg_dict = deep_merge(cfg_dict, dot_ov)

    cfg = config_from_dict(JEPAConfig, cfg_dict)

    _warn_inactive_optimizer_overrides(cfg)

    if args.print_config or args.dry_run:
        print(json.dumps(asdict(cfg), indent=2, default=str))
        return 0

    h5p = Path(cfg.data.h5_path)
    if not h5p.exists():
        print(f"error: HDF5 not found: {h5p}", file=sys.stderr)
        return 2

    run_dir = Path(cfg.training.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(cfg, run_dir / "resolved_config.yaml")  # type: ignore[arg-type]
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text("")

    def _append_metrics(record: Dict[str, Any]) -> None:
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=float) + "\n")

    t_start = time.time()
    created_at_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start))

    # ── device + seed ───────────────────────────────────────────────────────
    dev = cfg.training.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(str(dev))
    torch.manual_seed(int(cfg.training.seed))
    random.seed(int(cfg.training.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(cfg.training.seed))

    wb = _maybe_wandb(cfg)

    # ── dataset ─────────────────────────────────────────────────────────────
    ds_train = SandboxH5Dataset(
        h5p,
        cfg.data.regime,
        train=True,
        batch_size=int(cfg.training.batch_size),
        biosample_prefix="T_",
        dsf_list=tuple(cfg.training.dsf.dsf_list),
        dsf_sampling=str(cfg.training.dsf.sampling),
        seed=int(cfg.training.seed),
        shuffle=True,
        eval_include_vb_ground_truth=False,
        imp_prefixes=("V_", "B_"),
        h5_cache_ram=bool(cfg.data.h5_cache_ram),
        ram_cache_max_bytes=int(cfg.data.ram_cache_max_bytes),
        preserve_assay_id=bool(cfg.training.masking.preserve_assay_id),
    )

    masker = make_masker(
        p_full_loci=float(cfg.training.masking.p_full_loci),
        p_full_assay=float(cfg.training.masking.p_full_assay),
        p_chunks=float(cfg.training.masking.p_chunks),
        mask_fraction=float(cfg.training.masking.mask_fraction),
        chunk_size=int(cfg.training.masking.chunk_size),
        min_available_frac=float(cfg.training.masking.min_available_frac),
        preserve_assay_id=bool(cfg.training.masking.preserve_assay_id),
    )

    # ── model ────────────────────────────────────────────────────────────────
    signal_dim = len(SANDBOX_ASSAYS)
    meta_dim = int(cfg.model.metadata_embedding_dim_mult) * signal_dim

    if str(cfg.model_type) == "fresh":
        fresh_cfg = FreshJEPAModelConfig(
            num_assays=signal_dim,
            context_length=int(cfg.data.context_length),
            metadata_embed_dim=int(cfg.fresh.metadata_embed_dim),
            n_cnn_layers=int(cfg.fresh.n_cnn_layers),
            expansion_factor=int(cfg.fresh.expansion_factor),
            conv_kernel_size=int(cfg.fresh.conv_kernel_size),
            pool_size=int(cfg.fresh.pool_size),
            dna_pool_size=int(cfg.fresh.dna_pool_size),
            n_transformer_layers=int(cfg.fresh.n_transformer_layers),
            nhead=int(cfg.fresh.nhead),
            dropout=float(cfg.fresh.dropout),
            d_model=int(cfg.fresh.d_model),
            proj_dim=int(cfg.fresh.proj_dim),
            proj_hidden_dim=int(cfg.fresh.proj_hidden_dim),
            pred_hidden_dim=int(cfg.fresh.pred_hidden_dim),
            pred_proj_hidden_dim=int(cfg.fresh.pred_proj_hidden_dim),
            predictor_layers=int(cfg.fresh.predictor_layers),
            predictor_heads=int(cfg.fresh.predictor_heads),
            predictor_dim_head=int(cfg.fresh.predictor_dim_head),
            predictor_ff_mult=int(cfg.fresh.predictor_ff_mult),
            predictor_type=str(cfg.fresh.predictor_type),
            cond_source=str(cfg.fresh.cond_source),
            cond_embed_shared=str(cfg.fresh.cond_embed_shared),
            lambda_sigreg=float(cfg.fresh.lambda_sigreg),
            sigreg_num_proj=int(cfg.fresh.sigreg_num_proj),
            sigreg_knots=int(cfg.fresh.sigreg_knots),
            signal_transform=str(cfg.model.encode_input_transform),
            meta_embed_layernorm=bool(cfg.fresh.meta_embed_layernorm),
            pred_meta_embed_layernorm=bool(cfg.fresh.pred_meta_embed_layernorm),
            missing_data_mode=str(cfg.fresh.missing_data_mode),
            fusion_mode=str(cfg.fresh.fusion_mode),
            fusion_norm=str(cfg.fresh.fusion_norm),
            film_mode=str(cfg.fresh.film_mode),
            conv_norm=str(cfg.fresh.conv_norm),
            dna_pool_order=str(cfg.fresh.dna_pool_order),
            transformer_type=str(cfg.fresh.transformer_type),
        )
        model = JEPAModel(fresh_cfg).to(device)
    else:
        candi = build_sandbox_candi(
            context_bins=int(cfg.data.context_length),
            signal_dim=signal_dim,
            metadata_embedding_dim=meta_dim,
            n_cnn_layers=int(cfg.model.n_cnn_layers),
            expansion_factor=int(cfg.model.expansion_factor),
            nhead=int(cfg.model.nhead),
            n_sab_layers=int(cfg.model.n_transformer_layers),
            dropout=float(cfg.model.dropout),
            separate_decoders=bool(cfg.model.separate_decoders),
            mask_stem=bool(cfg.model.mask_stem),
            dist_type="gaussian",
            signal_transform=str(cfg.model.encode_input_transform),
            linear_film=bool(cfg.model.linear_film),
            single_shot_decoder_film=bool(cfg.model.single_shot_decoder_film),
            gaussian_var_min=float(cfg.model.gaussian_var_min),
        )

        # Resolve encoder output dim early so we can build the fresh predictor if needed.
        _encoder_out_dim = int(candi.latent_projection[0].in_features)
        _proj_dim = int(cfg.jepa.proj_dim) if int(cfg.jepa.proj_dim) > 0 else _encoder_out_dim
        _pred_hidden_dim = int(cfg.jepa.pred_hidden_dim) if int(cfg.jepa.pred_hidden_dim) > 0 else _proj_dim

        # Optional: inject JEPATransformerPredictor (E21 ablation, jepa.predictor_type=fresh_transformer).
        _injected_predictor = None
        _pred_meta_embedding = None
        if str(cfg.jepa.predictor_type) == "fresh_transformer":
            if str(cfg.jepa.pred_cond_source) == "meta_tgt_embed":
                _pred_meta_embedding = FreshMetadataEmbedding(
                    num_assays=signal_dim,
                    embed_dim=int(cfg.jepa.pred_meta_embed_dim),
                    use_layernorm=bool(cfg.jepa.pred_meta_embed_layernorm),
                )
                _cond_dim = (signal_dim + 1) * int(cfg.jepa.pred_meta_embed_dim)
            else:
                _cond_dim = 4 * (signal_dim + 1)
            _injected_predictor = JEPATransformerPredictor(
                proj_dim=_proj_dim,
                hidden_dim=_pred_hidden_dim,
                cond_dim=_cond_dim,
                depth=int(cfg.jepa.predictor_layers),
                heads=int(cfg.jepa.predictor_heads),
                dim_head=int(cfg.jepa.predictor_dim_head),
                ff_mult=int(cfg.jepa.predictor_ff_mult),
                dropout=float(cfg.model.dropout),
            )
            print(
                f"[jepa] using fresh_transformer predictor: "
                f"depth={cfg.jepa.predictor_layers} heads={cfg.jepa.predictor_heads} "
                f"dim_head={cfg.jepa.predictor_dim_head} cond_dim={_cond_dim}"
                f"{' (embedded)' if _pred_meta_embedding else ' (raw)'}",
                file=sys.stderr,
            )

        model = CANDIJepa(
            candi,
            proj_dim=int(cfg.jepa.proj_dim),
            proj_hidden_dim=int(cfg.jepa.proj_hidden_dim),
            pred_hidden_dim=int(cfg.jepa.pred_hidden_dim),
            pred_proj_hidden_dim=int(cfg.jepa.pred_proj_hidden_dim),
            num_assays=signal_dim,
            use_mask_cond=bool(cfg.jepa.pred_use_mask_cond),
            pred_mask_cond_type=str(cfg.jepa.pred_mask_cond_type),
            lambda_sigreg=float(cfg.jepa.lambda_sigreg),
            sigreg_num_proj=int(cfg.jepa.sigreg_num_proj),
            sigreg_knots=int(cfg.jepa.sigreg_knots),
            target_dsf=str(cfg.jepa.target_dsf),
            predictor=_injected_predictor,
            pred_metadata_embedding=_pred_meta_embedding,
        ).to(device)

    n_enc = sum(p.numel() for p in model.candi.parameters())
    n_jepa = sum(p.numel() for p in model.parameters()) - n_enc
    print(
        f"[jepa] encoder params: {n_enc:,}  |  JEPA head params: {n_jepa:,}",
        file=sys.stderr,
    )

    cfg_steps = int(cfg.training.steps_per_epoch)
    steps_per_epoch = cfg_steps if cfg_steps > 0 else ds_train.estimate_steps_per_epoch()
    total_steps = max(1, int(cfg.training.epochs)) * steps_per_epoch

    opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg, total_steps=total_steps)

    def _emit_snapshot(ep_idx: int, gstep: int, logd: Dict[str, float]) -> None:
        _append_metrics({
            "kind": "training_step",
            "epoch": int(ep_idx),
            "global_step": int(gstep),
            "lejepa": {k: float(v) for k, v in logd.items()},
        })

    # ── training loop ───────────────────────────────────────────────────────
    global_step = 0
    jepa_state: Dict[str, Any] = {}   # mutable cross-epoch state passed to train_one_epoch
    _save_best = bool(cfg.training.save_best_checkpoint)
    _best_cooldown = int(cfg.training.best_checkpoint_cooldown_epochs)
    _last_best_write_epoch = -_best_cooldown  # allow immediate first write
    _best_ckpt_path = run_dir / "jepa_checkpoint_best.pt"

    def _write_best_checkpoint() -> None:
        nonlocal _last_best_write_epoch
        sd = jepa_state.get("best_state_dict")
        if sd is None:
            return
        torch.save({
            "model_state_dict": sd,
            "global_step": int(jepa_state["best_combined_step"]),
            "epoch": int(jepa_state["best_combined_epoch"]),
            "combined_loss_scaled": float(jepa_state["best_combined_loss"]),
            "jepa_proj_dim": model.proj_dim,
            "encoder_out_dim": model.encoder_out_dim,
        }, _best_ckpt_path)
        jepa_state["best_state_dirty"] = False
        _last_best_write_epoch = jepa_state["best_combined_epoch"]
        print(
            f"[jepa] best checkpoint saved: epoch={jepa_state['best_combined_epoch']}  "
            f"combined_loss_scaled={jepa_state['best_combined_loss']:.4f}  "
            f"path={_best_ckpt_path}",
            file=sys.stderr,
        )

    for ep in range(int(cfg.training.epochs)):
        ep_t0 = time.time()
        global_step, epoch_summary = train_one_epoch(
            model,
            ds_train,
            device,
            masker,
            opt,
            cfg,
            scheduler=sched,
            use_amp=bool(cfg.training.amp),
            global_step=global_step,
            wandb_run=wb,
            max_batches=cfg.training.max_train_batches,
            geometry_log_every=int(cfg.training.geometry_log_every_n_steps),
            training_stats_jsonl_every=int(cfg.training.training_stats_jsonl_every_n_steps),
            on_snapshot=_emit_snapshot,
            epoch_index=ep,
            state=jepa_state,
        )

        epoch_record: Dict[str, Any] = {
            "kind": "epoch",
            "epoch": ep,
            "global_step": int(global_step),
            "epoch_seconds": float(time.time() - ep_t0),
        }
        epoch_record.update(epoch_summary)
        _append_metrics(epoch_record)

        if wb is not None:
            wb_epoch_log: Dict[str, Any] = {"lejepa/epoch": ep}
            wb_epoch_log.update({k: v for k, v in epoch_summary.items() if isinstance(v, float)})
            wb.log(wb_epoch_log, step=global_step)

        print(
            f"[jepa] epoch {ep}/{cfg.training.epochs}  step={global_step}  "
            f"elapsed={time.time()-t_start:.0f}s",
            file=sys.stderr,
        )

        if _save_best and jepa_state.get("best_state_dirty"):
            if ep - _last_best_write_epoch >= _best_cooldown:
                _write_best_checkpoint()

    # ── flush unsaved best checkpoint at end of training ─────────────────────
    if _save_best and jepa_state.get("best_state_dirty"):
        _write_best_checkpoint()

    # ── chr21 UMAP diagnostic ────────────────────────────────────────────────
    print("[jepa] running chr21 UMAP...", file=sys.stderr)
    run_chr21_umap(model, h5p, cfg, device, run_dir, global_step, wandb_run=wb)

    # ── save checkpoint ─────────────────────────────────────────────────────
    if bool(cfg.training.save_checkpoint):
        ckpt = run_dir / "jepa_checkpoint_last.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "global_step": int(global_step),
            "jepa_proj_dim": model.proj_dim,
            "encoder_out_dim": model.encoder_out_dim,
        }, ckpt)
        print(f"[jepa] checkpoint saved to {ckpt}", file=sys.stderr)

    elapsed = float(time.time() - t_start)
    (run_dir / "elapsed.txt").write_text(f"{elapsed:.3f}\n")
    if wb is not None:
        wb.log({"lejepa/run_elapsed_seconds": elapsed}, step=global_step)
        wb.finish()

    # ── HPO graph ────────────────────────────────────────────────────────────
    if not bool(getattr(cfg.hpo, "disable", False)):
        try:
            from sandbox.hpo import update_graph_for_run
            update_graph_for_run(
                run_id=run_dir.name,
                run_dir=run_dir,
                resolved_cfg_dict=asdict(cfg),
                parent_run_ids=list(cfg.hpo.parent),
                experiment_label=str(cfg.hpo.experiment_label),
                notes=str(cfg.hpo.notes),
                elapsed_seconds=elapsed,
                slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                wandb_run_name=cfg.wandb.run_name,
                graph_path=Path(cfg.hpo.graph_path),
                created_at=created_at_iso,
            )
        except Exception as e:
            print(f"[jepa] HPO graph update failed: {e}", file=sys.stderr)

    print(
        f"[jepa] done  epochs={cfg.training.epochs}  step={global_step}  elapsed={elapsed:.1f}s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
