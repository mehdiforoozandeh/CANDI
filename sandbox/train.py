"""Single-GPU sandbox trainer (HDF5 iterable dataset)."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from _utils import reverse_complement_dna, reverse_signal

from candi_loss import CANDI_LOSS
from sandbox.batch import make_masker, prepare_masked_batch
from sandbox.config import (
    add_sandbox_cli_args,
    deep_merge,
    dump_yaml,
    load_yaml,
    overrides_from_parsed_args,
)
from sandbox.config_types import SandboxConfig, config_from_dict
from sandbox.data import SandboxH5Dataset, build_canonical_meta
from sandbox.eval import (
    eval_batch_metrics,
    prompt_sensitivity_depth_count_ratio,
    prompt_sensitivity_readlen_mse,
    prompt_sensitivity_runtype_mse,
)
from sandbox.hpo import update_graph_for_run
from sandbox import SANDBOX_ASSAYS
from sandbox.losses import SandboxCompositeLoss
from sandbox.model import build_sandbox_candi

_DEFAULT_YAML = Path(__file__).resolve().parent / "configs" / "default.yaml"


def _maybe_wandb(cfg: SandboxConfig):
    if cfg.wandb.mode == "disabled":
        return None
    try:
        import wandb
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


def _dist_type_for_model(s: str) -> str:
    if s == "student_t":
        return "studentst"
    return s


def _active_base_lr(cfg: SandboxConfig) -> float:
    on = cfg.training.optimizer.name
    if on == "adam":
        return float(cfg.training.optimizer.adam.lr)
    if on == "adamw":
        return float(cfg.training.optimizer.adamw.lr)
    if on == "adamax":
        return float(cfg.training.optimizer.adamax.lr)
    if on == "sgd":
        return float(cfg.training.optimizer.sgd.lr)
    raise ValueError(on)


def _maybe_augment_reverse_complement(batch: Dict[str, Any], prob: float) -> Dict[str, Any]:
    """Match prod `train.py::_process_batch` RC aug (DNA + signal tracks)."""
    if prob <= 0.0 or float(torch.rand(1).item()) >= float(prob):
        return batch
    out = dict(batch)
    out["x_dna"] = reverse_complement_dna(out["x_dna"].clone())
    for k in ("x_data", "y_data", "y_pval", "y_peaks", "control_data"):
        if k in out and isinstance(out[k], torch.Tensor):
            out[k] = reverse_signal(out[k].clone())
    return out


def _build_mixed_meta(
    t_meta: torch.Tensor,
    vb_meta: torch.Tensor,
    masked_map: torch.Tensor,
) -> torch.Tensor:
    """Replace T_* y_meta with V_*/B_* metadata at cloze-masked assay slots.

    Args:
        t_meta:    [B, 4, F] T_* target metadata (current y_meta from prep)
        vb_meta:   [B, 4, F] V_*/B_* metadata from y_meta_imp
        masked_map:[B, L, F] bool — positions that are cloze-masked

    Returns:
        [B, 4, F] mixed metadata tensor where masked assay slots use V_*/B_* metadata.
    """
    masked_assays = masked_map.any(dim=1)  # [B, F] bool: True where the assay is cloze-masked
    mixed = t_meta.clone()
    mask_expanded = masked_assays.unsqueeze(1).expand_as(mixed)  # [B, 4, F]
    mixed[mask_expanded] = vb_meta[mask_expanded]
    return mixed


def _build_vb_natural_missing_meta(
    t_meta: torch.Tensor,
    vb_meta: torch.Tensor,
    y_avail: torch.Tensor,
    canonical_meta: Optional[torch.Tensor],
) -> torch.Tensor:
    """V/B natural metadata for assays missing in T (y_avail==0); canonical fallback.

    Used when ``use_canonical_missing_meta=False`` (E32 / CANDI v2 default): inject the
    paired V_*/B_* biosample's real covariates at imp-eval slots so depth_offset heads
    see the correct sequencing depth. Falls back to EIC canonical medians when V/B meta
    row 0 is invalid (-1).
    """
    device = t_meta.device
    mixed = t_meta.clone()
    missing = (y_avail.to(device) == 0).unsqueeze(1).expand_as(mixed)
    vb = vb_meta.to(device)
    valid_vb = (vb[:, 0:1, :] != -1.0).expand_as(mixed)
    use_vb = missing & valid_vb
    mixed[use_vb] = vb[use_vb]
    if canonical_meta is not None:
        can_exp = canonical_meta.to(device).unsqueeze(0).expand_as(mixed)
        still_missing = missing & (mixed[:, 0:1, :] == -1.0).expand_as(mixed)
        mixed[still_missing] = can_exp[still_missing]
    return mixed


_BRANCH_RAW_TO_LOG = {
    "count_obs_raw": "count_obs",
    "count_imp_raw": "count_imp",
    "pval_obs_raw": "pval_obs",
    "pval_imp_raw": "pval_imp",
    "peak_obs_raw": "peak_obs",
    "peak_imp_raw": "peak_imp",
}


def _global_grad_norm_for_loss(model: torch.nn.Module, loss: torch.Tensor) -> float:
    if not isinstance(loss, torch.Tensor) or (not loss.requires_grad):
        return float("nan")
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    sq_norm = None
    for g in grads:
        if g is None:
            continue
        g2 = g.detach().float().pow(2).sum()
        sq_norm = g2 if sq_norm is None else (sq_norm + g2)
    if sq_norm is None:
        return float("nan")
    return float(torch.sqrt(sq_norm).item())


def _max_abs_grad_value(model: torch.nn.Module) -> float:
    """Largest absolute gradient element across all trainable params."""
    out = 0.0
    for p in model.parameters():
        if (not p.requires_grad) or (p.grad is None) or (p.grad.numel() == 0):
            continue
        vmax = float(p.grad.detach().abs().max().item())
        if vmax > out:
            out = vmax
    return out


def _build_training_step_log(
    *,
    global_step: int,
    optimizer: torch.optim.Optimizer,
    pre_clip_grad_norm: float,
    clipped: int,
    clip_cap: float,
    clipped_running: Optional[float],
    clipped_window: Optional[float],
    loss_stats: Dict[str, float],
    loss_term_grad_norms: Dict[str, float],
) -> Dict[str, float]:
    logd: Dict[str, float] = {
        "training_stats/step": float(global_step),
        "training_stats/lr": float(optimizer.param_groups[0]["lr"]),
        "training_stats/total_loss": float(loss_stats["loss_total_weighted"]),
        "training_stats/grad_pre_clip_norm": float(pre_clip_grad_norm),
        "training_stats/grad_clipped": float(clipped),
    }
    if clip_cap > 0.0:
        logd["training_stats/grad_clip_cap"] = float(clip_cap)
    if clipped_running is not None:
        logd["training_stats/grad_clipped_frac_running"] = float(clipped_running)
    if clipped_window is not None:
        logd["training_stats/grad_clipped_frac_window"] = float(clipped_window)

    for raw_key, log_key in _BRANCH_RAW_TO_LOG.items():
        stats_key = f"loss_branch_{raw_key}"
        logd[f"training_losses/{log_key}"] = float(loss_stats[stats_key])
        grad_key = f"grad_norm_{raw_key}"
        if grad_key in loss_term_grad_norms:
            # Keep grad norms in a dedicated W&B family, separate from loss scalars.
            logd[f"training_grad_norms/{log_key}"] = float(loss_term_grad_norms[grad_key])

    return logd


def _split_eval_families(eval_values: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
    metrics: Dict[str, float] = {}
    losses: Dict[str, float] = {}
    for k, v in eval_values.items():
        if k.endswith("_loss"):
            losses[k] = float(v)
        else:
            metrics[k] = float(v)
    return metrics, losses


def _to_prefixed(prefix: str, values: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}/{k}": float(v) for k, v in values.items()}


def build_optimizer(model: torch.nn.Module, cfg: SandboxConfig) -> torch.optim.Optimizer:
    name = cfg.training.optimizer.name
    if name == "adam":
        h = cfg.training.optimizer.adam
        return torch.optim.Adam(
            model.parameters(),
            lr=h.lr,
            betas=(h.beta1, h.beta2),
            eps=h.eps,
            weight_decay=h.weight_decay,
            amsgrad=h.amsgrad,
        )
    if name == "adamw":
        h = cfg.training.optimizer.adamw
        return torch.optim.AdamW(
            model.parameters(),
            lr=h.lr,
            betas=(h.beta1, h.beta2),
            eps=h.eps,
            weight_decay=h.weight_decay,
            amsgrad=h.amsgrad,
        )
    if name == "adamax":
        h = cfg.training.optimizer.adamax
        return torch.optim.Adamax(
            model.parameters(),
            lr=h.lr,
            betas=(h.beta1, h.beta2),
            eps=h.eps,
            weight_decay=h.weight_decay,
        )
    if name == "sgd":
        h = cfg.training.optimizer.sgd
        return torch.optim.SGD(
            model.parameters(),
            lr=h.lr,
            momentum=h.momentum,
            dampening=h.dampening,
            weight_decay=h.weight_decay,
            nesterov=h.nesterov,
        )
    raise ValueError(f"unknown optimizer {name}")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: SandboxConfig, *, total_steps: int) -> Optional[Any]:
    """Per-step LR scheduler (matches prod _setup_cosine_scheduler). Caller steps once per optimizer step."""
    sch = cfg.training.schedule
    base_lr = _active_base_lr(cfg)
    eta_min = float(base_lr) * float(sch.min_lr_ratio)
    if sch.name == "constant" or total_steps <= 0:
        return None
    if sch.name == "linear":
        return LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=float(sch.min_lr_ratio),
            total_iters=total_steps,
        )
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


def train_one_epoch(
    model: torch.nn.Module,
    dataset: SandboxH5Dataset,
    device: torch.device,
    masker,
    criterion: SandboxCompositeLoss,
    optimizer: torch.optim.Optimizer,
    cfg: SandboxConfig,
    *,
    scheduler=None,
    use_amp: bool = False,
    global_step: int = 0,
    wandb_run=None,
    max_batches: Optional[int] = None,
    meta_sensitivity_probe_every_n_steps: int = 0,
    training_stats_jsonl_every_n_steps: int = 0,
    on_training_step_snapshot=None,
    epoch_index: int = 0,
) -> int:
    model.train()
    amp_on = bool(use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_on)
    batches = 0
    clip_cap = float(cfg.training.grad.clip_norm)
    clip_mode = str(cfg.training.grad.clip_mode)
    clip_win = max(1, int(cfg.training.grad.clip_log_window))
    clipped_hist: deque[int] = deque(maxlen=clip_win)
    rc_p = float(cfg.training.augment_reverse_complement_prob)
    all_clipped: list[int] = []

    for batch in dataset:
        batch = _maybe_augment_reverse_complement(dict(batch), rc_p)
        prep = prepare_masked_batch(batch, masker, device)
        if prep is None:
            continue
        with torch.amp.autocast("cuda", enabled=amp_on):
            p, n, mu, scale, df, peak = model(
                prep["x_data"],
                prep["x_dna"],
                prep["x_meta"],
                prep["y_meta"],
                query_mask=prep["query_mask"],
                query_mask_signal=prep["query_mask_signal"],
            )
            loss, stats, loss_terms = criterion.forward_with_terms(
                p,
                n,
                mu,
                scale,
                df,
                peak,
                prep["y_data"],
                prep["y_pval"],
                prep["y_peaks"],
                prep["observed_map"],
                prep["masked_map"],
                prep["signal_observed_map"],
                prep["signal_masked_map"],
                global_step=global_step,
                fallback_imp_to_observed_when_no_masked=False,
            )
        loss_term_grad_norms: Dict[str, float] = {}
        for raw_key in _BRANCH_RAW_TO_LOG:
            term_key = raw_key
            term = loss_terms.get(term_key)
            if term is None:
                continue
            if not torch.isfinite(term.detach()):
                loss_term_grad_norms[f"grad_norm_{raw_key}"] = float("nan")
                continue
            loss_term_grad_norms[f"grad_norm_{raw_key}"] = _global_grad_norm_for_loss(model, term)

        optimizer.zero_grad(set_to_none=True)
        if amp_on:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        if clip_mode == "norm":
            if clip_cap > 0.0:
                pre = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip_cap).item())
                clipped = int(pre > clip_cap + 1e-12)
            else:
                pre = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")).item())
                clipped = 0
        else:
            # For value clipping, keep logging pre-clip global norm and track whether any element
            # exceeded the cap before clipping.
            pre = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")).item())
            if clip_cap > 0.0:
                max_abs_pre = _max_abs_grad_value(model)
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_cap)
                clipped = int(max_abs_pre > clip_cap + 1e-12)
            else:
                clipped = 0
        clipped_hist.append(clipped)
        all_clipped.append(clipped)

        if amp_on:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # --- Scenario 3: Metadata sensitivity probe (no_grad, model.eval for this pass only) ---
        if meta_sensitivity_probe_every_n_steps > 0 and wandb_run is not None and global_step % meta_sensitivity_probe_every_n_steps == 0:
            model.eval()
            with torch.no_grad():
                _kw = dict(
                    query_mask=prep["query_mask"],
                    query_mask_signal=prep["query_mask_signal"],
                )
                out_norm = model(prep["x_data"], prep["x_dna"], prep["x_meta"], prep["y_meta"], **_kw)
                y_meta_null = torch.full_like(prep["y_meta"], -1.0)
                out_null = model(prep["x_data"], prep["x_dna"], prep["x_meta"], y_meta_null, **_kw)
            model.train()
            head_names = ["p", "n", "mu", "scale", "df", "peak"]
            sens_logs: Dict[str, Any] = {}
            for hi, hname in enumerate(head_names):
                o_n = out_norm[hi]
                o_z = out_null[hi]
                if o_n is None or o_z is None:
                    continue
                mse_val = float(((o_n.float() - o_z.float()) ** 2).mean().item())
                sens_logs[f"training_metadata_probes/meta_sensitivity_{hname}_mse"] = mse_val
            if sens_logs:
                sens_logs["training_metadata_probes/meta_sensitivity_mean_mse"] = float(sum(sens_logs.values()) / len(sens_logs))
                wandb_run.log(sens_logs, step=global_step)

        global_step += 1
        batches += 1
        clipped_running = float(sum(all_clipped)) / float(len(all_clipped)) if all_clipped else None
        clipped_window = float(sum(clipped_hist)) / float(len(clipped_hist)) if clipped_hist else None
        logd = _build_training_step_log(
            global_step=global_step,
            optimizer=optimizer,
            pre_clip_grad_norm=pre,
            clipped=clipped,
            clip_cap=clip_cap,
            clipped_running=clipped_running,
            clipped_window=clipped_window,
            loss_stats=stats,
            loss_term_grad_norms=loss_term_grad_norms,
        )
        if wandb_run is not None:
            wandb_run.log(logd, step=global_step)
        if (
            on_training_step_snapshot is not None
            and training_stats_jsonl_every_n_steps > 0
            and (global_step % training_stats_jsonl_every_n_steps == 0)
        ):
            on_training_step_snapshot(epoch_index, global_step, logd)
        if max_batches is not None and batches >= max_batches:
            break
    return global_step


@torch.no_grad()
def run_eval_pass(
    model: torch.nn.Module,
    criterion: SandboxCompositeLoss,
    h5_path: Path,
    regime: str,
    device: torch.device,
    masker,
    *,
    batch_size: int = 8,
    seed: int = 42,
    max_batches: int = 20,
    imp_prefixes: Tuple[str, ...] = ("V_", "B_"),
    h5_cache_ram: bool = True,
    ram_cache_max_bytes: int = 10 * 1024 * 1024 * 1024,
    canonical_meta: Optional[torch.Tensor] = None,
    use_canonical_missing_meta: bool = True,
) -> Dict[str, float]:
    """Single-pass eval with correctly prompted y_meta.

    y_meta construction per assay slot:
      - Observed (not cloze-masked, y_avail=1): T_* metadata  (always)
      - Cloze-masked (y_avail=1):               V_*/B_* natural metadata when
                                                 y_meta_imp is in the batch,
                                                 else T_* (no V_*/B_* biosample available)
      - Truly missing (y_avail=0):              canonical metadata from eic_metadata.csv
                                                 when use_canonical_missing_meta=True,
                                                 else -1 (MISSING token)
    """
    model.eval()
    _ = masker  # kept in signature for call-site compatibility; eval uses a dedicated no-mask masker.
    # Eval policy: do NOT cloze-mask T_* inputs. All available T_* assays/loci stay visible.
    eval_masker = make_masker(
        p_full_loci=0.0,
        p_full_assay=0.0,
        p_chunks=0.0,
        mask_fraction=0.0,
        chunk_size=40,
    )
    # shuffle=True with fixed seed so each epoch samples a diverse slice of
    # chr21 (peaks are sparse and concentrated in specific regions; sequential
    # batches from window index 0 often miss all positives -> AUROC=NaN).
    ds = SandboxH5Dataset(
        h5_path,
        regime,
        train=False,
        batch_size=batch_size,
        biosample_prefix="T_",
        dsf_sampling="off",
        seed=seed,
        shuffle=True,
        eval_include_vb_ground_truth=True,
        imp_prefixes=imp_prefixes,
        h5_cache_ram=h5_cache_ram,
        ram_cache_max_bytes=ram_cache_max_bytes,
    )
    metric_agg: Dict[str, list] = {}
    metric_keys_seen: set[str] = set()
    loss_agg: Dict[str, list] = {}
    # Pool (pred, target) vectors across batches for binary AUROC so that
    # rare positives in small eval batches don't yield per-batch NaNs (plan §9a).
    pool_den_peak_pred: list = []
    pool_den_peak_tgt: list = []
    pool_imp_peak_pred: list = []
    pool_imp_peak_tgt: list = []
    used = 0
    for i, ev_batch in enumerate(ds):
        if max_batches > 0 and i >= max_batches:
            break
        prep = prepare_masked_batch(ev_batch, eval_masker, device, apply_mask=False)
        if prep is None:
            continue

        # --- Build y_meta for the single forward pass ---
        # Start from T_* metadata (dataset always fills this for available assays).
        y_meta_fwd = prep["y_meta"].clone()
        query_mask_fwd = prep["query_mask"]

        # Natural metadata for cloze-masked assay slots: replace T_* with V_*/B_* where available.
        _ymi = ev_batch.get("y_meta_imp")
        if isinstance(_ymi, torch.Tensor):
            y_meta_fwd = _build_mixed_meta(y_meta_fwd, _ymi.to(device), prep["masked_map"])

        # Metadata for truly missing assay slots (y_avail=0) — these ARE the imp eval
        # targets (held-out unseen assays). The model must be conditioned on the
        # held-out assay's own covariates, else it imputes the wrong assay.
        y_avail_b = ev_batch.get("y_avail")
        if isinstance(y_avail_b, torch.Tensor):
            missing_mask = (y_avail_b.to(device) == 0)  # [B, F]
            if missing_mask.any():
                if use_canonical_missing_meta and canonical_meta is not None:
                    can_meta_d = canonical_meta.to(device)  # [4, F]
                    miss_exp = missing_mask.unsqueeze(1).expand_as(y_meta_fwd)
                    can_exp = can_meta_d.unsqueeze(0).expand_as(y_meta_fwd)
                    y_meta_fwd[miss_exp] = can_exp[miss_exp]
                elif not use_canonical_missing_meta and isinstance(_ymi, torch.Tensor):
                    # E32 / CANDI v2 default: inject the paired V_*/B_* biosample's real
                    # covariates at imp-eval slots so depth_offset heads see correct depth.
                    y_meta_fwd = _build_vb_natural_missing_meta(
                        y_meta_fwd,
                        _ymi.to(device),
                        y_avail_b.to(device),
                        canonical_meta,
                    )
                # Extend query_mask so FiLM conditioning applies to missing slots.
                query_mask_fwd = query_mask_fwd | missing_mask

        _rt = ev_batch.get("region_type")
        _ydi = ev_batch.get("y_data_imp")
        _ypi = ev_batch.get("y_pval_imp")
        _ypk = ev_batch.get("y_peaks_imp")
        _yav = ev_batch.get("y_avail")
        _ydsf = ev_batch.get("y_dsf")
        _rt_d = _rt.to(device) if isinstance(_rt, torch.Tensor) else _rt
        _ydi_d = _ydi.to(device) if isinstance(_ydi, torch.Tensor) else _ydi
        _ypi_d = _ypi.to(device) if isinstance(_ypi, torch.Tensor) else _ypi
        _ypk_d = _ypk.to(device) if isinstance(_ypk, torch.Tensor) else _ypk
        _yav_d = _yav.to(device) if isinstance(_yav, torch.Tensor) else _yav
        _ydsf_d = _ydsf.to(device) if isinstance(_ydsf, torch.Tensor) else _ydsf

        imp_eval_map = None
        imp_eval_signal_map = None
        if isinstance(_yav_d, torch.Tensor) and isinstance(_ypi_d, torch.Tensor):
            # Imputation eval positions: assays unavailable in T_* but available in V_*/B_* GT.
            t_unavail = (_yav_d <= 0).unsqueeze(1).expand_as(prep["masked_map"])
            vb_valid = (_ypi_d != -1)
            imp_eval_map = t_unavail & vb_valid
            if isinstance(_ydsf_d, torch.Tensor):
                dsf1 = (_ydsf_d == 1).unsqueeze(1).expand_as(prep["masked_map"])
                imp_eval_signal_map = imp_eval_map & dsf1
            else:
                imp_eval_signal_map = imp_eval_map

        _p, _n, mu, _sc, _df, peak = model(
            prep["x_data"],
            prep["x_dna"],
            prep["x_meta"],
            y_meta_fwd,
            query_mask=query_mask_fwd,
            query_mask_signal=prep["query_mask_signal"],
        )
        m = eval_batch_metrics(
            _p, _n, mu, peak,
            prep["y_data"], prep["y_pval"], prep["y_peaks"],
            prep["observed_map"], prep["masked_map"],
            prep["signal_observed_map"], prep["signal_masked_map"],
            regime=regime,
            region_type=_rt_d,
            imp_eval_map=imp_eval_map,
            imp_eval_signal_map=imp_eval_signal_map,
            y_data_imp=_ydi_d,
            y_pval_imp=_ypi_d,
            y_peaks_imp=_ypk_d,
        )
        for k, v in m.items():
            metric_keys_seen.add(k)
            if math.isfinite(v):
                metric_agg.setdefault(k, []).append(v)

        _, eval_loss_stats_den = criterion(
            _p,
            _n,
            mu,
            _sc,
            _df,
            peak,
            prep["y_data"],
            prep["y_pval"],
            prep["y_peaks"],
            prep["observed_map"],
            prep["masked_map"],
            prep["signal_observed_map"],
            prep["signal_masked_map"],
            global_step=0,
            fallback_imp_to_observed_when_no_masked=False,
        )
        eval_loss_stats_imp = None
        if (
            isinstance(_ydi_d, torch.Tensor)
            and isinstance(_ypi_d, torch.Tensor)
            and isinstance(_ypk_d, torch.Tensor)
            and isinstance(imp_eval_map, torch.Tensor)
        ):
            zeros_map = torch.zeros_like(imp_eval_map, dtype=torch.bool)
            imp_signal_map = (
                imp_eval_signal_map
                if isinstance(imp_eval_signal_map, torch.Tensor)
                else imp_eval_map
            )
            _, eval_loss_stats_imp = criterion(
                _p,
                _n,
                mu,
                _sc,
                _df,
                peak,
                _ydi_d,
                _ypi_d,
                _ypk_d,
                zeros_map,
                imp_eval_map,
                zeros_map,
                imp_signal_map,
                global_step=0,
                fallback_imp_to_observed_when_no_masked=False,
            )
        eval_losses = {
            "total_loss": (
                float(eval_loss_stats_den["loss_branch_count_obs_raw"])
                + float(eval_loss_stats_den["loss_branch_pval_obs_raw"])
                + float(eval_loss_stats_den["loss_branch_peak_obs_raw"])
                + (
                    float(eval_loss_stats_imp["loss_branch_count_imp_raw"])
                    + float(eval_loss_stats_imp["loss_branch_pval_imp_raw"])
                    + float(eval_loss_stats_imp["loss_branch_peak_imp_raw"])
                    if isinstance(eval_loss_stats_imp, dict)
                    else 0.0
                )
            ),
            "count_obs_loss": eval_loss_stats_den["loss_branch_count_obs_raw"],
            "count_imp_loss": (
                eval_loss_stats_imp["loss_branch_count_imp_raw"]
                if isinstance(eval_loss_stats_imp, dict)
                else float("nan")
            ),
            "pval_obs_loss": eval_loss_stats_den["loss_branch_pval_obs_raw"],
            "pval_imp_loss": (
                eval_loss_stats_imp["loss_branch_pval_imp_raw"]
                if isinstance(eval_loss_stats_imp, dict)
                else float("nan")
            ),
            "peak_obs_loss": eval_loss_stats_den["loss_branch_peak_obs_raw"],
            "peak_imp_loss": (
                eval_loss_stats_imp["loss_branch_peak_imp_raw"]
                if isinstance(eval_loss_stats_imp, dict)
                else float("nan")
            ),
        }
        for k, v in eval_losses.items():
            if math.isfinite(v):
                loss_agg.setdefault(k, []).append(float(v))

        # Accumulate for global peak AUROC.
        y_peaks = prep["y_peaks"]
        obs = prep["observed_map"]
        msk = prep["masked_map"]
        ok_d = (y_peaks == 0) | (y_peaks == 1)
        sel_d = (obs & ok_d).bool()
        if sel_d.any():
            pool_den_peak_pred.append(peak[sel_d].detach().float().cpu())
            pool_den_peak_tgt.append(y_peaks[sel_d].detach().float().cpu())
        # Only accumulate imp AUROC pool when real V_/B_* peak GT is available.
        if isinstance(_ypk_d, torch.Tensor):
            ok_i = (_ypk_d == 0) | (_ypk_d == 1)
            if imp_eval_map is not None:
                sel_i = (imp_eval_map & ok_i).bool()
            else:
                sel_i = torch.zeros_like(msk, dtype=torch.bool)
            if sel_i.any():
                pool_imp_peak_pred.append(peak[sel_i].detach().float().cpu())
                pool_imp_peak_tgt.append(_ypk_d[sel_i].detach().float().cpu())
        used += 1
    if used == 0:
        return {}
    out_metrics: Dict[str, float] = {}
    for k in metric_keys_seen:
        vals = metric_agg.get(k, [])
        out_metrics[k] = float(sum(vals) / len(vals)) if vals else float("nan")
    out_losses: Dict[str, float] = {k: float(sum(v) / len(v)) for k, v in loss_agg.items()}
    # Global peak AUROC (overrides per-batch average if pool is non-empty).
    def _global_auroc(pred_list, tgt_list) -> float:
        if not pred_list:
            return float("nan")
        p = torch.cat(pred_list).numpy()
        t = (torch.cat(tgt_list) > 0.5).long().numpy()
        if len(p) < 2 or t.min() == t.max():
            return float("nan")
        try:
            from sklearn.metrics import roc_auc_score
            return float(roc_auc_score(t, p))
        except Exception:
            return float("nan")

    gd = _global_auroc(pool_den_peak_pred, pool_den_peak_tgt)
    gi = _global_auroc(pool_imp_peak_pred, pool_imp_peak_tgt)
    if math.isfinite(gd):
        out_metrics["den_peak_auroc_gw"] = gd
    if math.isfinite(gi):
        out_metrics["imp_peak_auroc_gw"] = gi

    prefixed_metrics = _to_prefixed("eval_metrics", out_metrics)
    prefixed_losses = _to_prefixed("eval_losses", out_losses)
    merged = dict(prefixed_metrics)
    merged.update(prefixed_losses)
    return merged


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="Sandbox CANDI trainer")
    p.add_argument(
        "--config",
        type=Path,
        action="append",
        default=[],
        help="YAML overlays (merged after packaged default.yaml).",
    )
    p.add_argument("--print-config", action="store_true", help="Print resolved SandboxConfig and exit.")
    p.add_argument("--dry-run", action="store_true", help="Validate config and exit (no train).")
    p.add_argument("--h5", type=Path, default=None, help="Override data.h5_path.")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--run-name", type=str, default=None, help="W&B run name (wandb.run_name).")
    p.add_argument("--wandb", action="store_true", help="Force wandb online mode.")
    p.add_argument("--no-wandb", action="store_true", help="Disable wandb logging.")
    p.add_argument(
        "--save-checkpoint",
        action="store_true",
        help="Save final model checkpoint to <run_dir>/checkpoint_last.pt (default: off).",
    )
    p.add_argument(
        "--early-stop",
        action="store_true",
        help="Enable early stopping on rising eval_losses/total_loss (default: off).",
    )
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        help="Number of consecutive eval points where total_loss > best before stopping.",
    )
    add_sandbox_cli_args(p, SandboxConfig())
    args = p.parse_args(argv)

    cfg_dict: Dict[str, Any] = asdict(SandboxConfig())
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
    if args.lr is not None or args.weight_decay is not None:
        on = cfg_dict["training"]["optimizer"]["name"]
        short.setdefault("training", {}).setdefault("optimizer", {}).setdefault(str(on), {})
        if args.lr is not None:
            short["training"]["optimizer"][str(on)]["lr"] = float(args.lr)
        if args.weight_decay is not None:
            short["training"]["optimizer"][str(on)]["weight_decay"] = float(args.weight_decay)
    cfg_dict = deep_merge(cfg_dict, short)
    cfg_dict = deep_merge(cfg_dict, overrides_from_parsed_args(args))

    if args.no_wandb:
        cfg_dict.setdefault("wandb", {})["mode"] = "disabled"
    elif args.wandb:
        cfg_dict.setdefault("wandb", {})["mode"] = "online"
    if args.run_name is not None:
        cfg_dict.setdefault("wandb", {})["run_name"] = str(args.run_name)
    if args.save_checkpoint:
        cfg_dict.setdefault("training", {})["save_checkpoint"] = True
    if args.early_stop:
        cfg_dict.setdefault("training", {})["early_stop_enabled"] = True
    if args.early_stop_patience is not None:
        cfg_dict.setdefault("training", {})["early_stop_patience"] = int(args.early_stop_patience)

    cfg = config_from_dict(SandboxConfig, cfg_dict)

    if args.print_config or args.dry_run:
        print(json.dumps(asdict(cfg), indent=2, default=str))
        return 0

    h5p = Path(cfg.data.h5_path)
    if not h5p.exists():
        print(f"error: HDF5 not found: {h5p}", file=sys.stderr)
        return 2

    run_dir = Path(cfg.training.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(cfg, run_dir / "resolved_config.yaml")
    metrics_path = run_dir / "metrics.jsonl"
    # Truncate any stale metrics file from a previous run in the same dir.
    metrics_path.write_text("")

    def _append_metrics(record: Dict[str, Any]) -> None:
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=float) + "\n")

    t_start = time.time()
    created_at_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start))

    dev = cfg.training.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if str(dev).lower() == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(str(dev))
    torch.manual_seed(int(cfg.training.seed))
    random.seed(int(cfg.training.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(cfg.training.seed))

    wb = _maybe_wandb(cfg)

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
        imp_prefixes=tuple(cfg.eval.eval_imp_prefixes),
        h5_cache_ram=bool(cfg.data.h5_cache_ram),
        ram_cache_max_bytes=int(cfg.data.ram_cache_max_bytes),
    )

    # Load canonical per-assay metadata for Scenario 2 (truly missing assay imputation).
    _eic_path = str(cfg.eval.eic_metadata_path)
    canonical_meta: Optional[torch.Tensor] = None
    if Path(_eic_path).exists():
        from sandbox import SANDBOX_ASSAYS as _SASSAYS
        canonical_meta = build_canonical_meta(_eic_path, _SASSAYS)
        print(f"[sandbox] Loaded canonical metadata for {len(_SASSAYS)} assay types from {_eic_path}", file=sys.stderr)
    else:
        print(f"[sandbox] Warning: eic_metadata_path={_eic_path!r} not found; scenario 2 disabled.", file=sys.stderr)

    masker = make_masker(
        p_full_loci=float(cfg.training.masking.p_full_loci),
        p_full_assay=float(cfg.training.masking.p_full_assay),
        p_chunks=float(cfg.training.masking.p_chunks),
        mask_fraction=float(cfg.training.masking.mask_fraction),
        chunk_size=int(cfg.training.masking.chunk_size),
    )

    cand = CANDI_LOSS(
        dist_type=_dist_type_for_model(cfg.model.signal_dist),
        count_weight=cfg.training.loss_weights.count_weight,
        pval_weight=cfg.training.loss_weights.pval_weight,
        peak_weight=cfg.training.loss_weights.peak_weight,
        obs_weight=cfg.training.loss_weights.obs_weight,
        imp_weight=cfg.training.loss_weights.imp_weight,
    )
    crit = SandboxCompositeLoss(cand).to(device)

    meta_dim = int(cfg.model.metadata_embedding_dim_mult) * len(SANDBOX_ASSAYS)
    model = build_sandbox_candi(
        context_bins=int(cfg.data.context_length),
        signal_dim=len(SANDBOX_ASSAYS),
        metadata_embedding_dim=meta_dim,
        n_cnn_layers=int(cfg.model.n_cnn_layers),
        expansion_factor=int(cfg.model.expansion_factor),
        nhead=int(cfg.model.nhead),
        n_sab_layers=int(cfg.model.n_transformer_layers),
        dropout=float(cfg.model.dropout),
        separate_decoders=bool(cfg.model.separate_decoders),
        mask_stem=bool(cfg.model.mask_stem),
        dist_type=_dist_type_for_model(cfg.model.signal_dist),
        signal_transform=str(cfg.model.encode_input_transform),
    ).to(device)

    cfg_steps = int(cfg.training.steps_per_epoch)
    steps_per_epoch = cfg_steps if cfg_steps > 0 else ds_train.estimate_steps_per_epoch()
    total_steps = max(1, int(cfg.training.epochs)) * steps_per_epoch

    opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg, total_steps=total_steps)

    def _emit_training_step_snapshot(ep_idx: int, gstep: int, logd: Dict[str, float]) -> None:
        rec = {
            "kind": "training_step",
            "epoch": int(ep_idx),
            "global_step": int(gstep),
        }
        ts: Dict[str, float] = {}
        tl: Dict[str, float] = {}
        tg: Dict[str, float] = {}
        for k, v in logd.items():
            if k.startswith("training_stats/"):
                ts[k] = float(v)
            elif k.startswith("training_losses/"):
                tl[k] = float(v)
            elif k.startswith("training_grad_norms/"):
                tg[k] = float(v)
        if ts:
            rec["training_stats"] = ts
        if tl:
            rec["training_losses"] = tl
        if tg:
            rec["training_grad_norms"] = tg
        _append_metrics(rec)

    best_eval_total: Optional[float] = None
    early_stop_strikes = 0
    early_stop_triggered = False
    global_step = 0
    for ep in range(int(cfg.training.epochs)):
        ep_t0 = time.time()
        global_step = train_one_epoch(
            model,
            ds_train,
            device,
            masker,
            crit,
            opt,
            cfg,
            scheduler=sched,
            use_amp=bool(cfg.training.amp),
            global_step=global_step,
            wandb_run=wb,
            max_batches=cfg.training.max_train_batches,
            meta_sensitivity_probe_every_n_steps=int(cfg.eval.meta_sensitivity_probe_every_n_steps),
            training_stats_jsonl_every_n_steps=int(cfg.training.training_stats_jsonl_every_n_steps),
            on_training_step_snapshot=_emit_training_step_snapshot,
            epoch_index=ep,
        )
        if wb is not None:
            wb.log({"epoch": ep}, step=global_step)
        epoch_record: Dict[str, Any] = {
            "kind": "epoch",
            "epoch": ep,
            "global_step": int(global_step),
            "epoch_seconds": float(time.time() - ep_t0),
        }
        if cfg.training.eval_each_epoch and (ep + 1) % int(cfg.eval.eval_every_n_epochs) == 0:
            metrics = run_eval_pass(
                model,
                crit,
                h5p,
                str(cfg.data.regime),
                device,
                masker,
                batch_size=int(cfg.training.batch_size),
                seed=int(cfg.training.seed) + ep,
                max_batches=int(cfg.training.eval_max_batches),
                imp_prefixes=tuple(cfg.eval.eval_imp_prefixes),
                h5_cache_ram=bool(cfg.data.h5_cache_ram),
                ram_cache_max_bytes=int(cfg.data.ram_cache_max_bytes),
                canonical_meta=canonical_meta,
                use_canonical_missing_meta=bool(cfg.eval.use_canonical_missing_meta),
            )
            print(json.dumps({"epoch": ep, "eval_prefixed": metrics}, indent=2), file=sys.stderr)
            eval_metrics_dict, eval_losses_dict = _split_eval_families(metrics)
            if wb is not None:
                wb.log(metrics, step=global_step)
            epoch_record["eval_metrics"] = eval_metrics_dict
            epoch_record["eval_losses"] = eval_losses_dict
            ds_ev = SandboxH5Dataset(
                h5p,
                str(cfg.data.regime),
                train=False,
                batch_size=min(2, int(cfg.training.batch_size)),
                dsf_sampling="off",
                seed=int(cfg.training.seed),
                shuffle=False,
                eval_include_vb_ground_truth=True,
                imp_prefixes=tuple(cfg.eval.eval_imp_prefixes),
                h5_cache_ram=bool(cfg.data.h5_cache_ram),
                ram_cache_max_bytes=int(cfg.data.ram_cache_max_bytes),
            )
            batch0 = next(iter(ds_ev), None)
            if batch0 is not None:
                eval_masker_probe = make_masker(
                    p_full_loci=0.0,
                    p_full_assay=0.0,
                    p_chunks=0.0,
                    mask_fraction=0.0,
                    chunk_size=int(cfg.training.masking.chunk_size),
                )
                prep0 = prepare_masked_batch(batch0, eval_masker_probe, device, apply_mask=False)
                if prep0 is not None:
                    depth_ratio = prompt_sensitivity_depth_count_ratio(
                        model,
                        prep0,
                        prep0["y_meta"],
                        device,
                        depth_lo=float(cfg.eval.probe_depth_lo),
                        depth_hi=float(cfg.eval.probe_depth_hi),
                    )
                    runtype_mse = prompt_sensitivity_runtype_mse(model, prep0, prep0["y_meta"], device)
                    readlen_mse = prompt_sensitivity_readlen_mse(
                        model,
                        prep0,
                        prep0["y_meta"],
                        device,
                        readlen_a=float(cfg.eval.probe_read_length_lo),
                        readlen_b=float(cfg.eval.probe_read_length_hi),
                    )
                    pr = {
                        "training_metadata_probes/depth_count_ratio": depth_ratio,
                        "training_metadata_probes/runtype_mse": runtype_mse,
                        "training_metadata_probes/readlen_mse": readlen_mse,
                    }
                    if wb is not None:
                        wb.log(pr, step=global_step)
                    print(json.dumps({"epoch": ep, "training_metadata_probes": pr}, indent=2), file=sys.stderr)
                    epoch_record["training_metadata_probes"] = pr
        # Early-stop bookkeeping (only when an eval ran this epoch).
        if "eval_losses" in epoch_record:
            cur_total = epoch_record["eval_losses"].get("eval_losses/total_loss")
            if isinstance(cur_total, (int, float)) and math.isfinite(cur_total):
                if best_eval_total is None or cur_total < best_eval_total:
                    best_eval_total = float(cur_total)
                    early_stop_strikes = 0
                else:
                    early_stop_strikes += 1
                if (
                    bool(cfg.training.early_stop_enabled)
                    and early_stop_strikes >= int(cfg.training.early_stop_patience)
                ):
                    early_stop_triggered = True
                    epoch_record["early_stop_triggered"] = True
                    epoch_record["early_stop_best_total_loss"] = best_eval_total
                    epoch_record["early_stop_strikes"] = early_stop_strikes
        _append_metrics(epoch_record)
        if early_stop_triggered:
            print(
                f"[sandbox] early stopping at epoch {ep}: "
                f"strikes={early_stop_strikes} >= patience={int(cfg.training.early_stop_patience)}, "
                f"best_total_loss={best_eval_total:.4f}",
                file=sys.stderr,
            )
            break

    if bool(cfg.training.save_checkpoint):
        ckpt_path = run_dir / "checkpoint_last.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "global_step": int(global_step),
                "best_eval_total_loss": best_eval_total,
            },
            ckpt_path,
        )
        print(f"[sandbox] Saved checkpoint to {ckpt_path}", file=sys.stderr)

    elapsed = float(time.time() - t_start)
    (run_dir / "elapsed.txt").write_text(f"{elapsed:.3f}\n")
    if wb is not None:
        wb.log({"run/elapsed_seconds": elapsed}, step=global_step)
        wb.finish()

    # ── HPO graph upsert ────────────────────────────────────────────────────
    # Append-only graph of all runs (config axes + best-epoch results + parent
    # edges). Errors here never fail the run — we just log them.
    if not bool(getattr(cfg.hpo, "disable", False)):
        try:
            run_id = run_dir.name
            graph_path = Path(cfg.hpo.graph_path)
            if not graph_path.is_absolute():
                graph_path = Path.cwd() / graph_path
            update_graph_for_run(
                run_id=run_id,
                run_dir=run_dir,
                resolved_cfg_dict=asdict(cfg),
                parent_run_ids=list(cfg.hpo.parent),
                experiment_label=str(cfg.hpo.experiment_label),
                notes=str(cfg.hpo.notes),
                elapsed_seconds=elapsed,
                slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                wandb_run_name=cfg.wandb.run_name,
                graph_path=graph_path,
                created_at=created_at_iso,
            )
            print(f"[sandbox] HPO graph updated: {graph_path} (run_id={run_id})", file=sys.stderr)
        except Exception as e:  # pragma: no cover — best effort
            print(f"[sandbox] HPO graph update failed: {e}", file=sys.stderr)

    print(
        f"Finished {cfg.training.epochs} epoch(s), global_step={global_step}, elapsed={elapsed:.1f}s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
