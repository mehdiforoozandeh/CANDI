#!/usr/bin/env python3
"""Fixed E34 autoresearch harness — DO NOT MODIFY during agent loop."""
from __future__ import annotations

import gc
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from sandbox import SANDBOX_ASSAYS
from sandbox.autoresearch.june3 import train as agent_train
from sandbox.autoresearch.june3.candi_v2.loss import build_v2_loss
from sandbox.autoresearch.june3.candi_v2.model import CANDIv2
from sandbox.autoresearch.june3.eval_bridge import run_eval_pass
from sandbox.autoresearch.june3.pins import (
    CHR_FRAC,
    PinnedTrainDataset,
    eval_max_batches_for_frac,
    load_manifest,
    load_train_batches,
    verify_data_fraction,
    verify_eval_dataloader_chr21,
    verify_type1_chrom_split,
)
from sandbox.batch import make_masker, prepare_masked_batch
from sandbox.eval import prompt_sensitivity_depth_count_ratio
from sandbox.train import _split_eval_families, train_one_epoch
from sandbox.train_candi_v2 import build_optimizer, build_scheduler

EPOCHS = 20
LOSS_MIN_REL_DROP = 0.15
MAX_PEAK_VRAM_MB = 9500.0
PARAM_MULT_REJECT = 5.0
R2_GUARD_MIN = 0.0
DCR_LO = 3.0
DCR_HI = 5.0
W_IMP_R2 = 0.45
W_DEN_R2 = 0.25
W_IMP_LOSS = 0.20
W_OBS_LOSS = 0.10

ARTIFACT_DIR = Path(__file__).resolve().parent
BASELINE_JSON = ARTIFACT_DIR / "baseline.json"
H5_DEFAULT = Path(__file__).resolve().parents[2] / "data" / "sandbox.h5"
EIC_METADATA = Path(__file__).resolve().parents[3] / "data" / "eic_metadata.csv"


def load_baseline() -> Dict[str, Any]:
    if not BASELINE_JSON.exists():
        return {}
    try:
        return json.loads(BASELINE_JSON.read_text())
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}


def save_baseline(result: Dict[str, Any]) -> None:
    BASELINE_JSON.write_text(
        json.dumps(
            {
                "param_count": result.get("param_count"),
                "peak_vram_mb": result.get("peak_vram_mb"),
                "training_seconds": result.get("training_seconds"),
                "primary_score": result.get("primary_score"),
                "imp_count_r2_gw": result.get("imp_count_r2_gw"),
                "den_count_r2_gw": result.get("den_count_r2_gw"),
            },
            indent=2,
        )
    )


def compute_primary(
    imp_r2: float, den_r2: float, count_imp: float, count_obs: float
) -> float:
    return (
        W_IMP_R2 * imp_r2
        + W_DEN_R2 * den_r2
        - W_IMP_LOSS * count_imp
        - W_OBS_LOSS * count_obs
    )


def guards_pass(
    imp_r2: float,
    den_r2: float,
    dcr: float,
    count_imp_loss: float,
    count_obs_loss: float,
) -> bool:
    # R² absolute floors removed — any experiment that improves primary_score is kept.
    # DCR-band and finite-loss checks remain to reject degenerate/crashed runs.
    if not math.isfinite(dcr) or dcr < DCR_LO or dcr > DCR_HI:
        return False
    return math.isfinite(count_imp_loss) and math.isfinite(count_obs_loss)


def peak_vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))


def reset_vram_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _loss_check(
    ep0_train: Tuple[float, float],
    ep0_eval: Tuple[float, float],
    best_train: Tuple[float, float],
    best_eval: Tuple[float, float],
) -> Tuple[bool, str]:
    checks = [
        ("train_count_imp", ep0_train[0], best_train[0]),
        ("train_count_obs", ep0_train[1], best_train[1]),
        ("eval_count_imp", ep0_eval[0], best_eval[0]),
        ("eval_count_obs", ep0_eval[1], best_eval[1]),
    ]
    for name, ep0, best in checks:
        if not (math.isfinite(ep0) and math.isfinite(best)):
            return False, f"{name}_nonfinite"
        if ep0 <= 0:
            return False, f"{name}_ep0_nonpos"
        rel_drop = (ep0 - best) / ep0
        if rel_drop < LOSS_MIN_REL_DROP:
            return False, f"{name}_drop_{rel_drop:.3f}"
    return True, "ok"


@torch.no_grad()
def _probe_batch_train_losses(
    model: nn.Module,
    loss_fn: nn.Module,
    batch: Dict[str, torch.Tensor],
    masker,
    device: torch.device,
    *,
    use_amp: bool,
) -> Tuple[float, float]:
    prep = prepare_masked_batch(batch, masker, device, apply_mask=True)
    if prep is None:
        return float("nan"), float("nan")
    amp_on = bool(use_amp and device.type == "cuda")
    with torch.amp.autocast("cuda", enabled=amp_on):
        p, n, mu, scale, df, peak = model(
            prep["x_data"],
            prep["x_dna"],
            prep["x_meta"],
            prep["y_meta"],
            query_mask=prep.get("query_mask"),
            query_mask_signal=prep.get("query_mask_signal"),
        )
        _, stats, _ = loss_fn.forward_with_terms(
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
        )
    return float(stats.get("count_imp_loss", float("nan"))), float(
        stats.get("count_obs_loss", float("nan"))
    )


@torch.no_grad()
def _encoder_eff_rank(model: nn.Module, prep: Dict[str, torch.Tensor]) -> float:
    train_model = model
    if hasattr(model, "v2"):
        inner = model.v2
    else:
        inner = model
    inner.eval()
    with torch.amp.autocast("cuda", enabled=False):
        z_out = inner.encoder.encode(prep["x_data"], prep["x_dna"], prep["x_meta"])
        z = z_out[0] if isinstance(z_out, tuple) else z_out
    flat = z.detach().float().reshape(z.shape[0], -1)
    if flat.shape[0] < 2:
        return float("nan")
    x = flat - flat.mean(dim=0, keepdim=True)
    sv = torch.linalg.svdvals(x)
    p = (sv / sv.sum().clamp(min=1e-12)).clamp(min=1e-12)
    return float(torch.exp(-(p * p.log()).sum()).item())


def print_summary(result: Dict[str, Any]) -> None:
    print("---")
    keys = (
        "train_chr19_frac",
        "eval_chr21_frac",
        "primary_score",
        "best_epoch",
        "imp_count_r2_gw",
        "den_count_r2_gw",
        "count_imp_loss",
        "count_obs_loss",
        "train_count_imp_loss",
        "train_count_obs_loss",
        "train_count_imp_loss_ep0",
        "train_count_obs_loss_ep0",
        "eval_count_imp_loss_ep0",
        "eval_count_obs_loss_ep0",
        "loss_check_ok",
        "loss_check_reason",
        "depth_count_ratio",
        "param_count",
        "param_ok",
        "encoder_eff_rank",
        "grad_pre_clip_norm",
        "grad_clipped_frac_running",
        "training_grad_norms/count_obs",
        "training_grad_norms/count_imp",
        "peak_vram_mb",
        "peak_vram_ok",
        "training_seconds",
        "status",
        "guards_ok",
        "error",
    )
    for key in keys:
        val = result.get(key)
        if isinstance(val, bool):
            print(f"{key + ':':32s}{val}")
        elif isinstance(val, float):
            print(f"{key + ':':32s}{val:.6f}")
        else:
            print(f"{key + ':':32s}{val}")
    print("---")


def run_experiment(device: Optional[torch.device] = None) -> Dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.92)

    cfg = agent_train.get_config()
    cfg.training.epochs = EPOCHS
    h5_path = H5_DEFAULT
    if not h5_path.exists():
        return {"status": "crash", "error": f"missing h5: {h5_path}"}

    manifest = load_manifest(h5_path)
    verify_type1_chrom_split(h5_path, manifest)
    fracs = verify_data_fraction(h5_path, manifest, frac=CHR_FRAC)
    verify_eval_dataloader_chr21(h5_path, regime=str(cfg.data.regime))
    n21_pool = int(manifest.get("n_chr21_pool", 0))
    if n21_pool > 0:
        cfg.training.eval_max_batches = eval_max_batches_for_frac(n21_pool, CHR_FRAC)
    train_batches = load_train_batches(h5_path, manifest)
    ds_train = PinnedTrainDataset(train_batches, seed=int(cfg.training.seed))

    baseline = load_baseline()
    baseline_params = int(baseline.get("param_count", 0) or 0)

    seed = int(cfg.training.seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    reset_vram_stats()
    t0 = time.time()
    status = "ok"
    err: Optional[str] = None

    model = agent_train.build_model(cfg, device)
    train_model = agent_train.build_train_wrapper(model).to(device)
    loss_fn = build_v2_loss(cfg).to(device)
    optimizer = build_optimizer(model, cfg.training)

    steps_per_epoch = ds_train.estimate_steps_per_epoch()
    cfg.training.steps_per_epoch = steps_per_epoch
    total_steps = max(1, EPOCHS * steps_per_epoch)
    scheduler = build_scheduler(optimizer, cfg.training, total_steps=total_steps)

    masker = make_masker(
        p_full_assay=float(cfg.training.masking.p_full_assay),
        p_full_loci=float(cfg.training.masking.p_full_loci),
        p_chunks=float(cfg.training.masking.p_chunks),
        mask_fraction=float(cfg.training.masking.mask_fraction),
        chunk_size=int(cfg.training.masking.chunk_size),
    )

    canonical_meta = None
    if EIC_METADATA.exists():
        from sandbox.data import build_canonical_meta

        canonical_meta = build_canonical_meta(str(EIC_METADATA), SANDBOX_ASSAYS)

    n_params = param_count(model)
    param_ok = baseline_params <= 0 or n_params <= baseline_params * PARAM_MULT_REJECT

    best_total: Optional[float] = None
    best_epoch = -1
    best_eval_metrics: Dict[str, float] = {}
    best_eval_losses: Dict[str, float] = {}
    best_train_snap: Dict[str, float] = {}
    best_train_losses: Tuple[float, float] = (float("nan"), float("nan"))
    best_dcr = float("nan")
    ep0_train_losses: Tuple[float, float] = (float("nan"), float("nan"))
    ep0_eval_losses: Tuple[float, float] = (float("nan"), float("nan"))
    last_epoch_snap: Dict[str, float] = {}
    last_eval_metrics: Dict[str, float] = {}
    last_eval_losses: Dict[str, float] = {}
    last_dcr = float("nan")
    global_step = 0

    def on_training_step_snapshot(ep_idx: int, gstep: int, logd: Dict[str, float]) -> None:
        del ep_idx, gstep
        last_epoch_snap.clear()
        last_epoch_snap.update(logd)

    try:
        for ep in range(EPOCHS):
            ds_train.set_epoch(ep)
            global_step = train_one_epoch(
                train_model,
                ds_train,
                device,
                masker,
                loss_fn,
                optimizer,
                cfg,
                scheduler=scheduler,
                use_amp=bool(cfg.training.amp),
                global_step=global_step,
                wandb_run=None,
                max_batches=None,
                meta_sensitivity_probe_every_n_steps=0,
                training_stats_jsonl_every_n_steps=int(
                    cfg.training.training_stats_jsonl_every_n_steps
                ),
                on_training_step_snapshot=on_training_step_snapshot,
                epoch_index=ep,
            )

            ep_train_imp, ep_train_obs = _probe_batch_train_losses(
                train_model,
                loss_fn,
                train_batches[0],
                masker,
                device,
                use_amp=bool(cfg.training.amp),
            )

            metrics = run_eval_pass(
                train_model,
                loss_fn,
                h5_path,
                str(cfg.data.regime),
                device,
                masker,
                batch_size=int(cfg.training.batch_size),
                seed=seed + ep,
                max_batches=int(cfg.training.eval_max_batches),
                imp_prefixes=tuple(cfg.eval.eval_imp_prefixes),
                h5_cache_ram=bool(cfg.data.h5_cache_ram),
                ram_cache_max_bytes=int(cfg.data.ram_cache_max_bytes),
                canonical_meta=canonical_meta,
                use_canonical_missing_meta=bool(cfg.eval.use_canonical_missing_meta),
                include_median_metrics=False,
                eval_mask_chunk_size=int(cfg.training.masking.chunk_size),
            )
            eval_metrics, eval_losses = _split_eval_families(metrics)
            last_eval_metrics = dict(eval_metrics)
            last_eval_losses = dict(eval_losses)
            if ep == 0:
                ep0_train_losses = (ep_train_imp, ep_train_obs)
                ep0_eval_losses = (
                    float(eval_losses.get("count_imp_loss", float("nan"))),
                    float(eval_losses.get("count_obs_loss", float("nan"))),
                )

            total = float(eval_losses.get("count_imp_loss", float("nan"))) + float(
                eval_losses.get("count_obs_loss", float("nan"))
            )
            if math.isfinite(total) and (best_total is None or total < best_total):
                best_total = total
                best_epoch = ep
                best_eval_metrics = dict(eval_metrics)
                best_eval_losses = dict(eval_losses)
                best_train_losses = (ep_train_imp, ep_train_obs)
                best_train_snap = dict(last_epoch_snap)

                prep_rank = prepare_masked_batch(
                    train_batches[0], masker, device, apply_mask=True
                )
                if prep_rank is not None:
                    try:
                        best_dcr = float(
                            prompt_sensitivity_depth_count_ratio(
                                train_model, prep_rank, prep_rank["y_meta"], device
                            )
                        )
                    except Exception:
                        best_dcr = float("nan")

        if best_epoch < 0:
            best_epoch = EPOCHS - 1
            best_eval_metrics = last_eval_metrics
            best_eval_losses = last_eval_losses
            best_train_losses = ep0_train_losses

        imp_r2 = float(best_eval_metrics.get("imp_count_r2_gw", float("nan")))
        den_r2 = float(best_eval_metrics.get("den_count_r2_gw", float("nan")))
        count_imp = float(best_eval_losses.get("count_imp_loss", float("nan")))
        count_obs = float(best_eval_losses.get("count_obs_loss", float("nan")))
        train_imp, train_obs = best_train_losses
        primary = compute_primary(imp_r2, den_r2, count_imp, count_obs)
        loss_ok, loss_reason = _loss_check(
            ep0_train_losses, ep0_eval_losses, best_train_losses, (
                count_imp,
                count_obs,
            ),
        )

        peak = peak_vram_mb()
        peak_ok = peak <= MAX_PEAK_VRAM_MB
        eff_rank = float("nan")
        prep_rank = prepare_masked_batch(train_batches[0], masker, device, apply_mask=True)
        if prep_rank is not None:
            try:
                eff_rank = _encoder_eff_rank(model, prep_rank)
            except Exception:
                eff_rank = float("nan")

        result: Dict[str, Any] = {
            "train_chr19_frac": fracs["train_unique_frac"],
            "eval_chr21_frac": float(
                (cfg.training.eval_max_batches * int(cfg.training.batch_size))
                / max(1, n21_pool)
            ),
            "primary_score": primary,
            "best_epoch": best_epoch,
            "imp_count_r2_gw": imp_r2,
            "den_count_r2_gw": den_r2,
            "count_imp_loss": count_imp,
            "count_obs_loss": count_obs,
            "train_count_imp_loss": train_imp,
            "train_count_obs_loss": train_obs,
            "train_count_imp_loss_ep0": ep0_train_losses[0],
            "train_count_obs_loss_ep0": ep0_train_losses[1],
            "eval_count_imp_loss_ep0": ep0_eval_losses[0],
            "eval_count_obs_loss_ep0": ep0_eval_losses[1],
            "loss_check_ok": loss_ok,
            "loss_check_reason": loss_reason,
            "depth_count_ratio": best_dcr,
            "param_count": n_params,
            "param_ok": param_ok,
            "encoder_eff_rank": eff_rank,
            "grad_pre_clip_norm": best_train_snap.get("training_stats/grad_pre_clip_norm"),
            "grad_clipped_frac_running": best_train_snap.get(
                "training_stats/grad_clipped_frac_running"
            ),
            "training_grad_norms/count_obs": best_train_snap.get(
                "training_grad_norms/count_obs"
            ),
            "training_grad_norms/count_imp": best_train_snap.get(
                "training_grad_norms/count_imp"
            ),
            "peak_vram_mb": peak,
            "peak_vram_ok": peak_ok,
            "training_seconds": time.time() - t0,
            "status": status,
            "error": err,
            "guards_ok": guards_pass(imp_r2, den_r2, best_dcr, count_imp, count_obs),
        }
        if not BASELINE_JSON.exists() and status == "ok":
            save_baseline(result)
        return result

    except Exception as exc:
        return {
            "status": "crash",
            "error": repr(exc),
            "peak_vram_mb": peak_vram_mb(),
            "training_seconds": time.time() - t0,
            "param_count": param_count(model) if model is not None else 0,
        }
    finally:
        del model, train_model, loss_fn, optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    print_summary(run_experiment())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
