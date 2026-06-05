#!/usr/bin/env python3
"""Fixed autoresearch harness — DO NOT MODIFY (agent reads only).

Data loading, frozen v2 shell, training loop, 3-way eval, composite score, OOM guards.
"""
from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from sandbox.batch import make_masker, prepare_masked_batch
from sandbox.candi_v2.loss import build_v2_loss
from sandbox.candi_v2.model import CANDIv2
from sandbox.diagnostics.autoresearch import train as agent_train
from sandbox.diagnostics.meta_probes import run_probe_battery
from sandbox.diagnostics.real_data import build_real_v2_config, collect_batches, default_h5_path
from sandbox.diagnostics.real_overfit import _V2TupleWrapper, _imp_count_pearson
from sandbox.diagnostics.synthetic_overfit import nb_mean

# --- Fixed constants (agent cannot change) ---
MAX_STEPS = 1500
BATCH_SIZE = 4
N_TRAIN_BATCHES = 1  # pinned batch overfit (matches M02 diagnostic harness)
SEED = 42
MAX_PEAK_VRAM_MB = 9500.0
VRAM_BASELINE_MARGIN = 1.10
MAX_HEAD_PARAM_DELTA = 512
# Fixed masking: assay-only imputation (matches M02 / R17)
TRAIN_P_FULL_ASSAY = 1.0
TRAIN_P_FULL_LOCI = 0.0
TRAIN_P_CHUNKS = 0.0
EXPERIMENT_TIMEOUT_S = 900.0

SCORE_W_IMP = 0.35
SCORE_W_DCR = 0.25
SCORE_W_DENOISE = 0.20
SCORE_W_X_DEPTH = 0.10
SCORE_W_X_READLEN = 0.10

ARTIFACT_DIR = Path(__file__).resolve().parent
BASELINE_JSON = ARTIFACT_DIR / "baseline.json"


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def load_baseline() -> Dict[str, float]:
    if not BASELINE_JSON.exists():
        return {}
    try:
        return json.loads(BASELINE_JSON.read_text())
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}


def load_baseline_peak_vram() -> Optional[float]:
    v = load_baseline().get("peak_vram_mb")
    return float(v) if v is not None else None


def save_baseline(peak_vram_mb: float, composite_score: float, head_params: int) -> None:
    BASELINE_JSON.write_text(json.dumps({
        "peak_vram_mb": peak_vram_mb,
        "composite_score": composite_score,
        "head_params": head_params,
    }, indent=2))


def peak_vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)


def reset_vram_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def composite_score(metrics: Dict[str, float]) -> float:
    imp_p = metrics.get("imp_pearson", 0.0)
    if math.isnan(imp_p):
        imp_p = 0.0
    dcr = metrics.get("dcr_masked_bins", 0.0)
    if math.isnan(dcr):
        dcr = 0.0
    den_mae = metrics.get("denoise_rel_mae", 1.0)
    den_p = metrics.get("denoise_pearson", float("nan"))
    if not math.isnan(den_p):
        den_term = 1.0 - _clamp(den_p, 0.0, 1.0)
    else:
        den_term = _clamp(den_mae / 0.08, 0.0, 2.0) / 2.0
    x_d = metrics.get("x_depth_latent_delta", 0.0)
    if math.isnan(x_d):
        x_d = 0.0
    x_rl = metrics.get("x_readlen_latent_delta", 0.0)
    if math.isnan(x_rl):
        x_rl = 0.0

    return (
        SCORE_W_IMP * (1.0 - _clamp(imp_p, 0.0, 1.0))
        + SCORE_W_DCR * _clamp((4.0 - dcr) / 4.0, 0.0, 1.0)
        + SCORE_W_DENOISE * den_term
        + SCORE_W_X_DEPTH * _clamp((0.02 - x_d) / 0.02, 0.0, 1.0)
        + SCORE_W_X_READLEN * _clamp((0.02 - x_rl) / 0.02, 0.0, 1.0)
    )


def _rel_mae(mu: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    if not mask.any():
        return float("nan")
    pred = mu[mask].float()
    tgt = y[mask].float()
    denom = tgt.abs().clamp(min=1e-3)
    return float((pred - tgt).abs().div(denom).mean().item())


def _pearson(mu: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    if not mask.any():
        return float("nan")
    pred = mu[mask].flatten().float()
    tgt = y[mask].flatten().float()
    if pred.numel() < 2:
        return float("nan")
    pc = pred - pred.mean()
    tc = tgt - tgt.mean()
    denom = pc.norm() * tc.norm()
    if denom < 1e-12:
        return 0.0
    return float((pc @ tc / denom).item())


def build_shell_model(device: torch.device, tc: agent_train.TrainConfig) -> Tuple[CANDIv2, Any]:
    cfg = build_real_v2_config(dropout=0.0, heads="count_only", clip_norm=tc.clip_norm)
    cfg.training.loss_weights.obs_weight = tc.obs_weight
    cfg.training.loss_weights.imp_weight = tc.imp_weight
    cfg.training.loss_weights.count_weight = tc.count_weight
    torch.manual_seed(SEED)
    model = CANDIv2(cfg).to(device)
    agent_train.patch_count_head(model, tc)
    loss_fn = build_v2_loss(cfg)
    return model, loss_fn


def validate_head_params(model: CANDIv2, baseline_head_params: Optional[int]) -> Tuple[bool, str]:
    n = agent_train.count_head_param_count(model)
    if baseline_head_params is not None and n > baseline_head_params + MAX_HEAD_PARAM_DELTA:
        return False, f"head params {n} > baseline {baseline_head_params} + {MAX_HEAD_PARAM_DELTA}"
    return True, ""


def vram_preflight(
    model: CANDIv2,
    batch: Dict[str, torch.Tensor],
    masker,
    device: torch.device,
    baseline_peak: Optional[float],
) -> Tuple[bool, str]:
    reset_vram_stats()
    model.eval()
    try:
        prep = prepare_masked_batch(batch, masker, device)
        if prep is None:
            return False, "preflight: empty batch"
        with torch.no_grad():
            model.forward_tuple(
                prep["x_data"], prep["x_dna"], prep["x_meta"], prep["y_meta"],
            )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        return False, "preflight OOM"
    peak = peak_vram_mb()
    if peak > MAX_PEAK_VRAM_MB:
        return False, f"peak_vram {peak:.0f} > cap {MAX_PEAK_VRAM_MB:.0f}"
    if baseline_peak is not None and peak > baseline_peak * VRAM_BASELINE_MARGIN:
        return False, f"peak_vram {peak:.0f} > baseline {baseline_peak:.0f} * {VRAM_BASELINE_MARGIN}"
    return True, ""


def train_loop(
    model: CANDIv2,
    loss_fn,
    batches,
    masker,
    device: torch.device,
    tc: agent_train.TrainConfig,
) -> float:
    opt = agent_train.build_optimizer(model, tc)
    last_loss = float("nan")
    t_deadline = time.time() + EXPERIMENT_TIMEOUT_S

    for step in range(1, MAX_STEPS + 1):
        if time.time() > t_deadline:
            break
        batch = batches[(step - 1) % len(batches)]
        prep = prepare_masked_batch(batch, masker, device)
        if prep is None or not prep["masked_map"].any():
            continue
        model.train()
        opt.zero_grad(set_to_none=True)
        p, n, mu, var, df, peak = model.forward_tuple(
            prep["x_data"], prep["x_dna"], prep["x_meta"], prep["y_meta"],
        )
        loss, _, _ = loss_fn.forward_with_terms(
            p, n, mu, var, df, peak,
            prep["y_data"], prep["y_pval"], prep["y_peaks"],
            prep["observed_map"], prep["masked_map"],
            prep["signal_observed_map"], prep["signal_masked_map"],
            global_step=step,
            fallback_imp_to_observed_when_no_masked=False,
        )
        last_loss = float(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.clip_norm)
        opt.step()

    return last_loss


def evaluate_suite(
    model: CANDIv2,
    device: torch.device,
    h5_path: Path,
) -> Dict[str, float]:
    imp_masker = make_masker(p_full_assay=1.0, p_full_loci=0.0, preserve_assay_id=True)
    denoise_masker = make_masker(p_full_assay=0.0, p_full_loci=0.0, preserve_assay_id=True)
    probe = _V2TupleWrapper(model).to(device)

    imp_batches = collect_batches(h5_path, n_batches=1, batch_size=BATCH_SIZE, seed=SEED)
    den_batches = collect_batches(
        h5_path, n_batches=1, batch_size=BATCH_SIZE, seed=SEED + 1, dsf_sampling="uniform",
    )

    metrics: Dict[str, float] = {}

    # Imputation eval
    prep_imp = prepare_masked_batch(imp_batches[0], imp_masker, device)
    if prep_imp is not None and prep_imp["masked_map"].any():
        model.eval()
        with torch.no_grad():
            p, n, *_ = model.forward_tuple(
                prep_imp["x_data"], prep_imp["x_dna"], prep_imp["x_meta"], prep_imp["y_meta"],
            )
        metrics["imp_pearson"] = _imp_count_pearson(p, n, prep_imp["y_data"], prep_imp["masked_map"])
        probes = run_probe_battery(model, probe, prep_imp)
        metrics["dcr_masked_bins"] = probes.get("y_depth_dcr_on_masked_bins", float("nan"))
        metrics["y_depth_dcr_all"] = probes.get("y_depth_dcr_all", float("nan"))
        metrics["x_depth_latent_delta"] = probes.get("x_depth_latent_delta", float("nan"))
        metrics["x_readlen_latent_delta"] = probes.get("x_readlen_latent_delta", float("nan"))
        metrics["y_readlen_count_mse"] = probes.get("y_readlen_count_mse", float("nan"))

    # Denoising eval (DSF-corrupted input, full reconstruction)
    prep_den = prepare_masked_batch(den_batches[0], denoise_masker, device, apply_mask=False)
    if prep_den is not None:
        model.eval()
        with torch.no_grad():
            p, n, *_ = model.forward_tuple(
                prep_den["x_data"], prep_den["x_dna"], prep_den["x_meta"], prep_den["y_meta"],
            )
        mu = nb_mean(p, n)
        q = prep_den["query_mask"].unsqueeze(1).expand_as(mu)
        metrics["denoise_rel_mae"] = _rel_mae(mu, prep_den["y_data"], q)
        metrics["denoise_pearson"] = _pearson(mu, prep_den["y_data"], q)

    return metrics


def print_summary(result: Dict[str, Any]) -> None:
    print("---")
    for key in (
        "composite_score", "imp_pearson", "dcr_masked_bins", "denoise_rel_mae",
        "denoise_pearson", "x_depth_latent_delta", "x_readlen_latent_delta",
        "training_seconds", "peak_vram_mb", "peak_vram_ok", "num_steps",
        "num_params_M", "optimizer", "depth_center", "status", "device",
    ):
        val = result.get(key)
        if isinstance(val, float):
            print(f"{key + ':':18s}{val:.6f}")
        else:
            print(f"{key + ':':18s}{val}")
    print("---")


def run_experiment(device: Optional[torch.device] = None) -> Dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.92)

    tc = agent_train.get_config()
    h5_path = default_h5_path()
    baseline = load_baseline()
    baseline_peak = baseline.get("peak_vram_mb")
    baseline_peak = float(baseline_peak) if baseline_peak is not None else None
    baseline_head = baseline.get("head_params")
    baseline_head = int(baseline_head) if baseline_head is not None else None

    reset_vram_stats()
    t0 = time.time()
    status = "ok"
    err_msg = ""
    model = None

    try:
        model, loss_fn = build_shell_model(device, tc)
        head_n = agent_train.count_head_param_count(model)
        num_params = sum(p.numel() for p in model.parameters())

        ok, msg = validate_head_params(model, baseline_head)
        if not ok:
            status = "crash"
            err_msg = msg
            raise RuntimeError(msg)

        train_batches = collect_batches(
            h5_path, n_batches=N_TRAIN_BATCHES, batch_size=BATCH_SIZE,
            seed=SEED, dsf_sampling="uniform",
        )
        train_masker = make_masker(
            p_full_assay=TRAIN_P_FULL_ASSAY,
            p_full_loci=TRAIN_P_FULL_LOCI,
            p_chunks=TRAIN_P_CHUNKS,
            preserve_assay_id=True,
        )

        ok, msg = vram_preflight(
            model, train_batches[0], train_masker, device, baseline_peak,
        )
        if not ok:
            status = "crash"
            err_msg = msg
            raise RuntimeError(msg)

        reset_vram_stats()
        train_loss = train_loop(
            model, loss_fn, train_batches, train_masker, device, tc,
        )
        eval_metrics = evaluate_suite(model, device, h5_path)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        peak = peak_vram_mb()
        peak_ok = peak <= MAX_PEAK_VRAM_MB and (
            baseline_peak is None or peak <= baseline_peak * VRAM_BASELINE_MARGIN
        )

        score = composite_score(eval_metrics)
        if not peak_ok:
            status = "crash"
            score = 9.999

        result: Dict[str, Any] = {
            "composite_score": score,
            "training_loss": train_loss,
            "training_seconds": time.time() - t0,
            "peak_vram_mb": peak,
            "peak_vram_ok": peak_ok,
            "num_steps": MAX_STEPS,
            "num_params_M": num_params / 1e6,
            "optimizer": tc.optimizer,
            "depth_center": tc.depth_center,
            "use_depth_offset": tc.use_depth_offset,
            "head_params": head_n,
            "device": str(device),
            "status": status,
            **eval_metrics,
        }

        if baseline_peak is None and status == "ok" and peak_ok:
            save_baseline(peak, score, head_n)

    except torch.cuda.OutOfMemoryError:
        status = "crash"
        err_msg = "OOM during training"
        result = {
            "composite_score": 9.999,
            "training_seconds": time.time() - t0,
            "peak_vram_mb": peak_vram_mb(),
            "peak_vram_ok": False,
            "num_steps": MAX_STEPS,
            "depth_center": tc.depth_center,
            "status": "crash",
            "error": err_msg,
        }
    except Exception as exc:
        if status != "crash":
            status = "crash"
        result = {
            "composite_score": 9.999,
            "training_seconds": time.time() - t0,
            "peak_vram_mb": peak_vram_mb(),
            "peak_vram_ok": False,
            "num_steps": MAX_STEPS,
            "depth_center": tc.depth_center,
            "status": status,
            "error": err_msg or str(exc),
        }
    finally:
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print_summary(result)
    return result


def run_from_train() -> int:
    run_experiment()
    return 0


if __name__ == "__main__":
    raise SystemExit(run_from_train())
