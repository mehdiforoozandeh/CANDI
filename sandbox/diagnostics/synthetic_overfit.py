#!/usr/bin/env python3
"""Synthetic overfit harness for CANDI v2 count-head validation.

Runs phased overfit experiments on tiny synthetic data with rich diagnostics:
gradient norms, FiLM scale/shift stats, NB output monitoring, depth_count_ratio.

Usage:
    python -m sandbox.diagnostics.synthetic_overfit --phase p1
    python -m sandbox.diagnostics.synthetic_overfit --phase all
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from sandbox.candi_v2.config import CANDIv2Config, DecoderConfig, EncoderConfig
from sandbox.candi_v2.loss import build_v2_loss
from sandbox.candi_v2.model import CANDIv2
from sandbox.diagnostics.synthetic_data import (
    SyntheticDataConfig,
    generate_synthetic_batch,
    phase_config,
)
from sandbox.diagnostics.depth_offset_nb import apply_depth_offset_decoder


# Depth probe: +2 log2 units = 4× count scale (e.g. 5→20 or 10→40)
DEPTH_PROBE_LO = math.log2(5.0)
DEPTH_PROBE_HI = math.log2(20.0)

@dataclass
class PhaseCriteria:
    max_nll: float = 0.01
    min_pearson: float = 0.99
    max_rel_mae: float = 0.05  # |mu-y|/y mean; primary for NB count scale
    min_depth_ratio: float = 3.0
    max_imp_nll: float = 0.05
    min_imp_pearson: float = 0.95


PHASE_CRITERIA: Dict[str, PhaseCriteria] = {
    "p1": PhaseCriteria(max_nll=999.0, min_pearson=0.99, max_rel_mae=0.05, min_depth_ratio=0.0),
    "p2": PhaseCriteria(max_nll=999.0, min_pearson=0.985, max_rel_mae=0.05, min_depth_ratio=0.0),
    "p3": PhaseCriteria(max_nll=999.0, min_pearson=0.95, max_rel_mae=0.08, min_depth_ratio=3.0,
                        max_imp_nll=999.0, min_imp_pearson=0.95),
    "p4": PhaseCriteria(max_nll=999.0, min_pearson=0.95, max_rel_mae=0.08, min_depth_ratio=3.0,
                        max_imp_nll=999.0, min_imp_pearson=0.95),
    "p5": PhaseCriteria(max_nll=999.0, min_pearson=0.95, max_rel_mae=0.08, min_depth_ratio=3.0,
                        max_imp_nll=999.0, min_imp_pearson=0.95),
}


# ---------------------------------------------------------------------------
# Model config for fast overfit (~50K params)
# ---------------------------------------------------------------------------

def build_overfit_model_config(num_assays: int = 4, context_length: int = 96) -> CANDIv2Config:
    enc = EncoderConfig(
        num_assays=num_assays,
        context_length=context_length,
        metadata_embed_dim=32,
        n_cnn_layers=3,
        expansion_factor=2,
        n_transformer_layers=2,
        nhead=4,
        dropout=0.0,
        signal_transform="log1p",
        film_mode="per_conv_and_transformer",
        missing_data_mode="mask_token",
        fusion_norm="none",
    )
    dec = DecoderConfig(
        heads="count_only",
        film_mode="single_pre_decoder",
        meta_embed_dim=32,
        n_cnn_layers=3,
    )
    cfg = CANDIv2Config(encoder=enc, decoder=dec)
    cfg.training.epochs = 500
    cfg.training.batch_size = 4
    cfg.training.optimizer.adamax.lr = 1e-2
    cfg.training.grad.clip_norm = 10.0
    cfg.training.loss_weights.count_weight = 1.0
    cfg.training.loss_weights.peak_weight = 0.0
    cfg.training.loss_weights.pval_weight = 0.0
    cfg.training.loss_weights.obs_weight = 1.0
    cfg.training.loss_weights.imp_weight = 1.0
    return cfg


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

class GradientMonitor:
    """Register backward hooks to track per-module gradient norms."""

    MODULE_KEYS = (
        "encoder.metadata_embedding",
        "encoder.signal_tower",
        "encoder.transformer_blocks",
        "decoder.decoder_meta_embedding",
        "decoder.pre_decoder_film",
        "decoder.neg_binom_layer",
    )

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.handles: List[Any] = []
        self.grad_norms: Dict[str, float] = {}

    def _make_hook(self, name: str):
        def hook(_module, _grad_input, grad_output):
            if grad_output[0] is not None:
                self.grad_norms[name] = float(grad_output[0].detach().norm().item())
        return hook

    def register(self) -> None:
        self.clear()
        for key in self.MODULE_KEYS:
            parts = key.split(".")
            mod = self.model
            for p in parts:
                mod = getattr(mod, p, None)
                if mod is None:
                    break
            if mod is None:
                continue
            self.handles.append(mod.register_full_backward_hook(self._make_hook(key)))

    def clear(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.grad_norms = {}

    def collect_param_norms(self) -> Dict[str, float]:
        """Post-backward param grad norms for finer-grained modules."""
        out: Dict[str, float] = {}
        groups = {
            "enc_meta": "encoder.metadata_embedding",
            "enc_film": "encoder.signal_tower.per_conv_film_layers",
            "dec_film": "decoder.pre_decoder_film",
            "dec_meta": "decoder.decoder_meta_embedding",
            "nb_head": "decoder.neg_binom_layer",
        }
        for label, prefix in groups.items():
            sq = 0.0
            for name, p in self.model.named_parameters():
                if name.startswith(prefix) and p.grad is not None:
                    sq += float(p.grad.detach().norm().item() ** 2)
            out[f"grad_param_{label}"] = math.sqrt(sq) if sq > 0 else 0.0
        return out


def nb_mean(p: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return n * (1.0 - p) / (p + 1e-9)


def compute_pearson(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    if not mask.any():
        return float("nan")
    p = pred[mask].flatten().float()
    t = target[mask].flatten().float()
    if p.numel() < 2:
        return float("nan")
    p_c = p - p.mean()
    t_c = t - t.mean()
    denom = p_c.norm() * t_c.norm()
    if denom < 1e-12:
        return 0.0
    return float((p_c @ t_c / denom).item())


def compute_diagnostics(
    model: CANDIv2,
    batch: Dict[str, torch.Tensor],
    out: Dict[str, torch.Tensor],
    stats: Dict[str, float],
    grad_monitor: GradientMonitor,
) -> Dict[str, float]:
    """Aggregate high-signal diagnostic metrics."""
    p, n = out["p"], out["n"]
    mu = nb_mean(p, n)
    y = batch["y_data"]
    obs = batch["observed_map"]
    msk = batch["masked_map"]

    diag: Dict[str, float] = {
        "count_nll": stats.get("loss_branch_count_obs_raw", float("nan"))
        + stats.get("loss_branch_count_imp_raw", 0.0),
        "count_obs_nll": stats.get("loss_branch_count_obs_raw", float("nan")),
        "count_imp_nll": stats.get("loss_branch_count_imp_raw", float("nan")),
        "pearson_all": compute_pearson(mu, y, obs | msk),
        "pearson_obs": compute_pearson(mu, y, obs),
        "pearson_imp": compute_pearson(mu, y, msk) if msk.any() else float("nan"),
        "mu_mean": float(mu.detach().mean().item()),
        "mu_std": float(mu.detach().std().item()),
        "n_mean": float(n.detach().mean().item()),
        "p_mean": float(p.detach().mean().item()),
    }
    mask_all = obs | msk
    if mask_all.any():
        err = (mu - y).abs()[mask_all]
        denom = y[mask_all].clamp(min=1.0)
        diag["rel_mae"] = float((err / denom).mean().item())
        diag["mae"] = float(err.mean().item())
    else:
        diag["rel_mae"] = float("nan")
        diag["mae"] = float("nan")

    # Per-sample depth scaling of predicted mu
    depths = batch["depth_log2"]
    b = depths.shape[0]
    sample_mu = []
    for bi in range(b):
        m = (obs | msk)[bi]
        if m.any():
            sample_mu.append(float(mu[bi][m].mean().item()))
        else:
            sample_mu.append(float("nan"))
    if len(sample_mu) >= 2 and not math.isnan(sample_mu[0]) and sample_mu[0] > 0:
        diag["depth_count_ratio_samples"] = sample_mu[-1] / sample_mu[0]
    else:
        diag["depth_count_ratio_samples"] = float("nan")

    diag.update(grad_monitor.grad_norms)
    diag.update(grad_monitor.collect_param_norms())

    # Dead / exploding gradient flags
    for k, v in list(diag.items()):
        if k.startswith("grad_") or k.startswith("encoder.") or k.startswith("decoder."):
            if v < 1e-8 and v > 0:
                diag[f"{k}_DEAD"] = 1.0
            if v > 100:
                diag[f"{k}_EXPLODE"] = 1.0

    return diag


def depth_count_ratio_probe(
    model: CANDIv2,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    depth_lo: float = 22.0,
    depth_hi: float = 24.0,
) -> float:
    """Ratio of total NB mean at depth_hi vs depth_lo in y_meta (target ~4 for +2 log2)."""
    model.eval()
    m_lo = batch["y_meta"].clone()
    m_hi = batch["y_meta"].clone()
    m_lo[:, 0, :] = depth_lo
    m_hi[:, 0, :] = depth_hi
    with torch.no_grad():
        p_lo, n_lo, *_ = model.forward_tuple(
            batch["x_data"], batch["x_dna"], batch["x_meta"], m_lo,
        )
        p_hi, n_hi, *_ = model.forward_tuple(
            batch["x_data"], batch["x_dna"], batch["x_meta"], m_hi,
        )
    mean_lo = nb_mean(p_lo, n_lo)
    mean_hi = nb_mean(p_hi, n_hi)
    qm = batch["query_mask"].unsqueeze(1).expand_as(mean_lo)
    s_lo = mean_lo[qm].sum()
    s_hi = mean_hi[qm].sum()
    return float((s_hi / (s_lo + 1e-9)).item())


def _criteria_met(
    diag: Dict[str, float],
    criteria: PhaseCriteria,
    batch: Dict[str, torch.Tensor],
    *,
    require_dcr: bool,
) -> bool:
    pearson_ok = diag.get("pearson_all", 0.0) >= criteria.min_pearson
    rel_mae = diag.get("rel_mae", float("nan"))
    rel_mae_ok = not math.isnan(rel_mae) and rel_mae <= criteria.max_rel_mae
    imp_ok = True
    if batch["masked_map"].any():
        imp_ok = (
            math.isnan(diag.get("pearson_imp", float("nan")))
            or diag.get("pearson_imp", 0.0) >= criteria.min_imp_pearson
        )
        imp_rel = diag.get("imp_rel_mae", float("nan"))
        if not math.isnan(imp_rel) and imp_rel > criteria.max_rel_mae:
            imp_ok = False
    depth_ok = True
    if require_dcr and criteria.min_depth_ratio > 0:
        dcr = diag.get("depth_count_ratio", float("nan"))
        depth_ok = not math.isnan(dcr) and dcr >= criteria.min_depth_ratio
    return pearson_ok and rel_mae_ok and imp_ok and depth_ok



@dataclass
class RunResult:
    phase: str
    passed: bool
    steps: int
    final_metrics: Dict[str, float]
    history: List[Dict[str, float]] = field(default_factory=list)
    failure_reason: str = ""


def train_phase(
    phase: str,
    device: torch.device,
    *,
    max_steps: int = 2000,
    log_every: int = 100,
    patience: int = 500,
    depth_offset: bool = False,
) -> RunResult:
    data_cfg = phase_config(phase)
    criteria = PHASE_CRITERIA[phase]
    model_cfg = build_overfit_model_config(
        num_assays=data_cfg.num_assays,
        context_length=data_cfg.context_length,
    )

    torch.manual_seed(model_cfg.training.seed)
    model = CANDIv2(model_cfg).to(device)
    if depth_offset:
        apply_depth_offset_decoder(model)
        print(f"[{phase}] depth-offset NB head enabled (E29 prototype)", flush=True)
    loss_fn = build_v2_loss(model_cfg)
    lr = model_cfg.training.optimizer.adamax.lr
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    batch = generate_synthetic_batch(data_cfg, device)
    grad_mon = GradientMonitor(model)
    grad_mon.register()

    history: List[Dict[str, float]] = []
    best_metrics: Dict[str, float] = {}
    best_state: Optional[Dict[str, torch.Tensor]] = None
    steps_since_improve = 0
    passed = False
    failure_reason = ""

    t0 = time.time()
    for step in range(1, max_steps + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        grad_mon.grad_norms.clear()

        out = model(
            batch["x_data"], batch["x_dna"], batch["x_meta"], batch["y_meta"],
        )
        p, n, mu_out, var_out, df_out, peak_out = model.forward_tuple(
            batch["x_data"], batch["x_dna"], batch["x_meta"], batch["y_meta"],
        )

        loss, stats, _terms = loss_fn.forward_with_terms(
            p, n, mu_out, var_out, df_out, peak_out,
            batch["y_data"], batch["y_pval"], batch["y_peaks"],
            batch["observed_map"], batch["masked_map"],
            batch["signal_observed_map"], batch["signal_masked_map"],
            global_step=step,
            fallback_imp_to_observed_when_no_masked=False,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), model_cfg.training.grad.clip_norm)
        opt.step()

        with torch.no_grad():
            diag = compute_diagnostics(model, batch, out, stats, grad_mon)
            diag["step"] = float(step)
            diag["loss"] = float(loss.item())
            if step % 50 == 0 or step == 1:
                diag["depth_count_ratio"] = depth_count_ratio_probe(
                    model, batch, device,
                    depth_lo=DEPTH_PROBE_LO, depth_hi=DEPTH_PROBE_HI,
                )
            if batch["masked_map"].any():
                msk = batch["masked_map"]
                err = (nb_mean(out["p"], out["n"]) - batch["y_data"]).abs()[msk]
                diag["imp_rel_mae"] = float(
                    (err / batch["y_data"][msk].clamp(min=1.0)).mean().item()
                )

        if step % log_every == 0 or step == 1:
            elapsed = time.time() - t0
            dcr = diag.get("depth_count_ratio", float("nan"))
            print(
                f"[{phase}] step={step:4d} loss={diag['loss']:.4f} "
                f"obs_nll={diag['count_obs_nll']:.4f} rel_mae={diag.get('rel_mae', float('nan')):.4f} "
                f"pearson={diag['pearson_all']:.4f} dcr={dcr:.3f} "
                f"grad_dec_film={diag.get('grad_param_dec_film', 0):.2e} "
                f"({elapsed:.1f}s)",
                flush=True,
            )
            history.append(diag)

        if diag.get("rel_mae", float("inf")) < best_metrics.get("rel_mae", float("inf")):
            best_metrics = diag.copy()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            steps_since_improve = 0
        else:
            steps_since_improve += 1
            if steps_since_improve >= patience:
                failure_reason = f"no improvement for {patience} steps"
                break

        require_dcr = criteria.min_depth_ratio > 0
        if _criteria_met(diag, criteria, batch, require_dcr=require_dcr):
            passed = True
            best_metrics = diag.copy()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            break

    # Restore best weights and run final probes
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(batch["x_data"], batch["x_dna"], batch["x_meta"], batch["y_meta"])
        p, n, *_ = model.forward_tuple(
            batch["x_data"], batch["x_dna"], batch["x_meta"], batch["y_meta"],
        )
        _, stats, _ = loss_fn.forward_with_terms(
            p, n, torch.zeros_like(p), torch.zeros_like(p), None, torch.zeros_like(p),
            batch["y_data"], batch["y_pval"], batch["y_peaks"],
            batch["observed_map"], batch["masked_map"],
            batch["signal_observed_map"], batch["signal_masked_map"],
            fallback_imp_to_observed_when_no_masked=False,
        )
        final_diag = compute_diagnostics(model, batch, out, stats, grad_mon)
        final_diag["depth_count_ratio"] = depth_count_ratio_probe(
            model, batch, device, depth_lo=DEPTH_PROBE_LO, depth_hi=DEPTH_PROBE_HI,
        )
        if batch["masked_map"].any():
            msk = batch["masked_map"]
            err = (nb_mean(out["p"], out["n"]) - batch["y_data"]).abs()[msk]
            final_diag["imp_rel_mae"] = float(
                (err / batch["y_data"][msk].clamp(min=1.0)).mean().item()
            )
        best_metrics = final_diag
        if not passed:
            passed = _criteria_met(
                final_diag, criteria, batch,
                require_dcr=criteria.min_depth_ratio > 0,
            )

    if not passed:
        dcr = best_metrics.get("depth_count_ratio", float("nan"))
        if best_metrics.get("pearson_all", 0) < criteria.min_pearson:
            failure_reason = failure_reason or (
                f"pearson={best_metrics.get('pearson_all'):.4f} < {criteria.min_pearson}"
            )
        elif best_metrics.get("rel_mae", 999) > criteria.max_rel_mae:
            failure_reason = failure_reason or (
                f"rel_mae={best_metrics.get('rel_mae'):.4f} > {criteria.max_rel_mae}"
            )
        elif criteria.min_depth_ratio > 0 and (math.isnan(dcr) or dcr < criteria.min_depth_ratio):
            failure_reason = failure_reason or f"depth_count_ratio={dcr:.3f} < {criteria.min_depth_ratio}"
        elif batch["masked_map"].any() and best_metrics.get("pearson_imp", 0) < criteria.min_imp_pearson:
            failure_reason = failure_reason or (
                f"imp_pearson={best_metrics.get('pearson_imp'):.4f} < {criteria.min_imp_pearson}"
            )

    grad_mon.clear()
    status = "PASS" if passed else "FAIL"
    print(f"\n[{phase}] {status}: {failure_reason or 'all criteria met'}", flush=True)
    print(f"  best: rel_mae={best_metrics.get('rel_mae', float('nan')):.4f} "
          f"pearson={best_metrics.get('pearson_all', float('nan')):.4f} "
          f"obs_nll={best_metrics.get('count_obs_nll', float('nan')):.4f} "
          f"dcr={best_metrics.get('depth_count_ratio', float('nan')):.3f}", flush=True)

    return RunResult(
        phase=phase,
        passed=passed,
        steps=step,
        final_metrics=best_metrics,
        history=history,
        failure_reason=failure_reason,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CANDI v2 synthetic overfit diagnosis")
    p.add_argument("--phase", default="all", choices=["p1", "p2", "p3", "p4", "p5", "all"])
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--output-dir", type=str, default="sandbox/diagnostics/runs")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--depth-offset", action="store_true",
                   help="Use E29 library-size offset NB head (diagnostic fix for Q5)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}", flush=True)

    phases = ["p1", "p2", "p3", "p4", "p5"] if args.phase == "all" else [args.phase]
    results: List[RunResult] = []
    all_pass = True

    for ph in phases:
        print(f"\n{'='*60}\nPhase {ph.upper()}\n{'='*60}", flush=True)
        res = train_phase(
            ph, device, max_steps=args.max_steps, depth_offset=args.depth_offset,
        )
        results.append(res)
        if not res.passed:
            all_pass = False
            print(f"Stopping at {ph} — fix before continuing.", flush=True)
            break

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"overfit_{ts}.json"
    payload = {
        "all_pass": all_pass,
        "device": str(device),
        "results": [
            {
                "phase": r.phase,
                "passed": r.passed,
                "steps": r.steps,
                "failure_reason": r.failure_reason,
                "final_metrics": {k: v for k, v in r.final_metrics.items()
                                  if not k.endswith("_DEAD") and not k.endswith("_EXPLODE")},
            }
            for r in results
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nResults written to {out_path}", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
