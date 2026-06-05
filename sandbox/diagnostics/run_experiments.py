#!/usr/bin/env python3
"""Batch ablation matrix for synthetic overfit diagnostics.

Usage:
    python -m sandbox.diagnostics.run_experiments
    python -m sandbox.diagnostics.run_experiments --ids E01,E02
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn

from sandbox.candi_v2.config import CANDIv2Config, DecoderConfig, EncoderConfig
from sandbox.candi_v2.loss import build_v2_loss
from sandbox.candi_v2.model import CANDIv2
from sandbox.diagnostics.depth_offset_nb import apply_depth_offset_decoder
from sandbox.diagnostics.synthetic_data import generate_synthetic_batch, make_data_config
from sandbox.diagnostics.synthetic_overfit import (
    DEPTH_PROBE_HI,
    DEPTH_PROBE_LO,
    PHASE_CRITERIA,
    GradientMonitor,
    PhaseCriteria,
    _criteria_met,
    build_overfit_model_config,
    compute_diagnostics,
    depth_count_ratio_probe,
    nb_mean,
)


@dataclass
class ExperimentSpec:
    id: str
    description: str
    phase: str = "p3"
    max_steps: int = 5000
    patience: int = 500
    depth_offset: bool = False
    context_length: Optional[int] = None
    spatial_pattern: Optional[bool] = None
    stochastic: Optional[bool] = None
    stochastic_dsf: Optional[bool] = None
    encoder_film: Literal["default", "none"] = "default"
    decoder_film: Literal["default", "none"] = "default"
    optimizer: Literal["adam", "adamw", "sgd"] = "adam"
    lr: float = 1e-2
    meta_lr_mult: float = 1.0
    clip_norm: float = 10.0
    weight_decay: float = 0.0
    # If set, only check these metrics for "passed" (informational runs)
    require_dcr: Optional[bool] = None
    mask_counts: Optional[Tuple[int, ...]] = None
    production_depth: bool = False  # log2 depth 22–25 with matched count scale


EXPERIMENTS: List[ExperimentSpec] = [
    ExperimentSpec("E01", "P3 default NB — Q5 collapse repro", depth_offset=False),
    ExperimentSpec("E02", "P3 depth-offset — Q5 fix repro", depth_offset=True),
    ExperimentSpec("E03", "P1 spatial sine L=768 capacity", phase="p1", spatial_pattern=True,
                   context_length=768, depth_offset=False, max_steps=8000),
    ExperimentSpec("E04", "P2 spatial sine L=768 + depth vary", phase="p2", spatial_pattern=True,
                   context_length=768, depth_offset=False, max_steps=8000, require_dcr=False),
    ExperimentSpec("E05", "P3 offset + stochastic NB targets", depth_offset=True, stochastic=True),
    ExperimentSpec("E06", "P3 offset + decoder FiLM off", depth_offset=True, decoder_film="none"),
    ExperimentSpec("E07", "P3 offset + encoder FiLM off", depth_offset=True, encoder_film="none"),
    ExperimentSpec("E08", "P3 offset + enc/dec FiLM off", depth_offset=True,
                   encoder_film="none", decoder_film="none"),
    ExperimentSpec("E09", "P3 default lr=1e-3", depth_offset=False, lr=1e-3),
    ExperimentSpec("E10", "P3 default lr=3e-3", depth_offset=False, lr=3e-3),
    ExperimentSpec("E11", "P3 default AdamW wd=1e-4", depth_offset=False, optimizer="adamw",
                   weight_decay=1e-4),
    ExperimentSpec("E12", "P3 offset SGD lr=1e-2", depth_offset=True, optimizer="sgd", lr=1e-2),
    ExperimentSpec("E13", "P3 offset meta/film lr 10×", depth_offset=True, meta_lr_mult=10.0),
    ExperimentSpec("E14", "P4 offset + stochastic DSF", phase="p4", depth_offset=True,
                   stochastic_dsf=True),
    ExperimentSpec("E15", "P5 spatial motif + offset", phase="p5", depth_offset=True,
                   spatial_pattern=True),
    ExperimentSpec("E16", "P3 offset clip_norm=1", depth_offset=True, clip_norm=1.0),
    ExperimentSpec("E17", "P3 offset clip_norm=50", depth_offset=True, clip_norm=50.0),
    ExperimentSpec("E18", "P3 default clip_norm=1", depth_offset=False, clip_norm=1.0),
    # Round 2 — disambiguation / stress / production scale
    ExperimentSpec("E19", "P1 flat L=768 (length not spatial)", phase="p1",
                   context_length=768, spatial_pattern=False, max_steps=4000, patience=800),
    ExperimentSpec("E20", "P3 flat L=768 + offset", phase="p3",
                   context_length=768, spatial_pattern=False, depth_offset=True,
                   max_steps=4000, patience=800),
    ExperimentSpec("E21", "P3 offset mask 3/4 assays", phase="p3", depth_offset=True,
                   mask_counts=(3, 3, 3, 3)),
    ExperimentSpec("E22", "P3 offset production depth scale", phase="p3",
                   depth_offset=True, production_depth=True),
]


def _disable_encoder_film(model: CANDIv2) -> None:
    enc = model.encoder
    enc.signal_tower.pre_film = None
    enc.signal_tower.per_conv_film_layers = None
    enc.signal_tower.post_film = None
    enc.transformer_film_layers = None


def _measure_film_stats(model: CANDIv2, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    with torch.no_grad():
        x_data, x_dna, x_meta, y_meta = (
            batch["x_data"], batch["x_dna"], batch["x_meta"], batch["y_meta"],
        )
        if model.decoder.pre_decoder_film is not None and model.decoder.decoder_meta_embedding is not None:
            dec_emb = model.decoder.decoder_meta_embedding(y_meta.float())
            pooled = dec_emb.mean(dim=1)
            scale, shift = model.decoder.pre_decoder_film.proj(pooled).chunk(2, dim=-1)
            out["dec_film_scale_abs"] = float(scale.abs().mean().item())
            out["dec_film_shift_abs"] = float(shift.abs().mean().item())
        z, meta_emb = model.encoder.encode(x_data, x_dna, x_meta, return_meta=True)
        if model.encoder.signal_tower.per_conv_film_layers:
            sig = x_data[..., : model.encoder.num_tracks].float()
            from sandbox.candi_v2.encoder import _apply_signal_transform
            sig = _apply_signal_transform(sig, model.encoder.cfg.signal_transform)
            x = sig.permute(0, 2, 1)
            scales = []
            for i, block in enumerate(model.encoder.signal_tower.blocks):
                x = block(x)
                film = model.encoder.signal_tower.per_conv_film_layers[i]
                params = film.proj(meta_emb)
                sc, _ = params.chunk(2, dim=-1)
                scales.append(sc.abs().mean().item())
            if scales:
                out["enc_film_scale_abs"] = float(sum(scales) / len(scales))
        out["latent_std"] = float(z.std().item())
    return out


def _build_optimizer(
    model: nn.Module,
    spec: ExperimentSpec,
) -> torch.optim.Optimizer:
    lr = spec.lr
    meta_params: List[nn.Parameter] = list(model.encoder.metadata_embedding.parameters())
    if model.decoder.decoder_meta_embedding is not None:
        meta_params += list(model.decoder.decoder_meta_embedding.parameters())
    film_params: List[nn.Parameter] = []
    if model.decoder.pre_decoder_film is not None:
        film_params += list(model.decoder.pre_decoder_film.parameters())
    if model.encoder.signal_tower.per_conv_film_layers is not None:
        for layer in model.encoder.signal_tower.per_conv_film_layers:
            film_params += list(layer.parameters())
    meta_film_ids = {id(p) for p in meta_params + film_params}
    base_params = [p for p in model.parameters() if id(p) not in meta_film_ids]
    m_lr = lr * spec.meta_lr_mult

    if spec.meta_lr_mult != 1.0 and (meta_params or film_params):
        groups = [{"params": base_params, "lr": lr}]
        if meta_params:
            groups.append({"params": meta_params, "lr": m_lr})
        if film_params:
            groups.append({"params": film_params, "lr": m_lr})
        if spec.optimizer == "adamw":
            return torch.optim.AdamW(groups, lr=lr, weight_decay=spec.weight_decay)
        if spec.optimizer == "sgd":
            return torch.optim.SGD(groups, lr=lr, momentum=0.9)
        return torch.optim.Adam(groups, lr=lr)

    params = model.parameters()
    if spec.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=spec.weight_decay)
    if spec.optimizer == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    return torch.optim.Adam(params, lr=lr)


def run_experiment(spec: ExperimentSpec, device: torch.device) -> Dict[str, Any]:
    data_cfg = make_data_config(
        spec.phase,
        context_length=spec.context_length,
        spatial_pattern=spec.spatial_pattern,
        stochastic=spec.stochastic,
        stochastic_dsf=spec.stochastic_dsf,
        mask_counts=spec.mask_counts,
        production_depth=spec.production_depth,
    )
    criteria = PHASE_CRITERIA[spec.phase]
    if spec.require_dcr is False:
        criteria = replace(criteria, min_depth_ratio=0.0)

    model_cfg = build_overfit_model_config(
        num_assays=data_cfg.num_assays,
        context_length=data_cfg.context_length,
    )
    if spec.decoder_film == "none":
        model_cfg.decoder.film_mode = "none"
    model_cfg.training.optimizer.adamax.lr = spec.lr
    model_cfg.training.grad.clip_norm = spec.clip_norm

    torch.manual_seed(model_cfg.training.seed)
    model = CANDIv2(model_cfg).to(device)
    if spec.encoder_film == "none":
        _disable_encoder_film(model)
    if spec.depth_offset:
        apply_depth_offset_decoder(model)

    loss_fn = build_v2_loss(model_cfg)
    opt = _build_optimizer(model, spec)
    batch = generate_synthetic_batch(data_cfg, device)
    grad_mon = GradientMonitor(model)
    grad_mon.register()

    best_metrics: Dict[str, float] = {}
    best_state = None
    steps_since_improve = 0
    passed = False
    failure_reason = ""
    step = 0

    t0 = time.time()
    for step in range(1, spec.max_steps + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        grad_mon.grad_norms.clear()

        out = model(batch["x_data"], batch["x_dna"], batch["x_meta"], batch["y_meta"])
        p, n, mu_out, var_out, df_out, peak_out = model.forward_tuple(
            batch["x_data"], batch["x_dna"], batch["x_meta"], batch["y_meta"],
        )
        loss, stats, _ = loss_fn.forward_with_terms(
            p, n, mu_out, var_out, df_out, peak_out,
            batch["y_data"], batch["y_pval"], batch["y_peaks"],
            batch["observed_map"], batch["masked_map"],
            batch["signal_observed_map"], batch["signal_masked_map"],
            global_step=step,
            fallback_imp_to_observed_when_no_masked=False,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), spec.clip_norm)
        opt.step()

        with torch.no_grad():
            diag = compute_diagnostics(model, batch, out, stats, grad_mon)
            if step % 50 == 0 or step == 1:
                diag["depth_count_ratio"] = depth_count_ratio_probe(
                    model, batch, device, depth_lo=DEPTH_PROBE_LO, depth_hi=DEPTH_PROBE_HI,
                )
            if batch["masked_map"].any():
                msk = batch["masked_map"]
                err = (nb_mean(out["p"], out["n"]) - batch["y_data"]).abs()[msk]
                diag["imp_rel_mae"] = float(
                    (err / batch["y_data"][msk].clamp(min=1.0)).mean().item()
                )

        if diag.get("rel_mae", float("inf")) < best_metrics.get("rel_mae", float("inf")):
            best_metrics = diag.copy()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            steps_since_improve = 0
        else:
            steps_since_improve += 1
            if steps_since_improve >= spec.patience:
                failure_reason = f"patience={spec.patience}"
                break

        req_dcr = criteria.min_depth_ratio > 0
        if _criteria_met(diag, criteria, batch, require_dcr=req_dcr):
            passed = True
            best_metrics = diag.copy()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            break

    elapsed = time.time() - t0
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
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
        final = compute_diagnostics(model, batch, out, stats, grad_mon)
        final["depth_count_ratio"] = depth_count_ratio_probe(
            model, batch, device, depth_lo=DEPTH_PROBE_LO, depth_hi=DEPTH_PROBE_HI,
        )
        if batch["masked_map"].any():
            msk = batch["masked_map"]
            err = (nb_mean(out["p"], out["n"]) - batch["y_data"]).abs()[msk]
            final["imp_rel_mae"] = float(
                (err / batch["y_data"][msk].clamp(min=1.0)).mean().item()
            )
        final.update(_measure_film_stats(model, batch))
        best_metrics = final
        if not passed:
            passed = _criteria_met(final, criteria, batch, require_dcr=criteria.min_depth_ratio > 0)

    grad_mon.clear()

    key_metrics = {
        k: best_metrics.get(k)
        for k in (
            "rel_mae", "imp_rel_mae", "pearson_all", "pearson_imp",
            "depth_count_ratio", "count_obs_nll",
            "grad_param_enc_meta", "grad_param_dec_film", "grad_param_enc_film",
            "grad_param_nb_head", "grad_param_dec_meta",
            "dec_film_scale_abs", "dec_film_shift_abs", "enc_film_scale_abs", "latent_std",
        )
        if k in best_metrics or k.startswith("grad_")
    }

    status = "PASS" if passed else "FAIL"
    print(
        f"[{spec.id}] {status} {spec.description} | "
        f"rel_mae={best_metrics.get('rel_mae', float('nan')):.4f} "
        f"imp={best_metrics.get('imp_rel_mae', float('nan')):.4f} "
        f"dcr={best_metrics.get('depth_count_ratio', float('nan')):.3f} "
        f"pearson={best_metrics.get('pearson_all', float('nan')):.3f} "
        f"({elapsed:.0f}s)",
        flush=True,
    )

    return {
        "id": spec.id,
        "description": spec.description,
        "passed": passed,
        "failure_reason": failure_reason,
        "steps": step,
        "elapsed_s": elapsed,
        "config": asdict(spec),
        "metrics": key_metrics,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ids", type=str, default=None, help="Comma-separated experiment IDs")
    p.add_argument("--output", type=str, default="sandbox/diagnostics/runs/ablation_matrix.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    specs = EXPERIMENTS
    if args.ids:
        wanted = set(args.ids.split(","))
        specs = [s for s in EXPERIMENTS if s.id in wanted]

    results = []
    for spec in specs:
        print(f"\n{'='*60}\n{spec.id}: {spec.description}\n{'='*60}", flush=True)
        results.append(run_experiment(spec, device))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"device": str(device), "results": results}, indent=2))
    print(f"\nWrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
