"""Real chr19 sandbox overfit diagnostics for CANDI v2 (Q5 / E29 validation)."""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from sandbox.batch import make_masker, prepare_masked_batch
from sandbox.candi_v2.loss import build_v2_loss
from sandbox.candi_v2.model import CANDIv2
from sandbox.diagnostics.depth_offset_nb import apply_depth_offset_decoder
from sandbox.diagnostics.real_data import build_real_v2_config, collect_batches, default_h5_path
from sandbox.diagnostics.synthetic_overfit import GradientMonitor, nb_mean
from sandbox.eval import prompt_sensitivity_depth_count_ratio


@dataclass
class RealExperimentSpec:
    id: str
    description: str
    max_steps: int = 1500
    depth_offset: bool = False
    depth_center: float = 0.0  # subtract from log2 depth before 2^d (use ~24 for real EIC)
    n_batches: int = 1
    batch_size: int = 4
    # Masking knobs (sandbox defaults: p_full_assay=0.8, p_full_loci=0.5)
    p_full_assay: float = 0.8
    p_full_loci: float = 0.5
    p_chunks: float = 0.0
    mask_fraction: float = 0.2
    heads: str = "count_only"
    lr: float = 1e-3
    clip_norm: float = 2.0
    decoder_film: str = "default"  # default | none
    encoder_film: str = "default"  # default | none
    min_dcr: float = 3.0
    require_masked: bool = True


def _disable_encoder_film(model: CANDIv2) -> None:
    enc = model.encoder
    enc.signal_tower.pre_film = None
    enc.signal_tower.per_conv_film_layers = None
    enc.signal_tower.post_film = None
    enc.transformer_film_layers = None


class _V2TupleWrapper(nn.Module):
    """6-tuple forward for sandbox.eval probes."""

    def __init__(self, model: CANDIv2) -> None:
        super().__init__()
        self.v2 = model

    def forward(self, x_data, x_dna, x_meta, y_meta, **kwargs):
        return self.v2.forward_tuple(x_data, x_dna, x_meta, y_meta, **kwargs)


def _imp_count_pearson(
    p: torch.Tensor, n: torch.Tensor, y: torch.Tensor, masked: torch.Tensor,
) -> float:
    if not masked.any():
        return float("nan")
    mu = nb_mean(p, n)
    pred = mu[masked].flatten().float()
    tgt = y[masked].flatten().float()
    if pred.numel() < 2:
        return float("nan")
    pc = pred - pred.mean()
    tc = tgt - tgt.mean()
    denom = pc.norm() * tc.norm()
    if denom < 1e-12:
        return 0.0
    return float((pc @ tc / denom).item())


def run_real_experiment(
    spec: RealExperimentSpec,
    device: torch.device,
    *,
    h5_path: Optional[Path] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    h5 = h5_path or default_h5_path()
    batches = collect_batches(
        h5, n_batches=spec.n_batches, batch_size=spec.batch_size, seed=seed,
    )
    masker = make_masker(
        p_full_assay=spec.p_full_assay,
        p_full_loci=spec.p_full_loci,
        p_chunks=spec.p_chunks,
        mask_fraction=spec.mask_fraction,
        preserve_assay_id=True,
    )
    probe_masker = make_masker(p_full_assay=0.0, p_full_loci=0.0, preserve_assay_id=True)

    cfg = build_real_v2_config(
        heads=spec.heads, lr=spec.lr, clip_norm=spec.clip_norm, dropout=0.0,
    )
    if spec.decoder_film == "none":
        cfg.decoder.film_mode = "none"

    torch.manual_seed(seed)
    model = CANDIv2(cfg).to(device)
    if spec.encoder_film == "none":
        _disable_encoder_film(model)
    if spec.depth_offset:
        apply_depth_offset_decoder(model, depth_center=spec.depth_center)
    loss_fn = build_v2_loss(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=spec.lr)
    probe_model = _V2TupleWrapper(model).to(device)
    grad_mon = GradientMonitor(model)
    grad_mon.register()

    best: Dict[str, float] = {}
    best_state = None
    loss_start = None
    loss_end = None
    t0 = time.time()

    for step in range(1, spec.max_steps + 1):
        batch = batches[(step - 1) % len(batches)]
        model.train()
        prep = prepare_masked_batch(batch, masker, device)
        if prep is None:
            continue
        if spec.require_masked and not prep["masked_map"].any():
            continue

        opt.zero_grad(set_to_none=True)
        p, n, mu, var, df, peak = model.forward_tuple(
            prep["x_data"], prep["x_dna"], prep["x_meta"], prep["y_meta"],
        )
        loss, stats, _ = loss_fn.forward_with_terms(
            p, n, mu, var, df, peak,
            prep["y_data"], prep["y_pval"], prep["y_peaks"],
            prep["observed_map"], prep["masked_map"],
            prep["signal_observed_map"], prep["signal_masked_map"],
            global_step=step,
            fallback_imp_to_observed_when_no_masked=False,
        )
        if loss_start is None:
            loss_start = float(loss.item())
        loss_end = float(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), spec.clip_norm)
        opt.step()

        if step % 50 == 0 or step == 1:
            with torch.no_grad():
                prep_probe = prepare_masked_batch(batch, probe_masker, device, apply_mask=False)
                if prep_probe is not None:
                    dcr = prompt_sensitivity_depth_count_ratio(
                        probe_model, prep_probe, prep_probe["y_meta"], device,
                    )
                else:
                    dcr = float("nan")
                imp_p = _imp_count_pearson(p, n, prep["y_data"], prep["masked_map"])
                metrics = {
                    "total_loss": loss_end,
                    "count_obs_nll": stats.get("loss_branch_count_obs_raw", float("nan")),
                    "count_imp_nll": stats.get("loss_branch_count_imp_raw", float("nan")),
                    "depth_count_ratio": dcr,
                    "imp_count_pearson": imp_p,
                    "has_masked": float(prep["masked_map"].any().item()),
                }
                metrics.update(grad_mon.collect_param_norms())
                if metrics.get("total_loss", float("inf")) < best.get("total_loss", float("inf")):
                    best = metrics.copy()
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    elapsed = time.time() - t0
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()
    batch0 = batches[0]
    prep_probe = prepare_masked_batch(batch0, probe_masker, device, apply_mask=False)
    prep_mask = prepare_masked_batch(batch0, masker, device)
    final: Dict[str, float] = {"elapsed_s": elapsed}
    if prep_probe is not None:
        final["depth_count_ratio"] = prompt_sensitivity_depth_count_ratio(
            probe_model, prep_probe, prep_probe["y_meta"], device,
        )
    if prep_mask is not None:
        with torch.no_grad():
            p, n, mu, var, df, peak = model.forward_tuple(
                prep_mask["x_data"], prep_mask["x_dna"], prep_mask["x_meta"], prep_mask["y_meta"],
            )
            _, stats, _ = loss_fn.forward_with_terms(
                p, n, mu, var, df, peak,
                prep_mask["y_data"], prep_mask["y_pval"], prep_mask["y_peaks"],
                prep_mask["observed_map"], prep_mask["masked_map"],
                prep_mask["signal_observed_map"], prep_mask["signal_masked_map"],
                fallback_imp_to_observed_when_no_masked=False,
            )
            final["count_obs_nll"] = stats.get("loss_branch_count_obs_raw", float("nan"))
            final["count_imp_nll"] = stats.get("loss_branch_count_imp_raw", float("nan"))
            final["imp_count_pearson"] = _imp_count_pearson(
                p, n, prep_mask["y_data"], prep_mask["masked_map"],
            )
            final["total_loss"] = float(
                stats.get("loss_branch_count_obs_raw", 0.0)
                + stats.get("loss_branch_count_imp_raw", 0.0)
            )
    final.update(best)
    grad_mon.clear()

    dcr = final.get("depth_count_ratio", float("nan"))
    loss_drop = (loss_start - loss_end) / (abs(loss_start) + 1e-9) if loss_start is not None else 0.0
    dcr_ok = not math.isnan(dcr) and dcr >= spec.min_dcr
    loss_ok = loss_start is not None and loss_end is not None and loss_end < 0.85 * loss_start
    passed = loss_ok and (dcr_ok if spec.require_masked and spec.min_dcr > 0 else True)

    print(
        f"[{spec.id}] {'PASS' if passed else 'FAIL'} {spec.description} | "
        f"loss {loss_start:.3f}->{loss_end:.3f} dcr={dcr:.3f} "
        f"imp_p={final.get('imp_count_pearson', float('nan')):.3f} ({elapsed:.0f}s)",
        flush=True,
    )

    return {
        "id": spec.id,
        "description": spec.description,
        "passed": passed,
        "loss_start": loss_start,
        "loss_end": loss_end,
        "loss_drop_frac": loss_drop,
        "config": asdict(spec),
        "metrics": {k: final.get(k) for k in (
            "total_loss", "count_obs_nll", "count_imp_nll", "depth_count_ratio",
            "imp_count_pearson", "grad_param_nb_head", "grad_param_dec_film",
            "grad_param_dec_meta", "elapsed_s",
        ) if k in final},
    }
