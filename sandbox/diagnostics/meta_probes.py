"""Metadata conditioning probes: x_meta (encoder/latent) vs y_meta (decoder/output)."""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from sandbox.candi_v2.model import CANDIv2
from sandbox.diagnostics.synthetic_overfit import nb_mean


def _encode_latent(
    model: CANDIv2,
    prep: Dict[str, torch.Tensor],
    x_meta: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        z, _ = model.encoder.encode(
            prep["x_data"], prep["x_dna"], x_meta, return_meta=True,
        )
    return z


def latent_delta_ratio(
    model: CANDIv2,
    prep: Dict[str, torch.Tensor],
    x_meta_alt: torch.Tensor,
) -> float:
    """Relative L2 change in encoder latent when x_meta is perturbed."""
    z0 = _encode_latent(model, prep, prep["x_meta"])
    z1 = _encode_latent(model, prep, x_meta_alt)
    d = (z1 - z0).norm()
    base = z0.norm().clamp(min=1e-9)
    return float((d / base).item())


def count_output_from_y_meta(
    model: nn.Module,
    prep: Dict[str, torch.Tensor],
    y_meta: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        p, n, *_ = model(
            prep["x_data"], prep["x_dna"], prep["x_meta"], y_meta,
            query_mask=prep["query_mask"],
            query_mask_signal=prep["query_mask_signal"],
        )
    return nb_mean(p, n)


def prompt_count_delta_mse(
    model: nn.Module,
    prep: Dict[str, torch.Tensor],
    y_meta_a: torch.Tensor,
    y_meta_b: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """MSE between count NB means under two y_meta settings."""
    mu_a = count_output_from_y_meta(model, prep, y_meta_a)
    mu_b = count_output_from_y_meta(model, prep, y_meta_b)
    if mask is None:
        q = prep["query_mask"].unsqueeze(1).expand_as(mu_a)
    else:
        q = mask & prep["query_mask"].unsqueeze(1).expand_as(mu_a)
    if not q.any():
        return float("nan")
    d = (mu_a.float() - mu_b.float()).pow(2)
    return float(d[q].mean().item())


def prompt_count_depth_ratio(
    model: nn.Module,
    prep: Dict[str, torch.Tensor],
    base_y_meta: torch.Tensor,
    depth_lo: float = 22.0,
    depth_hi: float = 24.0,
    *,
    assay_mask: Optional[torch.Tensor] = None,
    position_mask: Optional[torch.Tensor] = None,
) -> float:
    """Count mean ratio depth_hi/depth_lo in y_meta.

    position_mask: if set (e.g. masked_map), ratio uses only those bins — fixes dilution.
    """
    m_lo = base_y_meta.clone()
    m_hi = base_y_meta.clone()
    if assay_mask is None:
        m_lo[:, 0, :] = depth_lo
        m_hi[:, 0, :] = depth_hi
    else:
        for b in range(m_lo.shape[0]):
            for a in range(m_lo.shape[2]):
                if assay_mask[b, a]:
                    m_lo[b, 0, a] = depth_lo
                    m_hi[b, 0, a] = depth_hi
    mu_lo = count_output_from_y_meta(model, prep, m_lo)
    mu_hi = count_output_from_y_meta(model, prep, m_hi)
    if position_mask is not None:
        q = position_mask
    else:
        q = prep["query_mask"].unsqueeze(1).expand_as(mu_lo)
    if not q.any():
        return float("nan")
    s_lo = mu_lo[q].sum()
    s_hi = mu_hi[q].sum()
    return float((s_hi / (s_lo + 1e-9)).item())


def prompt_count_depth_ratio_per_assay_median(
    model: nn.Module,
    prep: Dict[str, torch.Tensor],
    base_y_meta: torch.Tensor,
    depth_lo: float = 22.0,
    depth_hi: float = 24.0,
    *,
    position_mask: Optional[torch.Tensor] = None,
) -> float:
    """Median per-assay dcr (depth_hi/lo on full y_meta), evaluated on position_mask bins."""
    m_lo = base_y_meta.clone()
    m_hi = base_y_meta.clone()
    m_lo[:, 0, :] = depth_lo
    m_hi[:, 0, :] = depth_hi
    mu_lo = count_output_from_y_meta(model, prep, m_lo)
    mu_hi = count_output_from_y_meta(model, prep, m_hi)
    pos = prep["masked_map"] if position_mask is None else position_mask
    if not pos.any():
        return float("nan")
    ratios = []
    b, a = pos.shape[0], pos.shape[2]
    for ai in range(a):
        m = pos[:, :, ai]
        if not m.any():
            continue
        slo = mu_lo[:, :, ai][m].sum()
        shi = mu_hi[:, :, ai][m].sum()
        if slo > 1e-9:
            ratios.append(float((shi / slo).item()))
    if not ratios:
        return float("nan")
    ratios.sort()
    mid = len(ratios) // 2
    return ratios[mid] if len(ratios) % 2 else 0.5 * (ratios[mid - 1] + ratios[mid])


def ablate_x_meta_observed_assays(
    x_meta: torch.Tensor,
    observed_cols: torch.Tensor,
    *,
    fill: str = "missing",
) -> torch.Tensor:
    """Wipe x_meta rows on assay columns marked True in observed_cols [B, A]."""
    out = x_meta.clone()
    n_signal = observed_cols.shape[1]
    for b in range(observed_cols.shape[0]):
        for a in range(n_signal):
            if bool(observed_cols[b, a].item()):
                if fill == "missing":
                    out[b, :, a] = -1.0
                else:
                    out[b, :, a] = 0.0
    return out


def ablate_x_meta_observed_columns(
    x_meta: torch.Tensor,
    masked_map: torch.Tensor,
) -> torch.Tensor:
    """Remove x_meta for assay columns that are never masked (keep CLOZE on masked cols)."""
    masked_cols = masked_map.any(dim=1)  # [B, A]
    observed_cols = ~masked_cols
    return ablate_x_meta_observed_assays(x_meta, observed_cols, fill="missing")


def perturb_x_meta_row(
    x_meta: torch.Tensor,
    row: int,
    value,
    *,
    assay_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    out = x_meta.clone()
    if assay_mask is None:
        out[:, row, :] = value
    else:
        for b in range(out.shape[0]):
            for a in range(out.shape[2]):
                if assay_mask[b, a]:
                    out[b, row, a] = value
    return out


def perturb_x_meta_depth_delta(
    x_meta: torch.Tensor,
    delta: float,
    *,
    assay_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    out = x_meta.clone()
    if assay_mask is None:
        valid = out[:, 0, :] >= 0
        out[:, 0, :] = torch.where(valid, out[:, 0, :] + delta, out[:, 0, :])
    else:
        for b in range(out.shape[0]):
            for a in range(out.shape[2]):
                if assay_mask[b, a] and out[b, 0, a].item() >= 0:
                    out[b, 0, a] = out[b, 0, a] + delta
    return out


def masked_assay_columns(masked_map: torch.Tensor) -> torch.Tensor:
    """[B, A] True if assay has any masked positions."""
    return masked_map.any(dim=1)


def run_probe_battery(
    model: CANDIv2,
    probe_model: nn.Module,
    prep: Dict[str, torch.Tensor],
    *,
    depth_lo: float = 22.0,
    depth_hi: float = 24.0,
) -> Dict[str, float]:
    """Full x_meta + y_meta sensitivity battery on one prepared batch."""
    out: Dict[str, float] = {}
    y = prep["y_meta"]
    x = prep["x_meta"]
    masked_cols = masked_assay_columns(prep["masked_map"]) if prep["masked_map"].any() else None

    # --- y_meta / prompt / decoder path ---
    out["y_depth_dcr_all"] = prompt_count_depth_ratio(
        probe_model, prep, y, depth_lo, depth_hi,
    )
    if masked_cols is not None and masked_cols.any():
        out["y_depth_dcr_masked_assays"] = prompt_count_depth_ratio(
            probe_model, prep, y, depth_lo, depth_hi, assay_mask=masked_cols,
        )
        out["y_depth_dcr_observed_assays"] = prompt_count_depth_ratio(
            probe_model, prep, y, depth_lo, depth_hi,
            assay_mask=~masked_cols & (prep["query_mask"] > 0),
        )
        # M08: ratio on imputation bins only (no dilution from observed columns)
        out["y_depth_dcr_on_masked_bins"] = prompt_count_depth_ratio(
            probe_model, prep, y, depth_lo, depth_hi, position_mask=prep["masked_map"],
        )
        out["y_depth_dcr_median_per_assay_masked_bins"] = (
            prompt_count_depth_ratio_per_assay_median(
                probe_model, prep, y, depth_lo, depth_hi, position_mask=prep["masked_map"],
            )
        )

    y_rl_a = y.clone()
    y_rl_b = y.clone()
    y_rl_a[:, 2, :] = 36.0
    y_rl_b[:, 2, :] = 100.0
    out["y_readlen_count_mse"] = prompt_count_delta_mse(probe_model, prep, y_rl_a, y_rl_b)

    y_rt0 = y.clone()
    y_rt1 = y.clone()
    y_rt0[:, 3, :] = 0.0
    y_rt1[:, 3, :] = 1.0
    out["y_runtype_count_mse"] = prompt_count_delta_mse(probe_model, prep, y_rt0, y_rt1)

    if masked_cols is not None and masked_cols.any():
        out["y_readlen_count_mse_masked"] = prompt_count_delta_mse(
            probe_model, prep, y_rl_a, y_rl_b, mask=masked_cols.unsqueeze(1).expand_as(prep["masked_map"]),
        )

    # Wrong depth on masked assays only in y_meta (-2 log2)
    if masked_cols is not None and masked_cols.any():
        y_wrong = y.clone()
        for b in range(y.shape[0]):
            for a in range(y.shape[2]):
                if masked_cols[b, a] and y[b, 0, a].item() >= 0:
                    y_wrong[b, 0, a] = y[b, 0, a] - 2.0
        out["y_wrong_depth_masked_mse"] = prompt_count_delta_mse(
            probe_model, prep, y, y_wrong,
        )

    # --- x_meta / input / encoder path (signal assays only; last col may be control) ---
    n_signal = prep["masked_map"].shape[2]
    x_sig = x[:, :, :n_signal]
    x_d1 = perturb_x_meta_depth_delta(x_sig, +1.0)
    x_d1_full = x.clone()
    x_d1_full[:, :, :n_signal] = x_d1
    out["x_depth_latent_delta"] = latent_delta_ratio(model, prep, x_d1_full)

    x_rl = x.clone()
    valid_rl = x_sig[:, 2, :] >= 0
    x_rl[:, 2, :n_signal] = torch.where(
        valid_rl, torch.full_like(x_sig[:, 2, :], 100.0), x_sig[:, 2, :],
    )
    out["x_readlen_latent_delta"] = latent_delta_ratio(model, prep, x_rl)

    if masked_cols is not None and masked_cols.any():
        x_from_y = x.clone()
        x_wrong = x.clone()
        for b in range(masked_cols.shape[0]):
            for a in range(n_signal):
                if masked_cols[b, a]:
                    if y[b, 0, a].item() >= 0:
                        x_from_y[b, 0, a] = y[b, 0, a]
                        x_wrong[b, 0, a] = y[b, 0, a] - 2.0
        out["x_masked_fill_y_depth_latent_delta"] = latent_delta_ratio(model, prep, x_from_y)
        out["x_masked_wrong_depth_latent_delta"] = latent_delta_ratio(model, prep, x_wrong)

    return out
