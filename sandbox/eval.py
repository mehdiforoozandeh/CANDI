"""Reconstruction + prompt-sensitivity probes (plan §9)."""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def _gather_masked(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """pred/target [B,L,F], mask [B,L,F] bool -> 1D aligned vectors."""
    m = mask.bool()
    return pred[m].float(), target[m].float()


def r2_score(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    p, t = _gather_masked(pred, target, mask)
    if p.numel() < 2:
        return float("nan")
    ss_res = float((p - t).pow(2).sum().item())
    mean_t = float(t.mean().item())
    ss_tot = float((t - mean_t).pow(2).sum().item())
    if ss_tot <= 1e-12:
        return float("nan")
    return float(1.0 - (ss_res / ss_tot))


def pearson_r(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    p, t = _gather_masked(pred, target, mask)
    if p.numel() < 4:
        return float("nan")
    p = p - p.mean()
    t = t - t.mean()
    denom = p.norm() * t.norm() + 1e-8
    return float((p * t).sum().item() / denom)


def spearman_r(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """Exact Spearman rank correlation via scipy (handles ties correctly)."""
    p, t = _gather_masked(pred, target, mask)
    if p.numel() < 4:
        return float("nan")
    r, _ = spearmanr(p.cpu().numpy(), t.cpu().numpy())
    return float(r) if math.isfinite(r) else float("nan")


def auroc_binary(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """AUROC for binary targets in {0,1} using sklearn."""
    p, t = _gather_masked(pred, target, mask)
    if p.numel() == 0:
        return float("nan")
    tbin = (t > 0.5).long()
    if tbin.min() == tbin.max():
        return float("nan")
    try:
        auc = roc_auc_score(tbin.cpu().numpy(), p.cpu().numpy())
    except Exception:
        return float("nan")
    return float(auc)


def _region_expand(region_type: Optional[torch.Tensor], B: int, L: int, F: int) -> Optional[torch.Tensor]:
    if region_type is None:
        return None
    return region_type[:, None, None].expand(B, L, F)


def _top1pct_mask(target: torch.Tensor, base_mask: torch.Tensor) -> torch.Tensor:
    """Return base_mask & top-1% of target values within base_mask (matching prod METRICS.get_1obs_signals).

    Threshold = np.percentile(target[base_mask], 99). Positions where target >= threshold AND
    base_mask is True are kept. If fewer than 4 masked positions exist, returns an all-False mask.
    """
    m = base_mask.bool()
    vals = target[m].float().cpu().numpy()
    if len(vals) < 4:
        return torch.zeros_like(base_mask)
    thr = float(np.percentile(vals, 99))
    return m & (target.float() >= thr)


@torch.no_grad()
def eval_batch_metrics(
    output_p: torch.Tensor,
    output_n: torch.Tensor,
    output_mu: torch.Tensor,
    output_peak: torch.Tensor,
    y_data: torch.Tensor,
    y_pval: torch.Tensor,
    y_peaks: torch.Tensor,
    observed_map: torch.Tensor,
    masked_map: torch.Tensor,
    signal_observed_map: torch.Tensor,
    signal_masked_map: torch.Tensor,
    *,
    regime: str = "type2_loci",
    region_type: Optional[torch.Tensor] = None,
    imp_eval_map: Optional[torch.Tensor] = None,
    imp_eval_signal_map: Optional[torch.Tensor] = None,
    y_data_imp: Optional[torch.Tensor] = None,
    y_pval_imp: Optional[torch.Tensor] = None,
    y_peaks_imp: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Denoising uses `y_pval` / `y_peaks` (T_* targets). Imputation uses `y_pval_imp` when provided.
    Peak AUROC is **genome-wide only** (plan §9a).
    """
    out: Dict[str, float] = {}
    _ = regime
    _ = region_type

    # NB count head: expected value = n * (1 - p) / p
    nb_mean = output_n * (1.0 - output_p) / (output_p + 1e-8)

    # Imp GT must come from V_*/B_* only. Build valid masks from imp tensors.
    imp_count_valid = (y_data_imp != -1) if y_data_imp is not None else torch.zeros_like(masked_map)
    imp_pval_valid = (y_pval_imp != -1) if y_pval_imp is not None else torch.zeros_like(masked_map)
    imp_peak_valid = (
        ((y_peaks_imp == 0) | (y_peaks_imp == 1)) if y_peaks_imp is not None else torch.zeros_like(masked_map)
    )

    # Imp scopes are grounded on caller-provided imputation map (typically T-unavailable & V/B-available).
    # Fallback to masked_map when imp_eval_map is omitted (unit tests / legacy callers).
    imp_base = imp_eval_map.bool() if imp_eval_map is not None else masked_map.bool()
    imp_base_signal = imp_eval_signal_map.bool() if imp_eval_signal_map is not None else signal_masked_map.bool()
    imp_gw_mask = imp_base & imp_pval_valid
    imp_dsf1_mask = imp_base_signal & imp_pval_valid

    # --- three scopes: gw, dsf1, 1obs ---
    m_den_1obs = _top1pct_mask(y_pval, observed_map)
    m_imp_1obs = (
        _top1pct_mask(y_pval_imp, imp_gw_mask)
        if y_pval_imp is not None
        else torch.zeros_like(masked_map)
    )

    scope_defs = [
        ("gw",   observed_map,                              imp_gw_mask),
        ("dsf1", observed_map & signal_observed_map,        imp_dsf1_mask),
        ("1obs", m_den_1obs,                                m_imp_1obs),
    ]

    for scope, m_den, m_imp in scope_defs:
        for prefix, m in (("den", m_den), ("imp", m_imp)):
            if prefix == "imp" and y_pval_imp is None:
                out[f"{prefix}_pval_pearson_{scope}"] = float("nan")
                out[f"{prefix}_pval_spearman_{scope}"] = float("nan")
                out[f"{prefix}_pval_r2_{scope}"] = float("nan")
                continue

            tgt_p = y_pval if prefix == "den" else y_pval_imp
            out[f"{prefix}_pval_pearson_{scope}"] = pearson_r(output_mu, tgt_p, m)
            out[f"{prefix}_pval_spearman_{scope}"] = spearman_r(output_mu, tgt_p, m)
            out[f"{prefix}_pval_r2_{scope}"] = r2_score(output_mu, tgt_p, m)

            # Count metrics: gw scope only
            if scope == "gw":
                if prefix == "imp":
                    if y_data_imp is None:
                        out[f"{prefix}_count_pearson_gw"] = float("nan")
                        out[f"{prefix}_count_spearman_gw"] = float("nan")
                        out[f"{prefix}_count_r2_gw"] = float("nan")
                    else:
                        count_mask = m & imp_count_valid
                        out[f"{prefix}_count_pearson_gw"] = pearson_r(nb_mean, y_data_imp, count_mask)
                        out[f"{prefix}_count_spearman_gw"] = spearman_r(nb_mean, y_data_imp, count_mask)
                        out[f"{prefix}_count_r2_gw"] = r2_score(nb_mean, y_data_imp, count_mask)
                else:
                    out[f"{prefix}_count_pearson_gw"] = pearson_r(nb_mean, y_data, m)
                    out[f"{prefix}_count_spearman_gw"] = spearman_r(nb_mean, y_data, m)
                    out[f"{prefix}_count_r2_gw"] = r2_score(nb_mean, y_data, m)

        # Peak AUROC at gw scope only
        if scope == "gw":
            peak_gt_ok_d = (y_peaks == 0) | (y_peaks == 1)
            out["den_peak_auroc_gw"] = auroc_binary(output_peak, y_peaks, m_den & peak_gt_ok_d)
            if y_peaks_imp is None:
                out["imp_peak_auroc_gw"] = float("nan")
            else:
                out["imp_peak_auroc_gw"] = auroc_binary(output_peak, y_peaks_imp, m_imp & imp_peak_valid)

    return out


def _forward_heads(
    model: torch.nn.Module,
    prep: Dict[str, torch.Tensor],
    y_meta: torch.Tensor,
    device: torch.device,
):
    return model(
        prep["x_data"],
        prep["x_dna"],
        prep["x_meta"],
        y_meta.to(device),
        query_mask=prep["query_mask"],
        query_mask_signal=prep["query_mask_signal"],
    )


def prompt_sensitivity_depth_count_ratio(
    model: torch.nn.Module,
    prep: Dict[str, torch.Tensor],
    base_meta: torch.Tensor,
    device: torch.device,
    *,
    depth_lo: float = 22.0,
    depth_hi: float = 24.0,
) -> float:
    """
    Ratio of total expected NB counts (mean = n(1-p)/p) at depth_hi vs depth_lo.

    Interpretation (default depth_lo=22, depth_hi=24, i.e. +2 in log2-space = 4x sequencing depth):
      - ~4.0  -> model is reading and using depth metadata correctly (target).
      - ~1.0  -> model is essentially ignoring the depth token in y_meta (depth-invariant
                  output). This is a *failure mode*, not a healthy state.
      - >>4.0 or <<1.0 -> over/under-shooting, often indicates an unstable depth head.
    """
    model.eval()
    m_lo = base_meta.clone()
    m_hi = base_meta.clone()
    m_lo[:, 0, :] = depth_lo
    m_hi[:, 0, :] = depth_hi
    with torch.no_grad():
        p_lo, n_lo, *_rest = _forward_heads(model, prep, m_lo, device)
        p_hi, n_hi, *_r2 = _forward_heads(model, prep, m_hi, device)
    mean_lo = n_lo * (1.0 - p_lo) / (p_lo + 1e-9)
    mean_hi = n_hi * (1.0 - p_hi) / (p_hi + 1e-9)
    qm = prep["query_mask"].unsqueeze(1).expand_as(mean_lo)
    s_lo = mean_lo[qm].sum()
    s_hi = mean_hi[qm].sum()
    return float((s_hi / (s_lo + 1e-9)).item())


def _forward_mu(
    model: torch.nn.Module,
    prep: Dict[str, torch.Tensor],
    y_meta: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    return _forward_heads(model, prep, y_meta, device)[2]


def prompt_sensitivity_readlen_mse(
    model: torch.nn.Module,
    prep: Dict[str, torch.Tensor],
    base_meta: torch.Tensor,
    device: torch.device,
    *,
    assay_idx: int = 0,
    readlen_a: float = 36.0,
    readlen_b: float = 100.0,
) -> float:
    """Mean squared delta in mu when toggling read length (row 2) for one assay."""
    m_a = base_meta.clone()
    m_b = base_meta.clone()
    m_a[:, 2, assay_idx] = readlen_a
    m_b[:, 2, assay_idx] = readlen_b
    model.eval()
    with torch.no_grad():
        mu_a = _forward_mu(model, prep, m_a, device)
        mu_b = _forward_mu(model, prep, m_b, device)
    q = prep["query_mask_signal"].unsqueeze(1).expand_as(mu_a) & (mu_a != -1)
    d = (mu_a.float() - mu_b.float()).pow(2)
    return float(d[q].mean().item()) if q.any() else float("nan")


def prompt_sensitivity_runtype_mse(
    model: torch.nn.Module,
    prep: Dict[str, torch.Tensor],
    base_meta: torch.Tensor,
    device: torch.device,
    *,
    assay_idx: int = 0,
) -> float:
    """MSE(mu) when flipping run_type (row 3) between 0 and 1."""
    m0 = base_meta.clone()
    m1 = base_meta.clone()
    m0[:, 3, assay_idx] = 0.0
    m1[:, 3, assay_idx] = 1.0
    model.eval()
    with torch.no_grad():
        mu0 = _forward_mu(model, prep, m0, device)
        mu1 = _forward_mu(model, prep, m1, device)
    q = prep["query_mask_signal"].unsqueeze(1).expand_as(mu0) & (mu0 != -1)
    d = (mu0.float() - mu1.float()).pow(2)
    return float(d[q].mean().item()) if q.any() else float("nan")
