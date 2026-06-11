"""CANDI v2 loss — fully vendored, zero external model-code dependencies.

The agent may freely edit this file to add regularization terms, alternative
objectives, ELBO/CVAE-style KL penalties, auxiliary losses, etc.

Dependency graph (all vendored here):
  negative_binomial_loss, gamma_nll_loss, students_t_nll_loss  ← _utils.py
  CustomLaplaceNLLLoss, LaplaceNLLLoss                         ← candi_loss.py
  CANDI_LOSS                                                   ← candi_loss.py
  SandboxCompositeLoss                                         ← sandbox/losses.py
  build_v2_loss                                                ← this module

Only torch, numpy, and the standard library are imported from outside june3/.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Laplace

from sandbox.autoresearch.june3.candi_v2.config import CANDIv2Config


# ---------------------------------------------------------------------------
# DCR soft-penalty singleton (set by V2Decoder at build time)
# ---------------------------------------------------------------------------
_DCR_SLOPE_REF: Optional[nn.Parameter] = None
_DCR_PENALTY_WEIGHT: float = 0.0


def set_dcr_slope_ref(param: Optional[nn.Parameter], weight: float = 0.0) -> None:
    """Register decoder's log_depth_slope param for DCR penalty in _compute_terms."""
    global _DCR_SLOPE_REF, _DCR_PENALTY_WEIGHT
    _DCR_SLOPE_REF = param
    _DCR_PENALTY_WEIGHT = float(weight)


# ---------------------------------------------------------------------------
# Low-level loss utilities (vendored from _utils.py)
# ---------------------------------------------------------------------------

def negative_binomial_loss(y_true, n_pred, p_pred, invalid_penalty=1e6):
    """Numerically-stable NB NLL (closed-form, float32-forced)."""
    eps = 1e-8
    y = y_true.float()
    n = n_pred.float()
    p = p_pred.float()
    p = torch.clamp(p, min=eps, max=1.0 - eps)
    n = torch.clamp(n, min=eps)
    q = torch.clamp(1.0 - p, min=eps, max=1.0 - eps)
    y = torch.clamp(y, min=0.0)
    log_prob = (
        torch.lgamma(y + n)
        - torch.lgamma(y + 1.0)
        - torch.lgamma(n)
        + n * torch.log1p(-q)
        + y * torch.log(q)
    )
    nll = -log_prob
    bad = ~torch.isfinite(nll)
    if bad.any():
        nll = torch.where(bad, torch.full_like(nll, float(invalid_penalty)), nll)
    return nll


def gamma_nll_loss(y_true, mu_pred, alpha_pred, reduction="mean", eps=1e-6, invalid_penalty=1e6):
    """Gamma NLL loss."""
    y = torch.clamp(y_true.float(), min=eps)
    mu = torch.clamp(mu_pred.float(), min=eps)
    alpha = torch.clamp(alpha_pred.float(), min=eps)
    dist = torch.distributions.Gamma(concentration=alpha, rate=alpha / mu)
    nll = -dist.log_prob(y)
    bad = ~torch.isfinite(nll)
    if bad.any():
        nll = torch.where(bad, torch.full_like(nll, float(invalid_penalty)), nll)
    if reduction == "mean":
        return nll.mean()
    if reduction == "sum":
        return nll.sum()
    return nll


def students_t_nll_loss(y_true, mu_pred, sigma_pred, df_pred, reduction="none", invalid_penalty=1e6):
    """Student-t NLL loss."""
    eps = 1e-8
    y = y_true.float()
    sigma = torch.clamp(sigma_pred.float(), min=eps)
    df = torch.clamp(df_pred.float(), min=eps)
    dist = torch.distributions.StudentT(df=df, loc=mu_pred.float(), scale=sigma)
    nll = -dist.log_prob(y)
    bad = ~torch.isfinite(nll)
    if bad.any():
        nll = torch.where(bad, torch.full_like(nll, float(invalid_penalty)), nll)
    if reduction == "mean":
        return nll.mean()
    if reduction == "sum":
        return nll.sum()
    return nll


# ---------------------------------------------------------------------------
# Laplace loss helpers (vendored from candi_loss.py)
# ---------------------------------------------------------------------------

class CustomLaplaceNLLLoss(nn.Module):
    """SmoothL1-based Laplace NLL."""

    def __init__(self, reduction="mean", beta=1.0, eps=1e-7):
        super().__init__()
        self.beta = beta
        self.eps = eps
        self.reduction = reduction
        self.log2 = math.log(2.0)
        self.smooth_l1 = nn.SmoothL1Loss(reduction="none", beta=beta)

    def forward(self, mu, target, log_b):
        log_b = torch.clamp(log_b, min=-10.0, max=10.0)
        b = torch.exp(log_b)
        nll = (log_b + self.log2) + (self.smooth_l1(mu, target) + 0.5 * self.beta) / (b + self.eps)
        if self.reduction == "mean":
            return nll.mean()
        if self.reduction == "sum":
            return nll.sum()
        return nll


class LaplaceNLLLoss(nn.Module):
    """Exact Laplace NLL via torch.distributions."""

    def __init__(self, reduction="mean", eps=1e-7):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, mu, target, log_b):
        b = torch.exp(log_b) + self.eps
        nll = -Laplace(loc=mu, scale=b).log_prob(target)
        if self.reduction == "mean":
            return nll.mean()
        if self.reduction == "sum":
            return nll.sum()
        return nll


# ---------------------------------------------------------------------------
# CANDI_LOSS (vendored from candi_loss.py)
# ---------------------------------------------------------------------------

class CANDI_LOSS(nn.Module):
    """
    Unified loss for CANDI training.

    Computes element-wise count/signal/peak losses, reduces per branch, and
    applies static obs/imp/head weights.  Optional features (assay-EMA balancing,
    FG/BG balancing, uncertainty weighting, R-stable count objective) are off by
    default and preserved here so the agent can toggle them.

    **Agent edit surface:** add extra terms to forward(), override _reduce_head_branch(),
    add a KL / ELBO term, change the signal distribution, etc.
    """

    def __init__(
        self,
        reduction="mean",
        count_weight=1.0,
        pval_weight=1.0,
        peak_weight=1.0,
        obs_weight=1.0,
        imp_weight=1.0,
        dist_type="gaussian",
        enable_assay_ema_balance=False,
        enable_hier_reduction=False,
        assay_ema_decay=0.99,
        assay_ema_eps=1e-6,
        assay_ema_warmup_steps=100,
        assay_ema_weight_min=0.02,
        assay_ema_weight_max=1.0,
        enable_fg_bg_balance=False,
        fg_weight=0.5,
        fg_min_fraction=0.02,
        enable_uncertainty_weighting=False,
        uncertainty_warmup_steps=100,
        uncertainty_init_logvar=0.0,
        enable_count_rstable_objective=False,
        count_rstable_eps=1e-6,
        count_rstable_ema_decay=0.99,
        count_rstable_warmup_steps=100,
        count_rstable_denom_min=1e-4,
        count_rstable_r_max=5.0,
        count_rstable_dispersion_min=1e-3,
        count_rstable_dispersion_max=1e4,
    ):
        super().__init__()
        self.reduction = reduction
        self.dist_type = dist_type

        if self.dist_type in ("laplace", "laplace_const"):
            self.signal_loss = LaplaceNLLLoss(reduction="none")
        elif self.dist_type == "mae":
            self.signal_loss = nn.L1Loss(reduction="none")
        elif self.dist_type == "studentst":
            self.signal_loss = None
        elif self.dist_type == "gamma":
            self.signal_loss = gamma_nll_loss
        elif self.dist_type == "mse":
            self.signal_loss = nn.MSELoss(reduction="none")
        else:
            self.signal_loss = nn.GaussianNLLLoss(reduction="none", full=True)

        self.nbin_nll = negative_binomial_loss
        self.bce_loss = nn.BCELoss(reduction="none")

        self.count_weight = count_weight
        self.pval_weight = pval_weight
        self.peak_weight = peak_weight
        self.obs_weight = obs_weight
        self.imp_weight = imp_weight

        self.enable_assay_ema_balance = bool(enable_assay_ema_balance)
        self.enable_hier_reduction = bool(enable_hier_reduction)
        self.enable_fg_bg_balance = bool(enable_fg_bg_balance)
        self.enable_uncertainty_weighting = bool(enable_uncertainty_weighting)
        self.enable_count_rstable_objective = bool(enable_count_rstable_objective)

        self.assay_ema_decay = float(assay_ema_decay)
        self.assay_ema_eps = float(assay_ema_eps)
        self.assay_ema_warmup_steps = int(assay_ema_warmup_steps)
        self.assay_ema_weight_min = float(assay_ema_weight_min)
        self.assay_ema_weight_max = float(assay_ema_weight_max)

        self.fg_weight = float(fg_weight)
        self.fg_min_fraction = float(fg_min_fraction)

        self.uncertainty_warmup_steps = int(uncertainty_warmup_steps)
        self.logvar_count = nn.Parameter(torch.tensor(float(uncertainty_init_logvar)))
        self.logvar_pval = nn.Parameter(torch.tensor(float(uncertainty_init_logvar)))
        self.logvar_peak = nn.Parameter(torch.tensor(float(uncertainty_init_logvar)))

        self.count_rstable_eps = float(count_rstable_eps)
        self.count_rstable_ema_decay = float(count_rstable_ema_decay)
        self.count_rstable_warmup_steps = int(count_rstable_warmup_steps)
        self.count_rstable_denom_min = float(count_rstable_denom_min)
        self.count_rstable_r_max = float(count_rstable_r_max)
        self.count_rstable_dispersion_min = float(count_rstable_dispersion_min)
        self.count_rstable_dispersion_max = float(count_rstable_dispersion_max)

        self.register_buffer("assay_freq_ema", torch.empty(0))
        self.register_buffer("assay_freq_initialized", torch.tensor(False))
        self.register_buffer("count_rstable_ema_mu0", torch.empty(0))
        self.register_buffer("count_rstable_ema_r0", torch.empty(0))
        self.register_buffer("count_rstable_initialized", torch.tensor(False))
        self.last_debug_stats: Dict = {}

    def has_uncertainty_params(self) -> bool:
        return self.enable_uncertainty_weighting

    def get_debug_stats(self) -> Dict:
        return self.last_debug_stats

    def _maybe_init_ema(self, num_assays: int, device: torch.device) -> None:
        if self.assay_freq_initialized.item() and self.assay_freq_ema.shape[-1] == num_assays:
            return
        self.assay_freq_ema = torch.ones((3, 2, num_assays), device=device, dtype=torch.float32)
        self.assay_freq_initialized = torch.tensor(True, device=device)

    def _maybe_init_count_rstable(self, num_assays: int, device: torch.device) -> None:
        if self.count_rstable_initialized.item() and self.count_rstable_ema_mu0.shape[-1] == num_assays:
            return
        self.count_rstable_ema_mu0 = torch.ones((2, num_assays), device=device, dtype=torch.float32)
        self.count_rstable_ema_r0 = torch.full((2, num_assays), 10.0, device=device, dtype=torch.float32)
        self.count_rstable_initialized = torch.tensor(True, device=device)

    def _compute_signal_elementwise(self, mu_pred, scale_pred, df_pred, target):
        if self.dist_type == "studentst":
            if df_pred is None:
                raise ValueError("Student-t loss requires df_pred.")
            return students_t_nll_loss(target, mu_pred, scale_pred, df_pred, reduction="none")
        if self.dist_type == "gamma":
            return self.signal_loss(target, mu_pred, scale_pred, reduction="none")
        if self.dist_type in ("mse", "mae"):
            return self.signal_loss(mu_pred, target)
        return self.signal_loss(mu_pred, target, scale_pred)

    def _update_ema_and_get_weights(
        self, valid_map: torch.Tensor, head_idx: int, branch_idx: int, global_step: int
    ) -> torch.Tensor:
        per_assay_avail = valid_map.any(dim=1).any(dim=0).float()
        if self.enable_assay_ema_balance:
            with torch.no_grad():
                self.assay_freq_ema[head_idx, branch_idx, :] = (
                    self.assay_ema_decay * self.assay_freq_ema[head_idx, branch_idx, :]
                    + (1.0 - self.assay_ema_decay) * per_assay_avail
                )
        if (not self.enable_assay_ema_balance) or (global_step < self.assay_ema_warmup_steps):
            return torch.ones_like(per_assay_avail)
        inv = 1.0 / (self.assay_freq_ema[head_idx, branch_idx, :] + self.assay_ema_eps)
        inv = inv / torch.clamp(inv.max(), min=1e-8)
        return torch.clamp(inv, min=self.assay_ema_weight_min, max=self.assay_ema_weight_max)

    def _reduce_head_branch(
        self,
        elem_loss: torch.Tensor,
        valid_map: torch.Tensor,
        head_idx: int,
        branch_idx: int,
        global_step: int,
        peak_gt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if elem_loss.numel() == 0 or valid_map.numel() == 0:
            return elem_loss.new_tensor(0.0)
        if not self.enable_hier_reduction:
            vals = elem_loss[valid_map]
            if vals.numel() == 0:
                return elem_loss.new_tensor(0.0)
            return vals.sum() if self.reduction == "sum" else vals.mean()

        weights = self._update_ema_and_get_weights(valid_map, head_idx, branch_idx, global_step)
        per_assay_losses, per_assay_weights = [], []
        fg_fb_hits = fg_fb_total = 0

        for a in range(valid_map.shape[-1]):
            amask = valid_map[:, :, a]
            if not amask.any():
                continue
            if self.enable_fg_bg_balance and head_idx in (0, 1) and peak_gt is not None:
                peak_a = peak_gt[:, :, a]
                peak_valid = (peak_a == 0) | (peak_a == 1)
                joint = amask & peak_valid
                if joint.any():
                    fg_mask = joint & (peak_a == 1)
                    bg_mask = joint & (peak_a == 0)
                    fg_frac = float(fg_mask.sum().item()) / float(max(int(joint.sum().item()), 1))
                    if fg_mask.any() and bg_mask.any() and fg_frac >= self.fg_min_fraction:
                        assay_loss = (
                            self.fg_weight * elem_loss[:, :, a][fg_mask].mean()
                            + (1.0 - self.fg_weight) * elem_loss[:, :, a][bg_mask].mean()
                        )
                        fg_fb_hits += 1
                    else:
                        assay_loss = elem_loss[:, :, a][amask].mean()
                    fg_fb_total += 1
                else:
                    assay_loss = elem_loss[:, :, a][amask].mean()
            else:
                assay_loss = elem_loss[:, :, a][amask].mean()
            per_assay_losses.append(assay_loss)
            per_assay_weights.append(weights[a])

        if not per_assay_losses:
            return elem_loss.new_tensor(0.0)
        loss_vec = torch.stack(per_assay_losses)
        weight_vec = torch.stack(per_assay_weights).float()
        reduced = (loss_vec * weight_vec).sum() / torch.clamp(weight_vec.sum(), min=1e-8)
        self.last_debug_stats[f"fgbg_used_h{head_idx}_b{branch_idx}"] = int(fg_fb_hits)
        self.last_debug_stats[f"fgbg_total_h{head_idx}_b{branch_idx}"] = int(fg_fb_total)
        return reduced

    def _apply_uncertainty(self, obs_loss, imp_loss, logvar, global_step: int):
        if (not self.enable_uncertainty_weighting) or (global_step < self.uncertainty_warmup_steps):
            return obs_loss, imp_loss
        scale = torch.exp(-logvar)
        return scale * obs_loss + 0.5 * logvar, scale * imp_loss + 0.5 * logvar

    def _count_branch_rstable(
        self,
        elem_count: torch.Tensor,
        valid_map: torch.Tensor,
        true_count: torch.Tensor,
        true_peak: Optional[torch.Tensor],
        branch_idx: int,
        global_step: int,
    ) -> torch.Tensor:
        if (not self.enable_count_rstable_objective) or (global_step < self.count_rstable_warmup_steps):
            return self._reduce_head_branch(elem_count, valid_map, 0, branch_idx, global_step, peak_gt=true_peak)

        weights = self._update_ema_and_get_weights(valid_map, 0, branch_idx, global_step)
        per_assay_r, per_assay_w = [], []
        l_model_vals, l_null_vals, l_oracle_vals = [], [], []
        denom_min_seen, denom_clamp_hits, used_assays = float("inf"), 0, 0

        for a in range(valid_map.shape[-1]):
            amask = valid_map[:, :, a]
            if not amask.any():
                continue
            y = true_count[:, :, a][amask].float()
            l_model = elem_count[:, :, a][amask].float().mean()
            batch_mu = torch.clamp(y.mean(), min=0.0)
            batch_var = torch.var(y, unbiased=False) if y.numel() > 1 else batch_mu + 1.0
            batch_r = torch.clamp(
                (batch_mu * batch_mu) / torch.clamp(batch_var - batch_mu, min=1e-6),
                min=self.count_rstable_dispersion_min,
                max=self.count_rstable_dispersion_max,
            )
            with torch.no_grad():
                self.count_rstable_ema_mu0[branch_idx, a] = (
                    self.count_rstable_ema_decay * self.count_rstable_ema_mu0[branch_idx, a]
                    + (1.0 - self.count_rstable_ema_decay) * batch_mu
                )
                self.count_rstable_ema_r0[branch_idx, a] = (
                    self.count_rstable_ema_decay * self.count_rstable_ema_r0[branch_idx, a]
                    + (1.0 - self.count_rstable_ema_decay) * batch_r
                )
            mu0 = torch.clamp(self.count_rstable_ema_mu0[branch_idx, a], min=0.0)
            r0 = torch.clamp(self.count_rstable_ema_r0[branch_idx, a],
                             min=self.count_rstable_dispersion_min, max=self.count_rstable_dispersion_max)
            p_null = torch.clamp(r0 / (r0 + mu0 + 1e-8), min=1e-6, max=1.0 - 1e-6)
            l_null = self.nbin_nll(y, torch.full_like(y, r0), torch.full_like(y, p_null)).mean()
            p_oracle = torch.clamp(r0 / (r0 + y + 1e-8), min=1e-6, max=1.0 - 1e-6)
            l_oracle = self.nbin_nll(y, torch.full_like(y, r0), p_oracle).mean()
            denom_raw = l_null - l_oracle + self.count_rstable_eps
            denom_min_seen = min(denom_min_seen, float(torch.abs(denom_raw).item()))
            denom_sign = torch.where(denom_raw >= 0, torch.ones_like(denom_raw), -torch.ones_like(denom_raw))
            denom_safe = denom_sign * torch.clamp(torch.abs(denom_raw), min=self.count_rstable_denom_min)
            if float(torch.abs(denom_raw).item()) < self.count_rstable_denom_min:
                denom_clamp_hits += 1
            r_stable = torch.clamp((l_model - l_oracle) / denom_safe,
                                   min=-self.count_rstable_r_max, max=self.count_rstable_r_max)
            per_assay_r.append(r_stable)
            per_assay_w.append(weights[a])
            l_model_vals.append(float(l_model.item()))
            l_null_vals.append(float(l_null.item()))
            l_oracle_vals.append(float(l_oracle.item()))
            used_assays += 1

        if not per_assay_r:
            return elem_count.new_tensor(0.0)
        r_vec = torch.stack(per_assay_r)
        if self.enable_hier_reduction:
            w_vec = torch.stack(per_assay_w).float()
            reduced = (r_vec * w_vec).sum() / torch.clamp(w_vec.sum(), min=1e-8)
        else:
            reduced = r_vec.mean()
        bname = "obs" if branch_idx == 0 else "imp"
        self.last_debug_stats[f"rstable_{bname}_assays"] = int(used_assays)
        self.last_debug_stats[f"rstable_{bname}_lmodel_mean"] = float(np.mean(l_model_vals)) if l_model_vals else float("nan")
        self.last_debug_stats[f"rstable_{bname}_denom_min_abs"] = float(denom_min_seen if denom_min_seen != float("inf") else 0.0)
        self.last_debug_stats[f"rstable_{bname}_denom_clamp_hits"] = int(denom_clamp_hits)
        return reduced

    def forward(
        self,
        p_pred,
        n_pred,
        mu_pred,
        scale_pred,
        df_pred,
        peak_pred,
        true_count,
        true_pval,
        true_peak,
        obs_map,
        masked_map,
        obs_map_signal=None,
        masked_map_signal=None,
        global_step: int = 0,
    ):
        if obs_map_signal is None:
            obs_map_signal = obs_map
        if masked_map_signal is None:
            masked_map_signal = masked_map

        self.last_debug_stats = {}
        device = true_count.device
        num_assays = true_count.shape[-1]
        self._maybe_init_ema(num_assays, device)
        self._maybe_init_count_rstable(num_assays, device)

        elem_count = self.nbin_nll(true_count, n_pred, p_pred)
        elem_signal = self._compute_signal_elementwise(mu_pred, scale_pred, df_pred, true_pval).float()

        peak_valid = (true_peak == 0) | (true_peak == 1)
        with torch.amp.autocast("cuda", enabled=False):
            peak_pred_safe = torch.clamp(peak_pred.float(), min=1e-6, max=1.0 - 1e-6)
            peak_target_safe = torch.where(peak_valid, true_peak, torch.zeros_like(true_peak)).float()
            elem_peak = self.bce_loss(peak_pred_safe, peak_target_safe).float()
            elem_peak = torch.where(peak_valid, elem_peak, torch.zeros_like(elem_peak))

        observed_count_loss = self._count_branch_rstable(elem_count, obs_map, true_count, true_peak, 0, global_step)
        imputed_count_loss = self._count_branch_rstable(elem_count, masked_map, true_count, true_peak, 1, global_step)
        observed_pval_loss = self._reduce_head_branch(elem_signal, obs_map_signal, 1, 0, global_step, true_peak)
        imputed_pval_loss = self._reduce_head_branch(elem_signal, masked_map_signal, 1, 1, global_step, true_peak)
        obs_map_peak = obs_map_signal & peak_valid
        masked_map_peak = masked_map_signal & peak_valid
        observed_peak_loss = self._reduce_head_branch(elem_peak, obs_map_peak, 2, 0, global_step)
        imputed_peak_loss = self._reduce_head_branch(elem_peak, masked_map_peak, 2, 1, global_step)

        observed_count_loss = self.count_weight * observed_count_loss
        imputed_count_loss = self.count_weight * imputed_count_loss
        observed_pval_loss = self.pval_weight * observed_pval_loss
        imputed_pval_loss = self.pval_weight * imputed_pval_loss
        observed_peak_loss = self.peak_weight * observed_peak_loss
        imputed_peak_loss = self.peak_weight * imputed_peak_loss

        observed_count_loss, imputed_count_loss = self._apply_uncertainty(
            observed_count_loss, imputed_count_loss, self.logvar_count, global_step)
        observed_pval_loss, imputed_pval_loss = self._apply_uncertainty(
            observed_pval_loss, imputed_pval_loss, self.logvar_pval, global_step)
        observed_peak_loss, imputed_peak_loss = self._apply_uncertainty(
            observed_peak_loss, imputed_peak_loss, self.logvar_peak, global_step)

        observed_count_loss = self.obs_weight * observed_count_loss
        observed_pval_loss = self.obs_weight * observed_pval_loss
        observed_peak_loss = self.obs_weight * observed_peak_loss
        imputed_count_loss = self.imp_weight * imputed_count_loss
        imputed_pval_loss = self.imp_weight * imputed_pval_loss
        imputed_peak_loss = self.imp_weight * imputed_peak_loss

        if self.enable_uncertainty_weighting:
            self.last_debug_stats["logvar_count"] = float(self.logvar_count.detach().item())
            self.last_debug_stats["logvar_pval"] = float(self.logvar_pval.detach().item())
            self.last_debug_stats["logvar_peak"] = float(self.logvar_peak.detach().item())

        return (
            observed_count_loss,
            imputed_count_loss,
            observed_pval_loss,
            imputed_pval_loss,
            observed_peak_loss,
            imputed_peak_loss,
        )


# ---------------------------------------------------------------------------
# SandboxCompositeLoss (vendored from sandbox/losses.py)
# ---------------------------------------------------------------------------

class SandboxCompositeLoss(nn.Module):
    """
    Wraps CANDI_LOSS and adds the imp-fallback-to-observed edge-case fix.

    Agent edit surface: override forward_with_terms() to add extra terms
    (e.g. KL / ELBO on the encoder latent z), add regularization penalties,
    return additional stats keys, etc.  The `terms` dict returned by
    forward_with_terms is forwarded verbatim to agent_step → keep_rule → TSV.
    """

    def __init__(self, cand: CANDI_LOSS, consistency_weight: float = 0.0, aux_mse_imp_weight: float = 0.0, aux_mse_obs_weight: float = 0.0):
        super().__init__()
        self.cand = cand
        self._consistency_weight = float(consistency_weight)
        self._aux_mse_imp_weight = float(aux_mse_imp_weight)
        self._aux_mse_obs_weight = float(aux_mse_obs_weight)

    @staticmethod
    def _safe_unweight(weighted: torch.Tensor, weight: float) -> torch.Tensor:
        if abs(float(weight)) < 1e-12:
            return torch.full_like(weighted, float("nan"))
        return weighted / float(weight)

    def _compute_terms(
        self,
        output_p,
        output_n,
        output_mu,
        output_var,
        output_df,
        output_peak,
        y_data,
        y_pval,
        y_peaks,
        observed_map,
        masked_map,
        signal_observed_map,
        signal_masked_map,
        *,
        global_step: int = 0,
        fallback_imp_to_observed_when_no_masked: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if fallback_imp_to_observed_when_no_masked:
            has_masked = bool(masked_map.any().item())
            mm = masked_map if has_masked else observed_map
            smm = signal_masked_map if has_masked else signal_observed_map
        else:
            mm = masked_map
            smm = signal_masked_map

        count_obs_w, count_imp_w, pval_obs_w, pval_imp_w, peak_obs_w, peak_imp_w = self.cand(
            output_p, output_n, output_mu, output_var, output_df, output_peak,
            y_data, y_pval, y_peaks,
            observed_map, mm, signal_observed_map, smm,
            global_step=global_step,
        )

        c = self.cand
        w_count_obs = float(c.count_weight) * float(c.obs_weight)
        w_count_imp = float(c.count_weight) * float(c.imp_weight)
        w_pval_obs = float(c.pval_weight) * float(c.obs_weight)
        w_pval_imp = float(c.pval_weight) * float(c.imp_weight)
        w_peak_obs = float(c.peak_weight) * float(c.obs_weight)
        w_peak_imp = float(c.peak_weight) * float(c.imp_weight)

        terms: Dict[str, torch.Tensor] = {
            "count_obs_weighted": count_obs_w,
            "count_imp_weighted": count_imp_w,
            "pval_obs_weighted": pval_obs_w,
            "pval_imp_weighted": pval_imp_w,
            "peak_obs_weighted": peak_obs_w,
            "peak_imp_weighted": peak_imp_w,
            "count_obs_raw": self._safe_unweight(count_obs_w, w_count_obs),
            "count_imp_raw": self._safe_unweight(count_imp_w, w_count_imp),
            "pval_obs_raw": self._safe_unweight(pval_obs_w, w_pval_obs),
            "pval_imp_raw": self._safe_unweight(pval_imp_w, w_pval_imp),
            "peak_obs_raw": self._safe_unweight(peak_obs_w, w_peak_obs),
            "peak_imp_raw": self._safe_unweight(peak_imp_w, w_peak_imp),
        }
        terms["total_weighted"] = (
            terms["count_obs_weighted"] + terms["count_imp_weighted"]
            + terms["pval_obs_weighted"] + terms["pval_imp_weighted"]
            + terms["peak_obs_weighted"] + terms["peak_imp_weighted"]
        )
        # DCR soft penalty: bias alpha toward DCR≥3.0 without severing joint gradients
        if _DCR_SLOPE_REF is not None and _DCR_PENALTY_WEIGHT > 0:
            alpha = torch.exp(_DCR_SLOPE_REF)
            dcr = torch.pow(2.0, 2.0 * alpha)
            dcr_penalty = F.relu(3.015 - dcr).pow(2) * _DCR_PENALTY_WEIGHT
            terms["total_weighted"] = terms["total_weighted"] + dcr_penalty
        # Auxiliary log1p MSE on imputed positions: smooth direct gradient alongside NB NLL
        if self._aux_mse_imp_weight > 0.0 and mm.any():
            aux_mse = F.mse_loss(
                torch.log1p(output_mu[mm]),
                torch.log1p(y_data[mm].float()),
            )
            terms["total_weighted"] = terms["total_weighted"] + aux_mse * self._aux_mse_imp_weight
            terms["aux_mse_imp"] = aux_mse
        # Auxiliary log1p MSE on observed positions: smooth direct gradient for denoising alongside NB NLL
        if self._aux_mse_obs_weight > 0.0 and observed_map.any():
            aux_mse_obs = F.mse_loss(
                torch.log1p(output_mu[observed_map]),
                torch.log1p(y_data[observed_map].float()),
            )
            terms["total_weighted"] = terms["total_weighted"] + aux_mse_obs * self._aux_mse_obs_weight
            terms["aux_mse_obs"] = aux_mse_obs
        # Cross-assay consistency: imputed track means ≈ observed track means at same locus
        if self._consistency_weight > 0.0:
            obs_f = observed_map.float()
            imp_f = mm.float()
            obs_mean = (output_mu * obs_f).sum(-1) / obs_f.sum(-1).clamp(min=1.0)
            imp_mean = (output_mu * imp_f).sum(-1) / imp_f.sum(-1).clamp(min=1.0)
            valid = observed_map.any(-1) & mm.any(-1)
            if valid.any():
                cons = F.mse_loss(
                    torch.log1p(imp_mean[valid]),
                    torch.log1p(obs_mean[valid]).detach(),
                )
                terms["total_weighted"] = terms["total_weighted"] + cons * self._consistency_weight
                terms["consistency_loss"] = cons
        return terms

    @staticmethod
    def _stats_from_terms(terms: Dict[str, torch.Tensor]) -> Dict[str, float]:
        stats: Dict[str, float] = {
            "loss_total": float(terms["total_weighted"].detach().item()),
            "loss_total_weighted": float(terms["total_weighted"].detach().item()),
            "loss_branch_count_obs": float(terms["count_obs_weighted"].detach().item()),
            "loss_branch_count_imp": float(terms["count_imp_weighted"].detach().item()),
            "loss_branch_pval_obs": float(terms["pval_obs_weighted"].detach().item()),
            "loss_branch_pval_imp": float(terms["pval_imp_weighted"].detach().item()),
            "loss_branch_peak_obs": float(terms["peak_obs_weighted"].detach().item()),
            "loss_branch_peak_imp": float(terms["peak_imp_weighted"].detach().item()),
            "loss_branch_count_obs_raw": float(terms["count_obs_raw"].detach().item()),
            "loss_branch_count_imp_raw": float(terms["count_imp_raw"].detach().item()),
            "loss_branch_pval_obs_raw": float(terms["pval_obs_raw"].detach().item()),
            "loss_branch_pval_imp_raw": float(terms["pval_imp_raw"].detach().item()),
            "loss_branch_peak_obs_raw": float(terms["peak_obs_raw"].detach().item()),
            "loss_branch_peak_imp_raw": float(terms["peak_imp_raw"].detach().item()),
        }
        # count_imp_loss / count_obs_loss are the keys prepare.py reads.
        stats["count_imp_loss"] = stats["loss_branch_count_imp"]
        stats["count_obs_loss"] = stats["loss_branch_count_obs"]
        return stats

    def forward_with_terms(
        self,
        output_p,
        output_n,
        output_mu,
        output_var,
        output_df,
        output_peak,
        y_data,
        y_pval,
        y_peaks,
        observed_map,
        masked_map,
        signal_observed_map,
        signal_masked_map,
        *,
        global_step: int = 0,
        fallback_imp_to_observed_when_no_masked: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
        terms = self._compute_terms(
            output_p, output_n, output_mu, output_var, output_df, output_peak,
            y_data, y_pval, y_peaks,
            observed_map, masked_map, signal_observed_map, signal_masked_map,
            global_step=global_step,
            fallback_imp_to_observed_when_no_masked=fallback_imp_to_observed_when_no_masked,
        )
        stats = self._stats_from_terms(terms)
        return terms["total_weighted"], stats, terms

    @property
    def loss_weights(self) -> Dict[str, float]:
        c = self.cand
        return {
            "count_weight": float(c.count_weight),
            "pval_weight": float(c.pval_weight),
            "peak_weight": float(c.peak_weight),
            "obs_weight": float(c.obs_weight),
            "imp_weight": float(c.imp_weight),
        }

    def forward(
        self,
        output_p,
        output_n,
        output_mu,
        output_var,
        output_df,
        output_peak,
        y_data,
        y_pval,
        y_peaks,
        observed_map,
        masked_map,
        signal_observed_map,
        signal_masked_map,
        *,
        global_step: int = 0,
        fallback_imp_to_observed_when_no_masked: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total, stats, _ = self.forward_with_terms(
            output_p, output_n, output_mu, output_var, output_df, output_peak,
            y_data, y_pval, y_peaks,
            observed_map, masked_map, signal_observed_map, signal_masked_map,
            global_step=global_step,
            fallback_imp_to_observed_when_no_masked=fallback_imp_to_observed_when_no_masked,
        )
        return total, stats


# ---------------------------------------------------------------------------
# Factory (called by prepare.py — frozen harness)
# ---------------------------------------------------------------------------

def build_v2_loss(cfg: CANDIv2Config) -> SandboxCompositeLoss:
    """Build the loss module for CANDI v2.

    Agent: to add a new term (e.g. KL penalty on z), subclass SandboxCompositeLoss
    and override forward_with_terms(), or modify _compute_terms() directly here.
    """
    lw = cfg.training.loss_weights
    heads = str(cfg.decoder.heads)

    pval_weight = float(lw.pval_weight)
    count_weight = float(lw.count_weight)
    peak_weight = float(lw.peak_weight)

    if heads != "all":
        pval_weight = 0.0
    if heads == "peak_only":
        count_weight = 0.0
    if heads == "count_only":
        peak_weight = 0.0

    cand = CANDI_LOSS(
        dist_type="gaussian",
        count_weight=count_weight,
        pval_weight=pval_weight,
        peak_weight=peak_weight,
        obs_weight=float(lw.obs_weight),
        imp_weight=float(lw.imp_weight),
    )
    return SandboxCompositeLoss(
        cand,
        consistency_weight=float(getattr(cfg.decoder, 'consistency_weight', 0.0)),
        aux_mse_imp_weight=float(getattr(cfg.decoder, 'aux_mse_imp_weight', 0.0)),
        aux_mse_obs_weight=float(getattr(cfg.decoder, 'aux_mse_obs_weight', 0.0)),
    )
