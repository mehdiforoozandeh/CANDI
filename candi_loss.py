import math
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.distributions import Laplace

from _utils import negative_binomial_loss, students_t_nll_loss, gamma_nll_loss


class CustomLaplaceNLLLoss(nn.Module):
    """
    Combines Gaussian stability at the mode with Laplacian robustness in the tails.

    Formula: NLL = log(2b) + |x - mu| / b
    Uses SmoothL1 with offset correction to ensure exact L1 behavior in tails.
    """
    def __init__(self, reduction='mean', beta=1.0, eps=1e-7):
        super().__init__()
        self.beta = beta
        self.eps = eps
        self.reduction = reduction
        self.log2 = math.log(2.0)
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none', beta=beta)

    def forward(self, mu, target, log_b):
        log_b = torch.clamp(log_b, min=-10.0, max=10.0)
        b = torch.exp(log_b)
        numerator = self.smooth_l1(mu, target) + 0.5 * self.beta
        nll = (log_b + self.log2) + (numerator / (b + self.eps))

        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        return nll


class LaplaceNLLLoss(nn.Module):
    def __init__(self, reduction='mean', eps=1e-7):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, mu, target, log_b):
        b = torch.exp(log_b) + self.eps
        dist = Laplace(loc=mu, scale=b)
        nll = -dist.log_prob(target)

        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        else:
            return nll


class CANDI_LOSS(nn.Module):
    """
    Unified loss module for CANDI training.

    Key ideas:
    - Computes element-wise losses for count/signal/peak, then reduces per branch.
    - Supports hierarchical per-assay reduction with optional assay-frequency balancing.
    - Supports optional count-head R_stable objective replacement.
    - Handles peak supervision safely by masking invalid GT labels before BCE reduction.
    """
    def __init__(
        self,
        reduction='mean',
        count_weight=1.0,
        pval_weight=1.0,
        peak_weight=1.0,
        obs_weight=1.0,
        imp_weight=1.0,
        dist_type='gaussian',
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
        super(CANDI_LOSS, self).__init__()
        self.reduction = reduction
        self.dist_type = dist_type

        # Signal loss setup (element-wise to enable custom reductions).
        if self.dist_type in ['laplace', 'laplace_const']:
            self.signal_loss = LaplaceNLLLoss(reduction="none")
        elif self.dist_type == 'mae':
            self.signal_loss = nn.L1Loss(reduction="none")
        elif self.dist_type == 'studentst':
            self.signal_loss = None
        elif self.dist_type == 'gamma':
            self.signal_loss = gamma_nll_loss
        elif self.dist_type == 'mse':
            self.signal_loss = nn.MSELoss(reduction="none")
        else:
            self.signal_loss = nn.GaussianNLLLoss(reduction="none", full=True)

        self.nbin_nll = negative_binomial_loss
        self.bce_loss = nn.BCELoss(reduction="none")

        # Static priors/fallback weights.
        self.count_weight = count_weight
        self.pval_weight = pval_weight
        self.peak_weight = peak_weight
        self.obs_weight = obs_weight
        self.imp_weight = imp_weight

        # Feature toggles.
        self.enable_assay_ema_balance = bool(enable_assay_ema_balance)
        self.enable_hier_reduction = bool(enable_hier_reduction)
        self.enable_fg_bg_balance = bool(enable_fg_bg_balance)
        self.enable_uncertainty_weighting = bool(enable_uncertainty_weighting)
        self.enable_count_rstable_objective = bool(enable_count_rstable_objective)

        # EMA assay-frequency balancing config.
        self.assay_ema_decay = float(assay_ema_decay)
        self.assay_ema_eps = float(assay_ema_eps)
        self.assay_ema_warmup_steps = int(assay_ema_warmup_steps)
        self.assay_ema_weight_min = float(assay_ema_weight_min)
        self.assay_ema_weight_max = float(assay_ema_weight_max)

        # FG/BG config (count+signal).
        self.fg_weight = float(fg_weight)
        self.fg_min_fraction = float(fg_min_fraction)

        # Uncertainty weighting config.
        self.uncertainty_warmup_steps = int(uncertainty_warmup_steps)
        self.logvar_count = nn.Parameter(torch.tensor(float(uncertainty_init_logvar)))
        self.logvar_pval = nn.Parameter(torch.tensor(float(uncertainty_init_logvar)))
        self.logvar_peak = nn.Parameter(torch.tensor(float(uncertainty_init_logvar)))

        # Count R_stable objective config.
        self.count_rstable_eps = float(count_rstable_eps)
        self.count_rstable_ema_decay = float(count_rstable_ema_decay)
        self.count_rstable_warmup_steps = int(count_rstable_warmup_steps)
        self.count_rstable_denom_min = float(count_rstable_denom_min)
        self.count_rstable_r_max = float(count_rstable_r_max)
        self.count_rstable_dispersion_min = float(count_rstable_dispersion_min)
        self.count_rstable_dispersion_max = float(count_rstable_dispersion_max)

        # Stateful buffers.
        self.register_buffer("assay_freq_ema", torch.empty(0))
        self.register_buffer("assay_freq_initialized", torch.tensor(False))
        self.register_buffer("count_rstable_ema_mu0", torch.empty(0))
        self.register_buffer("count_rstable_ema_r0", torch.empty(0))
        self.register_buffer("count_rstable_initialized", torch.tensor(False))
        self.last_debug_stats = {}

    def has_uncertainty_params(self) -> bool:
        return self.enable_uncertainty_weighting

    def get_debug_stats(self) -> Dict:
        return self.last_debug_stats

    def _maybe_init_ema(self, num_assays: int, device: torch.device):
        # Lazy-init so we can infer assay dimension at runtime.
        if self.assay_freq_initialized.item() and self.assay_freq_ema.shape[-1] == num_assays:
            return
        init = torch.ones((3, 2, num_assays), device=device, dtype=torch.float32)
        self.assay_freq_ema = init
        self.assay_freq_initialized = torch.tensor(True, device=device)

    def _maybe_init_count_rstable(self, num_assays: int, device: torch.device):
        # Per (branch, assay) baseline state for count null/oracle builders.
        if self.count_rstable_initialized.item() and self.count_rstable_ema_mu0.shape[-1] == num_assays:
            return
        self.count_rstable_ema_mu0 = torch.ones((2, num_assays), device=device, dtype=torch.float32)
        self.count_rstable_ema_r0 = torch.full((2, num_assays), 10.0, device=device, dtype=torch.float32)
        self.count_rstable_initialized = torch.tensor(True, device=device)

    def _compute_signal_elementwise(self, mu_pred, scale_pred, df_pred, target):
        if self.dist_type == 'studentst':
            if df_pred is None:
                raise ValueError("Student-t loss requires df_pred.")
            return students_t_nll_loss(target, mu_pred, scale_pred, df_pred, reduction="none")
        if self.dist_type == 'gamma':
            return self.signal_loss(target, mu_pred, scale_pred, reduction="none")
        if self.dist_type in ['mse', 'mae']:
            return self.signal_loss(mu_pred, target)
        return self.signal_loss(mu_pred, target, scale_pred)

    def _update_ema_and_get_weights(self, valid_map: torch.Tensor, head_idx: int, branch_idx: int, global_step: int) -> torch.Tensor:
        # Availability proxy: assay considered available if any supervised token exists in the sample/window.
        per_assay_avail = valid_map.any(dim=1).any(dim=0).float()
        if self.enable_assay_ema_balance:
            with torch.no_grad():
                self.assay_freq_ema[head_idx, branch_idx, :] = (
                    self.assay_ema_decay * self.assay_freq_ema[head_idx, branch_idx, :]
                    + (1.0 - self.assay_ema_decay) * per_assay_avail
                )
        if (not self.enable_assay_ema_balance) or (global_step < self.assay_ema_warmup_steps):
            return torch.ones_like(per_assay_avail)

        # Inverse-frequency weighting with bounded dynamic range.
        inv = 1.0 / (self.assay_freq_ema[head_idx, branch_idx, :] + self.assay_ema_eps)
        inv = inv / torch.clamp(inv.max(), min=1e-8)
        inv = torch.clamp(inv, min=self.assay_ema_weight_min, max=self.assay_ema_weight_max)
        return inv

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
            if self.reduction == "sum":
                return vals.sum()
            return vals.mean()

        weights = self._update_ema_and_get_weights(valid_map, head_idx, branch_idx, global_step)
        per_assay_losses, per_assay_weights = [], []
        fg_fb_hits, fg_fb_total = 0, 0

        for a in range(valid_map.shape[-1]):
            amask = valid_map[:, :, a]
            if not amask.any():
                continue

            # Optional FG/BG split for count and signal only (peak head stays standard BCE).
            if self.enable_fg_bg_balance and head_idx in [0, 1] and peak_gt is not None:
                peak_a = peak_gt[:, :, a]
                peak_valid = (peak_a == 0) | (peak_a == 1)
                joint = amask & peak_valid
                if joint.any():
                    fg_mask = joint & (peak_a == 1)
                    bg_mask = joint & (peak_a == 0)
                    fg_frac = float(fg_mask.sum().item()) / float(max(int(joint.sum().item()), 1))
                    if fg_mask.any() and bg_mask.any() and fg_frac >= self.fg_min_fraction:
                        fg_loss = elem_loss[:, :, a][fg_mask].mean()
                        bg_loss = elem_loss[:, :, a][bg_mask].mean()
                        assay_loss = self.fg_weight * fg_loss + (1.0 - self.fg_weight) * bg_loss
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

        if len(per_assay_losses) == 0:
            return elem_loss.new_tensor(0.0)

        loss_vec = torch.stack(per_assay_losses)
        weight_vec = torch.stack(per_assay_weights).float()
        reduced = (loss_vec * weight_vec).sum() / torch.clamp(weight_vec.sum(), min=1e-8)

        self.last_debug_stats[f"fgbg_used_h{head_idx}_b{branch_idx}"] = int(fg_fb_hits)
        self.last_debug_stats[f"fgbg_total_h{head_idx}_b{branch_idx}"] = int(fg_fb_total)
        self.last_debug_stats[f"assay_w_min_h{head_idx}_b{branch_idx}"] = float(weight_vec.min().item())
        self.last_debug_stats[f"assay_w_max_h{head_idx}_b{branch_idx}"] = float(weight_vec.max().item())
        return reduced

    def _apply_uncertainty(self, obs_loss, imp_loss, logvar, global_step: int):
        if (not self.enable_uncertainty_weighting) or (global_step < self.uncertainty_warmup_steps):
            return obs_loss, imp_loss
        scale = torch.exp(-logvar)
        return scale * obs_loss + 0.5 * logvar, scale * imp_loss + 0.5 * logvar

    def _count_branch_legacy_reduced(self, elem_count: torch.Tensor, valid_map: torch.Tensor, branch_idx: int, global_step: int, peak_gt: Optional[torch.Tensor]) -> torch.Tensor:
        return self._reduce_head_branch(elem_count, valid_map, head_idx=0, branch_idx=branch_idx, global_step=global_step, peak_gt=peak_gt)

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
            return self._count_branch_legacy_reduced(elem_count, valid_map, branch_idx, global_step, peak_gt=true_peak)

        # Assay weights follow the same branch/head weighting path as standard hierarchical reduction.
        weights = self._update_ema_and_get_weights(valid_map, head_idx=0, branch_idx=branch_idx, global_step=global_step)
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
            mom_denom = torch.clamp(batch_var - batch_mu, min=1e-6)
            batch_r = torch.clamp((batch_mu * batch_mu) / mom_denom, min=self.count_rstable_dispersion_min, max=self.count_rstable_dispersion_max)

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
            r0 = torch.clamp(self.count_rstable_ema_r0[branch_idx, a], min=self.count_rstable_dispersion_min, max=self.count_rstable_dispersion_max)

            n_null = torch.full_like(y, r0)
            p_null = torch.full_like(y, torch.clamp(r0 / (r0 + mu0 + 1e-8), min=1e-6, max=1.0 - 1e-6))
            l_null = self.nbin_nll(y, n_null, p_null).mean()

            n_oracle = torch.full_like(y, r0)
            p_oracle = torch.clamp(r0 / (r0 + y + 1e-8), min=1e-6, max=1.0 - 1e-6)
            l_oracle = self.nbin_nll(y, n_oracle, p_oracle).mean()

            # Stable ratio denominator: keep finite even when null and oracle become numerically close.
            denom_raw = (l_null - l_oracle + self.count_rstable_eps)
            denom_min_seen = min(denom_min_seen, float(torch.abs(denom_raw).item()))
            denom_sign = torch.where(denom_raw >= 0, torch.ones_like(denom_raw), -torch.ones_like(denom_raw))
            denom_safe = denom_sign * torch.clamp(torch.abs(denom_raw), min=self.count_rstable_denom_min)
            if float(torch.abs(denom_raw).item()) < self.count_rstable_denom_min:
                denom_clamp_hits += 1

            r_stable = torch.clamp((l_model - l_oracle) / denom_safe, min=-self.count_rstable_r_max, max=self.count_rstable_r_max)
            per_assay_r.append(r_stable)
            per_assay_w.append(weights[a])
            l_model_vals.append(float(l_model.item()))
            l_null_vals.append(float(l_null.item()))
            l_oracle_vals.append(float(l_oracle.item()))
            used_assays += 1

        if len(per_assay_r) == 0:
            return elem_count.new_tensor(0.0)

        r_vec = torch.stack(per_assay_r)
        if self.enable_hier_reduction:
            w_vec = torch.stack(per_assay_w).float()
            reduced = (r_vec * w_vec).sum() / torch.clamp(w_vec.sum(), min=1e-8)
        else:
            reduced = r_vec.mean()

        bname = "obs" if branch_idx == 0 else "imp"
        self.last_debug_stats[f"rstable_{bname}_assays"] = int(used_assays)
        self.last_debug_stats[f"rstable_{bname}_lmodel_mean"] = float(np.mean(l_model_vals)) if l_model_vals else np.nan
        self.last_debug_stats[f"rstable_{bname}_lnull_mean"] = float(np.mean(l_null_vals)) if l_null_vals else np.nan
        self.last_debug_stats[f"rstable_{bname}_loracle_mean"] = float(np.mean(l_oracle_vals)) if l_oracle_vals else np.nan
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
        # Normalize optional masks and initialize state.
        if obs_map_signal is None:
            obs_map_signal = obs_map
        if masked_map_signal is None:
            masked_map_signal = masked_map

        self.last_debug_stats = {}
        device = true_count.device
        num_assays = true_count.shape[-1]
        self._maybe_init_ema(num_assays, device)
        self._maybe_init_count_rstable(num_assays, device)
        self.last_debug_stats["count_rstable_enabled"] = int(self.enable_count_rstable_objective)
        self.last_debug_stats["count_rstable_active"] = int(
            self.enable_count_rstable_objective and (global_step >= self.count_rstable_warmup_steps)
        )
        self.last_debug_stats["count_rstable_warmup_steps"] = int(self.count_rstable_warmup_steps)

        # Element-wise count/signal losses over full [B, L, F] tensors.
        elem_count = self.nbin_nll(true_count, n_pred, p_pred)
        elem_signal = self._compute_signal_elementwise(mu_pred, scale_pred, df_pred, true_pval).float()

        # Safe peak loss path:
        # - only GT labels in {0,1} are valid for BCE targets
        # - invalid labels are zeroed and then excluded by mask during reduction
        peak_valid = (true_peak == 0) | (true_peak == 1)
        with torch.amp.autocast("cuda", enabled=False):
            peak_pred_safe = torch.clamp(peak_pred.float(), min=1e-6, max=1.0 - 1e-6)
            peak_target_safe = torch.where(peak_valid, true_peak, torch.zeros_like(true_peak)).float()
            elem_peak = self.bce_loss(peak_pred_safe, peak_target_safe).float()
            elem_peak = torch.where(peak_valid, elem_peak, torch.zeros_like(elem_peak))

        # Count branches (legacy or R_stable replacement).
        observed_count_loss = self._count_branch_rstable(
            elem_count, obs_map, true_count=true_count, true_peak=true_peak, branch_idx=0, global_step=global_step
        )
        imputed_count_loss = self._count_branch_rstable(
            elem_count, masked_map, true_count=true_count, true_peak=true_peak, branch_idx=1, global_step=global_step
        )

        # Signal branches (always standard loss family).
        observed_pval_loss = self._reduce_head_branch(
            elem_signal, obs_map_signal, head_idx=1, branch_idx=0, global_step=global_step, peak_gt=true_peak
        )
        imputed_pval_loss = self._reduce_head_branch(
            elem_signal, masked_map_signal, head_idx=1, branch_idx=1, global_step=global_step, peak_gt=true_peak
        )

        # Peak branches: gate with both supervision mask and valid GT labels.
        obs_map_peak = obs_map_signal & peak_valid
        masked_map_peak = masked_map_signal & peak_valid
        observed_peak_loss = self._reduce_head_branch(
            elem_peak, obs_map_peak, head_idx=2, branch_idx=0, global_step=global_step, peak_gt=None
        )
        imputed_peak_loss = self._reduce_head_branch(
            elem_peak, masked_map_peak, head_idx=2, branch_idx=1, global_step=global_step, peak_gt=None
        )

        # Apply static priors.
        observed_count_loss = self.count_weight * observed_count_loss
        imputed_count_loss = self.count_weight * imputed_count_loss
        observed_pval_loss = self.pval_weight * observed_pval_loss
        imputed_pval_loss = self.pval_weight * imputed_pval_loss
        observed_peak_loss = self.peak_weight * observed_peak_loss
        imputed_peak_loss = self.peak_weight * imputed_peak_loss

        # Optional dynamic task weighting.
        observed_count_loss, imputed_count_loss = self._apply_uncertainty(
            observed_count_loss, imputed_count_loss, self.logvar_count, global_step=global_step
        )
        observed_pval_loss, imputed_pval_loss = self._apply_uncertainty(
            observed_pval_loss, imputed_pval_loss, self.logvar_pval, global_step=global_step
        )
        observed_peak_loss, imputed_peak_loss = self._apply_uncertainty(
            observed_peak_loss, imputed_peak_loss, self.logvar_peak, global_step=global_step
        )

        # Branch weights.
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
