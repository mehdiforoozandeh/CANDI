"""Sandbox loss stack: production `CANDI_LOSS` with the full six-branch forward."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from candi_loss import CANDI_LOSS


class SandboxCompositeLoss(nn.Module):
    """
    Thin wrapper around `CANDI_LOSS` that enforces production semantics for the
    no-masked-regions edge case.

    Production `train.py::_process_batch` behaviour when there are no cloze-masked
    positions: imputation heads still run with `masked_map` replaced by
    `observed_map` (lines 1303-1304 / 1316-1317 of prod train.py) so that
    uncertainty weights and imp_* terms stay on the optimisation graph.

    `CANDI_LOSS` applies `count_weight`, `pval_weight`, `peak_weight`,
    `obs_weight`, `imp_weight` from its constructor; pass those via
    `SandboxConfig.training.loss_weights` when building this module.
    """

    def __init__(self, cand: CANDI_LOSS):
        super().__init__()
        self.cand = cand

    @staticmethod
    def _safe_unweight(weighted: torch.Tensor, weight: float) -> torch.Tensor:
        if abs(float(weight)) < 1e-12:
            # If a head is disabled (weight 0), the unweighted value is not recoverable.
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
        # Optional prod compatibility: when no cloze-masked positions exist, route imp heads
        # through observed maps so imp branches stay on-graph.
        if fallback_imp_to_observed_when_no_masked:
            has_masked = bool(masked_map.any().item())
            mm = masked_map if has_masked else observed_map
            smm = signal_masked_map if has_masked else signal_observed_map
        else:
            mm = masked_map
            smm = signal_masked_map

        count_obs_w, count_imp_w, pval_obs_w, pval_imp_w, peak_obs_w, peak_imp_w = self.cand(
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
            mm,
            signal_observed_map,
            smm,
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
            terms["count_obs_weighted"]
            + terms["count_imp_weighted"]
            + terms["pval_obs_weighted"]
            + terms["pval_imp_weighted"]
            + terms["peak_obs_weighted"]
            + terms["peak_imp_weighted"]
        )
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
            "loss_branch_count_obs_weighted": float(terms["count_obs_weighted"].detach().item()),
            "loss_branch_count_imp_weighted": float(terms["count_imp_weighted"].detach().item()),
            "loss_branch_pval_obs_weighted": float(terms["pval_obs_weighted"].detach().item()),
            "loss_branch_pval_imp_weighted": float(terms["pval_imp_weighted"].detach().item()),
            "loss_branch_peak_obs_weighted": float(terms["peak_obs_weighted"].detach().item()),
            "loss_branch_peak_imp_weighted": float(terms["peak_imp_weighted"].detach().item()),
            "loss_branch_count_obs_raw": float(terms["count_obs_raw"].detach().item()),
            "loss_branch_count_imp_raw": float(terms["count_imp_raw"].detach().item()),
            "loss_branch_pval_obs_raw": float(terms["pval_obs_raw"].detach().item()),
            "loss_branch_pval_imp_raw": float(terms["pval_imp_raw"].detach().item()),
            "loss_branch_peak_obs_raw": float(terms["peak_obs_raw"].detach().item()),
            "loss_branch_peak_imp_raw": float(terms["peak_imp_raw"].detach().item()),
        }
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
            global_step=global_step,
            fallback_imp_to_observed_when_no_masked=fallback_imp_to_observed_when_no_masked,
        )
        stats = self._stats_from_terms(terms)
        return terms["total_weighted"], stats, terms

    @property
    def loss_weights(self) -> Dict[str, float]:
        """Mirrors `CANDI_LOSS` static weights for logging / HPO introspection."""
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
            global_step=global_step,
            fallback_imp_to_observed_when_no_masked=fallback_imp_to_observed_when_no_masked,
        )
        return total, stats


__all__ = ["CANDI_LOSS", "SandboxCompositeLoss"]
