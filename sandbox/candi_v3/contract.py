"""CANDI v3 — FROZEN candidate ↔ harness contract.

An ERA candidate is ONE self-contained `program.py` that defines a `Model` and an `Objective`
(the parts ERA rewrites) and calls the frozen `harness.run_and_score(...)`. This module pins
the interfaces and the tensor semantics so the harness, the seed, and every candidate agree.

Design intent: minimal and likelihood-agnostic (CRITIQUE §9 as a contract). The harness owns
data + a FIXED training budget + eval + scoring; the candidate owns the architecture
(`Model`), the corruption + loss + optimizer recipe (`Objective`).

--------------------------------------------------------------------------------------------
INPUT tensors the harness passes to `Model.forward` (batch B, bins L=768, assays F=8):
  x_counts   [B,L,F] float  raw counts of INPUT (observed) assays; 0 where not observed
  x_avail    [B,F]   float  1 if assay observed in the input, else 0
  x_mask     [B,L,F] bool   positions corrupted/to-predict (training: from Objective.corrupt;
                            eval: held-out V_/B_ assay columns). Model must NOT peek at truth here.
  x_meta     [B,4,F] float  input covariates (depth_log2, assay_idx, read_len, run_type)
  x_dna      [B,Wbp,4] float one-hot DNA (Wbp=19200)
  control    [B,L,1] float  ChIP control track (may be all-zero — see ctrl_avail)
  ctrl_avail [B]     float  1 if a control track is present, else 0  (CONTROL-OPTIONAL: a
                            candidate must run and score with ctrl_avail==0 for some/all samples)
  y_meta     [B,4,F] float  TARGET-assay covariates (condition the query; held-out assay's OWN
                            depth/run_type so the model imputes the right assay at the right depth)
  query_mask [B,F]   bool   which assays to produce outputs for (observed + held-out)

OUTPUT dict `Model.forward(...)` must return (values over [B,L,F]):
  "count_dist"  torch.distributions.Distribution over raw counts, broadcastable to [B,L,F].
                The ONE learned likelihood. Harness uses .mean (DCR + count metrics) and
                .cdf(y) (calibration PIT). Must support .mean and .cdf on integer counts.
  "signal_pred" Tensor[B,L,F]  enrichment prediction (scored vs arcsinh-pval by Spearman -> S_A).
                May be a deterministic readout of count_dist (CRITIQUE §9.1) or a separate head.
  "peak_prob"   Optional[Tensor[B,L,F]] in (0,1); enables peak AUROC (logged; not in the score).
--------------------------------------------------------------------------------------------
"""
from __future__ import annotations

from typing import Dict, Optional, Protocol, runtime_checkable

import torch


@runtime_checkable
class Model(Protocol):
    """A torch.nn.Module whose forward returns the output dict above."""
    def forward(
        self,
        x_counts: torch.Tensor, x_avail: torch.Tensor, x_mask: torch.Tensor,
        x_meta: torch.Tensor, x_dna: torch.Tensor,
        control: torch.Tensor, ctrl_avail: torch.Tensor,
        y_meta: torch.Tensor, query_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]: ...


@runtime_checkable
class Objective(Protocol):
    """Owns corruption + loss + the optimizer recipe (ERA rewrites this)."""

    def corrupt(self, batch: Dict[str, torch.Tensor], rng: torch.Generator) -> Dict[str, torch.Tensor]:
        """Given a CLEAN train batch, return the harness INPUT tensors above PLUS supervision
        targets/masks the loss needs (y_counts [B,L,F], y_signal, y_peaks, sup_mask bool).
        The clean batch carries depth levels for denoising: counts (=dsf1) + counts_dsf{2,4,8},
        meta (=dsf1) + meta_dsf{2,4,8}, avail, dna, control, ctrl_avail. The unified corruption
        (per-(assay,pos) dropout ⊕ depth-downsample, CRITIQUE §4) lives here."""
        ...

    def loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Scalar training loss from model outputs + the corrupted batch (targets/masks)."""

    def configure_optimizer(self, params) -> torch.optim.Optimizer:
        """Optimizer (+ implicitly LR/schedule/EMA if the candidate wants them)."""


# Tensor-shape constants (must match data_v3).
L_BINS = 768
F_ASSAYS = 8
DNA_BP = 19200
