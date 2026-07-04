"""CANDI v3 — minimal NEUTRAL seed candidate (the ERA root).

Deliberately minimal and architecture-neutral: a per-position MLP over (arcsinh input counts +
pooled DNA + target covariates) that mixes assays through a dense layer and emits an NB count
distribution + a signal prediction. NO transformer, NO committed cross-assay mechanism — a
clean compiling root for ERA to evolve, not a design. ERA rewrites Model + Objective.

A candidate is this whole file; it imports the FROZEN harness and prints the ERA_SCORE footer.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/project/6014832/mforooz/EpiDenoise/sandbox/candi_v3")

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from harness import run_and_score

F = 8           # assays
L = 768         # bins
POOL = 25       # 19200 bp / 768 bins


class Model(nn.Module):
    def __init__(self, d: int = 64):
        super().__init__()
        self.dna = nn.Conv1d(4, 16, kernel_size=7, padding=3)
        # per-position features: F input counts + F avail + 16 dna + 4 target-meta-per-assay summary
        self.mlp = nn.Sequential(
            nn.Linear(F + F + 16 + 4, d), nn.GELU(),
            nn.Linear(d, d), nn.GELU(),
        )
        self.head_r = nn.Linear(d, F)        # NB total_count (dispersion)
        self.head_mu = nn.Linear(d, F)       # NB base mean (enrichment, depth-free)
        self.head_sig = nn.Linear(d, F)      # signal prediction

    def forward(self, x_counts, x_avail, x_mask, x_meta, x_dna,
                control, ctrl_avail, y_meta, query_mask):
        B = x_counts.shape[0]
        xa = torch.arcsinh(x_counts) * x_avail[:, None, :]            # [B,L,F]
        dna = self.dna(x_dna.transpose(1, 2))                         # [B,16,19200]
        dna = Fn.avg_pool1d(dna, POOL).transpose(1, 2)               # [B,L,16]
        depth = y_meta[:, 0, :].clamp(min=0)                          # [B,F] target depth_log2
        meta_sum = y_meta.mean(dim=2, keepdim=True).transpose(1, 2)   # [B,1,4]
        meta_sum = meta_sum.expand(B, L, 4)
        feat = torch.cat([xa, x_avail[:, None, :].expand(B, L, F), dna, meta_sum], dim=-1)
        h = self.mlp(feat)                                            # [B,L,d]
        r = Fn.softplus(self.head_r(h)) + 1e-3                        # total_count > 0
        mu_base = Fn.softplus(self.head_mu(h)) + 1e-3                 # depth-free enrichment
        size = torch.pow(2.0, (depth - 23.0).clamp(-4, 4))[:, None, :]  # size-factor offset
        mu = mu_base * size                                           # mean scales with depth
        probs = (mu / (mu + r)).clamp(1e-4, 1 - 1e-4)               # NB mean == mu
        count_dist = torch.distributions.NegativeBinomial(total_count=r, probs=probs)
        return {"count_dist": count_dist, "signal_pred": self.head_sig(h), "peak_prob": None}


class Objective:
    """Simple corruption: hold out a random subset of available assays (imputation) + NB NLL."""

    def corrupt(self, batch, rng):
        counts, avail, meta = batch["counts"], batch["avail"], batch["meta"]
        B, Lc, Fc = counts.shape
        dev = counts.device
        x_avail = avail.clone()
        x_mask = torch.zeros(B, Lc, Fc, dtype=torch.bool, device=dev)
        for b in range(B):
            present = torch.where(avail[b] > 0)[0]
            if present.numel() <= 1:
                continue
            k = int(torch.randint(1, present.numel(), (1,), generator=rng).item())
            drop = present[torch.randperm(present.numel(), generator=rng)[:k]]
            x_avail[b, drop] = 0.0
            x_mask[b, :, drop] = True
        x_counts = counts * x_avail[:, None, :]
        return {
            "x_counts": x_counts, "x_avail": x_avail, "x_mask": x_mask, "x_meta": meta,
            "x_dna": batch["dna"], "control": batch["control"], "ctrl_avail": batch["ctrl_avail"],
            "y_meta": meta, "query_mask": (avail > 0),
            "y_counts": counts, "sup_mask": x_mask,
        }

    def loss(self, out, cb):
        cd = out["count_dist"]
        m = cb["sup_mask"]
        if m.sum() == 0:
            return (out["signal_pred"].sum() * 0.0).requires_grad_(True)
        nll = -cd.log_prob(cb["y_counts"].clamp(min=0))
        return nll[m].mean()

    def configure_optimizer(self, params):
        return torch.optim.Adam(params, lr=1e-3)


if __name__ == "__main__":
    # The harness owns the fixed training budget; candidates do not set train_steps.
    print(f"ERA_SCORE: {run_and_score(Model, Objective())}")
