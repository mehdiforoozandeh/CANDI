"""Menu-AR row-0 candidate: candi_v2 (VENDORED + editable) under the frozen v3 judge contract.

COMMON ORIGIN (iteration 0) of all five menu loops. The model lives in the editable vendored
package `candi_model/` (a fork of sandbox.candi_v2) so a loop agent can rewrite architecture —
e.g. the latent is un-detached (repr_first) and the transformer in candi_model/encoder.py is
editable (axial_longrange). This file (the adapter + objective) is also fully editable.

OPTION B — train on REAL pval/peak. The frozen harness train batch ships counts only, but the
h5 (`T_<bios>` groups, chr19) DOES contain real `pval` (arcsinh -log10 p) and `peaks` {0,1}.
We read them via a LOCKSTEP iterator that byte-reproduces `iter_train_batches`' (bios,window)
order (bs=4, seed=0 = run_and_score defaults) and a HARD counts-equality assertion every step,
so pval/peak are provably paired to each training batch (desync -> crash, never silent mislabel).

Three trunk-connected heads, all supervised on real labels at query/available positions:
  count : NB(n,p)            -> NB NLL vs real counts
  pval  : adapter Gaussian head (mu,var) on count+peak features -> Gaussian CRPS vs real arcsinh-pval
  peak  : candi_v2 native peak head -> BCE vs real peaks
signal_pred = mu_pval (the judge correlates this point estimate); var feeds the CRPS proper score.
Optimizer: Adamax. candi_v2's depth_offset head (depth_center=22.5) keeps DCR in band.

Run:  PYTHONPATH=<repo_root> candi_venv/bin/python train_v2_row0.py
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # must precede cuBLAS init (det. matmul)

import math
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Determinism: make the frozen harness's run_and_score reproducible at its fixed seed.
# (Also enforced in sbatch_wrap.sh so loop edits here can't reintroduce nondeterminism.)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

REPO = "/project/6014832/mforooz/EpiDenoise"
HERE = str(Path(__file__).resolve().parent)                         # this loop dir (holds candi_model/)
JUDGE = str(Path(__file__).resolve().parents[1] / "_judge")
for p in (HERE, JUDGE, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from harness import run_and_score, H5                              # noqa: E402  (frozen judge)
from data_v3 import _win_indices, _t_bases                         # noqa: E402  (frozen data layout)
from candi_model.model import CANDIv2                              # noqa: E402  VENDORED, editable per loop
from candi_model.config import CANDIv2Config, validate_v2_config   # noqa: E402  VENDORED, editable per loop

MISSING, CLOZE, EPS, DEPTH_CENTER = -1.0, -2.0, 1e-6, 22.5
TRAIN_BS, TRAIN_SEED = 4, 0          # MUST match run_and_score(batch_size=, seed=) defaults


def _gauss_crps(y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Closed-form Gaussian CRPS; proper score for calibrated (mu, sigma) pval predictions."""
    w = (y - mu) / sigma
    phi = torch.exp(-0.5 * w * w) / math.sqrt(2.0 * math.pi)
    Phi = 0.5 * (1.0 + torch.erf(w / math.sqrt(2.0)))
    return sigma * (w * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))


def _loo_ref_enr(x_counts: torch.Tensor, x_avail: torch.Tensor, x_meta: torch.Tensor) -> torch.Tensor:
    """Leak-free LOO cross-sample average enrichment (present context assays only)."""
    d = x_meta[:, 0, :]
    valid = (d != MISSING) & (d != CLOZE)
    size = torch.where(valid, torch.pow(2.0, (d - DEPTH_CENTER).clamp(-8, 8)), torch.ones_like(d))
    enr = torch.asinh(x_counts.clamp(min=0) / size[:, None, :]) * x_avail[:, None, :]
    ref_num = enr.sum(dim=0, keepdim=True) - enr
    ref_cnt = (x_avail.sum(dim=0)[None, None, :] - x_avail[:, None, :]).clamp(min=1.0)
    return ref_num / ref_cnt


def _v2_config() -> CANDIv2Config:
    cfg = CANDIv2Config()
    cfg.encoder.num_assays = 8
    cfg.encoder.context_length = 768
    cfg.decoder.heads = "count_peak"     # native NB count head + Bernoulli peak head (both trainable)
    cfg.decoder.norm = "group"           # CONSOLIDATION: single_lambda's win — GroupNorm deconv
    cfg.decoder.count_head = "depth_offset"
    cfg.decoder.depth_center = 22.5
    validate_v2_config(cfg)
    return cfg


class Model(nn.Module):
    """v3 contract -> candi_v2 -> v3 contract, plus an adapter Gaussian pval head."""

    def __init__(self):
        super().__init__()
        self.v2 = CANDIv2(_v2_config())
        # Gaussian pval head: maps trunk-connected count+peak features -> (mu, log_var) per position.
        # Trunk-connected (grads flow through n,p,peak into the shared trunk); fully editable.
        self.pval_head = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 2))
        # Zero-init count mean + dispersion residuals from (NB, ref) features (init parity).
        self.count_delta = nn.Linear(4, 1)
        self.disp_delta = nn.Linear(4, 1)
        for head in (self.count_delta, self.disp_delta):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, x_counts, x_avail, x_mask, x_meta, x_dna,
                control, ctrl_avail, y_meta, query_mask):
        B, L, Fa = x_counts.shape
        present = x_avail > 0                                  # observed context assays
        query = query_mask.bool()                             # assays to predict
        cloze = query & ~present                              # held-out (imputation) -> CLOZE
        absent = ~present & ~query                            # never observed -> MISSING

        xc = x_counts.float().clone()
        xc = torch.where(cloze[:, None, :].expand(B, L, Fa), torch.full_like(xc, CLOZE), xc)
        xc = torch.where(absent[:, None, :].expand(B, L, Fa), torch.full_like(xc, MISSING), xc)
        x_data = torch.cat([xc, control.float()], dim=-1)     # [B,L,F+1]; control = channel A, never masked

        xm = x_meta.float().clone()
        xm = torch.where(cloze[:, None, :].expand(B, 4, Fa), torch.full_like(xm, CLOZE), xm)
        xm = torch.where(absent[:, None, :].expand(B, 4, Fa), torch.full_like(xm, MISSING), xm)
        ctrl_meta = xm.new_zeros(B, 4, 1)
        x_meta_v2 = torch.cat([xm, ctrl_meta], dim=-1)        # [B,4,F+1]

        x_dna_v2 = x_dna.permute(0, 2, 1).contiguous().float()  # [B,4,G]
        out = self.v2(x_data, x_dna_v2, x_meta_v2, y_meta.float())

        n, p, peak = out["n"], out["p"], out["peak"]
        n = n.clamp(min=EPS)
        probs = (1.0 - p).clamp(EPS, 1.0 - EPS)               # torch NB mean = n(1-p)/p = mu_NB
        mean = (n * probs / (1.0 - probs)).clamp(min=EPS)     # depth-anchored mu before residual head

        # depth-FREE enrichment 2^eta = mean / 2^(d-c) (matches the depth-INVARIANT pval target);
        # mirrors the decoder's sentinel fallback (invalid-depth cols use factor 1, never scored).
        d = y_meta[:, 0, :]                                   # [B,F] log2(seq_depth) per assay
        valid = (d != MISSING) & (d != CLOZE)
        size = torch.where(valid, torch.pow(2.0, (d - DEPTH_CENTER).clamp(-8, 8)), torch.ones_like(d))
        asinh_enrich = torch.asinh(mean / size[:, None, :])   # [B,L,F]
        ref_enr = _loo_ref_enr(x_counts, x_avail, x_meta)   # leak-free avg-reference enrichment

        # Zero-init multiplicative residuals on count mean + NB dispersion (parity at init).
        feat = torch.stack([asinh_enrich, torch.log(n), peak, asinh_enrich - ref_enr], dim=-1)
        log_delta = self.count_delta(feat).squeeze(-1).clamp(-2.0, 2.0)
        log_n_delta = self.disp_delta(feat).squeeze(-1).clamp(-1.0, 1.0)
        adj_mean = mean * torch.exp(log_delta)
        n_adj = (n * torch.exp(log_n_delta)).clamp(min=EPS)
        probs_adj = (adj_mean / (n_adj + adj_mean)).clamp(EPS, 1.0 - EPS)
        count_dist = torch.distributions.NegativeBinomial(total_count=n_adj, probs=probs_adj)

        # Gaussian pval head on [arcsinh(enrichment), log n, peak_prob] -> (mu, var)
        pval_feat = torch.stack([asinh_enrich, torch.log(n), peak], dim=-1)   # [B,L,F,3]
        h = self.pval_head(pval_feat)
        mu_pval = F.softplus(h[..., 0])                        # >=0 (arcsinh-pval is >=0)
        var_pval = torch.exp(h[..., 1].clamp(-10.0, 10.0)) + EPS
        return {"count_dist": count_dist, "signal_pred": mu_pval,
                "peak_prob": peak, "pval_var": var_pval}


class Objective:
    """Unified corruption (imputation OR denoising per batch); 3 real-label heads (B)."""

    P_DEN = 0.35

    def __init__(self):
        self._labels = self._label_iter()    # lockstep with run_and_score's train iterator

    def _label_iter(self):
        """Byte-reproduce iter_train_batches' (bios,windows) order; yield (counts, pval, peaks)
        for the SAME windows so the per-batch counts-equality check proves correct pairing."""
        rng = np.random.default_rng(TRAIN_SEED)
        with h5py.File(H5, "r") as h5:
            widx = _win_indices(h5, train=True)
            bases = _t_bases(h5)
            items = [(b, s) for b in bases for s in range(0, len(widx), TRAIN_BS)]
            while True:
                rng.shuffle(items)
                for b, s in items:
                    g = h5["biosamples"][f"T_{b}"]
                    loc = widx[s:s + TRAIN_BS]
                    yield (np.asarray(g["counts_dsf1"][loc], dtype=np.float32),
                           np.asarray(g["pval"][loc], dtype=np.float32),
                           np.asarray(g["peaks"][loc], dtype=np.float32))

    def corrupt(self, batch, rng):
        counts = batch["counts"].float()
        B, L, Fa = counts.shape
        dev = counts.device
        avail = batch["avail"].float()
        meta = batch["meta"].float()

        # --- fetch REAL pval/peaks for THIS batch's windows; assert correct pairing ---
        ck, pv, pk = next(self._labels)
        if not np.array_equal(ck, counts.detach().cpu().numpy()):
            raise RuntimeError("pval/peak label pairing DESYNC: lockstep counts != batch counts "
                               "(check TRAIN_BS/TRAIN_SEED == run_and_score defaults).")
        y_pval = torch.as_tensor(pv, device=dev)               # [B,L,F] real arcsinh-pval (dsf1)
        y_peak = torch.as_tensor(pk, device=dev).float()       # [B,L,F] real peaks {0,1} (dsf1)

        denoise = torch.rand((), generator=rng).item() < self.P_DEN
        if denoise:                                            # low-depth in -> full-depth out
            x_counts = batch["counts_dsf8"].float().clone()
            x_meta = batch["meta_dsf8"].float().clone()
            x_avail = avail.clone()
            query = avail > 0
        else:                                                  # mask 1-2 present assays -> impute
            x_counts = counts.clone()
            x_meta = meta.clone()
            query = torch.zeros(B, Fa, dtype=torch.bool, device=dev)
            for b in range(B):
                pres = torch.where(avail[b] > 0)[0]
                if pres.numel() == 0:
                    continue
                k = 1 + int(torch.rand((), generator=rng).item() < 0.5)
                sel = pres[torch.randperm(pres.numel(), generator=rng)[:k]]
                query[b, sel] = True
            x_avail = avail.clone()
            x_avail[query] = 0.0
            x_counts[query[:, None, :].expand(B, L, Fa)] = 0.0

        # query is always a subset of available assays -> pval/peak labels there are real (no sentinels).
        sup = query[:, None, :].expand(B, L, Fa).contiguous()
        ref_enr = _loo_ref_enr(x_counts, x_avail, x_meta)
        return {
            "x_counts": x_counts, "x_avail": x_avail, "x_mask": sup, "x_meta": x_meta,
            "x_dna": batch["dna"].float(), "control": batch["control"].float(),
            "ctrl_avail": batch["ctrl_avail"].float(), "y_meta": meta.clone(), "query_mask": query,
            "y_counts": counts.clone(), "y_pval": y_pval, "y_peak": y_peak, "sup_mask": sup,
            "ref_enr": ref_enr,
        }

    @staticmethod
    def _pearson_global(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        p, t = pred[mask], tgt[mask]
        if p.numel() < 2:
            return pred.new_zeros(())
        p = p - p.mean()
        t = t - t.mean()
        num = (p * t).sum()
        den = torch.sqrt((p * p).sum().clamp(min=EPS) * (t * t).sum().clamp(min=EPS))
        return 1.0 - num / den

    def loss(self, out, cb):
        m = cb["sup_mask"]
        if m.sum() == 0:
            m = torch.ones_like(m)
        y = cb["y_counts"].clamp(min=0)
        nll_count = -out["count_dist"].log_prob(y)[m].mean()
        sigma = out["pval_var"][m].clamp(min=EPS).sqrt()
        crps_pval = _gauss_crps(cb["y_pval"][m], out["signal_pred"][m], sigma).mean()
        bce_peak = F.binary_cross_entropy(out["peak_prob"].clamp(EPS, 1 - EPS)[m], cb["y_peak"][m])
        ref = cb["ref_enr"].detach()
        pval_dev = self._pearson_global(out["signal_pred"] - ref, cb["y_pval"] - ref, m)
        d_tgt = cb["y_meta"][:, 0, :]
        valid = (d_tgt != MISSING) & (d_tgt != CLOZE)
        size = torch.where(valid, torch.pow(2.0, (d_tgt - DEPTH_CENTER).clamp(-8, 8)),
                           torch.ones_like(d_tgt))
        pred_enr = torch.asinh(out["count_dist"].mean / size[:, None, :])
        y_enr = torch.asinh(cb["y_counts"].clamp(min=0) / size[:, None, :])
        cnt_dev = self._pearson_global(pred_enr - ref, y_enr - ref, m)
        return nll_count + crps_pval + bce_peak + 0.02 * pval_dev + 0.02 * cnt_dev

    def configure_optimizer(self, params):
        return torch.optim.Adamax(params, lr=2e-3)


if __name__ == "__main__":
    kw = {}
    if os.environ.get("MENU_SMAX"):  kw["s_max"] = int(os.environ["MENU_SMAX"])
    if os.environ.get("MENU_MAXEP"): kw["max_epochs"] = float(os.environ["MENU_MAXEP"])
    if os.environ.get("MENU_TMAX"):  kw["t_max_sec"] = float(os.environ["MENU_TMAX"])
    print(f"[ceiling] AUGMENT={os.environ.get('MENU_AUGMENT','0')} budget_kwargs={kw}")
    print(f"ERA_SCORE: {run_and_score(Model, Objective(), **kw)}")
