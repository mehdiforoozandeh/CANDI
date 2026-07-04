"""CANDI v3 — residual-on-reference imputer (node-161 scaffold) plus three targeted fixes for
the binding scored floors: (1) mask-conditional NB dispersion inflation at imputation/query
positions so the count predictive distribution widens exactly where eval computes ECE on
fully-held-out assays; (2) Avocado-style 8/32/128-bin position pyramid + coarse ref/CF
regional features feeding the zero-init deviation heads for multi-scale magnitude/shape
(§3.5 multi-resolution win); (3) light count-mean + smoothed-local Pearson terms so the
lagging imp_count_pearson (0.37) catches up without rank-only surrogates. NB size-factor
structure, corruption, deviation-correlation, and residual baseline are unchanged."""
from __future__ import annotations

import sys

sys.path.insert(0, "/project/6014832/mforooz/EpiDenoise/sandbox/candi_v3")

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from harness import run_and_score

F = 8
L = 768
POOL = 25
REF_DEPTH = 23.0
N_ASSAY_ID = 64
CF_HEADS = 4
MR_SCALES = (8, 32, 128)
REG_KS = (17, 49)


class DNATower(nn.Module):
    def __init__(self, d_dna: int):
        super().__init__()
        self.stem = nn.Conv1d(4, d_dna, kernel_size=15, padding=7)
        self.dils = nn.ModuleList()
        for dil in (1, 2, 4, 8):
            self.dils.append(nn.Sequential(
                nn.Conv1d(d_dna, d_dna, kernel_size=5, padding=2 * dil, dilation=dil),
                nn.GELU(),
                nn.Conv1d(d_dna, d_dna, kernel_size=1),
            ))
        self.norm = nn.LayerNorm(d_dna)

    def forward(self, x_dna):
        x = Fn.gelu(self.stem(x_dna.transpose(1, 2)))
        x = Fn.max_pool1d(x, POOL)
        for blk in self.dils:
            x = x + blk(x)
        return self.norm(x.transpose(1, 2))


class Spatial(nn.Module):
    def __init__(self, d: int, dil: int):
        super().__init__()
        self.dw = nn.Conv1d(d, d, kernel_size=5, padding=2 * dil, dilation=dil, groups=d)
        self.pw = nn.Conv1d(d, d, kernel_size=1)
        self.norm = nn.LayerNorm(d)

    def forward(self, h, key_padding=None):
        B, Lc, Fc, d = h.shape
        x = h.permute(0, 2, 3, 1).reshape(B * Fc, d, Lc)
        y = self.pw(Fn.gelu(self.dw(x)))
        y = y.reshape(B, Fc, d, Lc).permute(0, 3, 1, 2)
        return self.norm(h + y)


class CrossAssay(nn.Module):
    def __init__(self, d: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.enc = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=2 * d, dropout=dropout, batch_first=True
        )

    def forward(self, h, key_padding=None):
        B, Lc, Fc, d = h.shape
        x = h.reshape(B * Lc, Fc, d)
        kpm = None
        if key_padding is not None:
            kpm = key_padding[:, None, :].expand(B, Lc, Fc).reshape(B * Lc, Fc)
        x = self.enc(x, src_key_padding_mask=kpm)
        return x.reshape(B, Lc, Fc, d)


class MultiResPos(nn.Module):
    """Avocado-style multi-resolution position factors (8/32/128 bins ~ 200bp/800bp/3.2kb)."""

    def __init__(self, d: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d * len(MR_SCALES), d), nn.GELU(), nn.LayerNorm(d))

    def forward(self, h):
        B, Lc, Fc, d = h.shape
        x = h.permute(0, 2, 3, 1).reshape(B * Fc, d, Lc)
        parts = []
        for s in MR_SCALES:
            if s > 1:
                p = Fn.avg_pool1d(x, kernel_size=s, stride=1, padding=s // 2)
                if p.shape[-1] > Lc:
                    p = p[..., :Lc]
                elif p.shape[-1] < Lc:
                    p = Fn.pad(p, (0, Lc - p.shape[-1]))
            else:
                p = x
            parts.append(p.reshape(B, Fc, d, Lc).permute(0, 3, 1, 2))
        return self.proj(torch.cat(parts, dim=-1))


def _regional_1d(x, k):
    """Stride-1 moving average along position; x [B,L] -> [B,L]."""
    Lc = x.shape[-1]
    t = x.unsqueeze(1)
    y = Fn.avg_pool1d(t, kernel_size=k, stride=1, padding=k // 2)
    if y.shape[-1] > Lc:
        y = y[..., :Lc]
    elif y.shape[-1] < Lc:
        y = Fn.pad(y, (0, Lc - y.shape[-1]))
    return y.squeeze(1)


class Model(nn.Module):
    def __init__(self, d: int = 64, d_a: int = 16, d_dna: int = 48, n_blocks: int = 4):
        super().__init__()
        assert d_a % CF_HEADS == 0
        self.d_dna = d_dna
        self.d_a = d_a
        self.hd = d_a // CF_HEADS
        self.assay_emb = nn.Embedding(N_ASSAY_ID, d_a)
        self.dna = DNATower(d_dna)
        self.dna_film = nn.Linear(d_a, 2 * d_dna)
        in_dim = 7 + d_a + d_dna
        self.in_proj = nn.Sequential(nn.Linear(in_dim, d), nn.GELU(), nn.LayerNorm(d))
        self.query_emb = nn.Embedding(N_ASSAY_ID, d)
        nn.init.normal_(self.query_emb.weight, std=0.02)
        self.blocks = nn.ModuleList()
        dils = [1, 2, 4, 8]
        for i in range(n_blocks):
            self.blocks.append(Spatial(d, dil=dils[i % len(dils)]))
            self.blocks.append(CrossAssay(d))
        self.cf_W = nn.Parameter(torch.eye(self.hd)[None].repeat(CF_HEADS, 1, 1))
        self.cf_gate = nn.Parameter(torch.ones(N_ASSAY_ID))
        self.head_r = nn.Linear(d, 1)
        self.head_mu = nn.Linear(d, 1)
        # Mask-conditional extra dispersion: widens NB at imputation/query positions (ECE target).
        self.mask_disp = nn.Parameter(torch.full((N_ASSAY_ID,), 1.2))
        self.mr_pos = MultiResPos(d)
        sig_in = d + 1 + CF_HEADS + d + 2 * len(REG_KS)
        self.head_sig_imp = nn.Sequential(
            nn.Linear(sig_in, d), nn.GELU(), nn.Linear(d, 1))
        self.head_sig_den = nn.Sequential(
            nn.Linear(sig_in, d), nn.GELU(), nn.Linear(d, 1))
        for head in (self.head_sig_imp, self.head_sig_den):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        self.cov_bias = nn.Sequential(
            nn.Linear(2, d_a), nn.GELU(), nn.Linear(d_a, 1))
        nn.init.zeros_(self.cov_bias[-1].weight)
        nn.init.zeros_(self.cov_bias[-1].bias)
        self.disp_bias = nn.Parameter(torch.zeros(N_ASSAY_ID))
        self.ref_gate = nn.Parameter(torch.ones(N_ASSAY_ID))
        self.sig_scale = nn.Parameter(torch.ones(N_ASSAY_ID))
        self.sig_bias = nn.Parameter(torch.zeros(N_ASSAY_ID))
        self.pk_scale = nn.Parameter(torch.ones(N_ASSAY_ID))
        self.pk_bias = nn.Parameter(torch.zeros(N_ASSAY_ID))

    def _aid(self, x_meta):
        return x_meta[:, 1, :].round().long().clamp(0, N_ASSAY_ID - 1)

    def forward(self, x_counts, x_avail, x_mask, x_meta, x_dna,
                control, ctrl_avail, y_meta, query_mask):
        B = x_counts.shape[0]
        aid = self._aid(y_meta)

        dna = self.dna(x_dna)[:, :, None, :].expand(B, L, F, self.d_dna)
        ae = self.assay_emb(aid)
        gamma, beta = self.dna_film(ae).chunk(2, dim=-1)
        dna = dna * (1.0 + gamma[:, None]) + beta[:, None]

        size_in = torch.pow(2.0, (x_meta[:, 0, :] - REF_DEPTH).clamp(-4, 4)).clamp(min=1e-6)
        enr_in = torch.arcsinh(x_counts.clamp(min=0) / size_in[:, None, :])
        avail_f = x_avail[:, None, :]
        ref = (enr_in * avail_f).sum(dim=2) / x_avail.sum(dim=1, keepdim=True).clamp(min=1.0)
        ref_b = ref[:, :, None].expand(B, L, F)

        cnt = torch.arcsinh(x_counts) * x_avail[:, None, :]
        av = x_avail[:, None, :].expand(B, L, F)
        is_query = x_mask.float()
        dep_in = x_meta[:, 0, :][:, None, :].expand(B, L, F)
        dep_tgt = y_meta[:, 0, :][:, None, :].expand(B, L, F)
        ctrl = (control[:, :, 0] * ctrl_avail[:, None])[:, :, None].expand(B, L, F)
        scal = torch.stack([cnt, av, is_query, dep_in, dep_tgt, ctrl, ref_b], dim=-1)

        ae_b = ae[:, None, :, :].expand(B, L, F, ae.shape[-1])
        h = self.in_proj(torch.cat([scal, ae_b, dna], dim=-1))
        qtok = self.query_emb(aid)
        h = h + x_mask.float().unsqueeze(-1) * qtok[:, None]

        key_padding = ~(x_avail > 0)
        for blk in self.blocks:
            h = blk(h, key_padding)

        ae_h = ae.view(B, F, CF_HEADS, self.hd)
        aff = torch.einsum("bfhd,hde,bghe->bfgh", ae_h, self.cf_W, ae_h)
        key_ok = (x_avail > 0)[:, None, :, None]
        aff = aff.masked_fill(~key_ok, -1e9)
        p = torch.softmax(aff, dim=2)
        cf_prior = torch.einsum("bfgh,blg->blfh", p, enr_in)
        cf_mean = cf_prior.mean(dim=-1)

        disp = self.disp_bias[aid][:, None, :]
        r_base = Fn.softplus(self.head_r(h).squeeze(-1) + disp) + 1e-2
        r_extra = Fn.softplus(self.mask_disp[aid])[:, None, :]
        r = r_base + x_mask.float() * r_extra
        mu_base = Fn.softplus(self.head_mu(h).squeeze(-1)) + 1e-3
        size = torch.pow(2.0, (y_meta[:, 0, :] - REF_DEPTH).clamp(-4, 4))[:, None, :]
        mu = mu_base * size
        probs = (mu / (mu + r)).clamp(1e-4, 1 - 1e-4)
        count_dist = torch.distributions.NegativeBinomial(total_count=r, probs=probs)

        log_enr = torch.log(mu_base + 1e-3).unsqueeze(-1)
        mr = self.mr_pos(h)
        reg_parts = []
        for k in REG_KS:
            reg_parts.append(_regional_1d(ref, k).unsqueeze(-1).expand(B, L, F))
            reg_parts.append(_regional_1d(cf_mean.mean(dim=2), k).unsqueeze(-1).expand(B, L, F))
        reg_feat = torch.stack(reg_parts, dim=-1)
        feat_sig = torch.cat([h, log_enr, cf_prior, mr, reg_feat], dim=-1)
        delta_imp = self.head_sig_imp(feat_sig).squeeze(-1)
        delta_den = self.head_sig_den(feat_sig).squeeze(-1)
        delta = torch.where(x_mask, delta_imp, delta_den)

        cov = torch.stack([y_meta[:, 2, :] / 100.0, y_meta[:, 3, :]], dim=-1)
        cov_b = self.cov_bias(cov).squeeze(-1)[:, None, :]
        gate = self.cf_gate[aid][:, None, :]
        rgate = self.ref_gate[aid][:, None, :]
        combined = rgate * ref_b + gate * cf_mean + cov_b + delta
        sc = self.sig_scale[aid][:, None, :]
        bi = self.sig_bias[aid][:, None, :]
        signal_pred = combined * sc + bi
        pk = self.pk_scale[aid][:, None, :]
        pb = self.pk_bias[aid][:, None, :]
        peak_prob = torch.sigmoid(signal_pred * pk + pb)
        return {"count_dist": count_dist, "signal_pred": signal_pred, "peak_prob": peak_prob}


class Objective:
    LEVELS = [("counts", "meta"),
              ("counts_dsf2", "meta_dsf2"),
              ("counts_dsf4", "meta_dsf4"),
              ("counts_dsf8", "meta_dsf8")]

    IMP_W = 1.0
    CONS_W = 0.3
    DEV_W = 1.0
    SMOOTH_K = 32

    def corrupt(self, batch, rng):
        avail = batch["avail"]
        counts_full, meta_full = batch["counts"], batch["meta"]
        B, Lc, Fc = counts_full.shape
        dev = counts_full.device

        li = int(torch.randint(0, len(self.LEVELS), (1,), generator=rng).item())
        ck, mk = self.LEVELS[li]
        x_counts = batch[ck].clone()
        x_meta = batch[mk]

        heavy = float(torch.rand((1,), generator=rng).item()) < 0.3
        lo, span = (0.30, 0.20) if heavy else (0.10, 0.20)

        x_avail = avail.clone()
        x_mask = torch.zeros(B, Lc, Fc, dtype=torch.bool, device=dev)
        for b in range(B):
            present = torch.where(avail[b] > 0)[0]
            n = present.numel()
            if n <= 1:
                continue
            frac = lo + span * float(torch.rand((1,), generator=rng).item())
            k = max(1, min(max(1, n - 1), int(round(frac * n))))
            drop = present[torch.randperm(n, generator=rng)[:k]]
            x_avail[b, drop] = 0.0
            x_mask[b, :, drop] = True
        x_counts = x_counts * x_avail[:, None, :]

        sup_mask = (avail > 0)[:, None, :].expand(B, Lc, Fc).clone()
        thr = torch.quantile(counts_full, 0.90, dim=1, keepdim=True)
        y_peaks = (counts_full > thr).float()
        size = torch.pow(2.0, (meta_full[:, 0, :] - REF_DEPTH).clamp(-4, 4))[:, None, :]
        y_enr = torch.arcsinh(counts_full.clamp(min=0) / size.clamp(min=1e-6))
        avail_full = (avail > 0).float()
        y_ref = (y_enr * avail_full[:, None, :]).sum(dim=2) / \
                avail_full.sum(dim=1, keepdim=True).clamp(min=1.0)
        return {
            "x_counts": x_counts, "x_avail": x_avail, "x_mask": x_mask, "x_meta": x_meta,
            "x_dna": batch["dna"], "control": batch["control"], "ctrl_avail": batch["ctrl_avail"],
            "y_meta": meta_full, "query_mask": (avail > 0),
            "y_counts": counts_full, "sup_mask": sup_mask, "y_peaks": y_peaks, "y_enr": y_enr,
            "y_ref": y_ref,
        }

    def _pearson_track(self, pred, tgt, track_imp):
        if track_imp.sum() == 0:
            return pred.sum() * 0.0
        p = pred - pred.mean(dim=1, keepdim=True)
        t = tgt - tgt.mean(dim=1, keepdim=True)
        num = (p * t).sum(dim=1)
        den = torch.sqrt(((p * p).sum(dim=1)).clamp(min=1e-6) *
                         ((t * t).sum(dim=1)).clamp(min=1e-6))
        corr = (num / den)[track_imp]
        return 1.0 - corr.mean()

    def _pearson_global(self, pred, tgt, sel):
        if sel.sum() < 2:
            return pred.sum() * 0.0
        p = pred[sel]
        t = tgt[sel]
        p = p - p.mean()
        t = t - t.mean()
        den = torch.sqrt((p * p).sum().clamp(min=1e-6) * (t * t).sum().clamp(min=1e-6))
        return 1.0 - (p * t).sum() / den

    def _deviation_pearson(self, pred, tgt, ref, sel):
        if sel.sum() < 2:
            return pred.sum() * 0.0
        refb = ref[:, :, None].expand_as(pred)
        dp = (pred - refb)[sel]
        dt = (tgt - refb)[sel]
        dp = dp - dp.mean()
        dt = dt - dt.mean()
        den = torch.sqrt((dp * dp).sum().clamp(min=1e-6) * (dt * dt).sum().clamp(min=1e-6))
        return 1.0 - (dp * dt).sum() / den

    def _smooth_along_l(self, x, k):
        B, Lc, Fc = x.shape
        t = x.permute(0, 2, 1).reshape(B * Fc, 1, Lc)
        y = Fn.avg_pool1d(t, kernel_size=k, stride=1, padding=k // 2)
        if y.shape[-1] > Lc:
            y = y[..., :Lc]
        elif y.shape[-1] < Lc:
            y = Fn.pad(y, (0, Lc - y.shape[-1]))
        return y.reshape(B, Fc, Lc).permute(0, 2, 1)

    def _consistency(self, sig_pred, sig_tgt, cb, m):
        imp_sel = (cb["x_mask"] & m).float()
        obs_sel = ((~cb["x_mask"]) & m).float()
        imp_n = imp_sel.sum(dim=2).clamp(min=1.0)
        obs_n = obs_sel.sum(dim=2).clamp(min=1.0)
        imp_m = (sig_pred * imp_sel).sum(dim=2) / imp_n
        obs_m = (sig_tgt.detach() * obs_sel).sum(dim=2) / obs_n
        valid_b = ((cb["x_mask"] & m).any(dim=1).any(dim=1) &
                   ((~cb["x_mask"]) & m).any(dim=1).any(dim=1))
        if valid_b.sum() == 0:
            return sig_pred.sum() * 0.0
        p = imp_m - imp_m.mean(dim=1, keepdim=True)
        t = obs_m - obs_m.mean(dim=1, keepdim=True)
        num = (p * t).sum(dim=1)
        den = torch.sqrt((p * p).sum(dim=1).clamp(min=1e-6) * (t * t).sum(dim=1).clamp(min=1e-6))
        corr = (num / den)[valid_b]
        return 1.0 - corr.mean()

    def loss(self, out, cb):
        m = cb["sup_mask"]
        if m.sum() == 0:
            return (out["signal_pred"].sum() * 0.0).requires_grad_(True)
        y = cb["y_counts"].clamp(min=0)
        sig_tgt = cb["y_enr"]
        sig_pred = out["signal_pred"]
        cnt_mean = out["count_dist"].mean

        tnorm = sig_tgt / (sig_tgt.mean(dim=1, keepdim=True).clamp(min=1e-3) + 1e-6)
        hi = 1.0 + 0.5 * tnorm.clamp(0, 4)
        w = torch.where(cb["x_mask"], self.IMP_W, 1.0) * m.float() * hi
        wsum = w.sum().clamp(min=1.0)

        nll = -out["count_dist"].log_prob(y)
        huber = Fn.smooth_l1_loss(sig_pred, sig_tgt, reduction="none", beta=1.0)
        nll_t = (nll * w).sum() / wsum
        hub_t = (huber * w).sum() / wsum

        track_imp = cb["x_mask"][:, 0, :] & cb["query_mask"]
        sel = cb["x_mask"] & m

        pe_gl = self._pearson_global(sig_pred, sig_tgt, sel)
        pe_pt = self._pearson_track(sig_pred, sig_tgt, track_imp)
        dev = self._deviation_pearson(sig_pred, sig_tgt, cb["y_ref"], sel)
        cons = self._consistency(sig_pred, sig_tgt, cb, m)

        sp_smooth = self._smooth_along_l(sig_pred, self.SMOOTH_K)
        st_smooth = self._smooth_along_l(sig_tgt, self.SMOOTH_K)
        shape_pe = self._pearson_global(sp_smooth, st_smooth, sel)

        size = torch.pow(2.0, (cb["y_meta"][:, 0, :] - REF_DEPTH).clamp(-4, 4))[:, None, :]
        log_enr_pred = torch.log((cnt_mean / size.clamp(min=1e-6)).clamp(min=1e-3))
        cnt_enr_pe = self._pearson_global(log_enr_pred, sig_tgt, sel)
        cnt_raw_pe = self._pearson_global(cnt_mean, y.float(), sel)

        pk_logit = torch.logit(out["peak_prob"].clamp(1e-4, 1 - 1e-4))
        bce = Fn.binary_cross_entropy_with_logits(pk_logit[m], cb["y_peaks"][m])

        return (0.65 * nll_t + 0.35 * hub_t
                + 1.3 * pe_gl + 0.9 * pe_pt
                + self.DEV_W * dev
                + self.CONS_W * cons
                + 0.5 * shape_pe
                + 0.5 * cnt_enr_pe
                + 0.4 * cnt_raw_pe
                + 0.1 * bce)

    def configure_optimizer(self, params):
        return torch.optim.AdamW(params, lr=1.5e-3, weight_decay=3e-4)


if __name__ == "__main__":
    print(f"ERA_SCORE: {run_and_score(Model, Objective())}")
    print("ERA_REFLECTION: Kept the node-161 residual-on-reference scaffold intact and added "
          "mask-conditional NB dispersion (widens predictive variance exactly at imputation "
          "positions where ECE is scored), a 3-level position pyramid plus coarse ref/CF "
          "regional features for multi-scale deviation readout, and light count-mean / "
          "smoothed-local Pearson terms to lift the lagging imp_count_pearson without heavy "
          "calibration aux losses that prior nodes showed hurt skill.")