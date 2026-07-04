"""CANDI v3 — FROZEN direct-H5 data provider (the single eval/data path).

Reads `sandbox/data/sandbox.h5` directly (layout from `sandbox/prepare_h5.py`) so the
baseline and the neural harness score on *identical* windows with full control. This is the
authoritative v3 eval protocol — do not edit during a search.

Biosample/assay split (see PLAN.md §2.5): each biosample `name` has up to three H5 groups
`T_<name>` (train/observed), `V_<name>`, `B_<name>` (reserved). Assay `a` is available in a
group iff `meta_dsf1[0, a] != -1`. Real imputation = predict the reserved V_/B_ assays of a
biosample (never in its T_) from that biosample's T_ assays.

Arrays per group: counts_dsf{1,2,4,8} [W,768,8] int, pval [W,768,8] f16 (arcsinh -log10 p),
peaks [W,768,8], meta_dsf{1,2,4,8} [4,8] (depth_log2, assay_idx, read_len, run_type), dna
[W,19200,4], control [W,768,1].
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np

# 8 sandbox assays, in column order (from h5 attrs selection_json["assays"]).
ASSAYS = ["ATAC-seq", "DNase-seq", "H3K4me3", "H3K4me1",
          "H3K27ac", "H3K27me3", "H3K36me3", "H3K9me3"]
F = len(ASSAYS)
L = 768          # signal bins per window
DNA_W = 19200    # bp per window (25 bp/bin * 768)
MISSING = -1.0   # meta sentinel for an unavailable assay
DEN_DSF = 8      # downsampling factor for the denoising INPUT (low depth -> predict dsf1 high depth)


def _base(name: str) -> str:
    return name.split("_", 1)[1]


@dataclass
class BiosampleEval:
    """Per-biosample eval tensors on the chr21 windows (numpy)."""
    name: str                       # base name, e.g. "DND-41"
    t_avail: np.ndarray             # [F] bool — assays available in T_
    t_pval: np.ndarray              # [W,L,F] f32 — T_ arcsinh-pval (avail cols meaningful)
    t_counts: np.ndarray            # [W,L,F] f32 — T_ counts at dsf1 (denoising TARGET, high depth)
    t_meta: np.ndarray              # [4,F] f32 — T_ covariates (dsf1, target depth)
    t_counts_lo: np.ndarray         # [W,L,F] f32 — T_ counts at DEN_DSF (denoising INPUT, low depth)
    t_meta_lo: np.ndarray           # [4,F] f32 — T_ covariates at DEN_DSF (low input depth)
    t_peaks: np.ndarray             # [W,L,F] f32 — T_ peaks {0,1} (for the baseline's avg-peak track)
    imp_avail: np.ndarray           # [F] bool — assays reserved in V_/B_ (the imp targets)
    imp_pval: np.ndarray            # [W,L,F] f32 — held-out GT arcsinh-pval (imp cols only)
    imp_counts: np.ndarray          # [W,L,F] f32 — held-out GT counts (dsf1)
    imp_peaks: np.ndarray           # [W,L,F] f32 — held-out GT peaks {0,1}
    imp_meta: np.ndarray            # [4,F] f32 — held-out assay covariates (for conditioning)


@dataclass
class EvalData:
    win_idx: np.ndarray             # [W] global window indices (chr21, sorted)
    biosamples: List[BiosampleEval]


def _read_group(h5: h5py.File, gname: str, wsel: np.ndarray):
    """Return (avail[F]bool, pval[W,L,F], counts[W,L,F], peaks[W,L,F], meta[4,F])."""
    g = h5["biosamples"][gname]
    meta = np.asarray(g["meta_dsf1"], dtype=np.float32)          # [4,F]
    avail = meta[0] != MISSING                                   # [F]
    pval = np.asarray(g["pval"][wsel], dtype=np.float32)         # [W,L,F]
    counts = np.asarray(g["counts_dsf1"][wsel], dtype=np.float32)
    peaks = np.asarray(g["peaks"][wsel], dtype=np.float32)
    return avail, pval, counts, peaks, meta


def _win_indices(h5: h5py.File, *, train: bool) -> np.ndarray:
    """Train = chr19 windows (type1 regime); eval = chr21."""
    ch = np.array([x.decode() if isinstance(x, bytes) else x for x in h5["windows/chrom"][:]])
    return np.where(ch == ("chr19" if train else "chr21"))[0]


def _t_bases(h5: h5py.File) -> List[str]:
    order = json.loads(h5["biosamples"].attrs["order"])
    return sorted({_base(n) for n in order if n.startswith("T_")})


def train_steps_per_epoch(h5_path: str | Path, batch_size: int) -> int:
    """One epoch = every chr19 window of every T_ biosample seen once (matches iter_train_batches)."""
    import math
    with h5py.File(Path(h5_path), "r") as h5:
        n_win = len(_win_indices(h5, train=True))
        n_bios = len(_t_bases(h5))
    return n_bios * math.ceil(n_win / batch_size)


def read_dna_control(h5: h5py.File, gname: str, gidx: np.ndarray):
    """DNA [B,DNA_W,4] f32 one-hot and control [B,L,1] f32 for global window indices `gidx`."""
    g = h5["biosamples"][gname]
    gidx = np.asarray(gidx)
    order = np.argsort(gidx); inv = np.argsort(order)           # h5 needs increasing index
    dna = np.asarray(g["dna"][gidx[order]], dtype=np.float32)[inv]
    ctrl = np.asarray(g["control"][gidx[order]], dtype=np.float32)[inv]
    return dna, ctrl


def iter_train_batches(h5_path: str | Path, batch_size: int, *, seed: int = 0):
    """Yield CLEAN train batches (T_ biosamples, chr19) for `Objective.corrupt` to corrupt.

    Each batch dict (numpy): counts [B,L,F] = dsf1 (full depth) + counts_dsf{2,4,8} (depth-
    downsampled, for denoising); meta [B,4,F] = dsf1 + meta_dsf{2,4,8} (depth_log2 lower per
    level); avail [B,F], dna [B,DNA_W,4], control [B,L,1], ctrl_avail [B]. An Objective can build
    (low-depth input -> high-depth target) denoising pairs from the dsf levels (CRITIQUE §4).
    Only T_-available assays carry real data; others are 0/avail=0. Infinite generator.
    """
    h5_path = Path(h5_path)
    rng = np.random.default_rng(seed)
    DSF = (1, 2, 4, 8)
    with h5py.File(h5_path, "r") as h5:
        widx = _win_indices(h5, train=True)
        bases = _t_bases(h5)
        items = [(b, s) for b in bases for s in range(0, len(widx), batch_size)]
        while True:
            rng.shuffle(items)
            for b, s in items:
                gname = f"T_{b}"
                g = h5["biosamples"][gname]
                meta = np.asarray(g["meta_dsf1"], dtype=np.float32)         # [4,F]
                avail = (meta[0] != MISSING).astype(np.float32)            # [F]
                loc = widx[s:s + batch_size]
                counts = np.asarray(g["counts_dsf1"][loc], dtype=np.float32)  # [B,L,F]
                dna, ctrl = read_dna_control(h5, gname, loc)
                B = counts.shape[0]
                out = {
                    "counts": counts,
                    "meta": np.broadcast_to(meta, (B, 4, F)).copy(),
                    "avail": np.broadcast_to(avail, (B, F)).copy(),
                    "dna": dna, "control": ctrl,
                    "ctrl_avail": np.ones(B, dtype=np.float32),
                }
                for k in DSF[1:]:                                          # add dsf2/4/8
                    out[f"counts_dsf{k}"] = np.asarray(g[f"counts_dsf{k}"][loc], dtype=np.float32)
                    mk = np.asarray(g[f"meta_dsf{k}"], dtype=np.float32)
                    out[f"meta_dsf{k}"] = np.broadcast_to(mk, (B, 4, F)).copy()
                yield out


def load_eval(h5_path: str | Path, chrom: str = "chr21") -> EvalData:
    """Load the chr21 eval split: per biosample, T_ observed + V_/B_ reserved GT."""
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as h5:
        order = json.loads(h5["biosamples"].attrs["order"])
        ch = np.array([x.decode() if isinstance(x, bytes) else x
                       for x in h5["windows/chrom"][:]])
        wsel = np.where(ch == chrom)[0]                          # sorted ascending
        bases = sorted({_base(n) for n in order if n.startswith("T_")})
        out: List[BiosampleEval] = []
        for b in bases:
            if f"T_{b}" not in h5["biosamples"]:
                continue
            t_avail, t_pval, t_counts, t_peaks, t_meta = _read_group(h5, f"T_{b}", wsel)
            tg = h5["biosamples"][f"T_{b}"]                       # low-depth T_ for denoising input
            t_counts_lo = np.asarray(tg[f"counts_dsf{DEN_DSF}"][wsel], dtype=np.float32)
            t_meta_lo = np.asarray(tg[f"meta_dsf{DEN_DSF}"], dtype=np.float32)
            imp_avail = np.zeros(F, dtype=bool)
            imp_pval = np.zeros_like(t_pval)
            imp_counts = np.zeros_like(t_counts)
            imp_peaks = np.zeros_like(t_pval)
            imp_meta = np.full((4, F), MISSING, dtype=np.float32)
            for pref in ("V_", "B_"):                            # reserved groups
                gname = f"{pref}{b}"
                if gname not in h5["biosamples"]:
                    continue
                av, pv, ct, pk, mt = _read_group(h5, gname, wsel)
                take = av & (~t_avail)                           # held-out: reserved & not in T
                imp_avail |= take
                for a in np.where(take)[0]:
                    imp_pval[:, :, a] = pv[:, :, a]
                    imp_counts[:, :, a] = ct[:, :, a]
                    imp_peaks[:, :, a] = pk[:, :, a]
                    imp_meta[:, a] = mt[:, a]
            out.append(BiosampleEval(
                name=b, t_avail=t_avail, t_pval=t_pval, t_counts=t_counts, t_meta=t_meta,
                t_counts_lo=t_counts_lo, t_meta_lo=t_meta_lo, t_peaks=t_peaks,
                imp_avail=imp_avail, imp_pval=imp_pval, imp_counts=imp_counts,
                imp_peaks=imp_peaks, imp_meta=imp_meta,
            ))
    return EvalData(win_idx=wsel, biosamples=out)
