"""Data layer for covariate decodability probes (MERGED dataset).

Instance = one (biosample_dir, assay) experiment. Counts from signal_DSF1_res25/chr{19,21}.npz
(int64 per-25bp-bin, full depth). Labels from per-instance JSON. DNA from data/hg38.fa.
Grouping for splits = base cell type (no replicate leakage).
"""
from __future__ import annotations
import os, re, json, glob
from dataclasses import dataclass
from functools import lru_cache
import numpy as np

MERGED_ROOT = "/home/mforooz/projects/def-maxwl/mforooz/DATA_CANDI_MERGED"
FASTA = "/project/6014832/mforooz/EpiDenoise/data/hg38.fa"
WIN_BINS = 768
RES = 25
WIN_BP = WIN_BINS * RES          # 19200
TRAIN_CHR, TEST_CHR = "chr19", "chr21"

_GRP_RE = re.compile(r'_(grp\d+_)?(rep\d+|nonrep)$')
def base_celltype(biosample: str) -> str:
    return _GRP_RE.sub('', biosample)

def _first(d):
    """file_metadata.json fields are {idx: value}; take the single value."""
    return list(d.values())[0]

@dataclass
class Instance:
    biosample: str
    assay: str
    group: str          # base cell type
    run_type: int       # 0 single, 1 paired
    read_length: float
    log_depth: float
    dsf1_dir: str       # .../signal_DSF1_res25

def census(assays=None) -> list[Instance]:
    """Walk the tree; one Instance per (biosample, assay) with both chr19 & chr21 npz present."""
    out = []
    for bdir in sorted(os.listdir(MERGED_ROOT)):
        bpath = os.path.join(MERGED_ROOT, bdir)
        if not os.path.isdir(bpath):
            continue
        for assay in sorted(os.listdir(bpath)):
            if assay == "chipseq-control":
                continue
            if assays is not None and assay not in assays:
                continue
            adir = os.path.join(bpath, assay)
            dsf1 = os.path.join(adir, "signal_DSF1_res25")
            if not (os.path.exists(os.path.join(dsf1, f"{TRAIN_CHR}.npz")) and
                    os.path.exists(os.path.join(dsf1, f"{TEST_CHR}.npz"))):
                continue
            fm = os.path.join(adir, "file_metadata.json")
            sm = os.path.join(dsf1, "metadata.json")
            if not (os.path.exists(fm) and os.path.exists(sm)):
                continue
            try:
                fmj = json.load(open(fm)); smj = json.load(open(sm))
                rt_str = str(_first(fmj["run_type"])).lower()
                rt = 1 if "pair" in rt_str else (0 if "single" in rt_str else None)
                rl = float(_first(fmj["read_length"]))
                depth = float(smj["depth"])
                if rt is None or depth <= 0:
                    continue
            except Exception:
                continue
            out.append(Instance(bdir, assay, base_celltype(bdir), rt, rl,
                                 float(np.log10(depth)), dsf1))
    return out

# ---- viability (base-cell-type grouping) ----
def _group_runtype(insts):
    """group -> 'S'/'P'/'mixed' by the run_types of its instances."""
    g = {}
    for it in insts:
        g.setdefault(it.group, set()).add(it.run_type)
    return {k: ('mixed' if len(v) > 1 else ('P' if 1 in v else 'S')) for k, v in g.items()}

def runtype_viable(insts, min_class=6) -> bool:
    gt = _group_runtype(insts)
    nS = sum(v == 'S' for v in gt.values()); nP = sum(v == 'P' for v in gt.values())
    return min(nS, nP) >= min_class

def readlength_viable(insts, min_distinct=3, min_std=10.0) -> bool:
    # spread across independent groups (one read_length per group, take the first instance's)
    by_g = {}
    for it in insts:
        by_g.setdefault(it.group, it.read_length)
    vals = list(by_g.values())
    return len(set(vals)) >= min_distinct and (np.std(vals) >= min_std)

# ---- counts ----
def load_counts(inst: Instance, chrom: str) -> np.ndarray:
    return np.load(os.path.join(inst.dsf1_dir, f"{chrom}.npz"))["arr_0"].astype(np.float32)

def instance_total(inst: Instance) -> float:
    return float(load_counts(inst, TRAIN_CHR).sum() + load_counts(inst, TEST_CHR).sum())

def _binom_thin(mat: np.ndarray, p: float, seed: int) -> np.ndarray:
    """Downsample counts by binomial thinning (removes depth: magnitude AND noise/sparsity signature)."""
    if p >= 1.0:
        return mat.astype(np.float32)
    rng = np.random.default_rng(seed)
    return rng.binomial(mat.astype(np.int64), p).astype(np.float32)

# ---- windows ----
def sample_windows(counts: np.ndarray, n_max=512, seed=0):
    """Non-overlapping 768-bin windows; drop all-zero; sample <=n_max. Returns (starts, mat[K,768])."""
    n_win = len(counts) // WIN_BINS
    mat = counts[:n_win * WIN_BINS].reshape(n_win, WIN_BINS)
    nz = np.where(mat.sum(1) > 0)[0]
    if len(nz) > n_max:
        nz = np.random.default_rng(seed).choice(nz, n_max, replace=False)
        nz.sort()
    return nz * WIN_BINS, mat[nz]

# ---- DNA ----
_FA = None
def _fasta():
    global _FA
    if _FA is None:
        import pysam
        _FA = pysam.FastaFile(FASTA)
    return _FA

_MAP = np.full(256, 4, np.uint8)
for i, b in enumerate("ACGT"):
    _MAP[ord(b)] = i; _MAP[ord(b.lower())] = i

@lru_cache(maxsize=200_000)
def dna_onehot(chrom: str, bin_start: int) -> np.ndarray:
    """[4, 19200] float32 one-hot (A,C,G,T; N/other -> 0). Cached per (chrom, bin_start)."""
    s = bin_start * RES
    seq = _fasta().fetch(chrom, s, s + WIN_BP) or ""
    idx = _MAP[np.frombuffer(seq.encode("ascii", "ignore"), np.uint8)]
    oh = np.zeros((4, WIN_BP), np.float32)
    valid = idx < 4
    pos = np.nonzero(valid)[0]
    oh[idx[valid], pos] = 1.0
    return oh

# ---- pool / split / dataset (training assembly) ----
def build_pool(insts, n_win=256):
    """One I/O pass: sample chr19 & chr21 windows (raw), then thin every instance to the common
    minimum depth (`thin`) to control depth. Stores both raw and depth-controlled windows."""
    pool = []
    for k, it in enumerate(insts):
        rec = dict(inst=k, group=it.group, run_type=it.run_type,
                   read_length=it.read_length, log_depth=it.log_depth, total=0.0)
        for chrom in (TRAIN_CHR, TEST_CHR):
            counts = load_counts(it, chrom)          # single load
            rec["total"] += float(counts.sum())
            starts, mat = sample_windows(counts, n_win, seed=k)
            rec[chrom] = dict(starts=starts, raw=mat)
        pool.append(rec)
    totals = [r["total"] for r in pool]
    T = min(totals)                                  # thinning: common target depth (binomial)
    C = float(np.median(totals))                     # scaling: common total (sum-normalization)
    for r in pool:
        r["p"] = T / r["total"] if r["total"] > 0 else 0.0       # thin prob
        r["scale"] = C / r["total"] if r["total"] > 0 else 0.0   # sum-norm scale
        for chrom in (TRAIN_CHR, TEST_CHR):
            r[chrom]["thin"] = _binom_thin(r[chrom]["raw"], r["p"], seed=r["inst"])
    pool_T[0], pool_C[0] = T, C
    return pool

pool_T = [None]  # last thinning target depth (diagnostics)
pool_C = [None]  # last scaling common total (diagnostics)

def _group_labels(insts, target):
    """group -> class. run_type: 'S'/'P' (mixed excluded). read_length/depth: tertile bins."""
    by_g = {}
    for it in insts:
        by_g.setdefault(it.group, []).append(it)
    if target == "run_type":
        return {g: c for g, c in _group_runtype(insts).items() if c in ('S', 'P')}
    key = (lambda it: it.read_length) if target == "read_length" else (lambda it: it.log_depth)
    gv = {g: float(np.mean([key(it) for it in its])) for g, its in by_g.items()}
    qs = np.quantile(list(gv.values()), [1/3, 2/3])
    return {g: int(np.digitize(v, qs)) for g, v in gv.items()}

def stratified_group_split(insts, target, seed, test_frac=0.3):
    labels = _group_labels(insts, target)
    rng = np.random.default_rng(seed)
    byc = {}
    for g, c in labels.items():
        byc.setdefault(c, []).append(g)
    train, test = set(), set()
    for c, gs in byc.items():
        gs = sorted(gs); rng.shuffle(gs)
        ntest = max(1, int(round(len(gs) * test_frac)))
        ntest = min(ntest, len(gs) - 1) if len(gs) > 1 else 0
        test.update(gs[:ntest]); train.update(gs[ntest:])
    return train, test

def assemble(pool, groups, chrom, target, norm, y_by_group=None):
    """Flatten windows of instances whose group in `groups`, on `chrom`. Returns (counts, starts, y, inst).
    norm: 'raw' (full depth) | 'scaled' (sum-normalized to common total) | 'thin' (binomial to common depth).
    y_by_group: optional {group: label} override (label-shuffle control)."""
    assert norm in ("raw", "scaled", "thin")
    counts, starts, y, inst = [], [], [], []
    for rec in pool:
        if rec["group"] not in groups:
            continue
        d = rec[chrom]; k = len(d["starts"])
        if k == 0:
            continue
        c = d["thin"] if norm == "thin" else (d["raw"] * rec["scale"] if norm == "scaled" else d["raw"])
        counts.append(c); starts.append(d["starts"])
        inst.append(np.full(k, rec["inst"]))
        if y_by_group is not None:
            yi = y_by_group[rec["group"]]
        else:
            yi = {"run_type": rec["run_type"], "read_length": rec["read_length"],
                  "depth": rec["log_depth"]}[target]
        y.append(np.full(k, yi, np.float32))
    if not counts:
        return None
    return (np.concatenate(counts), np.concatenate(starts),
            np.concatenate(y), np.concatenate(inst))

def dna_gpu(starts, chrom, device):
    """Resident DNA: unique one-hots [U,4,WIN_BP] on `device` + per-window index [N] (gather per batch).
    U is bounded by the tiling grid (~3k/chr), independent of #instances, so this fits the MIG slice."""
    import torch
    uniq, inv = np.unique(starts, return_inverse=True)
    arr = np.stack([dna_onehot(chrom, int(s)) for s in uniq])  # [U,4,WIN_BP]
    return torch.from_numpy(arr).to(device), torch.from_numpy(inv).to(device)
