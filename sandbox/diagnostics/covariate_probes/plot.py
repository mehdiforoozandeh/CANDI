"""Figures for covariate probes. Run: python -m sandbox.diagnostics.covariate_probes.plot"""
from __future__ import annotations
import csv, os, glob, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = "sandbox/diagnostics/covariate_probes"
RESDIR = f"{DIR}/results"

def _load(kind):
    """kind='results' or 'preds'. Merge all per-assay files under results/."""
    files = sorted(glob.glob(f"{RESDIR}/*_preds.tsv")) if kind == "preds" \
        else [f for f in sorted(glob.glob(f"{RESDIR}/*.tsv")) if not f.endswith("_preds.tsv")]
    rows = []
    for f in files:
        rows += list(csv.DictReader(open(f), delimiter="\t"))
    return rows

def _agg(rows, probe, metric, level="biosample"):
    """assay -> {dna: (mean,std)} over seeds."""
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r["probe"] != probe or r["level"] != level:
            continue
        v = r.get(metric, "")
        if v in ("", "nan"):
            continue
        by[r["assay"]][int(r["dna"])].append(float(v))
    return {a: {d: (np.mean(vs), np.std(vs)) for d, vs in dd.items()} for a, dd in by.items()}

def _grouped_bar(ax, agg, title, ylabel, chance=None, ylim=None):
    assays = sorted(agg, key=lambda a: -max((agg[a].get(d, (0,))[0]) for d in (0, 1)))
    x = np.arange(len(assays)); w = 0.38
    for i, d in enumerate((0, 1)):
        m = [agg[a].get(d, (np.nan, 0))[0] for a in assays]
        e = [agg[a].get(d, (0, 0))[1] for a in assays]
        ax.bar(x + (i - 0.5) * w, m, w, yerr=e, capsize=2,
               label=("signal+DNA" if d else "signal-only"))
    if chance is not None:
        ax.axhline(chance, ls="--", c="k", lw=1, label="chance")
    ax.set_xticks(x); ax.set_xticklabels(assays, rotation=45, ha="right", fontsize=8)
    ax.set_title(title); ax.set_ylabel(ylabel); ax.legend(fontsize=8)
    if ylim: ax.set_ylim(*ylim)

def fig1(rows):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _grouped_bar(axes[0], _agg(rows, "run_type", "auroc"),
                 "run_type decodability (per-biosample)", "AUROC", chance=0.5, ylim=(0, 1))
    _grouped_bar(axes[1], _agg(rows, "read_length", "r2"),
                 "read_length decodability (per-biosample)", "R²", chance=0.0)
    fig.tight_layout(); fig.savefig(f"{DIR}/fig1_decodability.png", dpi=130); plt.close(fig)

def fig2(rows):
    modes = [("depth_raw", "raw"), ("depth_scaled", "sum-normalized"), ("depth_thin", "thinned")]
    aggs = {p: _agg(rows, p, "r2") for p, _ in modes}
    assays = sorted(set().union(*[set(a) for a in aggs.values()]))
    x = np.arange(len(assays)); w = 0.27
    fig, ax = plt.subplots(figsize=(13, 4.5))
    for i, (p, lab) in enumerate(modes):
        vals = [aggs[p].get(a, {0: (np.nan, 0)})[0][0] for a in assays]
        ax.bar(x + (i - 1) * w, vals, w, label=lab)
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(assays, rotation=45, ha="right", fontsize=8)
    ax.set_title("Depth recoverability R² by normalization "
                 "(raw=present · sum-norm=NOT removed · thinned=removed)")
    ax.set_ylabel("depth R² (per-biosample)"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{DIR}/fig2_depth_validator.png", dpi=130); plt.close(fig)

def fig3(preds):
    rows = [p for p in preds if p["probe"] == "read_length" and p["dna"] == "1"]
    if not rows: return
    assays = sorted(set(p["assay"] for p in rows))
    n = len(assays); ncol = 4; nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3*ncol, 3*nrow), squeeze=False)
    for k, a in enumerate(assays):
        ax = axes[k // ncol][k % ncol]
        yt = np.array([float(p["y_true"]) for p in rows if p["assay"] == a])
        yp = np.array([float(p["y_pred"]) for p in rows if p["assay"] == a])
        ax.scatter(yt, yp, s=10, alpha=0.6)
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_title(a, fontsize=9); ax.set_xlabel("true bp", fontsize=8); ax.set_ylabel("pred bp", fontsize=8)
    for k in range(n, nrow*ncol): axes[k//ncol][k%ncol].axis("off")
    fig.suptitle("read_length: predicted vs true (signal+DNA, per-biosample)")
    fig.tight_layout(); fig.savefig(f"{DIR}/fig3_readlen_scatter.png", dpi=130); plt.close(fig)

def main():
    rows = _load("results"); preds = _load("preds")
    if not rows:
        print(f"no results under {RESDIR}/"); return
    fig1(rows); fig2(rows); fig3(preds)
    print("wrote fig1_decodability.png, fig2_depth_validator.png, fig3_readlen_scatter.png")

if __name__ == "__main__":
    main()
