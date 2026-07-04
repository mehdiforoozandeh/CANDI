"""Generate report.md from the merged sweep results + figures.
Run: python -m sandbox.diagnostics.covariate_probes.report"""
from __future__ import annotations
import numpy as np
from sandbox.diagnostics.covariate_probes.plot import _load, _agg

DIR = "sandbox/diagnostics/covariate_probes"

def _fmt(agg, a, d, dec=3):
    if a in agg and d in agg[a]:
        m, s = agg[a][d]
        return f"{m:.{dec}f}±{s:.{dec}f}"
    return "–"

def main():
    rows = _load("results")
    if not rows:
        open(f"{DIR}/report.md", "w").write("# Covariate probes — no results yet\n")
        print("no results"); return
    rt = _agg(rows, "run_type", "auroc")
    rl = _agg(rows, "read_length", "r2")
    d_raw = _agg(rows, "depth_raw", "r2"); d_sc = _agg(rows, "depth_scaled", "r2"); d_th = _agg(rows, "depth_thin", "r2")
    assays = sorted(set(rt) | set(rl))

    L = []
    L.append("# Covariate decodability probes — results\n")
    L.append("## What was done\n")
    L.append(
        "Per ENCODE assay, we ask whether a track's **run_type** (single/paired) and **read_length** "
        "are recoverable from the *shape* of its read-count signal (768×25 bp = 19.2 kb windows) and DNA "
        "sequence, **after controlling for sequencing depth**. A small CANDI-style encoder "
        "(Conv1D count tower + optional DNA tower → 2-layer transformer → linear head) is trained per "
        "assay/target. No metadata (`x_meta`) is given to the model.\n")
    L.append("## Data\n")
    L.append(
        "MERGED dataset (`DATA_CANDI_MERGED`), full-depth counts `signal_DSF1_res25` at 25 bp; DNA from "
        "hg38. Labels read per-instance from `file_metadata.json`. **Double holdout**: train = chr19 "
        "windows of train cell types, test = chr21 windows of *unseen* cell types (grouping = base cell "
        "type, so technical replicates never straddle the split). 5 stratified group splits; metrics "
        "reported **per-biosample** (window predictions averaged per held-out instance), mean±std over splits.\n")
    L.append("## Depth control & a key finding\n")
    thin_med = float(np.nanmedian([d_th[a][0][0] for a in d_th if 0 in d_th[a]])) if d_th else float("nan")
    L.append(
        "Depth is recoverable from raw counts. **Neither sum-count normalization NOR binomial thinning to "
        "a common depth removes it**: depth stays predictable on *held-out cell types* under both "
        f"(median thinned depth R² ≈ {thin_med:.2f}). The most likely cause is a **depth↔biology confound** "
        "in the cohort — deeper-sequenced samples are systematically different cell types, so the model "
        "reads a biological axis that co-varies with depth, which no count-normalization can strip. "
        "Probes use thinned input (the best available control). Per-assay depth R²:\n")
    L.append("| assay | raw | sum-normalized | thinned |")
    L.append("|---|---|---|---|")
    for a in sorted(set(d_raw) | set(d_th)):
        L.append(f"| {a} | {_fmt(d_raw,a,0)} | {_fmt(d_sc,a,0)} | {_fmt(d_th,a,0)} |")
    L.append("\n![depth validator](fig2_depth_validator.png)\n")
    L.append(
        "> **Caveat for the headline below:** because depth is not fully removable and run_type/read_length "
        "correlate with depth, part of their decodability may reflect the depth↔biology axis rather than a "
        "pure covariate fingerprint. The label-shuffle control (≈0.5) rules out trivial biology "
        "*relabeling*, but not this confound. Recommended follow-up: depth-*matched* contrasts "
        "(subsample so the two run_type classes share a depth distribution).\n")
    L.append("## Controls (from validation)\n")
    L.append(
        "- **label-shuffle** → AUROC ~0.5 (no spurious signal) · **DNA-only** → AUROC ~0.5 (DNA alone "
        "can't see a per-biosample label) · **overfit-tiny** → ~1.0 (model can learn). "
        "See the validation log for exact numbers.\n")
    L.append("## Headline: covariate decodability (per-biosample, depth-controlled)\n")
    L.append("| assay | run_type AUROC (signal) | run_type AUROC (+DNA) | read_length R² (signal) | read_length R² (+DNA) |")
    L.append("|---|---|---|---|---|")
    for a in assays:
        L.append(f"| {a} | {_fmt(rt,a,0)} | {_fmt(rt,a,1)} | {_fmt(rl,a,0)} | {_fmt(rl,a,1)} |")
    L.append("\n![decodability](fig1_decodability.png)\n")
    L.append("![read_length scatter](fig3_readlen_scatter.png)\n")
    L.append("## Interpretation\n")
    L.append(
        "AUROC>0.5 / R²>0 on *unseen cell types & chromosome at common depth* ⇒ a genuine, generalizable "
        "covariate fingerprint exists in the signal shape. A positive **DNA lift** (+DNA > signal-only) "
        "implicates mappability as the mechanism. This is the precondition for CANDI's metadata "
        "conditioning / supertrack goal.\n")
    open(f"{DIR}/report.md", "w").write("\n".join(L) + "\n")
    print(f"wrote {DIR}/report.md")

if __name__ == "__main__":
    main()
