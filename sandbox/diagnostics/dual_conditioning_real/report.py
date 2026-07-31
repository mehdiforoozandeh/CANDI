"""q19 report generator — reads a `results/*.json` ONLY and regenerates F1-F7 + T1-T2 + report.md.

Fully regenerable (no re-inference). Figures are best-effort (skipped if matplotlib is unavailable);
the markdown scorecard + inventory tables always render. Deterministic.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

from sandbox.diagnostics.dual_conditioning_real.tests.fixture import HELD_OUT_TARGETS


def _f(x, nd=3):
    try:
        v = float(x)
        return "nan" if (v != v) else f"{v:.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def _savefig(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _fig_m1(m1, figs: Path) -> Optional[str]:
    if not HAVE_MPL:
        return None
    imp, den = m1.get("imp", {}), m1.get("den", {})
    keys = ["spearman", "pearson", "crps", "ece"]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    x = np.arange(len(keys))
    ax.bar(x - 0.2, [imp.get(k, np.nan) for k in keys], 0.4, label="imp")
    ax.bar(x + 0.2, [den.get(k, np.nan) for k in keys], 0.4, label="den")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(keys); ax.set_title("F5 · M1 counts-only quality"); ax.legend()
    p = figs / "F5_m1_quality.png"; _savefig(fig, p); return p.name


def _fig_pit(m1, figs: Path) -> Optional[str]:
    if not HAVE_MPL:
        return None
    imp = m1.get("imp", {})
    grid, fbar = imp.get("calib_grid"), imp.get("calib_fbar")
    if not grid or not fbar:
        return None
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.plot(grid, fbar, "-o", ms=3)
    ax.set_xlabel("nominal u"); ax.set_ylabel("F̄(u)"); ax.set_title("F6 · PIT reliability (imp)")
    p = figs / "F6_pit.png"; _savefig(fig, p); return p.name


def _fig_m2_direction(m2, figs: Path) -> Optional[str]:
    if not HAVE_MPL:
        return None
    covs = ["run_type", "read_length", "depth"]
    means, los, his = [], [], []
    for c in covs:
        d = m2.get(c, {})
        agg = d.get("overall") or d.get("direction") or {}
        means.append(agg.get("mean", np.nan)); los.append(agg.get("lo", np.nan)); his.append(agg.get("hi", np.nan))
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    x = np.arange(len(covs)); m = np.array(means, float)
    yerr = np.vstack([m - np.array(los, float), np.array(his, float) - m])
    ax.bar(x, m, 0.5, yerr=np.abs(yerr), capsize=4)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(covs)
    ax.set_ylabel("CRPS(flip) − CRPS(true)"); ax.set_title("F1 · M2 direction (true prompt better ⇒ >0)")
    p = figs / "F1_m2_direction.png"; _savefig(fig, p); return p.name


def _fig_depth_curves(m2, figs: Path) -> Optional[str]:
    if not HAVE_MPL:
        return None
    pts = m2.get("depth", {}).get("per_target", [])
    if not pts:
        return None
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.3))
    for t in pts:
        td, cc, em = t.get("told_depth"), t.get("crps_curve"), t.get("eta_means")
        lab = t["target"][2] if isinstance(t.get("target"), list) and len(t["target"]) == 3 else "?"
        if td and cc:
            a1.plot(td, cc, "-o", ms=3, label=lab)
        if td and em:
            a2.plot(td, em, "-o", ms=3, label=lab)
    a1.set_xlabel("told depth"); a1.set_ylabel("CRPS vs GT"); a1.set_title("F2 · depth CRPS-vs-told (min@true)")
    a2.set_xlabel("told depth"); a2.set_ylabel("mean η (offset-independent)"); a2.set_title("F7 · η vs told depth")
    a1.legend(fontsize=6)
    p = figs / "F2_F7_depth.png"; _savefig(fig, p); return p.name


def _fig_runtype_scatter(m2, figs: Path) -> Optional[str]:
    if not HAVE_MPL:
        return None
    pts = m2.get("run_type", {}).get("per_target", [])
    if not pts:
        return None
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    for lab, col in (("single", "tab:orange"), ("paired", "tab:blue")):
        xs = [t["crps_true"] for t in pts if t.get("run_type") == lab]
        ys = [t["crps_flip"] for t in pts if t.get("run_type") == lab]
        if xs:
            ax.scatter(xs, ys, c=col, label=lab, s=30)
    lim = [0, max([t.get("crps_flip", 0) for t in pts] + [t.get("crps_true", 0) for t in pts] + [1e-6]) * 1.1]
    ax.plot(lim, lim, "k--", lw=0.7)
    ax.set_xlabel("CRPS(true)"); ax.set_ylabel("CRPS(flip)"); ax.set_title("F3 · run_type flip"); ax.legend()
    p = figs / "F3_runtype.png"; _savefig(fig, p); return p.name


def _fig_m3(m3, figs: Path) -> Optional[str]:
    if not HAVE_MPL:
        return None
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.bar(["within", "between"], [m3.get("within", np.nan), m3.get("between", np.nan)],
           color=["tab:green", "tab:gray"])
    ax.set_title(f"F4 · M3 latent cos-dist (ratio={_f(m3.get('ratio'))})")
    p = figs / "F4_m3.png"; _savefig(fig, p); return p.name


def _scorecard(res) -> str:
    m1, m2, m3 = res.get("M1", {}), res.get("M2", {}), res.get("M3", {})
    rt, rl, dp = m2.get("run_type", {}), m2.get("read_length", {}), m2.get("depth", {})
    rows = [
        "| metric | value | verifiable |",
        "|---|---|---|",
        f"| M1 imp Spearman | {_f(m1.get('imp', {}).get('spearman'))} | h40: healthy band |",
        f"| M1 imp Pearson | {_f(m1.get('imp', {}).get('pearson'))} | h40: > 0 |",
        f"| M1 den Spearman | {_f(m1.get('den', {}).get('spearman'))} | h40: den ≥ imp |",
        f"| M1 health gate (den≥imp) | {m1.get('health_gate_den_ge_imp')} | h40 |",
        f"| M1 imp CRPS | {_f(m1.get('imp', {}).get('crps'))} | h40: ≤ marginal ({_f(m1.get('marginal_crps'))}) |",
        f"| M1 imp PIT-ECE | {_f(m1.get('imp', {}).get('ece'))} | h40: ≲ 0.10 |",
        f"| M1 encoder eff-rank | {_f(m1.get('encoder_eff_rank'), 2)} | h40: > 1 |",
        f"| M2 depth frac min@true | {_f(dp.get('frac_min_at_true'), 2)} | h41: → 1 |",
        f"| M2 depth median η-slope | {_f(dp.get('median_eta_slope'))} | h41: offset-independent > 0 |",
        f"| M2 depth direction CI excl 0 | {dp.get('direction', {}).get('excludes_zero')} | h41 |",
        f"| M2 depth null Δ (≈0) | {_f(dp.get('null', {}).get('mean'))} | h41: shuffle null |",
        f"| M2 run_type direction CI excl 0 | {rt.get('overall', {}).get('excludes_zero')} | h42 |",
        f"| M2 run_type frac direction | {_f(rt.get('frac_direction'), 2)} | h42 |",
        f"| M2 run_type responsiveness | {_f(rt.get('mean_responsiveness'))} | h42 |",
        f"| M2 run_type natural-var-insufficient | {rt.get('natural_variance_insufficient')} | h42: honest null |",
        f"| M2 read_length direction CI excl 0 | {rl.get('overall', {}).get('excludes_zero')} | h42 (secondary) |",
        f"| M3 within/between ratio | {_f(m3.get('ratio'))} | h43: ≤ 0.3 |",
        f"| M3 encoder eff-rank | {_f(m3.get('encoder_eff_rank'), 2)} | h43: > 1 (guard) |",
        f"| M3 invariance_ok | {m3.get('invariance_ok')} | h43 |",
    ]
    return "\n".join(rows)


def _inventory_table() -> str:
    rows = ["| T_ biosample | imp | assay | idx | run_type |", "|---|---|---|---|---|"]
    for (t, pref, assay, idx, lab) in HELD_OUT_TARGETS:
        rows.append(f"| {t} | {pref} | {assay} | {idx} | {lab} |")
    return "\n".join(rows)


def generate(results_json, outdir: Optional[Path] = None) -> Path:
    results_json = Path(results_json)
    res = json.loads(results_json.read_text())
    outdir = Path(outdir) if outdir else results_json.parent / f"{results_json.stem}_report"
    figs = outdir / "figs"
    figs.mkdir(parents=True, exist_ok=True)

    names = {}
    names["F1"] = _fig_m2_direction(res.get("M2", {}), figs)
    names["F2F7"] = _fig_depth_curves(res.get("M2", {}), figs)
    names["F3"] = _fig_runtype_scatter(res.get("M2", {}), figs)
    names["F4"] = _fig_m3(res.get("M3", {}), figs)
    names["F5"] = _fig_m1(res.get("M1", {}), figs)
    names["F6"] = _fig_pit(res.get("M1", {}), figs)

    cfg = res.get("config", {})
    md = [f"# q19 — dual conditioning on real CANDI sandbox data — report",
          "",
          f"**tag:** `{cfg.get('tag', results_json.stem)}` · offset={cfg.get('use_offset')} · "
          f"dsf={cfg.get('dsf_sampling')} · epochs={cfg.get('epochs')} · n_units={res.get('n_units')} · "
          f"wall={res.get('wall_s')}s",
          "",
          "## T1 · Scorecard (M1 health · M2 flip steering · M3 invariance)",
          "", _scorecard(res), "",
          "## T2 · 12 held-out imputation targets (bios × assay × run_type)",
          "", _inventory_table(), ""]
    if HAVE_MPL:
        md += ["## Figures", ""]
        for key, cap in [("F1", "F1 · M2 direction bars (true vs flip per covariate)"),
                         ("F2F7", "F2 · depth CRPS-vs-told-depth (min@true) · F7 · η vs told-depth"),
                         ("F3", "F3 · run_type flip CRPS(true) vs CRPS(flip), single/paired"),
                         ("F4", "F4 · M3 within/between latent cos-dist"),
                         ("F5", "F5 · M1 counts-only quality"),
                         ("F6", "F6 · PIT reliability")]:
            if names.get(key):
                md += [f"**{cap}**", "", f"![{key}](figs/{names[key]})", ""]
    else:
        md += ["_matplotlib unavailable — figures skipped; tables above are complete._", ""]

    report_md = outdir / "report.md"
    report_md.write_text("\n".join(md))
    return report_md


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("results_json")
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    p = generate(a.results_json, a.outdir)
    print(f"[report] wrote {p}")


if __name__ == "__main__":
    main()
