"""candi_kit report generator — reads a results `*.json` ONLY and regenerates the figures + report.md.

VENDORED (COPY+EDIT) of sandbox/diagnostics/dual_conditioning_real/report.py:1-225.

Fully regenerable (no re-inference). Figures are best-effort (matplotlib lives in the [report] extra and
needs MPLBACKEND=Agg); the markdown scorecard + inventory tables always render. Deterministic.

THE ASSAY ORDER IS NEVER DECLARED HERE. Labels come from the results JSON — `config.assays` (the order
resolved at train time from the h5 attrs) — and the held-out-target inventory is reconstructed from the
per-target records rather than from a hand-written fixture.

The scorecard reports the VALIDATED subset only. Keys emitted under `--include-deprecated` are printed
in a separate, explicitly labelled table together with the verdict the run recorded for them.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


def _f(x, nd=3):
    try:
        v = float(x)
        return "nan" if (v != v) else f"{v:.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def _assays(res) -> List[str]:
    """Resolved assay order, in this preference: the config echo, the evaluate() echo, then the per-assay
    metric keys. Never a literal."""
    for src in (res.get("config", {}).get("assays"), res.get("assays")):
        if src:
            return list(src)
    return list(res.get("M1", {}).get("imp_per_assay", {}).keys())


def _savefig(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _fig_m1(m1, figs: Path) -> Optional[str]:
    if not HAVE_MPL:
        return None
    imp, den = m1.get("imp", {}), m1.get("den", {})
    keys = ["spearman_raw", "pearson_log1p", "crps", "ece"]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    x = np.arange(len(keys))
    ax.bar(x - 0.2, [imp.get(k, np.nan) for k in keys], 0.4, label="imp")
    ax.bar(x + 0.2, [den.get(k, np.nan) for k in keys], 0.4, label="den")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(keys, fontsize=7)
    ax.set_title("F5 · M1 counts-only quality"); ax.legend()
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
    """Target-CLUSTERED direction CIs — the only inference primitive in the kit. The position-level bars
    the original plotted were ~24x too narrow."""
    if not HAVE_MPL:
        return None
    covs = [c for c in ("run_type", "read_length", "depth") if c in m2]
    means, los, his = [], [], []
    for c in covs:
        d = m2.get(c, {})
        agg = d.get("overall_clustered") or d.get("direction_clustered") or {}
        means.append(agg.get("mean", np.nan)); los.append(agg.get("lo", np.nan)); his.append(agg.get("hi", np.nan))
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    x = np.arange(len(covs)); m = np.array(means, float)
    yerr = np.vstack([m - np.array(los, float), np.array(his, float) - m])
    ax.bar(x, m, 0.5, yerr=np.abs(yerr), capsize=4)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(covs)
    ax.set_ylabel("CRPS(flip) − CRPS(true)")
    ax.set_title("F1 · M2 direction, target-clustered (true prompt better ⇒ >0)")
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
    a1.set_xlabel("told depth"); a1.set_ylabel("CRPS vs GT"); a1.set_title("F2 · depth CRPS-vs-told")
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


def _fig_per_assay_crps(m1, assays: List[str], figs: Path) -> Optional[str]:
    """Per-assay CRPS split into the capability term and the fixable per-assay scale term, labelled with
    the assay order the run actually resolved."""
    if not HAVE_MPL:
        return None
    pa = m1.get("imp_per_assay", {})
    names = [a for a in assays if a in pa] or list(pa.keys())
    if not names:
        return None
    fig, ax = plt.subplots(figsize=(1.0 + 0.7 * len(names), 3.2))
    x = np.arange(len(names))
    oracle = [pa[a].get("crps_oracle_scaled", np.nan) for a in names]
    scale = [pa[a].get("scale_error", np.nan) for a in names]
    ax.bar(x, oracle, 0.6, label="crps_oracle_scaled (capability)")
    ax.bar(x, scale, 0.6, bottom=oracle, label="scale_error (calibration)")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("CRPS"); ax.set_title("F8 · per-assay imp CRPS decomposition"); ax.legend(fontsize=7)
    p = figs / "F8_per_assay_crps.png"; _savefig(fig, p); return p.name


def _scorecard(res) -> str:
    m1, m2, m3 = res.get("M1", {}), res.get("M2", {}), res.get("M3", {})
    s14 = res.get("S14", {})
    rt, dp = m2.get("run_type", {}), m2.get("depth", {})
    rt_cl, dp_cl = rt.get("overall_clustered", {}), dp.get("direction_clustered", {})
    abl = m2.get("ablation", {})
    abl_null = m2.get("ablation_within_batch", {})
    rows = [
        "| metric | value | bar |",
        "|---|---|---|",
        f"| M1 imp macro CRPS | {_f(m1.get('imp_macro_crps'))} | lower is better |",
        f"| M1 imp macro CRPS, oracle-scaled (CAPABILITY) | {_f(m1.get('imp_macro_crps_oracle_scaled'))} "
        f"| in-sample oracle ⇒ an upper bound |",
        f"| M1 imp macro scale_error (FIXABLE CALIBRATION) | {_f(m1.get('imp_macro_scale_error'))} "
        f"| may go slightly negative |",
        f"| M1 imp macro Spearman (raw counts) | {_f(m1.get('imp_macro_spearman_raw'))} | healthy band |",
        f"| M1 den macro Spearman (raw counts) | {_f(m1.get('den_macro_spearman_raw'))} | — |",
        f"| M1 imp pooled Spearman (raw counts) | {_f(m1.get('imp', {}).get('spearman_raw'))} | — |",
        f"| M1 imp pooled Pearson (log1p) | {_f(m1.get('imp', {}).get('pearson_log1p'))} | > 0 |",
        f"| M1 imp PIT-ECE | {_f(m1.get('imp', {}).get('ece'))} | ≲ 0.10 |",
        # Print the DENOMINATOR. A bare "1" against the reference panel's documented 7/8 reads as a
        # catastrophe when it is actually 1/1 -- a 3-assay panel simply has fewer held-out assays.
        f"| M1 imp beats honest marginal | {m1.get('imp_beats_marginal_n')}/{len(m1.get('imp_per_assay') or {})} assays | strict `<` |",
        f"| M1 encoder eff-rank (per-position) | {_f(m1.get('encoder_eff_rank_perpos'), 2)} | > 1 |",
        f"| M2 depth median total slope | {_f(dp.get('median_total_slope'))} | → 1.0 |",
        f"| M2 depth total_slope_err | {_f(dp.get('total_slope_err'))} | → 0 |",
        f"| M2 depth clamp: frac targets any clamp | {_f(dp.get('frac_targets_any_clamp'), 3)} "
        f"| a slope read through the clamp is not an exposure coefficient |",
        f"| M2 depth clamp: p90 / max frac at clamp | {_f(dp.get('p90_frac_log2mu_at_clamp'), 3)} / "
        f"{_f(dp.get('max_frac_log2mu_at_clamp'), 3)} | — |",
        f"| M2 depth clamp saturated | {dp.get('total_slope_clamp_saturated')} | must be False to read the slope |",
        f"| M2 depth direction (clustered) | {_f(dp_cl.get('mean'))} "
        f"[{_f(dp_cl.get('lo'))}, {_f(dp_cl.get('hi'))}] | supports_direction={dp_cl.get('supports_direction')} |",
        f"| M2 depth direction sign-test p | {_f(dp_cl.get('sign_test_p'), 4)} "
        f"| n_clusters={dp_cl.get('n_clusters')} (10/12 ⇒ p=0.039) |",
        f"| M2 run_type direction (clustered) | {_f(rt_cl.get('mean'))} "
        f"[{_f(rt_cl.get('lo'))}, {_f(rt_cl.get('hi'))}] | supports_direction={rt_cl.get('supports_direction')} |",
        f"| M2 run_type direction sign-test p | {_f(rt_cl.get('sign_test_p'), 4)} "
        f"| n_clusters={rt_cl.get('n_clusters')} |",
        f"| M2 run_type responsiveness | {_f(rt.get('mean_responsiveness'))} | GT-free prompt-induced shift |",
        f"| M2 run_type model_unresponsive | {rt.get('model_unresponsive')} "
        f"| mean abs(μ_true−μ_flip) < 1e-6 — a MODEL statistic |",
        f"| M3 within/between ratio | {_f(m3.get('ratio'))} | ≤ 0.3 (same-region pairs excluded) |",
        f"| M3 encoder eff-rank (pooled) | {_f(m3.get('encoder_eff_rank_pooled'), 2)} | > 1 (guard) |",
        f"| M3 invariance_ok | {m3.get('invariance_ok')} | — |",
        f"| S14 frac_min_at_true | {_f(s14.get('frac_min_at_true'), 3)} "
        f"| 0.25 is NOT chance; a perfect model caps at ≈0.73 |",
        f"| S14 frac_beats_told1 | {_f(s14.get('frac_beats_told1'), 3)} | > 0.5 |",
    ]
    for row, d in sorted(abl.items()):
        rows.append(
            f"| M2 ablation `{row}` (sentinel-free cross-target) | mean absΔη={_f(d.get('mean_abs_d_eta'), 4)} "
            f"max absΔη={_f(d.get('max_abs_d_eta'), 4)} ΔCRPS={_f(d.get('mean_d_crps'), 4)} "
            f"| n_sentinel_skipped={d.get('n_sentinel_skipped')} (0 on a full panel) |")
    for row, d in sorted(abl_null.items()):
        rows.append(
            f"| M2 ablation `{row}` STRUCTURAL NULL (within-batch) | mean absΔη={_f(d.get('mean_abs_d_eta'), 6)} "
            f"| must read exactly 0 |")
    return "\n".join(rows)


def _deprecated_table(res) -> str:
    """Everything emitted under --include-deprecated, printed with the verdict the run recorded. These
    keys may not back a claim."""
    m2 = res.get("M2", {})
    rows, seen = [], False
    for scope in ("depth", "run_type", "read_length"):
        d = m2.get(scope, {})
        verdicts = d.get("deprecated_verdicts", {})
        for k, v in sorted(verdicts.items()):
            seen = True
            val = d.get(k)
            val = _f(val.get("mean")) if isinstance(val, dict) else (_f(val) if val is not None else "—")
            rows.append(f"| `{scope}.{k}` | {val} | {v} |")
    for k, v in sorted(m2.get("deprecated_verdicts", {}).items()):
        seen = True
        rows.append(f"| `{k}` | (arm emitted) | {v} |")
    if not seen:
        return "_none — the run was scored without `--include-deprecated`._"
    return "\n".join(["| key | value | verdict |", "|---|---|---|"] + rows)


def _inventory_table(res) -> str:
    """Held-out imputation targets, reconstructed FROM THE RESULTS (not a fixture): the depth sweep and
    the run_type flip both record `(T_ biosample, imp biosample, assay)` per target."""
    m2 = res.get("M2", {})
    seen = {}
    for scope in ("run_type", "depth"):
        for t in m2.get(scope, {}).get("per_target", []):
            key = tuple(t.get("target", []))
            if len(key) != 3:
                continue
            rec = seen.setdefault(key, dict(run_type="?", n_fg=None, flagged=False))
            if t.get("run_type") in ("single", "paired"):
                rec["run_type"] = t["run_type"]
            if rec["n_fg"] is None:
                rec["n_fg"] = t.get("n_fg")
            rec["flagged"] = rec["flagged"] or bool(t.get("purity_fallback_fired"))
    if not seen:
        return "_no per-target records in this results file._"
    rows = ["| T_ biosample | imp biosample | assay | run_type | n_fg | purity fallback |",
            "|---|---|---|---|---|---|"]
    for (t, ib, assay), rec in sorted(seen.items()):
        rows.append(f"| {t} | {ib} | {assay} | {rec['run_type']} | {rec['n_fg']} | {rec['flagged']} |")
    rows.append("")
    rows.append(f"_{len(seen)} targets. Records with `purity fallback` = True were scored on background "
                f"(the `target >= 1` filter was dropped) and are excluded from the clustered CIs._")
    return "\n".join(rows)


def generate(results_json, outdir: Optional[Path] = None) -> Path:
    results_json = Path(results_json)
    res = json.loads(results_json.read_text())
    outdir = Path(outdir) if outdir else results_json.parent / f"{results_json.stem}_report"
    figs = outdir / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    assays = _assays(res)

    names = {}
    names["F1"] = _fig_m2_direction(res.get("M2", {}), figs)
    names["F2F7"] = _fig_depth_curves(res.get("M2", {}), figs)
    names["F3"] = _fig_runtype_scatter(res.get("M2", {}), figs)
    names["F4"] = _fig_m3(res.get("M3", {}), figs)
    names["F5"] = _fig_m1(res.get("M1", {}), figs)
    names["F6"] = _fig_pit(res.get("M1", {}), figs)
    names["F8"] = _fig_per_assay_crps(res.get("M1", {}), assays, figs)

    cfg = res.get("config", {})
    md = ["# candi_kit — dual-conditioning run report",
          "",
          f"**tag:** `{cfg.get('tag', results_json.stem)}` · offset={cfg.get('use_offset')} · "
          f"dsf={cfg.get('dsf_sampling')} · epochs={cfg.get('epochs')} · seed={cfg.get('seed')} · "
          f"n_units={res.get('n_units')} · wall={res.get('wall_s')}s",
          "",
          f"**resolved panel:** {len(assays)} assays · `{', '.join(assays) if assays else '—'}` · "
          f"context_bins={cfg.get('context_bins')} · resolution={cfg.get('resolution')} · "
          f"d_model={cfg.get('d_model')} · depth_center={cfg.get('depth_center')} · "
          f"dsf_list={cfg.get('dsf_list')}",
          "",
          "> Effective replication is the TARGET, not the position. Read the clustered CIs and the sign "
          "test; a single seed moves pooled imp CRPS by ~0.12, so gaps below that are uninterpretable.",
          "",
          "## T1 · Scorecard (validated subset)",
          "", _scorecard(res), "",
          "## T2 · Held-out imputation targets",
          "", _inventory_table(res), "",
          "## T3 · Deprecated keys (may NOT back a claim)",
          "", _deprecated_table(res), ""]
    if HAVE_MPL:
        md += ["## Figures", ""]
        for key, cap in [("F1", "F1 · M2 direction, target-clustered CIs"),
                         ("F2F7", "F2 · depth CRPS-vs-told-depth · F7 · η vs told-depth"),
                         ("F3", "F3 · run_type flip CRPS(true) vs CRPS(flip), single/paired"),
                         ("F4", "F4 · M3 within/between latent cos-dist"),
                         ("F5", "F5 · M1 counts-only quality"),
                         ("F6", "F6 · PIT reliability"),
                         ("F8", "F8 · per-assay imp CRPS: capability vs calibration")]:
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
