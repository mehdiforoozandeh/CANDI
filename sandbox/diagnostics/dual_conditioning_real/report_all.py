"""q19 consolidated results report — reads the 4 full-coverage arm JSONs and writes ONE `results/report.md`
with all cross-arm figures + tables. The q19 story is comparative (offset ON vs OFF), so the figures are
cross-arm. Deterministic, regenerable, reads JSON only.

  python -m sandbox.diagnostics.dual_conditioning_real.report_all
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sandbox.diagnostics.dual_conditioning_real.tests.fixture import HELD_OUT_TARGETS

RESULTS = Path("sandbox/diagnostics/dual_conditioning_real/results")
FIGS = RESULTS / "report_figs"

# (json tag, short label, role) — order = winning recipe (2 seeds), then the two controls
ARMS = [
    ("main_s0_full", "offset ON · s0", "★ winning recipe"),
    ("main_s1_full", "offset ON · s1", "★ winning recipe (seed 1)"),
    ("offoff_s0_full", "offset OFF", "control — learned steering"),
    ("copyable_s0_full", "x_eq_y", "control — copyable DSF"),
]
COLORS = ["#1f77b4", "#4a90d9", "#d62728", "#7f7f7f"]


def _load():
    out = {}
    for tag, lab, role in ARMS:
        p = RESULTS / f"{tag}.json"
        out[tag] = json.loads(p.read_text()) if p.exists() else None
    return out


def _f(x, nd=3):
    try:
        v = float(x)
        return "nan" if v != v else f"{v:.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGS / name, dpi=120)
    plt.close(fig)
    return name


def _depth_agg(res):
    """Mean eta and MEDIAN (true-depth-normalized) CRPS across depth targets, by told-depth index
    [true, -1, -2, -3]. CRPS is normalized by its value at the TRUE depth (index 0) and aggregated with
    the median so the 24% of targets whose min sits off-true don't blow up the mean (a per-curve
    min-normalization would spuriously push the aggregate minimum away from the true depth)."""
    pts = res["M2"]["depth"]["per_target"]
    etas = np.array([t["eta_means"] for t in pts if len(t.get("eta_means", [])) == 4], float)
    crps = np.array([t["crps_curve"] for t in pts if len(t.get("crps_curve", [])) == 4], float)
    eta_m = etas.mean(0) if etas.size else np.full(4, np.nan)
    crps_rel = (crps / crps[:, 0:1]) if crps.size else np.full((1, 4), np.nan)   # normalize by CRPS@true
    return eta_m, np.median(crps_rel, axis=0)


def _runtype_stats(res):
    """Correct run_type direction stats at instance and unique-biological-target levels (offset-OFF)."""
    import collections
    pts = res["M2"]["run_type"]["per_target"]
    frac_true = lambda s: float(np.mean([t["crps_flip"] > t["crps_true"] for t in s])) if s else float("nan")
    resp = lambda s: float(np.mean([t["responsiveness"] for t in s])) if s else float("nan")
    single = [t for t in pts if t.get("run_type") == "single"]
    paired = [t for t in pts if t.get("run_type") == "paired"]
    agg = collections.defaultdict(lambda: [[], []])
    for t in pts:
        agg[tuple(t["target"])][0].append(t["crps_true"]); agg[tuple(t["target"])][1].append(t["crps_flip"])
    n_true = sum(1 for ct, cf in agg.values() if np.mean(ct) < np.mean(cf))
    return dict(n_single=len(single), n_paired=len(paired), n_targets=len(agg), n_targets_true=n_true,
                frac_true_single=frac_true(single), frac_true_paired=frac_true(paired),
                resp_single=resp(single), resp_paired=resp(paired))


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def fig_eta_slope(data):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    vals = [data[t]["M2"]["depth"]["median_eta_slope"] if data[t] else np.nan for t, _, _ in ARMS]
    ax.bar(range(len(ARMS)), vals, color=COLORS)
    for i, v in enumerate(vals):
        ax.text(i, v + (0.02 if v >= 0 else -0.05), _f(v), ha="center", fontsize=9)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(range(len(ARMS))); ax.set_xticklabels([l for _, l, _ in ARMS], fontsize=8)
    ax.set_ylabel("median η-slope")
    ax.set_title("Fig 1 · η-slope (offset-independent depth lever): ~0 offset-ON, 0.88 OFF")
    return _save(fig, "fig1_eta_slope.png")


def fig_depth_curves(data):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    x = [0, -1, -2, -3]
    for (t, lab, _), c in zip(ARMS, COLORS):
        if not data[t]:
            continue
        eta_m, crps_m = _depth_agg(data[t])
        a1.plot(x, crps_m, "-o", ms=4, color=c, label=lab)
        a2.plot(x, eta_m, "-o", ms=4, color=c, label=lab)
    a1.set_xlabel("told depth − true depth (log2 units)"); a1.set_ylabel("median CRPS / CRPS@true")
    a1.axvline(0, color="k", ls=":", lw=0.7)
    a1.set_title("Fig 2 · Depth CRPS vs told-depth (min at true)"); a1.legend(fontsize=7)
    a2.set_xlabel("told depth − true depth (log2 units)"); a2.set_ylabel("mean η")
    a2.set_title("Fig 3 · η vs told-depth"); a2.legend(fontsize=7)
    return _save(fig, "fig2_3_depth.png")


def fig_runtype(data):
    # NOTE: the run_type direction-frac bar chart was removed — a degenerate 0 (ignored covariate ⇒
    # exactly-0 Δ) reads like "total failure" and is misleading. Responsiveness is the honest panel;
    # the direction-frac number still lives in the scorecard.
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    resp = [data[t]["M2"]["run_type"]["mean_responsiveness"] if data[t] else np.nan for t, _, _ in ARMS]
    ax.bar(range(len(ARMS)), resp, color=COLORS)
    for i, v in enumerate(resp):
        ax.text(i, v, _f(v, 2), ha="center", fontsize=9, va="bottom")
    ax.set_xticks(range(len(ARMS))); ax.set_xticklabels([l for _, l, _ in ARMS], fontsize=8)
    ax.set_ylabel("responsiveness (|Δμ| under flip)")
    ax.set_title("Fig 4 · run_type responsiveness (|Δμ| when the prompt's run_type is flipped)")
    return _save(fig, "fig4_runtype_resp.png")


def _delta_hist(ax, d, b, *, ylabel=None):
    """One signed-Δ histogram: green bars right of 0 (true-better), grey left; out-of-[-b,b] dropped+counted.
    Each histogram entry is one held-out target-assay measured on one chr21 window-batch."""
    bins = np.linspace(-b, b, 49)                 # 0 lands on a bin edge; fine resolution near zero
    centers = 0.5 * (bins[:-1] + bins[1:])
    colors = ["#2ca02c" if c > 0 else "#b7b7b7" for c in centers]
    counts, _ = np.histogram(d, bins=bins)
    ax.bar(centers, counts, width=(bins[1] - bins[0]) * 0.95, color=colors)
    ax.axvline(0, color="k", lw=1.1)
    ax.set_xlim(-b, b)
    ax.tick_params(axis="y", labelsize=7)         # keep y ticks = # measurements per bar
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    n_off = int((np.abs(d) > b).sum())
    ax.text(0.02, 0.94, f"median {np.median(d):+.3f}" + (f"   ({n_off} off-axis)" if n_off else ""),
            transform=ax.transAxes, va="top", fontsize=7.5, color="#333")


def _flip_dist_fig(data, covariate, out_name, suptitle, *, split, arm="offoff_s0_full"):
    """Win-rate + signed-Δ histograms (log-Δ AND raw-Δ) for one flip covariate, offset-OFF arm.
    Δ = CRPS(flip) − CRPS(true) (>0 ⇒ the true prompt imputes better). `split` → single/paired rows;
    else one pooled 'all targets' row (read_length is continuous, not single/paired)."""
    res = data.get(arm)
    if not res:
        return None
    pts = res["M2"][covariate]["per_target"]
    if split:
        groups = ["single", "paired"]
        pick = {g: [t for t in pts if t.get("run_type") == g] for g in groups}
    else:
        groups = ["all targets"]
        pick = {"all targets": list(pts)}
    logd, rawd = {}, {}
    for g in groups:
        ct = np.array([t["crps_true"] for t in pick[g]], float)
        cf = np.array([t["crps_flip"] for t in pick[g]], float)
        m = (ct > 0) & (cf > 0)
        logd[g] = np.log(cf[m]) - np.log(ct[m])   # log-space paired difference
        rawd[g] = cf - ct                          # raw paired difference (count-CRPS units)
    blog = min(max(float(np.percentile(np.abs(np.concatenate([logd[g] for g in groups])), 92)), 0.4), 1.0)
    braw = min(max(float(np.percentile(np.abs(np.concatenate([rawd[g] for g in groups])), 92)), 0.05), 60.0)

    mosaic = [["win", f"log_{g}", f"raw_{g}"] for g in groups]
    fig, axd = plt.subplot_mosaic(mosaic, figsize=(12.8, 2.3 + 1.55 * len(groups)),
                                  gridspec_kw={"width_ratios": [0.8, 1.2, 1.2]})
    axL = axd["win"]
    for i, g in enumerate(groups):
        win = 100 * float((rawd[g] > 0).mean()) if rawd[g].size else 0.0
        axL.barh(i, win, color="#2ca02c"); axL.barh(i, 100 - win, left=win, color="#cfcfcf")
        axL.text(win - 1.5, i, f"{win:.0f}%", va="center", ha="right",
                 color="white", fontsize=12, fontweight="bold")
        axL.text(101, i, f"n={rawd[g].size}", va="center", ha="left", fontsize=8, color="#555")
    axL.axvline(50, color="k", lw=1.0, ls="--")
    axL.set_yticks(range(len(groups))); axL.set_yticklabels(groups)
    axL.set_xlim(0, 118); axL.set_ylim(-0.6, len(groups) - 0.4); axL.invert_yaxis()
    axL.set_xlabel("% true-better")
    axL.set_title("win rate  (green = true-better; dashed = 50%)", fontsize=9)

    for i, g in enumerate(groups):
        _delta_hist(axd[f"log_{g}"], logd[g], blog, ylabel=f"{g}\n# measurements")
        _delta_hist(axd[f"raw_{g}"], rawd[g], braw)
        if i < len(groups) - 1:
            axd[f"log_{g}"].set_xticklabels([]); axd[f"raw_{g}"].set_xticklabels([])
    axd[f"log_{groups[0]}"].set_title("log-Δ = log CRPS(flip) − log CRPS(true)   (>0, green → true better)", fontsize=9)
    axd[f"raw_{groups[0]}"].set_title("raw-Δ = CRPS(flip) − CRPS(true)   (count-CRPS units)", fontsize=9)
    axd[f"log_{groups[-1]}"].set_xlabel("log-Δ")
    axd[f"raw_{groups[-1]}"].set_xlabel("raw-Δ")
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIGS / out_name, dpi=120)
    plt.close(fig)
    return out_name


def fig_runtype_scatter(data):
    return _flip_dist_fig(
        data, "run_type", "fig6_runtype_diff.png",
        "Fig 5 · offset-OFF run_type flip  ·  each measurement = one held-out target-assay × chr21 window-batch",
        split=True)


def fig_readlen(data):
    """Read_length direction-frac + responsiveness across arms (Figs 10–11)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    dirf = [data[t]["M2"]["read_length"]["frac_direction"] if data[t] else np.nan for t, _, _ in ARMS]
    resp = [data[t]["M2"]["read_length"]["mean_responsiveness"] if data[t] else np.nan for t, _, _ in ARMS]
    a1.bar(range(len(ARMS)), dirf, color=COLORS); a1.axhline(0.5, color="k", ls="--", lw=0.7)
    for i, v in enumerate(dirf):
        a1.text(i, v + 0.02, _f(v, 2), ha="center", fontsize=9)
    a1.set_xticks(range(len(ARMS))); a1.set_xticklabels([l for _, l, _ in ARMS], fontsize=8)
    a1.set_ylabel("direction-frac (true beats flip)"); a1.set_ylim(0, 1)
    a1.set_title("Fig 10 · read_length direction (dashed = chance)")
    a2.bar(range(len(ARMS)), resp, color=COLORS)
    for i, v in enumerate(resp):
        a2.text(i, v, _f(v, 2), ha="center", fontsize=9, va="bottom")
    a2.set_xticks(range(len(ARMS))); a2.set_xticklabels([l for _, l, _ in ARMS], fontsize=8)
    a2.set_ylabel("responsiveness (|Δμ| under flip)")
    a2.set_title("Fig 11 · read_length responsiveness")
    return _save(fig, "fig11_12_readlen.png")


def fig_readlen_dist(data):
    return _flip_dist_fig(
        data, "read_length", "fig13_readlen_diff.png",
        "Fig 12 · offset-OFF read_length flip  ·  each measurement = one held-out target-assay × chr21 window-batch",
        split=False)


def fig_m1(data):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    x = np.arange(len(ARMS))
    imp = [data[t]["M1"]["imp"].get("spearman") if data[t] else np.nan for t, _, _ in ARMS]
    den = [data[t]["M1"]["den"].get("spearman") if data[t] else np.nan for t, _, _ in ARMS]
    a1.bar(x - 0.2, imp, 0.4, label="imp", color="#2ca02c"); a1.bar(x + 0.2, den, 0.4, label="den", color="#98df8a")
    a1.axhline(0.4857, color="k", ls="--", lw=0.7, label="avg-track baseline 0.4857")
    a1.axhline(0.38, color="gray", ls=":", lw=0.7, label="candi_v2-base 0.38")
    a1.set_xticks(x); a1.set_xticklabels([l for _, l, _ in ARMS], fontsize=8)
    a1.set_ylabel("count Spearman"); a1.set_title("Fig 6 · M1 imp/den Spearman"); a1.legend(fontsize=6.5)
    crps = [data[t]["M1"]["imp"].get("crps") if data[t] else np.nan for t, _, _ in ARMS]
    marg = data["main_s0_full"]["M1"].get("marginal_crps") if data["main_s0_full"] else np.nan
    a2.bar(x, crps, 0.5, color="#2ca02c")
    a2.axhline(marg, color="k", ls="--", lw=0.8, label=f"marginal {_f(marg,2)}")
    a2.set_xticks(x); a2.set_xticklabels([l for _, l, _ in ARMS], fontsize=8)
    a2.set_ylabel("imp NB-CRPS (lower=better)"); a2.set_title("Fig 7 · M1 imputation CRPS"); a2.legend(fontsize=7)
    return _save(fig, "fig7_8_m1.png")


def fig_m3(data):
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ratio = [data[t]["M3"]["ratio"] if data[t] else np.nan for t, _, _ in ARMS]
    ok = [data[t]["M3"]["invariance_ok"] if data[t] else False for t, _, _ in ARMS]
    bars = ax.bar(range(len(ARMS)), ratio, color=["#2ca02c" if o else "#d62728" for o in ok])
    for i, v in enumerate(ratio):
        ax.text(i, v + 0.005, _f(v), ha="center", fontsize=9)
    ax.axhline(0.3, color="k", ls="--", lw=0.8, label="threshold 0.3")
    ax.set_xticks(range(len(ARMS))); ax.set_xticklabels([l for _, l, _ in ARMS], fontsize=8)
    ax.set_ylabel("within/between cos-dist ratio"); ax.legend(fontsize=8)
    ax.set_title("Fig 8 · M3 latent invariance (green = passes ≤0.3)")
    return _save(fig, "fig9_m3.png")


def fig_train_curves(data):
    """Per-epoch median training NLL for all arms — convergence diagnostic."""
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for (t, lab, _), c in zip(ARMS, COLORS):
        if not data[t] or "train_losses" not in data[t]:
            continue
        L = np.array(data[t]["train_losses"], float)
        ep = data[t]["config"]["epochs"]
        spe = max(1, len(L) // ep)
        pe = np.array([np.median(L[i * spe:(i + 1) * spe]) for i in range(ep)])
        ax.plot(np.arange(1, ep + 1), pe, "-o", ms=3, color=c, label=lab)
    ax.set_xlabel("epoch"); ax.set_ylabel("median masked-NB train NLL (chr19)")
    ax.set_title("Fig 13 · training convergence (per-epoch median loss)")
    ax.legend(fontsize=7)
    return _save(fig, "fig14_train_curves.png")


def fig_pit(data):
    res = data.get("main_s0_full")
    if not res:
        return None
    imp = res["M1"]["imp"]
    grid, fbar = imp.get("calib_grid"), imp.get("calib_fbar")
    if not grid:
        return None
    fig, ax = plt.subplots(figsize=(3.8, 3.6))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8); ax.plot(grid, fbar, "-o", ms=3, color="#1f77b4")
    ax.set_xlabel("nominal u"); ax.set_ylabel("F̄(u)")
    ax.set_title(f"Fig 9 · PIT reliability (main_s0, ECE={_f(imp.get('ece'))})")
    return _save(fig, "fig10_pit.png")


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def _scorecard(data):
    def row(name, fn):
        return "| " + name + " | " + " | ".join(fn(data[t]) if data[t] else "—" for t, _, _ in ARMS) + " |"
    hdr = "| metric | " + " | ".join(l for _, l, _ in ARMS) + " |"
    sep = "|" + "---|" * (len(ARMS) + 1)
    rows = [hdr, sep,
            row("**M1** imp Spearman", lambda r: _f(r["M1"]["imp"].get("spearman"))),
            row("M1 imp Pearson", lambda r: _f(r["M1"]["imp"].get("pearson"))),
            row("M1 den Spearman", lambda r: _f(r["M1"]["den"].get("spearman"))),
            row("M1 imp CRPS (marg 2.21)", lambda r: _f(r["M1"]["imp"].get("crps"), 2)),
            row("M1 imp ECE", lambda r: _f(r["M1"]["imp"].get("ece"))),
            row("M1 eff-rank", lambda r: _f(r["M1"].get("encoder_eff_rank"), 1)),
            row("M1 health (den≥imp)", lambda r: "✓" if r["M1"].get("health_gate_den_ge_imp") else "✗"),
            row("**M2 depth** min@true", lambda r: _f(r["M2"]["depth"].get("frac_min_at_true"), 2)),
            row("M2 depth **η-slope**", lambda r: _f(r["M2"]["depth"].get("median_eta_slope"))),
            row("M2 depth dir CI≠0", lambda r: "✓" if r["M2"]["depth"]["direction"].get("excludes_zero") else "✗"),
            row("M2 depth null Δ", lambda r: _f(r["M2"]["depth"]["null"].get("mean"))),
            row("**M2 run_type** dir-frac", lambda r: _f(r["M2"]["run_type"].get("frac_direction"), 2)),
            row("M2 run_type responsiveness", lambda r: _f(r["M2"]["run_type"].get("mean_responsiveness"), 2)),
            row("M2 run_type CI≠0 (paired·single)",
                lambda r: ("✓" if r["M2"]["run_type"]["paired"].get("excludes_zero") else "✗") + "·" +
                          ("✓" if r["M2"]["run_type"]["single"].get("excludes_zero") else "✗")),
            row("M2 run_type honest-null", lambda r: str(r["M2"]["run_type"].get("natural_variance_insufficient"))),
            row("**M2 read_length** responsiveness", lambda r: _f(r["M2"]["read_length"].get("mean_responsiveness"), 2)),
            row("M2 read_length CI≠0 (large-N)", lambda r: "✓" if r["M2"]["read_length"]["overall"].get("excludes_zero") else "✗"),
            row("**M3** ratio (≤0.3)", lambda r: _f(r["M3"].get("ratio"))),
            row("M3 invariance_ok", lambda r: "✓" if r["M3"].get("invariance_ok") else "✗")]
    return "\n".join(rows)


def _inventory():
    rows = ["| T_ biosample | imp | assay | idx | run_type |", "|---|---|---|---|---|"]
    for t, pref, assay, idx, lab in HELD_OUT_TARGETS:
        rows.append(f"| {t} | {pref} | {assay} | {idx} | {lab} |")
    return "\n".join(rows)


def _verdicts():
    return "\n".join([
        "| hypothesis | verdict | evidence |",
        "|---|---|---|",
        "| **h40** M1 health | **supported** | imp-Spearman 0.53–0.59, den 0.71 (den≥imp), beats marginal CRPS (2.21), ECE 0.026–0.062, eff-rank 52 |",
        "| **h41** depth steering | **partial** | passes on the mean, but η-slope ≈0 (arithmetic) on the winning recipe; offset-OFF → 0.88 (learned) |",
        "| **h42** run_type flip | **partial** | ignored on winning recipe (resp 0, honest-null); offset-OFF → dir 0.69, CI≠0 both single & paired |",
        "| **h43** M3 invariance | **supported** | ratio 0.24–0.29 ≤0.3; x_eq_y control breaks it (0.334) → genuine, per-assay-DSF load-bearing |",
    ])


# ---------------------------------------------------------------------------

def generate() -> Path:
    FIGS.mkdir(parents=True, exist_ok=True)
    data = _load()
    figs = dict(
        eta=fig_eta_slope(data), depth=fig_depth_curves(data), rt=fig_runtype(data),
        rt_sc=fig_runtype_scatter(data), m1=fig_m1(data), m3=fig_m3(data), pit=fig_pit(data),
        rl=fig_readlen(data), rl_dist=fig_readlen_dist(data), train=fig_train_curves(data))

    def g(tag, *ks):
        r = data.get(tag)
        for k in ks:
            r = r.get(k) if isinstance(r, dict) else None
        return r

    n_units = next((data[t]["n_units"] for t, _, _ in ARMS if data[t]), "—")
    imp_n = next((data[t]["M1"]["imp"].get("n_points") for t, _, _ in ARMS if data[t]), "—")
    imp0, imp1 = _f(g("main_s0_full", "M1", "imp", "spearman"), 2), _f(g("main_s1_full", "M1", "imp", "spearman"), 2)
    impoff, impcp = _f(g("offoff_s0_full", "M1", "imp", "spearman"), 2), _f(g("copyable_s0_full", "M1", "imp", "spearman"), 2)
    den0 = _f(g("main_s0_full", "M1", "den", "spearman"), 2)
    crps0, crps1 = _f(g("main_s0_full", "M1", "imp", "crps"), 2), _f(g("main_s1_full", "M1", "imp", "crps"), 2)
    crpsoff, marg = _f(g("offoff_s0_full", "M1", "imp", "crps"), 2), _f(g("main_s0_full", "M1", "marginal_crps"), 2)
    ece0, ece1 = _f(g("main_s0_full", "M1", "imp", "ece"), 3), _f(g("main_s1_full", "M1", "imp", "ece"), 3)
    _eta_on_raw = g("main_s0_full", "M2", "depth", "median_eta_slope")
    eta_on = "≈0" if (_eta_on_raw is not None and abs(float(_eta_on_raw)) < 0.01) else _f(_eta_on_raw, 2)
    eta_off = _f(g("offoff_s0_full", "M2", "depth", "median_eta_slope"), 2)
    rt_on = _f(g("main_s0_full", "M2", "run_type", "frac_direction"), 2)
    rt_resp_off = _f(g("offoff_s0_full", "M2", "run_type", "mean_responsiveness"), 2)
    rl_on = _f(g("main_s0_full", "M2", "read_length", "mean_responsiveness"), 3)
    rl_off = _f(g("offoff_s0_full", "M2", "read_length", "mean_responsiveness"), 2)
    rl_dir_on = _f(g("main_s0_full", "M2", "read_length", "frac_direction"), 2)
    rl_dir_off = _f(g("offoff_s0_full", "M2", "read_length", "frac_direction"), 2)
    m3_on = _f(g("main_s0_full", "M3", "ratio"), 3)
    m3_off = _f(g("offoff_s0_full", "M3", "ratio"), 3)
    m3_cp = _f(g("copyable_s0_full", "M3", "ratio"), 3)
    rts = _runtype_stats(data["offoff_s0_full"]) if data.get("offoff_s0_full") else {}
    fp, fs = _f(rts.get("frac_true_paired"), 2), _f(rts.get("frac_true_single"), 2)
    rp, rs = _f(rts.get("resp_paired"), 2), _f(rts.get("resp_single"), 2)
    ntt, nt = rts.get("n_targets_true", "—"), rts.get("n_targets", "—")
    sm = _f(g("offoff_s0_full", "M2", "run_type", "single", "mean"), 3)
    pm = _f(g("offoff_s0_full", "M2", "run_type", "paired", "mean"), 3)

    md = [
        "# q19 · Can we steer CANDI with real experimental metadata?",
        "*Dual conditioning on real CANDI sandbox data — results and how to read them.*",
        "",
        "## TL;DR",
        "",
        "- **What we asked.** CANDI is told, per assay, the experimental covariates of the output it should "
        "produce (sequencing depth, assay, read length, single/paired). Does it actually *use* that prompt "
        "on real data — or just look like it?",
        f"- **It imputes well.** Held-out imputation on all of chr21 is healthy: imp-Spearman "
        f"**{imp0}–{imp1}**, it clears the marginal-CRPS baseline, is calibrated, and denoising ≥ imputation. "
        "(h40 ✓)",
        "- **The catch.** With the depth-offset head (the winning recipe), depth \"steering\" is **free "
        f"arithmetic** (`μ ∝ 2^depth`), not *learned*: the honest lever `η` is flat (slope **{eta_on}**), "
        "and run_type / read_length are **ignored** (the prompt barely moves the prediction). (h41 / h42 "
        "partial)",
        f"- **The reveal.** Turn the offset **off** and the model is forced to read the prompt: `η` learns "
        f"depth (slope **{eta_off}**), and run_type steering becomes real — the true prompt imputes **{ntt} "
        f"of {nt}** held-out targets better, strongly for paired ({fp} of paired instances). The offset was "
        "*starving* the learned pathway.",
        f"- **The biology is robust.** The encoder builds one depth-invariant biological latent whether the "
        f"offset is on or off (M3 ratio {m3_on}/{m3_off} ≤ 0.3; a control breaks it). (h43 ✓)",
        f"- **Bottom line.** Dual conditioning is real and learnable, but there is an **offset on/off "
        f"tradeoff**: offset-on buys better imputation + free depth calibration; offset-off buys genuinely "
        f"learned steering at a real imputation cost (CRPS {crpsoff} vs {crps1}–{crps0}, near the {marg} "
        "floor). The follow-up (h45) is a hybrid that aims for both.",
        "",
        "---",
        "",
        "## 1 · The question, and the trap",
        "",
        "**Question (q19).** Before scaling to production, does the dual-conditioning recipe reproduce "
        "*metadata steering* on **real** CANDI sandbox data — i.e. does telling the model a covariate "
        "actually change, and improve, what it outputs?",
        "",
        "**The trap.** CANDI's count head is a depth-offset negative binomial: `log2(μ) = (depth − c) + η`. "
        "The `(depth − c)` term means the predicted mean scales with the told depth **by construction** — "
        "so if we only check \"did the output move when I changed the depth prompt?\", the answer is "
        "trivially yes, whether or not the model learned anything. A believable steering claim therefore "
        "has to separate two things: **(a) did the output move** (easy, and partly free), from "
        "**(b) did the model *learn to read* the prompt** (the real question). `η` — the part of the "
        "prediction the offset can't touch — is our honest lever for (b).",
        "",
        "## 2 · How the experiment was run",
        "",
        "**Data.** `sandbox.h5`: 8 ENCODE assays + a ChIP control, 5 biosamples stored as **T_/V_/B_** "
        "(Train / Validation / Blind-test) views that each hold a *different* subset of assays. We train "
        "on **T_ chr19** and evaluate on **chr21**. Because an assay present in a biosample's V_/B_ view "
        "but absent from its T_ input is never seen at input time, it gives a clean held-out **imputation "
        "target** whose only information channel is the prompt — there are **12** such targets "
        "(9 paired-end, 3 single-end; see T2). This run used the data *completely*: every epoch iterated "
        f"all of chr19 for all 5 biosamples, and eval covered all of chr21 ({n_units} eval batches, "
        f"{imp_n:,} scored positions, no subsampling).",
        "",
        "**Model.** The golden-reference CANDI architecture — per-assay **FiLM** (feature-wise linear "
        "modulation; the mechanism that injects the metadata prompt) in the encoder and decoder, and a "
        "**counts-only** depth-offset NB head — with the real 4-row metadata "
        "`[log2 depth, assay_id, read_length, run_type]`. Depth feeds *both* the offset and the decoder "
        "FiLM, so `η` is able to carry a learned depth response if the model chooses to learn one.",
        "",
        "**The four arms** (why each exists). *DSF = downsampling factor: in-silico reduction of "
        "sequencing depth, used to create depth counterfactuals.*",
        "",
        "| arm | config | what it isolates |",
        "|---|---|---|",
        "| `main_s0` / `main_s1` | **offset ON**, per-assay independent DSF | the winning recipe (two seeds; s0/s1 differ only in the random seed — weight init, per-step DSF draws, cloze masking, and shuffle order) |",
        "| `offoff_s0` | **offset OFF** | removes the `2^depth` shortcut → tests *learned* steering |",
        "| `copyable_s0` | offset ON, `x_eq_y` DSF | `x`/`y` = **input-DSF vs target-DSF** (depths), *not* input/output assay: here the context and the target get the **same** downsample, so depth is *copyable* — the model never has to learn depth-normalization. A training regime (applies to den + imp alike); its role is the M3 control below |",
        "",
        "**The readout — a counterfactual-prompt flip.** Real biology gives no ground-truth "
        "counterfactual (we can't observe the same position at a different depth), so for each held-out "
        "target we predict twice: once under the **true** prompt and once under a deliberately **wrong** "
        "one (flip run_type, or tell a wrong depth), and score *both* against the real held-out data with "
        "**CRPS** (a proper, full-distribution error — lower is better). Three questions follow: does the "
        "prediction *move* (**responsiveness** = mean absolute change in the predicted count mean μ when a "
        "covariate is flipped), does the *true* prompt score better (**direction**), and — for depth — "
        "does the offset-independent `η` track the told depth (**learned**, not arithmetic)? "
        "*Bootstrap CIs resample the pooled foreground positions (1000 resamples).*",
        "",
        "## 3 · Does it impute at all? — the health check (Figs 6–7, 9)",
        "",
        "**Experiment.** On the chr21 held-out targets, compare predicted vs real counts. "
        "**Metrics:** *count Spearman* (rank agreement of predicted vs true counts across positions, higher "
        "better), *NB-CRPS* (distributional error, lower better), and the sanity check that "
        "**denoising ≥ imputation** (reconstructing a seen assay shouldn't be harder than imputing an unseen "
        "one). Two *different* references anchor the plots, both for imputation: the **CRPS marginal** "
        f"(**{marg}**, Fig 7 dashed) is the score of a single position-independent NB fit to the pooled "
        "held-out counts (mean = their median, dispersion by method-of-moments) — the distribution you'd "
        "predict knowing nothing about position, so beating it means the model resolves *per-position* "
        "structure. The **Spearman references** in Fig 6 are external numbers from prior CANDI work — the "
        "position-wise average-track baseline (**0.4857**) and the candi_v2 production model (**0.38**) — "
        "not recomputed here (a constant marginal has no rank variance, so it has no Spearman of its own; "
        "these are the meaningful *per-position* baselines to clear).",
        "", f"![fig78](report_figs/{figs['m1']})", "",
        f"**Read it.** The winning recipe imputes well: imp-Spearman **{imp0} / {imp1}** (seed0/1) — above "
        "the ~0.38 candi_v2 imputation band and clearing the 0.4857 average-track reference (both imputation "
        f"baselines; compare to the *imp* bars, not den) — with den-Spearman **{den0} ≥ imp** (health "
        f"holds), and imp-CRPS **{crps0}** beating the CRPS marginal **{marg}**.",
    ]
    if figs["pit"]:
        md += ["", f"![fig10](report_figs/{figs['pit']})", "",
               f"**Fig 9 — calibration.** The PIT reliability curve (non-randomized probability-integral "
               f"transform of the held-out predictions: empirical F̄(u) vs nominal u; the diagonal = "
               f"perfectly calibrated) tracks near the diagonal with a mild below-diagonal bow "
               f"(ECE {ece0}/{ece1}) — a small calibration deviation, not a red flag."]
    md += [
        "",
        "**Takeaway:** the model is genuinely competent, so the steering readouts below measure a real "
        "model, not noise.",
        "",
        "### Did training converge? (Fig 13)",
        "", f"![fig14](report_figs/{figs['train']})", "",
        "**All four arms converge.** Per-epoch median NLL (masked-NB, on the chr19 training targets) "
        "plateaus by ~epoch 15–20 — the last-5-epoch slope is ≈ 0 and the last-3-epoch coefficient of "
        "variation is < 1% — so the readouts measure *trained* models, not a mid-descent snapshot. **No "
        "underfitting:** the curves flatten well before the 25-epoch budget (the two main arms improve < 2% "
        "over the final 10 epochs; only the `copyable` control is still inching down, ~4%). The *levels* "
        "match the story: `copyable` sits lowest (~0.96 vs ~1.14) because input-depth = target-depth is an "
        "easier fit, and `offset-OFF` sits highest (~1.18, slowest descent from a higher start) because "
        "removing the `2^depth` shortcut makes the task harder. **Overfitting can't be read off the "
        "training curve alone** — this harness logs no validation-loss trajectory and keeps no per-epoch "
        "checkpoints — but the indirect evidence argues against a severe case: train loss *plateaus* rather "
        "than continuing to fall (a memorizing model would keep driving it down), chr21 (a different, "
        "held-out chromosome) imputation stays healthy (Spearman 0.53–0.64, CRPS < marginal), and the two "
        "seeds land on the same loss *and* the same metrics. A definitive verdict would need per-epoch chr21 "
        "eval + checkpointing — a cheap harness addition if we want it.",
        "",
        "## 4 · Depth steering: learned, or just arithmetic? — the crux (Figs 2, 3, 1)",
        "",
        "**Experiment.** For each target, *sweep the told depth* over the depths achievable by in-silico "
        "downsampling (DSF 1→8, i.e. true depth down to −3 log2 units), holding everything else fixed, and "
        "measure two things: the CRPS-vs-GT curve, and `η` (the offset-independent mean statistic) as a "
        "function of the told depth.",
        "", f"![fig23](report_figs/{figs['depth']})", "",
        "**Fig 2 — the output responds to the prompt.** The y-axis is CRPS at the told depth ÷ CRPS at the "
        "true depth (so **1.0 marks the true depth**); the curve rising away from x=0 means the prediction "
        "is worst when we lie about the depth — for all arms. So the prediction tracks the prompt. "
        "**But** this is precisely what the offset arithmetic gives for free, so on its own it proves "
        "nothing about learning. (The offset-ON arms degrade *more steeply* when mis-told, because "
        "`2^depth` makes μ hyper-sensitive to the told value.)",
        "",
        "**Fig 3 & Fig 1 — is it *learned*?** `η` is the part of the prediction the `2^depth` offset "
        "cannot produce. If `η` rises with the told depth, the model *learned* to read depth; if `η` is "
        "flat, the depth response is pure arithmetic. The verdict is stark:",
        "", f"![fig1](report_figs/{figs['eta']})", "",
        f"With the offset **ON**, `η` is flat (slope **{eta_on}**) — the depth \"steering\" in Fig 2 is "
        f"entirely the hardwired `2^depth` term. With the offset **OFF**, and no shortcut available, the "
        f"model *learns* to carry depth in `η` (slope **{eta_off}**). **This is the finding:** the "
        "depth-offset head — the recipe that imputes best among the deployable arms — takes the free lunch "
        "and starves the learned depth pathway.",
        "",
        "## 5 · Run_type steering: the ENCODE-challenge covariate (Figs 4–5)",
        "",
        "**Why it matters.** Single- vs paired-end processing changes the count profile (dedup → "
        "read-start counts); train/test run_type mismatch was the headline bias in the ENCODE Imputation "
        "Challenge. **Experiment:** flip run_type (0↔1) in a held-out target's prompt and score true vs "
        "flipped against the real data — no counterfactual data needed.",
        "", f"![fig45](report_figs/{figs['rt']})", "",
        f"**Read it (Fig 4 — responsiveness).** Same story as depth. Under the offset (winning recipe) the "
        f"model **ignores** run_type — flipping it barely moves the prediction (responsiveness ≈ 0, Fig 4), "
        f"so the true prompt never strictly beats the flip and the scorecard's direction-frac collapses to "
        f"**{rt_on}**. *Why exactly 0 and not 0.5?* direction-frac counts targets where mean "
        "CRPS(flip) − CRPS(true) is strictly **> 0**; an ignored covariate flips to a **bit-identical** "
        "prediction, so that difference is **exactly 0** — not > 0 — and every target scores false → 0 "
        "(a coin-flip 0.5 would need a *symmetric* perturbation). We **omit the direction-frac bar chart** "
        "here because a 0 there reads like total failure when it just means *ignored*; responsiveness "
        "(Fig 4) is the honest panel — 0 responsiveness **and** 0 direction = ignored, not confused. The "
        "readout logs this as an honest null. **Offset-OFF, the model clearly reads run_type**: "
        "responsiveness jumps to "
        f"**{rt_resp_off}** (paired {rp}, single {rs}), and the true prompt imputes better for **{ntt} of "
        f"{nt}** held-out targets — strongly for paired (pooled Δ **+{pm}**, CI≠0; {fp} of paired "
        f"instances) and mixed for the 3 single-end targets ({fs} of instances; the position-pooled single "
        f"aggregate is marginally reversed at **{sm}**, dominated by one high-count target).",
    ]
    if figs["rt_sc"]:
        md += ["", f"![fig6](report_figs/{figs['rt_sc']})", "",
               "**Fig 5** shows it for the offset-OFF model. **Each measurement is one held-out target-assay "
               "scored on one chr21 window-batch**, summarized by the *paired* difference **Δ = CRPS(flip) − "
               "CRPS(true)**: Δ>0 means flipping the prompt *hurt*, i.e. the true run_type imputed better. A "
               "CRPS-vs-CRPS scatter buried this (values span orders of magnitude and hug the diagonal), so "
               "the figure encodes it three ways. **Left — win rate:** the share of measurements with Δ>0, "
               "read straight off the bar — **73% for paired, 56% for single** (both past the 50% dashed "
               "line). **Middle / right — effect size:** the per-group histogram of Δ in **log space** "
               "(log CRPS(flip) − log CRPS(true), scale-free) and in **raw count-CRPS units**, bars colored "
               "by sign (green = true-better) with the y-axis giving the number of measurements per bar. "
               "Effects are individually small (median log-Δ +0.05 paired, +0.02 single) but consistently "
               "green-leaning. Single carries a grey cluster of flip-better measurements near log-Δ≈−0.15 "
               "alongside one strongly true-better target near +0.95; paired is dominated by a green "
               "near-zero peak with a **concentrated left tail (a single high-count target, counted "
               "off-axis)** where flipping happened to help — the same target that swings the position-pooled "
               "single/paired *aggregates*, which is why we report the win rate rather than one pooled mean. "
               "The upshot: the natural run_type variance *is* sufficient (the model uses it once the offset "
               "shortcut is gone), so a paired→single FASTQ re-processing augmentation is **not** needed — "
               "attenuating the offset is.",
               "",
               "**Read_length — responsive, but not cleanly directional (Figs 10–12).** Read_length tracks "
               f"run_type only in the *first* step: offset-ON it is **ignored** (responsiveness {rl_on}, "
               f"direction-frac {rl_dir_on}); its scorecard \"CI≠0\" holds for every arm only because "
               "~3.7 M positions make a negligible mean effect statistically significant — not real "
               f"steering. But offset-**OFF** it diverges from run_type: the prediction becomes *highly* "
               f"responsive to a read-length flip (**{rl_off}**, larger than run_type's), yet its "
               f"direction-frac is only **{rl_dir_off}** (≈ chance) — the model *reads* read_length but its "
               "response does not consistently improve the imputation, unlike run_type. (The flip here jumps "
               "to the farthest observed read length, a large perturbation, which likely explains the big "
               "but undirected movement.) Fig 12 makes this concrete — a wide, roughly balanced Δ "
               "distribution rather than run_type's rightward lean.",
               "", f"![fig11_12](report_figs/{figs['rl']})", "",
               "**Figs 10–11 — read_length across arms.** These are the read_length counterparts of the "
               "run_type panels (we keep read_length's *direction* panel — unlike run_type's degenerate 0, "
               "read_length's direction-frac carries real signal). Direction-frac (Fig 10) hovers near "
               "chance for every arm; responsiveness (Fig 11) is ≈ 0 for all offset-ON arms and spikes only "
               "offset-OFF — i.e. the offset suppresses read_length reading, and removing it unlocks "
               "movement but not a correct direction.",
               "", f"![fig13](report_figs/{figs['rl_dist']})", "",
               "**Fig 12 — read_length flip distribution (offset-OFF), the analogue of Fig 5.** Pooled over "
               "all 12 targets (read_length is continuous, not single/paired). The win rate is **41%** "
               "(*below* the 50% line) and the log-Δ / raw-Δ histograms are wide with a slight **left** lean "
               "(median Δ < 0) — the flip to the farthest read length moves the prediction a lot yet, if "
               "anything, slightly *lowers* CRPS. So this is responsive-but-not-true-better — the visual "
               "signature of 'reads-it-but-doesn't-map-it-to-better-imputation', the opposite of run_type's "
               "clean rightward lean in Fig 5."]
    md += [
        "",
        "## 6 · Does the encoder build one biological latent? (Fig 8)",
        "",
        "**Experiment.** Encode the *same* chr21 region at several input depths (DSF 1→8), each with its "
        "*true* metadata, and compare the latent vectors. **Metric:** within-region vs between-region "
        "cosine distance. A **small within/between ratio** means the encoder maps the same biology to the "
        "same latent regardless of measurement depth — i.e. it *uses* the depth metadata to normalize the "
        "nuisance away, rather than ignoring metadata (which would be degenerate).",
        "", f"![fig9](report_figs/{figs['m3']})", "",
        f"**Read it.** The winning recipe is invariant (ratio **{m3_on}** ≤ 0.3, green; the latent is "
        "high-rank, not collapsed — see the eff-rank row in the scorecard). **How the `x_eq_y` arm acts as "
        "a control:** during training the main arms downsample the context and the target *independently* "
        "(per-assay DSF), so the encoder repeatedly sees the same biology at input depths that differ from "
        "the target and is *forced* to use the depth metadata to normalize that nuisance away — exactly the "
        "M3 test at eval time. The `x_eq_y` arm instead downsamples context and target by the **same** "
        "factor, so depth is copyable and the encoder never faces that pressure. Result: `x_eq_y` **breaks** "
        f"invariance (ratio **{m3_cp}**, red > 0.3) — the same-region-different-input-depth latents drift "
        "apart. That is the interpretation of *\"independent DSF improves the latent's invariance to "
        "depth\"*: the low ratio is a **learned, metadata-driven normalization** that only develops when "
        "training presents mismatched input/target depths — it is not a degenerate constant (which would "
        "also score low but collapse eff-rank) and not free. So **per-assay-independent depth (DSF) is "
        f"load-bearing** for the shared latent. **Tie-back:** invariance holds *both* offset-on and "
        f"offset-off ({m3_on}/{m3_off}), so the steering tradeoff lives in the count head, not the encoder "
        "— the biological representation is robust to the recipe.",
        "",
        "## 7 · The whole picture — an offset on/off tradeoff",
        "",
        "*(M1 = imputation health · M2 = metadata steering · M3 = latent invariance.)* Read the scorecard "
        "column by column: the **offset-ON** arms win on imputation (M1) and get calibrated depth-mean "
        "control for free, but their *learned* steering is null (η-slope ≈ 0; run_type and read_length "
        "responsiveness ≈ 0). The **offset-OFF** arm inverts it — genuinely learned depth **and** run_type "
        "steering, but at a **real imputation cost**: imp-Spearman "
        f"**{impoff}** (vs {imp0}–{imp1}) and, more tellingly, imp-CRPS **{crpsoff}**, which barely clears "
        f"the marginal floor of {marg} (the offset-ON arms clear it to ~{crps1}–{crps0}). That cost is "
        "exactly why the open follow-up (crux **h45**) matters: can a hybrid — an attenuated offset, an "
        "offset warmup, or an offset-off finetune — keep the imputation quality *and* recover the learned "
        "steering? *(Aside: the `x_eq_y` arm shows the best imp-Spearman "
        f"({impcp}) only because input-depth = target-depth is a trivially easier task — a diagnostic "
        "control, not a deployable recipe, which is also why it collapses the shared latent.)*",
        "",
        "So the q19 answer is **not an overall failure to steer** — steering is real and learnable; the "
        "offset-ON honest-nulls are a property of the head, not the model's ceiling.",
        "",
        "### Scorecard (all 4 arms)",
        "", _scorecard(data), "",
        "*Row notes.* **depth min@true** = fraction of the 12 targets whose CRPS curve bottoms out at the "
        "*true* told-depth (~0.76 everywhere — telling the true depth is usually best). **depth dir CI≠0** = "
        "the true depth beats the most-downsampled (k=8) told-depth on CRPS with a bootstrap 95% CI "
        "excluding 0 (✓ for all arms; on offset-ON this is mostly the `2^depth` arithmetic, hence the "
        "η-slope caveat). **depth null Δ = 0 for a degenerate reason, not a passed test:** the null shuffles "
        "the told depth *across the batch samples*, but a given (biosample, assay) has one sequencing depth "
        "shared by all windows in the batch, so the permutation is an **identity** and the effect is exactly "
        "0 — this particular null is uninformative; the real offset-arithmetic control is the **η-slope** "
        "(Fig 1), not this row.",
        "",
        "### Verdicts (crux h40–h43)",
        "", _verdicts(), "",
        "---",
        "",
        "## Appendix",
        "",
        "**T2 · the 12 held-out imputation targets** (each: an assay present in the V_/B_ view but absent "
        "from the T_ input).",
        "", _inventory(), "",
        "**T3 · arms**", "",
        "| tag | config | role |", "|---|---|---|",
    ]
    for tag, lab, role in ARMS:
        md.append(f"| `{tag}` | {lab} | {role} |")
    md += ["",
           "**Reproduce.** Full-coverage run = SLURM **49277527** "
           "(`jobs/sweep_full.sh`; whole chr19 × all biosamples/epoch, whole chr21 eval). "
           "Regenerate this report: `python -m sandbox.diagnostics.dual_conditioning_real.report_all`. "
           "Per-arm reports are under `results/<tag>_report/`.",
           "",
           "**Glossary.** *M1 / M2 / M3* — imputation health / metadata steering / latent invariance. "
           "*CRPS* — continuous ranked probability score, a proper full-distribution error (lower better). "
           "*Spearman* — rank correlation of predicted vs true counts. *η (eta)* — the offset-independent "
           "mean term of the NB head; the honest \"did-it-learn\" lever. *responsiveness* — mean absolute "
           "change in the predicted count mean μ when a prompt covariate is flipped (count units). "
           "*direction / direction-frac* — does the true prompt beat the flipped one on CRPS, and in what "
           "fraction of cases (exactly 0 when the covariate is ignored — a flip that changes nothing gives "
           "Δ=0, which fails the strict >0). *min@true* — fraction of targets whose depth-CRPS curve is "
           "lowest at the true told-depth. *dir CI≠0* — the true-vs-wrong-prompt CRPS gap has a bootstrap "
           "95% CI excluding 0. *null Δ* — direction effect under a shuffled depth prompt; here it is a "
           "degenerate no-op (constant within-batch depth ⇒ identity shuffle ⇒ exactly 0), so it is "
           "uninformative — the η-slope is the real arithmetic control. *DSF* — downsampling factor "
           "(in-silico depth reduction; `x_eq_y` = input DSF equals target DSF). *FiLM* — feature-wise linear "
           "modulation (how the metadata prompt conditions the network). *ECE / PIT* — calibration error / "
           "probability-integral-transform reliability. *imp / den* — imputation (held-out) vs denoising "
           "(observed) positions. *foreground* — the top-count positions (steering lives in high-count "
           "signal, so metrics are computed there)."]

    out = RESULTS / "report.md"
    out.write_text("\n".join(md))
    return out


if __name__ == "__main__":
    p = generate()
    print(f"[report_all] wrote {p} (+ {len(list(FIGS.glob('*.png')))} figures in {FIGS})")
