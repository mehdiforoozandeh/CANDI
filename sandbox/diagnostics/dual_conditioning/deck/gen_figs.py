#!/usr/bin/env python
"""Generate clean, SFU-branded, plain-language figures + math SVGs for the dual-conditioning deck.
Reads results/deck_data.json (exact measured numbers). Outputs vector SVGs into deck/assets/."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(open(os.path.join(HERE, "..", "results", "deck_data.json")))
OUT = os.path.join(HERE, "assets")
os.makedirs(OUT, exist_ok=True)

# --- SFU brand ------------------------------------------------------------
RED      = "#CC0633"   # primary
DARKRED  = "#A6192E"   # secondary
GREY     = "#54585A"   # dark grey
MIDGREY  = "#8A8D8F"
LIGHT    = "#D9D9D9"
PANEL    = "#F2F2F2"
INK      = "#2A2D2E"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "cm",
    "svg.fonttype": "path",     # embed glyphs as paths -> fully self-contained
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": GREY, "ytick.color": GREY, "axes.edgecolor": GREY,
    "axes.linewidth": 1.1,
})

def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)

def save(fig, name):
    fig.savefig(os.path.join(OUT, name), transparent=True, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print("wrote", name)

# ---- diagram helpers ------------------------------------------------------
def _tint(hexc, f=0.86):
    r, g, b = mcolors.to_rgb(hexc)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)

def _box(ax, cx, cy, w, h, text, *, fill="#fff", edge=GREY, tc=INK, lw=2.0,
         bold=False, dashed=False, fs=14):
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.10", linewidth=lw,
                 edgecolor=edge, facecolor=fill, linestyle=("--" if dashed else "-"), zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc,
            weight=("bold" if bold else "normal"), zorder=4, linespacing=1.2)

def _arrow(ax, x1, y1, x2, y2, *, color=GREY, lw=2.3, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15,
                 lw=lw, color=color, connectionstyle=f"arc3,rad={rad}", zorder=2,
                 shrinkA=1, shrinkB=3))

# =========================================================================
# 1. JOB-1 CLUSTERING  (illustrative layout, faithful to measured ratio)
# =========================================================================
def fig_job1():
    w = DATA["pooled"]["M3"]["within"]     # 0.075  (same-signal spread)
    b = DATA["pooled"]["M3"]["between"]    # 0.785  (different-signal spread)
    r = DATA["pooled"]["M3"]["ratio"]      # 0.096
    rng = np.random.default_rng(7)
    # 4 different "true signals": centers spread ~ between; corruptions jitter ~ within
    centers = np.array([[-1.0, 0.7], [1.15, 0.95], [-0.75, -0.95], [1.05, -0.7]]) * (b)
    cols = [RED, GREY, DARKRED, MIDGREY]
    fig, ax = plt.subplots(figsize=(6.1, 4.5))
    for c, col in zip(centers, cols):
        pts = c + rng.normal(0, w * 1.6, size=(6, 2))
        ax.scatter(pts[:, 0], pts[:, 1], s=95, color=col, edgecolor="white",
                   linewidth=1.3, zorder=3, alpha=0.95)
    # annotate one tight cluster (within) and the gap between two clusters (between)
    ax.annotate("", xy=centers[0] + [0.18, 0.14], xytext=centers[0] - [0.18, 0.14],
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.6))
    ax.text(centers[0][0], centers[0][1] + 0.42, "same signal,\ndifferent corruptions\n→ clustered",
            ha="center", va="bottom", fontsize=11.5, color=INK, linespacing=1.25)
    ax.annotate("", xy=centers[1], xytext=centers[2],
                arrowprops=dict(arrowstyle="<->", color=MIDGREY, lw=1.6, ls=(0, (4, 3))))
    ax.text(0.16, -0.02, "different signals\n→ far apart", ha="center", va="center",
            fontsize=11.5, color=GREY, linespacing=1.25)
    ax.text(0.99, 0.01, f"ratio  ρ ≈ {r:.2f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=14, color=RED, weight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect("equal"); ax.margins(0.22)
    save(fig, "job1_cluster.svg")

# =========================================================================
# 2. STEERING CURVE  (error vs how wrong the instruction was; 0 = the truth)
#    dip at 0 = obeys the instruction;  flat = ignores it.
#    aggregated over all transforms -> matches the headline steering score.
# =========================================================================
def _steer_curve(arm, title, dips):
    agg = {}
    for fam in ("mult", "add", "power"):
        C = np.array(DATA[arm]["crps_matrix"][fam]["C"])   # C[i true][j told]
        Cn = C / C.mean(axis=1, keepdims=True)             # relative to row mean
        P = C.shape[0]
        for i in range(P):
            for j in range(P):
                if abs(j - i) <= 2:
                    agg.setdefault(j - i, []).append(Cn[i, j])
    offs = sorted(agg)
    y = [float(np.mean(agg[o])) for o in offs]
    col = RED if dips else GREY
    fig, ax = plt.subplots(figsize=(6.1, 4.6))
    ax.axhline(1.0, color=LIGHT, lw=1.2, zorder=1)
    ax.axvline(0, color=INK, lw=1.0, ls=(0, (2, 3)), zorder=2)
    ax.plot(offs, y, color=col, lw=3.2, zorder=3, solid_capstyle="round", solid_joinstyle="round")
    k = offs.index(0)
    ax.scatter([0], [y[k]], s=210, color=col, edgecolor="white", lw=2.2, zorder=6)
    _clean(ax)
    ax.set_xticks(offs)
    ax.set_xticklabels(["−2", "−1", "the truth", "+1", "+2"], fontsize=12)
    ax.set_xlabel("how wrong our instruction was", fontsize=13, labelpad=8)
    ax.set_ylabel("prediction error\n(relative to its own average)", fontsize=13)
    ax.set_ylim(0.0, 1.75)
    ax.set_title(title, fontsize=15.5, weight="bold", color=INK, pad=10)
    cap = ("lowest error exactly at the truth → it obeys"
           if dips else "flat — the instruction barely changes the error")
    ax.text(0.5, 0.97, cap, transform=ax.transAxes, ha="center", va="top",
            fontsize=12, color=col, style="italic", weight="bold")
    save(fig, f"steer_{'fixed' if dips else 'broken'}.svg")

def fig_responses():
    _steer_curve("pooled", "Broken model", dips=False)
    _steer_curve("perassay_log_aware", "Fixed model", dips=True)

# =========================================================================
# 3. BAR CHARTS  (steering score across conditions)
# =========================================================================
def _bars(labels, vals, colors, name, figsize=(6.2, 4.4), gate=True, ymax=0.72,
          notes=None):
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(vals))
    bars = ax.bar(x, vals, width=0.62, color=colors, zorder=3, edgecolor="white", linewidth=1.5)
    if gate:
        ax.axhline(0.5, color=LIGHT, lw=1.4, ls=(0, (5, 4)), zorder=1)
        ax.text((len(vals) - 1) / 2.0, 0.508, "good steering ≈ 0.5", ha="center",
                va="bottom", fontsize=9.5, color=MIDGREY)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.018, f"{v:.2f}", ha="center", va="bottom", fontsize=15,
                color=INK, weight="bold")
    _clean(ax)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12.5)
    ax.set_yticks([0, 0.25, 0.5])
    ax.set_ylim(0, ymax)
    ax.set_ylabel("steering score  S", fontsize=13)
    if notes:
        for xi, txt in notes:
            ax.text(xi, -0.13, txt, ha="center", va="top", fontsize=10.5,
                    color=GREY, transform=ax.get_xaxis_transform())
    save(fig, name)

def fig_bars():
    P = DATA["pooled"]["M2_median"]
    A = DATA["perassay_log_aware"]["M2_median"]
    F = DATA["forced_input"]["M2_median"]
    # pooling reveal
    _bars(["blurred\n(v1)", "kept separate\n(fixed)"], [P, A], [GREY, RED], "bar_pooling.svg")
    # forced-input control
    _bars(["fixed\nmodel", "clean input\n(no shortcut)"], [A, F], [RED, DARKRED],
          "bar_forced.svg")

# =========================================================================
# 4. BACKGROUND (foreground vs whole-genome)  slide 8
# =========================================================================
def fig_bg():
    fams = ["mult", "add", "power"]
    agg = [DATA["perassay_log_aware"]["fg"][f]["agg"] for f in fams]
    fg  = [DATA["perassay_log_aware"]["fg"][f]["fg"] for f in fams]
    x = np.arange(len(fams)); w = 0.36
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.bar(x - w/2, agg, w, color=MIDGREY, label="whole genome", zorder=3, edgecolor="white")
    ax.bar(x + w/2, fg,  w, color=RED,     label="peaks only",  zorder=3, edgecolor="white")
    _clean(ax)
    ax.set_xticks(x); ax.set_xticklabels(["transform A", "transform B", "transform C"], fontsize=12)
    ax.set_ylabel("steering score  S", fontsize=12.5)
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=11, loc="upper right")
    save(fig, "bg.svg")

# =========================================================================
# 5. MATH  (compiled notation -> vector SVG, transparent)
# =========================================================================
def _math(tex, name, w=7.2, h=1.15, size=23):
    fig = plt.figure(figsize=(w, h))
    fig.text(0.5, 0.5, tex, ha="center", va="center", fontsize=size, color=INK)
    save(fig, name)

def fig_math():
    _math(r"$\rho=\dfrac{\langle\, d_{\cos}(z_{\mathrm{aug}},\,z_{\mathrm{id}})\,\rangle_{\mathrm{same\ signal}}}"
          r"{\langle\, d_{\cos}(z_a,\,z_b)\,\rangle_{\mathrm{different}}}$", "math_rho.svg", h=1.5)
    _math(r"$d_{\cos}(u,v)=1-\dfrac{u\cdot v}{\|u\|\,\|v\|}$", "math_dcos.svg", w=5.2)
    _math(r"$S=\dfrac{1}{P}\sum_{i=1}^{P}\dfrac{\bar C_i-C_{ii}}{\bar C_i}"
          r"\qquad \bar C_i=\dfrac{1}{P}\sum_{j=1}^{P}C_{ij}$", "math_S.svg", w=8.0, h=1.5)
    _math(r"$C_{ij}=\mathrm{CRPS}\left(\hat F_{\mathrm{told}\ j},\ y_i\right)$", "math_Cij.svg", w=5.6)
    _math(r"$\mathrm{CRPS}(F,y)=\int_{-\infty}^{\infty}\left(F(x)-\mathbb{1}[x\geq y]\right)^2\,dx$",
          "math_crps.svg", w=8.0)

# =========================================================================
# 6. SETUP SKELETON (slide 3) — Input / f_x / CANDI / Output / f_y / Target
# =========================================================================
def fig_skeleton():
    TF = "#ECECED"
    fig, ax = plt.subplots(figsize=(11, 4.25))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5); ax.axis("off")
    _box(ax, 1.15, 2.5, 1.7, 1.15, "Input", edge=GREY, bold=True, fs=17)
    _box(ax, 3.55, 3.75, 1.95, 1.05, "apply  $f_x$\n(transform\non the input)", fill=TF, edge=MIDGREY, fs=12.5)
    _box(ax, 6.05, 3.75, 1.75, 1.05, "CANDI", edge=RED, tc=DARKRED, bold=True, fs=17)
    _box(ax, 8.55, 3.75, 1.75, 1.05, "Output", edge=RED, tc=DARKRED, bold=True, fs=16)
    _box(ax, 3.55, 1.25, 1.95, 1.05, "apply  $f_y$\n(transform\non the target)", fill=TF, edge=MIDGREY, fs=12.5)
    _box(ax, 8.55, 1.25, 1.75, 1.05, "Target\n(the answer)", edge=RED, tc=DARKRED, dashed=True, bold=True, fs=13.5)
    # fork from Input to both paths
    _arrow(ax, 2.0, 2.75, 2.5, 3.55, color=GREY, rad=-0.18)
    _arrow(ax, 2.0, 2.25, 2.5, 1.45, color=GREY, rad=0.18)
    # top (model) path
    _arrow(ax, 4.6, 3.75, 5.15, 3.75, color=GREY)
    _arrow(ax, 6.95, 3.75, 7.65, 3.75, color=GREY)
    # bottom (answer) path
    _arrow(ax, 4.6, 1.25, 7.65, 1.25, color=GREY)
    # must-match link
    ax.plot([8.55, 8.55], [3.20, 1.80], ls=(0, (3, 3)), color=RED, lw=1.9, zorder=1)
    ax.text(9.55, 2.5, "must\nmatch", ha="left", va="center", fontsize=12.5, color=RED, weight="bold")
    ax.text(0.35, 4.62, "what the model does", fontsize=11.5, color=GREY, style="italic")
    ax.text(0.35, 0.34, "how we build the correct answer", fontsize=11.5, color=GREY, style="italic")
    save(fig, "skeleton.svg")

# =========================================================================
# 7. POOLING vs SEPARATE (new slide) — FiLM metadata routing
# =========================================================================
def fig_pooling():
    TEAL, GOLD = "#2E6E8E", "#B7860B"
    cols = [RED, TEAL, GOLD]
    ys = [3.35, 2.0, 0.65]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    for ax, pooled, title in [(axes[0], True, "Pooling  ·  v1 (broken)"),
                              (axes[1], False, "Kept separate  ·  the fix")]:
        ax.set_xlim(0, 6); ax.set_ylim(0, 4); ax.axis("off")
        ax.set_title(title, fontsize=15.5, weight="bold", color=(GREY if pooled else RED), pad=6)
        for k, (y, c) in enumerate(zip(ys, cols)):
            _box(ax, 0.98, y, 1.45, 0.78, f"assay {k+1}\ninstruction", fill=_tint(c), edge=c, fs=11)
        if pooled:
            _box(ax, 3.0, 2.0, 1.2, 0.95, "average\n(blur)", fill="#E4E5E6", edge=GREY, tc=GREY, bold=True, fs=12)
            for y, c in zip(ys, cols):
                r = 0.0 if abs(y - 2.0) < 0.1 else (0.22 if y < 2 else -0.22)
                _arrow(ax, 1.72, y, 2.42, 2.0, color=c, rad=r)
                r2 = 0.0 if abs(y - 2.0) < 0.1 else (-0.22 if y < 2 else 0.22)
                _arrow(ax, 3.58, 2.0, 4.28, y, color=GREY, rad=r2)
            for k, y in enumerate(ys):
                _box(ax, 5.0, y, 1.45, 0.78, f"assay {k+1}\noutput", fill="#E4E5E6", edge=GREY, fs=11)
        else:
            for y, c in zip(ys, cols):
                _arrow(ax, 1.72, y, 4.28, y, color=c)
            for k, (y, c) in enumerate(zip(ys, cols)):
                _box(ax, 5.0, y, 1.45, 0.78, f"assay {k+1}\noutput", fill=_tint(c), edge=c, fs=11)
    save(fig, "pooling.svg")


# =========================================================================
# 6. PHASE 2C — undoing vs redoing (h32) + composition (h31)
# =========================================================================
def _load(name):
    return json.load(open(os.path.join(HERE, "..", "results", name)))

_2C = "norm-none_enc-naive_off-on_mode-per_assay_2c.json"


def fig_h32():
    """Undoing a distortion on the INPUT costs real error; redoing one on the OUTPUT is free."""
    r = _load(_2C)["chr21"]; NON = {4, 5, 6}
    gap = r["M1"]["gap"]

    def od(pred):
        vs = [v for k, v in gap.items()
              for fx, fy in [tuple(map(int, k.split("_")))] if fx != fy and pred(fx, fy)]
        return float(np.mean(vs)) if vs else 0.0
    lo, li = max(0.0, od(lambda fx, fy: fy in NON)), od(lambda fx, fy: fx in NON)
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    x = [0, 1]; vals = [lo, li]
    ax.bar(x, vals, width=0.6, color=[MIDGREY, RED], zorder=3, edgecolor="white", linewidth=1.5)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.008, ("≈ free" if v < 0.02 else f"+{v:.2f}"), ha="center", va="bottom",
                fontsize=16, color=INK, weight="bold")
    _clean(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(["redo the OUTPUT\n(apply a distortion)", "undo the INPUT\n(remove a distortion)"],
                       fontsize=12.5)
    ax.set_yticks([]); ax.set_ylim(0, max(0.28, li * 1.3))
    ax.set_ylabel("extra reconstruction error", fontsize=12.5)
    save(fig, "h32_asymmetry.svg")


def fig_h31():
    """Extra error on input->output combinations the model never trained on — stays ~0."""
    ref = _load(_2C)
    full = ref["chr21"]["M1"]["cell_crps"]
    labs, vals, fbs = [], [], []
    for pct, tag in (("15", "rho0.15"), ("30", "rho0.3"), ("45", "rho0.45")):
        hr = _load(f"norm-none_enc-naive_off-on_mode-per_assay_{tag}_2c.json")
        held = {f"{a}_{b}" for a, b in hr["config"]["heldout"]}
        hold = hr["chr21"]["M1"]["cell_crps"]
        g = [hold[k] - full[k] for k in held if k in full and k in hold]
        labs.append(f"hid {pct}%"); vals.append(max(0.0, float(np.median(g)) if g else 0.0))
        fbs.append(hr["chr21"]["memorization"]["frac_beats"])
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    x = np.arange(3)
    ax.bar(x, vals, width=0.58, color=RED, zorder=3, edgecolor="white", linewidth=1.5)
    ax.axhline(0, color=GREY, lw=1.3, zorder=2)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.006, f"{v:.2f}", ha="center", va="bottom", fontsize=14.5, color=INK, weight="bold")
    _clean(ax)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=12.5)
    ax.set_yticks([]); ax.set_ylim(0, 0.30)
    ax.set_ylabel("extra error on\nunseen combinations", fontsize=12.5)
    ax.text(1, 0.255, "≈ 0  —  combinations it never saw still work", ha="center",
            fontsize=11.5, color=GREY)
    save(fig, "h31_composition.svg")


if __name__ == "__main__":
    fig_skeleton()
    fig_pooling()
    fig_job1()
    fig_responses()
    fig_bars()
    fig_bg()
    fig_math()
    fig_h32()
    fig_h31()
    print("\nAll figures written to", OUT)
