import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

sources = ["obs", "den", "denimp", "latent"]
colors  = ["#AEC6CF", "#77DD77", "#FFB347", "#C3B1E1"]

# Robustness ratios read from the original 6-panel figure (row-major).
data = {
    "Gm23248":              [0.660, 0.645, 0.720, 0.755],
    "Gm23338":              [0.430, 0.545, 0.550, 0.745],
    "Cardiac Muscle Cell":  [0.425, 0.895, 0.570, 0.490],
    "Foreskin Keratinocyte":[0.440, 0.355, 0.800, 0.840],
    "Hepatocyte":           [0.550, 0.585, 0.690, 0.640],
    "Neural Progenitor Cell":[0.600, 0.420, 0.350, 0.870],
}

fig, axes = plt.subplots(3, 2, figsize=(7.4, 7.0), sharey=True)
x = range(len(sources))

for ax, (title, vals) in zip(axes.ravel(), data.items()):
    ax.bar(x, vals, width=0.8, color=colors, edgecolor="none", zorder=3)
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(sources, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 0.9)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, color="#b0b0b0", alpha=0.7, zorder=0)
    for s in ax.spines.values():
        s.set_linewidth(0.8)
        s.set_color("#333333")
    ax.margins(x=0.06)

fig.supylabel("Fraction of Confident Assignments (Robustness Ratio)", fontsize=10)
fig.subplots_adjust(left=0.11, right=0.99, top=0.955, bottom=0.075, hspace=0.45, wspace=0.10)

fig.savefig("saga_repro.pdf")
fig.savefig("saga_repro.svg")
print("wrote saga_repro.pdf and saga_repro.svg")
