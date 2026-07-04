#!/usr/bin/env python3
"""Plot paper-grounded JEPA trajectory panels for one or more runs."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MASK_FRAC_THRESHOLD = 0.05

# (subplot_row, metric_key, y_label, threshold_val, threshold_label)
PANELS = [
    (0, "lejepa/combined_loss_scaled", "combined_loss_scaled", 0.0, "lower is better"),
    (1, "lejepa/pred_loss", "pred_loss", 0.025, "shortcut floor"),
    (2, "lejepa/sigreg_loss", "sigreg_loss", 2.0, "high-risk band"),
    (3, "lejepa/encoder_eff_rank", "encoder_eff_rank", 15.0, "collapse floor"),
    (4, "lejepa/cov_condition_number", "cov_condition_number", 50.0, "gate ceiling"),
    (5, "lejepa/meta_sens_runtype", "runtype_sens", 0.40, "min healthy"),
]


def load_steps(run_dir: Path) -> Tuple[List[Dict[str, Any]], List[bool]]:
    """Return (all_rows_with_lejepa, is_spike_flag_per_row)."""
    jsonl_path = run_dir / "metrics.jsonl"
    if not jsonl_path.exists():
        return [], []
    rows, spikes = [], []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("kind") != "training_step":
                continue
            lejepa = d.get("lejepa")
            if not isinstance(lejepa, dict):
                continue
            rows.append(lejepa)
            mask_frac = lejepa.get("lejepa/mask_frac", 1.0)
            spikes.append(mask_frac < MASK_FRAC_THRESHOLD)
    return rows, spikes


def get_series(rows, spikes, key):
    """Return (x_clean, y_clean, x_spike, y_spike) using sequential step indices."""
    xc, yc, xs, ys = [], [], [], []
    for i, (r, is_spike) in enumerate(zip(rows, spikes)):
        v = r.get(key)
        if v is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        x_val = i * 200  # approximate optimizer step
        if is_spike:
            xs.append(x_val)
            ys.append(v)
        else:
            xc.append(x_val)
            yc.append(v)
    return xc, yc, xs, ys


def main():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib not available. Install with: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    args = sys.argv[1:]
    out_path = "/tmp/jepa_trajectories.png"
    if "--out" in args:
        idx = args.index("--out")
        out_path = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    if not args:
        print(__doc__)
        sys.exit(1)

    run_dirs = [Path(p) for p in args]
    missing = [p for p in run_dirs if not p.exists()]
    if missing:
        print(f"ERROR: run directories not found: {missing}", file=sys.stderr)
        sys.exit(1)

    n_panels = len(PANELS)
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3.5 * n_panels), sharex=False)
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    for ri, run_dir in enumerate(run_dirs):
        rows, spikes = load_steps(run_dir)
        if not rows:
            print(f"WARNING: no training_step rows in {run_dir.name}", file=sys.stderr)
            continue
        color = colors[ri % len(colors)]
        label = run_dir.name

        for panel_idx, (_, key, ylabel, threshold, thr_label) in enumerate(PANELS):
            ax = axes[panel_idx]
            xc, yc, xs, ys = get_series(rows, spikes, key)
            if xc:
                ax.plot(xc, yc, color=color, label=label, linewidth=1.5, alpha=0.85)
            if xs:
                ax.scatter(xs, ys, color="lightgrey", marker="x", s=30, zorder=1,
                           label=("FJ2 spikes" if ri == 0 else None))

    # Finalize each panel
    for panel_idx, (_, key, ylabel, threshold, thr_label) in enumerate(PANELS):
        ax = axes[panel_idx]
        if threshold > 0.0:
            ax.axhline(
                y=threshold,
                color="red",
                linestyle="--",
                linewidth=1.0,
                label=f"{thr_label} ({threshold})",
            )
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=7, loc="upper right", ncol=2)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Optimizer step (approx.)")
    fig.suptitle("JEPA Training Trajectories", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
