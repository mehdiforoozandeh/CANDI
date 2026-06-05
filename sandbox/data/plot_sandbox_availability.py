#!/usr/bin/env python3
"""Plot sandbox H5 biosample × assay coverage and metadata heatmaps."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[2]
import sys
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sandbox import SANDBOX_ASSAYS

H5_PATH = Path(__file__).resolve().parent / "sandbox.h5"
OUT_DIR = Path(__file__).resolve().parent
MISSING = -1.0

META_ROWS = {
    0: ("depth_log2", "log2(seq depth)"),
    2: ("read_length", "read length (bp)"),
    3: ("run_type", "run type (0=SE, 1=PE)"),
}


def _bios_sort_key(name: str) -> tuple:
    prefix = name.split("_", 1)[0] if "_" in name else ""
    base = name[2:] if name[:2] in ("T_", "V_", "B_") else name
    porder = {"T": 0, "V": 1, "B": 2}.get(prefix, 3)
    return (base, porder, name)


def load_panel(h5_path: Path) -> tuple[list[str], np.ndarray]:
    """Return biosample names and meta_dsf1 array [n_bios, 4, n_assays]."""
    with h5py.File(h5_path, "r") as h5:
        order = json.loads(h5["biosamples"].attrs["order"])
        panels = []
        for bios in order:
            gname = bios.replace("/", "_")
            panels.append(np.array(h5["biosamples"][gname]["meta_dsf1"], dtype=np.float32))
    return order, np.stack(panels, axis=0)


def split_prefix(bios: str) -> tuple[str, str]:
    if bios.startswith("T_"):
        return "T", bios[2:]
    if bios.startswith("V_"):
        return "V", bios[2:]
    if bios.startswith("B_"):
        return "B", bios[2:]
    return "?", bios


def build_split_matrix(meta: np.ndarray, bios_order: list[str]) -> np.ndarray:
    """0=missing, 1=train (T_*), 2=test/val (V_* or B_*)."""
    n_bios, _, n_assays = meta.shape
    out = np.zeros((n_bios, n_assays), dtype=np.int8)
    avail = meta[:, 0, :] != MISSING
    for i, bios in enumerate(bios_order):
        prefix, _ = split_prefix(bios)
        for j in range(n_assays):
            if not avail[i, j]:
                continue
            if prefix == "T":
                out[i, j] = 1
            elif prefix in ("V", "B"):
                out[i, j] = 2
            else:
                out[i, j] = 1
    return out


def plot_split_heatmap(split: np.ndarray, bios_order: list[str], out_path: Path) -> None:
    bios_sorted = sorted(bios_order, key=_bios_sort_key)
    idx = [bios_order.index(b) for b in bios_sorted]
    mat = split[np.array(idx)]

    cmap = ListedColormap(["white", "#2ca02c", "#d62728"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    fig_h = max(4.0, 0.35 * len(bios_sorted) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    im = ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xticks(range(len(SANDBOX_ASSAYS)))
    ax.set_xticklabels(SANDBOX_ASSAYS, rotation=45, ha="right")
    ax.set_yticks(range(len(bios_sorted)))
    ax.set_yticklabels(bios_sorted, fontsize=8)
    ax.set_title("Sandbox H5 assay availability (meta_dsf1)\n"
                 "green = train (T_*), red = test/val (V_*/B_*), white = missing")
    legend = [
        Patch(facecolor="white", edgecolor="0.5", label="missing"),
        Patch(facecolor="#2ca02c", label="train (T_*)"),
        Patch(facecolor="#d62728", label="test/val (V_*/B_*)"),
    ]
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metadata_heatmap(
    meta: np.ndarray,
    bios_order: list[str],
    row_idx: int,
    field_key: str,
    field_label: str,
    out_path: Path,
) -> None:
    bios_sorted = sorted(bios_order, key=_bios_sort_key)
    idx = [bios_order.index(b) for b in bios_sorted]
    values = meta[np.array(idx), row_idx, :].copy()
    avail = values != MISSING
    values[~avail] = np.nan

    fig_h = max(4.0, 0.35 * len(bios_sorted) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="white")
    im = ax.imshow(values, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xticks(range(len(SANDBOX_ASSAYS)))
    ax.set_xticklabels(SANDBOX_ASSAYS, rotation=45, ha="right")
    ax.set_yticks(range(len(bios_sorted)))
    ax.set_yticklabels(bios_sorted, fontsize=8)
    ax.set_title(f"Sandbox H5 metadata — {field_label}\n(white = missing assay)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(field_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def depth_table(meta: np.ndarray, bios_order: list[str]) -> pd.DataFrame:
    rows = []
    for i, bios in enumerate(bios_order):
        prefix, base = split_prefix(bios)
        for j, assay in enumerate(SANDBOX_ASSAYS):
            d = float(meta[i, 0, j])
            if d == MISSING:
                continue
            rows.append({
                "biosample": bios,
                "prefix": prefix,
                "base_name": base,
                "assay": assay,
                "log2_depth": d,
                "seq_depth": 2.0 ** d,
                "read_length": float(meta[i, 2, j]),
                "run_type": float(meta[i, 3, j]),
            })
    return pd.DataFrame(rows)


def main() -> None:
    if not H5_PATH.exists():
        raise FileNotFoundError(H5_PATH)
    bios_order, meta = load_panel(H5_PATH)
    split = build_split_matrix(meta, bios_order)

    df = depth_table(meta, bios_order)
    csv_path = OUT_DIR / "sandbox_log2_depths.csv"
    df.to_csv(csv_path, index=False)

    plot_split_heatmap(split, bios_order, OUT_DIR / "sandbox_availability_split.png")
    for row_idx, (key, label) in META_ROWS.items():
        plot_metadata_heatmap(
            meta, bios_order, row_idx, key, label,
            OUT_DIR / f"sandbox_metadata_{key}.png",
        )

    print(f"Wrote {csv_path} ({len(df)} available biosample×assay entries)")
    print(f"Wrote heatmaps to {OUT_DIR}/sandbox_*.png")
    print("\nlog2(depth) for all available assays in H5:\n")
    for _, r in df.sort_values(["base_name", "prefix", "assay"]).iterrows():
        print(
            f"  {r['biosample']:28s} {r['assay']:12s}  "
            f"log2(depth)={r['log2_depth']:8.3f}  depth≈{r['seq_depth']:,.0f}"
        )


if __name__ == "__main__":
    main()
