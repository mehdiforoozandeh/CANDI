#!/usr/bin/env python3
"""
Visualize metadata distribution per assay from a metadata CSV.
Creates a multipanel figure: columns = assays, rows = log2_depth, read_length, run_type, control.
Output saved under data/ with a name derived from the input filename.
"""
import argparse
import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def mine_control_metadata(data_dir):
    """
    Mine ChIP-seq control metadata from the raw data directory.
    Returns DataFrame with columns: biosample_name, depth, read_length, run_type
    """
    control_records = []
    
    # Scan all biosample directories
    biosample_dirs = [d for d in glob.glob(os.path.join(data_dir, "*")) if os.path.isdir(d)]
    
    for bios_dir in biosample_dirs:
        biosample_name = os.path.basename(bios_dir)
        control_dir = os.path.join(bios_dir, "chipseq-control")
        
        if not os.path.exists(control_dir):
            continue
            
        file_meta_path = os.path.join(control_dir, "file_metadata.json")
        signal_meta_path = os.path.join(control_dir, "signal_DSF1_res25", "metadata.json")
        
        if not os.path.exists(file_meta_path) or not os.path.exists(signal_meta_path):
            continue
            
        try:
            with open(file_meta_path, 'r') as f:
                file_meta = json.load(f)
            with open(signal_meta_path, 'r') as f:
                signal_meta = json.load(f)
                
            # Extract fields (handle nested dict with "2" key or direct)
            read_length = file_meta.get("read_length", {})
            if isinstance(read_length, dict):
                read_length = read_length.get("2", None)
                
            run_type = file_meta.get("run_type", {})
            if isinstance(run_type, dict):
                run_type = run_type.get("2", None)
                
            depth = signal_meta.get("depth", None)
            
            if depth is not None and read_length is not None and run_type is not None:
                control_records.append({
                    'biosample_name': biosample_name,
                    'depth': float(depth),
                    'read_length': float(read_length),
                    'run_type': str(run_type)
                })
        except Exception as e:
            print(f"Warning: Failed to load control for {biosample_name}: {e}")
            continue
    
    if not control_records:
        return None
        
    return pd.DataFrame(control_records)


def main():
    parser = argparse.ArgumentParser(description="Visualize metadata distribution per assay + control")
    parser.add_argument("csv", type=str, help="Path to metadata CSV (e.g. data/eic_metadata.csv)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to raw data directory (e.g. DATA_CANDI_EIC) for mining control metadata")
    parser.add_argument("-o", "--outdir", type=str, default="data",
                        help="Output directory for figure (default: data)")
    args = parser.parse_args()

    # Filter to only include these 35 assays
    INCLUDED_ASSAYS = [
        'ATAC-seq', 'DNase-seq', 'H2AFZ', 'H2AK5ac', 'H2AK9ac', 'H2BK120ac', 'H2BK12ac', 'H2BK15ac',
        'H2BK20ac', 'H2BK5ac', 'H3F3A', 'H3K14ac', 'H3K18ac', 'H3K23ac', 'H3K23me2', 'H3K27ac', 'H3K27me3',
        'H3K36me3', 'H3K4ac', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K56ac', 'H3K79me1', 'H3K79me2', 'H3K9ac',
        'H3K9me1', 'H3K9me2', 'H3K9me3', 'H3T11ph', 'H4K12ac', 'H4K20me1', 'H4K5ac', 'H4K8ac', 'H4K91ac'
    ]

    df = pd.read_csv(args.csv)
    required = ["assay_name", "depth", "read_length", "run_type"]
    for c in required:
        if c not in df.columns:
            raise SystemExit(f"Missing column: {c}")

    # Filter to included assays only
    df = df[df["assay_name"].isin(INCLUDED_ASSAYS)].copy()

    df["log2_depth"] = np.log2(df["depth"].astype(float).replace(0, np.nan))
    df = df.dropna(subset=["log2_depth", "read_length", "run_type"])
    df["read_length"] = pd.to_numeric(df["read_length"], errors="coerce")
    df = df.dropna(subset=["read_length"])

    assays = sorted(df["assay_name"].unique())
    n_assays = len(assays)
    if n_assays == 0:
        raise SystemExit("No assays found after filtering.")
    
    # Mine control metadata if data_dir provided
    control_df = None
    if args.data_dir and os.path.exists(args.data_dir):
        print(f"Mining control metadata from {args.data_dir}...")
        control_df = mine_control_metadata(args.data_dir)
        if control_df is not None:
            control_df["log2_depth"] = np.log2(control_df["depth"])
            print(f"Found {len(control_df)} control samples")

    # Setup figure: 3 rows × (n_assays + 1 for control) columns
    n_rows = 3
    n_cols = n_assays + (1 if control_df is not None else 0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(max(3, n_cols * 1.8), 8), squeeze=False, sharex='row')

    # Plot assay metadata (rows 0-2)
    for col, assay in enumerate(assays):
        sub = df[df["assay_name"] == assay]

        ax_depth = axes[0, col]
        ax_depth.hist(sub["log2_depth"].dropna(), bins=min(30, max(2, len(sub) // 3)),
                     color="steelblue", edgecolor="white", alpha=0.8)
        ax_depth.set_title(assay, fontsize=9)
        if col == 0:
            ax_depth.set_ylabel("log2_depth\n(count)", fontsize=8)
        ax_depth.set_xlabel("log2(depth)", fontsize=7)
        ax_depth.tick_params(axis="x", labelsize=7)
        ax_depth.tick_params(axis="y", labelsize=7)

        ax_rl = axes[1, col]
        ax_rl.hist(sub["read_length"].dropna(), bins=min(25, max(2, sub["read_length"].nunique())),
                   color="seagreen", edgecolor="white", alpha=0.8)
        if col == 0:
            ax_rl.set_ylabel("read_length\n(count)", fontsize=8)
        ax_rl.set_xlabel("read_length", fontsize=7)
        ax_rl.tick_params(axis="x", labelsize=7)
        ax_rl.tick_params(axis="y", labelsize=7)

        ax_rt = axes[2, col]
        rt = sub["run_type"].astype(str).str.lower()
        single = (rt.str.contains("single", na=False)).sum()
        paired = (rt.str.contains("pair", na=False)).sum()
        labels = ["single", "paired"]
        vals = [single, paired]
        ax_rt.bar(labels, vals, color=["#1f77b4", "#ff7f0e"], edgecolor="white")
        ax_rt.tick_params(axis="x", labelsize=7)
        ax_rt.tick_params(axis="y", labelsize=7)
        if col == 0:
            ax_rt.set_ylabel("run_type\n(count)", fontsize=8)
        ax_rt.set_xlabel("")

    # Plot control metadata as an additional column (last column)
    if control_df is not None:
        col = n_assays  # Last column
        
        # Row 0: Control depth
        ax_ctrl_depth = axes[0, col]
        ax_ctrl_depth.hist(control_df["log2_depth"].dropna(), bins=min(30, max(5, len(control_df) // 3)),
                          color="coral", edgecolor="white", alpha=0.8)
        ax_ctrl_depth.set_title("Control", fontsize=9)
        ax_ctrl_depth.set_xlabel("log2(depth)", fontsize=7)
        ax_ctrl_depth.tick_params(axis="x", labelsize=7)
        ax_ctrl_depth.tick_params(axis="y", labelsize=7)
        
        # Row 1: Control read length
        ax_ctrl_rl = axes[1, col]
        ax_ctrl_rl.hist(control_df["read_length"].dropna(), 
                       bins=min(25, max(5, control_df["read_length"].nunique())),
                       color="coral", edgecolor="white", alpha=0.8)
        ax_ctrl_rl.set_xlabel("read_length", fontsize=7)
        ax_ctrl_rl.tick_params(axis="x", labelsize=7)
        ax_ctrl_rl.tick_params(axis="y", labelsize=7)
        
        # Row 2: Control run type
        ax_ctrl_rt = axes[2, col]
        rt_ctrl = control_df["run_type"].astype(str).str.lower()
        single = (rt_ctrl.str.contains("single", na=False)).sum()
        paired = (rt_ctrl.str.contains("pair", na=False)).sum()
        labels = ["single", "paired"]
        vals = [single, paired]
        ax_ctrl_rt.bar(labels, vals, color=["coral", "#ff6b6b"], edgecolor="white")
        ax_ctrl_rt.tick_params(axis="x", labelsize=7)
        ax_ctrl_rt.tick_params(axis="y", labelsize=7)
        ax_ctrl_rt.set_xlabel("")

    plt.suptitle(f"Metadata distribution by assay — {os.path.basename(args.csv)}", fontsize=11, y=1.00)
    plt.tight_layout()

    os.makedirs(args.outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.csv))[0]
    outpath = os.path.join(args.outdir, f"metadata_dist_{base}.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpath}")


if __name__ == "__main__":
    main()
