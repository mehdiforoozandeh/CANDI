#!/usr/bin/env python3
"""
Visualization of Confidence Calibration

This script generates calibration plots and genomic loci signal visualizations
for model predictions, handling both merged and EIC datasets.

Author: CANDI Team
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Import from project
sys.path.insert(0, str(Path(__file__).parent.parent))
from _utils import Gaussian


def load_predictions_from_npz(preds_dir: Path, biosample: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load predictions from NPZ files for a given biosample.
    Only loads mu, var, and observed_P (not z, peak_scores, observed_peak) for efficiency.
    
    Args:
        preds_dir: Path to preds directory (model_dir/preds/)
        biosample: Name of biosample
        
    Returns:
        Dictionary: {assay_dir_name: {mu, var, observed_P}}
    """
    bios_dir = preds_dir / biosample
    if not bios_dir.exists():
        raise FileNotFoundError(f"Biosample directory not found: {bios_dir}")
    
    results = {}
    
    # Get list of assay directories
    assay_dirs = [d for d in bios_dir.iterdir() if d.is_dir()]
    
    for assay_dir in assay_dirs:
        assay_dir_name = assay_dir.name
        
        try:
            # Load only the NPZ files we need for visualization
            mu_data = np.load(assay_dir / "mu.npz")
            var_data = np.load(assay_dir / "var.npz")
            obs_p_data = np.load(assay_dir / "observed_P.npz")
            
            # Extract arrays (NPZ files contain arrays with default key 'arr_0')
            results[assay_dir_name] = {
                "mu": mu_data['arr_0'],
                "var": var_data['arr_0'],
                "observed_P": obs_p_data['arr_0']
            }
            
        except Exception as e:
            print(f"Warning: Failed to load predictions for {biosample}/{assay_dir_name}: {e}")
            continue
    
    return results


def get_eic_assay_info(biosample: str, eic_metadata_path: str) -> Tuple[int, Set[str]]:
    """
    Get information about which assays are denoised vs imputed for EIC dataset.
    
    Args:
        biosample: Name of biosample (e.g., B_GM12878, V_K562)
        eic_metadata_path: Path to eic_metadata.csv
        
    Returns:
        Tuple of (available_assays_count, denoised_assays_set)
    """
    # Determine T_biosample name
    if biosample.startswith("B_"):
        T_biosample = biosample.replace("B_", "T_")
    elif biosample.startswith("V_"):
        T_biosample = biosample.replace("V_", "T_")
    elif biosample.startswith("T_"):
        T_biosample = biosample
    else:
        raise ValueError(f"Unexpected biosample name format: {biosample}")
    
    # Read EIC metadata
    metadata_df = pd.read_csv(eic_metadata_path)
    
    # Get assays available for T_biosample
    T_rows = metadata_df[metadata_df['biosample_name'] == T_biosample]
    
    if T_rows.empty:
        return 0, set()
    
    denoised_assays = set(T_rows['assay_name'].unique())
    available_assays_count = len(denoised_assays)
    
    return available_assays_count, denoised_assays


def organize_predictions_by_type(
    model_dir: Path,
    dataset: str,
    eic_metadata_path: Optional[str] = None
) -> Dict[str, Dict[str, Dict[str, Dict[str, np.ndarray]]]]:
    """
    Organize predictions into nested structure: data[comparison_type][assay][biosample] = {mu, var, obs}
    
    Args:
        model_dir: Path to model directory
        dataset: "merged" or "eic"
        eic_metadata_path: Path to eic_metadata.csv (required for EIC)
        
    Returns:
        Nested dictionary structure
    """
    preds_dir = model_dir / "preds"
    if not preds_dir.exists():
        raise FileNotFoundError(f"Predictions directory not found: {preds_dir}")
    
    # Get all biosamples
    biosamples = [d.name for d in preds_dir.iterdir() if d.is_dir()]
    print(f"Found {len(biosamples)} biosamples to process")
    
    # Initialize structure: data[comparison_type][assay][biosample] = {mu, var, obs}
    data = {"imputed": {}, "denoised": {}}
    
    # For EIC: cache metadata to avoid reading CSV multiple times
    eic_metadata_df = None
    if dataset == "eic":
        if eic_metadata_path is None:
            eic_metadata_path = Path(__file__).parent.parent / "data" / "eic_metadata.csv"
        if Path(eic_metadata_path).exists():
            eic_metadata_df = pd.read_csv(eic_metadata_path)
            print(f"Loaded EIC metadata with {len(eic_metadata_df)} rows")
        else:
            print(f"Warning: EIC metadata not found at {eic_metadata_path}")
    
    # Process each biosample
    for i, biosample in enumerate(biosamples):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processing biosample {i+1}/{len(biosamples)}: {biosample}")
        
        try:
            pred_data_dict = load_predictions_from_npz(preds_dir, biosample)
        except Exception as e:
            print(f"Warning: Failed to load {biosample}: {e}")
            continue
        
        if dataset == "merged":
            # For merged: directories end with _imputed or _denoised
            for assay_dir_name, pred_data in pred_data_dict.items():
                if assay_dir_name.endswith("_denoised"):
                    comparison_type = "denoised"
                    base_assay_name = assay_dir_name.replace("_denoised", "")
                elif assay_dir_name.endswith("_imputed"):
                    comparison_type = "imputed"
                    base_assay_name = assay_dir_name.replace("_imputed", "")
                else:
                    continue  # Skip unknown format
                
                if base_assay_name not in data[comparison_type]:
                    data[comparison_type][base_assay_name] = {}
                
                data[comparison_type][base_assay_name][biosample] = {
                    "mu": pred_data["mu"],
                    "var": pred_data["var"],
                    "obs": pred_data["observed_P"]
                }
        
        elif dataset == "eic":
            # For EIC: use cached metadata to determine imputed vs denoised
            if eic_metadata_df is not None:
                # Determine T_biosample name
                if biosample.startswith("B_"):
                    T_biosample = biosample.replace("B_", "T_")
                elif biosample.startswith("V_"):
                    T_biosample = biosample.replace("V_", "T_")
                elif biosample.startswith("T_"):
                    T_biosample = biosample
                else:
                    T_biosample = None
                
                # Get denoised assays for this biosample
                denoised_assays = set()
                if T_biosample:
                    T_rows = eic_metadata_df[eic_metadata_df['biosample_name'] == T_biosample]
                    if not T_rows.empty:
                        denoised_assays = set(T_rows['assay_name'].unique())
            else:
                # Fallback to function call if metadata not available
                _, denoised_assays = get_eic_assay_info(biosample, str(eic_metadata_path))
            
            for assay_dir_name, pred_data in pred_data_dict.items():
                # Extract base assay name (remove any suffixes)
                base_assay_name = assay_dir_name
                
                # Determine if this assay is denoised or imputed
                if base_assay_name in denoised_assays:
                    comparison_type = "denoised"
                else:
                    comparison_type = "imputed"
                
                if base_assay_name not in data[comparison_type]:
                    data[comparison_type][base_assay_name] = {}
                
                data[comparison_type][base_assay_name][biosample] = {
                    "mu": pred_data["mu"],
                    "var": pred_data["var"],
                    "obs": pred_data["observed_P"]
                }
    
    print(f"Data organization complete")
    return data


def get_calibration_curve(mu: np.ndarray, var: np.ndarray, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute calibration curve: empirical fraction within c% confidence interval vs c.
    
    Args:
        mu: Predicted means
        var: Predicted variances
        obs: Observed values
        
    Returns:
        Tuple of (confidence_levels, empirical_fractions)
    """
    # Create Gaussian distribution
    gaussian = Gaussian(mu, var)
    
    # Confidence levels from 0 to 1 in 0.01 intervals
    confidence_levels = np.arange(0.0, 1.01, 0.01)
    empirical_fractions = []
    
    for c in confidence_levels:
        if c == 0.0:
            empirical_fractions.append(0.0)
        elif c == 1.0:
            empirical_fractions.append(1.0)
        else:
            # Get confidence interval
            lower, upper = gaussian.interval(c)
            lower = lower.numpy() if hasattr(lower, 'numpy') else lower
            upper = upper.numpy() if hasattr(upper, 'numpy') else upper
            
            # Calculate fraction of observations within interval
            within_interval = np.logical_and(obs >= lower, obs <= upper)
            empirical_fractions.append(np.mean(within_interval))
    
    return confidence_levels, np.array(empirical_fractions)


def plot_calibration(
    data: Dict[str, Dict[str, Dict[str, Dict[str, np.ndarray]]]],
    comparison_type: str,
    output_path: Path
):
    """
    Generate calibration plots: one panel per assay type, one line per biosample.
    
    Args:
        data: Organized prediction data
        comparison_type: "imputed" or "denoised"
        output_path: Path to save figure
    """
    if comparison_type not in data or not data[comparison_type]:
        print(f"Warning: No data found for comparison type '{comparison_type}'")
        return
    
    assay_data = data[comparison_type]
    
    # Get unique assay types
    assay_types = sorted(assay_data.keys())
    
    if not assay_types:
        print(f"Warning: No assays found for comparison type '{comparison_type}'")
        return
    
    # Determine grid dimensions
    n_assays = len(assay_types)
    n_cols = min(3, n_assays)  # Max 3 columns
    n_rows = (n_assays + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows), squeeze=False)
    
    # Get color palette for biosamples
    all_biosamples = set()
    for assay in assay_data.values():
        all_biosamples.update(assay.keys())
    all_biosamples = sorted(all_biosamples)
    
    # Generate colors for biosamples
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_biosamples)))
    biosample_colors = {biosample: colors[i] for i, biosample in enumerate(all_biosamples)}
    
    # Count total combinations for progress reporting
    total_combinations = sum(len(assay_data[assay].keys()) for assay in assay_types)
    processed = 0
    
    # Plot each assay type
    for idx, assay_type in enumerate(assay_types):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        
        biosamples_in_assay = sorted(assay_data[assay_type].keys())
        
        print(f"  Processing assay {idx+1}/{n_assays}: {assay_type} ({len(biosamples_in_assay)} biosamples)")
        
        # Plot perfect calibration line (diagonal)
        ax.plot([0, 1], [0, 1], '--', color='orange', linewidth=2, label='Perfect calibration', alpha=0.7)
        
        # Plot calibration curve for each biosample
        for biosample_idx, biosample in enumerate(biosamples_in_assay):
            processed += 1
            if processed % 5 == 0 or processed == 1:
                print(f"    Computing calibration curve {processed}/{total_combinations}: {assay_type}/{biosample}")
            
            pred_data = assay_data[assay_type][biosample]
            mu = pred_data["mu"]
            var = pred_data["var"]
            obs = pred_data["obs"]
            
            try:
                conf_levels, emp_fracs = get_calibration_curve(mu, var, obs)
                ax.plot(conf_levels, emp_fracs, '-', color=biosample_colors[biosample], 
                       linewidth=1.5, label=biosample, alpha=0.8)
            except Exception as e:
                print(f"Warning: Failed to compute calibration for {assay_type}/{biosample}: {e}")
                continue
        
        ax.set_xlabel("Confidence level (c)", fontsize=12)
        ax.set_ylabel("Fraction within c% confidence interval", fontsize=12)
        ax.set_title(assay_type, fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=8, ncol=1)
    
    # Remove unused subplots
    for idx in range(n_assays, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        fig.delaxes(axes[row][col])
    
    plt.tight_layout()
    plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved calibration plot to {output_path}")


def plot_signal_loci(
    data: Dict[str, Dict[str, Dict[str, Dict[str, np.ndarray]]]],
    comparison_type: str,
    loci_coords: List[Tuple[int, int]],
    resolution: int,
    output_path: Path,
    use_sinh_transform: bool = True
):
    """
    Generate genomic loci signal visualization with confidence intervals.
    
    Two spaces:
    1. Normal space: sinh(arcsinh(P)) = P (original P values)
    2. Arcsinh space: arcsinh(P) (transformed space where mu/var are stored)
    
    Args:
        data: Organized prediction data (mu/var are in arcsinh space)
        comparison_type: "imputed" or "denoised"
        loci_coords: List of (start, end) tuples in base pairs
        resolution: Bin resolution in bp
        output_path: Path to save figure
        use_sinh_transform: If True, transform from arcsinh space to normal space using sinh()
                           (sinh(arcsinh(P)) = P) with log scale y-axis.
                           If False, stay in arcsinh space with linear scale y-axis.
    """
    if comparison_type not in data or not data[comparison_type]:
        print(f"Warning: No data found for comparison type '{comparison_type}'")
        return
    
    assay_data = data[comparison_type]
    
    # Convert loci coordinates to bin indices
    loci_bins = [(start // resolution, end // resolution) for start, end in loci_coords]
    
    # Collect all (assay, biosample) combinations, sorted by biosample then assay
    combinations = []
    biosample_order = []
    
    for assay_type in sorted(assay_data.keys()):
        for biosample in sorted(assay_data[assay_type].keys()):
            if biosample not in biosample_order:
                biosample_order.append(biosample)
            combinations.append((assay_type, biosample))
    
    # Sort combinations: group by biosample
    combinations_sorted = []
    for biosample in biosample_order:
        for assay_type in sorted(assay_data.keys()):
            if (assay_type, biosample) in combinations:
                combinations_sorted.append((assay_type, biosample))
    
    n_combinations = len(combinations_sorted)
    n_loci = len(loci_coords)
    
    if n_combinations == 0:
        print(f"Warning: No data combinations found for comparison type '{comparison_type}'")
        return
    
    # Create figure
    fig, axes = plt.subplots(n_combinations, n_loci, 
                            figsize=(3*n_loci, 2*n_combinations), squeeze=False)
    
    # Track biosample changes for separator lines
    prev_biosample = None
    
    print(f"  Processing {n_combinations} combinations across {n_loci} loci...")
    
    for row_idx, (assay_type, biosample) in enumerate(combinations_sorted):
        if (row_idx + 1) % 10 == 0 or row_idx == 0:
            print(f"    Processing combination {row_idx+1}/{n_combinations}: {assay_type}/{biosample}")
        pred_data = assay_data[assay_type][biosample]
        mu = pred_data["mu"]
        var = pred_data["var"]
        obs = pred_data["obs"]
        
        # Create Gaussian for confidence intervals
        gaussian = Gaussian(mu, var)
        lower_95, upper_95 = gaussian.interval(0.95)
        lower_95 = lower_95.numpy() if hasattr(lower_95, 'numpy') else lower_95
        upper_95 = upper_95.numpy() if hasattr(upper_95, 'numpy') else upper_95
        
        # Apply transform based on use_sinh_transform flag
        # Note: mu, var, obs are stored in arcsinh space
        if use_sinh_transform:
            # Transform from arcsinh space to normal space: sinh(arcsinh(P)) = P
            mu_plot = np.sinh(mu)
            lower_95_plot = np.sinh(lower_95)
            upper_95_plot = np.sinh(upper_95)
            obs_plot = np.sinh(obs)
        else:
            # Stay in arcsinh space (no transform)
            mu_plot = mu
            lower_95_plot = lower_95
            upper_95_plot = upper_95
            obs_plot = obs

        # Clip all values to min of 0
        mu_plot = np.clip(mu_plot, 0, None)
        lower_95_plot = np.clip(lower_95_plot, 0, None)
        upper_95_plot = np.clip(upper_95_plot, 0, None)
        obs_plot = np.clip(obs_plot, 0, None)
        
        for col_idx, (start_bp, end_bp) in enumerate(loci_coords):
            ax = axes[row_idx][col_idx]
            
            # Get bin coordinates
            start_bin, end_bin = loci_bins[col_idx]
            
            # Ensure indices are within bounds
            start_bin = max(0, min(start_bin, len(mu) - 1))
            end_bin = max(start_bin + 1, min(end_bin, len(mu)))
            
            x_values = np.arange(start_bin, end_bin)
            
            # Extract data slices
            mu_slice = mu_plot[start_bin:end_bin]
            lower_slice = lower_95_plot[start_bin:end_bin]
            upper_slice = upper_95_plot[start_bin:end_bin]
            obs_slice = obs_plot[start_bin:end_bin]
            
            # Fill between for 95% confidence interval
            ax.fill_between(x_values, lower_slice, upper_slice, 
                          color='coral', alpha=0.4, label='95% CI')
            
            # Plot mean prediction
            ax.plot(x_values, mu_slice, color='red', linewidth=0.5, label='Mean')
            
            # Plot observed
            ax.plot(x_values, obs_slice, color='royalblue', linewidth=0.4, 
                   alpha=0.8, label='Observed')
            
            # Set scale based on transform
            if use_sinh_transform:
                # Normal space: use log scale (sinh(arcsinh(P)) = P)
                # ax.set_yscale('log')
                # Ensure y-axis starts from a small positive value (log scale can't start at 0)
                # Collect all positive values from all arrays
                positive_vals = []
                for arr in [lower_slice, obs_slice, mu_slice]:
                    pos_arr = arr[arr > 0]
                    if len(pos_arr) > 0:
                        positive_vals.append(np.min(pos_arr))
                
                if len(positive_vals) > 0:
                    y_min = min(positive_vals)
                    if y_min > 0:
                        ax.set_ylim(bottom=y_min * 0.1)
            else:
                # Arcsinh space: use linear scale
                ax.set_yscale('linear')
            
            # Set labels and title
            if row_idx == 0:
                ax.set_title(f"chr21 {start_bp:,} : {end_bp:,}", fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(f"{assay_type}\n{biosample}", fontsize=8)
            else:
                ax.set_ylabel("")
            
            ax.set_xticklabels([])
            ax.grid(True, alpha=0.3)
            
            # Add legend only for first subplot
            if row_idx == 0 and col_idx == 0:
                ax.legend(loc='upper left', fontsize=6)
        
        prev_biosample = biosample
    
    # Add horizontal separator lines between different biosamples
    # We'll do this by adjusting subplot spacing and adding lines
    for row_idx in range(1, n_combinations):
        prev_assay, prev_biosample = combinations_sorted[row_idx - 1]
        curr_assay, curr_biosample = combinations_sorted[row_idx]
        
        if prev_biosample != curr_biosample:
            # Add a separator line at the top of current row
            for col_idx in range(n_loci):
                ax = axes[row_idx][col_idx]
                # Draw line at top of plot area
                y_max = ax.get_ylim()[1]
                ax.axhline(y=y_max, color='black', linewidth=1.5, clip_on=False, zorder=100)
    
    plt.tight_layout()
    plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved signal loci plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate confidence calibration and genomic loci visualizations"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to model directory containing preds/ subdirectory"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["merged", "eic"],
        help="Dataset type: 'merged' or 'eic'"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=25,
        help="Bin resolution in bp (default: 25)"
    )
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Create output directory
    viz_dir = model_dir / "viz"
    viz_dir.mkdir(exist_ok=True)
    
    # Load and organize predictions
    print("Loading predictions...")
    eic_metadata_path = None
    if args.dataset == "eic":
        eic_metadata_path = Path(__file__).parent.parent / "data" / "eic_metadata.csv"
    
    try:
        data = organize_predictions_by_type(model_dir, args.dataset, eic_metadata_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure pred.py has been run successfully before running this script.")
        sys.exit(1)
    
    # Define genomic loci coordinates (from old_eval.py)
    loci_coords = [
        (33481539, 33588914),  # GART
        (25800151, 26235914),  # APP
        (31589009, 31745788),  # SOD1
        (39526359, 39802081),  # B3GALT5
        (33577551, 33919338)   # ITSN1
    ]
    
    # Generate calibration plots
    print("\nGenerating calibration plots...")
    for comparison_type in ["imputed", "denoised"]:
        print(f"\n  Processing {comparison_type} predictions...")
        output_path = viz_dir / f"calibration_{comparison_type}.svg"
        plot_calibration(data, comparison_type, output_path)
    
    # Generate signal loci plots (both normal space and arcsinh space versions)
    print("\nGenerating signal loci plots...")
    for comparison_type in ["imputed", "denoised"]:
        print(f"\n  Processing {comparison_type} predictions...")
        
        # Version 1: Normal space (sinh(arcsinh(P)) = P) with log scale
        print(f"    Generating normal space version (sinh transform, log scale)...")
        output_path = viz_dir / f"signal_loci_{comparison_type}_sinh.svg"
        plot_signal_loci(data, comparison_type, loci_coords, args.resolution, output_path, use_sinh_transform=True)
        
        # Version 2: Arcsinh space with linear scale
        print(f"    Generating arcsinh space version (no transform, linear scale)...")
        output_path = viz_dir / f"signal_loci_{comparison_type}_arcsinh.svg"
        plot_signal_loci(data, comparison_type, loci_coords, args.resolution, output_path, use_sinh_transform=False)
    
    print("\n✓ All visualizations complete!")


if __name__ == "__main__":
    main()

