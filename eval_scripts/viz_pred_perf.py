#!/usr/bin/env python3
"""
Visualize Prediction Performance Metrics

This script loads metrics CSV files and generates boxplot visualizations
for imputed and denoised assays separately.

Author: CANDI Team
"""

import argparse
import sys
import math
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def generate_plots(df: pd.DataFrame, comparison_type: str, output_path: Path):
    """
    Generate boxplot visualizations for a given comparison type.
    
    Args:
        df: DataFrame with metrics data
        comparison_type: 'imputed' or 'denoised'
        output_path: Path to save the SVG file
    """
    # Filter to specific comparison type
    df_filtered = df[df['comparison'] == comparison_type]
    
    if df_filtered.empty:
        print(f"Warning: No data found for comparison type '{comparison_type}'")
        return
    
    # Identify metric columns (exclude non-metric fields)
    non_metric_cols = ['bios', 'assay', 'comparison', 'available_assays']
    metric_cols = [c for c in df_filtered.columns if c not in non_metric_cols]
    
    if not metric_cols:
        print(f"Warning: No metric columns found for comparison type '{comparison_type}'")
        return
    
    # Group metrics by type
    def get_metric_type(metric_name):
        """Extract the metric type from the metric name."""
        # Normalize the name
        name = metric_name.lower()
        
        # Identify base type
        if 'mse' in name:
            return 'MSE'
        elif 'pearson' in name:
            return 'Pearson'
        elif 'spearman' in name:
            return 'Spearman'
        elif 'aucroc' in name:
            return 'AUCROC'
        elif 'cidx' in name:
            return 'Cidx'
        elif '95ci_coverage' in name or '95ci' in name:
            return '95CI_coverage'
        else:
            return 'Other'
    
    # Group metrics by type
    metric_groups = {}
    for metric in metric_cols:
        mtype = get_metric_type(metric)
        if mtype not in metric_groups:
            metric_groups[mtype] = []
        metric_groups[mtype].append(metric)
    
    # Sort metrics within each group (arcsinh versions first, then alphabetical)
    for mtype in metric_groups:
        metric_groups[mtype].sort(key=lambda x: (not x.endswith('_arcsinh'), x))
    
    # Define row order
    row_order = ['MSE', 'Pearson', 'Spearman', 'AUCROC', 'Cidx', '95CI_coverage', 'Other']
    
    # Create ordered list of groups (only those that exist)
    ordered_groups = []
    for group_type in row_order:
        if group_type in metric_groups and metric_groups[group_type]:
            ordered_groups.append((group_type, metric_groups[group_type]))
    
    # Add any remaining groups not in the predefined order
    for group_type, metrics in metric_groups.items():
        if group_type not in row_order:
            ordered_groups.append((group_type, metrics))
    
    # Determine grid dimensions: one row per metric type, columns = max metrics in any group
    n_rows = len(ordered_groups)
    n_cols = max(len(metrics) for _, metrics in ordered_groups) if ordered_groups else 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), squeeze=False)
    
    # Plot metrics grouped by type (one row per type)
    for row_idx, (group_type, group_metrics) in enumerate(ordered_groups):
        for col_idx, metric in enumerate(group_metrics):
            ax = axes[row_idx][col_idx]
            
            # Compute medians and sort assays
            medians = df_filtered.groupby('assay')[metric].median().sort_values(ascending=False)
            assays_sorted = medians.index.tolist()
            
            # Prepare data for boxplot
            data = [df_filtered[df_filtered['assay']==assay][metric].dropna() for assay in assays_sorted]
            
            # Filter out empty data
            data_filtered = [(assay, vals) for assay, vals in zip(assays_sorted, data) if len(vals) > 0]
            if not data_filtered:
                ax.text(0.5, 0.5, f'No data for {metric}', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes)
                ax.set_title(metric)
                continue
            
            assays_plot = [a for a, v in data_filtered]
            data_plot = [v for a, v in data_filtered]
            
            # Draw boxplot without outliers
            bp = ax.boxplot(data_plot, patch_artist=True, showfliers=False)
            
            # Set box colors based on comparison type with transparency
            color = 'salmon' if comparison_type == 'imputed' else 'lightblue'
            for box in bp['boxes']:
                box.set_facecolor(color)
                box.set_alpha(0.6)  # Make boxes semi-transparent
            
            # Overlay stripplot with grey dots
            for box_idx, (assay, values) in enumerate(data_filtered):
                x_pos = box_idx + 1
                # Add small random jitter to x-position to avoid overplotting
                # Use a deterministic seed based on metric and assay for consistent positioning
                np.random.seed(hash(f"{metric}_{assay}") % (2**32))
                jitter = np.random.normal(0, 0.05, len(values))
                x_coords = x_pos + jitter
                ax.scatter(x_coords, values, color='grey', alpha=0.4, s=20, zorder=10)
            
            ax.set_title(metric)
            ax.set_xticks(range(1, len(assays_plot) + 1))
            ax.set_xticklabels(assays_plot, rotation=90, fontsize=8)
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)
        
        # Remove unused subplots in this row
        for col_idx in range(len(group_metrics), n_cols):
            fig.delaxes(axes[row_idx][col_idx])
    
    plt.tight_layout()
    
    # Save as SVG
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format='svg')
    plt.close()
    
    print(f"Saved plot to {output_path}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Visualize prediction performance metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using model directory (looks for preds/metrics.csv)
  python eval_scripts/viz_pred_perf.py --model-dir models/my_model/

  # Using direct path to CSV file
  python eval_scripts/viz_pred_perf.py --metrics-csv models/my_model/preds/metrics.csv
        """
    )
    
    # Arguments (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model-dir', type=str,
                       help='Path to model directory containing preds/metrics.csv')
    group.add_argument('--metrics-csv', type=str,
                       help='Direct path to metrics CSV file')
    
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for plots (default: model_dir/viz/ or metrics_csv_dir/viz/)')
    
    args = parser.parse_args()
    
    # Determine CSV path
    if args.metrics_csv:
        csv_path = Path(args.metrics_csv)
        if not csv_path.exists():
            print(f"Error: Metrics CSV file not found: {csv_path}")
            sys.exit(1)
        base_dir = csv_path.parent.parent  # Go up from preds/ to model_dir/
    else:
        model_dir = Path(args.model_dir)
        csv_path = model_dir / "preds" / "metrics.csv"
        if not csv_path.exists():
            print(f"Error: Metrics CSV file not found: {csv_path}")
            sys.exit(1)
        base_dir = model_dir
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = base_dir / "viz"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the data
    print(f"Loading metrics from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)
    
    # Check for required columns
    required_cols = ['bios', 'assay', 'comparison']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"Error: Required columns missing from CSV: {missing_cols}")
        sys.exit(1)
    
    # Check available comparison types
    available_comparisons = df['comparison'].unique()
    print(f"Found comparison types: {list(available_comparisons)}")
    
    # Generate plots for each comparison type
    for comparison_type in ['imputed', 'denoised']:
        if comparison_type in available_comparisons:
            output_path = output_dir / f"metrics_{comparison_type}.svg"
            print(f"\nGenerating plot for {comparison_type} assays...")
            generate_plots(df, comparison_type, output_path)
        else:
            print(f"\nSkipping {comparison_type} (no data available)")
    
    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()

