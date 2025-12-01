#!/usr/bin/env python3
"""
Visualize Prediction Performance with Hexbin Density Plots

This script generates hexbin density plots for observed vs predicted pval signals
across biosample-assay combinations. Includes density plots and uncertainty-colored
hexbin plots (by std and CV).

Author: CANDI Team
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from matplotlib.colors import LogNorm
import warnings
warnings.filterwarnings('ignore')


def load_predictions(preds_dir, biosample, assay):
    """
    Load prediction data for a specific biosample-assay pair.
    
    Args:
        preds_dir: Path to predictions directory
        biosample: Biosample name
        assay: Assay name
        
    Returns:
        Dictionary with obs, pred_mu, pred_std arrays or None if files don't exist
    """
    data_path = preds_dir / biosample / assay
    
    mu_path = data_path / "mu.npz"
    var_path = data_path / "var.npz"
    obs_path = data_path / "observed_P.npz"
    
    # Check if all required files exist
    if not (mu_path.exists() and var_path.exists() and obs_path.exists()):
        return None
    
    try:
        # Load data
        mu = np.load(mu_path)['arr_0']
        var = np.load(var_path)['arr_0']
        obs = np.load(obs_path)['arr_0']
        
        # Calculate std from variance
        pred_std = np.sqrt(var)
        
        # Filter out any NaN or inf values
        valid_mask = np.isfinite(obs) & np.isfinite(mu) & np.isfinite(pred_std)
        
        return {
            'obs': obs[valid_mask],
            'pred_mu': mu[valid_mask],
            'pred_std': pred_std[valid_mask]
        }
    except Exception as e:
        print(f"Warning: Error loading data for {biosample}/{assay}: {e}")
        return None


def calculate_pearson_r(x, y):
    """
    Calculate Pearson correlation coefficient.
    
    Args:
        x, y: Arrays to correlate
        
    Returns:
        Pearson r value
    """
    try:
        r, _ = pearsonr(x, y)
        return r
    except:
        return np.nan


def plot_density_scatter(data_dict, comparison_type, output_path, transform='none'):
    """
    Create hexbin density plots for all biosample-assay pairs.
    
    Args:
        data_dict: Dictionary mapping (biosample, assay) -> data
        comparison_type: 'imputed' or 'denoised'
        output_path: Path to save the plot
        transform: 'none' or 'arcsinh'
    """
    # Get unique biosamples and assays
    biosamples = sorted(set(k[0] for k in data_dict.keys()))
    assays = sorted(set(k[1] for k in data_dict.keys()))
    
    n_rows = len(biosamples)
    n_cols = len(assays)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
    
    for i, biosample in enumerate(biosamples):
        for j, assay in enumerate(assays):
            ax = axes[i, j]
            
            key = (biosample, assay)
            if key not in data_dict:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{biosample}\n{assay}')
                continue
            
            data = data_dict[key]
            
            # Data is stored in arcsinh space. Apply transformation based on request.
            # 'arcsinh': Keep in arcsinh space (as stored)
            # 'normal' or other: Convert to original space using sinh
            if transform == 'arcsinh':
                # Keep in arcsinh space
                obs = data['obs']
                pred = data['pred_mu']
                label_suffix = ' (arcsinh)'
            else:
                # Convert back to original space (consistent with P_Pearson-GW in compute_metrics.py)
                obs = np.sinh(data['obs'])
                pred = np.sinh(data['pred_mu'])
                label_suffix = ' (original)'
            
            # Calculate Pearson r
            r = calculate_pearson_r(obs, pred)
            
            # Create hexbin plot with LogNorm
            hb = ax.hexbin(obs, pred, gridsize=100, cmap='viridis', 
                          mincnt=1, bins='log', norm=LogNorm(), rasterized=True)
            
            # Add colorbar
            cbar = plt.colorbar(hb, ax=ax)
            cbar.set_label('Log10(Counts)', fontsize=10)
            
            # Add diagonal line
            lim_min = min(obs.min(), pred.min())
            lim_max = max(obs.max(), pred.max())
            ax.plot([lim_min, lim_max], [lim_min, lim_max], '--', color='red', 
                   alpha=0.8, linewidth=2, label='y=x')
            
            # Set labels and title
            ax.set_xlabel(f'Observed{label_suffix}')
            ax.set_ylabel(f'Predicted{label_suffix}')
            ax.set_title(f'{biosample} - {assay}\nPearson r = {r:.3f}')
            ax.legend(loc='upper left', fontsize=8)
            ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    plt.close()
    print(f"Saved hexbin density plot to {output_path}")


def plot_std_scatter(data_dict, comparison_type, output_path, transform='none'):
    """
    Create hexbin plots colored by predicted standard deviation.
    
    Args:
        data_dict: Dictionary mapping (biosample, assay) -> data
        comparison_type: 'imputed' or 'denoised'
        output_path: Path to save the plot
        transform: 'none' or 'arcsinh'
    """
    # Get unique biosamples and assays
    biosamples = sorted(set(k[0] for k in data_dict.keys()))
    assays = sorted(set(k[1] for k in data_dict.keys()))
    
    n_rows = len(biosamples)
    n_cols = len(assays)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
    
    for i, biosample in enumerate(biosamples):
        for j, assay in enumerate(assays):
            ax = axes[i, j]
            
            key = (biosample, assay)
            if key not in data_dict:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{biosample}\n{assay}')
                continue
            
            data = data_dict[key]
            
            # Data is stored in arcsinh space. Apply transformation based on request.
            if transform == 'arcsinh':
                # Keep in arcsinh space
                obs = data['obs']
                pred = data['pred_mu']
                std_color = data['pred_std']
                label_suffix = ' (arcsinh)'
            else:
                # Convert back to original space
                obs = np.sinh(data['obs'])
                pred = np.sinh(data['pred_mu'])
                std_color = np.sinh(data['pred_std'])
                label_suffix = ' (original)'
            
            # Calculate Pearson r
            r = calculate_pearson_r(obs, pred)
            
            # Hexbin plot colored by mean std in each bin
            hb = ax.hexbin(obs, pred, C=std_color, gridsize=100, 
                          cmap='viridis', reduce_C_function=np.mean, 
                          mincnt=1, rasterized=True)
            
            # Add colorbar
            cbar = plt.colorbar(hb, ax=ax)
            cbar.set_label(f'Mean Predicted Std{label_suffix}', fontsize=10)
            
            # Add diagonal line
            lim_min = min(obs.min(), pred.min())
            lim_max = max(obs.max(), pred.max())
            ax.plot([lim_min, lim_max], [lim_min, lim_max], '--', color='red', 
                   alpha=0.8, linewidth=2)
            
            # Set labels and title
            ax.set_xlabel(f'Observed{label_suffix}')
            ax.set_ylabel(f'Predicted{label_suffix}')
            ax.set_title(f'{biosample} - {assay}\nPearson r = {r:.3f}')
            ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    plt.close()
    print(f"Saved std-colored hexbin plot to {output_path}")


def plot_cv_scatter(data_dict, comparison_type, output_path, transform='none'):
    """
    Create hexbin plots colored by coefficient of variation (CV = std/mean).
    
    Args:
        data_dict: Dictionary mapping (biosample, assay) -> data
        comparison_type: 'imputed' or 'denoised'
        output_path: Path to save the plot
        transform: 'none' or 'arcsinh'
    """
    # Get unique biosamples and assays
    biosamples = sorted(set(k[0] for k in data_dict.keys()))
    assays = sorted(set(k[1] for k in data_dict.keys()))
    
    n_rows = len(biosamples)
    n_cols = len(assays)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
    
    for i, biosample in enumerate(biosamples):
        for j, assay in enumerate(assays):
            ax = axes[i, j]
            
            key = (biosample, assay)
            if key not in data_dict:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{biosample}\n{assay}')
                continue
            
            data = data_dict[key]
            
            # Data is stored in arcsinh space. Apply transformation based on request.
            if transform == 'arcsinh':
                # Keep in arcsinh space
                obs = data['obs']
                pred = data['pred_mu']
                # Calculate CV in arcsinh space
                cv = np.divide(data['pred_std'], data['pred_mu'], 
                              out=np.zeros_like(data['pred_std']), 
                              where=np.abs(data['pred_mu']) > 1e-8)
                label_suffix = ' (arcsinh)'
            else:
                # Convert back to original space
                obs = np.sinh(data['obs'])
                pred = np.sinh(data['pred_mu'])
                pred_std_orig = np.sinh(data['pred_std'])
                # Calculate CV in original space
                cv = np.divide(pred_std_orig, pred, 
                              out=np.zeros_like(pred_std_orig), 
                              where=np.abs(pred) > 1e-8)
                label_suffix = ' (original)'
            
            # Calculate Pearson r
            r = calculate_pearson_r(obs, pred)
            
            # Hexbin plot colored by mean CV in each bin
            hb = ax.hexbin(obs, pred, C=cv, gridsize=100, 
                          cmap='plasma', reduce_C_function=np.mean, 
                          mincnt=1, rasterized=True)
            
            # Add colorbar
            cbar = plt.colorbar(hb, ax=ax)
            cbar.set_label(f'Mean CV (std/mean){label_suffix}', fontsize=10)
            
            # Add diagonal line
            lim_min = min(obs.min(), pred.min())
            lim_max = max(obs.max(), pred.max())
            ax.plot([lim_min, lim_max], [lim_min, lim_max], '--', color='red', 
                   alpha=0.8, linewidth=2)
            
            # Set labels and title
            ax.set_xlabel(f'Observed{label_suffix}')
            ax.set_ylabel(f'Predicted{label_suffix}')
            ax.set_title(f'{biosample} - {assay}\nPearson r = {r:.3f}')
            ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    plt.close()
    print(f"Saved CV-colored hexbin plot to {output_path}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Visualize prediction performance with hexbin density and uncertainty scatter plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using model directory (looks for preds/ subdirectory)
  python eval_scripts/viz_scatter_density.py --model-dir models/my_model/

  # Using direct path to preds directory
  python eval_scripts/viz_scatter_density.py --preds-dir models/my_model/preds/
        """
    )
    
    # Arguments (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model-dir', type=str,
                       help='Path to model directory containing preds/ subdirectory')
    group.add_argument('--preds-dir', type=str,
                       help='Direct path to predictions directory')
    
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for plots (default: model_dir/viz/ or preds_dir/../viz/)')
    
    parser.add_argument('--comparison-type', type=str, default='both', 
                       choices=['imputed', 'denoised', 'both'],
                       help='Which comparison type to plot (default: both)')
    
    args = parser.parse_args()
    
    # Determine preds directory
    if args.preds_dir:
        preds_dir = Path(args.preds_dir)
        if not preds_dir.exists():
            print(f"Error: Predictions directory not found: {preds_dir}")
            sys.exit(1)
        base_dir = preds_dir.parent
    else:
        model_dir = Path(args.model_dir)
        preds_dir = model_dir / "preds"
        if not preds_dir.exists():
            print(f"Error: Predictions directory not found: {preds_dir}")
            sys.exit(1)
        base_dir = model_dir
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = base_dir / "viz"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metrics CSV to get biosample-assay pairs and comparison types
    metrics_path = preds_dir / "metrics.csv"
    if not metrics_path.exists():
        print(f"Error: Metrics CSV file not found: {metrics_path}")
        sys.exit(1)
    
    print(f"Loading metrics from: {metrics_path}")
    df = pd.read_csv(metrics_path)
    
    # Get comparison types to process
    available_comparisons = df['comparison'].unique()
    if args.comparison_type == 'both':
        comparison_types = [ct for ct in ['imputed', 'denoised'] if ct in available_comparisons]
    else:
        if args.comparison_type not in available_comparisons:
            print(f"Error: Comparison type '{args.comparison_type}' not found in data")
            sys.exit(1)
        comparison_types = [args.comparison_type]
    
    print(f"Processing comparison types: {comparison_types}")
    
    # Process each comparison type
    for comparison_type in comparison_types:
        print(f"\n{'='*60}")
        print(f"Processing {comparison_type} assays...")
        print(f"{'='*60}")
        
        # Filter to specific comparison type
        df_filtered = df[df['comparison'] == comparison_type]
        
        # Load all data for this comparison type
        data_dict = {}
        for _, row in df_filtered.iterrows():
            biosample = row['bios']
            assay = row['assay']
            
            print(f"  Loading {biosample}/{assay}...", end=' ')
            data = load_predictions(preds_dir, biosample, assay)
            if data is not None:
                data_dict[(biosample, assay)] = data
                print(f"✓ ({len(data['obs'])} points)")
            else:
                print("✗ (no data)")
        
        if not data_dict:
            print(f"Warning: No data found for {comparison_type} assays")
            continue
        
        print(f"\nLoaded {len(data_dict)} biosample-assay pairs")
        
        # Generate all plots
        print("\nGenerating plots...")
        
        # 1. Hexbin density plot - normal space
        print("  1/6: Hexbin density plot (normal space)...")
        plot_density_scatter(data_dict, comparison_type, 
                           output_dir / f"{comparison_type}_density_scatter_normal.svg",
                           transform='none')
        
        # 2. Hexbin density plot - arcsinh space
        print("  2/6: Hexbin density plot (arcsinh space)...")
        plot_density_scatter(data_dict, comparison_type,
                           output_dir / f"{comparison_type}_density_scatter_arcsinh.svg",
                           transform='arcsinh')
        
        # 3. Std-colored hexbin - normal space
        print("  3/6: Std-colored hexbin (normal space)...")
        plot_std_scatter(data_dict, comparison_type,
                        output_dir / f"{comparison_type}_std_scatter_normal.svg",
                        transform='none')
        
        # 4. Std-colored hexbin - arcsinh space
        print("  4/6: Std-colored hexbin (arcsinh space)...")
        plot_std_scatter(data_dict, comparison_type,
                        output_dir / f"{comparison_type}_std_scatter_arcsinh.svg",
                        transform='arcsinh')
        
        # 5. CV-colored hexbin - normal space
        print("  5/6: CV-colored hexbin (normal space)...")
        plot_cv_scatter(data_dict, comparison_type,
                       output_dir / f"{comparison_type}_cv_scatter_normal.svg",
                       transform='none')
        
        # 6. CV-colored hexbin - arcsinh space
        print("  6/6: CV-colored hexbin (arcsinh space)...")
        plot_cv_scatter(data_dict, comparison_type,
                       output_dir / f"{comparison_type}_cv_scatter_arcsinh.svg",
                       transform='arcsinh')
        
        print(f"\n✓ Completed all plots for {comparison_type} assays")
    
    print(f"\n{'='*60}")
    print(f"All plots saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

