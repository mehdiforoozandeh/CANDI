#!/usr/bin/env python3
"""
Visualize Model Performance Across Quantiles

This script evaluates model calibration by computing metrics at different quantile
levels of the predicted distribution (using icdf). It produces line charts showing
how metrics vary with quantile, aggregated across biosample-assay experiments.

Author: CANDI Team
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Import from project
sys.path.insert(0, str(Path(__file__).parent.parent))
from _utils import Gaussian, Laplace, StudentsT, Gamma, METRICS


def get_dist_type(model_dir: Path) -> str:
    """Load model configuration to determine distribution type."""
    config_files = list(model_dir.glob("*_config.json"))
    if not config_files:
        print(f"Warning: No config JSON file found in {model_dir}. Assuming gaussian.")
        return "gaussian"
    
    config_path = config_files[0]
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config.get('dist-type', config.get('dist_type', 'gaussian'))


def load_signal_predictions(preds_dir: Path, biosample: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load signal predictions from NPZ files for a given biosample.
    
    Returns:
        Dictionary: {assay_dir_name: {mu, var, df (optional), observed_P}}
    """
    bios_dir = preds_dir / biosample
    if not bios_dir.exists():
        raise FileNotFoundError(f"Biosample directory not found: {bios_dir}")
    
    results = {}
    assay_dirs = [d for d in bios_dir.iterdir() if d.is_dir()]
    
    for assay_dir in assay_dirs:
        assay_dir_name = assay_dir.name
        
        try:
            mu_data = np.load(assay_dir / "mu.npz")['arr_0']
            var_data = np.load(assay_dir / "var.npz")['arr_0']
            obs_p_data = np.load(assay_dir / "observed_P.npz")['arr_0']
            
            result = {
                "mu": mu_data,
                "var": var_data,
                "observed_P": obs_p_data
            }
            
            # Load df if available (for Student's T)
            df_path = assay_dir / "df.npz"
            if df_path.exists():
                result["df"] = np.load(df_path)['arr_0']
            
            results[assay_dir_name] = result
            
        except Exception as e:
            print(f"Warning: Failed to load predictions for {biosample}/{assay_dir_name}: {e}")
            continue
    
    return results


def build_distribution(mu: np.ndarray, var: np.ndarray, dist_type: str, 
                       df: Optional[np.ndarray] = None):
    """Build the appropriate distribution object based on dist_type."""
    mu_t = torch.tensor(mu, dtype=torch.float32)
    var_t = torch.tensor(var, dtype=torch.float32)
    
    if dist_type in ['laplace', 'laplace_const', 'mae']:
        return Laplace(mu_t, var_t)
    elif dist_type == 'studentst':
        if df is not None:
            df_t = torch.tensor(df, dtype=torch.float32)
            return StudentsT(mu_t, var_t, df_t)
        else:
            print("Warning: Student's t requested but df not found. Using Gaussian.")
            return Gaussian(mu_t, var_t)
    elif dist_type == 'gamma':
        return Gamma(mu_t, var_t)
    else:
        # gaussian, gaussian_const, mse, etc.
        return Gaussian(mu_t, var_t)


def compute_metrics_at_quantile(pred: np.ndarray, obs: np.ndarray, metrics_obj: METRICS, signal_transform: str = 'arcsinh') -> Dict[str, float]:
    """
    Compute metrics comparing predicted values (at a specific quantile) to observed.
    Values are transformed back to original space using inverse transform before computing metrics.
    Computes both GW (genome-wide) and 1obs (top 1% observed positions) versions.
    
    Args:
        pred: Predicted values in transformed space
        obs: Observed values in transformed space
        metrics_obj: METRICS object for computing metrics
        signal_transform: Signal transformation type ('arcsinh', 'log1p', 'none')
    """
    # Transform back to original space
    if signal_transform == 'arcsinh':
        pred_orig = np.sinh(pred)
        obs_orig = np.sinh(obs)
    elif signal_transform == 'log1p':
        pred_orig = np.expm1(pred)
        obs_orig = np.expm1(obs)
    else:  # 'none'
        pred_orig = pred
        obs_orig = obs
    
    # Filter out NaN/Inf
    mask = np.isfinite(pred_orig) & np.isfinite(obs_orig)
    pred_clean = pred_orig[mask]
    obs_clean = obs_orig[mask]
    
    if len(pred_clean) < 10:
        return {
            'mse_gw': np.nan, 'mae_gw': np.nan, 'pearson_gw': np.nan, 'spearman_gw': np.nan,
            'mse_1obs': np.nan, 'mae_1obs': np.nan, 'pearson_1obs': np.nan, 'spearman_1obs': np.nan
        }
    
    # Genome-wide metrics
    mse_gw = metrics_obj.mse(obs_clean, pred_clean)
    mae_gw = metrics_obj.mae(obs_clean, pred_clean)
    pearson_gw = metrics_obj.pearson(obs_clean, pred_clean)
    spearman_gw = metrics_obj.spearman(obs_clean, pred_clean)
    
    # Top 1% observed positions metrics
    mse_1obs = metrics_obj.mse1obs(obs_clean, pred_clean)
    mae_1obs = metrics_obj.mae1obs(obs_clean, pred_clean)
    pearson_1obs = metrics_obj.pearson1_obs(obs_clean, pred_clean)
    spearman_1obs = metrics_obj.spearman1_obs(obs_clean, pred_clean)
    
    return {
        'mse_gw': mse_gw, 'mae_gw': mae_gw, 'pearson_gw': pearson_gw, 'spearman_gw': spearman_gw,
        'mse_1obs': mse_1obs, 'mae_1obs': mae_1obs, 'pearson_1obs': pearson_1obs, 'spearman_1obs': spearman_1obs
    }


def get_eic_denoised_assays(biosample: str, eic_metadata_path: Path) -> Set[str]:
    """Get set of assay names that are denoised for EIC dataset."""
    if biosample.startswith("B_"):
        T_biosample = biosample.replace("B_", "T_")
    elif biosample.startswith("V_"):
        T_biosample = biosample.replace("V_", "T_")
    elif biosample.startswith("T_"):
        T_biosample = biosample
    else:
        return set()
    
    try:
        metadata_df = pd.read_csv(eic_metadata_path)
        T_rows = metadata_df[metadata_df['biosample_name'] == T_biosample]
        return set(T_rows['assay_name'].unique())
    except Exception as e:
        print(f"Warning: Could not load EIC metadata: {e}")
        return set()


def process_biosample(
    preds_dir: Path,
    biosample: str,
    dist_type: str,
    quantiles: np.ndarray,
    dataset: str,
    metrics_obj: METRICS,
    eic_metadata_path: Optional[Path] = None,
    signal_transform: str = 'arcsinh'
) -> List[Dict]:
    """
    Process a single biosample and compute metrics at each quantile.
    
    Returns:
        List of dictionaries with columns: biosample, assay, assay_type, mode, quantile, metric, value
    """
    pred_data_dict = load_signal_predictions(preds_dir, biosample)
    
    # For EIC, determine which assays are denoised
    denoised_assays = set()
    if dataset == 'eic' and eic_metadata_path is not None:
        denoised_assays = get_eic_denoised_assays(biosample, eic_metadata_path)
    
    results = []
    
    for assay_dir_name, pred_data in pred_data_dict.items():
        # Determine mode (imp vs ups)
        if dataset == 'merged':
            if assay_dir_name.endswith("_denoised"):
                mode = "ups"
                assay_name = assay_dir_name.replace("_denoised", "")
            elif assay_dir_name.endswith("_imputed"):
                mode = "imp"
                assay_name = assay_dir_name.replace("_imputed", "")
            else:
                mode = "unknown"
                assay_name = assay_dir_name
        else:  # eic
            assay_name = assay_dir_name
            mode = "ups" if assay_name in denoised_assays else "imp"
        
        # Determine assay type (group similar assays)
        assay_type = get_assay_type(assay_name)
        
        # Build distribution
        dist = build_distribution(
            pred_data['mu'],
            pred_data['var'],
            dist_type,
            pred_data.get('df')
        )
        
        obs = pred_data['observed_P']
        
        # Compute metrics at each quantile
        for q in quantiles:
            pred_at_q = dist.icdf(torch.tensor(q)).numpy()
            metrics = compute_metrics_at_quantile(pred_at_q, obs, metrics_obj, signal_transform)
            
            for metric_name, metric_value in metrics.items():
                results.append({
                    'biosample': biosample,
                    'assay': assay_name,
                    'assay_type': assay_type,
                    'mode': mode,
                    'quantile': q,
                    'metric': metric_name,
                    'value': metric_value
                })
    
    return results


def get_assay_type(assay_name: str) -> str:
    """Group assays into types for visualization."""
    assay_upper = assay_name.upper()
    
    # Histone marks
    if 'H3K4ME3' in assay_upper:
        return 'H3K4me3'
    elif 'H3K4ME1' in assay_upper:
        return 'H3K4me1'
    elif 'H3K27AC' in assay_upper:
        return 'H3K27ac'
    elif 'H3K27ME3' in assay_upper:
        return 'H3K27me3'
    elif 'H3K36ME3' in assay_upper:
        return 'H3K36me3'
    elif 'H3K9ME3' in assay_upper:
        return 'H3K9me3'
    elif 'H3K9AC' in assay_upper:
        return 'H3K9ac'
    
    # DNase/ATAC
    elif 'DNASE' in assay_upper or 'DHS' in assay_upper:
        return 'DNase'
    elif 'ATAC' in assay_upper:
        return 'ATAC-seq'
    
    # TFs
    elif 'CTCF' in assay_upper:
        return 'CTCF'
    elif 'POL2' in assay_upper or 'POLR2A' in assay_upper:
        return 'Pol2'
    
    # Other
    else:
        return 'Other'


def plot_quantile_performance(df: pd.DataFrame, output_dir: Path, model_name: str):
    """
    Create multipanel figure with assay types as rows and metrics as columns.
    8 columns: MAE-GW, MAE-1obs, MSE-GW, MSE-1obs, Pearson-GW, Pearson-1obs, Spearman-GW, Spearman-1obs
    Rows: Different assay types
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter to only imp and ups modes
    df_plot = df[df['mode'].isin(['imp', 'ups'])].copy()
    
    if df_plot.empty:
        print("No data to plot")
        return
    
    # Get unique assay types (sorted for consistent ordering)
    assay_types = sorted(df_plot['assay_type'].unique())
    n_assay_types = len(assay_types)
    
    # Define metrics and their properties
    metrics_info = [
        ('mae_gw', 'MAE (GW)', False),
        ('mae_1obs', 'MAE (1obs)', False),
        ('mse_gw', 'MSE (GW)', True),  # True = log scale
        ('mse_1obs', 'MSE (1obs)', True),
        ('pearson_gw', 'Pearson (GW)', False),
        ('pearson_1obs', 'Pearson (1obs)', False),
        ('spearman_gw', 'Spearman (GW)', False),
        ('spearman_1obs', 'Spearman (1obs)', False),
    ]
    
    n_metrics = len(metrics_info)
    
    palette = {'imp': 'salmon', 'ups': 'skyblue'}
    
    # Create figure
    fig, axes = plt.subplots(n_assay_types, n_metrics, figsize=(24, 4 * n_assay_types))
    
    # Handle case of single row
    if n_assay_types == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, assay_type in enumerate(assay_types):
        df_assay = df_plot[df_plot['assay_type'] == assay_type]
        
        for col_idx, (metric_name, metric_label, use_log_scale) in enumerate(metrics_info):
            ax = axes[row_idx, col_idx]
            df_metric = df_assay[df_assay['metric'] == metric_name]
            
            if not df_metric.empty:
                sns.lineplot(
                    data=df_metric,
                    x='quantile',
                    y='value',
                    hue='mode',
                    palette=palette,
                    errorbar='sd',
                    ax=ax
                )
                
                # Set log scale for MSE metrics
                if use_log_scale:
                    ax.set_yscale('log')
                
                # Mark median (0.5) with vertical line
                ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
                
                # Set labels
                ax.set_xlabel('Quantile' if row_idx == n_assay_types - 1 else '', fontsize=9)
                ax.set_ylabel(metric_label if col_idx == 0 else '', fontsize=9)
                
                # Set title only for top row
                if row_idx == 0:
                    ax.set_title(metric_label, fontsize=10, fontweight='bold')
                
                # Add assay type label on the left
                if col_idx == 0:
                    ax.text(-0.3, 0.5, assay_type, transform=ax.transAxes,
                           fontsize=11, fontweight='bold', va='center', ha='right', rotation=90)
                
                # Legend only in top-right plot
                if row_idx == 0 and col_idx == n_metrics - 1:
                    ax.legend(title='Mode', loc='best', fontsize=8)
                else:
                    ax.get_legend().remove() if ax.get_legend() else None
                
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=8)
            else:
                # No data for this combination
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=10, color='gray')
                ax.set_xlabel('')
                ax.set_ylabel('')
                if row_idx == 0:
                    ax.set_title(metric_label, fontsize=10, fontweight='bold')
    
    plt.suptitle(f'Quantile Performance Analysis by Assay Type - {model_name}', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    fig.savefig(output_dir / 'quantile_performance_multipanel.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_dir / 'quantile_performance_multipanel.png'}")
    
    # Also save individual metric plots for backward compatibility
    for metric_name, metric_label, use_log_scale in metrics_info:
        df_metric = df_plot[df_plot['metric'] == metric_name]
        
        if df_metric.empty:
            continue
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.lineplot(
            data=df_metric,
            x='quantile',
            y='value',
            hue='mode',
            palette=palette,
            errorbar='sd',
            ax=ax
        )
        
        if use_log_scale:
            ax.set_yscale('log')
        
        ax.set_xlabel('Quantile', fontsize=12)
        ax.set_ylabel(metric_label, fontsize=12)
        ax.set_title(f'{metric_label} vs Quantile - {model_name}', fontsize=14)
        ax.legend(title='Mode', loc='best')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        safe_metric_name = metric_name.replace('_', '-')
        fig.savefig(output_dir / f'quantile_{safe_metric_name}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {output_dir / f'quantile_{safe_metric_name}.png'}")


def compute_calibration_data(
    preds_dir: Path,
    biosample: str,
    dist_type: str,
    dataset: str,
    metrics_obj: METRICS,
    eic_metadata_path: Optional[Path] = None,
    signal_transform: str = 'arcsinh'
) -> List[Dict]:
    """
    Compute calibration data: what quantile of the predicted distribution corresponds to observed values.
    
    Args:
        signal_transform: Signal transformation type ('arcsinh', 'log1p', 'none')
    
    Returns:
        List of dictionaries with: biosample, assay, assay_type, mode, subset, quantile_values
    """
    pred_data_dict = load_signal_predictions(preds_dir, biosample)
    
    # For EIC, determine which assays are denoised
    denoised_assays = set()
    if dataset == 'eic' and eic_metadata_path is not None:
        denoised_assays = get_eic_denoised_assays(biosample, eic_metadata_path)
    
    results = []
    
    for assay_dir_name, pred_data in pred_data_dict.items():
        # Determine mode (imp vs ups)
        if dataset == 'merged':
            if assay_dir_name.endswith("_denoised"):
                mode = "ups"
                assay_name = assay_dir_name.replace("_denoised", "")
            elif assay_dir_name.endswith("_imputed"):
                mode = "imp"
                assay_name = assay_dir_name.replace("_imputed", "")
            else:
                mode = "unknown"
                assay_name = assay_dir_name
        else:  # eic
            assay_name = assay_dir_name
            mode = "ups" if assay_name in denoised_assays else "imp"
        
        # Determine assay type
        assay_type = get_assay_type(assay_name)
        
        # Build distribution
        dist = build_distribution(
            pred_data['mu'],
            pred_data['var'],
            dist_type,
            pred_data.get('df')
        )
        
        obs = pred_data['observed_P']
        
        # Compute CDF of observed values (what quantile they correspond to)
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        quantile_values = dist.cdf(obs_tensor).numpy()
        
        # Transform back to original space for subset filtering
        if signal_transform == 'arcsinh':
            pred_orig = np.sinh(pred_data['mu'])
            obs_orig = np.sinh(obs)
        elif signal_transform == 'log1p':
            pred_orig = np.expm1(pred_data['mu'])
            obs_orig = np.expm1(obs)
        else:  # 'none'
            pred_orig = pred_data['mu']
            obs_orig = obs
        
        # Filter out NaN/Inf
        mask = np.isfinite(pred_orig) & np.isfinite(obs_orig) & np.isfinite(quantile_values)
        quantile_values_clean = quantile_values[mask]
        obs_orig_clean = obs_orig[mask]
        pred_orig_clean = pred_orig[mask]
        
        if len(quantile_values_clean) < 10:
            continue
        
        # Genome-wide
        results.append({
            'biosample': biosample,
            'assay': assay_name,
            'assay_type': assay_type,
            'mode': mode,
            'subset': 'gw',
            'quantile_values': quantile_values_clean
        })
        
        # Top 1% observed positions
        perc_99 = np.percentile(obs_orig_clean, 99)
        mask_1obs = obs_orig_clean >= perc_99
        if mask_1obs.sum() > 0:
            results.append({
                'biosample': biosample,
                'assay': assay_name,
                'assay_type': assay_type,
                'mode': mode,
                'subset': '1obs',
                'quantile_values': quantile_values_clean[mask_1obs]
            })
        
        # Promoter positions
        try:
            prom_indices = np.concatenate([
                np.arange(row['start'], min(row['end'], len(quantile_values_clean))) 
                for _, row in metrics_obj.prom_df.iterrows()
            ])
            prom_indices = prom_indices[prom_indices < len(quantile_values_clean)]
            if len(prom_indices) > 0:
                results.append({
                    'biosample': biosample,
                    'assay': assay_name,
                    'assay_type': assay_type,
                    'mode': mode,
                    'subset': 'prom',
                    'quantile_values': quantile_values_clean[prom_indices]
                })
        except Exception as e:
            pass  # Skip if promoter data not available
        
        # Gene body positions
        try:
            gene_indices = np.concatenate([
                np.arange(row['start'], min(row['end'], len(quantile_values_clean))) 
                for _, row in metrics_obj.gene_df.iterrows()
            ])
            gene_indices = gene_indices[gene_indices < len(quantile_values_clean)]
            if len(gene_indices) > 0:
                results.append({
                    'biosample': biosample,
                    'assay': assay_name,
                    'assay_type': assay_type,
                    'mode': mode,
                    'subset': 'gene',
                    'quantile_values': quantile_values_clean[gene_indices]
                })
        except Exception as e:
            pass  # Skip if gene data not available
    
    return results


def plot_calibration_histograms(df_calib: pd.DataFrame, output_dir: Path, model_name: str, n_bins: int = 20, mode_filter: Optional[str] = None):
    """
    Create multipanel calibration histogram plot.
    Rows: Assay types
    Columns: gw, 1obs, prom, gene
    Different colors for different biosamples (if multiple).
    
    Args:
        df_calib: DataFrame with calibration data
        output_dir: Output directory
        model_name: Model name for plot title
        n_bins: Number of bins for histogram
        mode_filter: If specified, filter to only 'imp' or 'ups' mode
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter by mode if specified
    if mode_filter is not None:
        df_calib = df_calib[df_calib['mode'] == mode_filter].copy()
        if df_calib.empty:
            print(f"Warning: No data for mode '{mode_filter}'")
            return
    
    # Get unique assay types
    assay_types = sorted(df_calib['assay_type'].unique())
    n_assay_types = len(assay_types)
    
    subsets = ['gw', '1obs', 'prom', 'gene']
    subset_labels = {'gw': 'Genome-wide', '1obs': 'Top 1% Obs', 'prom': 'Promoters', 'gene': 'Gene Bodies'}
    
    # Create figure
    fig, axes = plt.subplots(n_assay_types, 4, figsize=(16, 4 * n_assay_types))
    
    # Handle case of single row
    if n_assay_types == 1:
        axes = axes.reshape(1, -1)
    
    # Get unique biosamples for color mapping
    biosamples = sorted(df_calib['biosample'].unique())
    n_biosamples = len(biosamples)
    
    # Create color palette for biosamples
    if n_biosamples <= 10:
        bios_colors = sns.color_palette("tab10", n_biosamples)
    else:
        bios_colors = sns.color_palette("husl", n_biosamples)
    bios_palette = dict(zip(biosamples, bios_colors))
    
    bins = np.linspace(0, 1, n_bins + 1)
    
    for row_idx, assay_type in enumerate(assay_types):
        df_assay = df_calib[df_calib['assay_type'] == assay_type]
        
        for col_idx, subset in enumerate(subsets):
            ax = axes[row_idx, col_idx]
            df_subset = df_assay[df_assay['subset'] == subset]
            
            if not df_subset.empty:
                # Plot histogram for each biosample
                for biosample in sorted(df_subset['biosample'].unique()):
                    df_bios = df_subset[df_subset['biosample'] == biosample]
                    
                    # Aggregate quantile values across all assays for this biosample
                    all_quantiles = []
                    for _, row in df_bios.iterrows():
                        all_quantiles.extend(row['quantile_values'])
                    
                    if len(all_quantiles) > 0:
                        all_quantiles = np.array(all_quantiles)
                        
                        # Compute histogram
                        hist, _ = np.histogram(all_quantiles, bins=bins, density=True)
                        hist = hist / hist.sum()  # Normalize to sum to 1
                        
                        # Plot as line
                        bin_centers = (bins[:-1] + bins[1:]) / 2
                        ax.plot(bin_centers, hist, 
                               label=biosample if n_biosamples > 1 else None,
                               color=bios_palette[biosample],
                               alpha=0.7, linewidth=2)
                
                # Add uniform distribution reference line
                uniform_height = 1.0 / n_bins
                ax.axhline(y=uniform_height, color='black', linestyle='--', 
                          alpha=0.5, linewidth=1, label='Uniform (ideal)')
                
                # Set labels
                ax.set_xlabel('Quantile' if row_idx == n_assay_types - 1 else '', fontsize=9)
                ax.set_ylabel('Fraction' if col_idx == 0 else '', fontsize=9)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, None)
                
                # Set title only for top row
                if row_idx == 0:
                    ax.set_title(subset_labels[subset], fontsize=10, fontweight='bold')
                
                # Add assay type label on the left
                if col_idx == 0:
                    ax.text(-0.25, 0.5, assay_type, transform=ax.transAxes,
                           fontsize=11, fontweight='bold', va='center', ha='right', rotation=90)
                
                # Legend only in top-right plot
                if row_idx == 0 and col_idx == 3 and (n_biosamples > 1 or True):  # Always show legend for uniform line
                    ax.legend(loc='best', fontsize=7)
                
                ax.grid(True, alpha=0.3, axis='y')
                ax.tick_params(labelsize=8)
            else:
                # No data for this combination
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=10, color='gray')
                ax.set_xlabel('')
                ax.set_ylabel('')
                if row_idx == 0:
                    ax.set_title(subset_labels[subset], fontsize=10, fontweight='bold')
    
    # Add mode to title if filtered
    title_suffix = f" ({mode_filter.upper()})" if mode_filter else ""
    plt.suptitle(f'Calibration Histogram{title_suffix} - {model_name}', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save with mode-specific filename
    filename = f'calibration_histograms_{mode_filter}.png' if mode_filter else 'calibration_histograms.png'
    fig.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_dir / filename}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Visualize model performance across quantiles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all plots (performance + calibration)
  python eval_scripts/viz_quantile_performance.py --model-dir models/my_model/ --dataset merged

  # Generate only calibration histograms
  python eval_scripts/viz_quantile_performance.py --model-dir models/my_model/ --dataset eic --plot-type calibration
  
  # Generate only performance plots
  python eval_scripts/viz_quantile_performance.py --model-dir models/my_model/ --dataset eic --plot-type performance
        """
    )
    
    parser.add_argument('--model-dir', type=str, required=True,
                       help='Path to model directory containing preds/ subdirectory')
    parser.add_argument('--dataset', type=str, required=True, choices=['merged', 'eic'],
                       help='Dataset type (merged or eic)')
    parser.add_argument('--biosample', type=str, default=None,
                       help='Specific biosample name (default: process all biosamples)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for plots (default: model_dir/preds/quantile_plots/)')
    parser.add_argument('--num-quantiles', type=int, default=None,
                       help='[Deprecated] Number of quantiles - now uses fixed set: 1,5,10,15,...,95,99')
    parser.add_argument('--plot-type', type=str, default='all', 
                       choices=['all', 'performance', 'calibration'],
                       help='Type of plot to generate (default: all)')
    parser.add_argument('--signal-transform', type=str, default='arcsinh',
                       choices=['arcsinh', 'log1p', 'none'],
                       help='Signal transformation used during training (default: arcsinh)')
    
    args = parser.parse_args()
    
    # Setup paths
    model_dir = Path(args.model_dir)
    preds_dir = model_dir / "preds"
    
    if not preds_dir.exists():
        print(f"Error: Predictions directory not found: {preds_dir}")
        sys.exit(1)
    
    # Set output directory
    if args.output_dir is None:
        output_dir = preds_dir / "quantile_plots"
    else:
        output_dir = Path(args.output_dir)
    
    # Get model name for plot titles
    model_name = model_dir.name
    
    # Determine distribution type
    dist_type = get_dist_type(model_dir)
    print(f"Distribution type: {dist_type}")
    
    # Initialize METRICS object
    print("Initializing METRICS object...")
    metrics_obj = METRICS()
    
    # Define quantiles - use specific percentiles for faster computation
    # 1, 5, 10, 15, 20, ..., 85, 90, 95, 99
    quantiles = np.array([0.01, 0.05] + [i/100.0 for i in range(10, 100, 5)] + [0.99])
    print(f"Evaluating {len(quantiles)} quantiles: {[int(q*100) for q in quantiles]}")
    
    # Get list of biosamples
    if args.biosample is not None:
        biosamples = [args.biosample]
        print(f"Processing single biosample: {args.biosample}")
    else:
        biosamples = [d.name for d in preds_dir.iterdir() if d.is_dir()]
        print(f"Processing all biosamples: {len(biosamples)} found")
    
    # EIC metadata path
    eic_metadata_path = None
    if args.dataset == 'eic':
        script_dir = Path(__file__).parent.parent
        eic_metadata_path = script_dir / "data" / "eic_metadata.csv"
        if not eic_metadata_path.exists():
            print(f"Warning: eic_metadata.csv not found at {eic_metadata_path}")
            eic_metadata_path = None
    
    # Process biosamples based on plot type
    if args.plot_type in ['all', 'performance']:
        print("\n" + "="*60)
        print("GENERATING PERFORMANCE PLOTS")
        print("="*60)
        
        all_results = []
        
        for i, biosample in enumerate(biosamples):
            print(f"\nProcessing {i+1}/{len(biosamples)}: {biosample}")
            
            try:
                results = process_biosample(
                    preds_dir, biosample, dist_type, quantiles, 
                    args.dataset, metrics_obj, eic_metadata_path, args.signal_transform
                )
                all_results.extend(results)
                print(f"  Processed {len(results)} performance entries")
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        if not all_results:
            print("\nNo performance data to plot.")
        else:
            # Create DataFrame
            df_perf = pd.DataFrame(all_results)
            
            # Save raw data
            output_dir.mkdir(parents=True, exist_ok=True)
            csv_path = output_dir / "quantile_metrics.csv"
            df_perf.to_csv(csv_path, index=False)
            print(f"\nSaved performance data to: {csv_path}")
            
            # Print summary
            print("\nPerformance data summary:")
            print(f"  Total entries: {len(df_perf)}")
            print(f"  Biosamples: {df_perf['biosample'].nunique()}")
            print(f"  Assays: {df_perf['assay'].nunique()}")
            print(f"  Assay types: {sorted(df_perf['assay_type'].unique())}")
            print(f"  Modes: {sorted(df_perf['mode'].unique())}")
            
            # Create plots
            print("\nGenerating performance plots...")
            plot_quantile_performance(df_perf, output_dir, model_name)
    
    if args.plot_type in ['all', 'calibration']:
        print("\n" + "="*60)
        print("GENERATING CALIBRATION PLOTS")
        print("="*60)
        
        all_calib_results = []
        
        for i, biosample in enumerate(biosamples):
            print(f"\nProcessing {i+1}/{len(biosamples)}: {biosample}")
            
            try:
                calib_results = compute_calibration_data(
                    preds_dir, biosample, dist_type, 
                    args.dataset, metrics_obj, eic_metadata_path, args.signal_transform
                )
                all_calib_results.extend(calib_results)
                print(f"  Processed {len(calib_results)} calibration entries")
            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if not all_calib_results:
            print("\nNo calibration data to plot.")
        else:
            # Create DataFrame
            df_calib = pd.DataFrame(all_calib_results)
            
            # Save raw calibration data
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Print summary
            print("\nCalibration data summary:")
            print(f"  Total entries: {len(df_calib)}")
            print(f"  Biosamples: {df_calib['biosample'].nunique()}")
            print(f"  Assays: {df_calib['assay'].nunique()}")
            print(f"  Assay types: {sorted(df_calib['assay_type'].unique())}")
            print(f"  Subsets: {sorted(df_calib['subset'].unique())}")
            
            # Create calibration plots - separate files for imp and ups
            print("\nGenerating calibration histograms...")
            
            # Check which modes are available
            available_modes = df_calib['mode'].unique()
            
            if 'imp' in available_modes:
                print("  Creating calibration histogram for IMP mode...")
                plot_calibration_histograms(df_calib, output_dir, model_name, n_bins=20, mode_filter='imp')
            
            if 'ups' in available_modes:
                print("  Creating calibration histogram for UPS mode...")
                plot_calibration_histograms(df_calib, output_dir, model_name, n_bins=20, mode_filter='ups')
            
            # Also create combined plot for comparison
            print("  Creating combined calibration histogram...")
            plot_calibration_histograms(df_calib, output_dir, model_name, n_bins=20, mode_filter=None)
    
    print(f"\nDone. Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
