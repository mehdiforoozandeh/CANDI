#!/usr/bin/env python3
"""
Compute Metrics from Saved Predictions

This script loads saved predictions from NPZ files and computes comprehensive metrics
for both merged and EIC datasets.

Author: CANDI Team
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
import pandas as pd
import numpy as np
import torch

# Import from project
sys.path.insert(0, str(Path(__file__).parent.parent))
from _utils import METRICS, Gaussian


def load_predictions_from_npz(preds_dir: Path, biosample: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load predictions from NPZ files for a given biosample.
    
    Args:
        preds_dir: Path to preds directory (model_dir/preds/)
        biosample: Name of biosample
        
    Returns:
        Dictionary: {assay_dir_name: {mu, var, peak_scores, observed_P, observed_peak, z}}
    """
    bios_dir = preds_dir / biosample
    if not bios_dir.exists():
        raise FileNotFoundError(f"Biosample directory not found: {bios_dir}")
    
    results = {}
    
    # Load Z from biosample level (shared across all assays)
    z_path = bios_dir / "z.npz"
    z_data_biosample = None
    if z_path.exists():
        try:
            z_data_biosample = np.load(z_path)['arr_0']
        except Exception as e:
            print(f"Warning: Failed to load biosample-level Z for {biosample}: {e}")
    
    # Get list of assay directories
    assay_dirs = [d for d in bios_dir.iterdir() if d.is_dir()]
    
    for assay_dir in assay_dirs:
        assay_dir_name = assay_dir.name
        
        try:
            # Load NPZ files
            mu_data = np.load(assay_dir / "mu.npz")
            var_data = np.load(assay_dir / "var.npz")
            peak_data = np.load(assay_dir / "peak_scores.npz")
            obs_p_data = np.load(assay_dir / "observed_P.npz")
            obs_peak_data = np.load(assay_dir / "observed_peak.npz")
            
            # Try loading z from assay dir for backward compatibility, fall back to biosample level
            z_data = z_data_biosample
            assay_z_path = assay_dir / "z.npz"
            if assay_z_path.exists():
                try:
                    z_data = np.load(assay_z_path)['arr_0']
                except:
                    pass  # Use biosample-level z
            
            # Extract arrays (NPZ files contain arrays with default key 'arr_0')
            results[assay_dir_name] = {
                "mu": mu_data['arr_0'],
                "var": var_data['arr_0'],
                "peak_scores": peak_data['arr_0'],
                "observed_P": obs_p_data['arr_0'],
                "observed_peak": obs_peak_data['arr_0'],
                "z": z_data
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
        - available_assays_count: Number of assays available in T_biosample
        - denoised_assays_set: Set of assay names that exist in T_biosample (these are denoised)
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
    
    # Debug: Print column names to verify
    if 'biosample_name' not in metadata_df.columns:
        print(f"Error: Column 'biosample_name' not found. Available columns: {metadata_df.columns.tolist()}")
        return 0, set()
    
    # Get assays available for T_biosample
    # Note: CSV uses 'biosample_name' column, not 'bios_name'
    T_rows = metadata_df[metadata_df['biosample_name'] == T_biosample]
    
    if T_rows.empty:
        # Debug: Show available biosample names that start with T_
        available_T_biosamples = metadata_df[metadata_df['biosample_name'].str.startswith('T_')]['biosample_name'].unique()
        print(f"Warning: No metadata found for {T_biosample}")
        print(f"  Looking for biosample: {T_biosample} (from input: {biosample})")
        print(f"  Sample T_ biosamples in CSV: {list(available_T_biosamples[:5])}")
        return 0, set()
    
    # Get unique assay names for T_biosample
    # Note: CSV uses 'assay_name' column, not 'experiment_target'
    if 'assay_name' not in metadata_df.columns:
        print(f"Error: Column 'assay_name' not found. Available columns: {metadata_df.columns.tolist()}")
        return 0, set()
    
    denoised_assays = set(T_rows['assay_name'].unique())
    available_assays_count = len(denoised_assays)
    
    return available_assays_count, denoised_assays


def compute_metrics_for_assay(
    assay_name: str,
    pred_data: Dict[str, np.ndarray],
    comparison_type: str,
    available_assays: int,
    metrics_obj: METRICS,
    biosample: str
) -> Dict[str, Any]:
    """
    Compute comprehensive metrics for a single assay.
    
    Args:
        assay_name: Name of assay
        pred_data: Dictionary with mu, var, peak_scores, observed_P, observed_peak, z
        comparison_type: "imputed" or "denoised"
        available_assays: Number of available assays
        metrics_obj: METRICS object for computing metrics
        biosample: Name of biosample
        
    Returns:
        Dictionary with all metrics plus metadata
    """
    # Extract data
    mu = pred_data['mu']
    var = pred_data['var']
    peak_scores = pred_data['peak_scores']
    observed_P = pred_data['observed_P']
    observed_peak = pred_data['observed_peak']
    
    # Create Gaussian distribution
    pval_dist = Gaussian(torch.tensor(mu), torch.tensor(var))
    
    # Compute 95% CI
    lower_95, upper_95 = pval_dist.interval(confidence=0.95)
    lower_95 = lower_95.numpy()
    upper_95 = upper_95.numpy()
    
    # Define safe_metric wrapper
    def safe_metric(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            print(f"Error calculating metric {fn.__name__} for {biosample}/{assay_name}: {e}")
            return np.nan
    
    # Initialize results dictionary
    metrics_dict = {
        'bios': biosample,
        'assay': assay_name,
        'comparison': comparison_type,
        'available_assays': available_assays
    }
    
    # Compute MSE and Pearson metrics TWICE - with and without sinh transform
    
    # WITHOUT sinh (arcsinh space)
    pred_pval_arcsinh = mu
    P_target_arcsinh = observed_P
    
    metrics_dict['P_MSE-GW_arcsinh'] = safe_metric(metrics_obj.mse, P_target_arcsinh, pred_pval_arcsinh)
    metrics_dict['P_Pearson-GW_arcsinh'] = safe_metric(metrics_obj.pearson, P_target_arcsinh, pred_pval_arcsinh)
    metrics_dict['P_MSE-gene_arcsinh'] = safe_metric(metrics_obj.mse_gene, P_target_arcsinh, pred_pval_arcsinh)
    metrics_dict['P_Pearson_gene_arcsinh'] = safe_metric(metrics_obj.pearson_gene, P_target_arcsinh, pred_pval_arcsinh)
    metrics_dict['P_MSE-prom_arcsinh'] = safe_metric(metrics_obj.mse_prom, P_target_arcsinh, pred_pval_arcsinh)
    metrics_dict['P_Pearson_prom_arcsinh'] = safe_metric(metrics_obj.pearson_prom, P_target_arcsinh, pred_pval_arcsinh)
    metrics_dict['P_MSE-1obs_arcsinh'] = safe_metric(metrics_obj.mse1obs, P_target_arcsinh, pred_pval_arcsinh)
    metrics_dict['P_Pearson_1obs_arcsinh'] = safe_metric(metrics_obj.pearson1_obs, P_target_arcsinh, pred_pval_arcsinh)
    metrics_dict['P_MSE-1imp_arcsinh'] = safe_metric(metrics_obj.mse1imp, P_target_arcsinh, pred_pval_arcsinh)
    metrics_dict['P_Pearson_1imp_arcsinh'] = safe_metric(metrics_obj.pearson1_imp, P_target_arcsinh, pred_pval_arcsinh)
    
    # WITH sinh (original space)
    pred_pval = np.sinh(mu)
    P_target = np.sinh(observed_P)
    
    metrics_dict['P_MSE-GW'] = safe_metric(metrics_obj.mse, P_target, pred_pval)
    metrics_dict['P_Pearson-GW'] = safe_metric(metrics_obj.pearson, P_target, pred_pval)
    metrics_dict['P_MSE-gene'] = safe_metric(metrics_obj.mse_gene, P_target, pred_pval)
    metrics_dict['P_Pearson_gene'] = safe_metric(metrics_obj.pearson_gene, P_target, pred_pval)
    metrics_dict['P_MSE-prom'] = safe_metric(metrics_obj.mse_prom, P_target, pred_pval)
    metrics_dict['P_Pearson_prom'] = safe_metric(metrics_obj.pearson_prom, P_target, pred_pval)
    metrics_dict['P_MSE-1obs'] = safe_metric(metrics_obj.mse1obs, P_target, pred_pval)
    metrics_dict['P_Pearson_1obs'] = safe_metric(metrics_obj.pearson1_obs, P_target, pred_pval)
    metrics_dict['P_MSE-1imp'] = safe_metric(metrics_obj.mse1imp, P_target, pred_pval)
    metrics_dict['P_Pearson_1imp'] = safe_metric(metrics_obj.pearson1_imp, P_target, pred_pval)
    
    # Other metrics (computed once in original space with sinh)
    metrics_dict['P_Spearman-GW'] = safe_metric(metrics_obj.spearman, P_target, pred_pval)
    metrics_dict['P_Spearman_gene'] = safe_metric(metrics_obj.spearman_gene, P_target, pred_pval)
    metrics_dict['P_Spearman_prom'] = safe_metric(metrics_obj.spearman_prom, P_target, pred_pval)
    metrics_dict['P_Spearman_1obs'] = safe_metric(metrics_obj.spearman1_obs, P_target, pred_pval)
    metrics_dict['P_Spearman_1imp'] = safe_metric(metrics_obj.spearman1_imp, P_target, pred_pval)
    
    # Peak AUCROC metrics
    metrics_dict['Peak_AUCROC-GW'] = safe_metric(metrics_obj.aucroc, observed_peak, peak_scores)
    metrics_dict['Peak_AUCROC-gene'] = safe_metric(metrics_obj.aucroc_gene, observed_peak, peak_scores)
    metrics_dict['Peak_AUCROC-prom'] = safe_metric(metrics_obj.aucroc_prom, observed_peak, peak_scores)
    
    # Concordance index metrics (use raw mu, sigma from Gaussian, no sinh)
    sigma = pval_dist.std().numpy()
    metrics_dict['P_Cidx_GW'] = safe_metric(metrics_obj.c_index_gauss, mu, sigma, observed_P, num_pairs=5000)
    metrics_dict['P_Cidx_prom'] = safe_metric(metrics_obj.c_index_gauss_prom, mu, sigma, observed_P, num_pairs=5000)
    metrics_dict['P_Cidx_gene'] = safe_metric(metrics_obj.c_index_gauss_gene, mu, sigma, observed_P, num_pairs=5000)
    
    # 95% CI coverage metrics (sinh transform lower_95, upper_95 and observed_P)
    lower_95_sinh = np.sinh(lower_95)
    upper_95_sinh = np.sinh(upper_95)
    P_target_sinh = np.sinh(observed_P)
    
    metrics_dict['P_95CI_coverage_GW'] = safe_metric(metrics_obj.coverage_95ci, P_target_sinh, lower_95_sinh, upper_95_sinh)
    metrics_dict['P_95CI_coverage_gene'] = safe_metric(metrics_obj.coverage_95ci_gene, P_target_sinh, lower_95_sinh, upper_95_sinh)
    metrics_dict['P_95CI_coverage_prom'] = safe_metric(metrics_obj.coverage_95ci_prom, P_target_sinh, lower_95_sinh, upper_95_sinh)
    metrics_dict['P_95CI_coverage_1obs'] = safe_metric(metrics_obj.coverage_95ci_1obs, P_target_sinh, lower_95_sinh, upper_95_sinh)
    
    return metrics_dict


def process_merged_biosample(
    model_dir: Path,
    biosample: str,
    metrics_obj: METRICS
) -> List[Dict[str, Any]]:
    """
    Process a biosample from merged dataset.
    
    Args:
        model_dir: Path to model directory
        biosample: Name of biosample
        metrics_obj: METRICS object
        
    Returns:
        List of metric dictionaries
    """
    preds_dir = model_dir / "preds"
    
    # Load predictions
    try:
        pred_data_dict = load_predictions_from_npz(preds_dir, biosample)
    except Exception as e:
        print(f"Error loading predictions for {biosample}: {e}")
        return []
    
    results = []
    
    # Count available assays (unique base assay names)
    base_assay_names = set()
    for assay_dir_name in pred_data_dict.keys():
        if assay_dir_name.endswith("_denoised"):
            base_name = assay_dir_name.replace("_denoised", "")
        elif assay_dir_name.endswith("_imputed"):
            base_name = assay_dir_name.replace("_imputed", "")
        else:
            base_name = assay_dir_name
        base_assay_names.add(base_name)
    
    available_assays = len(base_assay_names)
    
    # Process each assay
    for assay_dir_name, pred_data in pred_data_dict.items():
        # Determine comparison type and base assay name
        if assay_dir_name.endswith("_denoised"):
            comparison_type = "denoised"
            base_assay_name = assay_dir_name.replace("_denoised", "")
        elif assay_dir_name.endswith("_imputed"):
            comparison_type = "imputed"
            base_assay_name = assay_dir_name.replace("_imputed", "")
        else:
            # Fallback for backward compatibility
            comparison_type = "unknown"
            base_assay_name = assay_dir_name
        
        print(f"  Computing metrics for {base_assay_name} ({comparison_type})...")
        
        metrics = compute_metrics_for_assay(
            assay_name=base_assay_name,
            pred_data=pred_data,
            comparison_type=comparison_type,
            available_assays=available_assays,
            metrics_obj=metrics_obj,
            biosample=biosample
        )
        
        results.append(metrics)
    
    return results


def process_eic_biosample(
    model_dir: Path,
    biosample: str,
    metrics_obj: METRICS,
    eic_metadata_path: str
) -> List[Dict[str, Any]]:
    """
    Process a biosample from EIC dataset.
    
    Args:
        model_dir: Path to model directory
        biosample: Name of biosample
        metrics_obj: METRICS object
        eic_metadata_path: Path to eic_metadata.csv
        
    Returns:
        List of metric dictionaries
    """
    preds_dir = model_dir / "preds"
    
    # Get EIC assay info
    try:
        available_assays, denoised_assays = get_eic_assay_info(biosample, eic_metadata_path)
    except Exception as e:
        print(f"Error getting EIC assay info for {biosample}: {e}")
        return []
    
    print(f"  Available assays in T_: {available_assays}")
    print(f"  Denoised assays: {len(denoised_assays)}")
    
    # Load predictions
    try:
        pred_data_dict = load_predictions_from_npz(preds_dir, biosample)
        print(f"  Loaded {len(pred_data_dict)} assay predictions")
    except Exception as e:
        print(f"Error loading predictions for {biosample}: {e}")
        return []
    
    if not pred_data_dict:
        print(f"  Warning: No predictions found for {biosample}")
        return []
    
    results = []
    
    # Process each assay
    for assay_dir_name, pred_data in pred_data_dict.items():
        # Assay name is the directory name (no suffix for EIC)
        assay_name = assay_dir_name
        
        # Determine if denoised or imputed
        if assay_name in denoised_assays:
            comparison_type = "denoised"
        else:
            comparison_type = "imputed"
        
        print(f"  Computing metrics for {assay_name} ({comparison_type})...")
        
        metrics = compute_metrics_for_assay(
            assay_name=assay_name,
            pred_data=pred_data,
            comparison_type=comparison_type,
            available_assays=available_assays,
            metrics_obj=metrics_obj,
            biosample=biosample
        )
        
        results.append(metrics)
    
    return results


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Compute metrics from saved predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # For merged dataset
  python eval_scripts/compute_metrics.py --model-dir models/my_model/ --dataset merged

  # For EIC dataset
  python eval_scripts/compute_metrics.py --model-dir models/my_eic_model/ --dataset eic

  # For specific biosample
  python eval_scripts/compute_metrics.py --model-dir models/my_model/ --dataset merged --biosample GM12878
        """
    )
    
    # Required arguments
    parser.add_argument('--model-dir', type=str, required=True,
                       help='Path to model directory containing preds/ subdirectory')
    parser.add_argument('--dataset', type=str, required=True, choices=['merged', 'eic'],
                       help='Dataset type (merged or eic)')
    
    # Optional arguments
    parser.add_argument('--biosample', type=str, default='all',
                       help='Specific biosample name or "all" (default: all)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV path (default: model_dir/preds/metrics.csv)')
    
    args = parser.parse_args()
    
    # Setup paths
    model_dir = Path(args.model_dir)
    preds_dir = model_dir / "preds"
    
    if not preds_dir.exists():
        print(f"Error: Predictions directory not found: {preds_dir}")
        sys.exit(1)
    
    # Set output path
    if args.output is None:
        output_path = preds_dir / "metrics.csv"
    else:
        output_path = Path(args.output)
    
    # Initialize METRICS object
    print("Initializing METRICS object...")
    metrics_obj = METRICS()
    
    # Get list of biosamples
    if args.biosample.lower() == 'all':
        biosamples = [d.name for d in preds_dir.iterdir() if d.is_dir()]
        print(f"Found {len(biosamples)} biosamples: {biosamples}")
    else:
        biosamples = [args.biosample]
        print(f"Processing single biosample: {args.biosample}")
    
    # Process each biosample
    all_results = []
    
    for i, biosample in enumerate(biosamples):
        print(f"\n{'='*60}")
        print(f"Processing biosample {i+1}/{len(biosamples)}: {biosample}")
        print(f"{'='*60}")
        
        try:
            if args.dataset == 'merged':
                results = process_merged_biosample(model_dir, biosample, metrics_obj)
            else:  # eic
                # Look for eic_metadata.csv in the project's data/ directory
                script_dir = Path(__file__).parent.parent
                eic_metadata_path = script_dir / "data" / "eic_metadata.csv"
                
                if not eic_metadata_path.exists():
                    print(f"Error: eic_metadata.csv not found at {eic_metadata_path}")
                    continue
                
                results = process_eic_biosample(model_dir, biosample, metrics_obj, str(eic_metadata_path))
            
            all_results.extend(results)
            print(f"✅ Completed {biosample}: {len(results)} metrics computed")
            
        except Exception as e:
            print(f"❌ Error processing {biosample}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results to CSV
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(output_path, index=False)
        print(f"\n🎉 Metrics saved to {output_path}")
        print(f"Total metrics computed: {len(all_results)}")
        
        # Print summary
        print("\nSummary:")
        print(f"  Unique biosamples: {df['bios'].nunique()}")
        print(f"  Unique assays: {df['assay'].nunique()}")
        print(f"  Imputed predictions: {len(df[df['comparison'] == 'imputed'])}")
        print(f"  Denoised predictions: {len(df[df['comparison'] == 'denoised'])}")
    else:
        print("\n⚠️ No metrics computed")
        sys.exit(1)


if __name__ == "__main__":
    main()

