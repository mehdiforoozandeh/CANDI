#!/usr/bin/env python3
"""
Test Iterative Refinement (Scenario A) for CANDI.
Strategies:
1. Hard Input Replacement (Baseline): default
2. Gibbs Sampling (Re-masking): --remask-prob > 0
3. Soft Constraint (Relaxation): --alpha > 0
4. Probabilistic Sampling: --sample
5. Confidence Gated Update: --confidence-gated
"""
import argparse
import sys
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import glob
from scipy.stats import pearsonr, spearmanr
import shutil
from PIL import Image

# Add project root and post_training to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "post_training"))

# Import from _utils
try:
    from _utils import DataMasker
except ImportError:
    print("Error: Could not import DataMasker from _utils.")
    sys.exit(1)

# Import from viz script
try:
    from viz_candi_outputs import (
        load_model_from_checkpoint,
        load_eic_validation_data,
        run_inference,
        CANDIDataHandler
    )
except ImportError:
    try:
        from post_training.viz_candi_outputs import (
            load_model_from_checkpoint,
            load_eic_validation_data,
            run_inference,
            CANDIDataHandler
        )
    except ImportError:
        print("Error: Could not import viz_candi_outputs.")
        sys.exit(1)

def compute_metrics(y_true, y_pred, n_pred, upsampled_idx, imputed_idx):
    """
    Compute metrics for current iteration.
    y_true, y_pred, n_pred: [L, F] tensors/arrays
    """
    metrics = {
        'upsampled': {'pearson': [], 'spearman': [], 'min_err': []},
        'imputed': {'pearson': [], 'spearman': [], 'min_err': []}
    }
    
    # Convert to numpy
    if torch.is_tensor(y_true): y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred): y_pred = y_pred.cpu().numpy()
    if torch.is_tensor(n_pred): n_pred = n_pred.cpu().numpy()
    
    F = y_true.shape[1]
    
    for f in range(F):
        # Determine type
        if f in upsampled_idx:
            cat = 'upsampled'
        elif f in imputed_idx:
            cat = 'imputed'
        else:
            continue
            
        # Get valid ground truth indices (y_true != -1)
        valid = y_true[:, f] != -1
        if not valid.any():
            continue
            
        t = y_true[valid, f]
        p = y_pred[valid, f]
        
        # Pearson
        if len(t) > 1 and np.std(p) > 1e-6 and np.std(t) > 1e-6:
            r, _ = pearsonr(t, p)
            metrics[cat]['pearson'].append(r)
        
        # Spearman
        if len(t) > 1 and np.std(p) > 1e-6:
            rho, _ = spearmanr(t, p)
            metrics[cat]['spearman'].append(rho)
            
        # Min % Error = sqrt(1/n)
        # Average over the track
        n_vals = n_pred[:, f]
        n_vals = np.maximum(n_vals, 1e-6)
        min_err = np.sqrt(1.0 / n_vals)
        metrics[cat]['min_err'].append(np.mean(min_err))
        
    # Aggregate
    res = {}
    for cat in ['upsampled', 'imputed']:
        res[f'{cat}_pearson'] = np.nanmean(metrics[cat]['pearson']) if metrics[cat]['pearson'] else 0
        res[f'{cat}_spearman'] = np.nanmean(metrics[cat]['spearman']) if metrics[cat]['spearman'] else 0
        res[f'{cat}_min_err'] = np.nanmean(metrics[cat]['min_err']) if metrics[cat]['min_err'] else 0
        
    return res

def create_frame(data, mu_pred, n_pred, history_metrics, iter_idx, biosample, chrom, start_loci, resolution, save_path):
    """
    Create a single frame with metrics plots and tracks.
    """
    Y_T = data['Y_T'][0].float().cpu()
    Y_V = data['Y_V'][0].float().cpu()
    expnames = data['expnames']
    upsampled_indices = set(data['available_T_indices'])
    imputed_indices = set(data['available_V_indices']) - upsampled_indices
    
    L, F = Y_T.shape
    
    # Setup Figure with GridSpec
    track_h = 0.5
    metrics_h = 4
    spacer_h = 1.0
    total_h = metrics_h + spacer_h + F * track_h
    
    fig = plt.figure(figsize=(20, total_h))
    gs = GridSpec(F + 2, 3, figure=fig, height_ratios=[metrics_h, spacer_h] + [track_h]*F)
    
    # --- Metrics Plots (Top Row) ---
    ax_pearson = fig.add_subplot(gs[0, 0])
    ax_spearman = fig.add_subplot(gs[0, 1])
    ax_error = fig.add_subplot(gs[0, 2])
    
    iterations = range(len(history_metrics))
    
    ups_p = [m['upsampled_pearson'] for m in history_metrics]
    imp_p = [m['imputed_pearson'] for m in history_metrics]
    ups_s = [m['upsampled_spearman'] for m in history_metrics]
    imp_s = [m['imputed_spearman'] for m in history_metrics]
    ups_e = [m['upsampled_min_err'] for m in history_metrics]
    imp_e = [m['imputed_min_err'] for m in history_metrics]
    
    # Plot Pearson
    ax_pearson.plot(iterations, ups_p, 'g-o', label='Observed')
    ax_pearson.plot(iterations, imp_p, 'b-o', label='Imputed')
    ax_pearson.set_title('Pearson Correlation')
    ax_pearson.set_xlabel('Iteration')
    ax_pearson.set_xticks(iterations)
    ax_pearson.grid(True, alpha=0.3)
    ax_pearson.legend()
    ax_pearson.axvline(iter_idx, color='r', linestyle='--', alpha=0.5)
    
    # Plot Spearman
    ax_spearman.plot(iterations, ups_s, 'g-o', label='Observed')
    ax_spearman.plot(iterations, imp_s, 'b-o', label='Imputed')
    ax_spearman.set_title('Spearman Correlation')
    ax_spearman.set_xticks(iterations)
    ax_spearman.grid(True, alpha=0.3)
    ax_spearman.axvline(iter_idx, color='r', linestyle='--', alpha=0.5)
    
    # Plot Min Error
    ax_error.plot(iterations, ups_e, 'g-o', label='Observed')
    ax_error.plot(iterations, imp_e, 'b-o', label='Imputed')
    ax_error.set_title('Mean Min % Error')
    ax_error.set_ylabel('Uncertainty (sqrt(1/n))')
    ax_error.set_xticks(iterations)
    ax_error.grid(True, alpha=0.3)
    ax_error.axvline(iter_idx, color='r', linestyle='--', alpha=0.5)
    
    # --- Tracks ---
    x_coords = np.arange(L)
    xtick_locs = np.linspace(0, L-1, 10)
    xtick_labels = [f"{int(start_loci + x * resolution):,}" for x in xtick_locs]
    
    for f in range(F):
        row = f + 2
        assay_name = expnames[f] if f < len(expnames) else f"Assay_{f}"
        
        ax_gt = fig.add_subplot(gs[row, 0])
        ax_mu = fig.add_subplot(gs[row, 1])
        ax_n = fig.add_subplot(gs[row, 2])
        
        # Plot GT
        gt_max = 0
        if f in upsampled_indices:
            vals = Y_T[:, f].numpy()
            mask = vals != -1
            if mask.any():
                ax_gt.fill_between(x_coords, 0, np.where(mask, vals, 0), step='post', color='green', alpha=1.0)
                ax_gt.text(0.02, 0.8, "Observed", transform=ax_gt.transAxes, color='green', fontsize=8)
                gt_max = vals[mask].max()
        elif f in imputed_indices:
            vals = Y_V[:, f].numpy()
            mask = vals != -1
            if mask.any():
                ax_gt.fill_between(x_coords, 0, np.where(mask, vals, 0), step='post', color='blue', alpha=1.0)
                ax_gt.text(0.02, 0.8, "Target", transform=ax_gt.transAxes, color='blue', fontsize=8)
                gt_max = vals[mask].max()
        
        if gt_max > 0: ax_gt.set_ylim(0, gt_max*1.1)
        ax_gt.set_ylabel(assay_name, rotation=0, ha='right', fontsize=9)
        ax_gt.spines['top'].set_visible(False)
        ax_gt.spines['right'].set_visible(False)
        ax_gt.set_xticks([])
        
        # Plot Mu
        mu_vals = mu_pred[:, f].numpy()
        ax_mu.fill_between(x_coords, 0, mu_vals, step='post', color='red', alpha=0.8)
        mu_max = mu_vals.max()
        if mu_max > 0: ax_mu.set_ylim(0, mu_max*1.1)
        ax_mu.spines['top'].set_visible(False)
        ax_mu.spines['right'].set_visible(False)
        ax_mu.set_xticks([])
        
        # Plot N
        n_vals = n_pred[:, f].numpy()
        ax_n.fill_between(x_coords, 0, n_vals, step='post', color='purple', alpha=0.8)
        n_max = n_vals.max()
        if n_max > 0: ax_n.set_ylim(0, n_max*1.1)
        ax_n.spines['top'].set_visible(False)
        ax_n.spines['right'].set_visible(False)
        ax_n.set_xticks([])
        
        # Titles
        if f == 0:
            ax_gt.set_title('Ground Truth', fontsize=12, fontweight='bold')
            ax_mu.set_title('Predicted Mean', fontsize=12, fontweight='bold')
            ax_n.set_title('Predicted Dispersion', fontsize=12, fontweight='bold')

        if f == F - 1:
            for ax in [ax_gt, ax_mu, ax_n]:
                ax.set_xticks(xtick_locs)
                ax.set_xticklabels(xtick_labels, rotation=45, ha='right')
                ax.set_xlabel(f'Genomic Position ({chrom})')
    
    plt.suptitle(f'Iterative Refinement - Iteration {iter_idx}', fontsize=16, y=0.995)
    # plt.tight_layout() # Conflict with manual adjustments
    plt.subplots_adjust(top=0.88, hspace=0.5)
    
    plt.savefig(save_path, dpi=100)
    plt.close(fig)

def iterative_loop(model, data, device, iterations=10, unmask_metadata=True, output_dir=None, args=None):
    current_X = data['X'].clone()
    current_mX = data['mX'].clone()
    signal_dim = data['Y_T'].shape[-1] 
    
    # Store original signal and move to correct device immediately
    original_X_signal = data['X'][:, :, :signal_dim].clone().to(device)
    missing_mask = (original_X_signal == -1)
    
    # Move current buffers to device
    current_X = current_X.to(device)
    current_mX = current_mX.to(device)
    
    history_metrics = []
    best_n = None # Initialize for confidence gating
    
    # Determine remask probability
    if args.adaptive_remask:
        n_obs = len(data['available_T_indices'])
        current_remask_prob = n_obs / 35.0
    else:
        current_remask_prob = args.remask_prob
    
    print(f"Starting loop for {iterations} iterations...")
    if current_remask_prob > 0:
        print(f"Strategy: Gibbs Sampling (Re-masking {current_remask_prob*100:.1f}%)")
    if args.sample:
        print("Strategy: Probabilistic Sampling (from NB)")
    if args.alpha > 0:
        print(f"Strategy: Soft Constraint (alpha={args.alpha})")
    if args.confidence_gated:
        print("Strategy: Confidence Gated Update (update missing only if n increases)")

    for t in range(iterations + 1):
        print(f"Iter {t}...", end=" ", flush=True)
        
        mu, n = run_inference(model, current_X, current_mX, data['mY'], data['seq'], device)
        mu_cpu = mu.cpu()
        n_cpu = n.cpu()
        
        # Metrics & Visualization
        Y_merged = data['Y_T'].clone()
        imputed_idx = set(data['available_V_indices']) - set(data['available_T_indices'])
        for f in imputed_idx:
            Y_merged[:, :, f] = data['Y_V'][:, :, f]
            
        mets = compute_metrics(Y_merged[0], mu_cpu[0], n_cpu[0], 
                               data['available_T_indices'], imputed_idx)
        history_metrics.append(mets)
        print(f"P: {mets.get('imputed_pearson',0):.3f}")
        
        frame_path = output_dir / f"frame_{t:03d}.png"
        create_frame(data, mu_cpu[0], n_cpu[0], history_metrics, t, 
                     args.biosample, args.chrom, args.start_loci, args.resolution, frame_path)
        
        if t < iterations:
            # === PREPARE INPUT FOR NEXT ITERATION ===
            
            # 1. Determine Prediction Values (Mean vs Sample)
            if args.sample:
                # Recalculate probs from mu and n
                n_t = n.to(current_X.device)
                mu_t = mu.to(current_X.device)
                # PyTorch probs = mu / (n + mu) based on _utils.py check
                probs = mu_t / (n_t + mu_t + 1e-6)
                nb_dist = torch.distributions.NegativeBinomial(total_count=n_t, probs=probs)
                pred_vals = nb_dist.sample()
            else:
                pred_vals = mu.to(current_X.device)
            
            # 2. Update Input Buffer
            x_signal = current_X[:, :, :signal_dim].clone()
            
            # B. Update OBSERVED Values (Soft Constraint)
            obs_mask = ~missing_mask
            
            # Calculate Target Values
            if args.alpha > 0:
                target_obs = (1.0 - args.alpha) * original_X_signal[obs_mask] + args.alpha * pred_vals[obs_mask]
            else:
                target_obs = original_X_signal[obs_mask]
                
            # Apply Update (Gated vs Ungated)
            if args.confidence_gated and t > 0:
                n_tensor = n.to(device)
                improved = n_tensor > best_n
                
                improved_obs = improved[obs_mask]
                current_obs = x_signal[obs_mask]
                is_hole_obs = (current_obs == -1)
                
                do_update = is_hole_obs | improved_obs
                
                final_obs = torch.where(do_update, target_obs, current_obs)
                x_signal[obs_mask] = final_obs
                
                # Update best_n for observed regions
                best_n_obs = best_n[obs_mask]
                n_new_obs = n_tensor[obs_mask]
                best_n[obs_mask] = torch.where(do_update, n_new_obs, best_n_obs)
            else:
                x_signal[obs_mask] = target_obs
            
            # C. Update MISSING Values (Imputation Targets)
            if t == 0:
                # First pass: Always fill all original missing values
                x_signal[missing_mask] = pred_vals[missing_mask]
                
                # Initialize best_n for gating
                if args.confidence_gated:
                    best_n = n.to(device).clone()
                    
                # Update Metadata for future passes (Switch to mY)
                if unmask_metadata:
                    true_meta = data['mY'].to(current_mX.device)
                    current_mX[:, :, :signal_dim] = true_meta.clone()
                    
            else:
                # Subsequent passes: Refine missing values
                if args.confidence_gated:
                    n_tensor = n.to(device)
                    improved = n_tensor > best_n
                    
                    # 1. Always fill explicit holes (e.g. from Gibbs) in missing regions
                    holes_mask = missing_mask & (x_signal == -1)
                    x_signal[holes_mask] = pred_vals[holes_mask]
                    
                    # 2. Refine existing values only if confidence improves
                    existing_mask = missing_mask & (x_signal != -1)
                    refine_mask = existing_mask & improved
                    x_signal[refine_mask] = pred_vals[refine_mask]
                    
                    # Update best_n for locations we updated
                    updated_locs = holes_mask | refine_mask
                    best_n[updated_locs] = n_tensor[updated_locs]
                else:
                    # Ungated: Always update missing values with latest prediction
                    x_signal[missing_mask] = pred_vals[missing_mask]
            
            # D. Gibbs Re-masking (Apply at end for NEXT iteration)
            if current_remask_prob > 0:
                av_signal = data['avX'][:, :signal_dim].to(current_X.device)
                meta_signal = current_mX[:, :, :signal_dim].to(current_X.device)
                
                masker = DataMasker(mask_value=-1, chunk_size=40, mask_fraction=current_remask_prob,
                                    p_full_loci=0.0, p_full_assay=1.0, p_chunks=0.0)
                
                x_masked, _, _ = masker.apply_mask(x_signal, meta_signal, av_signal)
                x_signal = x_masked
            
            current_X[:, :, :signal_dim] = x_signal
        
    return history_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--biosample', required=True)
    parser.add_argument('--start-loci', type=int, default=1000000)
    parser.add_argument('--chrom', default='chr21')
    parser.add_argument('--resolution', type=int, default=25)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--device', default=None)
    
    # New Args
    parser.add_argument('--remask-prob', type=float, default=0.0, help='Probability to re-mask inputs at each step (Gibbs Sampling)')
    parser.add_argument('--adaptive-remask', action='store_true', help='Set remask-prob based on n_observed / 35')
    parser.add_argument('--alpha', type=float, default=0.0, help='Soft constraint weight (0.0 = Hard, 1.0 = Pure Prediction)')
    parser.add_argument('--sample', action='store_true', help='Use sampling from NB distribution instead of mean')
    parser.add_argument('--confidence-gated', action='store_true', help='Update input only if confidence (n) increases')
    
    args = parser.parse_args()
    
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")
        
    base_path = os.environ.get('CANDI_EIC_DATA_PATH', '/project/6014832/mforooz/DATA_CANDI_EIC')
    data_handler = CANDIDataHandler(base_path=base_path, resolution=args.resolution, dataset_type="eic", DNA=True)
    
    model, config = load_model_from_checkpoint(args.model_path, device)
    context_length = config.get('context-length', 1200)
    
    end_loci = args.start_loci + context_length * args.resolution
    locus = [args.chrom, args.start_loci, end_loci]
    data = load_eic_validation_data(data_handler, args.biosample, locus, context_length, args.resolution)
    
    model_path = Path(args.model_path)
    model_dir = model_path.parent if model_path.is_file() else model_path
    viz_dir = model_dir / "iterative_frames"
    if viz_dir.exists(): shutil.rmtree(viz_dir)
    viz_dir.mkdir(parents=True)
    
    iterative_loop(model, data, device, iterations=args.iterations, unmask_metadata=True, output_dir=viz_dir, args=args)
    
    print("Creating GIF...")
    frames = sorted(list(viz_dir.glob("frame_*.png")))
    try:
        viz_final_dir = model_dir / "viz"
        viz_final_dir.mkdir(exist_ok=True)
        
        # Suffix based on method
        method = "iterative"
        
        remask_p = args.remask_prob
        if args.adaptive_remask:
             remask_p = len(data['available_T_indices']) / 35.0
        
        if remask_p > 0: method += f"_gibbs{remask_p:.2f}"
        if args.sample: method += "_sample"
        if args.alpha > 0: method += f"_soft{args.alpha}"
        if args.confidence_gated: method += "_gated"
        
        gif_path = viz_final_dir / f"{args.biosample}_{args.chrom}_{args.start_loci}_{method}.gif"
        
        # Use PIL for better duration control
        images = []
        for frame_file in frames:
            img = Image.open(frame_file)
            images.append(img.copy())
            img.close()
            
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=500,
            loop=0
        )
        print(f"Saved GIF to {gif_path} (0.5s per frame)")
    except Exception as e:
        print(f"GIF creation failed: {e}. Frames are in {viz_dir}")

if __name__ == '__main__':
    main()
