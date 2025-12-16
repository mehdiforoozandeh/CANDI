#!/usr/bin/env python3
"""
Visualize CANDI model outputs for EIC validation samples.

This script loads a pretrained CANDI model and visualizes:
1. Ground truth (T_* in green for upsampled, V_* in blue for imputation targets)
2. Predicted mu (mean) of Negative Binomial distribution (red)
3. Predicted n (dispersion parameter) of Negative Binomial distribution (purple)

Following the data loading pattern from EIC_VALIDATION_MONITOR in model.py.
"""

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend, faster for saving files
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# Add project root to path to allow imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data import CANDIDataHandler
from model import CANDI

# Token dictionary (same as in EIC_VALIDATION_MONITOR)
TOKEN_DICT = {"missing_mask": -1, "cloze_mask": -2, "pad": -3}


def load_model_from_checkpoint(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Load CANDI model from checkpoint and config.
    
    Args:
        model_path: Path to model checkpoint (.pt file) or model directory
        device: Device to load model on
    
    Returns:
        model: Loaded CANDI model in eval mode
        config: Model configuration dictionary
    """
    model_path = Path(model_path)
    
    # If it's a directory, find the checkpoint and config
    if model_path.is_dir():
        model_dir = model_path
        # Find checkpoint
        checkpoint_files = list(model_dir.glob("*.pt"))
        if not checkpoint_files:
            checkpoints_dir = model_dir / "checkpoints"
            if checkpoints_dir.exists():
                checkpoint_files = list(checkpoints_dir.glob("*.pt"))
            else:
                raise FileNotFoundError(f"No .pt checkpoint file found in {model_dir}")
        checkpoint_path = checkpoint_files[0]
        
        # Find config
        config_files = list(model_dir.glob("*_config.json"))
        if not config_files:
            raise FileNotFoundError(f"No config JSON file found in {model_dir}")
        config_path = config_files[0]
    else:
        # Assume it's a checkpoint file, find config in same directory
        checkpoint_path = model_path
        model_dir = checkpoint_path.parent
        config_files = list(model_dir.glob("*_config.json"))
        if not config_files:
            raise FileNotFoundError(f"No config JSON file found in {model_dir}")
        config_path = config_files[0]
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Loaded config from {config_path}")
    
    # Extract model parameters from config
    signal_dim = config.get('signal_dim', 35)
    metadata_embedding_dim = signal_dim * 4
    dropout = config.get('dropout', 0.1)
    nhead = config.get('nhead', 9)
    n_sab_layers = config.get('n-sab-layers', config.get('n_sab_layers', 4))
    n_cnn_layers = config.get('n-cnn-layers', config.get('n_cnn_layers', 3))
    conv_kernel_size = config.get('conv-kernel-size', config.get('conv_kernel_size', 3))
    pool_size = config.get('pool-size', config.get('pool_size', 2))
    context_length = config.get('context-length', config.get('context_length', 1200))
    separate_decoders = config.get('separate-decoders', config.get('separate_decoders', True))
    unet = config.get('unet', False)
    pos_enc = config.get('pos-enc', config.get('pos_enc', 'relative'))
    expansion_factor = config.get('expansion-factor', config.get('expansion_factor', 3))
    attention_type = config.get('attention-type', config.get('attention_type', 'dual'))
    norm = config.get('norm-type', config.get('norm', 'batch'))
    output_ff = config.get('output_ff', False)
    
    # Get metadata dimensions
    num_sequencing_platforms = config.get('num_sequencing_platforms', 10)
    num_runtypes = config.get('num_runtypes', 2)
    
    # Create model
    if unet:
        from model import CANDI_UNET
        model = CANDI_UNET(
            signal_dim=signal_dim,
            metadata_embedding_dim=metadata_embedding_dim,
            conv_kernel_size=conv_kernel_size,
            n_cnn_layers=n_cnn_layers,
            nhead=nhead,
            n_sab_layers=n_sab_layers,
            pool_size=pool_size,
            dropout=dropout,
            context_length=context_length,
            pos_enc=pos_enc,
            expansion_factor=expansion_factor,
            separate_decoders=separate_decoders,
            num_sequencing_platforms=num_sequencing_platforms,
            num_runtypes=num_runtypes,
            norm=norm,
            attention_type=attention_type
        )
    else:
        model = CANDI(
            signal_dim=signal_dim,
            metadata_embedding_dim=metadata_embedding_dim,
            conv_kernel_size=conv_kernel_size,
            n_cnn_layers=n_cnn_layers,
            nhead=nhead,
            n_sab_layers=n_sab_layers,
            pool_size=pool_size,
            dropout=dropout,
            context_length=context_length,
            pos_enc=pos_enc,
            expansion_factor=expansion_factor,
            separate_decoders=separate_decoders,
            num_sequencing_platforms=num_sequencing_platforms,
            num_runtypes=num_runtypes,
            norm=norm,
            attention_type=attention_type,
            output_ff=output_ff
        )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, config


def load_eic_validation_data(data_handler, V_biosample, locus, context_length, resolution=25):
    """
    Load data for a V_* biosample and its corresponding T_*.
    Following the pattern from EIC_VALIDATION_MONITOR._load_validation_data.
    
    Args:
        data_handler: CANDIDataHandler instance
        V_biosample: Name of V_* biosample (e.g., "V_heart_left_ventricle")
        locus: Genomic locus as [chrom, start, end]
        context_length: Context length in bins
        resolution: Genomic resolution in bp
        
    Returns:
        Dictionary with:
        - X: Input data [B, L, F] (from T_* with control)
        - mX: Input metadata [B, 4, F]
        - avX: Input availability [B, F]
        - Y_T: T_* ground truth [B, L, F] (for upsampled assays)
        - Y_V: V_* ground truth [B, L, F] (for imputed assays)
        - mY_T, mY_V: Metadata for T_* and V_*
        - available_T_indices: List of assay indices available in T_*
        - available_V_indices: List of assay indices available in V_*
        - seq: DNA sequence [B, L*resolution, 4]
        - expnames: List of experiment names
    """
    # Find corresponding T_* biosample
    T_biosample = V_biosample.replace("V_", "T_")
    
    if T_biosample not in data_handler.navigation:
        raise ValueError(f"T_* biosample {T_biosample} not found for {V_biosample}")
    
    # Load T_* data and ground truth (identical data loading)
    temp_x, temp_mx = data_handler.load_bios_Counts(T_biosample, locus, DSF=1)
    X, mX, avX = data_handler.make_bios_tensor_Counts(temp_x, temp_mx)
    Y_T, mY_T, avY_T = X.clone(), mX.clone(), avX.clone()  # Same source
    del temp_x, temp_mx
    
    # Load V_* ground truth (for imputed assays)
    temp_y_V, temp_my_V = data_handler.load_bios_Counts(V_biosample, locus, DSF=1)
    Y_V, mY_V, avY_V = data_handler.make_bios_tensor_Counts(temp_y_V, temp_my_V)
    del temp_y_V, temp_my_V
    
    # Create unified mY by integrating mY_T into mY_V where mY_V is missing
    mY = mY_V.clone()
    mask = (mY == -1)
    mY[mask] = mY_T[mask]
    
    # Fill in prompt for y metadata (median mode)
    # Note: fill_in_prompt expects [4, E] but sometimes unsqueezes 0-th dim inside?
    # Let's check data.py:
    #   filled = md.clone().squeeze(0) -> assumes input has a batch dim 1 at pos 0?
    #   If md is [4, E], squeeze(0) does nothing if dim 0 is 4.
    #   Wait, if md is [4, E], squeeze(0) doesn't remove dim 0 (size 4).
    #   So if input is [4, E], filled becomes [4, E].
    #   Then it processes columns.
    #   Finally returns filled.unsqueeze(0) -> [1, 4, E].
    
    # So if we pass [4, E], we get [1, 4, E].
    mY = data_handler.fill_in_prompt(mY, sample=False, use_mode=False)
    
    # If mY is now [1, 4, E], we should squeeze it back to [4, E] before expanding?
    # Or just use it as [1, 4, E] and expand dim 0.
    if mY.dim() == 3 and mY.shape[0] == 1:
        mY = mY.squeeze(0)
    # Now mY is [4, E]
    
    # Load control data
    try:
        temp_control_data, temp_control_metadata = data_handler.load_bios_Control(T_biosample, locus, DSF=1)
        if temp_control_data and "chipseq-control" in temp_control_data:
            control_data, control_meta, control_avail = data_handler.make_bios_tensor_Control(temp_control_data, temp_control_metadata)
        else:
            temp_control_data, temp_control_metadata = data_handler.load_bios_Control(V_biosample, locus, DSF=1)
            if temp_control_data and "chipseq-control" in temp_control_data:
                control_data, control_meta, control_avail = data_handler.make_bios_tensor_Control(temp_control_data, temp_control_metadata)
            else:
                raise ValueError("No control data found")
    except Exception as e:
        L = X.shape[0]
        control_data = torch.full((L, 1), -1.0)
        control_meta = torch.full((4, 1), -1.0)
        control_avail = torch.zeros(1)
    
    # Concatenate control data to input
    X = torch.cat([X, control_data], dim=1)
    mX = torch.cat([mX, control_meta], dim=1)
    avX = torch.cat([avX, control_avail], dim=0)
    
    # Prepare data for model (reshape to context windows)
    num_rows = (X.shape[0] // context_length) * context_length
    X = X[:num_rows, :]
    Y_T = Y_T[:num_rows, :]
    Y_V = Y_V[:num_rows, :]
    
    # Reshape to context windows
    X = X.view(-1, context_length, X.shape[-1])
    Y_T = Y_T.view(-1, context_length, Y_T.shape[-1])
    Y_V = Y_V.view(-1, context_length, Y_V.shape[-1])
    
    # Expand metadata to match batch dimension
    # mX is [4, F], X is [B, L, F] (reshaped) -> we need mX to be [B, 4, F]
    # mY is [4, F] -> we need mY to be [B, 4, F]
    
    mX = mX.unsqueeze(0).expand(X.shape[0], -1, -1)
    mY = mY.unsqueeze(0).expand(X.shape[0], -1, -1)
    avX_expanded = avX.unsqueeze(0).expand(X.shape[0], -1)
    avY_T_expanded = avY_T.unsqueeze(0).expand(X.shape[0], -1)
    avY_V_expanded = avY_V.unsqueeze(0).expand(X.shape[0], -1)
    
    # Load DNA sequence
    seq = data_handler._dna_to_onehot(
        data_handler._get_DNA_sequence(locus[0], locus[1], locus[2])
    )
    seq = seq[:num_rows * resolution, :]
    seq = seq.view(-1, context_length * resolution, seq.shape[-1])
    
    # Get experiment names
    expnames = list(data_handler.aliases["experiment_aliases"].keys())
    
    # Get available indices for T_* (input) - remove control if present
    available_T_indices = torch.where(avY_T == 1)[0].tolist()
    # Remove control index if present (control is at the end)
    available_T_indices = [idx for idx in available_T_indices if idx < len(expnames)]
    
    # Get available indices for V_* (target)
    available_V_indices = torch.where(avY_V == 1)[0].tolist()
    
    return {
        'X': X,
        'mX': mX,
        'avX': avX_expanded,
        'Y_T': Y_T,
        'Y_V': Y_V,
        'mY': mY,
        'avY_T': avY_T,
        'avY_V': avY_V,
        'available_T_indices': available_T_indices,
        'available_V_indices': available_V_indices,
        'seq': seq,
        'expnames': expnames
    }


def run_inference(model, X, mX, mY, seq, device):
    """
    Run model inference following EIC_VALIDATION_MONITOR._predict pattern.
    
    Args:
        model: CANDI model
        X: Input data [B, L, F] (includes control track)
        mX: Input metadata [B, 4, F]
        mY: Output metadata [B, 4, F] (should be 35 features, no control)
        seq: DNA sequence [B, L*resolution, 4]
        device: Device to run on
    
    Returns:
        mu_pred: Predicted mean [B, L, F]
        n_pred: Predicted dispersion parameter [B, L, F]
    """
    # Apply masking (convert missing to cloze)
    X = X.clone()
    mX = mX.clone()
    X[X == TOKEN_DICT["missing_mask"]] = float(TOKEN_DICT["cloze_mask"])
    mX[mX == TOKEN_DICT["missing_mask"]] = float(TOKEN_DICT["cloze_mask"])
    
    # Get signal_dim (35 features, no control)
    signal_dim = mY.shape[-1]
    
    # Move to device
    X = X.to(device).float()
    mX = mX.to(device).float()
    mY = mY[:, :, :signal_dim].to(device).float()  # Ensure 35 features
    seq = seq.to(device).float()
    
    # Run inference
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=device != 'cpu'):
        outputs_p, outputs_n, outputs_mu, outputs_var, outputs_peak = model(X, seq, mX, mY)
    
    # Convert to float32 for post-processing
    n_pred = outputs_n.float().cpu()
    p_pred = outputs_p.float().cpu()
    
    # Compute mu from n and p: mean = n * (1-p) / p
    mu_pred = n_pred * (1 - p_pred) / p_pred
    
    return mu_pred, n_pred


def visualize_outputs(data, mu_pred, n_pred, biosample, chrom, start_loci, save_path, resolution=25):
    """
    Create visualization with tracks using fast fill_between rendering.
    
    Structure:
    - Rows: Assays (F rows)
    - Columns: 3 (Ground Truth, Predicted Mu, Predicted N)
    
    Args:
        data: Dictionary from load_eic_validation_data
        mu_pred: Predicted mean [B, L, F]
        n_pred: Predicted dispersion [B, L, F]
        biosample: Biosample name
        chrom: Chromosome
        start_loci: Start position
        save_path: Path to save figure
        resolution: Resolution in bp per bin
    """
    # Get data from dict
    Y_T = data['Y_T']  # [B, L, F]
    Y_V = data['Y_V']  # [B, L, F]
    expnames = data['expnames']
    available_T_indices = set(data['available_T_indices'])
    available_V_indices = set(data['available_V_indices'])
    
    # Identify imputed assays (in V_* but not in T_*)
    imputed_indices = available_V_indices - available_T_indices
    upsampled_indices = available_T_indices
    
    # For visualization, use first batch window
    batch_idx = 0
    Y_T_vis = Y_T[batch_idx].clone().float()  # [L, F]
    Y_V_vis = Y_V[batch_idx].clone().float()  # [L, F]
    mu_vis = mu_pred[batch_idx].clone().float()  # [L, F]
    n_vis = n_pred[batch_idx].clone().float()  # [L, F]
    
    L, F = Y_T_vis.shape
    
    # X-axis coordinates
    x_coords = np.arange(L)
    
    # Determine plot dimensions
    # Vertical space per track (in inches)
    track_height = 0.8
    total_height = max(10, F * track_height)
    
    # Create figure and axes
    # Share x axis across all plots
    fig, axes = plt.subplots(F, 3, figsize=(24, total_height), sharex=True)
    
    # Handle F=1 case where axes is 1D
    if F == 1:
        axes = np.array([axes])
    
    # X-axis ticks (only on bottom rows)
    num_xticks = 10
    xtick_locs = np.linspace(0, L-1, num_xticks)
    xtick_labels = [f"{int(start_loci + x * resolution):,}" for x in xtick_locs]
    
    print(f"Rendering {F} tracks...")
    
    # Iterate over assays
    for f_idx in range(F):
        assay_name = expnames[f_idx] if f_idx < len(expnames) else f"Assay_{f_idx}"
        
        # Row axes
        ax_gt = axes[f_idx, 0]
        ax_mu = axes[f_idx, 1]
        ax_n = axes[f_idx, 2]
        
        # --- Column 1: Ground Truth ---
        gt_max = 0
        if f_idx in upsampled_indices:
            # Upsampled (T_*) - Green
            vals = Y_T_vis[:, f_idx].numpy()
            # Replace -1 with 0 for plotting (masked values won't show if 0, but technically -1 means missing)
            # Better to mask -1s or treat as 0
            plot_vals = np.where(vals == -1, 0, vals)
            
            # FAST RENDER: fill_between instead of bar
            # step='post' gives the histogram/bar-like appearance
            ax_gt.fill_between(x_coords, 0, plot_vals, step='post', 
                             color='green', alpha=1.0, label='Observed (T_*)')
            
            ax_gt.set_ylabel(assay_name, rotation=0, ha='right', fontsize=10)
            
            # Add label indicating type
            ax_gt.text(0.02, 0.85, "Observed", transform=ax_gt.transAxes, color='green', fontsize=8, fontweight='bold')
            
            if (vals != -1).any():
                gt_max = vals[vals != -1].max()
            
        elif f_idx in imputed_indices:
            # Imputed (V_*) - Blue
            vals = Y_V_vis[:, f_idx].numpy()
            plot_vals = np.where(vals == -1, 0, vals)
            
            # FAST RENDER
            ax_gt.fill_between(x_coords, 0, plot_vals, step='post',
                             color='blue', alpha=1.0, label='Target (V_*)')
            
            ax_gt.set_ylabel(assay_name, rotation=0, ha='right', fontsize=10, color='blue')
            
            # Add label indicating type
            ax_gt.text(0.02, 0.85, "Imputation Target", transform=ax_gt.transAxes, color='blue', fontsize=8, fontweight='bold')
            
            if (vals != -1).any():
                gt_max = vals[vals != -1].max()
            
        else:
            # Missing in both
            ax_gt.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax_gt.transAxes, color='gray')
            ax_gt.set_ylabel(assay_name, rotation=0, ha='right', fontsize=10, color='gray')
        
        # Remove spines for cleaner look
        ax_gt.spines['top'].set_visible(False)
        ax_gt.spines['right'].set_visible(False)
        
        # Set y-limit explicitly to start at 0
        if gt_max > 0:
            ax_gt.set_ylim(0, gt_max * 1.1)
            ax_gt.text(0.98, 0.85, f"Max: {gt_max:.1f}", transform=ax_gt.transAxes, ha='right', fontsize=8)
        
        # --- Column 2: Predicted Mu ---
        mu_vals = mu_vis[:, f_idx].numpy()
        
        # FAST RENDER
        ax_mu.fill_between(x_coords, 0, mu_vals, step='post', color='red', alpha=0.8)
        
        # Remove spines
        ax_mu.spines['top'].set_visible(False)
        ax_mu.spines['right'].set_visible(False)
        
        mu_max = mu_vals.max()
        if mu_max > 0:
            ax_mu.set_ylim(0, mu_max * 1.1)
            ax_mu.text(0.98, 0.85, f"Max: {mu_max:.1f}", transform=ax_mu.transAxes, ha='right', fontsize=8)
            
        # --- Column 3: Predicted N ---
        n_vals = n_vis[:, f_idx].numpy()
        
        # FAST RENDER
        ax_n.fill_between(x_coords, 0, n_vals, step='post', color='purple', alpha=0.8)
        
        # Remove spines
        ax_n.spines['top'].set_visible(False)
        ax_n.spines['right'].set_visible(False)
        
        n_max = n_vals.max()
        if n_max > 0:
            ax_n.set_ylim(0, n_max * 1.1)
            ax_n.text(0.98, 0.85, f"Max: {n_max:.1f}", transform=ax_n.transAxes, ha='right', fontsize=8)

    # Set Column Titles (only on top row)
    axes[0, 0].set_title('Ground Truth\n(Green=Input, Blue=Target)', fontsize=14, fontweight='bold')
    axes[0, 1].set_title('Predicted μ (Mean)', fontsize=14, fontweight='bold', color='darkred')
    axes[0, 2].set_title('Predicted n (Dispersion)', fontsize=14, fontweight='bold', color='indigo')
    
    # Set X-axis labels (only on bottom row)
    for col in range(3):
        ax = axes[F-1, col]
        ax.set_xticks(xtick_locs)
        ax.set_xticklabels(xtick_labels, rotation=45, ha='right')
        ax.set_xlabel(f'Genomic Position ({chrom})', fontsize=12)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.95) # Make room for suptitle
    
    # Title
    plt.suptitle(f'CANDI Outputs: {biosample} at {chrom}:{start_loci:,}', 
                 fontsize=16, fontweight='bold', y=0.998)
    
    # Save figure
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight') # Lower dpi slightly due to large size
    print(f"Saved visualization to {save_path}")
    
    # Print summary
    print(f"\n=== Summary ===")
    print(f"Upsampled assays (in T_*): {len(upsampled_indices)}")
    for idx in sorted(upsampled_indices):
        if idx < len(expnames):
            print(f"  - {expnames[idx]}")
    print(f"\nImputed assays (in V_* only): {len(imputed_indices)}")
    for idx in sorted(imputed_indices):
        if idx < len(expnames):
            print(f"  - {expnames[idx]}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize CANDI model outputs for EIC validation samples')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to model checkpoint (.pt) or model directory')
    parser.add_argument('--biosample', type=str, required=False,
                       help='V_* biosample name (e.g., V_heart_left_ventricle or V_chorion). '
                            'The corresponding T_ biosample will be used as input. '
                            'Not required if --list-biosamples is used.')
    parser.add_argument('--list-biosamples', action='store_true',
                       help='List available biosamples and exit')
    parser.add_argument('--start-loci', type=int, default=1000000,
                       help='Start position in base pairs (default: 1000000)')
    parser.add_argument('--chrom', type=str, default='chr21',
                       help='Chromosome (default: chr21)')
    parser.add_argument('--resolution', type=int, default=25,
                       help='Resolution in base pairs (default: 25)')
    parser.add_argument('--context-length', type=int, default=None,
                       help='Context length in bins (will try to load from model config if not provided)')
    parser.add_argument('--base-path', type=str, default=None,
                       help='Path to EIC dataset (default: will try to infer from data.py defaults)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu, default: auto-detect)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Initialize data handler first (needed for --list-biosamples)
    if args.base_path is None:
        base_path = os.environ.get('CANDI_EIC_DATA_PATH', '/project/6014832/mforooz/DATA_CANDI_EIC')
    else:
        base_path = args.base_path
    
    print(f"Using data path: {base_path}")
    data_handler = CANDIDataHandler(
        base_path=base_path,
        resolution=args.resolution,
        dataset_type="eic",
        DNA=True
    )
    
    # List biosamples if requested
    if args.list_biosamples:
        available_biosamples = list(data_handler.navigation.keys())
        t_biosamples = sorted([b for b in available_biosamples if b.startswith('T_')])
        v_biosamples = sorted([b for b in available_biosamples if b.startswith('V_')])
        
        print(f"\nAvailable V_ biosamples with corresponding T_ biosamples:")
        v_with_t = []
        for v_bios in v_biosamples:
            t_bios = v_bios.replace("V_", "T_")
            if t_bios in t_biosamples:
                v_with_t.append(v_bios)
        
        for b in v_with_t:
            print(f"  {b}")
        
        print(f"\nTotal: {len(v_with_t)} V_ biosamples with matching T_ biosamples")
        return
    
    # Check if biosample is provided
    if not args.biosample:
        print("Error: --biosample is required unless --list-biosamples is used.")
        parser.print_help()
        return
    
    # Ensure biosample starts with V_
    biosample = args.biosample
    if not biosample.startswith("V_"):
        if biosample.startswith("T_"):
            biosample = "V_" + biosample[2:]
        else:
            biosample = "V_" + biosample
        print(f"Note: Converted biosample to {biosample}")
    
    # Check biosample exists
    T_biosample = biosample.replace("V_", "T_")
    if T_biosample not in data_handler.navigation:
        print(f"Error: T_ biosample '{T_biosample}' not found.")
        print("Use --list-biosamples to see available biosamples.")
        return
    if biosample not in data_handler.navigation:
        print(f"Error: V_ biosample '{biosample}' not found.")
        print("Use --list-biosamples to see available biosamples.")
        return
    
    # Load model
    print("Loading model...")
    model, config = load_model_from_checkpoint(args.model_path, device)
    
    # Get context length from config if not provided (in bins, not bp)
    context_length = args.context_length
    if context_length is None:
        context_length = config.get('context-length', config.get('context_length', 1200))
    print(f"Using context length: {context_length} bins ({context_length * args.resolution} bp)")
    
    # Calculate locus end
    end_loci = args.start_loci + context_length * args.resolution
    locus = [args.chrom, args.start_loci, end_loci]
    print(f"Loading data for {args.chrom}:{args.start_loci:,}-{end_loci:,}")
    
    # Load data following EIC_VALIDATION_MONITOR pattern
    print("Loading data...")
    data = load_eic_validation_data(
        data_handler, biosample, locus, context_length, args.resolution
    )
    
    print(f"  X shape: {data['X'].shape}")
    print(f"  Y_T shape: {data['Y_T'].shape}")
    print(f"  Y_V shape: {data['Y_V'].shape}")
    print(f"  seq shape: {data['seq'].shape}")
    print(f"  Available T_* assays: {len(data['available_T_indices'])}")
    print(f"  Available V_* assays: {len(data['available_V_indices'])}")
    
    # Run inference following EIC_VALIDATION_MONITOR._predict pattern
    print("Running inference...")
    mu_pred, n_pred = run_inference(
        model, data['X'], data['mX'], data['mY'], data['seq'], device
    )
    print(f"  mu_pred shape: {mu_pred.shape}")
    print(f"  n_pred shape: {n_pred.shape}")
    
    # Determine save path
    model_path = Path(args.model_path)
    if model_path.is_file():
        model_dir = model_path.parent
    else:
        model_dir = model_path
    
    viz_dir = model_dir / "viz"
    save_path = viz_dir / f"{biosample}_{args.chrom}_{args.start_loci}.png"
    
    # Visualize
    print("Creating visualization...")
    visualize_outputs(
        data, mu_pred, n_pred,
        biosample, args.chrom, args.start_loci, save_path,
        resolution=args.resolution
    )
    
    print("Done!")


if __name__ == '__main__':
    main()

