from _utils import *    
from data import * 

import torch, math, random, time, json, os, pickle, sys, gc
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import datetime
from scipy.stats import nbinom
import imageio.v2 as imageio
from io import BytesIO
from torchinfo import summary
from typing import List, Dict

try:
    from x_transformers import Encoder as XTransformerEncoder, Attention as XAttention, CrossAttender
    XTRANSFORMERS_AVAILABLE = True
except ImportError:
    XTRANSFORMERS_AVAILABLE = False
    XTransformerEncoder = None
    XAttention = None
    CrossAttender = None

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

##=========================================== EIC Validation Monitor =============================================##

class EIC_VALIDATION_MONITOR(object):
    """
    Validation monitor for EIC dataset during training.
    Evaluates model on V_* biosamples using T_* as input.
    Computes NLL losses (not weighted) for imputed and upsampled assays across chr21.
    """
    
    def __init__(self, context_length, training_batch_size, device=None, resolution=25):
        """
        Initialize EIC validation monitor.
        
        Args:
            context_length: Context length for genomic windows (in bins)
            training_batch_size: Training batch size (validation uses 2x this)
            device: Device to use for validation (auto-detect if None)
            resolution: Genomic resolution in bp
        """
        self.data_path = "/project/6014832/mforooz/DATA_CANDI_EIC"
        self.context_length = context_length
        self.resolution = resolution
        self.validation_batch_size = int(training_batch_size * 4)  
        print(f"Validation batch size: {self.validation_batch_size}")
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Initialize data handler
        self.data_handler = CANDIDataHandler(
            base_path=self.data_path,
            resolution=self.resolution,
            dataset_type="eic",
            DNA=True
        )
        self.data_handler._load_files()
        
        # Filter to V_* biosamples only (validation split)
        self.v_biosamples = []
        for bios in list(self.data_handler.navigation.keys()):
            if bios.startswith("V_"):
                self.v_biosamples.append(bios)
        
        print(f"EIC_VALIDATION_MONITOR initialized with {len(self.v_biosamples)} V_* biosamples")
        
        # Get experiment names
        self.expnames = list(self.data_handler.aliases["experiment_aliases"].keys())
        
        # Load chromosome sizes
        self.chr_sizes = {}
        chr_sizes_file = "data/hg38.chrom.sizes"
        main_chrs = ["chr" + str(x) for x in range(1, 23)] + ["chrX"]
        if os.path.exists(chr_sizes_file):
            with open(chr_sizes_file, 'r') as f:
                for line in f:
                    chr_name, chr_size = line.strip().split('\t')
                    if chr_name in main_chrs:
                        self.chr_sizes[chr_name] = int(chr_size)
        else:
            # Fallback
            self.chr_sizes = {"chr21": 46709983}
        
        # Initialize loss functions
        self.nbin_nll = negative_binomial_loss
        self.gaus_nll = torch.nn.GaussianNLLLoss(reduction="mean", full=True)
        self.peak_bce = torch.nn.BCEWithLogitsLoss(reduction="mean")
        
        # Token dictionary
        self.token_dict = {"missing_mask": -1, "cloze_mask": -2, "pad": -3}
    
    def _load_validation_data(self, V_biosample: str, locus: List, cached_seq=None):
        """
        Load data for a V_* biosample and its corresponding T_*.
        
        Args:
            V_biosample: Name of V_* biosample
            locus: Genomic locus as [chrom, start, end]
            cached_seq: Pre-loaded DNA sequence tensor (optional, for speed)
            
        Returns:
            Dictionary with:
            - X, mX, avX (from T_*)
            - Y_T, P_T, Peak_T (T_* ground truth for upsampled)
            - Y_V, P_V, Peak_V (V_* ground truth for imputed)
            - available_T_indices (upsampled assays)
            - available_V_indices (all V_* assays)
            - seq (DNA sequence)
        """
        # Find corresponding T_* biosample
        T_biosample = V_biosample.replace("V_", "T_")
        
        if T_biosample not in self.data_handler.navigation:
            raise ValueError(f"T_* biosample {T_biosample} not found for {V_biosample}")
        
        # Load T_* data and ground truth (identical data loading)
        temp_x, temp_mx = self.data_handler.load_bios_Counts(T_biosample, locus, DSF=1)
        X, mX, avX = self.data_handler.make_bios_tensor_Counts(temp_x, temp_mx)
        Y_T, mY_T, avY_T = X, mX, avX  # identical tensors since same source
        del temp_x, temp_mx
        
        # Load V_* ground truth (for imputed assays)
        temp_y_V, temp_my_V = self.data_handler.load_bios_Counts(V_biosample, locus, DSF=1)
        Y_V, mY_V, avY_V = self.data_handler.make_bios_tensor_Counts(temp_y_V, temp_my_V)
        del temp_y_V, temp_my_V
        
        # Load P-value data separately for T_* and V_*
        temp_p_T = self.data_handler.load_bios_BW(T_biosample, locus)
        temp_p_V = self.data_handler.load_bios_BW(V_biosample, locus)
        
        # Create merged P for model input (contains all assays)
        temp_p_merged = {**temp_p_V, **temp_p_T}
        P_merged, avlP = self.data_handler.make_bios_tensor_BW(temp_p_merged)
        
        # Create separate P tensors for T_* and V_*
        P_T, _ = self.data_handler.make_bios_tensor_BW(temp_p_T)
        P_V, _ = self.data_handler.make_bios_tensor_BW(temp_p_V)
        
        del temp_p_T, temp_p_V, temp_p_merged
        
        # Load Peak data separately for T_* and V_*
        temp_peak_T = self.data_handler.load_bios_Peaks(T_biosample, locus)
        temp_peak_V = self.data_handler.load_bios_Peaks(V_biosample, locus)
        
        # Create merged Peak for model input (contains all assays)
        temp_peak_merged = {**temp_peak_V, **temp_peak_T}
        Peak_merged, avlPeak = self.data_handler.make_bios_tensor_Peaks(temp_peak_merged)
        
        # Create separate Peak tensors for T_* and V_*
        Peak_T, _ = self.data_handler.make_bios_tensor_Peaks(temp_peak_T)
        Peak_V, _ = self.data_handler.make_bios_tensor_Peaks(temp_peak_V)
        
        del temp_peak_T, temp_peak_V, temp_peak_merged
        
        # Load control data
        try:
            temp_control_data, temp_control_metadata = self.data_handler.load_bios_Control(T_biosample, locus, DSF=1)
            if temp_control_data and "chipseq-control" in temp_control_data:
                control_data, control_meta, control_avail = self.data_handler.make_bios_tensor_Control(temp_control_data, temp_control_metadata)
            else:
                temp_control_data, temp_control_metadata = self.data_handler.load_bios_Control(V_biosample, locus, DSF=1)
                if temp_control_data and "chipseq-control" in temp_control_data:
                    control_data, control_meta, control_avail = self.data_handler.make_bios_tensor_Control(temp_control_data, temp_control_metadata)
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
        num_rows = (X.shape[0] // self.context_length) * self.context_length
        X = X[:num_rows, :]
        Y_T = Y_T[:num_rows, :]
        Y_V = Y_V[:num_rows, :]
        
        # Reshape P and Peak data to match X/Y length
        # Ensure all have the same number of rows
        min_rows = min(X.shape[0], P_merged.shape[0], Peak_merged.shape[0], 
                      P_T.shape[0], P_V.shape[0], Peak_T.shape[0], Peak_V.shape[0])
        num_rows = (min_rows // self.context_length) * self.context_length
        
        X = X[:num_rows, :]
        Y_T = Y_T[:num_rows, :]
        Y_V = Y_V[:num_rows, :]
        P_merged = P_merged[:num_rows, :]
        P_T = P_T[:num_rows, :]
        P_V = P_V[:num_rows, :]
        Peak_merged = Peak_merged[:num_rows, :]
        Peak_T = Peak_T[:num_rows, :]
        Peak_V = Peak_V[:num_rows, :]
        
        # Reshape to context windows
        X = X.view(-1, self.context_length, X.shape[-1])
        Y_T = Y_T.view(-1, self.context_length, Y_T.shape[-1])
        Y_V = Y_V.view(-1, self.context_length, Y_V.shape[-1])
        P_T = P_T.view(-1, self.context_length, P_T.shape[-1])
        P_V = P_V.view(-1, self.context_length, P_V.shape[-1])
        Peak_T = Peak_T.view(-1, self.context_length, Peak_T.shape[-1])
        Peak_V = Peak_V.view(-1, self.context_length, Peak_V.shape[-1])
        
        # Expand metadata to match batch dimension
        mX = mX.expand(X.shape[0], -1, -1)
        mY_T = mY_T.expand(X.shape[0], -1, -1)
        mY_V = mY_V.expand(X.shape[0], -1, -1)
        avX = avX.expand(X.shape[0], -1)
        
        # Load DNA sequence (use cached if available)
        if cached_seq is not None:
            # Use cached sequence, just slice to match num_rows
            seq = cached_seq[:num_rows, :, :]
        else:
            seq = self.data_handler._dna_to_onehot(
                self.data_handler._get_DNA_sequence(locus[0], locus[1], locus[2])
            )
            seq = seq[:num_rows * self.resolution, :]
            seq = seq.view(-1, self.context_length * self.resolution, seq.shape[-1])
        
        # Get available indices for T_* (input) - remove control if present
        if avX.ndim == 1:
            available_T_indices = torch.where(avX == 1)[0].tolist()
        else:
            available_T_indices = torch.where(avX[0, :] == 1)[0].tolist()
        
        # Remove control index if present (control is at the end)
        if len(available_T_indices) > 0 and available_T_indices[-1] >= len(self.expnames):
            available_T_indices = [idx for idx in available_T_indices if idx < len(self.expnames)]
        
        # Get available indices for V_* (target)
        if avY_V.ndim == 1:
            available_V_indices = torch.where(avY_V == 1)[0].tolist()
        else:
            available_V_indices = torch.where(avY_V[0, :] == 1)[0].tolist()
        
        return {
            'X': X,
            'mX': mX,
            'avX': avX,
            'Y_T': Y_T,
            'P_T': P_T,
            'Peak_T': Peak_T,
            'Y_V': Y_V,
            'P_V': P_V,
            'Peak_V': Peak_V,
            'mY_T': mY_T,
            'mY_V': mY_V,
            'available_T_indices': available_T_indices,
            'available_V_indices': available_V_indices,
            'seq': seq
        }
    
    def _predict(self, model, X, mX, mY, avX, seq):
        """
        Run model prediction in batches.
        
        Returns:
            output_p, output_n, output_mu, output_var, output_peak
        """
        # Unwrap DDP if needed
        if hasattr(model, 'module'):
            model_to_use = model.module
        else:
            model_to_use = model
        
        model_to_use.train()  # Use batch statistics (avoids corrupted running stats in BatchNorm)
        
        # Initialize output tensors
        original_feature_dim = X.shape[-1] - 1  # Subtract control
        n = torch.empty(X.shape[0], X.shape[1], original_feature_dim, device="cpu", dtype=torch.float32)
        p = torch.empty(X.shape[0], X.shape[1], original_feature_dim, device="cpu", dtype=torch.float32)
        mu = torch.empty(X.shape[0], X.shape[1], original_feature_dim, device="cpu", dtype=torch.float32)
        var = torch.empty(X.shape[0], X.shape[1], original_feature_dim, device="cpu", dtype=torch.float32)
        peak = torch.empty(X.shape[0], X.shape[1], original_feature_dim, device="cpu", dtype=torch.float32)
        
        # Process in batches with mixed precision for speed
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=True):
            for i in range(0, len(X), self.validation_batch_size):
                x_batch = X[i:i + self.validation_batch_size]
                mX_batch = mX[i:i + self.validation_batch_size]
                mY_batch = mY[i:i + self.validation_batch_size]
                seq_batch = seq[i:i + self.validation_batch_size]
                
                # Apply masking (in-place on clones)
                x_batch = x_batch.clone()
                mX_batch = mX_batch.clone()
                x_batch[x_batch == self.token_dict["missing_mask"]] = float(self.token_dict["cloze_mask"])
                mX_batch[mX_batch == self.token_dict["missing_mask"]] = float(self.token_dict["cloze_mask"])
                
                # Move to device
                x_batch = x_batch.to(self.device, non_blocking=True)
                mX_batch = mX_batch.to(self.device, non_blocking=True)
                mY_batch = mY_batch.to(self.device, non_blocking=True)
                seq_batch = seq_batch.to(self.device, non_blocking=True)
                
                # Forward pass
                outputs_p, outputs_n, outputs_mu, outputs_var, outputs_peak = model_to_use(
                    x_batch.float(), seq_batch, mX_batch.float(), mY_batch
                )
                
                # Store predictions (convert to float32 for metrics)
                batch_end = min(i + self.validation_batch_size, len(X))
                n[i:batch_end] = outputs_n.float().cpu()
                p[i:batch_end] = outputs_p.float().cpu()
                mu[i:batch_end] = outputs_mu.float().cpu()
                var[i:batch_end] = outputs_var.float().cpu()
                peak[i:batch_end] = outputs_peak.float().cpu()
        
        return n, p, mu, var, peak
    
    def _compute_count_nll(self, n_pred, p_pred, y_true):
        """Compute mean NB NLL across positions."""
        # Flatten to [B*L, F]
        n_flat = n_pred.view(-1, n_pred.shape[-1])
        p_flat = p_pred.view(-1, p_pred.shape[-1])
        y_flat = y_true.view(-1, y_true.shape[-1])
        
        # Compute NLL per position
        nll = self.nbin_nll(y_flat, n_flat, p_flat)
        return nll.mean().item()
    
    def _compute_signal_nll(self, mu_pred, var_pred, y_true):
        """Compute mean Gaussian NLL across positions."""
        # Flatten to [B*L, F]
        mu_flat = mu_pred.view(-1, mu_pred.shape[-1])
        var_flat = var_pred.view(-1, var_pred.shape[-1])
        y_flat = y_true.view(-1, y_true.shape[-1])
        
        # Compute NLL per position
        nll = self.gaus_nll(mu_flat, y_flat, var_flat)
        return nll.item()
    
    def _compute_peak_bce(self, peak_pred, peak_true):
        """Compute mean BCE across positions."""
        # Flatten to [B*L, F]
        peak_pred_flat = peak_pred.view(-1, peak_pred.shape[-1])
        peak_true_flat = peak_true.view(-1, peak_true.shape[-1]).float()  # Convert to float for BCE
        
        # Compute BCE
        bce = self.peak_bce(peak_pred_flat, peak_true_flat)
        return bce.item()
    
    def run_validation(self, model, batch_idx, total_batches):
        """
        Run validation on all V_* biosamples in EIC dataset.
        
        Args:
            model: CANDI model to evaluate
            batch_idx: Current batch index
            total_batches: Total number of batches in training
            
        Returns:
            Dictionary with:
            - iteration, progress_pct
            - mean metrics across all assays
            - per-assay-type metrics (averaged across biosamples)
        """
        print(f"Running EIC validation at batch {batch_idx} ({100.0 * batch_idx / total_batches:.1f}% progress)...")
        
        # Use full chr21
        locus = ["chr21", 0, self.chr_sizes["chr21"]]
        
        # Collect metrics per (biosample, assay_name, comparison_type)
        all_metrics = []
        
        # Pre-load DNA sequence once (same for all biosamples on chr21)
        print(f"  Loading DNA sequence for {locus[0]}...")
        chr_length = locus[2] - locus[1]
        num_windows = chr_length // (self.context_length * self.resolution)
        num_rows = num_windows * self.context_length
        
        cached_seq = self.data_handler._dna_to_onehot(
            self.data_handler._get_DNA_sequence(locus[0], locus[1], locus[2])
        )
        cached_seq = cached_seq[:num_rows * self.resolution, :]
        cached_seq = cached_seq.view(-1, self.context_length * self.resolution, cached_seq.shape[-1])
        print(f"  DNA sequence cached: {cached_seq.shape}")
        
        # Process all biosamples one at a time to avoid OOM
        num_biosamples = len(self.v_biosamples)
        print(f"  Processing {num_biosamples} biosamples on full {locus[0]}")
        
        for bios_idx, V_biosample in enumerate(self.v_biosamples):
            try:
                print(f"  [{bios_idx+1}/{num_biosamples}] Validating {V_biosample}...", end=" ", flush=True)
                
                # Clear memory before loading new biosample
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Load validation data (with cached DNA sequence)
                data = self._load_validation_data(V_biosample, locus, cached_seq=cached_seq)
                
                # Run prediction
                n, p, mu, var, peak = self._predict(
                    model, data['X'], data['mX'], data['mY_T'], data['avX'], data['seq']
                )
                
                # Determine which assays are upsampled vs imputed
                available_T_set = set(data['available_T_indices'])
                available_V_set = set(data['available_V_indices'])
                
                upsampled_assays = available_T_set  # Assays in T_*
                imputed_assays = available_V_set - available_T_set  # Assays in V_* but not in T_*
                
                # Compute metrics for upsampled assays (compare against T_* ground truth)
                for assay_idx in upsampled_assays:
                    if assay_idx >= len(self.expnames):
                        continue
                    
                    assay_name = self.expnames[assay_idx]
                    
                    # Extract predictions for this assay
                    n_assay = n[:, :, assay_idx]
                    p_assay = p[:, :, assay_idx]
                    mu_assay = mu[:, :, assay_idx]
                    var_assay = var[:, :, assay_idx]
                    peak_assay = peak[:, :, assay_idx]
                    
                    # Extract ground truth from T_*
                    y_T_assay = data['Y_T'][:, :, assay_idx]
                    p_T_assay = data['P_T'][:, :, assay_idx]
                    peak_T_assay = data['Peak_T'][:, :, assay_idx]
                    
                    # Compute losses
                    count_nll = self._compute_count_nll(n_assay, p_assay, y_T_assay)
                    signal_nll = self._compute_signal_nll(mu_assay, var_assay, p_T_assay)
                    peak_bce = self._compute_peak_bce(peak_assay, peak_T_assay)
                    
                    all_metrics.append({
                        'biosample': V_biosample,
                        'assay_name': assay_name,
                        'comparison': 'upsampled',
                        'count_nll': count_nll,
                        'signal_nll': signal_nll,
                        'peak_bce': peak_bce
                    })
                
                # Compute metrics for imputed assays (compare against V_* ground truth)
                for assay_idx in imputed_assays:
                    if assay_idx >= len(self.expnames):
                        continue
                    
                    assay_name = self.expnames[assay_idx]
                    
                    # Extract predictions for this assay
                    n_assay = n[:, :, assay_idx]
                    p_assay = p[:, :, assay_idx]
                    mu_assay = mu[:, :, assay_idx]
                    var_assay = var[:, :, assay_idx]
                    peak_assay = peak[:, :, assay_idx]
                    
                    # Extract ground truth from V_*
                    y_V_assay = data['Y_V'][:, :, assay_idx]
                    p_V_assay = data['P_V'][:, :, assay_idx]
                    peak_V_assay = data['Peak_V'][:, :, assay_idx]
                    
                    # Compute losses
                    count_nll = self._compute_count_nll(n_assay, p_assay, y_V_assay)
                    signal_nll = self._compute_signal_nll(mu_assay, var_assay, p_V_assay)
                    peak_bce = self._compute_peak_bce(peak_assay, peak_V_assay)
                    
                    all_metrics.append({
                        'biosample': V_biosample,
                        'assay_name': assay_name,
                        'comparison': 'imputed',
                        'count_nll': count_nll,
                        'signal_nll': signal_nll,
                        'peak_bce': peak_bce
                    })
                
                # Clean up memory after each biosample
                del data, n, p, mu, var, peak
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                print(f"done ({len(upsampled_assays)} ups, {len(imputed_assays)} imp)")
                
            except Exception as e:
                print(f"failed: {e}")
                import traceback
                traceback.print_exc()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
        
        if not all_metrics:
            print("Warning: No validation metrics computed")
            return {
                'iteration': batch_idx,
                'progress_pct': 100.0 * batch_idx / total_batches,
                'imp_count_nll_mean': 0.0,
                'imp_signal_nll_mean': 0.0,
                'imp_peak_bce_mean': 0.0,
                'ups_count_nll_mean': 0.0,
                'ups_signal_nll_mean': 0.0,
                'ups_peak_bce_mean': 0.0
            }
        
        # Convert to DataFrame for easier aggregation
        import pandas as pd
        df = pd.DataFrame(all_metrics)
        
        # Compute mean metrics across all assays
        imp_df = df[df['comparison'] == 'imputed']
        ups_df = df[df['comparison'] == 'upsampled']
        
        result = {
            'iteration': batch_idx,
            'progress_pct': 100.0 * batch_idx / total_batches,
            'imp_count_nll_mean': imp_df['count_nll'].mean() if len(imp_df) > 0 else 0.0,
            'imp_signal_nll_mean': imp_df['signal_nll'].mean() if len(imp_df) > 0 else 0.0,
            'imp_peak_bce_mean': imp_df['peak_bce'].mean() if len(imp_df) > 0 else 0.0,
            'ups_count_nll_mean': ups_df['count_nll'].mean() if len(ups_df) > 0 else 0.0,
            'ups_signal_nll_mean': ups_df['signal_nll'].mean() if len(ups_df) > 0 else 0.0,
            'ups_peak_bce_mean': ups_df['peak_bce'].mean() if len(ups_df) > 0 else 0.0
        }
        
        # Compute per-assay-type metrics (averaged across biosamples)
        for assay_name in df['assay_name'].unique():
            assay_imp_df = imp_df[imp_df['assay_name'] == assay_name]
            assay_ups_df = ups_df[ups_df['assay_name'] == assay_name]
            
            # Imputed metrics
            if len(assay_imp_df) > 0:
                result[f'{assay_name}_imp_count_nll'] = assay_imp_df['count_nll'].mean()
                result[f'{assay_name}_imp_signal_nll'] = assay_imp_df['signal_nll'].mean()
                result[f'{assay_name}_imp_peak_bce'] = assay_imp_df['peak_bce'].mean()
            
            # Upsampled metrics
            if len(assay_ups_df) > 0:
                result[f'{assay_name}_ups_count_nll'] = assay_ups_df['count_nll'].mean()
                result[f'{assay_name}_ups_signal_nll'] = assay_ups_df['signal_nll'].mean()
                result[f'{assay_name}_ups_peak_bce'] = assay_ups_df['peak_bce'].mean()
        
        print(f"Validation completed: {len(imp_df)} imputed, {len(ups_df)} upsampled assays evaluated")
        
        return result

##=========================================== Loss Functions =============================================##

class CANDI_LOSS(nn.Module):
    def __init__(self, reduction='mean', count_weight=1.0, pval_weight=1.0, peak_weight=1.0, obs_weight=1.0, imp_weight=1.0):
        super(CANDI_LOSS, self).__init__()
        self.reduction = reduction
        self.gaus_nll = nn.GaussianNLLLoss(reduction=self.reduction, full=True)
        self.nbin_nll = negative_binomial_loss
        self.bce_loss = nn.BCELoss(reduction=self.reduction)
        # Loss weights for multi-task learning
        self.count_weight = count_weight
        self.pval_weight = pval_weight
        self.peak_weight = peak_weight
        self.obs_weight = obs_weight
        self.imp_weight = imp_weight

    def forward(self, p_pred, n_pred, mu_pred, var_pred, peak_pred, true_count, true_pval, true_peak, obs_map, masked_map):
        ups_true_count, ups_true_pval, ups_true_peak = true_count[obs_map], true_pval[obs_map], true_peak[obs_map]
        ups_n_pred, ups_p_pred = n_pred[obs_map], p_pred[obs_map]
        ups_mu_pred, ups_var_pred = mu_pred[obs_map], var_pred[obs_map]
        ups_peak_pred = peak_pred[obs_map]

        imp_true_count, imp_true_pval, imp_true_peak = true_count[masked_map], true_pval[masked_map], true_peak[masked_map]
        imp_n_pred, imp_p_pred = n_pred[masked_map], p_pred[masked_map]
        imp_mu_pred, imp_var_pred = mu_pred[masked_map], var_pred[masked_map]
        imp_peak_pred = peak_pred[masked_map]

        observed_count_loss = self.nbin_nll(ups_true_count, ups_n_pred, ups_p_pred) 
        imputed_count_loss = self.nbin_nll(imp_true_count, imp_n_pred, imp_p_pred)

        if self.reduction == "mean":
            observed_count_loss = observed_count_loss.mean()
            imputed_count_loss = imputed_count_loss.mean()
        elif self.reduction == "sum":
            observed_count_loss = observed_count_loss.sum()
            imputed_count_loss = imputed_count_loss.sum()

        observed_pval_loss = self.gaus_nll(ups_mu_pred, ups_true_pval, ups_var_pred)
        imputed_pval_loss = self.gaus_nll(imp_mu_pred, imp_true_pval, imp_var_pred)

        observed_pval_loss = observed_pval_loss.float()
        imputed_pval_loss = imputed_pval_loss.float()
        
        # Peak losses using Binary Cross Entropy (disable autocast for BCE)
        with torch.amp.autocast("cuda", enabled=False):
            observed_peak_loss = self.bce_loss(ups_peak_pred.float(), ups_true_peak.float())
            imputed_peak_loss = self.bce_loss(imp_peak_pred.float(), imp_true_peak.float())
        
        # Apply weights to losses for multi-task learning
        # First apply task-specific weights (count, pval, peak)
        observed_count_loss = self.count_weight * observed_count_loss
        imputed_count_loss = self.count_weight * imputed_count_loss
        observed_pval_loss = self.pval_weight * observed_pval_loss
        imputed_pval_loss = self.pval_weight * imputed_pval_loss
        observed_peak_loss = self.peak_weight * observed_peak_loss
        imputed_peak_loss = self.peak_weight * imputed_peak_loss
        
        # Then apply obs/imp weights
        observed_count_loss = self.obs_weight * observed_count_loss
        observed_pval_loss = self.obs_weight * observed_pval_loss
        observed_peak_loss = self.obs_weight * observed_peak_loss
        imputed_count_loss = self.imp_weight * imputed_count_loss
        imputed_pval_loss = self.imp_weight * imputed_pval_loss
        imputed_peak_loss = self.imp_weight * imputed_peak_loss
        
        return observed_count_loss, imputed_count_loss, observed_pval_loss, imputed_pval_loss, observed_peak_loss, imputed_peak_loss

##=========================================== CANDI Architecture =============================================##

class CANDI_Decoder(nn.Module):
    def __init__(self, signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, context_length, pool_size=2, expansion_factor=3, num_sequencing_platforms=10, num_runtypes=2, norm="batch"):
        super(CANDI_Decoder, self).__init__()

        self.l1 = context_length
        self.l2 = self.l1 // (pool_size**n_cnn_layers)
        
        self.f1 = signal_dim
        self.signal_dim = signal_dim
        self.f2 = (self.f1 * (expansion_factor**(n_cnn_layers)))
        self.d_model =  self.latent_dim = self.f2

        conv_channels = [(self.f1)*(expansion_factor**l) for l in range(n_cnn_layers)]
        reverse_conv_channels = [expansion_factor * x for x in conv_channels[::-1]]
        conv_kernel_size = [conv_kernel_size for _ in range(n_cnn_layers)]

        self.deconv = nn.ModuleList()
        for i in range(n_cnn_layers):

            is_last_layer = (i == n_cnn_layers - 1)
            layer_norm_type = norm
            
            self.deconv.append(DeconvTower(
                reverse_conv_channels[i], 
                reverse_conv_channels[i + 1] if i + 1 < n_cnn_layers else int(reverse_conv_channels[i] / expansion_factor),
                conv_kernel_size[-(i + 1)], S=pool_size, D=1, residuals=True,
                groups=self.f1, pool_size=pool_size, norm=layer_norm_type)
            )
        
        # Per-layer cross-attention for Y-side metadata (after each deconv layer)
        self.ymd_cross_attn_layers = nn.ModuleList([
            MetadataCrossAttention(
                latent_dim=reverse_conv_channels[i + 1] if i + 1 < n_cnn_layers else int(reverse_conv_channels[i] / expansion_factor),
                num_heads=1,
                num_assays=signal_dim,
                num_sequencing_platforms=num_sequencing_platforms,
                num_runtypes=num_runtypes
            ) for i in range(n_cnn_layers)
        ])
    
    def forward(self, src, y_metadata):
        # Apply deconv with per-layer metadata injection
        src = src.permute(0, 2, 1)  # to N, F2, L'
        for i, dconv in enumerate(self.deconv):
            src = dconv(src)
            # Apply metadata cross-attention after each deconv layer
            src = src.permute(0, 2, 1)  # to N, L', C
            src = self.ymd_cross_attn_layers[i](y_metadata, src)
            src = src.permute(0, 2, 1)  # back to N, C, L'
        src = src.permute(0, 2, 1)  # final permute to N, L, F1
        return src    

class CANDI_DNA_Encoder(nn.Module):
    def __init__(self, 
        signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, nhead, n_sab_layers, pool_size=2, 
        dropout=0.1, context_length=1600, pos_enc="relative", expansion_factor=3, num_sequencing_platforms=10, 
        num_runtypes=2, norm="batch", attention_type="dual"):

        super(CANDI_DNA_Encoder, self).__init__()

        self.pos_enc = pos_enc
        self.l1 = context_length
        self.l2 = self.l1 // (pool_size**n_cnn_layers)
        
        self.f1 = signal_dim 
        self.f2 = (self.f1 * (expansion_factor**(n_cnn_layers)))
        self.f3 = self.f2 + metadata_embedding_dim
        d_model = self.f2
        self.latent_dim = self.f2

        # DNA_conv_channels = exponential_linspace_int(4, self.f2, n_cnn_layers+3)
        # DNA_kernel_size = [conv_kernel_size for _ in range(n_cnn_layers+2)]

        # --- REDESIGN: Wide Stem & Biophysical Kernels ---
        # 1. Wide Stem: Start with 64 channels instead of 4 to prevent feature collapse.
        #    This moves closer to Borzoi's "High-Capacity" philosophy.
        start_channels = 64 
        
        tower_channels = exponential_linspace_int(start_channels, self.f2, n_cnn_layers + 2)
        DNA_conv_channels = [4] + tower_channels
        
        # 2. Biophysical Kernels: 
        #    Layer 0: 15bp to capture complete TF motifs (The "Motif Scanner").
        #    Layer 1+: 5bp to capture motif syntax/arrangement.
        DNA_kernel_size = [15] + [5 for _ in range(n_cnn_layers+1)]
        # -------------------------------------------------

        self.convEncDNA = nn.ModuleList(
            [ConvTower(
                DNA_conv_channels[i], DNA_conv_channels[i + 1],
                DNA_kernel_size[i], S=1, D=1,
                pool_type="max", residuals=True, SE=False,
                groups=1, pool_size=5 if i >= n_cnn_layers else pool_size, norm=norm) for i in range(n_cnn_layers + 2)])

        conv_channels = [(self.f1)*(expansion_factor**l) for l in range(n_cnn_layers)]
        reverse_conv_channels = [expansion_factor * x for x in conv_channels[::-1]]
        # conv_kernel_size_list = [conv_kernel_size for _ in range(n_cnn_layers)]

        # --- REDESIGN: Signal Features Only ---
        # 1. Kernel Size: 9 (225bp) 
        #    At 25bp resolution, a kernel of 9 covers ~225bp. This matches the typical
        #    width of a ChIP-seq peak or ATAC domain, allowing the model to see the 
        #    "shape" of the feature rather than just local noise (kernel 3).
        conv_kernel_size_list = [9] + [5 for _ in range(n_cnn_layers - 1)]
        # --------------------------------------
        
        self.convEnc = nn.ModuleList(
            [ConvTower(
                conv_channels[i], conv_channels[i + 1] if i + 1 < n_cnn_layers else expansion_factor * conv_channels[i],
                conv_kernel_size_list[i], S=1, D=1,
                pool_type="max", residuals=True,
                groups=self.f1, SE=False,
                pool_size=pool_size, norm=norm) for i in range(n_cnn_layers)])
        
        # Per-layer cross-attention for X-side metadata (after each conv layer)
        self.xmd_cross_attn_layers = nn.ModuleList([
            MetadataCrossAttention(
                latent_dim=conv_channels[i + 1] if i + 1 < n_cnn_layers else expansion_factor * conv_channels[i],
                num_heads=1,
                num_assays=signal_dim,
                num_sequencing_platforms=num_sequencing_platforms,
                num_runtypes=num_runtypes
            ) for i in range(n_cnn_layers)
        ])

        # Store attention type for reference
        self.attention_type = attention_type

        # Validate x-transformers availability if needed
        if attention_type == "xtransformers" and not XTRANSFORMERS_AVAILABLE:
            raise ImportError(
                "x-transformers library is required for attention_type='xtransformers' but not installed. "
                "Install it with: pip install x-transformers"
            )

        # Hybrid fusion: cross-attention + concat-linear for best of both
        self.fusion = HybridFusion(d_model=self.f2, nhead=nhead, dropout=dropout)

        # Initialize transformer encoder based on attention type
        if attention_type == "xtransformers":
            self.transformer_encoder = nn.ModuleList([
                XTransformerEncoderBlock(
                    d_model=self.f2, 
                    num_heads=nhead, 
                    seq_length=self.l2, 
                    dropout=dropout
                ) for _ in range(n_sab_layers)])
                
        else:  # "dual" or default
            self.transformer_encoder = nn.ModuleList([
                DualAttentionEncoderBlock(
                    self.f2, nhead, self.l2, dropout=dropout, 
                    max_distance=self.l2, pos_encoding_type=pos_enc, max_len=self.l2
                ) for _ in range(n_sab_layers)])

    def forward(self, src, seq, x_metadata):
        if len(seq.shape) != len(src.shape):
            seq = seq.unsqueeze(0).expand(src.shape[0], -1, -1)

        seq = seq.permute(0, 2, 1)  # to N, 4, 25*L
        seq = seq.float()

        ### DNA CONV ENCODER ###
        for seq_conv in self.convEncDNA:
            seq = seq_conv(seq)
        seq = seq.permute(0, 2, 1)  # to N, L', F2

        ### SIGNAL CONV ENCODER WITH PER-LAYER METADATA INJECTION ###
        src = src.permute(0, 2, 1) # to N, F1, L
        for i, conv in enumerate(self.convEnc):
            src = conv(src)
            # Apply metadata cross-attention after each conv layer
            src = src.permute(0, 2, 1)  # to N, L', C
            src = self.xmd_cross_attn_layers[i](x_metadata, src)
            src = src.permute(0, 2, 1)  # back to N, C, L'
        src = src.permute(0, 2, 1)  # final permute to N, L', F2

        ### CROSS-ATTENTION FUSION (signal queries DNA) ###
        src = self.fusion(signal=src, dna=seq)  # [N, L', F2]

        ### TRANSFORMER ENCODER ###
        for enc in self.transformer_encoder:
            src = enc(src)

        return src

class CANDI(nn.Module):
    def __init__(self, signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, nhead,
        n_sab_layers, pool_size=2, dropout=0.1, context_length=1600, pos_enc="relative", 
        expansion_factor=3, separate_decoders=True, num_sequencing_platforms=10, num_runtypes=2, 
        norm="batch", attention_type="dual", output_ff=False):
        super(CANDI, self).__init__()

        self.pos_enc = pos_enc
        self.separate_decoders = separate_decoders
        self.l1 = context_length
        self.l2 = self.l1 // (pool_size**n_cnn_layers)
        
        self.f1 = signal_dim 
        self.f2 = (self.f1 * (expansion_factor**(n_cnn_layers)))
        self.f3 = self.f2 + metadata_embedding_dim
        self.d_model = self.latent_dim = self.f2

        self.encoder = CANDI_DNA_Encoder(signal_dim+1, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, nhead,
            n_sab_layers, pool_size, dropout, context_length, pos_enc, expansion_factor, num_sequencing_platforms, num_runtypes, norm, attention_type=attention_type)

        self.latent_projection = nn.Linear(
            ((signal_dim+1) * (expansion_factor**(n_cnn_layers))), 
            signal_dim * (expansion_factor**(n_cnn_layers)) 
            )
        
        if self.separate_decoders:
            self.count_decoder = CANDI_Decoder(signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, context_length, pool_size, expansion_factor, num_sequencing_platforms, num_runtypes, norm)
            self.pval_decoder = CANDI_Decoder(signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, context_length, pool_size, expansion_factor, num_sequencing_platforms, num_runtypes, norm)
            self.peak_decoder = CANDI_Decoder(signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, context_length, pool_size, expansion_factor, num_sequencing_platforms, num_runtypes, norm)
        else:
            self.decoder = CANDI_Decoder(signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, context_length, pool_size, expansion_factor, num_sequencing_platforms, num_runtypes, norm)

        self.neg_binom_layer = NegativeBinomialLayer(self.f1, self.f1, FF=output_ff)
        self.gaussian_layer = GaussianLayer(self.f1, self.f1, FF=output_ff)
        self.peak_layer = PeakLayer(self.f1, self.f1, FF=output_ff)
    
    def encode(self, src, seq, x_metadata, apply_arcsinh_transform=True):
        """Encode input data into latent representation.
        
        Args:
            src: Source data tensor
            seq: Sequence data
            x_metadata: Metadata tensor
            apply_arcsinh_transform: If True, apply arcsinh transformation to non-missing values in src
        """
        src = torch.where(src == -2, torch.tensor(-1, device=src.device), src)
        x_metadata = torch.where(x_metadata == -2, torch.tensor(-1, device=x_metadata.device), x_metadata)
        
        # Apply arcsinh transformation to non-missing values if requested
        if apply_arcsinh_transform:
            mask = src != -1
            src = torch.where(mask, torch.arcsinh(src), src)
        
        z = self.encoder(src, seq, x_metadata)
        return z
    
    def decode(self, z, y_metadata):
        """Decode latent representation into predictions."""
        y_metadata = torch.where(y_metadata == -2, torch.tensor(-1, device=y_metadata.device), y_metadata)
        
        if self.separate_decoders:
            count_decoded = self.count_decoder(z, y_metadata)
            pval_decoded = self.pval_decoder(z, y_metadata)
            peak_decoded = self.peak_decoder(z, y_metadata)

            p, n = self.neg_binom_layer(count_decoded)
            mu, var = self.gaussian_layer(pval_decoded)
            peak = self.peak_layer(peak_decoded)  # Use count decoder for peak prediction
        else:
            decoded = self.decoder(z, y_metadata)

            p, n = self.neg_binom_layer(decoded)
            mu, var = self.gaussian_layer(decoded)
            peak = self.peak_layer(decoded)
            
        return p, n, mu, var, peak

    def forward(self, src, seq, x_metadata, y_metadata, availability=None, return_z=False):
        z = self.encode(src, seq, x_metadata)

        z = self.latent_projection(z)

        p, n, mu, var, peak = self.decode(z, y_metadata)
        
        if return_z:
            return p, n, mu, var, peak, z
        else:
            return p, n, mu, var, peak

class CANDI_UNET(CANDI):
    """
    CANDI with U-Net skip connections.
    
    Identical architecture to CANDI, but adds skip connections from the signal encoder
    to the decoder. The skip connections:
    1. Are computed with the same operations as the encoder (including arcsinh transform
       and per-layer metadata cross-attention)
    2. Are added to the decoder before each deconv layer
    3. Preserve the decoder's per-layer metadata cross-attention
    """
    def __init__(self, signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers,
                 nhead, n_sab_layers, pool_size=2, dropout=0.1, context_length=1600,
                 pos_enc="relative", expansion_factor=3, separate_decoders=True, 
                 num_sequencing_platforms=10, num_runtypes=4, norm="batch", attention_type="dual",
                 output_ff=False):
        super(CANDI_UNET, self).__init__(
            signal_dim, metadata_embedding_dim,
            conv_kernel_size, n_cnn_layers,
            nhead, n_sab_layers,
            pool_size, dropout,
            context_length, pos_enc,
            expansion_factor,
            separate_decoders, num_sequencing_platforms, num_runtypes, norm, attention_type,
            output_ff=output_ff
        )

    def _compute_skips(self, src, x_metadata):
        """
        Compute skip connections from signal encoder with per-layer metadata injection.
        
        This mirrors EXACTLY what happens in CANDI_DNA_Encoder.forward() for the signal path:
        1. Replace cloze_mask with missing_mask
        2. Apply arcsinh transformation
        3. Apply convolutions with per-layer metadata cross-attention
        
        Args:
            src: Input signal tensor [B, L, F+1] (includes control)
            x_metadata: Input metadata tensor [B, 4, F+1]
            
        Returns:
            List of skip tensors, one per conv layer, in encoder order (shallow to deep)
        """
        # Match encoder preprocessing: replace cloze_mask (-2) with missing_mask (-1)
        src = torch.where(src == -2, torch.tensor(-1, device=src.device), src)
        x_metadata = torch.where(x_metadata == -2, torch.tensor(-1, device=x_metadata.device), x_metadata)
        
        # Match encoder preprocessing: apply arcsinh transformation to non-missing values
        mask = src != -1
        src = torch.where(mask, torch.arcsinh(src), src)
        
        # Process through signal convolutions (matching encoder.forward signal path)
        x = src.permute(0, 2, 1)  # (N, F+1, L)
        skips = []
        
        for i, conv in enumerate(self.encoder.convEnc):
            x = conv(x)
            # Apply metadata cross-attention after each conv layer (matching encoder)
            x = x.permute(0, 2, 1)  # to N, L', C
            x = self.encoder.xmd_cross_attn_layers[i](x_metadata, x)
            x = x.permute(0, 2, 1)  # back to N, C, L'
            skips.append(x)
        
        return skips

    def _unet_decode(self, z, y_metadata, skips, decoder):
        """
        Decode with U-Net skip connections and per-layer metadata injection.
        
        This mirrors EXACTLY what happens in CANDI_Decoder.forward(), but adds
        skip connections before each deconv layer:
        1. Add skip connection (additive, matching resolution from encoder)
        2. Apply deconv
        3. Apply per-layer metadata cross-attention
        
        Args:
            z: Latent tensor [B, L', D]
            y_metadata: Output metadata tensor [B, 4, F]
            skips: List of skip tensors from _compute_skips
            decoder: The CANDI_Decoder to use
            
        Returns:
            Decoded tensor [B, L, F]
        """
        x = z.permute(0, 2, 1)  # (N, C, L')

        for i, dconv in enumerate(decoder.deconv):
            # Get matching resolution skip from encoder (reverse order: deep to shallow)
            skip = skips[-(i + 1)]
            
            # Handle dimension mismatch: encoder has F+1 (with control), decoder has F
            # Skip shape: (N, C_enc, L), x shape: (N, C_dec, L)
            # C_enc includes control assay, C_dec does not
            if skip.shape[1] != x.shape[1]:
                skip = skip[:, :x.shape[1], :]
            
            # Add skip connection before deconv (additive residual)
            x = x + skip
            
            # Apply deconv
            x = dconv(x)
            
            # Apply metadata cross-attention after each deconv layer (matching decoder)
            x = x.permute(0, 2, 1)  # to N, L', C
            x = decoder.ymd_cross_attn_layers[i](y_metadata, x)
            x = x.permute(0, 2, 1)  # back to N, C, L'

        x = x.permute(0, 2, 1)  # (N, L, F)
        return x

    def forward(self, src, seq, x_metadata, y_metadata, availability=None, return_z=False):
        """
        Forward pass with U-Net skip connections.
        
        Note: Skip connections are computed via a separate pass through the signal encoder.
        This is necessary because the main encode() fuses signal+DNA, but skips should
        only contain signal features (DNA context is added via cross-attention in encoder).
        """
        # Compute skip features from signal branch with metadata injection
        skips = self._compute_skips(src, x_metadata)
        
        # Standard encode (fuses seq + signal + metadata via HybridFusion + Transformer)
        z = self.encode(src, seq, x_metadata)
        z = self.latent_projection(z)

        # UNet-style decode with skip connections for each task
        if self.separate_decoders:
            count_decoded = self._unet_decode(z, y_metadata, skips, self.count_decoder)
            pval_decoded = self._unet_decode(z, y_metadata, skips, self.pval_decoder)
            peak_decoded = self._unet_decode(z, y_metadata, skips, self.peak_decoder)
        else:
            decoded = self._unet_decode(z, y_metadata, skips, self.decoder)
            count_decoded = pval_decoded = peak_decoded = decoded
        
        p, n = self.neg_binom_layer(count_decoded)
        mu, var = self.gaussian_layer(pval_decoded)
        peak = self.peak_layer(peak_decoded)

        if return_z:
            return p, n, mu, var, peak, z
        return p, n, mu, var, peak

#========================================================================================================#
#===========================================Building Blocks==============================================#
#========================================================================================================#

# ---------------------------
# Metadata Cross-Attention
# ---------------------------
class MetadataCrossAttention(nn.Module):
    """
    Per-assay cross-attention where each assay's metadata queries its own spatial features.
    Includes rich per-field metadata embedding for better OOD robustness.
    """
    def __init__(self, latent_dim, num_heads=4, num_assays=35, 
                 num_sequencing_platforms=10, num_runtypes=2):
        super().__init__()
        self.num_assays = num_assays
        self.latent_dim = latent_dim
        self.per_assay_dim = latent_dim // num_assays
        self.num_sequencing_platforms = num_sequencing_platforms
        self.num_runtypes = num_runtypes
        
        assert latent_dim % num_assays == 0, \
            f"latent_dim {latent_dim} must be divisible by num_assays {num_assays}"
        
        # Rich metadata embedding per field
        # Use max to ensure at least 1 dimension per field
        # field_embed_dim = max(1, self.per_assay_dim // 4)
        field_embed_dim = max(16, self.per_assay_dim)
        self.field_embed_dim = field_embed_dim
        
        # Continuous features: depth and read_length
        self.depth_proj = nn.Sequential(
            nn.Linear(1, field_embed_dim),
        )
        
        self.read_length_proj = nn.Sequential(
            nn.Linear(1, field_embed_dim),
        )
        
        # Categorical features: platform and runtype
        # Add 1 extra token for missing/cloze (-1)
        self.platform_embedding = nn.Embedding(
            num_sequencing_platforms + 1,  # +1 for special token
            field_embed_dim
        )
        
        self.runtype_embedding = nn.Embedding(
            num_runtypes + 1,  # +1 for special token
            field_embed_dim
        )
        
        # Final projection to query dimension
        self.metadata_fusion = nn.Sequential(
            nn.Linear(4 * field_embed_dim, self.per_assay_dim)
            )
        
        # single-head cross-attention (batched over all assays)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.per_assay_dim,
            num_heads=1, # hardcoded to 1 for now
            batch_first=True
        )
        
        # FiLM-style conditioning: separate scale and shift projections
        self.scale_proj = nn.Sequential(
            nn.Linear(self.per_assay_dim, self.per_assay_dim)
        )
        
        self.shift_proj = nn.Sequential(
            nn.Linear(self.per_assay_dim, self.per_assay_dim)
        )
        
        # Feed-forward network (like in transformers) for richer non-linear processing
        # Expands to 4x then projects back
        ffn_hidden_dim = self.per_assay_dim * 4
        self.ffn = nn.Sequential(
            nn.Linear(self.per_assay_dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(ffn_hidden_dim, self.per_assay_dim),
            nn.Dropout(0.1)
        )
        self.ffn_norm = nn.LayerNorm(self.per_assay_dim)

    def _embed_metadata(self, metadata):
        """
        Process and embed metadata fields.
        
        Args:
            metadata: [B, 4, F] - metadata for F assays
                      [depth, platform, read_length, runtype]
        Returns:
            metadata_queries: [B, F, per_assay_dim]
        """
        B, _, F = metadata.shape
        
        # Extract each metadata field
        depth = metadata[:, 0, :].float()  # [B, F]
        platform = metadata[:, 1, :].long()  # [B, F]
        read_length = metadata[:, 2, :].float()  # [B, F]
        runtype = metadata[:, 3, :].long()  # [B, F]
        
        # Handle special tokens for categorical features
        # Map -1 (missing/cloze) -> num_classes
        platform = torch.where(platform == -1, 
                              torch.full_like(platform, self.num_sequencing_platforms),
                              platform)
        
        runtype = torch.where(runtype == -1, 
                             torch.full_like(runtype, self.num_runtypes),
                             runtype)
        
        depth_embed = self.depth_proj(depth.unsqueeze(-1))  # [B, F, field_dim]
        platform_embed = self.platform_embedding(platform)  # [B, F, field_dim]
        read_length_embed = self.read_length_proj(read_length.unsqueeze(-1))  # [B, F, field_dim]
        runtype_embed = self.runtype_embedding(runtype)  # [B, F, field_dim]
        
        # Concatenate and fuse
        metadata_concat = torch.cat([
            depth_embed, 
            platform_embed, 
            read_length_embed, 
            runtype_embed
        ], dim=-1)  # [B, F, 4*field_dim]
        
        metadata_queries = self.metadata_fusion(metadata_concat)  # [B, F, per_assay_dim]
        return metadata_queries

    def forward(self, metadata, latent):
        """
        Args:
            metadata: [B, 4, F] - metadata for F assays
            latent: [B, L, D] where D = F * per_assay_dim
        Returns:
            [B, L, D] - per-assay conditioned latent
        """
        B, L, D = latent.shape
        F = metadata.shape[2]
        d = self.per_assay_dim
        
        # Reshape latent to expose per-assay structure
        latent_per_assay = latent.view(B, L, F, d)
        
        # Create rich queries from metadata
        metadata_queries = self._embed_metadata(metadata)  # [B, F, d]
        
        # Batch all assays for parallel attention
        queries_all = metadata_queries.reshape(B*F, 1, d)  # [B*F, 1, d]
        kv_all = latent_per_assay.transpose(1, 2).reshape(B*F, L, d)  # [B*F, L, d]
        
        # Cross-attention: each assay's metadata queries its spatial features
        attended_all, _ = self.cross_attn(
            query=queries_all,
            key=kv_all,
            value=kv_all
        )  # [B*F, 1, d]
        
        attended = attended_all.view(B, F, d)  # [B, F, d]
        
        # Apply feed-forward network with residual connection (standard transformer practice)
        # This allows richer non-linear processing of the attended features
        attended_ffn = self.ffn_norm(attended + self.ffn(attended))  # [B, F, d]
        
        # FiLM-style conditioning: generate scale and shift parameters from metadata-attended features
        scale = self.scale_proj(attended_ffn)  # [B, F, d]
        shift = self.shift_proj(attended_ffn)  # [B, F, d]
        
        # Broadcast scale and shift to spatial dimensions
        scale_spatial = scale.unsqueeze(1).expand(-1, L, -1, -1)  # [B, L, F, d]
        shift_spatial = shift.unsqueeze(1).expand(-1, L, -1, -1)  # [B, L, F, d]
        
        # Apply FiLM conditioning to the latent features
        # Clamp scale to prevent overflow/NaNs (e.g. exp(10) explodes in fp16)
        scale_spatial = torch.clamp(scale_spatial, min=-4.0, max=4.0) 
        modulated_latent = torch.exp(scale_spatial) * latent_per_assay + shift_spatial
        
        # Reshape back to [B, L, D]
        return modulated_latent.view(B, L, D)

# ---------------------------
# Absolute Positional Encoding
# ---------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Creates positional encodings of shape (1, max_len, d_model).
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term[:pe.size(2)//2])  # odd indices
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, L, d_model)
        Returns:
            x with added positional encoding for positions [0, L)
        """
        L = x.size(1)
        return x + self.pe[:, :L]

# ---------------------------
# Relative Positional Bias Module
# ---------------------------
class RelativePositionBias(nn.Module):
    def __init__(self, num_heads, max_distance):
        """
        Args:
            num_heads (int): number of attention heads.
            max_distance (int): maximum sequence length to support.
        """
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.relative_bias = nn.Parameter(torch.zeros(2 * max_distance - 1, num_heads))
        nn.init.trunc_normal_(self.relative_bias, std=0.02)

    def forward(self, L):
        """
        Args:
            L (int): current sequence length.
        Returns:
            Tensor of shape (num_heads, L, L) to add as bias.
        """
        device = self.relative_bias.device
        pos = torch.arange(L, device=device)
        rel_pos = pos[None, :] - pos[:, None]  # shape (L, L)
        rel_pos = rel_pos + self.max_distance - 1  # shift to [0, 2*max_distance-2]
        bias = self.relative_bias[rel_pos]  # (L, L, num_heads)
        bias = bias.permute(2, 0, 1)  # (num_heads, L, L)
        return bias

# ---------------------------
# XTransformer Encoder Block with RoPE
# ---------------------------
class XTransformerEncoderBlock(nn.Module):
    """Standard transformer with RoPE using x-transformers library."""
    def __init__(self, d_model, num_heads, seq_length, dropout=0.1, 
                 ff_mult=4, **kwargs):
        super().__init__()
        
        if not XTRANSFORMERS_AVAILABLE:
            raise ImportError(
                "x-transformers library is required but not installed. "
                "Install it with: pip install x-transformers"
            )
        
        self.encoder = XTransformerEncoder(
            dim=d_model,
            depth=1,  # Single block (will be stacked in ModuleList)
            heads=num_heads,
            use_rmsnorm=True,
            ff_mult=ff_mult,
            attn_dropout=dropout,
            ff_dropout=dropout,
            rotary_pos_emb=True  # Enable RoPE (let x-transformers auto-calculate rotary_emb_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, L, d_model)
        Returns:
            Tensor of shape (B, L, d_model)
        """
        return self.encoder(x)

# ---------------------------
# Cross-Attention Fusion with Zero-Init Projection
# ---------------------------
class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion with depthwise-preserved Q and full-DNA K/V.
    Uses PyTorch's optimized scaled_dot_product_attention.
    
    - Pre-normalization with RMSNorm for stable training
    - Q: depthwise projection preserves per-assay structure
    - K/V: full DNA projections so each head sees all DNA features
    """
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        # Pre-normalization (RMSNorm for stability)
        # Use PyTorch's native RMSNorm (available in PyTorch 2.4+)
        self.signal_norm = nn.RMSNorm(d_model)
        self.dna_norm = nn.RMSNorm(d_model)
        
        # Q: depthwise projection (preserves per-assay structure)
        self.q_proj = nn.Conv1d(d_model, d_model, kernel_size=1, groups=nhead)
        
        # K/V: full DNA projections (each head sees all DNA features)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout
        
        # Zero-init for stable training
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        
    def forward(self, signal, dna):
        """
        Args:
            signal: (B, L, d_model) - query source (epigenomic features)
            dna: (B, L, d_model) - key/value source (DNA features)
        Returns:
            (B, L, d_model) - fused representation
        """
        B, L, d = signal.shape
        
        # Pre-normalize inputs
        signal_normed = self.signal_norm(signal)
        dna_normed = self.dna_norm(dna)
        
        # Q: depthwise projection on normalized signal
        Q = self.q_proj(signal_normed.transpose(1, 2)).transpose(1, 2)
        Q = Q.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        
        # K/V: full DNA projected and split into heads
        K = self.k_proj(dna_normed).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        V = self.v_proj(dna_normed).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        
        # PyTorch optimized attention (Flash Attention when available)
        attn_output = F.scaled_dot_product_attention(
            Q, K, V,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False
        )
        
        out = attn_output.transpose(1, 2).contiguous().view(B, L, d)
        return signal + self.out_proj(out)

class CrossAttentionFusionXT(nn.Module):
    """
    Cross-attention fusion using x-transformers library.
    Conventional cross-attention where signal attends to DNA features.
    
    Uses x-transformers CrossAttender for optimized attention with:
    - Rotary positional embeddings (optional)
    - Flash attention support
    - Pre-normalization
    """
    def __init__(self, d_model, nhead=4, dropout=0.1, use_rotary=True, depth=1):
        super().__init__()
        
        if not XTRANSFORMERS_AVAILABLE:
            raise ImportError("x-transformers is required for CrossAttentionFusionXT")
        
        self.d_model = d_model
        self.nhead = nhead
        
        # x-transformers CrossAttender handles cross-attention
        # CrossAttender wraps AttentionLayers with cross_attend=True
        self.cross_attn = CrossAttender(
            dim=d_model,
            depth=depth,
            heads=nhead,
            attn_dropout=dropout,
            ff_dropout=dropout,
            rotary_pos_emb=use_rotary,
            attn_flash=False,  # Use Flash Attention when available
            pre_norm=True,  # Pre-normalization for stable training
        )
        
    def forward(self, signal, dna):
        """
        Args:
            signal: (B, L, d_model) - query source (epigenomic features)
            dna: (B, L, d_model) - key/value source (DNA features)
        Returns:
            (B, L, d_model) - fused representation with residual connection
        """
        # CrossAttender: signal attends to dna context
        out = self.cross_attn(signal, context=dna)
        return out

# ---------------------------
# Hybrid Fusion (Cross-Attention + Concat-Linear)
# ---------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g

class HybridFusion(nn.Module):
    """
    Combines cross-attention (position-specific) with concat+linear (full feature access).
    
    - Cross-attention path: position-specific DNA context routing
    - Linear path: direct feature combination without bottleneck
    - Learnable alpha parameter controls the balance between pathways
    """
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        
        # Cross-attention path (position-specific routing)
        # self.cross_attn = CrossAttentionFusion(d_model, nhead, dropout)
        self.cross_attn = CrossAttentionFusionXT(d_model, nhead, dropout)
        
        # Linear path (direct feature combination, no bottleneck)
        try:
            self.linear_norm = nn.RMSNorm(2 * d_model)
        except AttributeError:
            self.linear_norm = RMSNorm(2 * d_model)
            
        self.linear_fusion = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
        )
        # Zero-init the linear output for stable training
        nn.init.zeros_(self.linear_fusion[0].weight)
        nn.init.zeros_(self.linear_fusion[0].bias)
        
        # Learnable mixing parameter
        # This biases the model to rely on concat+linear at first (alpha * attn + (1-alpha) * linear)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, signal, dna):
        """
        Args:
            signal: (B, L, d_model) - epigenomic features
            dna: (B, L, d_model) - DNA features
        Returns:
            (B, L, d_model) - fused representation
        """
        # Cross-attention path: position-specific DNA context
        attn_out = self.cross_attn(signal, dna)
        
        # Linear path: direct feature combination (normalized)
        concat_features = torch.cat([signal, dna], dim=-1)
        concat_normed = self.linear_norm(concat_features)
        linear_out = signal + self.linear_fusion(concat_normed)
        
        # Mix both pathways (sigmoid ensures alpha stays in [0, 1])
        alpha = torch.sigmoid(self.alpha)
        # print(alpha.item())
        return alpha * attn_out + (1 - alpha) * linear_out

# ---------------------------
# Dual Attention Encoder Block (Post-Norm)
# ---------------------------
class DualAttentionEncoderBlock(nn.Module):
    """
    Dual Attention Encoder Block with post-norm style.
    It has two parallel branches:
      - MHA1 (sequence branch): optionally uses relative or absolute positional encodings.
      - MHA2 (channel branch): operates along the channel dimension (no positional encoding).
    The outputs of the two branches are concatenated and fused via a FFN.
    Residual connections and layer norms are applied following the post-norm convention.
    """
    def __init__(self, d_model, num_heads, seq_length, dropout=0.1, 
                max_distance=128, pos_encoding_type="relative", max_len=5000):
        """
        Args:
            d_model (int): model (feature) dimension.
            num_heads (int): number of attention heads.
            seq_length (int): expected sequence length (used for channel branch).
            dropout (float): dropout rate.
            max_distance (int): max distance for relative bias.
            pos_encoding_type (str): "relative" or "absolute" for MHA1.
            max_len (int): max sequence length for absolute positional encoding.
        """
        super().__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.num_heads = num_heads
        self.pos_encoding_type = pos_encoding_type

        # Automatically determine the number of heads for each branch.
        self.num_heads_seq = get_divisible_heads(d_model, num_heads)
        self.num_heads_chan = get_divisible_heads(seq_length, num_heads)
        
        # Sequence branch (MHA1)
        if pos_encoding_type == "relative":
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)
            self.relative_bias = RelativePositionBias(num_heads, max_distance)
        elif pos_encoding_type == "absolute":
            # Use PyTorch's built-in MHA; we'll add absolute pos encodings.
            self.mha_seq = nn.MultiheadAttention(embed_dim=d_model, num_heads=self.num_heads_seq, 
                                                  dropout=dropout, batch_first=True)
            self.abs_pos_enc = SinusoidalPositionalEncoding(d_model, max_len)
        else:
            raise ValueError("pos_encoding_type must be 'relative' or 'absolute'")
            
        # Channel branch (MHA2)
        # We transpose so that channels (d_model) become sequence tokens.
        # We set embed_dim for channel attention to seq_length.
        self.mha_channel = nn.MultiheadAttention(embed_dim=seq_length, num_heads=self.num_heads_chan,
                                                  dropout=dropout, batch_first=True)
        
        # Fusion: concatenate outputs from both branches (dimension becomes 2*d_model)
        # and then use an FFN to map it back to d_model.
        self.ffn = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer Norms (applied after each sublayer, i.e., post-norm)
        self.norm_seq = nn.LayerNorm(d_model)
        self.norm_chan = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)

    def relative_multihead_attention(self, x):
        """
        Custom multi-head self-attention with relative positional bias.
        Args:
            x: Tensor of shape (B, L, d_model)
        Returns:
            Tensor of shape (B, L, d_model)
        """
        B, L, _ = x.shape
        head_dim = self.d_model // self.num_heads
        q = self.q_proj(x)  # (B, L, d_model)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # Reshape: (B, L, num_heads, head_dim) -> (B, num_heads, L, head_dim)
        q = q.view(B, L, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)  # (B, num_heads, L, L)
        bias = self.relative_bias(L)  # (num_heads, L, L)
        scores = scores + bias.unsqueeze(0)  # (B, num_heads, L, L)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        out = torch.matmul(attn_weights, v)  # (B, num_heads, L, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        out = self.out_proj(out)
        return out

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, L, d_model)
        Returns:
            Tensor of shape (B, L, d_model)
        """
        B, L, _ = x.shape
        
        # ----- Sequence Branch (MHA1) using post-norm -----
        if self.pos_encoding_type == "relative":
            # Compute sequence attention without pre-norm.
            seq_attn = self.relative_multihead_attention(x)  # (B, L, d_model)
        else:
            # Absolute positional encodings: add pos encoding and use default MHA.
            x_abs = self.abs_pos_enc(x)
            seq_attn, _ = self.mha_seq(x_abs, x_abs, x_abs)  # (B, L, d_model)
        # Add residual and then norm (post-norm)
        x_seq = self.norm_seq(x + seq_attn)  # (B, L, d_model)
        
        # ----- Channel Branch (MHA2) using post-norm -----
        # Transpose: (B, L, d_model) -> (B, d_model, L)
        x_trans = x.transpose(1, 2)
        # Apply channel attention (without pre-norm).
        chan_attn, _ = self.mha_channel(x_trans, x_trans, x_trans)  # (B, d_model, L)
        # Transpose back: (B, L, d_model)
        chan_attn = chan_attn.transpose(1, 2)
        # Add residual and norm
        x_chan = self.norm_chan(x + chan_attn)
        
        # ----- Fusion via FFN -----
        # Concatenate along feature dimension: (B, L, 2*d_model)
        fusion_input = torch.cat([x_seq, x_chan], dim=-1)
        ffn_out = self.ffn(fusion_input)  # (B, L, d_model)
        # Residual connection and final norm (post-norm)
        # out = self.norm_ffn(x + ffn_out)
        out = self.norm_ffn(x_seq + x_chan + ffn_out)
        return out

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization for 1D convolutions.
    
    Normalizes by RMS without mean centering. For input shape (B, C, L),
    normalizes across the channel dimension.
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # x shape: (B, C, L) for Conv1d
        # Transpose to (B, L, C) for normalization
        x = x.permute(0, 2, 1)
        # Compute RMS over the channel dimension
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x * rms * self.weight
        # Transpose back to (B, C, L)
        x = x.permute(0, 2, 1)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_C, out_C, W, S, D, norm, groups=1, apply_act=False):
        super(ConvBlock, self).__init__()
        self.normtype = norm
        self.apply_act = apply_act
        
        # Create conv layer
        if S == 1:
            padding_val = "same"
        else:
            padding_val = (D * (W - 1)) // 2

        self.conv = nn.Conv1d(
            in_C, out_C, kernel_size=W, dilation=D, stride=S, groups=groups, padding=padding_val)
        
        # Apply normalization
        if self.normtype == "weight":
            # WeightNorm wraps the conv layer itself
            self.conv = nn.utils.weight_norm(self.conv)
        elif self.normtype == "batch":
            self.norm = nn.BatchNorm1d(out_C)
        elif self.normtype == "layer":
            self.norm = nn.LayerNorm(out_C)
        elif self.normtype == "group":
            self.norm = nn.GroupNorm(groups, out_C)
        elif self.normtype == "instance":
            # Use affine=True and larger eps to avoid gradient issues
            self.norm = nn.InstanceNorm1d(out_C, affine=True, eps=1e-5)
        elif self.normtype == "rms":
            self.norm = RMSNorm(out_C)
    
    def forward(self, x):
        x = self.conv(x)
        
        # WeightNorm doesn't need activation normalization
        if self.normtype == "layer":
            x = x.permute(0, 2, 1)
            x = self.norm(x)
            x = x.permute(0, 2, 1)
        elif self.normtype in ["batch", "group", "instance", "rms"]:
            x = self.norm(x)
        
        if self.apply_act:
            x = F.gelu(x)
        
        return x

class ConvTower(nn.Module):
    def __init__(self, in_C, out_C, W, S=1, D=1, pool_type="max", residuals=True, groups=1, pool_size=2, SE=False, norm="batch"):
        super(ConvTower, self).__init__()
        
        if pool_type == "max" or pool_type == "attn" or pool_type == "avg":
            self.do_pool = True
        else:
            self.do_pool = False
        
        if pool_type == "attn":
            self.pool = SoftmaxPooling1D(pool_size)
        elif pool_type == "max":
            self.pool = nn.MaxPool1d(pool_size)
        elif pool_type == "avg":
            self.pool = nn.AvgPool1d(pool_size)
        
        self.conv1 = ConvBlock(in_C, out_C, W, S, D, norm=norm, groups=groups, apply_act=False)
        self.resid = residuals
        
        if self.resid:
            self.rconv = nn.Conv1d(in_C, out_C, kernel_size=1, stride=S, groups=groups)
    
    def forward(self, x):
        y = self.conv1(x)  # Output before activation
        
        if self.resid:
            y = y + self.rconv(x)
        
        y = F.gelu(y)  # Activation after residual
                
        if self.do_pool:
            y = self.pool(y)
        return y

class DeconvBlock(nn.Module):
    def __init__(self, in_C, out_C, W, S, D, norm, groups=1, apply_act=False):
        super(DeconvBlock, self).__init__()
        self.normtype = norm
        self.apply_act = apply_act
        
        # Create deconv layer
        padding = (W - 1) // 2
        output_padding = S - 1
        
        self.deconv = nn.ConvTranspose1d(
            in_C, out_C, kernel_size=W, dilation=D, stride=S,
            padding=padding, output_padding=output_padding, groups=groups)
        
        # Apply normalization
        if self.normtype == "weight":
            # WeightNorm wraps the deconv layer itself
            self.deconv = nn.utils.weight_norm(self.deconv)
        elif self.normtype == "batch":
            self.norm = nn.BatchNorm1d(out_C)
        elif self.normtype == "layer":
            self.norm = nn.LayerNorm(out_C)
        elif self.normtype == "group":
            self.norm = nn.GroupNorm(groups, out_C)
        elif self.normtype == "instance":
            # Use affine=True and larger eps to avoid gradient issues
            self.norm = nn.InstanceNorm1d(out_C, affine=True, eps=1e-5)
        elif self.normtype == "rms":
            self.norm = RMSNorm(out_C)
    
    def forward(self, x):
        x = self.deconv(x)
        
        # WeightNorm doesn't need activation normalization
        if self.normtype == "layer":
            x = x.permute(0, 2, 1)
            x = self.norm(x)
            x = x.permute(0, 2, 1)
        elif self.normtype in ["batch", "group", "instance", "rms"]:
            x = self.norm(x)
        
        if self.apply_act:
            x = F.gelu(x)
        
        return x

class DeconvTower(nn.Module):
    def __init__(self, in_C, out_C, W, S=1, D=1, residuals=True, groups=1, pool_size=2, norm="batch"):
        super(DeconvTower, self).__init__()
        
        self.deconv1 = DeconvBlock(in_C, out_C, W, S, D, norm=norm, groups=groups, apply_act=False)
        self.resid = residuals
        
        if self.resid:
            self.rdeconv = nn.ConvTranspose1d(in_C, out_C, kernel_size=1, stride=S, output_padding=S - 1, groups=groups)
    
    def forward(self, x):
        y = self.deconv1(x)  # Output before activation
        
        if self.resid:
            y = y + self.rdeconv(x)
        
        y = F.gelu(y)  # Activation after residual
        return y

class SE_Block_1D(nn.Module):
    """
    Squeeze-and-Excitation block for 1D convolutional layers.
    This module recalibrates channel-wise feature responses by modeling interdependencies between channels.
    """
    def __init__(self, c, r=8):
        super(SE_Block_1D, self).__init__()
        # Global average pooling for 1D
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        # Excitation network to produce channel-wise weights
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, recal=True):
        bs, c, l = x.shape  # Batch size, number of channels, length
        # Squeeze: Global average pooling to get the channel-wise statistics
        y = self.squeeze(x).view(bs, c)  # Shape becomes (bs, c)
        # Excitation: Fully connected layers to compute weights for each channel
        y = self.excitation(y).view(bs, c, 1)  # Shape becomes (bs, c, 1)
        # Recalibrate: Multiply the original input by the computed weights
        if recal:
            return x * y.expand_as(x)  # Shape matches (bs, c, l)
        else:
            return y.expand_as(x)  # Shape matches (bs, c, l)

class Sqeeze_Extend(nn.Module):
    def __init__(self, k=1):
        super(Sqeeze_Extend, self).__init__()
        self.k = k
        self.squeeze = nn.AdaptiveAvgPool1d(k)

    def forward(self, x):
        bs, c, l = x.shape  
        y = self.squeeze(x).view(bs, c, self.k)
        return y.expand_as(x)

class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position

        # Get the current device from embeddings_table
        device = self.embeddings_table.device

        # Move final_mat to the same device as embeddings_table
        final_mat = final_mat.to(device)

        embeddings = self.embeddings_table[final_mat]

        return embeddings

class RelativeMultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.max_relative_position = 2

        self.relative_position_k = RelativePosition(self.head_dim, self.max_relative_position)
        self.relative_position_v = RelativePosition(self.head_dim, self.max_relative_position)

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))#.to(device)
        
    def forward(self, query, key, value, mask = None):
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
        batch_size = query.shape[0]
        len_k = key.shape[1]
        len_q = query.shape[1]
        len_v = value.shape[1]

        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)

        r_q1 = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        r_k1 = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2)) 

        r_q2 = query.permute(1, 0, 2).contiguous().view(len_q, batch_size*self.n_heads, self.head_dim)
        r_k2 = self.relative_position_k(len_q, len_k)
        attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
        attn2 = attn2.contiguous().view(batch_size, self.n_heads, len_q, len_k)
        self.scale = self.scale.to(attn1.device)
        attn = (attn1 + attn2) / self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e10)

        attn = self.dropout(torch.softmax(attn, dim = -1))

        #attn = [batch size, n heads, query len, key len]
        r_v1 = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        weight1 = torch.matmul(attn, r_v1)
        r_v2 = self.relative_position_v(len_q, len_v)
        weight2 = attn.permute(2, 0, 1, 3).contiguous().view(len_q, batch_size*self.n_heads, len_k)
        weight2 = torch.matmul(weight2, r_v2)
        weight2 = weight2.transpose(0, 1).contiguous().view(batch_size, self.n_heads, len_q, self.head_dim)

        x = weight1 + weight2
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x

class RelativeEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, feed_forward_hidden, dropout):
        super().__init__()
        
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)

        self.relative_multihead_attn = RelativeMultiHeadAttentionLayer(d_model, heads, dropout)
        self.positionwise_feedforward = nn.Sequential(
            nn.Linear(d_model, feed_forward_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feed_forward_hidden, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, src len]

        # Self-attention
        _src = self.relative_multihead_attn(src, src, src, src_mask)
        
        # Residual connection and layer norm
        src = self.layer_norm_1(src + self.dropout(_src))

        # Position-wise feedforward
        _src = self.positionwise_feedforward(src)

        # Another residual connection and layer norm
        src = self.layer_norm_2(src + self.dropout(_src))

        return src

class RelativeDecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout):
        super().__init__()

        self.layer_norm_cross_attn = nn.LayerNorm(hid_dim)
        self.layer_norm_ff = nn.LayerNorm(hid_dim)

        self.encoder_attention = RelativeMultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = nn.Sequential(
            nn.Linear(hid_dim, pf_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pf_dim, hid_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, src_mask=None):
        # trg = [batch size, trg len, hid dim]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, trg len]
        # src_mask = [batch size, src len]

        # Encoder-decoder attention
        query = trg
        key = enc_src
        value = enc_src

        # Using the decoder input as the query, and the encoder output as key and value
        _trg = self.encoder_attention(query, key, value, src_mask)

        # Residual connection and layer norm
        trg = self.layer_norm_cross_attn(trg + self.dropout(_trg))

        # Positionwise feedforward
        _trg = self.positionwise_feedforward(trg)

        # Residual connection and layer norm
        trg = self.layer_norm_ff(trg + self.dropout(_trg))

        return trg

class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_hidden_layers):
        super(FeedForwardNN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        
        # Input Layer
        self.hidden_layers.append(nn.Linear(input_size, hidden_size))
        
        # Hidden Layers
        for _ in range(n_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        
        # Output Layer
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # Activation Function
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Pass through each layer
        for hidden_layer in self.hidden_layers:
            x = self.relu(hidden_layer(x))
        
        x = self.output_layer(x)
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        # Use the full div_term for both even and odd indices, handling odd d_model
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term[:pe.size(2)//2])  # Ensure matching size

        self.register_buffer('pe', pe.permute(1, 0, 2))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

#========================================================================================================#
#========================================= Negative Binomial ============================================#
#========================================================================================================#

class NegativeBinomialLayer(nn.Module):
    def __init__(self, input_dim, output_dim, FF=False, eps=1e-6):
        super(NegativeBinomialLayer, self).__init__()
        self.FF = FF
        self.eps = eps  # Small constant for numerical stability
        
        if self.FF:
            self.feed_forward = FeedForwardNN(input_dim, input_dim, input_dim, n_hidden_layers=2)
        
        # 1. Head for the Mean (mu)
        self.linear_mean = nn.Linear(input_dim, output_dim)
        
        # 2. Head for the Dispersion (n) - predict n directly for stability
        # n controls overdispersion: large n -> Poisson-like, small n -> overdispersed
        self.linear_n = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        if self.FF:
            x = self.feed_forward(x)
        
        # Predict Mean (mu) - must be positive
        mu_logits = self.linear_mean(x)
        mu = F.softplus(mu_logits) + self.eps

        # Predict n (total_count/dispersion) directly - must be positive
        n_logits = self.linear_n(x)
        n = F.softplus(n_logits) + self.eps

        # Convert to p using the codebase convention: mean = n(1-p)/p
        # Solving for p: p = n / (n + mu)
        p = n / (n + mu)

        # Return p, n to match the existing interface of the codebase
        return p, n

class GaussianLayer(nn.Module):
    def __init__(self, input_dim, output_dim, FF=False):
        super(GaussianLayer, self).__init__()

        self.FF = FF
        if self.FF:
            self.feed_forward = FeedForwardNN(input_dim, input_dim, input_dim, n_hidden_layers=2)

        # Linear layers with controlled initialization
        self.linear_mu = nn.Linear(input_dim, output_dim)
        self.linear_var = nn.Linear(input_dim, output_dim)
        

    def forward(self, x):
        if self.FF:
            x = self.feed_forward(x)

        mu_logits = self.linear_mu(x)
        mu = F.softplus(mu_logits)
        
        var_logits = self.linear_var(x)
        var = F.softplus(var_logits)

        return mu, var

class PeakLayer(nn.Module):
    def __init__(self, input_dim, output_dim, FF=False):
        super(PeakLayer, self).__init__()

        self.FF = FF
        if self.FF:
            self.feed_forward = FeedForwardNN(input_dim, input_dim, input_dim, n_hidden_layers=2)
        
        # Linear layer with controlled initialization
        self.linear_peak = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        if self.FF:
            x = self.feed_forward(x)

        peak_logits = self.linear_peak(x)
        peak = torch.sigmoid(peak_logits)

        return peak

#========================================================================================================#
#=============================================== Main ===================================================#
#========================================================================================================#

if __name__ == "__main__":
    hyper_parameters1678 = {
        "data_path": "/project/compbio-lab/EIC/training_data/",
        "input_dim": 35,
        "dropout": 0.05,
        "nhead": 4,
        "d_model": 192,
        "nlayers": 3,
        "epochs": 4,
        "mask_percentage": 0.2,
        "chunk": True,
        "context_length": 200,
        "batch_size": 200,
        "learning_rate": 0.0001
    }  

    if sys.argv[1] == "epd16":
        train_epidenoise16(
            hyper_parameters1678, 
            checkpoint_path=None, 
            start_ds=0)

    elif sys.argv[1] == "epd17":
        train_epidenoise17(
            hyper_parameters1678, 
            checkpoint_path=None, 
            start_ds=0)

    elif sys.argv[1] == "epd18":
        train_epidenoise18(
            hyper_parameters1678, 
            checkpoint_path=None, 
            start_ds=0)

    elif sys.argv[1] == "epd20":
        hyper_parameters20 = {
            "data_path": "/project/compbio-lab/EIC/training_data/",
            "input_dim": 35,
            "dropout": 0.05,
            "nhead": 4,
            "d_model": 128,
            "nlayers": 2,
            "epochs": 10,
            "mask_percentage": 0.3,
            "kernel_size": [1, 3, 3],
            "conv_out_channels": [64, 64, 128],
            "dilation":1,
            "context_length": 800,
            "batch_size": 100,
            "learning_rate": 0.0001,
        }
        train_epidenoise20(
            hyper_parameters20, 
            checkpoint_path=None, 
            start_ds=0)
    
    elif sys.argv[1] == "epd21":
        hyper_parameters21 = {
            "data_path": "/project/compbio-lab/EIC/training_data/",
            "input_dim": 35,
            "dropout": 0.1,
            "nhead": 4,
            "d_model": 256,
            "nlayers": 2,
            "epochs": 2,
            "kernel_size": [1, 9, 7, 5],
            "conv_out_channels": [64, 128, 192, 256],
            "dilation":1,
            "context_length": 800,
            "learning_rate": 1e-3,
        }
        train_epidenoise21(
            hyper_parameters21, 
            checkpoint_path=None, 
            start_ds=0)
    
    elif sys.argv[1] == "epd22":
        hyper_parameters22 = {
            "data_path": "/project/compbio-lab/EIC/training_data/",
            "input_dim": 35,
            "dropout": 0.01,
            "context_length": 200,
            
            "kernel_size": [1, 3, 3, 3],
            "conv_out_channels": [128, 144, 192, 256],
            "dilation":1,

            "nhead": 2,
            "n_enc_layers": 1,
            "n_dec_layers": 1,
            
            "mask_percentage":0.15,
            "batch_size":400,
            "epochs": 10,
            "outer_loop_epochs":2,
            "learning_rate": 1e-4
        }
        train_epidenoise22(
            hyper_parameters22, 
            checkpoint_path=None, 
            start_ds=0)

    elif sys.argv[1] == "epd30a":
        
        hyper_parameters30a = {
            "data_path": "/project/compbio-lab/encode_data/",
            "input_dim": 45,
            "metadata_embedding_dim": 45,
            "dropout": 0.01,
            "nhead": 5,
            "d_model": 450,
            "nlayers": 2,
            "epochs": 1,
            "inner_epochs": 10,
            "mask_percentage": 0.15,
            "context_length": 200,
            "batch_size": 50,
            "learning_rate": 1e-4,
            "num_loci": 200,
            "lr_halflife":1,
            "min_avail":10
        }
        if len(sys.argv) >= 3:
            if sys.argv[2] == "synth":
                synth_hyper_parameters30a = {
                    "data_path": "/project/compbio-lab/encode_data/",
                    "input_dim": 47,
                    "metadata_embedding_dim": 40,
                    "dropout": 0.01,
                    "nhead": 8,
                    "d_model": 416,
                    "nlayers": 6,
                    "epochs": 2000,
                    "inner_epochs": 100,
                    "mask_percentage": 0.1,
                    "context_length": 400,
                    "batch_size": 36,
                    "learning_rate": 1e-4,
                    "num_loci": 1600,
                    "lr_halflife":1,
                    "min_avail":5
                }
            
                train_epd30_synthdata(
                    synth_hyper_parameters30a, arch="a")

        else:
            train_epidenoise30(
                hyper_parameters30a, 
                checkpoint_path=None, 
                arch="a")
    
    elif sys.argv[1]  == "epd30b":
        hyper_parameters30b = {
            "data_path": "/project/compbio-lab/encode_data/",
            "input_dim": 45,
            "metadata_embedding_dim": 40,
            "dropout": 0.01,

            "n_cnn_layers": 4,
            "conv_kernel_size" : 5,
            "n_decoder_layers" : 1,

            "nhead": 5,
            "d_model": 768,
            "nlayers": 6,
            "epochs": 10,
            "inner_epochs": 5,
            "mask_percentage": 0.15,
            "context_length": 810,
            "batch_size": 50,
            "learning_rate": 1e-4,
            "num_loci": 1600,
            "lr_halflife":1,
            "min_avail":5
        }
        if len(sys.argv) >= 3:
            if sys.argv[2] == "synth":
                synth_hyper_parameters30b = {
                    "data_path": "/project/compbio-lab/encode_data/",
                    "input_dim": 47,
                    "metadata_embedding_dim": 40,
                    "dropout": 0.1,

                    "n_cnn_layers": 3,
                    "conv_kernel_size" : 7,
                    "n_decoder_layers" : 1,

                    "nhead": 8,
                    "d_model": 768,
                    "nlayers": 2,
                    "epochs": 4000,
                    "inner_epochs": 50,
                    "mask_percentage": 0.1,
                    "context_length": 810,
                    "batch_size": 50,
                    "learning_rate": 5e-4,
                    "num_loci": 800,
                    "lr_halflife":2,
                    "min_avail":8
                }
                train_epd30_synthdata(
                    synth_hyper_parameters30b, arch="b")

        else:
            if sys.argv[1] == "epd30b":
                train_epidenoise30(
                    hyper_parameters30b, 
                    checkpoint_path=None, 
                    arch="b")

    elif sys.argv[1] == "epd30c":
        hyper_parameters30c = {
            "data_path": "/project/compbio-lab/encode_data/",
            "input_dim": 45,
            "metadata_embedding_dim": 45,
            "dropout": 0.05,

            "n_cnn_layers": 3,
            "conv_kernel_size" : 7,
            "pool_size" : 3,

            "nhead": 6,
            "d_model": (90)*(2**3),
            "nlayers": 2,
            "epochs": 1,
            "inner_epochs": 1,
            "mask_percentage": 0.1,
            "context_length": 810,
            "batch_size": 20,
            "learning_rate": 1e-4,
            "num_loci": 200,
            "lr_halflife":2,
            "min_avail":10
        }
        if len(sys.argv) >= 3:
            if sys.argv[2] == "synth":
                synth_hyper_parameters30cd = {
                    "data_path": "/project/compbio-lab/encode_data/",
                    "input_dim": 47,
                    "metadata_embedding_dim": 49,
                    "dropout": 0.05,

                    "n_cnn_layers": 3,
                    "conv_kernel_size" : 7,
                    "pool_size" : 3,

                    "nhead": 6,
                    "d_model": (47+49)*(2**3),
                    "nlayers": 3,
                    "epochs": 2000,
                    "inner_epochs": 50,
                    "mask_percentage": 0.1,
                    "context_length": 810,
                    "batch_size": 20,
                    "learning_rate": 1e-4,
                    "num_loci": 800,
                    "lr_halflife":2,
                    "min_avail":8
                }
                train_epd30_synthdata(
                    synth_hyper_parameters30cd, arch="c")

        else:
            train_epidenoise30(
                hyper_parameters30c, 
                checkpoint_path=None, 
                arch="c")
    
    elif sys.argv[1] == "epd30d":
        hyper_parameters30d = {
            "data_path": "/project/compbio-lab/encode_data/",
            "input_dim": 45,
            "metadata_embedding_dim": 45,
            "dropout": 0.1,

            "n_cnn_layers": 5,
            "conv_kernel_size" : 5,
            "pool_size": 2,

            "nhead": 16,
            "d_model": 768,
            "nlayers": 8,
            "epochs": 10,
            "inner_epochs": 5,
            "mask_percentage": 0.2,
            "context_length": 1600,
            "batch_size": 50,
            "learning_rate": 1e-4,
            "num_loci": 3200,
            "lr_halflife":1,
            "min_avail":5
        }
        train_epidenoise30(
            hyper_parameters30d, 
            checkpoint_path=None, 
            arch="d")
    
    elif sys.argv[1] == "epd30d_eic":
        hyper_parameters30d_eic = {
            "data_path": "/project/compbio-lab/encode_data/",
            "input_dim": 35,
            "metadata_embedding_dim": 35,
            "dropout": 0.1,

            "n_cnn_layers": 5,
            "conv_kernel_size" : 5,
            "pool_size": 2,

            "nhead": 16,
            "d_model": 768,
            "nlayers": 8,
            "epochs": 10,
            "inner_epochs": 1,
            "mask_percentage": 0.25,
            "context_length": 3200,
            "batch_size": 50,
            "learning_rate": 1e-4,
            "num_loci": 3200,
            "lr_halflife":1,
            "min_avail":1
        }
        train_epd30_eic(
            hyper_parameters30d_eic, 
            checkpoint_path=None, 
            arch="d")