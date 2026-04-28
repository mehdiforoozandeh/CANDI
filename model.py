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
from torch.distributions import Laplace
from typing import List, Dict, Optional
from candi_loss import CANDI_LOSS, LaplaceNLLLoss, CustomLaplaceNLLLoss

try:
    from x_transformers import Encoder as XTransformerEncoder, Attention as XAttention
    XTRANSFORMERS_AVAILABLE = True
except ImportError:
    XTRANSFORMERS_AVAILABLE = False
    XTransformerEncoder = None
    XAttention = None

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

##=========================================== EIC Validation Monitor =============================================##

class EIC_VALIDATION_MONITOR(object):
    """
    Validation monitor for EIC dataset during training.
    Evaluates model on V_* biosamples using T_* as input.
    Computes NLL losses and evaluation metrics for imputed and upsampled assays across chr21.
    
    Aggregation strategy:
    - Within a single assay: mean across genomic positions
    - Across multiple assays: median of per-assay values
    """
    
    def __init__(self, context_length, training_batch_size, device=None, resolution=25, dist_type='gaussian'):
        """
        Initialize EIC validation monitor.
        
        Args:
            context_length: Context length for genomic windows (in bins)
            training_batch_size: Training batch size (validation uses 4x this)
            device: Device to use for validation (auto-detect if None)
            resolution: Genomic resolution in bp
            dist_type: Distribution type for signal prediction ('gaussian' or 'laplace')
        """
        self.data_path = "/project/6014832/mforooz/DATA_CANDI_EIC"
        self.context_length = context_length
        self.resolution = resolution
        self.dist_type = dist_type
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
        
        # Initialize loss functions (element-wise, reduction='none' for median aggregation)
        self.nbin_nll = negative_binomial_loss  # Already returns element-wise
        
        if self.dist_type in ['laplace', 'laplace_const']:
            # laplace_const uses same NLL but with learned constant scale per assay
            self.signal_loss_fn = LaplaceNLLLoss(reduction="none")
        elif self.dist_type == 'mae':
            # MAE is equivalent to Laplace NLL with constant scale
            self.signal_loss_fn = torch.nn.L1Loss(reduction="none")
        elif self.dist_type == 'studentst':
            # Student's t uses students_t_nll_loss inline
            self.signal_loss_fn = None
        elif self.dist_type == 'gamma':
            self.signal_loss_fn = gamma_nll_loss
        elif self.dist_type == 'mse':
            # MSE is equivalent to Gaussian NLL with constant variance
            self.signal_loss_fn = torch.nn.MSELoss(reduction="none")
        else:
            # Default: Gaussian NLL (including gaussian_const which uses same NLL)
            self.signal_loss_fn = torch.nn.GaussianNLLLoss(reduction="none", full=True)
        
        # Peak uses BCELoss (not BCEWithLogitsLoss) since model output is already probability
        self.peak_bce_fn = torch.nn.BCELoss(reduction="none")
        
        # Token dictionary
        self.token_dict = {"missing_mask": -1, "cloze_mask": -2, "pad": -3}
        
        # Initialize METRICS helper for evaluation metrics
        self.metrics_helper = METRICS(chrom='chr21', bin_size=resolution)
    
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
        
        # Create unified mY for supertrack prompting (merge T and V metadata)
        # Prioritize V metadata (target), fill missing with T metadata (input)
        # mY_V and mY_T have shape [B, 4, F]
        mY_unified = mY_V.clone()
        missing_mask = (mY_unified == -1)
        if missing_mask.any():
            mY_unified[missing_mask] = mY_T[missing_mask]
        
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
            'mY_unified': mY_unified,
            'available_T_indices': available_T_indices,
            'available_V_indices': available_V_indices,
            'seq': seq,
            'T_biosample': T_biosample
        }
    
    def _load_dsf4_input(self, T_biosample: str, locus: List, cached_seq=None):
        """
        Load DSF=4 (low depth) input data for a T_* biosample.
        Used for DSF Invariance check in supertrack validation.
        
        Args:
            T_biosample: Name of T_* biosample
            locus: Genomic locus as [chrom, start, end]
            cached_seq: Pre-loaded DNA sequence tensor (optional)
            
        Returns:
            Dictionary with X, mX, avX, seq for DSF=4 data, or None if loading fails.
        """
        try:
            # Load DSF=4 count data
            temp_x, temp_mx = self.data_handler.load_bios_Counts(T_biosample, locus, DSF=4)
            if temp_x is None or len(temp_x) == 0:
                return None
            
            X, mX, avX = self.data_handler.make_bios_tensor_Counts(temp_x, temp_mx)
            del temp_x, temp_mx
            
            # Load control data (try DSF=4, fallback to DSF=1)
            try:
                temp_control_data, temp_control_metadata = self.data_handler.load_bios_Control(T_biosample, locus, DSF=4)
                if temp_control_data and "chipseq-control" in temp_control_data:
                    control_data, control_meta, control_avail = self.data_handler.make_bios_tensor_Control(temp_control_data, temp_control_metadata)
                else:
                    # Fallback to DSF=1 control
                    temp_control_data, temp_control_metadata = self.data_handler.load_bios_Control(T_biosample, locus, DSF=1)
                    if temp_control_data and "chipseq-control" in temp_control_data:
                        control_data, control_meta, control_avail = self.data_handler.make_bios_tensor_Control(temp_control_data, temp_control_metadata)
                    else:
                        raise ValueError("No control data found")
            except Exception:
                L = X.shape[0]
                control_data = torch.full((L, 1), -1.0)
                control_meta = torch.full((4, 1), -1.0)
                control_avail = torch.zeros(1)
            
            # Concatenate control data to input
            X = torch.cat([X, control_data], dim=1)
            mX = torch.cat([mX, control_meta], dim=1)
            avX = torch.cat([avX, control_avail], dim=0)
            
            # Reshape to context windows
            num_rows = (X.shape[0] // self.context_length) * self.context_length
            X = X[:num_rows, :]
            X = X.view(-1, self.context_length, X.shape[-1])
            
            # Expand metadata to match batch dimension
            mX = mX.expand(X.shape[0], -1, -1)
            avX = avX.expand(X.shape[0], -1)
            
            # Load DNA sequence (use cached if available)
            if cached_seq is not None:
                seq = cached_seq[:X.shape[0], :, :]
            else:
                seq = self.data_handler._dna_to_onehot(
                    self.data_handler._get_DNA_sequence(locus[0], locus[1], locus[2])
                )
                seq = seq[:num_rows * self.resolution, :]
                seq = seq.view(-1, self.context_length * self.resolution, seq.shape[-1])
            
            return {
                'X': X,
                'mX': mX,
                'avX': avX,
                'seq': seq
            }
            
        except Exception as e:
            print(f"    Warning: Failed to load DSF=4 data for {T_biosample}: {e}")
            return None
    
    def _predict(self, model, X, mX, mY, avX, seq):
        """
        Run model prediction in batches.
        
        Returns:
            output_p, output_n, output_mu, output_var, output_df, output_peak
            (output_df is None for Gaussian/Laplace distributions)
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
        
        # Initialize df tensor only if using Student's t distribution
        if self.dist_type == 'studentst':
            df = torch.empty(X.shape[0], X.shape[1], original_feature_dim, device="cpu", dtype=torch.float32)
        else:
            df = None
        
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
                
                # Forward pass - now returns 6 values
                outputs_p, outputs_n, outputs_mu, outputs_var, outputs_df, outputs_peak = model_to_use(
                    x_batch.float(), seq_batch, mX_batch.float(), mY_batch
                )
                
                # Store predictions (convert to float32 for metrics)
                batch_end = min(i + self.validation_batch_size, len(X))
                n[i:batch_end] = outputs_n.float().cpu()
                p[i:batch_end] = outputs_p.float().cpu()
                mu[i:batch_end] = outputs_mu.float().cpu()
                var[i:batch_end] = outputs_var.float().cpu()
                peak[i:batch_end] = outputs_peak.float().cpu()
                
                # Store df if using Student's t
                if self.dist_type == 'studentst' and outputs_df is not None:
                    df[i:batch_end] = outputs_df.float().cpu()
        
        return n, p, mu, var, df, peak
    
    def _compute_count_nll(self, n_pred, p_pred, y_true):
        """Compute mean NB NLL across positions for a single assay."""
        # Flatten: n_pred, p_pred, y_true are 2D [B*L] already (single assay)
        n_flat = n_pred.view(-1)
        p_flat = p_pred.view(-1)
        y_flat = y_true.view(-1)
        
        # Filter out invalid values (sentinel tokens)
        valid_mask = (y_flat >= 0) & (n_flat > 0) & (p_flat > 0) & (p_flat < 1)
        if valid_mask.sum() == 0:
            return np.nan
        
        n_valid = n_flat[valid_mask]
        p_valid = p_flat[valid_mask]
        y_valid = y_flat[valid_mask]
        
        # Compute element-wise NLL, then mean across positions
        nll = self.nbin_nll(y_valid.unsqueeze(-1), n_valid.unsqueeze(-1), p_valid.unsqueeze(-1))
        return float(nll.mean().item())
    
    def _compute_signal_nll(self, mu_pred, var_pred, y_true, df_pred=None):
        """Compute mean Gaussian/Laplace/StudentT NLL (or MSE/MAE for deterministic modes) across positions for a single assay."""
        mu_flat = mu_pred.view(-1)
        var_flat = var_pred.view(-1)
        y_flat = y_true.view(-1)
        
        # Filter out invalid values (sentinel tokens)
        # For deterministic losses, var is untrained so we only check mu and y
        if self.dist_type in ['mse', 'mae']:
            valid_mask = (y_flat > -100)
        else:
            valid_mask = (y_flat > -100) & (var_flat > 0) 
        if valid_mask.sum() == 0:
            return np.nan
        
        mu_valid = mu_flat[valid_mask]
        var_valid = var_flat[valid_mask]
        y_valid = y_flat[valid_mask]
        
        # Compute element-wise loss, then mean across positions
        if self.dist_type == 'studentst' and df_pred is not None:
            df_flat = df_pred.view(-1)
            df_valid = df_flat[valid_mask]
            nll = students_t_nll_loss(y_valid, mu_valid, var_valid, df_valid, reduction='none')
        elif self.dist_type == 'gamma':
            # Gamma distribution: var_valid holds alpha
            nll = self.signal_loss_fn(y_valid, mu_valid, var_valid, reduction='none')
        elif self.dist_type in ['mse', 'mae']:
            # Deterministic losses: only use mu, ignore variance
            nll = self.signal_loss_fn(mu_valid, y_valid)
        else:
            nll = self.signal_loss_fn(mu_valid, y_valid, var_valid)
        return float(nll.mean().item())
    
    def _compute_peak_bce(self, peak_pred, peak_true):
        """Compute mean BCE across positions for a single assay. peak_pred is probability (sigmoid already applied)."""
        peak_pred_flat = peak_pred.view(-1)
        peak_true_flat = peak_true.view(-1).float()
        
        # Filter out invalid values
        valid_mask = (peak_true_flat >= 0) & (peak_pred_flat >= 0) & (peak_pred_flat <= 1)
        if valid_mask.sum() == 0:
            return np.nan
        
        pred_valid = peak_pred_flat[valid_mask]
        true_valid = peak_true_flat[valid_mask]
        
        # Clamp predictions to avoid log(0)
        pred_valid = torch.clamp(pred_valid, min=1e-7, max=1-1e-7)
        
        # Compute element-wise BCE, then mean across positions
        bce = self.peak_bce_fn(pred_valid, true_valid)
        return float(bce.mean().item())
    
    def _compute_assay_metrics(self, pred_count, y_count, pred_pval, y_pval, pred_peak, y_peak):
        """
        Compute all evaluation metrics for a single assay.
        
        Args:
            pred_count: Predicted count (NB mean), flattened numpy array
            y_count: Ground truth count, flattened numpy array
            pred_pval: Predicted p-value (Gaussian mean), flattened numpy array
            y_pval: Ground truth p-value, flattened numpy array
            pred_peak: Predicted peak probability, flattened numpy array
            y_peak: Ground truth peak (binary), flattened numpy array
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import roc_auc_score
        
        metrics = {}
        
        def safe_metric(fn, *args, **kwargs):
            try:
                result = fn(*args, **kwargs)
                if np.isnan(result) or np.isinf(result):
                    return np.nan
                return result
            except Exception:
                return np.nan
        
        # === Count metrics ===
        # Filter valid count values
        count_valid_mask = (y_count >= 0) & (pred_count >= 0)
        if count_valid_mask.sum() > 10:
            y_c = y_count[count_valid_mask]
            p_c = pred_count[count_valid_mask]
            
            # GW metrics
            metrics['count_pearson_GW'] = safe_metric(self.metrics_helper.pearson, y_c, p_c)
            metrics['count_spearman_GW'] = safe_metric(self.metrics_helper.spearman, y_c, p_c)
            metrics['count_mae_GW'] = safe_metric(self.metrics_helper.mae, y_c, p_c)
            metrics['count_mae_r2_GW'] = safe_metric(self.metrics_helper.mae_r2, y_c, p_c)
            
            # Gene metrics
            metrics['count_pearson_gene'] = safe_metric(self.metrics_helper.pearson_gene, y_c, p_c)
            metrics['count_spearman_gene'] = safe_metric(self.metrics_helper.spearman_gene, y_c, p_c)
            metrics['count_mae_gene'] = safe_metric(self.metrics_helper.mae_gene, y_c, p_c)
            metrics['count_mae_r2_gene'] = safe_metric(self.metrics_helper.mae_r2_gene, y_c, p_c)
            
            # Prom metrics
            metrics['count_pearson_prom'] = safe_metric(self.metrics_helper.pearson_prom, y_c, p_c)
            metrics['count_spearman_prom'] = safe_metric(self.metrics_helper.spearman_prom, y_c, p_c)
            metrics['count_mae_prom'] = safe_metric(self.metrics_helper.mae_prom, y_c, p_c)
            metrics['count_mae_r2_prom'] = safe_metric(self.metrics_helper.mae_r2_prom, y_c, p_c)
            
            # 1obs metrics
            metrics['count_pearson_1obs'] = safe_metric(self.metrics_helper.pearson1_obs, y_c, p_c)
            metrics['count_spearman_1obs'] = safe_metric(self.metrics_helper.spearman1_obs, y_c, p_c)
            metrics['count_mae_1obs'] = safe_metric(self.metrics_helper.mae1obs, y_c, p_c)
            metrics['count_mae_r2_1obs'] = safe_metric(self.metrics_helper.mae_r2_1obs, y_c, p_c)
        
        # === Signal (p-value) metrics ===
        pval_valid_mask = (y_pval > -100) & (pred_pval > -100)  # p-values can be negative (arcsinh transformed)
        if pval_valid_mask.sum() > 10:
            y_p = y_pval[pval_valid_mask]
            p_p = pred_pval[pval_valid_mask]
            
            # GW metrics
            metrics['pval_pearson_GW'] = safe_metric(self.metrics_helper.pearson, y_p, p_p)
            metrics['pval_spearman_GW'] = safe_metric(self.metrics_helper.spearman, y_p, p_p)
            metrics['pval_mae_GW'] = safe_metric(self.metrics_helper.mae, y_p, p_p)
            metrics['pval_mae_r2_GW'] = safe_metric(self.metrics_helper.mae_r2, y_p, p_p)
            
            # Gene metrics
            metrics['pval_pearson_gene'] = safe_metric(self.metrics_helper.pearson_gene, y_p, p_p)
            metrics['pval_spearman_gene'] = safe_metric(self.metrics_helper.spearman_gene, y_p, p_p)
            metrics['pval_mae_gene'] = safe_metric(self.metrics_helper.mae_gene, y_p, p_p)
            metrics['pval_mae_r2_gene'] = safe_metric(self.metrics_helper.mae_r2_gene, y_p, p_p)
            
            # Prom metrics
            metrics['pval_pearson_prom'] = safe_metric(self.metrics_helper.pearson_prom, y_p, p_p)
            metrics['pval_spearman_prom'] = safe_metric(self.metrics_helper.spearman_prom, y_p, p_p)
            metrics['pval_mae_prom'] = safe_metric(self.metrics_helper.mae_prom, y_p, p_p)
            metrics['pval_mae_r2_prom'] = safe_metric(self.metrics_helper.mae_r2_prom, y_p, p_p)
            
            # 1obs metrics
            metrics['pval_pearson_1obs'] = safe_metric(self.metrics_helper.pearson1_obs, y_p, p_p)
            metrics['pval_spearman_1obs'] = safe_metric(self.metrics_helper.spearman1_obs, y_p, p_p)
            metrics['pval_mae_1obs'] = safe_metric(self.metrics_helper.mae1obs, y_p, p_p)
            metrics['pval_mae_r2_1obs'] = safe_metric(self.metrics_helper.mae_r2_1obs, y_p, p_p)
        
        # === Peak metrics (AUCROC) ===
        peak_valid_mask = (y_peak >= 0) & (pred_peak >= 0) & (pred_peak <= 1)
        if peak_valid_mask.sum() > 10:
            y_pk = y_peak[peak_valid_mask]
            p_pk = pred_peak[peak_valid_mask]
            
            # Only compute AUCROC if we have both classes
            if len(np.unique(y_pk)) > 1:
                metrics['peak_aucroc_GW'] = safe_metric(self.metrics_helper.aucroc, y_pk, p_pk)
                metrics['peak_aucroc_gene'] = safe_metric(self.metrics_helper.aucroc_gene, y_pk, p_pk)
                metrics['peak_aucroc_prom'] = safe_metric(self.metrics_helper.aucroc_prom, y_pk, p_pk)
        
        return metrics
    
    def _run_supertrack_checks(self, model, T_biosample, locus, X_dsf1, mX_dsf1, avX_dsf1, seq, mY_template, cached_seq=None):
        """
        Run supertrack prompt sensitivity checks to verify metadata affects outputs.
        
        Implements Part D of issue_supertrack.md:
        1. Depth Sensitivity Ratio: Output ratio when prompting high vs low depth
        2. RunType Identity MSE: MSE between Single vs Paired run type prompts
        3. ReadLength Identity MSE: MSE between short vs long read length prompts
        4. DSF Invariance Ratio: Output ratio for DSF=1 vs DSF=4 inputs with same canonical prompt
        
        Args:
            model: CANDI model to evaluate
            T_biosample: Name of T_* biosample
            locus: Genomic locus as [chrom, start, end]
            X_dsf1: Input tensor (DSF=1) [B, L, F+1]
            mX_dsf1: Input metadata [B, 4, F+1]
            avX_dsf1: Input availability [B, F+1]
            seq: DNA sequence tensor [B, L*res, 4]
            mY_template: Template metadata tensor for Y-side prompts [B, 4, F] (Should be mY_unified)
            cached_seq: Pre-loaded DNA sequence (optional)
            
        Returns:
            Dictionary with supertrack check metrics
        """
        results = {}
        
        # Unwrap DDP if needed
        if hasattr(model, 'module'):
            model_to_use = model.module
        else:
            model_to_use = model
        
        model_to_use.train()  # Use batch statistics (consistent with _predict)
        
        # Use a small subset for speed - reduced to 1x batch size to avoid OOM
        subset_size = min(self.validation_batch_size, X_dsf1.shape[0])
        X_sub = X_dsf1[:subset_size]
        mX_sub = mX_dsf1[:subset_size]
        avX_sub = avX_dsf1[:subset_size] if avX_dsf1.dim() > 1 else avX_dsf1.unsqueeze(0).expand(subset_size, -1)
        seq_sub = seq[:subset_size]
        mY_base = mY_template[:subset_size].clone()
        
        # Metadata tensor layout: [B, 4, F]
        # row 0: depth (log2)
        # row 1: assay_id (categorical, identifies assay type - fixed per column, NOT modified in ST checks)
        # row 2: read_length
        # row 3: run_type (0=single, 1=paired)
        
        def run_forward(X, mX, mY, seq_batch):
            """Run model forward pass and return NB mean output."""
            nb_mean_cpu = None
            mu_cpu = None
            try:
                with torch.no_grad(), torch.amp.autocast('cuda', enabled=True):
                    X_dev = X.to(self.device, non_blocking=True).float()
                    mX_dev = mX.to(self.device, non_blocking=True).float()
                    mY_dev = mY.to(self.device, non_blocking=True).float()
                    seq_dev = seq_batch.to(self.device, non_blocking=True)
                    
                    # Apply masking (consistent with _predict)
                    X_dev = X_dev.clone()
                    mX_dev = mX_dev.clone()
                    X_dev[X_dev == self.token_dict["missing_mask"]] = float(self.token_dict["cloze_mask"])
                    mX_dev[mX_dev == self.token_dict["missing_mask"]] = float(self.token_dict["cloze_mask"])
                    
                    outputs_p, outputs_n, outputs_mu, outputs_var, outputs_df, outputs_peak = model_to_use(
                        X_dev, seq_dev, mX_dev, mY_dev
                    )
                    
                    # Compute NB mean: n*(1-p)/p and move to CPU immediately
                    nb_mean = (outputs_n * (1 - outputs_p)) / outputs_p
                    nb_mean_cpu = nb_mean.float().cpu()
                    mu_cpu = outputs_mu.float().cpu()
                    
                    # Explicitly delete GPU tensors
                    del X_dev, mX_dev, mY_dev, seq_dev
                    del outputs_p, outputs_n, outputs_mu, outputs_var, outputs_df, outputs_peak, nb_mean
            finally:
                # Always clean up GPU memory after forward pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            return nb_mean_cpu, mu_cpu
        
        # Mask of valid prompts in base metadata (where we have actual metadata to start with)
        # shape: [B, F] (checking across all 4 metadata channels, if any is missing/sentinel)
        # Here we check specifically the metadata channel we are about to modify, but generally
        # if mY_base is -1, it means that assay is missing in unified view.
        # We assume sentinel is -1.
        valid_prompt_mask = (mY_base[:, 0, :] != -1) # Check depth channel as proxy
        
        try:
            # ========== Check 1: Depth Sensitivity Ratio ==========
            # Prompt with log2(depth)=23 (Low) vs log2(depth)=25 (High)
            # Ideal ratio: ~4 (since 2^25 / 2^23 = 4)
            # Only modify where valid_prompt_mask is True
            
            mY_low = mY_base.clone()
            # Replace depth (dim 1 index 0) only for valid assays
            mY_low[:, 0, :][valid_prompt_mask] = 23.0
            
            mY_high = mY_base.clone()
            # Replace depth (dim 1 index 0) only for valid assays
            mY_high[:, 0, :][valid_prompt_mask] = 25.0
            
            print(f"\n    ST Check 1 (Depth): Modifying {valid_prompt_mask.sum().item()//mY_high.shape[0]} features (depth 23 vs 25)")
            
            nb_mean_low, _ = run_forward(X_sub, mX_sub, mY_low, seq_sub)
            nb_mean_high, _ = run_forward(X_sub, mX_sub, mY_high, seq_sub)
            
            # Filter valid values AND ensure we only look at prompted assays
            # valid_prompt_mask needs to be expanded to [B, L, F] to match nb_mean output?
            # nb_mean output is [B, L, F]. valid_prompt_mask is [B, F].
            # We need to broadcast valid_prompt_mask over L dimension.
            valid_prompt_mask_expanded = valid_prompt_mask.unsqueeze(1).expand(-1, nb_mean_low.shape[1], -1)
            
            mask = (nb_mean_low > 0) & (nb_mean_high > 0) & \
                   ~torch.isnan(nb_mean_low) & ~torch.isnan(nb_mean_high) & \
                   valid_prompt_mask_expanded
                   
            if mask.sum() > 0:
                sum_low = nb_mean_low[mask].sum().item()
                sum_high = nb_mean_high[mask].sum().item()
                if sum_low > 1e-6:
                    results['st_depth_ratio'] = sum_high / sum_low
                else:
                    results['st_depth_ratio'] = np.nan
            else:
                results['st_depth_ratio'] = np.nan
            
            # Clean up Check 1 tensors
            del nb_mean_low, nb_mean_high, mY_low, mY_high
            
            # ========== Check 2: RunType Identity MSE ==========
            # Prompt with RunType=Single (0) vs RunType=Paired (1)
            # Ideal: MSE > 0 (non-identical outputs)
            mY_single = mY_base.clone()
            mY_single[:, 3, :][valid_prompt_mask] = 0  # Single-end
            
            mY_paired = mY_base.clone()
            mY_paired[:, 3, :][valid_prompt_mask] = 1  # Paired-end
            
            print(f"    ST Check 2 (RunType): Modifying {valid_prompt_mask.sum().item()//mY_base.shape[0]} features (single vs paired)")
            
            nb_mean_single, _ = run_forward(X_sub, mX_sub, mY_single, seq_sub)
            nb_mean_paired, _ = run_forward(X_sub, mX_sub, mY_paired, seq_sub)
            
            mask = ~torch.isnan(nb_mean_single) & ~torch.isnan(nb_mean_paired) & valid_prompt_mask_expanded
            if mask.sum() > 0:
                mse = ((nb_mean_single[mask] - nb_mean_paired[mask]) ** 2).mean().item()
                results['st_runtype_mse'] = mse
            else:
                results['st_runtype_mse'] = np.nan
            
            # Clean up Check 2 tensors
            del nb_mean_single, nb_mean_paired, mY_single, mY_paired
            
            # ========== Check 3: ReadLength Identity MSE ==========
            # Prompt with ReadLength=36 (Short) vs ReadLength=100 (Long)
            # Ideal: MSE > 0 (non-identical outputs)
            mY_short = mY_base.clone()
            mY_short[:, 2, :][valid_prompt_mask] = 36  # Short reads
            
            mY_long = mY_base.clone()
            mY_long[:, 2, :][valid_prompt_mask] = 100  # Long reads
            
            print(f"    ST Check 3 (ReadLen): Modifying {valid_prompt_mask.sum().item()//mY_base.shape[0]} features (36 vs 100)")
            
            nb_mean_short, _ = run_forward(X_sub, mX_sub, mY_short, seq_sub)
            nb_mean_long, _ = run_forward(X_sub, mX_sub, mY_long, seq_sub)
            
            mask = ~torch.isnan(nb_mean_short) & ~torch.isnan(nb_mean_long) & valid_prompt_mask_expanded
            if mask.sum() > 0:
                mse = ((nb_mean_short[mask] - nb_mean_long[mask]) ** 2).mean().item()
                results['st_readlen_mse'] = mse
            else:
                results['st_readlen_mse'] = np.nan
            
            # Clean up Check 3 tensors
            del nb_mean_short, nb_mean_long, mY_short, mY_long
            
            # ========== Check 4: DSF Invariance Ratio ==========
            # Input DSF=1 vs DSF=4, both with same canonical prompt (Depth=24, Paired, ReadLen=100)
            # Ideal ratio: ~1 (identical outputs for same target prompt)
            dsf4_data = self._load_dsf4_input(T_biosample, locus, cached_seq=cached_seq)
            
            if dsf4_data is not None:
                # Canonical "supertrack" prompt
                mY_canon = mY_base.clone()
                mY_canon[:, 0, :][valid_prompt_mask] = 24.0   # Canonical depth
                mY_canon[:, 2, :][valid_prompt_mask] = 100    # Long reads
                mY_canon[:, 3, :][valid_prompt_mask] = 1      # Paired-end
                
                print(f"    ST Check 4 (DSF Inv): Modifying {valid_prompt_mask.sum().item()//mY_base.shape[0]} features (canonical prompt)")
                
                # Match subset size with DSF4 data - use same subset_size to avoid OOM
                dsf4_subset_size = min(subset_size, dsf4_data['X'].shape[0])
                X_dsf4_sub = dsf4_data['X'][:dsf4_subset_size]
                mX_dsf4_sub = dsf4_data['mX'][:dsf4_subset_size]
                seq_dsf4_sub = dsf4_data['seq'][:dsf4_subset_size]
                
                # Use matching subset size for DSF1
                X_dsf1_sub = X_sub[:dsf4_subset_size]
                mX_dsf1_sub = mX_sub[:dsf4_subset_size]
                seq_dsf1_sub = seq_sub[:dsf4_subset_size]
                mY_canon_sub = mY_canon[:dsf4_subset_size]
                
                # Run forward passes first
                nb_mean_dsf1, _ = run_forward(X_dsf1_sub, mX_dsf1_sub, mY_canon_sub, seq_dsf1_sub)
                nb_mean_dsf4, _ = run_forward(X_dsf4_sub, mX_dsf4_sub, mY_canon_sub, seq_dsf4_sub)
                
                # Expand mask for subset (after forward passes so we know the L dimension)
                valid_prompt_mask_sub = valid_prompt_mask[:dsf4_subset_size]
                valid_prompt_mask_sub_expanded = valid_prompt_mask_sub.unsqueeze(1).expand(-1, nb_mean_dsf1.shape[1], -1)
                
                mask = (nb_mean_dsf1 > 0) & (nb_mean_dsf4 > 0) & \
                       ~torch.isnan(nb_mean_dsf1) & ~torch.isnan(nb_mean_dsf4) & \
                       valid_prompt_mask_sub_expanded
                       
                if mask.sum() > 0:
                    sum_dsf1 = nb_mean_dsf1[mask].sum().item()
                    sum_dsf4 = nb_mean_dsf4[mask].sum().item()
                    if sum_dsf4 > 1e-6:
                        results['st_dsf_invariance_ratio'] = sum_dsf1 / sum_dsf4
                    else:
                        results['st_dsf_invariance_ratio'] = np.nan
                    
                    # Also compute MSE for more detailed comparison
                    mse = ((nb_mean_dsf1[mask] - nb_mean_dsf4[mask]) ** 2).mean().item()
                    results['st_dsf_invariance_mse'] = mse
                else:
                    results['st_dsf_invariance_ratio'] = np.nan
                    results['st_dsf_invariance_mse'] = np.nan
            else:
                results['st_dsf_invariance_ratio'] = np.nan
                results['st_dsf_invariance_mse'] = np.nan
            
        except Exception as e:
            print(f"    Warning: Supertrack check failed for {T_biosample}: {e}")
            import traceback
            traceback.print_exc()
            results = {
                'st_depth_ratio': np.nan,
                'st_runtype_mse': np.nan,
                'st_readlen_mse': np.nan,
                'st_dsf_invariance_ratio': np.nan,
                'st_dsf_invariance_mse': np.nan
            }
        
        # Clean up GPU memory after supertrack checks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return results
    
    def run_validation(self, model, batch_idx, total_batches):
        """
        Run validation on all V_* biosamples in EIC dataset.
        
        Args:
            model: CANDI model to evaluate
            batch_idx: Current batch index
            total_batches: Total number of batches in training
        
        Returns:
            Dictionary with keys namespaced for W&B groups:
            - val_loss/*: median losses (no per-assay breakdown)
            - val_metrics/*: median metrics (no per-assay breakdown)
            - val_loss_per_assay/<assay_name>/*: per-assay losses
            - val_metrics_per_assay/<assay_name>/*: per-assay metrics
            - supertrack/*: supertrack prompt sensitivity checks
        """
        print(f"Running EIC validation at batch {batch_idx} ({100.0 * batch_idx / total_batches:.1f}% progress)...")
        
        # Use full chr21
        locus = ["chr21", 0, self.chr_sizes["chr21"]]
        
        # Collect all data per (biosample, assay_name, comparison_type)
        all_records = []
        
        # Collect supertrack check results per biosample
        supertrack_results = []
        
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
                
                # Run prediction on FULL chr21 data (no subsetting)
                n, p, mu, var, df, peak = self._predict(
                    model, data['X'], data['mX'], data['mY_T'], data['avX'], data['seq']
                )
                
                # Immediately free GPU memory after prediction
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Compute NB mean for count predictions
                nb_mean = (n * (1 - p)) / p  # NB mean = n*(1-p)/p
                
                # Determine which assays are upsampled vs imputed
                available_T_set = set(data['available_T_indices'])
                available_V_set = set(data['available_V_indices'])
                
                upsampled_assays = available_T_set  # Assays in T_*
                imputed_assays = available_V_set - available_T_set  # Assays in V_* but not in T_*
                
                # Process upsampled assays (compare against T_* ground truth subset)
                for assay_idx in upsampled_assays:
                    if assay_idx >= len(self.expnames):
                        continue
                    
                    assay_name = self.expnames[assay_idx]
                    
                    # Extract predictions for this assay
                    n_assay = n[:, :, assay_idx]
                    p_assay = p[:, :, assay_idx]
                    mu_assay = mu[:, :, assay_idx]
                    var_assay = var[:, :, assay_idx]
                    df_assay = df[:, :, assay_idx] if df is not None else None
                    peak_assay = peak[:, :, assay_idx]
                    nb_mean_assay = nb_mean[:, :, assay_idx]
                    
                    # Extract ground truth from T_* (full chr21)
                    y_T_assay = data['Y_T'][:, :, assay_idx]
                    p_T_assay = data['P_T'][:, :, assay_idx]
                    peak_T_assay = data['Peak_T'][:, :, assay_idx]
                    
                    # Compute losses (mean across positions for single assay)
                    count_nll = self._compute_count_nll(n_assay, p_assay, y_T_assay)
                    signal_nll = self._compute_signal_nll(mu_assay, var_assay, p_T_assay, df_pred=df_assay)
                    peak_bce = self._compute_peak_bce(peak_assay, peak_T_assay)
                    
                    # Compute perplexity from NLL (perplexity = exp(NLL))
                    count_perplexity = np.exp(count_nll) if (count_nll is not None and not np.isnan(count_nll)) else None
                    signal_perplexity = np.exp(signal_nll) if (signal_nll is not None and not np.isnan(signal_nll)) else None
                    
                    # Compute evaluation metrics
                    eval_metrics = self._compute_assay_metrics(
                        pred_count=nb_mean_assay.numpy().flatten(),
                        y_count=y_T_assay.numpy().flatten(),
                        pred_pval=mu_assay.numpy().flatten(),
                        y_pval=p_T_assay.numpy().flatten(),
                        pred_peak=peak_assay.numpy().flatten(),
                        y_peak=peak_T_assay.numpy().flatten()
                    )
                    
                    record = {
                        'biosample': V_biosample,
                        'assay_name': assay_name,
                        'comparison': 'ups',
                        'count_nll': count_nll,
                        'signal_nll': signal_nll,
                        'peak_bce': peak_bce,
                        'count_perplexity': count_perplexity,
                        'signal_perplexity': signal_perplexity,
                        **eval_metrics
                    }
                    all_records.append(record)
                
                # Process imputed assays (compare against V_* ground truth subset)
                for assay_idx in imputed_assays:
                    if assay_idx >= len(self.expnames):
                        continue
                    
                    assay_name = self.expnames[assay_idx]
                    
                    # Extract predictions for this assay
                    n_assay = n[:, :, assay_idx]
                    p_assay = p[:, :, assay_idx]
                    mu_assay = mu[:, :, assay_idx]
                    var_assay = var[:, :, assay_idx]
                    df_assay = df[:, :, assay_idx] if df is not None else None
                    peak_assay = peak[:, :, assay_idx]
                    nb_mean_assay = nb_mean[:, :, assay_idx]
                    
                    # Extract ground truth from V_* (full chr21)
                    y_V_assay = data['Y_V'][:, :, assay_idx]
                    p_V_assay = data['P_V'][:, :, assay_idx]
                    peak_V_assay = data['Peak_V'][:, :, assay_idx]
                    
                    # Compute losses (mean across positions for single assay)
                    count_nll = self._compute_count_nll(n_assay, p_assay, y_V_assay)
                    signal_nll = self._compute_signal_nll(mu_assay, var_assay, p_V_assay, df_pred=df_assay)
                    peak_bce = self._compute_peak_bce(peak_assay, peak_V_assay)
                    
                    # Compute perplexity from NLL (perplexity = exp(NLL))
                    count_perplexity = np.exp(count_nll) if (count_nll is not None and not np.isnan(count_nll)) else None
                    signal_perplexity = np.exp(signal_nll) if (signal_nll is not None and not np.isnan(signal_nll)) else None
                    
                    # Compute evaluation metrics
                    eval_metrics = self._compute_assay_metrics(
                        pred_count=nb_mean_assay.numpy().flatten(),
                        y_count=y_V_assay.numpy().flatten(),
                        pred_pval=mu_assay.numpy().flatten(),
                        y_pval=p_V_assay.numpy().flatten(),
                        pred_peak=peak_assay.numpy().flatten(),
                        y_peak=peak_V_assay.numpy().flatten()
                    )
                    
                    record = {
                        'biosample': V_biosample,
                        'assay_name': assay_name,
                        'comparison': 'imp',
                        'count_nll': count_nll,
                        'signal_nll': signal_nll,
                        'peak_bce': peak_bce,
                        'count_perplexity': count_perplexity,
                        'signal_perplexity': signal_perplexity,
                        **eval_metrics
                    }
                    all_records.append(record)
                
                # Run supertrack prompt sensitivity checks
                st_result = self._run_supertrack_checks(
                    model=model,
                    T_biosample=data['T_biosample'],
                    locus=locus,
                    X_dsf1=data['X'],
                    mX_dsf1=data['mX'],
                    avX_dsf1=data['avX'],
                    seq=data['seq'],
                    mY_template=data['mY_unified'],
                    cached_seq=cached_seq
                )
                st_result['biosample'] = V_biosample
                supertrack_results.append(st_result)
                
                # Clean up memory after each biosample
                del data, n, p, mu, var, df, peak, nb_mean
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                print(f"done ({len(upsampled_assays)} ups, {len(imputed_assays)} imp)")
                
            except Exception as e:
                print(f"failed: {e}")
                import traceback
                traceback.print_exc()
                gc.collect()
                # Only try to clear cache if CUDA isn't in a corrupted state
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except RuntimeError as cuda_err:
                        print(f"Warning: CUDA cache clear failed (GPU may be in bad state): {cuda_err}")
                continue
        
        if not all_records:
            print("Warning: No validation metrics computed")
            return self._get_empty_result(batch_idx, total_batches)
        
        # Convert to DataFrame for aggregation
        import pandas as pd
        df = pd.DataFrame(all_records)
        
        # Split by comparison type
        imp_df = df[df['comparison'] == 'imp']
        ups_df = df[df['comparison'] == 'ups']
        
        # Build result dictionary with proper namespacing
        result = {
            'iteration': batch_idx,
            'progress_pct': 100.0 * batch_idx / total_batches
        }
        
        # === val_loss/* : median of per-assay losses (each per-assay loss is mean across positions) ===
        loss_cols = ['count_nll', 'signal_nll', 'peak_bce', 'count_perplexity', 'signal_perplexity']
        for col in loss_cols:
            if col in imp_df.columns and len(imp_df) > 0:
                vals = imp_df[col].dropna()
                if len(vals) > 0:
                    result[f'val_loss/imp_{col}'] = float(np.median(vals))
            if col in ups_df.columns and len(ups_df) > 0:
                vals = ups_df[col].dropna()
                if len(vals) > 0:
                    result[f'val_loss/ups_{col}'] = float(np.median(vals))
        
        # === val_metrics/* : median of per-assay metrics (each metric computed on full assay positions) ===
        metric_cols = [c for c in df.columns if c not in ['biosample', 'assay_name', 'comparison'] + loss_cols]
        for col in metric_cols:
            if col in imp_df.columns and len(imp_df) > 0:
                vals = imp_df[col].dropna()
                if len(vals) > 0:
                    result[f'val_metrics/imp_{col}'] = float(np.median(vals))
            if col in ups_df.columns and len(ups_df) > 0:
                vals = ups_df[col].dropna()
                if len(vals) > 0:
                    result[f'val_metrics/ups_{col}'] = float(np.median(vals))
        
        # === val_loss_per_assay/* and val_metrics_per_assay/* : median across biosamples for each assay ===
        for assay_name in df['assay_name'].unique():
            assay_imp_df = imp_df[imp_df['assay_name'] == assay_name]
            assay_ups_df = ups_df[ups_df['assay_name'] == assay_name]
            
            # Per-assay losses
            for col in loss_cols:
                if col in assay_imp_df.columns and len(assay_imp_df) > 0:
                    vals = assay_imp_df[col].dropna()
                    if len(vals) > 0:
                        result[f'val_loss_per_assay/{assay_name}/imp_{col}'] = float(np.median(vals))
                if col in assay_ups_df.columns and len(assay_ups_df) > 0:
                    vals = assay_ups_df[col].dropna()
                    if len(vals) > 0:
                        result[f'val_loss_per_assay/{assay_name}/ups_{col}'] = float(np.median(vals))
            
            # Per-assay metrics
            for col in metric_cols:
                if col in assay_imp_df.columns and len(assay_imp_df) > 0:
                    vals = assay_imp_df[col].dropna()
                    if len(vals) > 0:
                        result[f'val_metrics_per_assay/{assay_name}/imp_{col}'] = float(np.median(vals))
                if col in assay_ups_df.columns and len(assay_ups_df) > 0:
                    vals = assay_ups_df[col].dropna()
                    if len(vals) > 0:
                        result[f'val_metrics_per_assay/{assay_name}/ups_{col}'] = float(np.median(vals))
        
        print(f"Validation completed: {len(imp_df)} imputed, {len(ups_df)} upsampled assay evaluations")
        
        # === supertrack/* : aggregate supertrack prompt sensitivity checks ===
        if supertrack_results:
            st_df = pd.DataFrame(supertrack_results)
            st_metric_cols = ['st_depth_ratio', 'st_runtype_mse', 'st_readlen_mse', 
                             'st_dsf_invariance_ratio', 'st_dsf_invariance_mse']
            
            for col in st_metric_cols:
                if col in st_df.columns:
                    vals = st_df[col].dropna()
                    if len(vals) > 0:
                        result[f'supertrack/{col}'] = float(np.median(vals))
            
            # Report how many biosamples had valid supertrack checks
            result['supertrack/num_valid_checks'] = len(st_df)
            
            # Log ideal values for reference (as constants)
            # st_depth_ratio ideal: ~4 (2^25 / 2^23 = 4)
            # st_runtype_mse ideal: > 0
            # st_readlen_mse ideal: > 0
            # st_dsf_invariance_ratio ideal: ~1
            print(f"  Supertrack checks: depth_ratio={result.get('supertrack/st_depth_ratio', 'N/A'):.3f} "
                  f"(ideal~4), dsf_inv={result.get('supertrack/st_dsf_invariance_ratio', 'N/A'):.3f} (ideal~1)")
        
        return result
    
    def _get_empty_result(self, batch_idx, total_batches):
        """Return empty result dict when no validation metrics are computed."""
        return {
            'iteration': batch_idx,
            'progress_pct': 100.0 * batch_idx / total_batches,
            'val_loss/imp_count_nll': 0.0,
            'val_loss/imp_signal_nll': 0.0,
            'val_loss/imp_peak_bce': 0.0,
            'val_loss/imp_count_perplexity': 1.0,
            'val_loss/imp_signal_perplexity': 1.0,
            'val_loss/ups_count_nll': 0.0,
            'val_loss/ups_signal_nll': 0.0,
            'val_loss/ups_peak_bce': 0.0,
            'val_loss/ups_count_perplexity': 1.0,
            'val_loss/ups_signal_perplexity': 1.0,
            # Supertrack check defaults (NaN indicates no data)
            'supertrack/st_depth_ratio': np.nan,
            'supertrack/st_runtype_mse': np.nan,
            'supertrack/st_readlen_mse': np.nan,
            'supertrack/st_dsf_invariance_ratio': np.nan,
            'supertrack/st_dsf_invariance_mse': np.nan,
            'supertrack/num_valid_checks': 0
        }

##=========================================== Loss Functions =============================================##
# Loss implementations are modularized in `candi_loss.py` and imported above.
# They are re-exported from this module for backward compatibility.

##=========================================== CANDI Architecture =============================================##

class CANDI_Decoder(nn.Module):
    def __init__(self, signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, context_length, pool_size=2, expansion_factor=3, norm="layer"):
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
        
        # Per-layer FiLM Adapter for Y-side metadata
        self.film_layers = nn.ModuleList()
        for i in range(n_cnn_layers):
            layer_channels = reverse_conv_channels[i + 1] if i + 1 < n_cnn_layers else int(reverse_conv_channels[i] / expansion_factor)
            # Check for divisibility just in case
            if layer_channels % signal_dim == 0:
                d_per_assay = layer_channels // signal_dim
                self.film_layers.append(
                    FiLMLayer(input_dim=metadata_embedding_dim, output_dim=d_per_assay * 2)
                )
            else:
                 # Fallback: simple linear if dimensions don't align cleanly (unlikely with this arch)
                self.film_layers.append(
                    FiLMLayer(input_dim=metadata_embedding_dim, output_dim=layer_channels * 2 // signal_dim)
                )

    def forward(self, src, y_metadata_embed):
        # Apply deconv with per-layer metadata injection
        src = src.permute(0, 2, 1)  # to N, F2, L'
        for i, dconv in enumerate(self.deconv):
            src = dconv(src)
            # Apply metadata FiLM modulation
            # src is [N, C, L']
            # y_metadata_embed is [N, F, emb_dim]
            src = self.film_layers[i](src, y_metadata_embed)
            
        src = src.permute(0, 2, 1)  # final permute to N, L, F1
        return src    

class QueryDeconvStage(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, pool_size=2, norm="layer"):
        super().__init__()
        self.pool_size = pool_size
        # Keep decoder behavior deconvolution-based (transpose conv upsampling),
        # then refine with a same-resolution conv.
        self.deconv = nn.ConvTranspose1d(
            in_ch,
            out_ch,
            kernel_size=pool_size,
            stride=pool_size
        )
        self.conv = nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2)
        if norm in ["layer", "rms"]:
            self.norm = nn.GroupNorm(1, out_ch)
        elif norm == "batch":
            self.norm = nn.BatchNorm1d(out_ch)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        x = self.deconv(x)
        x = self.conv(x)
        x = self.norm(x)
        return F.gelu(x)

class CNP_MoE_Decoder(nn.Module):
    """
    Strict query decoder: f(context, query) -> track features.
    Context is provided per-query as z_q = z[pair_b], query is q[pair_b, pair_f].
    """
    def __init__(self, context_dim, query_dim, n_cnn_layers, context_length, pool_size=2,
                 expansion_factor=3, norm="layer", moe_experts=4, hidden_dim=64):
        super().__init__()
        self.context_length = context_length
        self.n_cnn_layers = n_cnn_layers
        self.pool_size = pool_size
        self.moe_experts = max(1, int(moe_experts))
        self.kernel_sizes = self._build_kernel_sizes(self.moe_experts)

        stage_channels = [context_dim]
        for i in range(n_cnn_layers):
            stage_channels.append(max(hidden_dim, context_dim // (expansion_factor ** (i + 1))))

        self.stage_experts = nn.ModuleList()
        self.stage_gates = nn.ModuleList()
        self.stage_film = nn.ModuleList()

        for i in range(n_cnn_layers):
            in_ch, out_ch = stage_channels[i], stage_channels[i + 1]
            experts = nn.ModuleList([
                QueryDeconvStage(in_ch, out_ch, kernel_size=k, pool_size=pool_size, norm=norm)
                for k in self.kernel_sizes
            ])
            self.stage_experts.append(experts)
            self.stage_gates.append(nn.Linear(query_dim, self.moe_experts))
            self.stage_film.append(nn.Linear(query_dim, out_ch * 2))

        self.out_dim = stage_channels[-1]

    @staticmethod
    def _build_kernel_sizes(n_experts):
        # User-approved schedule: base [3,5,7,9,11], then cycle duplicates from 3 upward.
        base = [3, 5, 7, 9, 11]
        if n_experts <= 5:
            return base[:n_experts]
        extras = n_experts - 5
        cyc = [3, 5, 7, 9, 11]
        ext = [cyc[i % len(cyc)] for i in range(extras)]
        return sorted(base + ext)

    @staticmethod
    def _apply_film(x, film_params):
        # x: [Nq, C, L], film_params: [Nq, 2C]
        Nq, C, _ = x.shape
        scale, shift = film_params.chunk(2, dim=-1)
        scale = torch.clamp(scale, min=-4.0, max=4.0).view(Nq, C, 1)
        shift = shift.view(Nq, C, 1)
        return x * torch.exp(scale) + shift

    def forward(self, z_q, q_q):
        """
        z_q: [Nq, L', Cctx], q_q: [Nq, Eq]
        returns hidden: [Nq, L, H]
        """
        x = z_q.permute(0, 2, 1)  # [Nq, Cctx, L']
        for i, experts in enumerate(self.stage_experts):
            eouts = torch.stack([e(x) for e in experts], dim=1)  # [Nq, E, C, L]
            gate = torch.softmax(self.stage_gates[i](q_q), dim=-1).unsqueeze(-1).unsqueeze(-1)  # [Nq,E,1,1]
            x = (eouts * gate).sum(dim=1)  # [Nq, C, L]
            x = self._apply_film(x, self.stage_film[i](q_q))
        if x.shape[-1] != self.context_length:
            x = F.interpolate(x, size=self.context_length, mode="nearest")
        return x.permute(0, 2, 1)  # [Nq, L, H]

class CNP_DynConv_Decoder(nn.Module):
    """
    Strict query decoder with FiLM-like activation modulation at each stage.
    """
    def __init__(self, context_dim, query_dim, n_cnn_layers, context_length, pool_size=2,
                 expansion_factor=3, norm="layer", hidden_dim=64):
        super().__init__()
        self.context_length = context_length
        stage_channels = [context_dim]
        for i in range(n_cnn_layers):
            stage_channels.append(max(hidden_dim, context_dim // (expansion_factor ** (i + 1))))

        self.stages = nn.ModuleList()
        self.stage_film = nn.ModuleList()
        for i in range(n_cnn_layers):
            in_ch, out_ch = stage_channels[i], stage_channels[i + 1]
            self.stages.append(QueryDeconvStage(in_ch, out_ch, kernel_size=3, pool_size=pool_size, norm=norm))
            self.stage_film.append(nn.Linear(query_dim, out_ch * 2))

        self.out_dim = stage_channels[-1]

    @staticmethod
    def _apply_film(x, film_params):
        Nq, C, _ = x.shape
        scale, shift = film_params.chunk(2, dim=-1)
        scale = torch.clamp(scale, min=-4.0, max=4.0).view(Nq, C, 1)
        shift = shift.view(Nq, C, 1)
        return x * torch.exp(scale) + shift

    def forward(self, z_q, q_q):
        x = z_q.permute(0, 2, 1)  # [Nq, Cctx, L']
        for i, stage in enumerate(self.stages):
            x = stage(x)
            x = self._apply_film(x, self.stage_film[i](q_q))
        if x.shape[-1] != self.context_length:
            x = F.interpolate(x, size=self.context_length, mode="nearest")
        return x.permute(0, 2, 1)  # [Nq, L, H]

class CNP_CondConv_Decoder(nn.Module):
    """
    Query decoder with weight-space CondConv mixing.
    Supports query-only, feature-only, or hybrid routing inputs.
    Mixing applies to stage conv weights only; deconv upsampling weights remain fixed.
    """
    def __init__(self, context_dim, query_dim, n_cnn_layers, context_length, pool_size=2,
                 expansion_factor=3, norm="layer", condconv_k=3, hidden_dim=64,
                 conv_kernel_size=3,
                 condconv_routing="hybrid", condconv_gate_activation="sigmoid"):
        super().__init__()
        self.context_length = context_length
        self.condconv_k = max(1, int(condconv_k))
        self.conv_kernel_size = None if conv_kernel_size is None else max(1, int(conv_kernel_size))
        self.condconv_routing = str(condconv_routing).lower()
        self.condconv_gate_activation = str(condconv_gate_activation).lower()
        if self.condconv_routing not in {"query", "feature", "hybrid"}:
            raise ValueError(
                f"Unsupported condconv_routing: {condconv_routing}. "
                f"Expected one of ['query', 'feature', 'hybrid']."
            )
        if self.condconv_gate_activation not in {"sigmoid", "softmax"}:
            raise ValueError(
                f"Unsupported condconv_gate_activation: {condconv_gate_activation}. "
                f"Expected one of ['sigmoid', 'softmax']."
            )
        self.basis_kernel_sizes = self._build_basis_kernel_sizes(self.condconv_k, self.conv_kernel_size)
        self.max_basis_kernel = max(self.basis_kernel_sizes)

        stage_channels = [context_dim]
        for i in range(n_cnn_layers):
            stage_channels.append(max(hidden_dim, context_dim // (expansion_factor ** (i + 1))))

        self.deconvs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.conv_weight_basis = nn.ParameterList()
        self.conv_bias_basis = nn.ParameterList()
        self.register_buffer("basis_kernel_mask", self._build_basis_kernel_mask(self.basis_kernel_sizes))

        for i in range(n_cnn_layers):
            in_ch, out_ch = stage_channels[i], stage_channels[i + 1]
            self.deconvs.append(
                nn.ConvTranspose1d(in_ch, out_ch, kernel_size=pool_size, stride=pool_size)
            )
            if norm in ["layer", "rms"]:
                self.norms.append(nn.GroupNorm(1, out_ch))
            elif norm == "batch":
                self.norms.append(nn.BatchNorm1d(out_ch))
            else:
                self.norms.append(nn.Identity())

            if self.condconv_routing == "query":
                gate_in_dim = query_dim
            elif self.condconv_routing == "feature":
                gate_in_dim = out_ch
            else:
                gate_in_dim = query_dim + out_ch
            self.gates.append(nn.Linear(gate_in_dim, self.condconv_k))

            # Basis conv kernels (weight-space mixture): [K, Cout, Cin, Kmax]
            weight_basis = nn.Parameter(
                torch.empty(self.condconv_k, out_ch, out_ch, self.max_basis_kernel)
            )
            bias_basis = nn.Parameter(torch.zeros(self.condconv_k, out_ch))
            nn.init.kaiming_uniform_(weight_basis, a=math.sqrt(5))
            self.conv_weight_basis.append(weight_basis)
            self.conv_bias_basis.append(bias_basis)

        self.out_dim = stage_channels[-1]

    @staticmethod
    def _build_basis_kernel_sizes(k, kernel_size):
        k = max(1, int(k))
        if kernel_size is None:
            # Legacy mode: repeat-cycle odd kernels up to 11.
            # Examples:
            #   K=2 -> [3,5]
            #   K=3 -> [3,5,7]
            #   K=7 -> [3,3,5,5,7,9,11]
            #   K=8 -> [3,3,5,5,7,7,9,11]
            base = [3, 5, 7, 9, 11]
            if k <= len(base):
                return base[:k]
            extras = k - len(base)
            cyc = [3, 5, 7]
            ext = [cyc[i % len(cyc)] for i in range(extras)]
            return sorted(base + ext)
        # New default mode: all basis kernels share the CANDI conv kernel size.
        return [max(1, int(kernel_size)) for _ in range(k)]

    @staticmethod
    def _build_basis_kernel_mask(kernel_sizes):
        kmax = max(kernel_sizes)
        mask = torch.zeros(len(kernel_sizes), 1, 1, kmax)
        center = kmax // 2
        for i, ks in enumerate(kernel_sizes):
            half = ks // 2
            mask[i, 0, 0, center - half:center + half + 1] = 1.0
        return mask

    def _condconv1d(self, x, gate, weight_basis, bias_basis):
        """
        x: [Nq, C, L], gate: [Nq, K], weight_basis: [K, C, C, W], bias_basis: [K, C]
        Applies per-query mixed conv using grouped conv.
        """
        Nq, C, L = x.shape
        K = gate.shape[-1]
        W = weight_basis.shape[-1]
        masked_basis = weight_basis * self.basis_kernel_mask.to(weight_basis.dtype)

        mixed_w = torch.einsum("nk,kocw->nocw", gate, masked_basis)  # [Nq, C, C, W]
        mixed_b = torch.einsum("nk,kc->nc", gate, bias_basis)        # [Nq, C]

        xg = x.reshape(1, Nq * C, L)
        wg = mixed_w.reshape(Nq * C, C, W)
        bg = mixed_b.reshape(Nq * C)
        yg = F.conv1d(xg, wg, bias=bg, padding=W // 2, groups=Nq)
        y = yg.reshape(Nq, C, yg.shape[-1])
        return y

    def _compute_gates(self, gate_logits):
        if self.condconv_gate_activation == "softmax":
            return torch.softmax(gate_logits, dim=-1)
        return torch.sigmoid(gate_logits)

    def forward(self, z_q, q_q):
        x = z_q.permute(0, 2, 1)  # [Nq, Cctx, L']
        for i in range(len(self.deconvs)):
            x = self.deconvs[i](x)
            x_pool = x.mean(dim=-1)  # [Nq, C]
            if self.condconv_routing == "query":
                gate_in = q_q
            elif self.condconv_routing == "feature":
                gate_in = x_pool
            else:
                gate_in = torch.cat([q_q, x_pool], dim=-1)
            gate = self._compute_gates(self.gates[i](gate_in))  # [Nq, K]
            x = self._condconv1d(
                x, gate, self.conv_weight_basis[i], self.conv_bias_basis[i]
            )
            x = self.norms[i](x)
            x = F.gelu(x)
        if x.shape[-1] != self.context_length:
            x = F.interpolate(x, size=self.context_length, mode="nearest")
        return x.permute(0, 2, 1)  # [Nq, L, H]

class CANDI_DNA_Encoder(nn.Module):
    def __init__(self, 
        signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, nhead, n_sab_layers, pool_size=2, 
        dropout=0.1, context_length=1600, pos_enc="relative", expansion_factor=3, norm="layer", attention_type="dual", xl_dna=False, mask_stem=False):

        super(CANDI_DNA_Encoder, self).__init__()

        self.pos_enc = pos_enc
        self.l1 = context_length
        self.l2 = self.l1 // (pool_size**n_cnn_layers)
        
        self.f1 = signal_dim 
        self.f2 = (self.f1 * (expansion_factor**(n_cnn_layers)))
        self.f3 = self.f2 + metadata_embedding_dim
        d_model = self.f2
        self.latent_dim = self.f2
        
        # MaskStem for handling missing data
        self.use_mask_stem = mask_stem
        if self.use_mask_stem:
            self.mask_stem = MaskStem(n_channels=signal_dim, missing_token=-1)

        # DNA Encoder Configuration
        if xl_dna:
            # XL DNA Encoder: Wider channels with biophysical kernels
            # Kernel Size: 15bp for initial layer (complete TF motifs), then 5bp for syntax
            # Channels: Exponential growth starting at 96
            DNA_kernel_size = [15] + [5 for _ in range(n_cnn_layers + 1)]
            tower_channels = [96 * (2 ** l) for l in range(n_cnn_layers + 2)]
            DNA_conv_channels = [4] + tower_channels
        else:
            # Standard DNA Encoder: Matches signal encoder dimensionality
            DNA_kernel_size = [conv_kernel_size for _ in range(n_cnn_layers + 2)]
            start_channels = 4
            tower_channels = exponential_linspace_int(start_channels, self.f2, n_cnn_layers + 2)
            DNA_conv_channels = [4] + tower_channels

        self.convEncDNA = nn.ModuleList(
            [ConvTower(
                DNA_conv_channels[i], DNA_conv_channels[i + 1],
                DNA_kernel_size[i], S=1, D=1,
                pool_type="max", residuals=True, SE=False,
                groups=1, pool_size=5 if i >= n_cnn_layers else pool_size, norm=norm) for i in range(n_cnn_layers + 2)])

        conv_channels = [(self.f1)*(expansion_factor**l) for l in range(n_cnn_layers)]
        reverse_conv_channels = [expansion_factor * x for x in conv_channels[::-1]]
        conv_kernel_size_list = [conv_kernel_size for _ in range(n_cnn_layers)]
        
        self.convEnc = nn.ModuleList(
            [ConvTower(
                conv_channels[i], conv_channels[i + 1] if i + 1 < n_cnn_layers else expansion_factor * conv_channels[i],
                conv_kernel_size_list[i], S=1, D=1,
                pool_type="max", residuals=True,
                groups=self.f1, SE=False,
                pool_size=pool_size, norm=norm) for i in range(n_cnn_layers)])
        
        # Per-layer FiLM Adapter for X-side metadata
        self.film_layers = nn.ModuleList()
        for i in range(n_cnn_layers):
            layer_channels = conv_channels[i + 1] if i + 1 < n_cnn_layers else expansion_factor * conv_channels[i]
            # Assumes layer_channels is divisible by signal_dim (f1)
            d_per_assay = layer_channels // self.f1
            self.film_layers.append(
                FiLMLayer(input_dim=metadata_embedding_dim, output_dim=d_per_assay * 2)
            )

        # Store attention type for reference
        self.attention_type = attention_type

        # Validate x-transformers availability if needed
        if attention_type == "xtransformers" and not XTRANSFORMERS_AVAILABLE:
            raise ImportError(
                "x-transformers library is required for attention_type='xtransformers' but not installed. "
                "Install it with: pip install x-transformers"
            )

        # Linear fusion: concat + linear projection (no residuals)
        self.fusion = LinearFusion(
            signal_dim=self.f2, 
            dna_dim=DNA_conv_channels[-1], 
            output_dim=self.f2, 
            dropout=dropout
        )

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

    def forward(self, src, seq, x_metadata_embed):
        if len(seq.shape) != len(src.shape):
            seq = seq.unsqueeze(0).expand(src.shape[0], -1, -1)

        seq = seq.permute(0, 2, 1)  # to N, 4, 25*L
        seq = seq.float()

        ### DNA CONV ENCODER ###
        for seq_conv in self.convEncDNA:
            seq = seq_conv(seq)
        seq = seq.permute(0, 2, 1)  # to N, L', F2
        
        ### SIGNAL CONV ENCODER WITH PER-LAYER METADATA INJECTION ###
        src = src.permute(0, 2, 1)  # to N, F1, L
        
        # Apply MaskStem if enabled (handles missing data before convolutions)
        if self.use_mask_stem:
            src = self.mask_stem(src)
        
        for i, conv in enumerate(self.convEnc):
            src = conv(src)
            # Apply metadata FiLM modulation
            # src is [N, C, L]
            # x_metadata_embed is [N, F1, emb_dim]
            src = self.film_layers[i](src, x_metadata_embed)
            
        src = src.permute(0, 2, 1)  # final permute to N, L', F2

        ### FUSION (signal queries DNA) ###
        src = self.fusion(signal=src, dna=seq)  # [N, L', F2]

        ### TRANSFORMER ENCODER ###
        for enc in self.transformer_encoder:
            src = enc(src)

        return src

class CANDI(nn.Module):
    def __init__(self, signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, nhead,
        n_sab_layers, pool_size=2, dropout=0.1, context_length=1600, pos_enc="relative", 
        expansion_factor=3, separate_decoders=True, num_assays=35, num_runtypes=2, 
        norm="layer", attention_type="dual", output_ff=False, dist_type="gaussian", xl_dna=False, mask_stem=False,
        signal_transform="arcsinh", decoder_type="fixed", moe_experts=4, nq_chunk_multiplier=1, condconv_k=3,
        condconv_routing="hybrid", condconv_gate_activation="sigmoid", condconv_conv_kernel_size="shared",
        enable_latent_kl=False, latent_std_min=0.01, latent_std_max=1.0, latent_reparam_mode="clamp",
        latent_sample_train_only=True, latent_deterministic_warmup_steps=0):
        """
        CANDI model for epigenomic signal imputation.
        
        Args:
            num_assays: Number of distinct assay types (e.g., H3K4me3, CTCF).
                       Used for assay_embedding in MetadataEncoder.
                       Note: replaces num_sequencing_platforms per issue_supertrack.md ToDo 1.
        """
        super(CANDI, self).__init__()

        self.pos_enc = pos_enc
        self.separate_decoders = separate_decoders
        self.dist_type = dist_type
        self.decoder_type = decoder_type
        self.moe_experts = int(moe_experts)
        self.nq_chunk_multiplier = max(1, int(nq_chunk_multiplier))
        self.condconv_k = max(1, int(condconv_k))
        self.condconv_routing = str(condconv_routing).lower()
        self.condconv_gate_activation = str(condconv_gate_activation).lower()
        self.condconv_conv_kernel_size = condconv_conv_kernel_size
        self.mask_stem = mask_stem
        self.signal_transform = signal_transform
        self.enable_latent_kl = bool(enable_latent_kl)
        self.latent_std_min = float(latent_std_min)
        self.latent_std_max = float(latent_std_max)
        self.latent_reparam_mode = str(latent_reparam_mode).lower()
        self.latent_sample_train_only = bool(latent_sample_train_only)
        self.latent_deterministic_warmup_steps = int(latent_deterministic_warmup_steps)
        if self.latent_std_min <= 0 or self.latent_std_max <= 0 or self.latent_std_min >= self.latent_std_max:
            raise ValueError(
                f"Invalid latent std bounds: min={self.latent_std_min}, max={self.latent_std_max}. "
                "Require 0 < min < max."
            )
        if self.latent_reparam_mode not in {"clamp", "softplus"}:
            raise ValueError(f"Unsupported latent_reparam_mode={self.latent_reparam_mode}")
        self.l1 = context_length
        self.l2 = self.l1 // (pool_size**n_cnn_layers)
        
        self.f1 = signal_dim 
        self.f2 = (self.f1 * (expansion_factor**(n_cnn_layers)))
        self.f3 = self.f2 + metadata_embedding_dim
        self.d_model = self.latent_dim = self.f2

        # Separate Metadata Encoders for X (encoder) and Y (decoder)
        # X has signal_dim+1 assays (includes control), Y has signal_dim assays
        self.x_metadata_encoder = MetadataEncoder(
            num_assays=num_assays, 
            num_runtypes=num_runtypes, 
            embed_dim=metadata_embedding_dim
        )
        self.y_query_metadata_encoder = QueryMetadataEncoder(
            num_assays=num_assays, 
            num_runtypes=num_runtypes, 
            embed_dim=metadata_embedding_dim
        )
        # Backward-compatible alias used in existing code paths.
        self.y_metadata_encoder = self.y_query_metadata_encoder

        self.encoder = CANDI_DNA_Encoder(signal_dim+1, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, nhead,
            n_sab_layers, pool_size, dropout, context_length, pos_enc, expansion_factor, norm, attention_type=attention_type, xl_dna=xl_dna, mask_stem=mask_stem)

        self.latent_projection = nn.Sequential(
            nn.Linear(
                ((signal_dim+1) * (expansion_factor**(n_cnn_layers))), 
                signal_dim * (expansion_factor**(n_cnn_layers)),
            ),
            nn.GELU(),
            nn.LayerNorm(signal_dim * (expansion_factor**(n_cnn_layers)))
        )

        if self.enable_latent_kl:
            # Keep shared-module initialization invariant when toggling latent KL.
            # The latent heads are fully reinitialized below, so restoring the RNG
            # state here prevents them from perturbing decoder/output head init.
            cpu_rng_state = torch.get_rng_state()
            try:
                self.latent_mu_head = nn.Linear(self.latent_dim, self.latent_dim)
                self.latent_logvar_head = nn.Linear(self.latent_dim, self.latent_dim)
            finally:
                torch.set_rng_state(cpu_rng_state)
            # Stabilizing init:
            # - mu head starts as near-identity so decoded latent keeps useful encoder signal.
            # - logvar head starts with low variance to avoid noisy latent sampling at step 0.
            nn.init.eye_(self.latent_mu_head.weight)
            nn.init.zeros_(self.latent_mu_head.bias)
            nn.init.zeros_(self.latent_logvar_head.weight)
            std_init = min(max(0.05, self.latent_std_min), self.latent_std_max)
            nn.init.constant_(self.latent_logvar_head.bias, 2.0 * math.log(std_init))
        else:
            self.latent_mu_head = None
            self.latent_logvar_head = None

        self._last_latent_kl = None
        self._last_latent_stats = {}
        self._latent_global_step = 0
        self._latent_force_deterministic_train = False
        self._latent_posterior_heads_frozen = False
        self._latent_blend_alpha_train = 0.0
        self._latent_enable_sampling_train = True
        
        self.query_decoder = decoder_type != "fixed"
        if not self.query_decoder:
            if self.separate_decoders:
                self.count_decoder = CANDI_Decoder(signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, context_length, pool_size, expansion_factor, norm)
                self.pval_decoder = CANDI_Decoder(signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, context_length, pool_size, expansion_factor, norm)
                self.peak_decoder = CANDI_Decoder(signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, context_length, pool_size, expansion_factor, norm)
            else:
                self.decoder = CANDI_Decoder(signal_dim, metadata_embedding_dim, conv_kernel_size, n_cnn_layers, context_length, pool_size, expansion_factor, norm)
            self.neg_binom_layer = NegativeBinomialLayer(self.f1, self.f1, FF=output_ff)
        else:
            if decoder_type == "query_moe":
                q_decoder_cls = CNP_MoE_Decoder
                q_kwargs = {"moe_experts": self.moe_experts}
            elif decoder_type == "query_dynconv":
                q_decoder_cls = CNP_DynConv_Decoder
                q_kwargs = {}
            elif decoder_type == "query_condconv":
                q_decoder_cls = CNP_CondConv_Decoder
                if condconv_conv_kernel_size == "shared":
                    condconv_kernel = conv_kernel_size
                else:
                    condconv_kernel = condconv_conv_kernel_size
                q_kwargs = {
                    "condconv_k": self.condconv_k,
                    "conv_kernel_size": condconv_kernel,
                    "condconv_routing": self.condconv_routing,
                    "condconv_gate_activation": self.condconv_gate_activation,
                }
            else:
                raise ValueError(f"Unsupported decoder_type: {decoder_type}")

            context_dim = signal_dim * (expansion_factor ** n_cnn_layers)
            if self.separate_decoders:
                self.count_decoder = q_decoder_cls(context_dim, metadata_embedding_dim, n_cnn_layers, context_length, pool_size, expansion_factor, norm, **q_kwargs)
                self.pval_decoder = q_decoder_cls(context_dim, metadata_embedding_dim, n_cnn_layers, context_length, pool_size, expansion_factor, norm, **q_kwargs)
                self.peak_decoder = q_decoder_cls(context_dim, metadata_embedding_dim, n_cnn_layers, context_length, pool_size, expansion_factor, norm, **q_kwargs)
                h_count = self.count_decoder.out_dim
                h_pval = self.pval_decoder.out_dim
                h_peak = self.peak_decoder.out_dim
            else:
                self.decoder = q_decoder_cls(context_dim, metadata_embedding_dim, n_cnn_layers, context_length, pool_size, expansion_factor, norm, **q_kwargs)
                h_count = h_pval = h_peak = self.decoder.out_dim

            # Query heads: map per-query hidden tracks [Nq,L,H] -> scalar outputs [Nq,L].
            self.query_count_mu_head = nn.Linear(h_count, 1)
            self.query_count_n_head = nn.Linear(h_count, 1)
            self.query_pval_mu_head = nn.Linear(h_pval, 1)
            self.query_pval_scale_head = nn.Linear(h_pval, 1)
            if self.dist_type == 'studentst':
                self.query_pval_df_head = nn.Linear(h_pval, 1)
            else:
                self.query_pval_df_head = None
            self.query_peak_head = nn.Linear(h_peak, 1)

        # Signal layer based on distribution type
        # For deterministic losses (mse, mae), use matching probabilistic layer to keep architecture identical
        if self.dist_type in ['laplace', 'mae']:
            self.signal_layer = LaplacianLayer(self.f1, self.f1, FF=output_ff)
        elif self.dist_type == 'laplace_const':
            self.signal_layer = LaplacianLayerConstantScale(self.f1, self.f1, FF=output_ff)
        elif self.dist_type == 'studentst':
            self.signal_layer = StudentsTLayer(self.f1, self.f1, FF=output_ff)
        elif self.dist_type == 'gamma':
            self.signal_layer = GammaLayer(self.f1, self.f1, FF=output_ff)
        elif self.dist_type == 'gaussian_const':
            self.signal_layer = GaussianLayerConstantVar(self.f1, self.f1, FF=output_ff)
        else:
            # gaussian or mse
            self.signal_layer = GaussianLayer(self.f1, self.f1, FF=output_ff)
        self.peak_layer = PeakLayer(self.f1, self.f1, FF=output_ff)
    
    def encode(self, src, seq, x_metadata, apply_arcsinh_transform=True):
        """Encode input data into latent representation.
        
        Args:
            src: Source data tensor [B, L, F+1] (includes control)
            seq: Sequence data
            x_metadata: Metadata tensor [B, 4, F+1] (includes control metadata)
            apply_arcsinh_transform: Deprecated. Use self.signal_transform instead.
        
        Note: -2 (cloze) is now passed through to MetadataEncoder for distinct handling.
              Previously, -2 was converted to -1 here (removed per S3 in issue_supertrack.md).
        """
        # Apply transformation to non-missing values based on self.signal_transform
        # apply_arcsinh_transform is kept for backward compatibility but is deprecated
        # Note: Both -1 (missing) and -2 (cloze) are treated as special tokens, not transformed
        if apply_arcsinh_transform:
            mask = (src != -1) & (src != -2)
            if self.signal_transform == 'arcsinh':
                src = torch.where(mask, torch.arcsinh(src), src)
            elif self.signal_transform == 'log1p':
                src = torch.where(mask, torch.log1p(src), src)
            # else: 'none' - no transformation
        
        # Embed X metadata (includes control, F+1 assays)
        # MetadataEncoder now handles -1 (missing) and -2 (cloze) distinctly
        x_metadata_embed = self.x_metadata_encoder(x_metadata)
        
        z = self.encoder(src, seq, x_metadata_embed)
        return z

    def get_last_latent_kl(self):
        return self._last_latent_kl

    def get_last_latent_stats(self) -> Dict[str, float]:
        return dict(self._last_latent_stats)

    def _set_latent_posterior_heads_frozen(self, freeze: bool):
        if not self.enable_latent_kl:
            return
        freeze = bool(freeze)
        if freeze == self._latent_posterior_heads_frozen:
            return
        for p in self.latent_mu_head.parameters():
            p.requires_grad = not freeze
        for p in self.latent_logvar_head.parameters():
            p.requires_grad = not freeze
        self._latent_posterior_heads_frozen = freeze

    def set_latent_train_controls(
        self,
        global_step: int,
        force_deterministic_train: bool = False,
        freeze_posterior_heads: bool = False,
        blend_alpha_train: float = 1.0,
        enable_sampling_train: bool = True,
    ):
        self._latent_global_step = int(global_step)
        self._latent_force_deterministic_train = bool(force_deterministic_train)
        self._set_latent_posterior_heads_frozen(bool(freeze_posterior_heads))
        self._latent_blend_alpha_train = float(max(0.0, min(1.0, blend_alpha_train)))
        self._latent_enable_sampling_train = bool(enable_sampling_train)

    def _apply_latent_regularization(self, z):
        """
        Variational latent path with diagonal Gaussian posterior q(z|x).
        Returns latent tensor used for decoding and scalar KL[q||p].
        """
        if not self.enable_latent_kl:
            self._last_latent_kl = z.new_tensor(0.0)
            self._last_latent_stats = {}
            return z, self._last_latent_kl

        mu_z = self.latent_mu_head(z)
        raw = self.latent_logvar_head(z)

        if self.latent_reparam_mode == "clamp":
            logvar_min = 2.0 * math.log(self.latent_std_min)
            logvar_max = 2.0 * math.log(self.latent_std_max)
            logvar_z = torch.clamp(raw, min=logvar_min, max=logvar_max)
            std_z = torch.exp(0.5 * logvar_z)
        else:
            std_z = self.latent_std_min + F.softplus(raw)
            std_z = torch.clamp(std_z, min=self.latent_std_min, max=self.latent_std_max)
            logvar_z = 2.0 * torch.log(std_z)

        use_eval_posterior_mean = self.latent_sample_train_only and (not self.training)
        if self.training and self._latent_force_deterministic_train:
            # Deterministic parity path during warmup/debug mode: decode from raw projected latent.
            z_used = z
        elif self.training and (not self._latent_enable_sampling_train):
            # Phase-B bridge: smoothly transition decode context from raw z to posterior mean mu_z.
            a = self._latent_blend_alpha_train
            z_used = (1.0 - a) * z + a * mu_z
        elif use_eval_posterior_mean:
            z_used = mu_z
        else:
            eps = torch.randn_like(std_z)
            z_used = mu_z + std_z * eps

        # KL[q(z|x)||p(z)] with p(z)=N(0, I), diagonal q.
        kl_per_token = 0.5 * (mu_z.pow(2) + torch.exp(logvar_z) - 1.0 - logvar_z)
        kl_loss = kl_per_token.sum(dim=-1).mean()

        self._last_latent_kl = kl_loss
        with torch.no_grad():
            self._last_latent_stats = {
                "latent_std_min_seen": float(std_z.min().detach().item()),
                "latent_std_max_seen": float(std_z.max().detach().item()),
                "latent_mu_abs_mean": float(mu_z.abs().mean().detach().item()),
                "latent_logvar_mean": float(logvar_z.mean().detach().item()),
            }
        return z_used, kl_loss
    
    @staticmethod
    def _mask_to_pairs(mask):
        if mask is None:
            return None, None
        if mask.dim() != 2:
            raise ValueError(f"query mask must be [B,F], got shape {tuple(mask.shape)}")
        pair_b, pair_f = torch.nonzero(mask, as_tuple=True)
        return pair_b, pair_f

    @staticmethod
    def _scatter_sparse_tracks(sparse_tracks, pair_b, pair_f, B, L, F, fill_value=-1.0):
        """
        Scatter [Nq, L] -> [B, L, F] using explicit (sample, assay) pairs.
        """
        out = sparse_tracks.new_full((B, L, F), fill_value=fill_value)
        if pair_b is None or pair_b.numel() == 0:
            return out
        if pair_b.numel() != pair_f.numel():
            raise ValueError("pair_b and pair_f must have same length")
        if pair_b.min() < 0 or pair_b.max() >= B or pair_f.min() < 0 or pair_f.max() >= F:
            raise ValueError("scatter indices out of bounds")
        out[pair_b, :, pair_f] = sparse_tracks
        return out

    @staticmethod
    def _scatter_sparse_tracks_inplace(out, sparse_tracks, pair_b, pair_f):
        """
        In-place scatter [Nq, L] into existing dense [B, L, F].
        """
        if pair_b is None or pair_b.numel() == 0:
            return
        if pair_b.numel() != pair_f.numel():
            raise ValueError("pair_b and pair_f must have same length")
        B, _, F = out.shape
        if pair_b.min() < 0 or pair_b.max() >= B or pair_f.min() < 0 or pair_f.max() >= F:
            raise ValueError("scatter indices out of bounds")
        if sparse_tracks.dtype != out.dtype:
            sparse_tracks = sparse_tracks.to(out.dtype)
        out[pair_b, :, pair_f] = sparse_tracks

    def _project_with_query_mask(self, decoded, query_mask, fill_value=-1.0):
        """
        Project dense [B,L,F] predictions to queried lanes only using sentinel fill elsewhere.
        """
        B, L, F = decoded.shape
        if query_mask is None:
            return decoded
        pair_b, pair_f = self._mask_to_pairs(query_mask)
        if pair_b is None or pair_b.numel() == 0:
            return decoded.new_full((B, L, F), fill_value=fill_value)
        sparse = decoded[pair_b, :, pair_f]
        return self._scatter_sparse_tracks(sparse, pair_b, pair_f, B, L, F, fill_value=fill_value)

    def decode(self, z, y_metadata, query_mask=None, query_mask_signal=None):
        """Decode latent representation into predictions.
        
        Args:
            z: Latent tensor
            y_metadata: Metadata tensor [B, 4, F] (no control, just signal_dim assays)
        
        Returns:
            p, n: Negative binomial parameters
            mu, scale: Signal distribution parameters (var for Gaussian, log_b for Laplace, sigma for StudentT)
            df: Degrees of freedom for Student's t (None for Gaussian/Laplace)
            peak: Peak predictions
        
        Note: -2 (cloze) is now passed through to MetadataEncoder for distinct handling.
              This allows the model to distinguish "predict this assay" (-2) from "missing" (-1).
              Previously, -2 was converted to -1 here (removed per S3 in issue_supertrack.md).
        """
        y_metadata_embed = self.y_query_metadata_encoder(y_metadata)
        B, _, num_assays = y_metadata.shape
        max_nq_per_chunk = 16 # max(1, int(self.nq_chunk_multiplier) * int(B))

        # Legacy fixed decoder path stays unchanged for strict backward compatibility.
        if not self.query_decoder:
            if self.separate_decoders:
                count_decoded = self.count_decoder(z, y_metadata_embed)
                pval_decoded = self.pval_decoder(z, y_metadata_embed)
                peak_decoded = self.peak_decoder(z, y_metadata_embed)
                p, n = self.neg_binom_layer(count_decoded)
                if self.dist_type == 'studentst':
                    mu, scale, df = self.signal_layer(pval_decoded)
                else:
                    mu, scale = self.signal_layer(pval_decoded)
                    df = None
                peak = self.peak_layer(peak_decoded)
            else:
                decoded = self.decoder(z, y_metadata_embed)
                p, n = self.neg_binom_layer(decoded)
                if self.dist_type == 'studentst':
                    mu, scale, df = self.signal_layer(decoded)
                else:
                    mu, scale = self.signal_layer(decoded)
                    df = None
                peak = self.peak_layer(decoded)
            return p, n, mu, scale, df, peak

        # Query decoder path: strict f(context, query)->track.
        if query_mask is None:
            query_mask = torch.ones((B, num_assays), dtype=torch.bool, device=z.device)
        if query_mask_signal is None:
            query_mask_signal = query_mask

        pair_b_c, pair_f_c = self._mask_to_pairs(query_mask)
        pair_b_s, pair_f_s = self._mask_to_pairs(query_mask_signal)

        L_full = self.l1
        # Initialize dense outputs with sentinels.
        p = z.new_full((B, L_full, num_assays), -1.0)
        n = z.new_full((B, L_full, num_assays), -1.0)
        mu = z.new_full((B, L_full, num_assays), -1.0)
        scale = z.new_full((B, L_full, num_assays), -1.0)
        peak = z.new_full((B, L_full, num_assays), -1.0)
        df = z.new_full((B, L_full, num_assays), -1.0) if self.dist_type == 'studentst' else None

        if self.separate_decoders:
            # Count branch
            if pair_b_c.numel() > 0:
                Nqc = pair_b_c.numel()
                for start in range(0, Nqc, max_nq_per_chunk):
                    end = min(start + max_nq_per_chunk, Nqc)
                    pb = pair_b_c[start:end]
                    pf = pair_f_c[start:end]
                    h_c = self.count_decoder(z[pb], y_metadata_embed[pb, pf])  # [Nc, L, H]
                    mu_c = F.softplus(self.query_count_mu_head(h_c).squeeze(-1)) + 1e-6
                    n_c = F.softplus(self.query_count_n_head(h_c).squeeze(-1)) + 1e-6
                    p_c = torch.clamp(n_c / (n_c + mu_c), min=1e-6, max=1.0 - 1e-6)
                    self._scatter_sparse_tracks_inplace(p, p_c, pb, pf)
                    self._scatter_sparse_tracks_inplace(n, n_c, pb, pf)
                    # print(f"Count branch: start: {start}, end: {end}, Nqc: {Nqc}, max_nq_per_chunk: {max_nq_per_chunk}")

            # Signal branch
            if pair_b_s.numel() > 0:
                Nqs = pair_b_s.numel()
                for start in range(0, Nqs, max_nq_per_chunk):
                    end = min(start + max_nq_per_chunk, Nqs)
                    pb = pair_b_s[start:end]
                    pf = pair_f_s[start:end]
                    h_s = self.pval_decoder(z[pb], y_metadata_embed[pb, pf])  # [Ns, L, H]
                    mu_s = F.softplus(self.query_pval_mu_head(h_s).squeeze(-1))
                    scale_s = self.query_pval_scale_head(h_s).squeeze(-1)
                    if self.dist_type in ['gaussian', 'gaussian_const', 'mse']:
                        scale_s = F.softplus(scale_s) + 1e-6
                    elif self.dist_type in ['laplace', 'laplace_const', 'mae']:
                        scale_s = torch.clamp(scale_s, min=-10.0, max=10.0)
                    elif self.dist_type == 'studentst':
                        scale_s = F.softplus(scale_s) + 1e-6
                    elif self.dist_type == 'gamma':
                        scale_s = F.softplus(scale_s) + 1e-6
                    self._scatter_sparse_tracks_inplace(mu, mu_s, pb, pf)
                    self._scatter_sparse_tracks_inplace(scale, scale_s, pb, pf)
                    if self.dist_type == 'studentst':
                        df_s = 2.0 + F.softplus(self.query_pval_df_head(h_s).squeeze(-1))
                        df_s = torch.clamp(df_s, min=2.01, max=100.0)
                        self._scatter_sparse_tracks_inplace(df, df_s, pb, pf)
                    # print(f"Signal branch: start: {start}, end: {end}, Nqs: {Nqs}, max_nq_per_chunk: {max_nq_per_chunk}")

            # Peak branch
            if pair_b_s.numel() > 0:
                Nqs = pair_b_s.numel()
                for start in range(0, Nqs, max_nq_per_chunk):
                    end = min(start + max_nq_per_chunk, Nqs)
                    pb = pair_b_s[start:end]
                    pf = pair_f_s[start:end]
                    h_k = self.peak_decoder(z[pb], y_metadata_embed[pb, pf])  # [Ns, L, H]
                    peak_s = torch.sigmoid(self.query_peak_head(h_k).squeeze(-1))
                    self._scatter_sparse_tracks_inplace(peak, peak_s, pb, pf)
                    # print(f"Peak branch: start: {start}, end: {end}, Nqs: {Nqs}, max_nq_per_chunk: {max_nq_per_chunk}")

        else:
            if pair_b_c.numel() > 0:
                Nqc = pair_b_c.numel()
                for start in range(0, Nqc, max_nq_per_chunk):
                    end = min(start + max_nq_per_chunk, Nqc)
                    pb = pair_b_c[start:end]
                    pf = pair_f_c[start:end]
                    h_c = self.decoder(z[pb], y_metadata_embed[pb, pf])  # [Nc, L, H]
                    mu_c = F.softplus(self.query_count_mu_head(h_c).squeeze(-1)) + 1e-6
                    n_c = F.softplus(self.query_count_n_head(h_c).squeeze(-1)) + 1e-6
                    p_c = torch.clamp(n_c / (n_c + mu_c), min=1e-6, max=1.0 - 1e-6)
                    self._scatter_sparse_tracks_inplace(p, p_c, pb, pf)
                    self._scatter_sparse_tracks_inplace(n, n_c, pb, pf)

            if pair_b_s.numel() > 0:
                Nqs = pair_b_s.numel()
                for start in range(0, Nqs, max_nq_per_chunk):
                    end = min(start + max_nq_per_chunk, Nqs)
                    pb = pair_b_s[start:end]
                    pf = pair_f_s[start:end]
                    h_s = self.decoder(z[pb], y_metadata_embed[pb, pf])  # [Ns, L, H]
                    mu_s = F.softplus(self.query_pval_mu_head(h_s).squeeze(-1))
                    scale_s = self.query_pval_scale_head(h_s).squeeze(-1)
                    if self.dist_type in ['gaussian', 'gaussian_const', 'mse']:
                        scale_s = F.softplus(scale_s) + 1e-6
                    elif self.dist_type in ['laplace', 'laplace_const', 'mae']:
                        scale_s = torch.clamp(scale_s, min=-10.0, max=10.0)
                    elif self.dist_type == 'studentst':
                        scale_s = F.softplus(scale_s) + 1e-6
                    elif self.dist_type == 'gamma':
                        scale_s = F.softplus(scale_s) + 1e-6
                    self._scatter_sparse_tracks_inplace(mu, mu_s, pb, pf)
                    self._scatter_sparse_tracks_inplace(scale, scale_s, pb, pf)
                    if self.dist_type == 'studentst':
                        df_s = 2.0 + F.softplus(self.query_pval_df_head(h_s).squeeze(-1))
                        df_s = torch.clamp(df_s, min=2.01, max=100.0)
                        self._scatter_sparse_tracks_inplace(df, df_s, pb, pf)
                    peak_s = torch.sigmoid(self.query_peak_head(h_s).squeeze(-1))
                    self._scatter_sparse_tracks_inplace(peak, peak_s, pb, pf)

        return p, n, mu, scale, df, peak

    def forward(self, src, seq, x_metadata, y_metadata, availability=None, return_z=False, query_mask=None, query_mask_signal=None):
        z = self.encode(src, seq, x_metadata)

        z = self.latent_projection(z)
        z, _ = self._apply_latent_regularization(z)

        p, n, mu, scale, df, peak = self.decode(z, y_metadata, query_mask=query_mask, query_mask_signal=query_mask_signal)
        
        if return_z:
            return p, n, mu, scale, df, peak, z
        else:
            return p, n, mu, scale, df, peak

class CANDI_UNET(CANDI):
    """
    Archived model placeholder.

    The maintained CANDI_UNET implementation was moved to legacy/__archive__.py.
    """
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "CANDI_UNET has been archived to legacy/__archive__.py and is no longer "
            "maintained in model.py."
        )

#========================================================================================================#
#===========================================Building Blocks==============================================#
#========================================================================================================#

# ---------------------------
# Metadata Encoder & FiLM
# ---------------------------
class MetadataEncoder(nn.Module):
    """
    Encode per-assay metadata into embeddings for FiLM conditioning.
    
    Metadata tensor layout: [B, 4, F] where rows are:
        - index 0: depth_log2 (continuous)
        - index 1: assay_id (categorical, identifies assay type like H3K4me3, CTCF)
        - index 2: read_length (continuous)
        - index 3: run_type (categorical, 0=single-end, 1=paired-end)
    
    Special tokens (per S3 in issue_supertrack.md):
        - -1: missing (data not available)
        - -2: cloze (assay is requested/to be predicted)
    
    These are now handled DISTINCTLY to allow the model to differentiate between
    "this data is missing" vs "predict this assay from context".
    """
    def __init__(self, num_assays=35, num_runtypes=2, embed_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_assays = num_assays
        self.num_runtypes = num_runtypes
        
        # Continuous features: depth and read_length
        self.depth_proj = nn.Linear(1, embed_dim)
        self.read_length_proj = nn.Linear(1, embed_dim)
        
        # Special token embeddings for continuous features (S3: distinct cloze)
        # These allow the model to learn different representations for -1 (missing) vs -2 (cloze)
        self.depth_missing_emb = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.depth_cloze_emb = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.read_length_missing_emb = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.read_length_cloze_emb = nn.Parameter(torch.randn(embed_dim) * 0.02)
        
        # Categorical features: +3 for control, missing (-1), and cloze (-2)
        # Index mapping for assay_id:
        #   0 to num_assays-1: regular assays
        #   num_assays: control
        #   num_assays+1: missing (-1)
        #   num_assays+2: cloze (-2)
        self.assay_embedding = nn.Embedding(num_assays + 3, embed_dim)
        
        # Index mapping for runtype:
        #   0: single-end, 1: paired-end
        #   num_runtypes: missing (-1)
        #   num_runtypes+1: cloze (-2)
        self.runtype_embedding = nn.Embedding(num_runtypes + 2, embed_dim)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(4 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
    
    def _embed_continuous(self, values, proj, missing_emb, cloze_emb):
        """
        Embed continuous values with special handling for -1 (missing) and -2 (cloze).
        
        Args:
            values: [B, F] continuous values
            proj: nn.Linear projection for normal values
            missing_emb: nn.Parameter embedding for -1 (missing)
            cloze_emb: nn.Parameter embedding for -2 (cloze)
            
        Returns:
            emb: [B, F, embed_dim]
        """
        B, F = values.shape
        device = values.device
        
        # Masks for special tokens
        missing_mask = (values == -1)
        cloze_mask = (values == -2)
        normal_mask = ~missing_mask & ~cloze_mask
        
        # Project all values first (handles dtype from autocast properly)
        # Then overwrite special token positions
        all_vals = values.unsqueeze(-1).float()  # [B, F, 1]
        emb = proj(all_vals)  # [B, F, embed_dim] - dtype matches autocast context
        
        # Overwrite special token positions with learned embeddings
        if missing_mask.any():
            emb[missing_mask] = missing_emb.to(emb.dtype)
        if cloze_mask.any():
            emb[cloze_mask] = cloze_emb.to(emb.dtype)
        
        return emb
        
    def forward(self, metadata):
        """
        Args:
            metadata: [B, 4, F] with rows [depth_log2, assay_id, read_length, run_type]
        Returns:
            embedded: [B, F, embed_dim]
        """
        # Extract fields
        depth = metadata[:, 0, :]       # [B, F]
        assay_id = metadata[:, 1, :]    # [B, F]
        read_length = metadata[:, 2, :] # [B, F]
        runtype = metadata[:, 3, :]     # [B, F]
        
        # Embed continuous features with special token handling
        depth_emb = self._embed_continuous(
            depth, self.depth_proj,
            self.depth_missing_emb, self.depth_cloze_emb
        )
        read_length_emb = self._embed_continuous(
            read_length, self.read_length_proj,
            self.read_length_missing_emb, self.read_length_cloze_emb
        )
        
        # Handle categorical features: map -1 -> missing index, -2 -> cloze index
        # Assay ID: -1 -> num_assays+1 (missing), -2 -> num_assays+2 (cloze)
        assay_id = assay_id.long()
        assay_id = torch.where(assay_id == -1, 
                              torch.full_like(assay_id, self.num_assays + 1),
                              assay_id)
        assay_id = torch.where(assay_id == -2, 
                              torch.full_like(assay_id, self.num_assays + 2),
                              assay_id)
        assay_emb = self.assay_embedding(assay_id)
        
        # Runtype: -1 -> num_runtypes (missing), -2 -> num_runtypes+1 (cloze)
        runtype = runtype.long()
        runtype = torch.where(runtype == -1, 
                             torch.full_like(runtype, self.num_runtypes),
                             runtype)
        runtype = torch.where(runtype == -2, 
                             torch.full_like(runtype, self.num_runtypes + 1),
                             runtype)
        runtype_emb = self.runtype_embedding(runtype)
        
        # Concat and fuse
        concat = torch.cat([depth_emb, assay_emb, read_length_emb, runtype_emb], dim=-1)
        return self.fusion(concat)

class QueryMetadataEncoder(MetadataEncoder):
    """
    Decoder-side metadata encoder that explicitly serves per-assay query vectors.
    Inherits token handling and embedding semantics from MetadataEncoder.
    """
    def forward(self, metadata):
        # Return shape remains [B, F, E], used directly for sparse query gather.
        return super().forward(metadata)

class FiLMLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # input_dim = embed_dim
        # output_dim = channels per assay * 2 (scale + shift)
        self.proj = nn.Linear(input_dim, output_dim)
        
        # S4: Random initialization to break symmetry (per issue_supertrack.md)
        # Previously used near-identity init (std=0.02, zero bias) which allowed
        # the model to ignore prompts by applying near-identity FiLM transformation.
        # Xavier init forces the model to actively use metadata embeddings.
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.normal_(self.proj.bias, mean=0.0, std=0.1)

    def forward(self, x, metadata_embed):
        """
        x: [B, C, L] where C = F * d_per_assay
        metadata_embed: [B, F, embed_dim]
        """
        B, C, L = x.shape
        F = metadata_embed.shape[1]
        
        if C % F != 0:
            raise ValueError(f"C % F != 0 for FiLMLayer. C: {C}, F: {F}")

        d_per_assay = C // F
        
        # Project metadata to scale/shift parameters
        # [B, F, embed_dim] -> [B, F, d_per_assay * 2]
        params = self.proj(metadata_embed)
        
        # CORRECT: Split FIRST within each assay, THEN flatten
        scale, shift = params.chunk(2, dim=-1)  # [B, F, d_per_assay] each
        scale = scale.contiguous().view(B, C)   # [B, C]
        shift = shift.contiguous().view(B, C)   # [B, C]
        
        # Clamp scale to prevent overflow/NaNs in exp()
        scale = torch.clamp(scale, min=-4.0, max=4.0)
        
        # Reshape for broadcasting [B, C, 1]
        scale = scale.unsqueeze(-1)
        shift = shift.unsqueeze(-1)
        
        return x * torch.exp(scale) + shift

# ---------------------------
# MaskStem for Missing Data Handling
# ---------------------------
class MaskStem(nn.Module):
    """
    MaskStem: A learnable stem that processes (value, mask) pairs before the main encoder.
    
    This is a lightweight alternative to partial convolution that:
    1. Takes input signal and generates a binary mask (present/missing)
    2. Zeros out missing values explicitly
    3. Interleaves value and mask channels per-assay
    4. Applies a depth-wise 1x1 convolution to learn how to combine them
    
    Input: [B, C, L] where C = number of assays (F+1 including control)
    Output: [B, C, L] with missing data handled cleanly
    
    The grouped convolution ensures each assay learns its own (value, mask) -> feature mapping.
    """
    def __init__(self, n_channels, missing_token=-1):
        """
        Args:
            n_channels: Number of input channels (assays, including control = signal_dim + 1)
            missing_token: The sentinel value used to mark missing data (default: -1)
        """
        super().__init__()
        self.n_channels = n_channels
        self.missing_token = missing_token
        
        # Depth-wise 1x1 convolution: each assay gets its own 2->1 mapping
        # Input: 2 * n_channels (interleaved [value, mask] per assay)
        # Output: n_channels
        # groups=n_channels ensures each assay is processed independently
        self.stem_conv = nn.Conv1d(
            in_channels=2 * n_channels,
            out_channels=n_channels,
            kernel_size=1,
            groups=n_channels,
            bias=True
        )
        
        # Initialize to favor the value channel (near-identity for non-missing data)
        # This ensures stable training from the start
        with torch.no_grad():
            # For each group (assay), the conv has shape [1, 2, 1]
            # Initialize to: output = 1.0 * value + 0.0 * mask
            # Weight shape: [n_channels, 2, 1]
            self.stem_conv.weight.zero_()
            self.stem_conv.weight[:, 0, :] = 1.0  # Value channel weight = 1
            # Mask channel weight starts at 0 but can learn
            self.stem_conv.bias.zero_()
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C, L] with missing values marked by missing_token
            
        Returns:
            Processed tensor [B, C, L] with missing data handled via learned transformation
        """
        B, C, L = x.shape
        
        # 1. Generate binary mask (1 = present, 0 = missing)
        mask = (x != self.missing_token).float()
        
        # 2. Clean the values: zero out missing entries
        # This ensures the sentinel value doesn't leak into learned features
        x_clean = x * mask
        
        
        # 3. Interleave value and mask channels per-assay for grouped conv
        # We need: [B, 2*C, L] with layout [val_0, mask_0, val_1, mask_1, ...]
        # Stack to [B, C, 2, L] then reshape to [B, 2*C, L]
        x_interleaved = torch.stack([x_clean, mask], dim=2)  # [B, C, 2, L]
        x_interleaved = x_interleaved.view(B, 2 * C, L)  # [B, 2*C, L]
        
        # 4. Apply depth-wise convolution
        out = self.stem_conv(x_interleaved)  # [B, C, L]
        
        return out

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
            use_rmsnorm=False,
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

class LinearFusion(nn.Module):
    """
    Simple fusion: Concatenate Signal and DNA, then project linearly.
    Allows for different input dimensions for signal and DNA.
    """
    def __init__(self, signal_dim, dna_dim, output_dim, dropout=0.1):
        super().__init__()
        # Project combined dims to output_dim
        self.fusion_proj = nn.Linear(signal_dim + dna_dim, output_dim)
        
        # GELU activation for non-linearity
        self.gelu = nn.GELU()
        # Normalization and Dropout for stability before Transformer
        self.norm = nn.LayerNorm(output_dim)  
        self.dropout = nn.Dropout(dropout)

    def forward(self, signal, dna):
        """
        Args:
            signal: (B, L, signal_dim)
            dna: (B, L, dna_dim)
        Returns:
            (B, L, output_dim) fused representation
        """
        # 1. Concatenate along feature dimension
        combined = torch.cat([signal, dna], dim=-1)  # [B, L, signal_dim + dna_dim]
        
        # 2. Linear Projection
        fused = self.fusion_proj(combined)           # [B, L, output_dim]
        
        # 3. GELU activation
        fused = self.gelu(fused)

        # 4. Norm + Dropout
        return self.dropout(self.norm(fused))

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
    def __init__(self, in_C, out_C, W, S=1, D=1, pool_type="max", residuals=True, groups=1, pool_size=2, SE=False, norm="layer"):
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
            # Safety check: ensure sequence length is sufficient for pooling
            seq_len = y.size(2)
            pool_kernel = getattr(self.pool, 'kernel_size', 2)
            if isinstance(pool_kernel, tuple):
                pool_kernel = pool_kernel[0]
            if seq_len < pool_kernel:
                raise ValueError(f"ConvTower: sequence length {seq_len} is smaller than pool kernel {pool_kernel}. Check input dimensions.")
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
    def __init__(self, in_C, out_C, W, S=1, D=1, residuals=True, groups=1, pool_size=2, norm="layer"):
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
        
        # Clamp p to strictly (0, 1) for NegativeBinomial validity
        p = torch.clamp(p, min=self.eps, max=1.0 - self.eps)

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
        
        # Initialize weights with small random values to break symmetry but keep predictions close to bias
        # This prevents "ConstantInputWarning" while maintaining stability
        nn.init.normal_(self.linear_mu.weight, mean=0.0, std=1e-4)
        nn.init.constant_(self.linear_mu.bias, 0.0) # Start with mean ~0.69 (softplus(0))
        
        nn.init.normal_(self.linear_var.weight, mean=0.0, std=1e-4)
        nn.init.constant_(self.linear_var.bias, 0.5)

    def forward(self, x):
        if self.FF:
            x = self.feed_forward(x)

        mu_logits = self.linear_mu(x)
        mu = F.softplus(mu_logits)
        
        var_logits = self.linear_var(x)
        # Add epsilon for numerical stability
        var = F.softplus(var_logits) + 1e-6

        return mu, var

class LaplacianLayer(nn.Module):
    """
    Predicts Location (mu) and Log-Scale (log_b) for Laplace distribution.
    Outputting log_b is numerically stable and allows the loss function to handle exponentiation.
    """
    def __init__(self, input_dim, output_dim, FF=False):
        super(LaplacianLayer, self).__init__()
        self.FF = FF
        if self.FF:
            self.feed_forward = FeedForwardNN(input_dim, input_dim, input_dim, n_hidden_layers=2)

        self.linear_mu = nn.Linear(input_dim, output_dim)
        self.linear_log_b = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        if self.FF:
            x = self.feed_forward(x)

        # Mu: Signal is non-negative, use Softplus
        mu = F.softplus(self.linear_mu(x))
        
        # Log Scale (log_b): predict log_b directly
        # Clamp to reasonable range to prevent numerical explosion in exp() later
        log_b = self.linear_log_b(x)
        log_b = torch.clamp(log_b, min=-10.0, max=10.0)

        return mu, log_b

class GaussianLayerConstantVar(nn.Module):
    """
    Predicts Location (mu) per position, but learns a single constant variance per assay.
    
    This is useful for ablation studies where we want to learn assay-level noise
    without allowing the model to vary uncertainty per genomic position.
    """
    def __init__(self, input_dim, output_dim, FF=False):
        super(GaussianLayerConstantVar, self).__init__()
        self.FF = FF
        self.output_dim = output_dim
        
        if self.FF:
            self.feed_forward = FeedForwardNN(input_dim, input_dim, input_dim, n_hidden_layers=2)

        self.linear_mu = nn.Linear(input_dim, output_dim)
        
        # Learnable constant variance per output dimension (assay)
        # Initialize raw_var so that softplus(raw_var) approx 5.0
        self.raw_var = nn.Parameter(torch.ones(1, output_dim) * 5.0)
        
        # Zero out weights for mean prediction
        nn.init.zeros_(self.linear_mu.weight)
        nn.init.constant_(self.linear_mu.bias, 0.0)

    def forward(self, x):
        if self.FF:
            x = self.feed_forward(x)

        mu = F.softplus(self.linear_mu(x))
        
        # Expand constant variance to match batch/sequence dimensions
        # softplus ensures variance is positive + epsilon for stability
        var = (F.softplus(self.raw_var) + 1e-6).expand_as(mu)
        
        return mu, var

class LaplacianLayerConstantScale(nn.Module):
    """
    Predicts Location (mu) per position, but learns a single constant scale (log_b) per assay.
    
    This is useful for ablation studies where we want to learn assay-level noise
    without allowing the model to vary uncertainty per genomic position.
    """
    def __init__(self, input_dim, output_dim, FF=False):
        super(LaplacianLayerConstantScale, self).__init__()
        self.FF = FF
        self.output_dim = output_dim
        
        if self.FF:
            self.feed_forward = FeedForwardNN(input_dim, input_dim, input_dim, n_hidden_layers=2)

        self.linear_mu = nn.Linear(input_dim, output_dim)
        
        # Learnable constant log_b per output dimension (assay)
        # Initialize to 0.0 so that b = exp(0) = 1.0
        self.log_b = nn.Parameter(torch.zeros(1, output_dim))

    def forward(self, x):
        if self.FF:
            x = self.feed_forward(x)

        # Mu: Signal is non-negative, use Softplus
        mu = F.softplus(self.linear_mu(x))
        
        # Expand constant log_b to match batch/sequence dimensions
        # Clamp for stability just like original layer
        log_b_clamped = torch.clamp(self.log_b, min=-10.0, max=10.0)
        log_b_expanded = log_b_clamped.expand_as(mu)
        
        return mu, log_b_expanded

class GammaLayer(nn.Module):
    """
    Predicts mean (mu) and concentration (alpha) for Gamma distribution.
    
    mu > 0, alpha > 0.
    """
    def __init__(self, input_dim, output_dim, FF=False, alpha_init=2.0):
        super(GammaLayer, self).__init__()
        self.FF = FF
        if self.FF:
            self.feed_forward = FeedForwardNN(input_dim, input_dim, input_dim, n_hidden_layers=2)

        self.linear_mu = nn.Linear(input_dim, output_dim)
        self.linear_alpha = nn.Linear(input_dim, output_dim)

        # Initialize alpha bias so softplus(bias) ≈ alpha_init
        with torch.no_grad():
            self.linear_alpha.bias.fill_(math.log(math.exp(alpha_init) - 1.0))

    def forward(self, x):
        if self.FF:
            x = self.feed_forward(x)

        mu = F.softplus(self.linear_mu(x))
        alpha = F.softplus(self.linear_alpha(x))

        return mu, alpha

class StudentsTLayer(nn.Module):
    """
    Predicts Location (mu), Scale (sigma), and Degrees of Freedom (df) for Student's t-distribution.
    
    All parameters are constrained to be positive using softplus.
    df is offset by +2 to ensure finite variance (df > 2).
    df is initialized to ~30 (Gaussian-like) for stable training.
    """
    def __init__(self, input_dim, output_dim, FF=False, df_init=10.0):
        super(StudentsTLayer, self).__init__()
        self.FF = FF
        self.df_init = df_init
        
        if self.FF:
            self.feed_forward = FeedForwardNN(input_dim, input_dim, input_dim, n_hidden_layers=2)

        self.linear_mu = nn.Linear(input_dim, output_dim)
        self.linear_sigma = nn.Linear(input_dim, output_dim)
        self.linear_df = nn.Linear(input_dim, output_dim)
        
        # Initialize df bias to produce ~df_init after 2 + softplus()
        # softplus(x) ≈ x for large x, so bias ≈ df_init - 2
        with torch.no_grad():
            self.linear_df.bias.fill_(df_init - 2.0)

    def forward(self, x):
        if self.FF:
            x = self.feed_forward(x)

        # Mu: Signal is non-negative, use Softplus
        mu = F.softplus(self.linear_mu(x))
        
        # Sigma: Scale parameter, must be positive
        sigma = F.softplus(self.linear_sigma(x))
        
        # df: Degrees of freedom, offset by 2 to ensure variance is defined (df > 2)
        # Using 2 + softplus(x) ensures df > 2
        df = 2.0 + F.softplus(self.linear_df(x))
        
        # Clamp df to reasonable range to prevent numerical issues
        df = torch.clamp(df, min=2.01, max=100.0)

        return mu, sigma, df

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

