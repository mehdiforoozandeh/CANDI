#!/usr/bin/env python3
"""
CANDI Post-Training Script with Iterative Refinement (Scenario B).

Implements Training-Aware Refinement with configurable constraints:
1. Soft Constraints (Lambda) - Blend predictions with observations
2. Confidence Gating - Update only when model certainty increases
3. Gibbs Sampling (Re-masking) - Randomly mask regions for robustness

Usage Examples:
    # Basic refinement with 1 pass (Teacher Forcing baseline)
    python post_training/train_refinement.py --refinement-model-path /path/to/model.pt --epochs 5
    
    # Soft constraint experiment (20% prediction blend)
    python post_training/train_refinement.py --refinement-model-path /path/to/model.pt --refinement-lambda 0.2
    
    # Confidence gating (only update predictions if confidence improves)
    python post_training/train_refinement.py --refinement-model-path /path/to/model.pt --gating-strategy preds
    
    # Gibbs sampling (re-mask 15% of observations)
    python post_training/train_refinement.py --refinement-model-path /path/to/model.pt --gibbs-prob 0.15 --gibbs-scope obs
    
    # Full differentiable (BPTT) training
    python post_training/train_refinement.py --refinement-model-path /path/to/model.pt --gradient-flow bptt
"""

import sys
import os
from pathlib import Path
import time
import argparse
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler
from datetime import datetime
import pandas as pd
import random
import json
import gc
import multiprocessing
import math
from sklearn.metrics import roc_auc_score

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from model import CANDI, CANDI_LOSS, CANDI_UNET, EIC_VALIDATION_MONITOR
from _utils import DataMasker, reverse_complement_dna, reverse_signal, negative_binomial_loss
from data import CANDIDataHandler, CANDIIterableDataset

# Import utilities from train.py
from train import (
    CANDI_TRAINER,
    create_argument_parser as create_base_argument_parser,
    validate_arguments as validate_base_arguments,
    setup_device,
    init_distributed,
    cleanup_distributed,
    create_model_from_args,
    generate_model_name,
    print_training_summary,
    save_config_file,
    load_config_file,
    check_gpu_availability
)

# Token dictionary
TOKEN_DICT = {"missing_mask": -1, "cloze_mask": -2, "pad": -3}

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return float('nan')

ANSI_GREEN = "\033[92m"
ANSI_RED = "\033[91m"
ANSI_RESET = "\033[0m"

def _colorize_delta(val: float, *, improvement_if_positive: bool) -> str:
    """
    Color a delta value:
    - green if improved
    - red if worse
    - no color if NaN
    """
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return "nan"
    improved = (val > 0) if improvement_if_positive else (val < 0)
    color = ANSI_GREEN if improved else ANSI_RED
    return f"{color}{val:+.4f}{ANSI_RESET}"

def _pearson_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Fast Pearson correlation on torch tensors.
    Returns NaN if insufficient samples or zero-variance.
    """
    if x.numel() < 2 or y.numel() < 2:
        return float('nan')
    # ensure float32 for stability
    x = x.float()
    y = y.float()
    x = x - x.mean()
    y = y - y.mean()
    denom = x.std(unbiased=False) * y.std(unbiased=False)
    if denom.item() == 0.0 or torch.isnan(denom):
        return float('nan')
    r = (x * y).mean() / denom
    if torch.isnan(r):
        return float('nan')
    return float(r.item())

def _auc_roc_numpy(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    AUC-ROC with guardrails (returns NaN if undefined).
    """
    try:
        if y_true.size < 2:
            return float('nan')
        uniq = np.unique(y_true)
        if uniq.size < 2:
            return float('nan')
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float('nan')


##=========================================== Refinement Validation Monitor =============================================##

class RefinementValidationMonitor(EIC_VALIDATION_MONITOR):
    """
    Extended validation monitor that runs 2-pass evaluation to measure refinement improvement.
    Computes Pass 1 metrics, Pass 2 metrics, and Delta metrics.
    """
    
    def __init__(self, context_length, training_batch_size, device=None, resolution=25,
                 refinement_lambda=0.0, gating_strategy='none', gibbs_prob=0.0, gibbs_scope='obs',
                 dist_type='gaussian'):
        super().__init__(context_length, training_batch_size, device, resolution, dist_type=dist_type)
        
        # Refinement constraints for validation (should match training)
        self.refinement_lambda = refinement_lambda
        self.gating_strategy = gating_strategy
        self.gibbs_prob = gibbs_prob
        self.gibbs_scope = gibbs_scope
        
    def _apply_refinement_constraints(self, x_input, pred_mu, pred_n, observed_map, masked_map, 
                                       original_obs, best_n=None, t=0, imputation_map=None, x_avail=None):
        """
        Apply refinement constraints to construct next iteration input.
        This mirrors the training logic for consistency.
        """
        x_new = x_input.clone()
        
        # Use imputation_map if provided, otherwise fall back to masked_map
        if imputation_map is None:
            imputation_map = masked_map
        
        # Constraint 1: Lambda blending for observed regions
        if self.refinement_lambda > 0:
            blend_obs = (1.0 - self.refinement_lambda) * original_obs + self.refinement_lambda * pred_mu
            x_new[observed_map] = blend_obs[observed_map]
        else:
            x_new[observed_map] = original_obs[observed_map]
        
        # Fill ALL imputed regions (masked + missing) with predictions
        x_new[imputation_map] = pred_mu[imputation_map]
        
        # Constraint 2: Confidence Gating
        if self.gating_strategy != 'none' and t > 0 and best_n is not None:
            improved_mask = pred_n > best_n
            
            if self.gating_strategy in ['preds', 'both']:
                # Revert ALL imputed slots (masked + missing) if confidence didn't improve
                gate_mask = imputation_map & (~improved_mask)
                x_new[gate_mask] = x_input[gate_mask]
                
            if self.gating_strategy in ['obs', 'both']:
                gate_mask = observed_map & (~improved_mask)
                x_new[gate_mask] = x_input[gate_mask]
        
        # Constraint 3: Gibbs Sampling (Note: disabled during validation for consistency)
        # We don't apply random re-masking during validation to get deterministic metrics
        
        return x_new
    
    def run_validation(self, model, batch_idx, total_batches):
        """
        Run 2-pass validation on all V_* biosamples.
        Returns metrics for Pass 1, Pass 2, and Delta.
        """
        print(f"Running Refinement Validation at batch {batch_idx} ({100.0 * batch_idx / total_batches:.1f}% progress)...")
        
        # Use full chr21
        locus = ["chr21", 0, self.chr_sizes["chr21"]]
        
        # Collect metrics per (biosample, assay_name, comparison_type, pass_num)
        all_metrics_pass1 = []
        all_metrics_pass2 = []
        
        # Pre-load DNA sequence once
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
        
        # Process all biosamples
        num_biosamples = len(self.v_biosamples)
        print(f"  Processing {num_biosamples} biosamples with 2-pass refinement")
        
        for bios_idx, V_biosample in enumerate(self.v_biosamples):
            try:
                print(f"  [{bios_idx+1}/{num_biosamples}] Validating {V_biosample}...", end=" ", flush=True)
                
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Load validation data
                data = self._load_validation_data(V_biosample, locus, cached_seq=cached_seq)
                
                # ===== PASS 1: Standard Prediction =====
                n1, p1, mu1, var1, peak1 = self._predict(
                    model, data['X'], data['mX'], data['mY_T'], data['avX'], data['seq']
                )
                
                # ===== Construct Refined Input for Pass 2 =====
                # Reconstruct the observed/masked maps from the input
                X_input = data['X'][:, :, :-1]  # Remove control track for signal manipulation
                signal_dim = X_input.shape[-1]
                
                observed_map = (X_input != TOKEN_DICT["missing_mask"]) & (X_input != TOKEN_DICT["cloze_mask"])
                masked_map = (X_input == TOKEN_DICT["cloze_mask"])
                missing_map = (X_input == TOKEN_DICT["missing_mask"])
                imputation_map = masked_map | missing_map  # All positions that need imputation

                # IMPORTANT: refined input should be in count space -> use NB mean, not Gaussian mu
                eps = 1e-8
                p1_clamped = torch.clamp(p1, min=eps, max=1.0 - eps)
                n1_clamped = torch.clamp(n1, min=eps)
                mu1_nb = (n1_clamped * (1.0 - p1_clamped)) / p1_clamped
                
                # Apply refinement constraints
                X_refined = self._apply_refinement_constraints(
                    x_input=X_input,
                    pred_mu=mu1_nb,
                    pred_n=n1,
                    observed_map=observed_map,
                    masked_map=masked_map,
                    original_obs=X_input,
                    best_n=n1,
                    t=0,
                    imputation_map=imputation_map
                )
                
                # Reconstruct full input with control track
                control_track = data['X'][:, :, -1:]
                X_refined_full = torch.cat([X_refined, control_track], dim=-1)

                # Apply the same metadata-fill behavior used in training (deterministic; no Gibbs in validation):
                # fill any missing/cloze metadata in mX using the prompt metadata (mY_T).
                # This ensures pass-2 input metadata matches the prompt conditions.
                try:
                    mX_full = data.get('mX', None)
                    mY_full = data.get('mY_T', None)
                    if mX_full is not None and mY_full is not None:
                        # remove control metadata track for signal-assay metadata manipulation
                        mX_sig = mX_full[:, :, :-1]
                        mX_ctrl = mX_full[:, :, -1:]
                        mY_sig = mY_full

                        meta_bad = (mX_sig == TOKEN_DICT["missing_mask"]) | (mX_sig == TOKEN_DICT["cloze_mask"])
                        meta_bad_assay = meta_bad.any(dim=1)  # [B, F]
                        if meta_bad_assay.any():
                            meta_bad_broadcast = meta_bad_assay.unsqueeze(1).expand_as(mX_sig)
                            mX_sig = mX_sig.clone()
                            mX_sig[meta_bad_broadcast] = mY_sig[meta_bad_broadcast]

                        mX_refined_full = torch.cat([mX_sig, mX_ctrl], dim=-1)
                    else:
                        mX_refined_full = mX_full
                except Exception:
                    # If metadata shapes differ across datasets, fall back to original mX.
                    mX_refined_full = data.get('mX', None)
                
                # ===== PASS 2: Refined Prediction =====
                n2, p2, mu2, var2, peak2 = self._predict(
                    model, X_refined_full, mX_refined_full, data['mY_T'], data['avX'], data['seq']
                )
                
                # Determine upsampled vs imputed assays
                available_T_set = set(data['available_T_indices'])
                available_V_set = set(data['available_V_indices'])
                upsampled_assays = available_T_set
                imputed_assays = available_V_set - available_T_set
                
                # Compute metrics for both passes
                for assay_idx in upsampled_assays:
                    if assay_idx >= len(self.expnames):
                        continue
                    assay_name = self.expnames[assay_idx]
                    
                    # Pass 1 metrics
                    count_nll_1 = self._compute_count_nll(n1[:, :, assay_idx], p1[:, :, assay_idx], data['Y_T'][:, :, assay_idx])
                    signal_nll_1 = self._compute_signal_nll(mu1[:, :, assay_idx], var1[:, :, assay_idx], data['P_T'][:, :, assay_idx])
                    peak_bce_1 = self._compute_peak_bce(peak1[:, :, assay_idx], data['Peak_T'][:, :, assay_idx])
                    
                    all_metrics_pass1.append({
                        'biosample': V_biosample, 'assay_name': assay_name, 'comparison': 'upsampled',
                        'count_nll': count_nll_1, 'signal_nll': signal_nll_1, 'peak_bce': peak_bce_1
                    })
                    
                    # Pass 2 metrics
                    count_nll_2 = self._compute_count_nll(n2[:, :, assay_idx], p2[:, :, assay_idx], data['Y_T'][:, :, assay_idx])
                    signal_nll_2 = self._compute_signal_nll(mu2[:, :, assay_idx], var2[:, :, assay_idx], data['P_T'][:, :, assay_idx])
                    peak_bce_2 = self._compute_peak_bce(peak2[:, :, assay_idx], data['Peak_T'][:, :, assay_idx])
                    
                    all_metrics_pass2.append({
                        'biosample': V_biosample, 'assay_name': assay_name, 'comparison': 'upsampled',
                        'count_nll': count_nll_2, 'signal_nll': signal_nll_2, 'peak_bce': peak_bce_2
                    })
                
                for assay_idx in imputed_assays:
                    if assay_idx >= len(self.expnames):
                        continue
                    assay_name = self.expnames[assay_idx]
                    
                    # Pass 1 metrics
                    count_nll_1 = self._compute_count_nll(n1[:, :, assay_idx], p1[:, :, assay_idx], data['Y_V'][:, :, assay_idx])
                    signal_nll_1 = self._compute_signal_nll(mu1[:, :, assay_idx], var1[:, :, assay_idx], data['P_V'][:, :, assay_idx])
                    peak_bce_1 = self._compute_peak_bce(peak1[:, :, assay_idx], data['Peak_V'][:, :, assay_idx])
                    
                    all_metrics_pass1.append({
                        'biosample': V_biosample, 'assay_name': assay_name, 'comparison': 'imputed',
                        'count_nll': count_nll_1, 'signal_nll': signal_nll_1, 'peak_bce': peak_bce_1
                    })
                    
                    # Pass 2 metrics
                    count_nll_2 = self._compute_count_nll(n2[:, :, assay_idx], p2[:, :, assay_idx], data['Y_V'][:, :, assay_idx])
                    signal_nll_2 = self._compute_signal_nll(mu2[:, :, assay_idx], var2[:, :, assay_idx], data['P_V'][:, :, assay_idx])
                    peak_bce_2 = self._compute_peak_bce(peak2[:, :, assay_idx], data['Peak_V'][:, :, assay_idx])
                    
                    all_metrics_pass2.append({
                        'biosample': V_biosample, 'assay_name': assay_name, 'comparison': 'imputed',
                        'count_nll': count_nll_2, 'signal_nll': signal_nll_2, 'peak_bce': peak_bce_2
                    })
                
                del data, n1, p1, mu1, var1, peak1, n2, p2, mu2, var2, peak2
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
        
        if not all_metrics_pass1:
            print("Warning: No validation metrics computed")
            return self._empty_result(batch_idx, total_batches)
        
        # Aggregate metrics
        df1 = pd.DataFrame(all_metrics_pass1)
        df2 = pd.DataFrame(all_metrics_pass2)
        
        imp_df1 = df1[df1['comparison'] == 'imputed']
        ups_df1 = df1[df1['comparison'] == 'upsampled']
        imp_df2 = df2[df2['comparison'] == 'imputed']
        ups_df2 = df2[df2['comparison'] == 'upsampled']
        
        result = {
            'iteration': batch_idx,
            'progress_pct': 100.0 * batch_idx / total_batches,
            # Pass 1 metrics
            'pass1_imp_count_nll_mean': imp_df1['count_nll'].mean() if len(imp_df1) > 0 else 0.0,
            'pass1_imp_signal_nll_mean': imp_df1['signal_nll'].mean() if len(imp_df1) > 0 else 0.0,
            'pass1_imp_peak_bce_mean': imp_df1['peak_bce'].mean() if len(imp_df1) > 0 else 0.0,
            'pass1_ups_count_nll_mean': ups_df1['count_nll'].mean() if len(ups_df1) > 0 else 0.0,
            'pass1_ups_signal_nll_mean': ups_df1['signal_nll'].mean() if len(ups_df1) > 0 else 0.0,
            'pass1_ups_peak_bce_mean': ups_df1['peak_bce'].mean() if len(ups_df1) > 0 else 0.0,
            # Pass 2 metrics
            'pass2_imp_count_nll_mean': imp_df2['count_nll'].mean() if len(imp_df2) > 0 else 0.0,
            'pass2_imp_signal_nll_mean': imp_df2['signal_nll'].mean() if len(imp_df2) > 0 else 0.0,
            'pass2_imp_peak_bce_mean': imp_df2['peak_bce'].mean() if len(imp_df2) > 0 else 0.0,
            'pass2_ups_count_nll_mean': ups_df2['count_nll'].mean() if len(ups_df2) > 0 else 0.0,
            'pass2_ups_signal_nll_mean': ups_df2['signal_nll'].mean() if len(ups_df2) > 0 else 0.0,
            'pass2_ups_peak_bce_mean': ups_df2['peak_bce'].mean() if len(ups_df2) > 0 else 0.0,
        }
        
        # Compute Delta metrics (Pass2 - Pass1, lower is better for NLL/BCE)
        result['delta_imp_count_nll'] = result['pass2_imp_count_nll_mean'] - result['pass1_imp_count_nll_mean']
        result['delta_imp_signal_nll'] = result['pass2_imp_signal_nll_mean'] - result['pass1_imp_signal_nll_mean']
        result['delta_imp_peak_bce'] = result['pass2_imp_peak_bce_mean'] - result['pass1_imp_peak_bce_mean']
        result['delta_ups_count_nll'] = result['pass2_ups_count_nll_mean'] - result['pass1_ups_count_nll_mean']
        result['delta_ups_signal_nll'] = result['pass2_ups_signal_nll_mean'] - result['pass1_ups_signal_nll_mean']
        result['delta_ups_peak_bce'] = result['pass2_ups_peak_bce_mean'] - result['pass1_ups_peak_bce_mean']
        
        print(f"Refinement Validation completed:")
        print(f"  Pass 1 Imp NLL: {result['pass1_imp_count_nll_mean']:.4f}")
        print(f"  Pass 2 Imp NLL: {result['pass2_imp_count_nll_mean']:.4f}")
        print(f"  Delta Imp NLL: {result['delta_imp_count_nll']:.4f} ({'improved' if result['delta_imp_count_nll'] < 0 else 'degraded'})")
        
        return result
    
    def _empty_result(self, batch_idx, total_batches):
        return {
            'iteration': batch_idx,
            'progress_pct': 100.0 * batch_idx / total_batches,
            'pass1_imp_count_nll_mean': 0.0, 'pass1_imp_signal_nll_mean': 0.0, 'pass1_imp_peak_bce_mean': 0.0,
            'pass1_ups_count_nll_mean': 0.0, 'pass1_ups_signal_nll_mean': 0.0, 'pass1_ups_peak_bce_mean': 0.0,
            'pass2_imp_count_nll_mean': 0.0, 'pass2_imp_signal_nll_mean': 0.0, 'pass2_imp_peak_bce_mean': 0.0,
            'pass2_ups_count_nll_mean': 0.0, 'pass2_ups_signal_nll_mean': 0.0, 'pass2_ups_peak_bce_mean': 0.0,
            'delta_imp_count_nll': 0.0, 'delta_imp_signal_nll': 0.0, 'delta_imp_peak_bce': 0.0,
            'delta_ups_count_nll': 0.0, 'delta_ups_signal_nll': 0.0, 'delta_ups_peak_bce': 0.0,
        }


##=========================================== Refinement Trainer =============================================##

class REFINEMENT_TRAINER(CANDI_TRAINER):
    """
    Extended Trainer that implements Iterative Refinement logic in _process_batch.
    
    This class inherits from CANDI_TRAINER and ONLY overrides _process_batch to add
    refinement logic. All other training loop logic (progress monitoring, validation,
    checkpointing, etc.) is inherited from the parent class.
    
    Supports:
    - Multiple refinement iterations
    - Soft Lambda constraint (blend predictions with observations)
    - Confidence Gating (update only when certainty improves)
    - Gibbs Sampling (random re-masking for robustness)
    - Gradient flow control (stop-gradient vs BPTT)
    """
    
    def __init__(self, model, dataset_params, training_params, device=None, rank=None, world_size=None):
        # Extract refinement params BEFORE calling super().__init__ 
        # so they're available for any setup that happens there
        self.refinement_iterations = training_params.get('refinement_iterations', 1)
        self.refinement_lambda = training_params.get('refinement_lambda', 0.0)
        self.gating_strategy = training_params.get('gating_strategy', 'none')
        self.gibbs_prob = training_params.get('gibbs_prob', 0.0)
        self.gibbs_scope = training_params.get('gibbs_scope', 'obs')
        self.gradient_flow = training_params.get('gradient_flow', 'stop')
        
        # Call parent __init__ which sets up everything
        super().__init__(model, dataset_params, training_params, device, rank, world_size)
        
        # Print refinement configuration
        if self.is_main_process:
            print("\n" + "="*60)
            print("REFINEMENT TRAINING CONFIGURATION")
            print("="*60)
            print(f"  Refinement Iterations: {self.refinement_iterations}")
            print(f"  Soft Constraint (Lambda): {self.refinement_lambda}")
            print(f"    (0.0 = Hard Obs Replacement, 1.0 = Full Prediction)")
            print(f"  Confidence Gating: {self.gating_strategy}")
            print(f"  Gibbs Sampling: prob={self.gibbs_prob}, scope={self.gibbs_scope}")
            print(f"  Gradient Flow: {self.gradient_flow}")
            print(f"    (stop = Teacher Forcing, bptt = Backprop Through Time)")
            print("="*60 + "\n")
    
        # Buffers for refinement-aware monitoring (populated in _process_batch)
        self._last_refine_step_summaries = None
        self._last_refine_deltas = None
        self._last_refine_probe_info = None

        # EMA over delta metrics (refinement learning signal)
        self.delta_ema = {}
        self.delta_ema_alpha = float(self.training_params.get('specific_ema_alpha', 0.005))

    def _update_delta_ema(self, deltas: dict) -> None:
        """
        Update EMA for ALL delta metrics and losses. Skips NaNs/Infs.
        """
        if not deltas:
            return

        # All delta keys that should be tracked with EMA
        keys = [
            # Loss deltas
            'delta_imp_count_loss',
            'delta_obs_count_loss',
            'delta_imp_pval_loss',
            'delta_obs_pval_loss',
            'delta_imp_peak_loss',
            'delta_obs_peak_loss',
            # Metric deltas
            'delta_imp_peak_auc',
            'delta_obs_peak_auc',
            'delta_imp_count_pcc',
            'delta_obs_count_pcc',
            'delta_imp_pval_pcc',
            'delta_obs_pval_pcc',
        ]
        alpha = self.delta_ema_alpha
        for k in keys:
            v = deltas.get(k, float('nan'))
            v = _safe_float(v)
            if math.isnan(v) or math.isinf(v):
                continue
            if k not in self.delta_ema:
                self.delta_ema[k] = v
            else:
                self.delta_ema[k] = alpha * v + (1.0 - alpha) * self.delta_ema[k]

    def _setup(self):
        """
        Setup method - calls parent _setup() but uses RefinementValidationMonitor
        for validation if enabled.
        """
        # First, call parent setup for DDP wrapping and basic initialization
        if self.is_main_process:
            print(f"Setting up Refinement trainer...")
            print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"Training parameters: {self.training_params}")
            
            # Watch model with W&B
            try:
                import wandb
                if wandb.run is not None:
                    wandb.watch(self.model, log="all", log_freq=100)
            except ImportError:
                pass
        
        # Wrap model in DDP if multi-GPU (same as parent)
        if self.is_ddp:
            if self.is_main_process:
                print(f"Wrapping model in DDP for {self.world_size} processes")
            self.model = DDP(
                self.model,
                device_ids=[self.device.index] if self.device.type == 'cuda' else None,
                output_device=self.device.index if self.device.type == 'cuda' else None,
                find_unused_parameters=False
            )
        
        # Setup validation - use RefinementValidationMonitor instead of EIC_VALIDATION_MONITOR
        if self.enable_validation:
            if self.is_main_process:
                print("Validation is enabled (using RefinementValidationMonitor)")
            try:
                context_length_bp = self.dataset_params.get('context_length', 1200 * 25)
                context_length_bins = context_length_bp // 25
                training_batch_size = self.training_params.get('batch_size', 25)
                dist_type = self.training_params.get('dist_type', 'gaussian')
                
                self.validation_monitor = RefinementValidationMonitor(
                    context_length=context_length_bins,
                    training_batch_size=training_batch_size,
                    device=self.device,
                    refinement_lambda=self.refinement_lambda,
                    gating_strategy=self.gating_strategy,
                    gibbs_prob=self.gibbs_prob,
                    gibbs_scope=self.gibbs_scope,
                    dist_type=dist_type
                )
                if self.is_main_process:
                    print("RefinementValidationMonitor setup completed successfully")
                    print(f"Validation frequency: {self.val_freq} (every {100.0 * self.val_freq:.1f}% of training)")
            except Exception as e:
                if self.is_main_process:
                    print(f"Warning: Failed to setup RefinementValidationMonitor: {e}")
                    import traceback
                    traceback.print_exc()
                self.validation_monitor = None
        else:
            if self.is_main_process:
                print("Validation is disabled")
            self.validation_monitor = None
    
    def _apply_refinement_constraints(self,
                                      current_input,
                                      current_meta,
                                      y_meta_prompt,
                                      pred_mu,
                                      pred_n,
                                      observed_map,
                                      masked_map,
                                      original_obs,
                                      best_n,
                                      t,
                                      imputation_map=None,
                                      x_avail=None):
        """
        Apply all refinement constraints to construct next iteration input.
        
        Args:
            current_input: Current input signal [B, L, F]
            current_meta: Current input metadata [B, 4, F]
            y_meta_prompt: Prompt metadata for Y-side [B, 4, F]
            pred_mu: Predicted mean from model [B, L, F]
            pred_n: Predicted dispersion from model [B, L, F]
            observed_map: Boolean mask for observed positions [B, L, F]
            masked_map: Boolean mask for masked (cloze) positions [B, L, F]
            original_obs: Original observed data [B, L, F]
            best_n: Best confidence seen so far [B, L, F]
            t: Current refinement iteration (0-indexed)
            imputation_map: Boolean mask for ALL imputed positions (masked | missing) [B, L, F]
                           If None, defaults to masked_map (backward compatibility)
            x_avail: Availability mask [B, F] indicating originally available assays.
                     Required for whole-assay Gibbs sampling.
            
        Returns:
            next_input: Updated input for next iteration [B, L, F]
            next_meta: Updated input metadata for next iteration [B, 4, F]
            new_best_n: Updated best confidence [B, L, F]
        """
        next_input = current_input.clone()
        next_meta = current_meta.clone()
        
        # Use imputation_map if provided, otherwise fall back to masked_map
        if imputation_map is None:
            imputation_map = masked_map
        
        # ===== Constraint 1: Lambda Blending for Observed Regions =====
        if self.refinement_lambda > 0:
            blend_obs = (1.0 - self.refinement_lambda) * original_obs + self.refinement_lambda * pred_mu
            next_input[observed_map] = blend_obs[observed_map]
        else:
            # Hard replacement: keep original observed values
            next_input[observed_map] = original_obs[observed_map]
        
        # ===== Fill ALL Imputed Regions with Predictions =====
        # This includes both cloze-masked and originally missing positions
        next_input[imputation_map] = pred_mu[imputation_map]

        # NOTE: prompt-filling of metadata is handled in the refinement loop (after pass 0)
        # by constructing a meta_baseline and applying Gibbs masking on top of it.
        
        # ===== Constraint 2: Confidence Gating =====
        new_best_n = best_n.clone() if best_n is not None else pred_n.clone()
        
        if self.gating_strategy != 'none' and t > 0 and best_n is not None:
            improved_mask = pred_n > best_n
            
            if self.gating_strategy in ['preds', 'both']:
                # Revert ALL imputed slots (masked + missing) if confidence didn't improve
                gate_mask = imputation_map & (~improved_mask)
                next_input[gate_mask] = current_input[gate_mask]
            
            if self.gating_strategy in ['obs', 'both']:
                # Revert observed slots if confidence didn't improve (when using lambda)
                gate_mask = observed_map & (~improved_mask)
                next_input[gate_mask] = current_input[gate_mask]
            
            # Update best_n where improvement occurred
            new_best_n = torch.where(improved_mask, pred_n, best_n)
        
        # ===== Constraint 3: Gibbs Sampling (Re-masking) =====
        if self.gibbs_prob > 0 and x_avail is not None:
            # Whole-assay masking based on x_avail
            B_mb, _, F = next_input.shape
            
            for b in range(B_mb):
                # Identify assays
                avail_indices = torch.nonzero(x_avail[b]).squeeze(-1)
                missing_indices = torch.nonzero(x_avail[b] == 0).squeeze(-1)
                
                # Scope: Observed (Available) Assays
                if self.gibbs_scope in ['obs', 'both']:
                    num_avail = len(avail_indices)
                    num_to_mask = int(math.ceil(num_avail * self.gibbs_prob)) if num_avail > 0 else 0
                    # cap to num_avail - 1 so at least one available assay remains
                    num_to_mask = min(num_to_mask, max(0, num_avail - 1))
                    
                    if num_to_mask > 0:
                        perm = torch.randperm(num_avail, device=self.device)
                        mask_indices = avail_indices[perm[:num_to_mask]]
                        next_input[b, :, mask_indices] = float(TOKEN_DICT["cloze_mask"])
                        next_meta[b, :, mask_indices] = float(TOKEN_DICT["cloze_mask"])
                
                # Scope: Predicted (Imputed) Assays
                if self.gibbs_scope in ['preds', 'both']:
                    num_missing = len(missing_indices)
                    num_to_mask = int(math.ceil(num_missing * self.gibbs_prob)) if num_missing > 0 else 0
                    # cap to num_missing - 1 so at least one imputed assay remains
                    num_to_mask = min(num_to_mask, max(0, num_missing - 1))
                    
                    if num_to_mask > 0:
                        perm = torch.randperm(num_missing, device=self.device)
                        mask_indices = missing_indices[perm[:num_to_mask]]
                        next_input[b, :, mask_indices] = float(TOKEN_DICT["cloze_mask"])
                        next_meta[b, :, mask_indices] = float(TOKEN_DICT["cloze_mask"])
        
        return next_input, next_meta, new_best_n

    def _compute_refine_step_summary(self,
                                    *,
                                    step_idx: int,
                                    output_p: torch.Tensor,
                                    output_n: torch.Tensor,
                                    output_mu: torch.Tensor,
                                    output_peak: torch.Tensor,
                                    y_data: torch.Tensor,
                                    y_pval: torch.Tensor,
                                    y_peaks: torch.Tensor,
                                    observed_map: torch.Tensor,
                                    masked_map: torch.Tensor,
                                    losses: dict) -> dict:
        """
        Compute the refinement-aware metrics requested:
        - Imp/Obs losses (count, pval, peak)
        - Imp/Obs Peak AUCROC
        - Imp/Obs Pearson for Count (NB mean) and Gaussian mu

        Note: This is computed on the provided tensors (often a probe microbatch).
        """
        eps = 1e-8

        # NB expected mean for counts: mean = n*(1-p)/p
        p = torch.clamp(output_p, min=eps, max=1 - eps)
        n = torch.clamp(output_n, min=eps)
        count_pred = (n * (1.0 - p)) / p
        pval_pred = output_mu
        peak_prob = torch.sigmoid(output_peak)

        def _extract(mask: torch.Tensor, pred: torch.Tensor, true: torch.Tensor):
            # flatten masked values
            if not mask.any():
                return None, None
            pv = pred[mask].detach()
            tv = true[mask].detach()
            # drop inf/nan
            finite = torch.isfinite(pv) & torch.isfinite(tv)
            pv = pv[finite]
            tv = tv[finite]
            if pv.numel() < 2:
                return None, None
            return pv, tv

        # Pearson correlations
        imp_count_pv, imp_count_tv = _extract(masked_map, count_pred, y_data)
        obs_count_pv, obs_count_tv = _extract(observed_map, count_pred, y_data)
        imp_pval_pv, imp_pval_tv = _extract(masked_map, pval_pred, y_pval)
        obs_pval_pv, obs_pval_tv = _extract(observed_map, pval_pred, y_pval)

        imp_count_pcc = _pearson_torch(imp_count_pv, imp_count_tv) if imp_count_pv is not None else float('nan')
        obs_count_pcc = _pearson_torch(obs_count_pv, obs_count_tv) if obs_count_pv is not None else float('nan')
        imp_pval_pcc = _pearson_torch(imp_pval_pv, imp_pval_tv) if imp_pval_pv is not None else float('nan')
        obs_pval_pcc = _pearson_torch(obs_pval_pv, obs_pval_tv) if obs_pval_pv is not None else float('nan')

        # AUCROC for peak (subsample if huge to keep overhead bounded)
        def _auc(mask: torch.Tensor):
            if not mask.any():
                return float('nan'), 0, 0
            y_true = y_peaks[mask].detach().float()
            y_score = peak_prob[mask].detach().float()

            # Binarize robustly (peaks should already be 0/1 but guard anyway)
            y_true_bin = (y_true > 0.5)
            pos = int(y_true_bin.sum().item())
            neg = int((~y_true_bin).sum().item())
            if pos == 0 or neg == 0:
                return float('nan'), pos, neg

            # move to CPU and optionally subsample
            y_true_cpu = y_true_bin.flatten().cpu().numpy().astype(np.int32)
            y_score_cpu = y_score.flatten().cpu().numpy()
            max_n = 20000
            if y_true_cpu.size > max_n:
                # Stratified subsample to preserve both classes
                pos_idx = np.where(y_true_cpu == 1)[0]
                neg_idx = np.where(y_true_cpu == 0)[0]
                # take up to half from each class
                take_pos = min(len(pos_idx), max_n // 2)
                take_neg = min(len(neg_idx), max_n - take_pos)
                # if one side is smaller, fill remainder from the other
                if take_pos < max_n // 2:
                    take_neg = min(len(neg_idx), max_n - take_pos)
                if take_neg < (max_n - take_pos):
                    take_pos = min(len(pos_idx), max_n - take_neg)
                sel = []
                if take_pos > 0:
                    sel.append(np.random.choice(pos_idx, size=take_pos, replace=False))
                if take_neg > 0:
                    sel.append(np.random.choice(neg_idx, size=take_neg, replace=False))
                sel = np.concatenate(sel) if len(sel) > 0 else np.array([], dtype=np.int64)
                if sel.size > 0:
                    y_true_cpu = y_true_cpu[sel]
                    y_score_cpu = y_score_cpu[sel]

            return _auc_roc_numpy(y_true_cpu, y_score_cpu), pos, neg

        imp_peak_auc, imp_peak_pos, imp_peak_neg = _auc(masked_map)
        obs_peak_auc, obs_peak_pos, obs_peak_neg = _auc(observed_map)

        return {
            'step': int(step_idx),
            # losses
            'imp_count_loss': _safe_float(losses.get('imp_count_loss', float('nan'))),
            'obs_count_loss': _safe_float(losses.get('obs_count_loss', float('nan'))),
            'imp_pval_loss': _safe_float(losses.get('imp_pval_loss', float('nan'))),
            'obs_pval_loss': _safe_float(losses.get('obs_pval_loss', float('nan'))),
            'imp_peak_loss': _safe_float(losses.get('imp_peak_loss', float('nan'))),
            'obs_peak_loss': _safe_float(losses.get('obs_peak_loss', float('nan'))),
            # metrics
            'imp_peak_auc': float(imp_peak_auc),
            'obs_peak_auc': float(obs_peak_auc),
            'imp_peak_pos': int(imp_peak_pos),
            'imp_peak_neg': int(imp_peak_neg),
            'obs_peak_pos': int(obs_peak_pos),
            'obs_peak_neg': int(obs_peak_neg),
            'imp_count_pcc': float(imp_count_pcc),
            'obs_count_pcc': float(obs_count_pcc),
            'imp_pval_pcc': float(imp_pval_pcc),
            'obs_pval_pcc': float(obs_pval_pcc),
        }

    def _print_batch_log(self, metrics, loss_dict, batch_idx, epoch, batch_processing_time=None):
        """
        Refinement-aware batch logging.

        For each batch prints:
        - A visually separated block
        - Delta-only summary across refinement steps (colored)
        - Microbatching context (minimal)

        Also records CSV progress with only deltas (per user request).
        """
        if not self.is_main_process:
            return

        # Current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        grad_norm = getattr(self, 'grad_norm', float('nan'))

        # Masking probabilities info
        mask_info = "N/A"
        if hasattr(self, 'last_masking_probs'):
            probs = self.last_masking_probs
            mask_info = f"A:{probs.get('p_full_assay', 0.0):.1f}/L:{probs.get('p_full_loci', 0.0):.1f}/C:{probs.get('p_chunks', 0.0):.1f}"

        # Batch time
        batch_time_str = "N/A"
        batch_time = 0.0
        if batch_processing_time is not None:
            batch_time = float(batch_processing_time)
            batch_time_str = f"{batch_processing_time:.2f}s"

        # Get refinement step summaries/deltas from last _process_batch
        deltas = self._last_refine_deltas or {}
        probe = self._last_refine_probe_info or {}

        # Header block
        sep_big = "=" * 92
        print(sep_big)
        print(f"REFINE Δ | Epoch {epoch+1} | Batch {batch_idx}/{getattr(self, 'estimated_batches_per_epoch', '?')} "
              f"| LR {current_lr:.2e} | GradNorm {grad_norm:.2f} | time {batch_time_str}")
        if probe:
            print(f"Microbatching: microbatch_size={probe.get('microbatch_size')} | total_microbatches={probe.get('num_microbatches')} "
                  f"| probe_microbatch={probe.get('probe_microbatch_idx')} (rows {probe.get('probe_rows')})")
        print(f"Mask Probs: {mask_info}")

        # Delta summary (final - initial)
        if deltas:
            # Loss deltas: negative is improvement
            imp_count = _colorize_delta(deltas.get('delta_imp_count_loss', float('nan')), improvement_if_positive=False)
            imp_pval = _colorize_delta(deltas.get('delta_imp_pval_loss', float('nan')), improvement_if_positive=False)
            imp_peak = _colorize_delta(deltas.get('delta_imp_peak_loss', float('nan')), improvement_if_positive=False)
            obs_count = _colorize_delta(deltas.get('delta_obs_count_loss', float('nan')), improvement_if_positive=False)
            obs_pval = _colorize_delta(deltas.get('delta_obs_pval_loss', float('nan')), improvement_if_positive=False)
            obs_peak = _colorize_delta(deltas.get('delta_obs_peak_loss', float('nan')), improvement_if_positive=False)

            # Metric deltas: positive is improvement
            imp_auc = _colorize_delta(deltas.get('delta_imp_peak_auc', float('nan')), improvement_if_positive=True)
            imp_cpcc = _colorize_delta(deltas.get('delta_imp_count_pcc', float('nan')), improvement_if_positive=True)
            imp_ppcc = _colorize_delta(deltas.get('delta_imp_pval_pcc', float('nan')), improvement_if_positive=True)
            obs_auc = _colorize_delta(deltas.get('delta_obs_peak_auc', float('nan')), improvement_if_positive=True)
            obs_cpcc = _colorize_delta(deltas.get('delta_obs_count_pcc', float('nan')), improvement_if_positive=True)
            obs_ppcc = _colorize_delta(deltas.get('delta_obs_pval_pcc', float('nan')), improvement_if_positive=True)

            print("Δ over refinement steps (final - initial):")
            print(f"  ΔLoss  (Imp): count={imp_count} | pval={imp_pval} | peak={imp_peak}")
            print(f"  ΔLoss  (Obs): count={obs_count} | pval={obs_pval} | peak={obs_peak}")
            print(f"  ΔMetric(Imp): peak_auc={imp_auc} | count_pcc={imp_cpcc} | pval_pcc={imp_ppcc}")
            print(f"  ΔMetric(Obs): peak_auc={obs_auc} | count_pcc={obs_cpcc} | pval_pcc={obs_ppcc}")

            # EMA of delta PCCs (learning to refine)
            ema_imp_cpcc = self.delta_ema.get('delta_imp_count_pcc', float('nan'))
            ema_imp_ppcc = self.delta_ema.get('delta_imp_pval_pcc', float('nan'))
            ema_obs_cpcc = self.delta_ema.get('delta_obs_count_pcc', float('nan'))
            ema_obs_ppcc = self.delta_ema.get('delta_obs_pval_pcc', float('nan'))

            ema_imp_cpcc_s = _colorize_delta(ema_imp_cpcc, improvement_if_positive=True)
            ema_imp_ppcc_s = _colorize_delta(ema_imp_ppcc, improvement_if_positive=True)
            ema_obs_cpcc_s = _colorize_delta(ema_obs_cpcc, improvement_if_positive=True)
            ema_obs_ppcc_s = _colorize_delta(ema_obs_ppcc, improvement_if_positive=True)

            print(f"  EMA(ΔPCC): Imp count_pcc={ema_imp_cpcc_s} | Imp pval_pcc={ema_imp_ppcc_s} | "
                  f"Obs count_pcc={ema_obs_cpcc_s} | Obs pval_pcc={ema_obs_ppcc_s}  (alpha={self.delta_ema_alpha})")
        else:
            print("Δ over refinement steps: (not available for this batch)")

        # Record CSV progress with deltas ONLY (plus minimal context)
        self._record_progress(epoch, batch_idx, deltas, current_lr=current_lr, grad_norm=grad_norm,
                              mask_info=mask_info, batch_time=batch_time)
        self._save_progress_to_csv(epoch, batch_idx)
        print(sep_big)
        print("")

        # Log to W&B
        try:
            import wandb
            if wandb.run is not None:
                log_data = {
                    "epoch": epoch + 1,
                    "batch": batch_idx,
                    "lr": current_lr,
                    "grad_norm": grad_norm,
                    **loss_dict
                }
                
                # Log deltas
                if deltas:
                    for k, v in deltas.items():
                        log_data[f"deltas/{k}"] = v
                
                # Log EMA trends
                for k, v in self.delta_ema.items():
                    log_data[f"ema/{k}"] = v
                    
                wandb.log(log_data)
        except ImportError:
            pass

    def _record_progress(self, epoch, batch_idx, deltas, *, current_lr, grad_norm, mask_info, batch_time):
        """
        Override: record ONLY deltas for refinement-aware monitoring.
        """
        if not hasattr(self, 'progress_data'):
            self.progress_data = []

        # Update EMA from this batch's deltas
        self._update_delta_ema(deltas or {})

        record = {
            'epoch': epoch + 1,
            'batch_idx': int(batch_idx),
            'timestamp': datetime.now().isoformat(),
            'learning_rate': float(current_lr),
            'gradient_norm': float(grad_norm) if grad_norm is not None else float('nan'),
            'mask_info': str(mask_info),
            'batch_time': float(batch_time),
            'refinement_iterations': int(self.refinement_iterations),
            'gradient_flow': str(self.gradient_flow),
            'microbatch_size': int(self.training_params.get('microbatch_size', 0) or 0),
            # deltas only
            **{k: float(v) for k, v in (deltas or {}).items()}
        }

        # EMA of all delta metrics and losses (trend signal for CSV)
        record.update({
            # EMA of loss deltas
            'ema_delta_imp_count_loss': float(self.delta_ema.get('delta_imp_count_loss', float('nan'))),
            'ema_delta_obs_count_loss': float(self.delta_ema.get('delta_obs_count_loss', float('nan'))),
            'ema_delta_imp_pval_loss': float(self.delta_ema.get('delta_imp_pval_loss', float('nan'))),
            'ema_delta_obs_pval_loss': float(self.delta_ema.get('delta_obs_pval_loss', float('nan'))),
            'ema_delta_imp_peak_loss': float(self.delta_ema.get('delta_imp_peak_loss', float('nan'))),
            'ema_delta_obs_peak_loss': float(self.delta_ema.get('delta_obs_peak_loss', float('nan'))),
            # EMA of metric deltas
            'ema_delta_imp_peak_auc': float(self.delta_ema.get('delta_imp_peak_auc', float('nan'))),
            'ema_delta_obs_peak_auc': float(self.delta_ema.get('delta_obs_peak_auc', float('nan'))),
            'ema_delta_imp_count_pcc': float(self.delta_ema.get('delta_imp_count_pcc', float('nan'))),
            'ema_delta_obs_count_pcc': float(self.delta_ema.get('delta_obs_count_pcc', float('nan'))),
            'ema_delta_imp_pval_pcc': float(self.delta_ema.get('delta_imp_pval_pcc', float('nan'))),
            'ema_delta_obs_pval_pcc': float(self.delta_ema.get('delta_obs_pval_pcc', float('nan'))),
            'ema_alpha': float(self.delta_ema_alpha),
        })
        self.progress_data.append(record)

    def _save_progress_to_csv(self, epoch, batch_idx):
        """
        Override: save refinement progress CSV (deltas-only).
        """
        if not self.progress_data:
            return

        if self.progress_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.progress_file = Path(self.progress_dir) / f"refinement_progress_deltas_{timestamp}.csv"
            if self.is_main_process:
                print(f"Progress will be saved to: {self.progress_file}")

        df = pd.DataFrame(self.progress_data)
        df.to_csv(self.progress_file, index=False)
        if self.is_main_process and (batch_idx % 100 == 0 or batch_idx < 5):
            print(f"Progress updated in {self.progress_file} ({len(df)} records)")
    
    def _process_batch(self, batch):
        """
        Process a single batch with Iterative Refinement.
        
        This method closely mirrors the parent CANDI_TRAINER._process_batch but adds:
        1. Multiple forward passes (Pass 0 + K refinement iterations)
        2. Refinement constraints between passes
        3. Accumulated loss across all passes
        
        The structure follows train.py exactly for consistency.
        """
        # ========== 1. EXTRACT DATA FROM BATCH (same as train.py) ==========
        x_data = batch['x_data'].float()
        x_meta = batch['x_meta'].float()
        x_avail = batch['x_avail']
        x_dna = batch['x_dna'].float()

        y_data = batch['y_data'].float()
        y_meta = batch['y_meta'].float()
        y_pval = batch['y_pval'].float()
        y_peaks = batch['y_peaks'].float()
        
        control_data = batch['control_data'].float()
        control_meta = batch['control_meta'].float()
        control_avail = batch['control_avail']
        
        # ========== 2. AUGMENTATION (same as train.py) ==========
        reverse_complement_prob = self.training_params.get('reverse_complement_prob', 0.5)
        if torch.rand(1).item() < reverse_complement_prob:
            x_dna = reverse_complement_dna(x_dna)
            x_data = reverse_signal(x_data)
            y_data = reverse_signal(y_data)
            y_pval = reverse_signal(y_pval)
            y_peaks = reverse_signal(y_peaks)
            control_data = reverse_signal(control_data)
        
        # ========== 3. MASKING SETUP (same as train.py) ==========
        if not hasattr(self, 'masker'):
            self.masker = DataMasker(
                mask_value=TOKEN_DICT["cloze_mask"],
                chunk_size=self.training_params.get('chunk_size', 40),
                mask_fraction=self.training_params.get('mask_fraction', 0.20),
                p_full_loci=self.training_params.get('p_full_loci', 0.0),
                p_full_assay=self.training_params.get('p_full_assay', 1.0),
                p_chunks=self.training_params.get('p_chunks', 0.0)
            )
        
        B, L, F = x_data.shape
        x_data_masked = x_data.clone()
        x_meta_masked = x_meta.clone()
        x_avail_masked = x_avail.clone()
        
        x_data_masked, x_meta_masked, x_avail_masked = self.masker.apply_mask(
            x_data_masked, x_meta_masked, x_avail_masked
        )
        
        # Create masks for loss computation (same as train.py)
        masked_map = (x_data_masked == TOKEN_DICT["cloze_mask"])
        observed_map = (x_data_masked != TOKEN_DICT["missing_mask"]) & (x_data_masked != TOKEN_DICT["cloze_mask"])
        observed_map = observed_map.clone()
        masked_map = masked_map.clone()
        
        # Create missing_mask: positions that were originally missing (before cloze masking)
        missing_mask = (x_data == TOKEN_DICT["missing_mask"])
        missing_mask = missing_mask.clone()
        
        # Create imputation_map: ALL positions that need imputation (cloze-masked OR originally missing)
        imputation_map = masked_map | missing_mask
        imputation_map = imputation_map.clone()
        
        # Store masking info for logging (same as train.py)
        self.last_masking_probs = self.masker.get_probabilities()
        
        # Validate (same as train.py)
        if not observed_map.any():
            if self.is_main_process:
                print("Warning: No observed regions found! Skipping batch...")
            return None
        
        has_masked_regions = masked_map.any()
        
        # Move masks to device (same as train.py)
        masked_map = masked_map.to(self.device)
        observed_map = observed_map.to(self.device)
        missing_mask = missing_mask.to(self.device)
        imputation_map = imputation_map.to(self.device)
        
        # ========== 4. ZERO GRADIENTS (same as train.py) ==========
        # set_to_none=True reduces peak memory and improves performance
        self.optimizer.zero_grad(set_to_none=True)

        # Optional microbatching (gradient accumulation inside a batch)
        microbatch_size = self.training_params.get('microbatch_size', None)
        if microbatch_size is None or microbatch_size <= 0:
            microbatch_size = B
        microbatch_size = min(int(microbatch_size), B)
        num_microbatches = int(math.ceil(B / microbatch_size)) if microbatch_size > 0 else 1

        # We'll compute per-step refinement summaries on a probe microbatch (first one) for reporting
        probe_microbatch_idx = 0
        probe_step_summaries = None

        # Ensure tensors are on device (train.py already moved batch to device, but keep this safe)
        y_data = y_data.to(self.device)
        y_pval = y_pval.to(self.device)
        y_peaks = y_peaks.to(self.device)
        x_dna = x_dna.to(self.device)
        y_meta = y_meta.to(self.device)
        control_data = control_data.to(self.device)
        control_meta = control_meta.to(self.device)
        x_data = x_data.to(self.device)
        x_meta = x_meta.to(self.device)
        x_data_masked = x_data_masked.to(self.device)
        x_meta_masked = x_meta_masked.to(self.device)
        x_avail = x_avail.to(self.device)

        # Accumulators for logging (CPU floats)
        loss_sums = {
            'total_loss': 0.0,
            'obs_count_loss': 0.0,
            'imp_count_loss': 0.0,
            'obs_pval_loss': 0.0,
            'imp_pval_loss': 0.0,
            'obs_peak_loss': 0.0,
            'imp_peak_loss': 0.0,
        }
        metrics = None  # will compute from the last microbatch for simplicity

        def _oom_cleanup():
            """Best-effort cleanup after CUDA OOM to allow training to continue."""
            try:
                self.optimizer.zero_grad(set_to_none=True)
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass

        # Process microbatches sequentially
        for mb_start in range(0, B, microbatch_size):
            mb_end = min(B, mb_start + microbatch_size)
            mb_n = mb_end - mb_start
            weight = float(mb_n) / float(B)  # gradient weight to match full-batch averaging
            mb_idx = mb_start // microbatch_size if microbatch_size > 0 else 0

            # Slice microbatch tensors
            x_data_mb = x_data[mb_start:mb_end]
            x_meta_mb = x_meta[mb_start:mb_end]
            x_data_masked_mb = x_data_masked[mb_start:mb_end]
            x_meta_masked_mb = x_meta_masked[mb_start:mb_end]
            x_dna_mb = x_dna[mb_start:mb_end]
            y_data_mb = y_data[mb_start:mb_end]
            y_pval_mb = y_pval[mb_start:mb_end]
            y_peaks_mb = y_peaks[mb_start:mb_end]
            y_meta_mb = y_meta[mb_start:mb_end]
            control_data_mb = control_data[mb_start:mb_end]
            control_meta_mb = control_meta[mb_start:mb_end]
            masked_map_mb = masked_map[mb_start:mb_end]
            observed_map_mb = observed_map[mb_start:mb_end]
            missing_mask_mb = missing_mask[mb_start:mb_end]
            imputation_map_mb = imputation_map[mb_start:mb_end]
            x_avail_mb = x_avail[mb_start:mb_end]

            # ========== 5. REFINEMENT LOOP (per-microbatch) ==========
            current_signal = x_data_masked_mb
            current_meta = x_meta_masked_mb
            original_obs = x_data_mb  # already on device
            best_n = None
            meta_baseline = None  # prompt-filled metadata baseline (non-sticky across Gibbs steps)

            total_loss_accum = None  # accumulate step losses without building extra graph on a requires_grad leaf

            final_output_p = final_output_n = final_output_mu = final_output_var = final_output_peak = None
            final_obs_count_loss = final_imp_count_loss = None
            final_obs_pval_loss = final_imp_pval_loss = None
            final_obs_peak_loss = final_imp_peak_loss = None

            num_passes = self.refinement_iterations + 1

            try:
                for t in range(num_passes):
                    model_input = torch.cat([current_signal, control_data_mb], dim=2)
                    model_meta = torch.cat([current_meta, control_meta_mb], dim=2)

                    # Forward
                    if self.use_mixed_precision:
                        with autocast('cuda'):
                            output_p, output_n, output_mu, output_var, output_peak = self.model(
                                model_input, x_dna_mb, model_meta, y_meta_mb
                            )
                    else:
                        output_p, output_n, output_mu, output_var, output_peak = self.model(
                            model_input, x_dna_mb, model_meta, y_meta_mb
                        )

                    # Validate outputs
                    if torch.isnan(output_p).any() or torch.isnan(output_n).any() or \
                       torch.isnan(output_mu).any() or torch.isnan(output_var).any() or \
                       torch.isnan(output_peak).any():
                        if self.is_main_process:
                            print("Warning: NaN in model outputs! Skipping batch...")
                        _oom_cleanup()
                        return None

                    # Loss
                    if self.use_mixed_precision:
                        with autocast('cuda'):
                            if has_masked_regions:
                                obs_count_loss, imp_count_loss, obs_pval_loss, imp_pval_loss, obs_peak_loss, imp_peak_loss = self.criterion(
                                    output_p, output_n, output_mu, output_var, output_peak,
                                    y_data_mb, y_pval_mb, y_peaks_mb, observed_map_mb, masked_map_mb
                                )
                                step_loss = obs_count_loss + obs_pval_loss + obs_peak_loss + imp_count_loss + imp_pval_loss + imp_peak_loss
                            else:
                                obs_count_loss, _, obs_pval_loss, _, obs_peak_loss, _ = self.criterion(
                                    output_p, output_n, output_mu, output_var, output_peak,
                                    y_data_mb, y_pval_mb, y_peaks_mb, observed_map_mb, observed_map_mb
                                )
                                imp_count_loss = torch.tensor(0.0, device=self.device)
                                imp_pval_loss = torch.tensor(0.0, device=self.device)
                                imp_peak_loss = torch.tensor(0.0, device=self.device)
                                step_loss = obs_count_loss + obs_pval_loss + obs_peak_loss
                    else:
                        if has_masked_regions:
                            obs_count_loss, imp_count_loss, obs_pval_loss, imp_pval_loss, obs_peak_loss, imp_peak_loss = self.criterion(
                                output_p, output_n, output_mu, output_var, output_peak,
                                y_data_mb, y_pval_mb, y_peaks_mb, observed_map_mb, masked_map_mb
                            )
                            step_loss = obs_count_loss + obs_pval_loss + obs_peak_loss + imp_count_loss + imp_pval_loss + imp_peak_loss
                        else:
                            obs_count_loss, _, obs_pval_loss, _, obs_peak_loss, _ = self.criterion(
                                output_p, output_n, output_mu, output_var, output_peak,
                                y_data_mb, y_pval_mb, y_peaks_mb, observed_map_mb, observed_map_mb
                            )
                            imp_count_loss = torch.tensor(0.0, device=self.device)
                            imp_pval_loss = torch.tensor(0.0, device=self.device)
                            imp_peak_loss = torch.tensor(0.0, device=self.device)
                            step_loss = obs_count_loss + obs_pval_loss + obs_peak_loss

                    total_loss_accum = step_loss if total_loss_accum is None else (total_loss_accum + step_loss)

                    # Collect per-step summaries on probe microbatch ONLY (to keep overhead manageable)
                    if mb_idx == probe_microbatch_idx:
                        if probe_step_summaries is None:
                            probe_step_summaries = []
                        step_losses_dict = {
                            'obs_count_loss': obs_count_loss.detach(),
                            'imp_count_loss': imp_count_loss.detach(),
                            'obs_pval_loss': obs_pval_loss.detach(),
                            'imp_pval_loss': imp_pval_loss.detach(),
                            'obs_peak_loss': obs_peak_loss.detach(),
                            'imp_peak_loss': imp_peak_loss.detach(),
                        }
                        probe_step_summaries.append(
                            self._compute_refine_step_summary(
                                step_idx=t,
                                output_p=output_p,
                                output_n=output_n,
                                output_mu=output_mu,
                                output_peak=output_peak,
                                y_data=y_data_mb,
                                y_pval=y_pval_mb,
                                y_peaks=y_peaks_mb,
                                observed_map=observed_map_mb,
                                masked_map=masked_map_mb,
                                losses=step_losses_dict
                            )
                        )

                    # Save final pass outputs/losses for metrics/logging
                    if t == num_passes - 1:
                        final_output_p = output_p
                        final_output_n = output_n
                        final_output_mu = output_mu
                        final_output_var = output_var
                        final_output_peak = output_peak
                        final_obs_count_loss = obs_count_loss
                        final_imp_count_loss = imp_count_loss
                        final_obs_pval_loss = obs_pval_loss
                        final_imp_pval_loss = imp_pval_loss
                        final_obs_peak_loss = obs_peak_loss
                        final_imp_peak_loss = imp_peak_loss

                    # Next input
                    if t < num_passes - 1:
                        # IMPORTANT: Feed back NB mean (count space), NOT Gaussian mu.
                        # NB mean: mu_nb = n * (1-p) / p
                        eps = 1e-8
                        p_clamped = torch.clamp(output_p, min=eps, max=1.0 - eps)
                        n_clamped = torch.clamp(output_n, min=eps)
                        mu_nb = (n_clamped * (1.0 - p_clamped)) / p_clamped

                        if self.gradient_flow == 'stop':
                            pred_mu = mu_nb.detach()
                            pred_n = output_n.detach()
                        else:
                            pred_mu = mu_nb
                            pred_n = output_n

                        # Build meta_baseline right after pass 0 so the refined inputs match the prompt metadata.
                        # This also prevents Gibbs-masked metadata from persisting ("sticking") across steps:
                        # each step's metadata mask is applied fresh on top of meta_baseline.
                        if meta_baseline is None:
                            if y_meta_mb is not None:
                                meta_baseline = x_meta_masked_mb.clone()
                                meta_bad = (meta_baseline == TOKEN_DICT["missing_mask"]) | (meta_baseline == TOKEN_DICT["cloze_mask"])
                                meta_bad_assay = meta_bad.any(dim=1)  # [B, F]
                                if meta_bad_assay.any():
                                    meta_bad_broadcast = meta_bad_assay.unsqueeze(1).expand_as(meta_baseline)
                                    meta_baseline[meta_bad_broadcast] = y_meta_mb[meta_bad_broadcast]
                            else:
                                meta_baseline = x_meta_masked_mb.clone()

                        next_signal, next_meta, best_n = self._apply_refinement_constraints(
                            current_input=current_signal,
                            # IMPORTANT: apply Gibbs metadata masking on top of the prompt-filled baseline,
                            # not on top of previously masked metadata (prevents masks from "sticking").
                            current_meta=meta_baseline,
                            y_meta_prompt=y_meta_mb,
                            pred_mu=pred_mu,
                            pred_n=pred_n,
                            observed_map=observed_map_mb,
                            masked_map=masked_map_mb,
                            original_obs=original_obs,
                            best_n=best_n,
                            t=t,
                            imputation_map=imputation_map_mb,
                            x_avail=x_avail_mb
                        )
                        current_signal = next_signal
                        current_meta = next_meta

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if self.is_main_process:
                        alloc = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
                        reserved = torch.cuda.memory_reserved() / (1024**3) if torch.cuda.is_available() else 0.0
                        print(f"Warning: CUDA Out of Memory during refinement (microbatch {mb_start}:{mb_end}, size {mb_n}/{B}). "
                              f"allocated={alloc:.2f}GB reserved={reserved:.2f}GB. Skipping batch...")
                    _oom_cleanup()
                    return None
                raise e

            # ========== 6. COMPUTE AVERAGE LOSS (per microbatch) ==========
            total_loss_mb = total_loss_accum / num_passes
            if torch.isnan(total_loss_mb).any():
                if self.is_main_process:
                    print("Warning: Encountered NaN loss! Skipping batch...")
                _oom_cleanup()
                return None

            # Backprop immediately to free the graph for this microbatch
            # Weight by mb_n/B so gradients match full-batch averaging.
            try:
                if self.use_mixed_precision:
                    self.scaler.scale(total_loss_mb * weight).backward()
                else:
                    (total_loss_mb * weight).float().backward()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if self.is_main_process:
                        print(f"Warning: CUDA Out of Memory during backward (microbatch {mb_start}:{mb_end}). Skipping batch...")
                    _oom_cleanup()
                    return None
                raise e

            # Accumulate logging scalars (detach to avoid keeping graphs)
            loss_sums['total_loss'] += float(total_loss_mb.detach().item()) * weight
            loss_sums['obs_count_loss'] += float(final_obs_count_loss.detach().item()) * weight
            loss_sums['imp_count_loss'] += float(final_imp_count_loss.detach().item()) * weight
            loss_sums['obs_pval_loss'] += float(final_obs_pval_loss.detach().item()) * weight
            loss_sums['imp_pval_loss'] += float(final_imp_pval_loss.detach().item()) * weight
            loss_sums['obs_peak_loss'] += float(final_obs_peak_loss.detach().item()) * weight
            loss_sums['imp_peak_loss'] += float(final_imp_peak_loss.detach().item()) * weight

            # Metrics (compute only for the last microbatch to keep overhead reasonable)
            if mb_end == B:
                metrics = {}
        
        # ========== 7. OPTIMIZER STEP (same as train.py) ==========
        # Unscale once (after all microbatches), clip once, step once.
        try:
            clip_mode = self.training_params.get('clip_mode', 'norm')
            clip_value = self.training_params.get('clip_value', 2.0)

            if self.use_mixed_precision:
                self.scaler.unscale_(self.optimizer)

            if clip_mode == 'norm':
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_value)
            else:
                parameters = [p for p in self.model.parameters() if p.grad is not None]
                if len(parameters) > 0:
                    device = parameters[0].grad.device
                    grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0).to(device) for p in parameters]), 2.0)
                else:
                    grad_norm = torch.tensor(0.0, device=self.device)
                if clip_mode == 'value':
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=clip_value)

            if self.use_mixed_precision:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if self.is_main_process:
                    print("Warning: CUDA Out of Memory during optimizer step / clipping. Skipping batch...")
                _oom_cleanup()
                return None
            raise e
        
        # Store gradient norm (same as train.py)
        self.grad_norm = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

        # ========== 8. BUILD LOSS DICT (same as train.py) ==========
        loss_dict = {k: float(v) for k, v in loss_sums.items()}
        if metrics is None:
            metrics = {}

        # Build deltas (final - initial) from probe step summaries for monitoring/CSV
        deltas = {}
        if probe_step_summaries and len(probe_step_summaries) >= 2:
            s0 = probe_step_summaries[0]
            sf = probe_step_summaries[-1]
            delta_keys = [
                ('delta_imp_count_loss', 'imp_count_loss'),
                ('delta_obs_count_loss', 'obs_count_loss'),
                ('delta_imp_pval_loss', 'imp_pval_loss'),
                ('delta_obs_pval_loss', 'obs_pval_loss'),
                ('delta_imp_peak_loss', 'imp_peak_loss'),
                ('delta_obs_peak_loss', 'obs_peak_loss'),
                ('delta_imp_peak_auc', 'imp_peak_auc'),
                ('delta_obs_peak_auc', 'obs_peak_auc'),
                ('delta_imp_count_pcc', 'imp_count_pcc'),
                ('delta_obs_count_pcc', 'obs_count_pcc'),
                ('delta_imp_pval_pcc', 'imp_pval_pcc'),
                ('delta_obs_pval_pcc', 'obs_pval_pcc'),
            ]
            for dk, sk in delta_keys:
                deltas[dk] = _safe_float(sf.get(sk, float('nan'))) - _safe_float(s0.get(sk, float('nan')))

        # Store for _print_batch_log
        self._last_refine_step_summaries = probe_step_summaries
        self._last_refine_deltas = deltas
        self._last_refine_probe_info = {
            'microbatch_size': microbatch_size,
            'num_microbatches': num_microbatches,
            'probe_microbatch_idx': probe_microbatch_idx,
            'probe_rows': f"{0}:{min(microbatch_size, B)}",
            'batch_size': B,
        }
        
        # ========== 11. RETURN (same as train.py) ==========
        return_dict = {**loss_dict, **metrics, **deltas}
        return return_dict


##=========================================== CLI Interface =============================================##

def create_refinement_argument_parser():
    """Create argument parser for refinement training."""
    parser = create_base_argument_parser()
    
    # Add Refinement-specific Arguments
    refine_group = parser.add_argument_group('Refinement Configuration')
    
    refine_group.add_argument('--refinement-model-path', type=str, required=True,
                             help='Path to the pretrained model checkpoint (.pt file) or model directory')
    refine_group.add_argument('--refinement-iterations', type=int, default=1,
                             help='Number of refinement passes (default 1 = 1 initial + 1 refine)')
    refine_group.add_argument('--refinement-lambda', type=float, default=0.0,
                             help='Soft constraint lambda: 0.0=Hard Obs Replacement, 1.0=Full Prediction Blend')
    refine_group.add_argument('--gating-strategy', type=str, default='none',
                             choices=['none', 'preds', 'obs', 'both'],
                             help='Confidence gating: none=disabled, preds=gate predictions, obs=gate observed, both=gate all')
    refine_group.add_argument('--gibbs-prob', type=float, default=0.0,
                             help='Probability of re-masking for Gibbs sampling (0.0=disabled)')
    refine_group.add_argument('--gibbs-scope', type=str, default='obs',
                             choices=['obs', 'preds', 'both'],
                             help='Scope for Gibbs re-masking: obs=observed only, preds=predictions only, both=all')
    refine_group.add_argument('--gradient-flow', type=str, default='stop',
                             choices=['stop', 'bptt'],
                             help='Gradient flow: stop=detach (Teacher Forcing), bptt=backprop through time')

    refine_group.add_argument('--microbatch-size', type=int, default=None,
                             help='Optional microbatch size for gradient accumulation inside each batch. '
                                  'Useful to avoid CUDA OOM, especially for --gradient-flow bptt. '
                                  'If not set, uses full batch-size.')
    
    return parser


def load_pretrained_model_config(model_path):
    """Load configuration from pretrained model directory."""
    model_path = Path(model_path)
    
    if model_path.is_file():
        model_dir = model_path.parent
    else:
        model_dir = model_path
    
    # Find config file
    config_files = list(model_dir.glob("*_config.json"))
    if not config_files:
        raise FileNotFoundError(f"No config JSON file found in {model_dir}")
    
    config_path = config_files[0]
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Loaded pretrained config from: {config_path}")
    return config, model_dir


def main():
    """Main entry point for refinement training."""
    parser = create_refinement_argument_parser()
    args = parser.parse_args()
    
    # Load config file if specified
    if args.config:
        try:
            config = load_config_file(args.config)
            for key, value in config.items():
                key_attr = key.replace('-', '_')
                if not hasattr(args, key_attr) or getattr(args, key_attr) == parser.get_default(key_attr):
                    setattr(args, key_attr, value)
        except Exception as e:
            print(f"Error loading configuration file: {e}")
            return 1
    
    # Handle GPU check
    if args.check_gpus:
        check_gpu_availability()
        return 0
    
    # Load pretrained model config
    try:
        pretrained_config, pretrained_model_dir = load_pretrained_model_config(args.refinement_model_path)
    except Exception as e:
        print(f"Error loading pretrained model config: {e}")
        return 1
    
    # Override model architecture args from pretrained config
    arch_keys = [
        'signal_dim', 'nhead', 'n-sab-layers', 'n-cnn-layers', 'conv-kernel-size',
        'pool-size', 'expansion-factor', 'dropout', 'pos-enc', 'attention-type',
        'separate-decoders', 'norm-type', 'unet', 'context-length', 'output-ff',
        'dist-type'
    ]
    for key in arch_keys:
        key_attr = key.replace('-', '_')
        if key in pretrained_config:
            setattr(args, key_attr, pretrained_config[key])
    
    # Set dataset type: default to pretrained config, but allow CLI override
    # Check if user explicitly specified --eic or --merged in CLI
    # (Both will be False if not specified, since they're action='store_true')
    user_specified_eic = getattr(args, 'eic', False)
    user_specified_merged = getattr(args, 'merged', False)
    
    if user_specified_eic or user_specified_merged:
        # User explicitly specified --eic or --merged, use that (already set by argparse)
        if user_specified_eic:
            print(f"Using dataset type from CLI: EIC (overriding pretrained config)")
        else:
            print(f"Using dataset type from CLI: merged (overriding pretrained config)")
    else:
        # User didn't specify, so use the pretrained config's dataset type
        if pretrained_config.get('eic', False):
            args.eic = True
            args.merged = False
            print(f"Using dataset type from pretrained config: EIC")
        elif pretrained_config.get('merged', False):
            args.eic = False
            args.merged = True
            print(f"Using dataset type from pretrained config: merged")
        else:
            # Fallback: infer from model directory name
            if 'eic' in str(pretrained_model_dir).lower():
                args.eic = True
                args.merged = False
                print(f"Inferred dataset type from model directory: EIC")
            else:
                args.eic = False
                args.merged = True
                print(f"Inferred dataset type from model directory: merged (default)")

    # ===========================================
    # Refinement-specific constraint: DSF must be 1
    # ===========================================
    # Iterative refinement is defined at the native resolution (DSF=1). Using dsf_list>1
    # changes the tensor shapes and makes refinement behavior inconsistent.
    if getattr(args, 'dsf_list', None) != [1]:
        print(f"Refinement training: forcing --dsf-list to [1] (was {getattr(args, 'dsf_list', None)})")
    args.dsf_list = [1]
    
    # Handle mixed precision
    if args.no_mixed_precision:
        args.mixed_precision = False
    
    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    
    # Setup device
    device = setup_device(args)
    
    # Initialize DDP if requested
    if args.ddp:
        try:
            rank, world_size, local_rank = init_distributed()
            if rank is not None:
                args.rank, args.world_size = rank, world_size
            else:
                print("Failed to initialize DDP, falling back to single-GPU")
                args.ddp = False
                args.rank = None
                args.world_size = None
        except Exception as e:
            print(f"DDP initialization failed: {e}")
            args.ddp = False
            args.rank = None
            args.world_size = None
    
    # Create dataset parameters
    base_path = args.data_path if args.data_path.endswith('/') else args.data_path + '/'
    data_path = base_path + ("DATA_CANDI_EIC/" if args.eic else "DATA_CANDI_MERGED/")
    
    dataset_params = {
        'base_path': data_path,
        'dataset_type': "eic" if args.eic else "merged",
        'm': args.num_loci,
        'context_length': args.context_length * 25,
        'split': 'train',
        'loci_gen_strategy': args.loci_gen,
        'dsf_list': args.dsf_list,
        'DNA': True,
        'must_have_chr_access': args.must_have_chr_access,
        'bios_min_exp_avail_threshold': args.min_avail,
        'shuffle_bios': True,
        'fill_prompt_mode': args.fill_prompt_mode
    }
    
    # Create training parameters
    training_params = {
        'optimizer': args.optimizer,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'momentum': args.momentum,
        'beta1': args.beta1,
        'beta2': args.beta2,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'inner_epochs': args.inner_epochs,
        'enable_validation': not args.disable_validation,
        'val_freq': args.val_freq,
        'use_mixed_precision': args.mixed_precision,
        'specific_ema_alpha': args.specific_ema_alpha,
        'progress_dir': args.progress_dir,
        'debug': args.debug,
        'DNA': True,
        'no_save': args.no_save,
        'count_weight': args.count_weight,
        'pval_weight': args.pval_weight,
        'peak_weight': args.peak_weight,
        'obs_weight': args.obs_weight,
        'imp_weight': args.imp_weight,
        'p_full_loci': args.p_full_loci,
        'p_full_assay': args.p_full_assay,
        'p_chunks': args.p_chunks,
        'mask_fraction': args.mask_fraction,
        'chunk_size': args.chunk_size,
        'reverse_complement_prob': args.reverse_complement_prob,
        'clip_mode': args.clip_mode,
        'clip_value': args.clip_value,
        'microbatch_size': args.microbatch_size,
        'dist_type': getattr(args, 'dist_type', 'gaussian'),
        # Refinement-specific
        'refinement_iterations': args.refinement_iterations,
        'refinement_lambda': args.refinement_lambda,
        'gating_strategy': args.gating_strategy,
        'gibbs_prob': args.gibbs_prob,
        'gibbs_scope': args.gibbs_scope,
        'gradient_flow': args.gradient_flow,
    }
    
    # Create model
    temp_dataset = CANDIIterableDataset(**dataset_params)
    signal_dim = len(temp_dataset.aliases['experiment_aliases'])
    # Use num_assays instead of num_sequencing_platforms per issue_supertrack.md ToDo 1
    num_assays = temp_dataset.num_assays
    num_runtypes = 4
    
    model = create_model_from_args(args, signal_dim, num_assays, num_runtypes)
    
    # Load pretrained weights
    model_path = Path(args.refinement_model_path)
    if model_path.is_dir():
        checkpoint_files = list(model_path.glob("*.pt"))
        if not checkpoint_files:
            checkpoints_dir = model_path / "checkpoints"
            if checkpoints_dir.exists():
                checkpoint_files = list(checkpoints_dir.glob("*.pt"))
        if not checkpoint_files:
            print(f"Error: No .pt checkpoint found in {model_path}")
            return 1
        checkpoint_path = checkpoint_files[0]
    else:
        checkpoint_path = model_path
    
    print(f"Loading pretrained weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # Initialize Refinement Trainer
    trainer = REFINEMENT_TRAINER(
        model=model,
        dataset_params=dataset_params,
        training_params=training_params,
        device=device,
        rank=args.rank if args.ddp else None,
        world_size=args.world_size if args.ddp else None
    )
    
    # Generate model name with _REFINE suffix
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_model_name = pretrained_model_dir.name
    model_name = f"{base_model_name}_REFINE_{timestamp}"
    
    # Add refinement config to name
    refine_suffix = f"_iter{args.refinement_iterations}"
    if args.refinement_lambda > 0:
        refine_suffix += f"_lambda{args.refinement_lambda}"
    if args.gating_strategy != 'none':
        refine_suffix += f"_gate{args.gating_strategy}"
    if args.gibbs_prob > 0:
        refine_suffix += f"_gibbs{args.gibbs_prob}"
    if args.gradient_flow == 'bptt':
        refine_suffix += "_bptt"
    model_name += refine_suffix
    
    # Initialize W&B
    if not args.ddp or args.rank == 0:  # Only log from main process
        try:
            import wandb
            run = wandb.init(
                project="Refinement_CANDI",
                name=model_name,
                config=vars(args)
            )
            
            # Save W&B URL to a file
            output_dir = pretrained_model_dir / model_name
            output_dir.mkdir(parents=True, exist_ok=True)
            wandb_url_file = output_dir / "wandb_url.txt"
            with open(wandb_url_file, "w") as f:
                f.write(f"W&B Run URL: {run.get_url()}\n")
                f.write(f"Project URL: {run.get_project_url()}\n")
                f.write(f"Run ID: {run.id}\n")
            print(f"W&B URL saved to: {wandb_url_file}")
            
        except ImportError:
            print("Warning: wandb not installed. Skipping W&B initialization.")
    
    trainer.model_name = model_name
    print(f"Model name: {model_name}")
    
    # Setup output directory in the original model folder
    if not args.no_save and (not args.ddp or args.rank == 0):
        output_dir = pretrained_model_dir / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        trainer.progress_dir = str(output_dir)
        trainer.progress_file = None
        trainer.validation_progress_file = None
        
        # Save refinement config
        refine_config = {
            **pretrained_config,
            'dsf_list': [1],
            'refinement_iterations': args.refinement_iterations,
            'refinement_lambda': args.refinement_lambda,
            'gating_strategy': args.gating_strategy,
            'gibbs_prob': args.gibbs_prob,
            'gibbs_scope': args.gibbs_scope,
            'gradient_flow': args.gradient_flow,
            'microbatch_size': args.microbatch_size,
            'refinement_epochs': args.epochs,
            'refinement_learning_rate': args.learning_rate,
            'pretrained_model': str(checkpoint_path),
        }
        config_path = output_dir / f"{model_name}_config.json"
        save_config_file(refine_config, config_path)
        print(f"Refinement config saved to: {config_path}")
    
    # Run training
    start_time = time.time()
    print_training_summary(args, model, device)
    
    trained_model = trainer.train()
    
    end_time = time.time()
    training_duration = end_time - start_time
    hours, remainder = divmod(training_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\nRefinement Training Complete!")
    print(f"Duration: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    print("="*60)
    
    # Save final model
    if not args.no_save and (not args.ddp or args.rank == 0):
        output_dir = pretrained_model_dir / model_name
        model_save_path = output_dir / f"{model_name}.pt"
        
        model_to_save = trained_model.module if hasattr(trained_model, 'module') else trained_model
        torch.save(model_to_save.state_dict(), model_save_path)
        print(f"Refined model saved to: {model_save_path}")
    
    # Cleanup DDP
    if args.ddp:
        cleanup_distributed()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

