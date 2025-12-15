#!/usr/bin/env python3
"""
CANDI Prediction Module

This module provides functionality for loading trained CANDI models and running inference
on genomic data. It supports both merged and EIC datasets with configurable metadata
filling and prediction options.

Author: Refactored from old_eval.py
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import torch.nn as nn

# Import from current codebase
from data import CANDIDataHandler
from model import CANDI, CANDI_UNET
from _utils import NegativeBinomial, Gaussian, DataMasker


class CANDIPredictor:
    """
    CANDI model predictor for loading trained models and running inference.
    
    This class handles model loading from JSON config files and .pt checkpoints,
    data loading using CANDIDataHandler, and running predictions with optional
    latent representation extraction.
    """
    
    def __init__(self, model_dir: str, device: Optional[str] = None, DNA: bool = True):
        """
        Initialize CANDI predictor.
        
        Args:
            model_dir: Path to model directory containing config JSON and .pt checkpoint
            device: Device to use for inference (auto-detect if None)
            DNA: Whether to use DNA sequence input (must be True for current models)
        """
        self.model_dir = Path(model_dir)
        self.DNA = DNA
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize data handler (will be set up when needed)
        self.data_handler = None
        
        # Model and config
        self.model = None
        self.config = None
        
        # Token dictionary for masking
        self.token_dict = {"missing_mask": -1, "cloze_mask": -2, "pad": -3}
        
        # Cached max batch size (auto-detected)
        self._max_batch_size = None
        
        # Load model and config
        self._load_config()
        self._load_model()
        
        print(f"CANDI Predictor initialized on {self.device}")
        print(f"Model: {self.config.get('unet', False) and 'CANDI_UNET' or 'CANDI'}")
        print(f"Signal dim: {self.config.get('signal_dim', 'unknown')}")
    
    def _load_config(self):
        """Load model configuration from JSON file."""
        config_files = list(self.model_dir.glob("*_config.json"))
        if not config_files:
            raise FileNotFoundError(f"No config JSON file found in {self.model_dir}")
        
        # Use the first (and typically only) config file
        config_path = config_files[0]
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        print(f"Loaded config from {config_path}")
    
    def _load_model(self):
        """Load CANDI model from checkpoint."""
        # Find checkpoint file
        checkpoint_files = list(self.model_dir.glob("*.pt"))
        if not checkpoint_files:
            checkpoints_dir = self.model_dir / "checkpoints"
            if checkpoints_dir.exists():
                checkpoint_files = list(checkpoints_dir.glob("*.pt"))
            else:
                raise FileNotFoundError(f"No .pt checkpoint file found in {self.model_dir}")
        
        # Use the first checkpoint file (typically the final model)
        checkpoint_path = checkpoint_files[0]
        
        # Extract model parameters from config
        signal_dim = self.config.get('signal_dim', 35)
        metadata_embedding_dim = signal_dim * 4
        dropout = self.config.get('dropout', 0.1)
        nhead = self.config.get('nhead', 9)
        n_sab_layers = self.config.get('n-sab-layers', 4)
        n_cnn_layers = self.config.get('n-cnn-layers', 3)
        conv_kernel_size = self.config.get('conv-kernel-size', 3)
        pool_size = self.config.get('pool-size', 2)
        context_length = self.config.get('context-length', 1200)
        separate_decoders = self.config.get('separate-decoders', True)
        unet = self.config.get('unet', False)
        pos_enc = self.config.get('pos-enc', 'relative')
        expansion_factor = self.config.get('expansion-factor', 3)
        # Read attention_type and norm from config (important for model architecture compatibility)
        attention_type = self.config.get('attention-type', self.config.get('attention_type', 'dual'))
        norm = self.config.get('norm-type', self.config.get('norm', 'batch'))

        self.context_length = context_length
        
        # Get metadata dimensions
        num_sequencing_platforms = self.config.get('num_sequencing_platforms', 10)
        num_runtypes = self.config.get('num_runtypes', 4) # Based on the mapping in EmbedMetadata: 0, 1, 2 (missing), 3 (cloze_masked)
        
        # Create model
        if unet:
            self.model = CANDI_UNET(
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
            self.model = CANDI(
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
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded model from {checkpoint_path}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def setup_data_handler(self, data_path: str, dataset_type: str = "merged", 
                          context_length: int = 1200, resolution: int = 25, split: str = "test"):
        """
        Setup data handler for loading genomic data.
        
        Args:
            data_path: Path to dataset directory
            dataset_type: Type of dataset ("merged" or "eic")
            context_length: Context length for genomic windows
            resolution: Genomic resolution in bp
        """
        self.data_handler = CANDIDataHandler(
            base_path=data_path,
            resolution=resolution,
            dataset_type=dataset_type,
            DNA=self.DNA
        )
        
        # Load required data files
        self.data_handler._load_files()
        
        print(f"Filtering navigation for split: {split}...")
        for bios in list(self.data_handler.navigation.keys()):
            # Skip invalid keys (like environment variable names or paths)
            if bios not in self.data_handler.split_dict:
                print(f"Warning: Skipping invalid navigation key: {bios}")
                if dataset_type == "merged":
                    del self.data_handler.navigation[bios]
                continue
            if self.data_handler.split_dict[bios] != split:
                if dataset_type == "merged":
                    del self.data_handler.navigation[bios]
        
        print(f"Data handler setup for {dataset_type} dataset at {data_path}")
        print(f"Available experiments: {len(self.data_handler.aliases['experiment_aliases'])}")
    
    def _find_max_batch_size(self, X: torch.Tensor, mX: torch.Tensor, mY: torch.Tensor,
                             avail: torch.Tensor, seq: Optional[torch.Tensor] = None,
                             start_size: int = 1, max_size: int = 256) -> int:
        """
        Automatically find the maximum batch size that fits in GPU memory.
        
        Uses exponential search followed by binary search to efficiently find optimal batch size.
        
        Args:
            X: Sample input count data [B, L, F]
            mX: Sample input metadata [B, 4, F]
            mY: Sample target metadata [B, 4, F]
            avail: Sample availability mask [B, F]
            seq: Sample DNA sequence [B, L*25, 4] (required if DNA=True)
            start_size: Starting batch size to test
            max_size: Maximum batch size to test
            
        Returns:
            Maximum batch size that fits in GPU memory
        """
        if not torch.cuda.is_available() or self.device.type != 'cuda':
            # For CPU, use a conservative default
            return 50
        
        # Clear GPU cache before starting
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Get sample dimensions from first sample
        sample_X = X[0:1]
        sample_mX = mX[0:1]
        sample_mY = mY[0:1]
        sample_avail = avail[0:1]
        sample_seq = seq[0:1] if seq is not None else None
        
        print(f"Finding optimal batch size for GPU {self.device}...")
        
        # First, do exponential search to find upper bound
        current_size = start_size
        best_size = start_size
        
        # Exponential search: double until we hit OOM
        while current_size <= max_size:
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Create batch of size current_size
                test_X = sample_X.repeat(current_size, 1, 1).to(self.device)
                test_mX = sample_mX.repeat(current_size, 1, 1).to(self.device)
                test_mY = sample_mY.repeat(current_size, 1, 1).to(self.device)
                test_avail = sample_avail.repeat(current_size, 1).to(self.device)
                
                if self.DNA and sample_seq is not None:
                    test_seq = sample_seq.repeat(current_size, 1, 1).to(self.device)
                    # Test forward pass
                    with torch.no_grad():
                        _ = self.model(
                            test_X.float(), test_seq, test_mX.float(), test_mY
                        )
                else:
                    # Test forward pass
                    with torch.no_grad():
                        _ = self.model(
                            test_X.float(), test_mX.float(), test_mY, test_avail
                        )
                
                # If successful, record and try larger
                best_size = current_size
                current_size *= 2
                
                # Clean up
                del test_X, test_mX, test_mY, test_avail
                if self.DNA and sample_seq is not None:
                    del test_seq
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                error_str = str(e).lower()
                if "out of memory" in error_str or "cublas" in error_str or "alloc" in error_str:
                    # Hit memory limit
                    if current_size == start_size:
                        # Even batch size 1 failed, return 1 as fallback
                        print(f"Warning: Even batch size {start_size} failed. Using {start_size} as fallback.")
                        torch.cuda.empty_cache()
                        return start_size
                    # Now binary search between best_size and current_size
                    break
                else:
                    # Other error, re-raise
                    raise
        
        # Binary search between best_size and current_size (or max_size if we didn't hit OOM)
        if current_size > max_size:
            current_size = max_size
        
        low, high = best_size, current_size
        
        while low < high:
            mid = (low + high + 1) // 2  # Round up to avoid infinite loop
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Create batch of size mid
                test_X = sample_X.repeat(mid, 1, 1).to(self.device)
                test_mX = sample_mX.repeat(mid, 1, 1).to(self.device)
                test_mY = sample_mY.repeat(mid, 1, 1).to(self.device)
                test_avail = sample_avail.repeat(mid, 1).to(self.device)
                
                if self.DNA and sample_seq is not None:
                    test_seq = sample_seq.repeat(mid, 1, 1).to(self.device)
                    # Test forward pass
                    with torch.no_grad():
                        _ = self.model(
                            test_X.float(), test_seq, test_mX.float(), test_mY
                        )
                else:
                    # Test forward pass
                    with torch.no_grad():
                        _ = self.model(
                            test_X.float(), test_mX.float(), test_mY, test_avail
                        )
                
                # If successful, try larger batch size
                best_size = mid
                low = mid
                
                # Clean up
                del test_X, test_mX, test_mY, test_avail
                if self.DNA and sample_seq is not None:
                    del test_seq
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                error_str = str(e).lower()
                if "out of memory" in error_str or "cublas" in error_str or "alloc" in error_str:
                    # Batch size too large, try smaller
                    high = mid - 1
                    torch.cuda.empty_cache()
                else:
                    # Other error, re-raise
                    raise
        
        # Final cleanup
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        print(f"Found optimal batch size: {best_size}")
        return best_size
    
    def load_data(self, bios_name: str, locus: List, dsf: int = 1, 
                  fill_y_prompt_spec: Optional[Dict] = None,
                  fill_prompt_mode: str = "median") -> Tuple:
        """
        Load data for a specific biosample and genomic locus.
        
        Args:
            bios_name: Name of the biosample
            locus: Genomic locus as [chrom, start, end]
            dsf: Downsampling factor
            fill_y_prompt_spec: Optional dictionary specifying custom metadata values
            fill_prompt_mode: Mode for filling missing metadata. Options:
                - "none": Don't fill missing metadata (leave as -1)
                - "median": Use median for numeric fields (depth, read_length), mode for categorical (platform, run_type) (default)
                - "mode": Use mode for all fields
                - "sample": Use random sampling from dataset distribution
                - "custom": Use custom metadata from fill_y_prompt_spec
            
        Returns:
            Tuple of (X, Y, P, seq, mX, mY, avX, avY) for DNA models
            Tuple of (X, Y, P, mX, mY, avX, avY) for non-DNA models
        """
        if self.data_handler is None:
            raise RuntimeError("Data handler not setup. Call setup_data_handler() first.")
        
        print(f"Loading data for {bios_name} at {locus}")
        
        # Load count data
        temp_x, temp_mx = self.data_handler.load_bios_Counts(bios_name, locus, dsf)
        X, mX, avX = self.data_handler.make_bios_tensor_Counts(temp_x, temp_mx)
        del temp_x, temp_mx
        
        # Load target data
        temp_y, temp_my = self.data_handler.load_bios_Counts(bios_name, locus, 1)
        Y, mY, avY = self.data_handler.make_bios_tensor_Counts(temp_y, temp_my)
        del temp_y, temp_my
        
        # Fill in Y prompt metadata based on mode
        if fill_prompt_mode == "none":
            # Don't fill - leave missing values as -1
            print("Fill-in-prompt disabled: leaving missing metadata as -1")
        elif fill_prompt_mode == "custom" and fill_y_prompt_spec is not None:
            # Use custom metadata specification
            mY = self.data_handler.fill_in_prompt_manual(mY, fill_y_prompt_spec, overwrite=True)
            print("Using custom metadata specification for fill-in-prompt")
        elif fill_prompt_mode == "sample":
            # Use random sampling (sample=True)
            mY = self.data_handler.fill_in_prompt(mY, missing_value=-1, sample=True)
            print("Using random sampling for fill-in-prompt")
        elif fill_prompt_mode == "mode":
            # Use mode for all fields
            mY = self.data_handler.fill_in_prompt(mY, missing_value=-1, sample=False, use_mode=True)
            print("Using mode statistics for fill-in-prompt")
        else:
            # Default: Use median for numeric fields, mode for categorical (sample=False, use_mode=False)
            mY = self.data_handler.fill_in_prompt(mY, missing_value=-1, sample=False, use_mode=False)
            print("Using median/mode statistics for fill-in-prompt (median for numeric, mode for categorical)")
        
        # Load p-value data
        temp_p = self.data_handler.load_bios_BW(bios_name, locus)
        P, avlP = self.data_handler.make_bios_tensor_BW(temp_p)
        del temp_p
        
        # Verify availability consistency
        assert (avlP == avY).all(), "Availability masks for P and Y do not match"
        
        # Load control data
        try:
            temp_control_data, temp_control_metadata = self.data_handler.load_bios_Control(bios_name, locus, dsf)
            control_data, control_meta, control_avail = self.data_handler.make_bios_tensor_Control(temp_control_data, temp_control_metadata)
            del temp_control_data, temp_control_metadata
        except Exception as e:
            print(f"Warning: Failed to load control data for {bios_name}: {e}")
            print("Using missing values for control data")
            # Create control data with missing values
            L = X.shape[0]
            control_data = torch.full((L, 1), -1.0)  # missing_value
            control_meta = torch.full((4, 1), -1.0)  # missing_value
            control_avail = torch.zeros(1)  # not available
        
        # Concatenate control data to input data (same as in training)
        X = torch.cat([X, control_data], dim=1)      # (L, F+1)
        mX = torch.cat([mX, control_meta], dim=1)    # (4, F+1)
        avX = torch.cat([avX, control_avail], dim=0) # (F+1,)
        
        # Load DNA sequence if needed
        seq = None
        if self.DNA:
            seq = self.data_handler._dna_to_onehot(
                self.data_handler._get_DNA_sequence(locus[0], locus[1], locus[2])
            )
        
        # Reshape data to context windows
        context_length = self.context_length
        num_rows = (X.shape[0] // self.context_length) * self.context_length
        X, Y, P = X[:num_rows, :], Y[:num_rows, :], P[:num_rows, :]
        
        if self.DNA:
            seq = seq[:num_rows * self.data_handler.resolution, :]
        
        # Reshape to context windows
        X = X.view(-1, self.context_length, X.shape[-1])
        Y = Y.view(-1, self.context_length, Y.shape[-1])
        P = P.view(-1, self.context_length, P.shape[-1])
        
        if self.DNA:
            seq = seq.view(-1, self.context_length * self.data_handler.resolution, seq.shape[-1])
        
        # Expand metadata and availability to match batch dimension
        mX, mY = mX.expand(X.shape[0], -1, -1), mY.expand(Y.shape[0], -1, -1)
        avX, avY = avX.expand(X.shape[0], -1), avY.expand(Y.shape[0], -1)
        
        if self.DNA:
            return X, Y, P, seq, mX, mY, avX, avY
        else:
            return X, Y, P, mX, mY, avX, avY
    
    def predict(self, X: torch.Tensor, mX: torch.Tensor, mY: torch.Tensor, 
                avail: torch.Tensor, seq: Optional[torch.Tensor] = None,
                imp_target: List[int] = []) -> Tuple[torch.Tensor, ...]:
        """
        Run inference on input data.
        
        Args:
            X: Input count data [B, L, F]
            mX: Input metadata [B, 4, F]
            mY: Target metadata [B, 4, F]
            avail: Availability mask [B, F]
            seq: DNA sequence [B, L*25, 4] (required if DNA=True)
            imp_target: List of feature indices to treat as imputation targets
            
        Returns:
            Tuple of (output_n, output_p, output_mu, output_var, output_peak)
        """
        # Set model to training mode to use batch statistics in BatchNorm
        # (avoids corrupted running statistics while keeping no_grad for efficiency)
        self.model.train()
        
        # Auto-detect batch size if not set or explicitly set to None
        batch_size = None # self.config.get('batch_size', None)

        if batch_size is None:
            # Check if we've already computed max batch size
            if self._max_batch_size is None:
                # Use first sample to determine optimal batch size
                sample_X = X[:1]
                sample_mX = mX[:1]
                sample_mY = mY[:1]
                sample_avail = avail[:1]
                sample_seq = seq[:1] if seq is not None else None
                
                self._max_batch_size = self._find_max_batch_size(
                    sample_X, sample_mX, sample_mY, sample_avail, sample_seq
                )
            batch_size = self._max_batch_size
            # print(f"Using auto-detected batch size: {batch_size}")
        else:
            # Use configured batch size
            batch_size = int(batch_size)
        
        # Initialize output tensors - model outputs only for original features (without control)
        # Control is only used as input, not predicted as output
        original_feature_dim = X.shape[-1] - 1  # Subtract 1 for control
        n = torch.empty(X.shape[0], X.shape[1], original_feature_dim, device="cpu", dtype=torch.float32)
        p = torch.empty(X.shape[0], X.shape[1], original_feature_dim, device="cpu", dtype=torch.float32)
        mu = torch.empty(X.shape[0], X.shape[1], original_feature_dim, device="cpu", dtype=torch.float32)
        var = torch.empty(X.shape[0], X.shape[1], original_feature_dim, device="cpu", dtype=torch.float32)
        peak = torch.empty(X.shape[0], X.shape[1], original_feature_dim, device="cpu", dtype=torch.float32)
        
        # Process in batches
        for i in range(0, len(X), batch_size):
            # Get batch
            x_batch = X[i:i + batch_size]
            mX_batch = mX[i:i + batch_size]
            mY_batch = mY[i:i + batch_size]
            avail_batch = avail[i:i + batch_size]
            
            if self.DNA:
                seq_batch = seq[i:i + batch_size]
            
            with torch.no_grad():
                # Clone and prepare batch
                x_batch = x_batch.clone()
                mX_batch = mX_batch.clone()
                mY_batch = mY_batch.clone()
                avail_batch = avail_batch.clone()
                
                # Apply masking - use float tokens to match model expectations
                x_batch_missing = (x_batch == self.token_dict["missing_mask"])
                mX_batch_missing = (mX_batch == self.token_dict["missing_mask"])
                avail_batch_missing = (avail_batch == 0)
                
                x_batch[x_batch_missing] = float(self.token_dict["cloze_mask"])
                mX_batch[mX_batch_missing] = float(self.token_dict["cloze_mask"])
                
                # Apply imputation targets
                if len(imp_target) > 0:
                    x_batch[:, :, imp_target] = float(self.token_dict["cloze_mask"])
                    mX_batch[:, :, imp_target] = float(self.token_dict["cloze_mask"])
                    avail_batch[:, imp_target] = 0
                
                # Move to device first
                x_batch = x_batch.to(self.device)
                mX_batch = mX_batch.to(self.device)
                mY_batch = mY_batch.to(self.device)
                avail_batch = avail_batch.to(self.device)
                
                if self.DNA:
                    seq_batch = seq_batch.to(self.device)
                    # Run model forward pass - convert to float for model input
                    outputs_p, outputs_n, outputs_mu, outputs_var, outputs_peak = self.model(
                        x_batch.float(), seq_batch, mX_batch.float(), mY_batch
                    )
                else:
                    outputs_p, outputs_n, outputs_mu, outputs_var, outputs_peak = self.model(
                        x_batch.float(), mX_batch.float(), mY_batch, avail_batch
                    )
            
            # Store predictions
            batch_end = min(i + batch_size, len(X))
            n[i:batch_end] = outputs_n.cpu()
            p[i:batch_end] = outputs_p.cpu()
            mu[i:batch_end] = outputs_mu.cpu()
            var[i:batch_end] = outputs_var.cpu()
            peak[i:batch_end] = outputs_peak.cpu()
            
            # Clean up
            del x_batch, mX_batch, mY_batch, avail_batch, outputs_p, outputs_n, outputs_mu, outputs_var, outputs_peak
            if self.DNA:
                del seq_batch
        
        return n, p, mu, var, peak
    
    def get_latent_z(self, X: torch.Tensor, mX: torch.Tensor, mY: torch.Tensor,
                     avail: torch.Tensor, seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract latent representations from the model encoder.
        
        Args:
            X: Input count data [B, L, F]
            mX: Input metadata [B, 4, F]
            mY: Target metadata [B, 4, F]
            avail: Availability mask [B, F]
            seq: DNA sequence [B, L*25, 4] (required if DNA=True)
            
        Returns:
            Latent representations Z [B, L, D]
        """
        # Set model to training mode to use batch statistics in BatchNorm
        # (avoids corrupted running statistics while keeping no_grad for efficiency)
        self.model.train()
        
        # Auto-detect batch size if not set or explicitly set to None
        batch_size = self.config.get('batch_size', None)
        if batch_size is None:
            # Check if we've already computed max batch size
            if self._max_batch_size is None:
                # Use first sample to determine optimal batch size
                sample_X = X[:1]
                sample_mX = mX[:1]
                sample_mY = mY[:1]
                sample_avail = avail[:1]
                sample_seq = seq[:1] if seq is not None else None
                
                self._max_batch_size = self._find_max_batch_size(
                    sample_X, sample_mX, sample_mY, sample_avail, sample_seq
                )
            batch_size = self._max_batch_size
        else:
            # Use configured batch size
            batch_size = int(batch_size)
        Z_all = []
        
        for i in range(0, len(X), batch_size):
            # Get batch
            x_batch = X[i:i + batch_size]
            mX_batch = mX[i:i + batch_size]
            mY_batch = mY[i:i + batch_size]
            avail_batch = avail[i:i + batch_size]
            
            if self.DNA:
                seq_batch = seq[i:i + batch_size]
            
            with torch.no_grad():
                # Clone and prepare batch
                x_batch = x_batch.clone()
                mX_batch = mX_batch.clone()
                mY_batch = mY_batch.clone()
                avail_batch = avail_batch.clone()
                
                # Apply masking
                x_batch_missing = (x_batch == self.token_dict["missing_mask"])
                mX_batch_missing = (mX_batch == self.token_dict["missing_mask"])
                
                x_batch[x_batch_missing] = self.token_dict["cloze_mask"]
                mX_batch[mX_batch_missing] = self.token_dict["cloze_mask"]
                
                # Move to device
                x_batch = x_batch.to(self.device)
                mX_batch = mX_batch.to(self.device)
                mY_batch = mY_batch.to(self.device)
                avail_batch = avail_batch.to(self.device)
                
                if self.DNA:
                    seq_batch = seq_batch.to(self.device)
                    # Get latent representation
                    Z = self.model.encode(x_batch.float(), seq_batch, mX_batch)
                else:
                    Z = self.model.encode(x_batch.float(), mX_batch)
            
            Z_all.append(Z.cpu())
            
            # Clean up
            del x_batch, mX_batch, mY_batch, avail_batch, Z
            if self.DNA:
                del seq_batch
        
        return torch.cat(Z_all, dim=0)
    
    def predict_biosample(self, bios_name: str, x_dsf: int = 1, 
                         fill_y_prompt_spec: Optional[Dict] = None,
                         fill_prompt_mode: str = "median",
                         locus: Optional[List] = None,
                         get_latent_z: bool = False,
                         return_raw_predictions: bool = False) -> Dict[str, Any]:
        """
        High-level method to predict for an entire biosample.
        
        Args:
            bios_name: Name of the biosample
            x_dsf: Downsampling factor
            fill_y_prompt_spec: Optional custom metadata specification
            fill_prompt_mode: Mode for filling missing metadata ("none", "median", "sample", "custom")
            locus: Genomic locus (default: chr21)
            get_latent_z: Whether to extract latent representations
            return_raw_predictions: Whether to return raw prediction tensors
            
        Returns:
            Dictionary with organized predictions by biosample and experiment
        """
        if locus is None:
            # Default to chr21
            locus = ["chr21", 0, self.data_handler.chr_sizes["chr21"]]
        
        # Load data
        if self.DNA:
            X, Y, P, seq, mX, mY, avX, avY = self.load_data(
                bios_name, locus, x_dsf, fill_y_prompt_spec, fill_prompt_mode
            )
        else:
            X, Y, P, mX, mY, avX, avY = self.load_data(
                bios_name, locus, x_dsf, fill_y_prompt_spec, fill_prompt_mode
            )
            seq = None
        
        print(f"Loaded data: {X.shape}, {Y.shape}, {P.shape}")
        
        # Get available experiments
        available_indices = torch.where(avX[0, :] == 1)[0].tolist()
        if 35 in available_indices: #remove control
            available_indices.remove(35)


        experiment_names = list(self.data_handler.aliases['experiment_aliases'].keys())
        
        # Initialize results structure
        results = {
            bios_name: {}
        }
        
        # Run leave-one-out predictions for imputation
        print(f"Running leave-one-out predictions for {len(available_indices)} experiments...")
        
        for leave_one_out in available_indices:
            exp_name = experiment_names[leave_one_out]
            print(f"  Predicting {exp_name} (index {leave_one_out})")
            
            # Run prediction
            if self.DNA:
                n, p, mu, var, peak = self.predict(X, mX, mY, avX, seq, [leave_one_out])
            else:
                n, p, mu, var, peak = self.predict(X, mX, mY, avX, None, [leave_one_out])
            
            p = p.view((p.shape[0] * p.shape[1]), p.shape[-1])
            n = n.view((n.shape[0] * n.shape[1]), n.shape[-1])
            mu = mu.view((mu.shape[0] * mu.shape[1]), mu.shape[-1])
            var = var.view((var.shape[0] * var.shape[1]), var.shape[-1])
            peak = peak.view((peak.shape[0] * peak.shape[1]), peak.shape[-1])
            
            # Create distributions
            count_dist = NegativeBinomial(p[:, leave_one_out], n[:, leave_one_out])
            pval_dist = Gaussian(mu[:, leave_one_out], var[:, leave_one_out])
            
            # Store results
            results[bios_name][exp_name] = {
                'type': 'imputed',
                'experiment_name': exp_name,
                'count_dist': count_dist,
                'count_params': {'p': p[:, leave_one_out], 'n': n[:, leave_one_out]},
                'pval_dist': pval_dist,
                'pval_params': {'mu': mu[:, leave_one_out], 'var': var[:, leave_one_out]},
                'peak_scores': peak[:, leave_one_out]
            }
            
            # Add raw predictions if requested
            if return_raw_predictions:
                results[bios_name][exp_name]['raw_predictions'] = {
                    'output_p': p[:, leave_one_out],
                    'output_n': n[:, leave_one_out],
                    'output_mu': mu[:, leave_one_out],
                    'output_var': var[:, leave_one_out],
                    'output_peak': peak[:, leave_one_out]
                }
        
        # Run upsampling predictions (predict all available experiments)
        print("Running upsampling predictions...")
        
        if self.DNA:
            n_ups, p_ups, mu_ups, var_ups, peak_ups = self.predict(X, mX, mY, avX, seq, [])
        else:
            n_ups, p_ups, mu_ups, var_ups, peak_ups = self.predict(X, mX, mY, avX, None, [])

        p_ups = p_ups.view((p_ups.shape[0] * p_ups.shape[1]), p_ups.shape[-1])
        n_ups = n_ups.view((n_ups.shape[0] * n_ups.shape[1]), n_ups.shape[-1])
        mu_ups = mu_ups.view((mu_ups.shape[0] * mu_ups.shape[1]), mu_ups.shape[-1])
        var_ups = var_ups.view((var_ups.shape[0] * var_ups.shape[1]), var_ups.shape[-1])
        peak_ups = peak_ups.view((peak_ups.shape[0] * peak_ups.shape[1]), peak_ups.shape[-1])
        
        # Store upsampling results for available experiments (denoised)
        for exp_idx in available_indices:
            exp_name = experiment_names[exp_idx]
            
            # Create distributions
            count_dist_ups = NegativeBinomial(p_ups[:, exp_idx], n_ups[:, exp_idx])
            pval_dist_ups = Gaussian(mu_ups[:, exp_idx], var_ups[:, exp_idx])
            
            # Add upsampling results
            results[bios_name][f"{exp_name}_upsampled"] = {
                'type': 'denoised',
                'experiment_name': exp_name,
                'count_dist': count_dist_ups,
                'count_params': {'p': p_ups[:, exp_idx], 'n': n_ups[:, exp_idx]},
                'pval_dist': pval_dist_ups,
                'pval_params': {'mu': mu_ups[:, exp_idx], 'var': var_ups[:, exp_idx]},
                'peak_scores': peak_ups[:, exp_idx]
            }
            
            # Add raw predictions if requested
            if return_raw_predictions:
                results[bios_name][f"{exp_name}_upsampled"]['raw_predictions'] = {
                    'output_p': p_ups[:, exp_idx],
                    'output_n': n_ups[:, exp_idx],
                    'output_mu': mu_ups[:, exp_idx],
                    'output_var': var_ups[:, exp_idx],
                    'output_peak': peak_ups[:, exp_idx]
                }
        
        # Optionally store predictions for non-available experiments (imputed from upsampling pass)
        all_experiment_indices = list(range(len(experiment_names)))
        non_available_indices = [idx for idx in all_experiment_indices if idx not in available_indices]
        
        if non_available_indices:
            print(f"Storing predictions for {len(non_available_indices)} non-available experiments...")
            for exp_idx in non_available_indices:
                exp_name = experiment_names[exp_idx]
                
                # Create distributions
                count_dist_imp = NegativeBinomial(p_ups[:, exp_idx], n_ups[:, exp_idx])
                pval_dist_imp = Gaussian(mu_ups[:, exp_idx], var_ups[:, exp_idx])
                
                # Add imputation results (from upsampling pass)
                results[bios_name][f"{exp_name}_imputed_from_upsampling"] = {
                    'type': 'imputed',
                    'experiment_name': exp_name,
                    'count_dist': count_dist_imp,
                    'count_params': {'p': p_ups[:, exp_idx], 'n': n_ups[:, exp_idx]},
                    'pval_dist': pval_dist_imp,
                    'pval_params': {'mu': mu_ups[:, exp_idx], 'var': var_ups[:, exp_idx]},
                    'peak_scores': peak_ups[:, exp_idx]
                }
                
                # Add raw predictions if requested
                if return_raw_predictions:
                    results[bios_name][f"{exp_name}_imputed_from_upsampling"]['raw_predictions'] = {
                        'output_p': p_ups[:, exp_idx],
                        'output_n': n_ups[:, exp_idx],
                        'output_mu': mu_ups[:, exp_idx],
                        'output_var': var_ups[:, exp_idx],
                        'output_peak': peak_ups[:, exp_idx]
                    }
        
        # Extract latent representations if requested
        if get_latent_z:
            print("Extracting latent representations...")
            if self.DNA:
                Z = self.get_latent_z(X, mX, mY, avX, seq)
            else:
                Z = self.get_latent_z(X, mX, mY, avX, None)
            
            results['latent_z'] = Z
        
        return results
    
    def predict_all_biosamples(self, dataset_type: str, split: str = "test",
                              x_dsf: int = 1, fill_y_prompt_spec: Optional[Dict] = None,
                              fill_prompt_mode: str = "median",
                              locus: Optional[List] = None):
        """
        Run predictions for all biosamples in the dataset.
        
        Args:
            dataset_type: "merged" or "eic"
            split: Data split to use (train/val/test)
            x_dsf: Downsampling factor
            fill_y_prompt_spec: Optional custom metadata specification
            fill_prompt_mode: Mode for filling missing metadata
            locus: Genomic locus (default: chr21)
        """
        if locus is None:
            locus = ["chr21", 0, self.data_handler.chr_sizes["chr21"]]
        
        # Get list of biosamples filtered by split
        biosample_names = []
        for bios in list(self.data_handler.navigation.keys()):
            # Skip biosamples not in split_dict
            if bios not in self.data_handler.split_dict:
                print(f"Warning: Skipping {bios} - not found in split_dict")
                continue
            
            if self.data_handler.split_dict[bios] == split:
                # For EIC dataset, only process B_ or V_ biosamples (skip T_)
                if dataset_type == "eic":
                    if split == "test" and bios.startswith("B_"):
                        biosample_names.append(bios)
                    elif split == "val" and bios.startswith("V_"):
                        biosample_names.append(bios)
                    elif not (bios.startswith("T_") or bios.startswith("B_") or bios.startswith("V_")):
                        # Handle non-prefixed biosamples (assume B_)
                        biosample_names.append(bios)
                else:
                    biosample_names.append(bios)
        
        print(f"Found {len(biosample_names)} biosamples in {split} split")
        
        for bios_name in biosample_names:
            print(f"\n{'='*60}")
            print(f"Processing biosample: {bios_name}")
            print(f"{'='*60}")
            
            try:
                if dataset_type == "merged":
                    # Use predict_biosample for merged dataset
                    prediction_dict = self.predict_biosample(
                        bios_name=bios_name,
                        x_dsf=x_dsf,
                        fill_y_prompt_spec=fill_y_prompt_spec,
                        fill_prompt_mode=fill_prompt_mode,
                        locus=locus,
                        get_latent_z=True,
                        return_raw_predictions=False
                    )
                    
                    # Extract latent Z
                    Z = prediction_dict.get('latent_z', None)
                    if Z is None:
                        # Need to extract Z separately
                        if self.DNA:
                            X, Y, P, seq, mX, mY, avX, avY = self.load_data(
                                bios_name, locus, x_dsf, fill_y_prompt_spec, fill_prompt_mode
                            )
                            Z = self.get_latent_z(X, mX, mY, avX, seq)
                        else:
                            X, Y, P, mX, mY, avX, avY = self.load_data(
                                bios_name, locus, x_dsf, fill_y_prompt_spec, fill_prompt_mode
                            )
                            Z = self.get_latent_z(X, mX, mY, avX, None)
                    
                    experiment_names = list(self.data_handler.aliases['experiment_aliases'].keys())
                    
                    # Load observed P and Peak data once for this biosample
                    try:
                        temp_p = self.data_handler.load_bios_BW(bios_name, locus)
                        P_obs_all, _ = self.data_handler.make_bios_tensor_BW(temp_p)
                        # Reshape to match prediction length
                        num_rows = (P_obs_all.shape[0] // self.context_length) * self.context_length
                        P_obs_all = P_obs_all[:num_rows, :]
                        P_obs_all = P_obs_all.view(-1, self.context_length, P_obs_all.shape[-1])
                        P_obs_all = P_obs_all.view(-1, P_obs_all.shape[-1])  # [B*L, F]
                        
                        temp_peak = self.data_handler.load_bios_Peaks(bios_name, locus)
                        Peak_obs_all, _ = self.data_handler.make_bios_tensor_Peaks(temp_peak)
                        num_rows = (Peak_obs_all.shape[0] // self.context_length) * self.context_length
                        Peak_obs_all = Peak_obs_all[:num_rows, :]
                        Peak_obs_all = Peak_obs_all.view(-1, self.context_length, Peak_obs_all.shape[-1])
                        Peak_obs_all = Peak_obs_all.view(-1, Peak_obs_all.shape[-1])  # [B*L, F]
                    except Exception as e:
                        print(f"Warning: Failed to load observed data for {bios_name}: {e}")
                        P_obs_all = None
                        Peak_obs_all = None
                    
                    # Save predictions
                    self.save_predictions_to_npz(
                        prediction_dict=prediction_dict,
                        bios_name=bios_name,
                        dataset_type="merged",
                        experiment_names=experiment_names,
                        Z=Z,
                        locus=locus,
                        P_obs_all=P_obs_all,
                        Peak_obs_all=Peak_obs_all
                    )
                    
                elif dataset_type == "eic":
                    # Follow bios_pipeline_eic logic exactly
                    # Determine T_ and B_ biosample names
                    if split == "test":
                        if bios_name.startswith("B_"):
                            T_biosname = bios_name.replace("B_", "T_")
                            B_biosname = bios_name
                        elif bios_name.startswith("T_"):
                            T_biosname = bios_name
                            B_biosname = bios_name.replace("T_", "B_")
                        else:
                            print(f"Warning: Unexpected biosample name format: {bios_name}. Skipping.")
                            continue
                    elif split == "val":
                        if bios_name.startswith("V_"):
                            T_biosname = bios_name.replace("V_", "T_")
                            B_biosname = bios_name
                        elif bios_name.startswith("T_"):
                            T_biosname = bios_name
                            B_biosname = bios_name.replace("T_", "V_")
                        else:
                            print(f"Warning: Unexpected biosample name format: {bios_name}. Skipping.")
                            continue
                    else:
                        # Default: assume B_ prefix
                        if bios_name.startswith("B_"):
                            T_biosname = bios_name.replace("B_", "T_")
                            B_biosname = bios_name
                        elif bios_name.startswith("T_"):
                            T_biosname = bios_name
                            B_biosname = bios_name.replace("T_", "B_")
                        else:
                            print(f"Warning: Unexpected biosample name format: {bios_name}. Skipping.")
                            continue
                    
                    print(f"Loading T_ data from {T_biosname} and B_ data from {B_biosname}")
                    
                    # Check if T_ biosample exists
                    if T_biosname not in self.data_handler.navigation:
                        print(f"Warning: T_ biosample {T_biosname} not found. Skipping {bios_name}")
                        continue
                    
                    # Load T_ data (input side)
                    try:
                        temp_x, temp_mx = self.data_handler.load_bios_Counts(T_biosname, locus, DSF=x_dsf)
                        X, mX, avX = self.data_handler.make_bios_tensor_Counts(temp_x, temp_mx)
                        del temp_x, temp_mx
                        available_X_indices = torch.where(avX[0, :] == 1)[0] if avX.ndim > 1 else torch.where(avX == 1)[0]
                    except Exception as e:
                        print(f"Warning: Failed to load T_ data for {T_biosname}: {e}. Skipping {bios_name}")
                        continue
                    
                    # Load B_ data (target side)
                    try:
                        temp_y, temp_my = self.data_handler.load_bios_Counts(B_biosname, locus, DSF=1)
                        Y, mY, avY = self.data_handler.make_bios_tensor_Counts(temp_y, temp_my)
                        # Apply fill-in-prompt based on mode
                        if fill_prompt_mode == "none":
                            pass
                        elif fill_prompt_mode == "custom" and fill_y_prompt_spec is not None:
                            mY = self.data_handler.fill_in_prompt_manual(mY, fill_y_prompt_spec, overwrite=True)
                        elif fill_prompt_mode == "sample":
                            mY = self.data_handler.fill_in_prompt(mY, missing_value=-1, sample=True)
                        elif fill_prompt_mode == "mode":
                            mY = self.data_handler.fill_in_prompt(mY, missing_value=-1, sample=False, use_mode=True)
                        else:
                            mY = self.data_handler.fill_in_prompt(mY, missing_value=-1, sample=False, use_mode=False)
                        del temp_y, temp_my
                        available_Y_indices = torch.where(avY[0, :] == 1)[0] if avY.ndim > 1 else torch.where(avY == 1)[0]
                    except Exception as e:
                        print(f"Warning: Failed to load B_ data for {B_biosname}: {e}. Skipping {bios_name}")
                        continue
                    
                    # Load and merge P-value data
                    try:
                        temp_py = self.data_handler.load_bios_BW(B_biosname, locus)
                        temp_px = self.data_handler.load_bios_BW(T_biosname, locus)
                        temp_p = {**temp_py, **temp_px}
                        P, avlP = self.data_handler.make_bios_tensor_BW(temp_p)
                        del temp_py, temp_px, temp_p
                    except Exception as e:
                        print(f"Warning: Failed to load P-value data: {e}. Skipping {bios_name}")
                        continue
                    
                    # Load and merge Peak data
                    try:
                        temp_peak_t = self.data_handler.load_bios_Peaks(T_biosname, locus)
                        temp_peak_b = self.data_handler.load_bios_Peaks(B_biosname, locus)
                        temp_peak = {**temp_peak_b, **temp_peak_t}
                        Peak, avlPeak = self.data_handler.make_bios_tensor_Peaks(temp_peak)
                        del temp_peak_t, temp_peak_b, temp_peak
                    except Exception as e:
                        print(f"Warning: Failed to load Peak data: {e}. Skipping {bios_name}")
                        continue
                    
                    # Load control data
                    control_data = None
                    control_meta = None
                    control_avail = None
                    try:
                        temp_control_data, temp_control_metadata = self.data_handler.load_bios_Control(T_biosname, locus, DSF=x_dsf)
                        if temp_control_data and "chipseq-control" in temp_control_data:
                            control_data, control_meta, control_avail = self.data_handler.make_bios_tensor_Control(temp_control_data, temp_control_metadata)
                        else:
                            temp_control_data, temp_control_metadata = self.data_handler.load_bios_Control(B_biosname, locus, DSF=x_dsf)
                            if temp_control_data and "chipseq-control" in temp_control_data:
                                control_data, control_meta, control_avail = self.data_handler.make_bios_tensor_Control(temp_control_data, temp_control_metadata)
                    except Exception as e:
                        pass
                    
                    if control_data is None:
                        L = X.shape[0]
                        control_data = torch.full((L, 1), -1.0)
                        control_meta = torch.full((4, 1), -1.0)
                        control_avail = torch.zeros(1)
                    
                    # Concatenate control data
                    X = torch.cat([X, control_data], dim=1)
                    mX = torch.cat([mX, control_meta], dim=1)
                    avX = torch.cat([avX, control_avail], dim=0)
                    
                    # Prepare data for model
                    num_rows = (X.shape[0] // self.context_length) * self.context_length
                    X = X[:num_rows, :]
                    Y = Y[:num_rows, :]
                    P = P[:num_rows, :]
                    Peak = Peak[:num_rows, :]
                    
                    # Ensure mY matches Y's feature dimension
                    if mY.shape[-1] != Y.shape[-1]:
                        print(f"Warning: mY feature dim {mY.shape[-1]} != Y feature dim {Y.shape[-1]}. Trimming mY.")
                        mY = mY[:, :Y.shape[-1]]
                    
                    X = X.view(-1, self.context_length, X.shape[-1])
                    Y = Y.view(-1, self.context_length, Y.shape[-1])
                    P = P.view(-1, self.context_length, P.shape[-1])
                    Peak = Peak.view(-1, self.context_length, Peak.shape[-1])
                    
                    # Expand masks to match batch dimension
                    mX = mX.expand(X.shape[0], -1, -1)
                    mY = mY.expand(Y.shape[0], -1, -1)
                    avX = avX.expand(X.shape[0], -1)
                    
                    # Load DNA sequence if needed
                    seq = None
                    if self.DNA:
                        seq = self.data_handler._dna_to_onehot(
                            self.data_handler._get_DNA_sequence(locus[0], locus[1], locus[2])
                        )
                        seq = seq[:num_rows*self.data_handler.resolution, :]
                        seq = seq.view(-1, self.context_length*self.data_handler.resolution, seq.shape[-1])
                    
                    # Single forward pass
                    print(f"Running single forward pass for EIC...")
                    if self.DNA:
                        n_ups, p_ups, mu_ups, var_ups, peak_ups = self.predict(
                            X, mX, mY, avX, seq=seq, imp_target=[]
                        )
                    else:
                        n_ups, p_ups, mu_ups, var_ups, peak_ups = self.predict(
                            X, mX, mY, avX, seq=None, imp_target=[]
                        )
                    
                    # Flatten predictions
                    p_ups = p_ups.view((p_ups.shape[0] * p_ups.shape[1]), p_ups.shape[-1])
                    n_ups = n_ups.view((n_ups.shape[0] * n_ups.shape[1]), n_ups.shape[-1])
                    mu_ups = mu_ups.view((mu_ups.shape[0] * mu_ups.shape[1]), mu_ups.shape[-1])
                    var_ups = var_ups.view((var_ups.shape[0] * var_ups.shape[1]), var_ups.shape[-1])
                    peak_ups = peak_ups.view((peak_ups.shape[0] * peak_ups.shape[1]), peak_ups.shape[-1])
                    
                    # Flatten ground truth data
                    X_flat = X.view((X.shape[0] * X.shape[1]), X.shape[-1])
                    Y_flat = Y.view((Y.shape[0] * Y.shape[1]), Y.shape[-1])
                    P_flat = P.view((P.shape[0] * P.shape[1]), P.shape[-1])
                    Peak_flat = Peak.view((Peak.shape[0] * Peak.shape[1]), Peak.shape[-1])
                    
                    # Extract latent Z
                    if self.DNA:
                        Z = self.get_latent_z(X, mX, mY, avX, seq)
                    else:
                        Z = self.get_latent_z(X, mX, mY, avX, None)
                    
                    experiment_names = list(self.data_handler.aliases['experiment_aliases'].keys())
                    
                    # Save predictions (P_flat and Peak_flat are already the observed data)
                    self.save_predictions_to_npz(
                        prediction_dict=None,
                        bios_name=bios_name,
                        dataset_type="eic",
                        experiment_names=experiment_names,
                        Z=Z,
                        P_obs_all=P_flat,
                        Peak_obs_all=Peak_flat,
                        mu_ups=mu_ups,
                        var_ups=var_ups,
                        peak_ups=peak_ups,
                        available_X_indices=available_X_indices,
                        available_Y_indices=available_Y_indices,
                        P=P_flat,
                        Peak=Peak_flat,
                        X=X_flat,
                        Y=Y_flat,
                        locus=locus
                    )
                    
                print(f"✅ Completed predictions for {bios_name}")
                
            except Exception as e:
                print(f"❌ Error processing {bios_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    def save_predictions_to_npz(self, prediction_dict: Optional[Dict[str, Any]], 
                                bios_name: str, dataset_type: str,
                                experiment_names: List[str], Z: torch.Tensor,
                                locus: List,
                                P_obs_all: Optional[torch.Tensor] = None,
                                Peak_obs_all: Optional[torch.Tensor] = None,
                                mu_ups: Optional[torch.Tensor] = None,
                                var_ups: Optional[torch.Tensor] = None,
                                peak_ups: Optional[torch.Tensor] = None,
                                available_X_indices: Optional[torch.Tensor] = None,
                                available_Y_indices: Optional[torch.Tensor] = None,
                                P: Optional[torch.Tensor] = None,
                                Peak: Optional[torch.Tensor] = None,
                                X: Optional[torch.Tensor] = None,
                                Y: Optional[torch.Tensor] = None):
        """
        Save predictions to NPZ files in structured directory format.
        
        Args:
            prediction_dict: For merged: dict from predict_biosample(). For EIC: None
            bios_name: Name of biosample
            dataset_type: "merged" or "eic"
            experiment_names: List of experiment names
            Z: Latent representations (shared across assays)
            For EIC: mu_ups, var_ups, peak_ups, available_X_indices, available_Y_indices, P, Peak, X, Y
        """
        # Create directory structure: model_dir/preds/biosample/assay/
        preds_dir = self.model_dir / "preds" / bios_name
        preds_dir.mkdir(parents=True, exist_ok=True)
        
        # Flatten Z to 1D or 2D for saving
        Z_np = Z.numpy()
        if Z_np.ndim == 3:  # [B, L, D]
            Z_np = Z_np.reshape(-1, Z_np.shape[-1])  # [B*L, D]
        Z_flat = Z_np.flatten() if Z_np.ndim == 2 else Z_np
        
        if dataset_type == "merged":
            # Iterate through prediction_dict keys
            for exp_key, pred_data in prediction_dict[bios_name].items():
                # Determine assay name and type
                if exp_key.endswith("_upsampled"):
                    assay_name = exp_key.replace("_upsampled", "")
                    assay_dir = preds_dir / f"{assay_name}_denoised"
                    is_denoised = True
                elif exp_key.endswith("_imputed_from_upsampling"):
                    assay_name = exp_key.replace("_imputed_from_upsampling", "")
                    assay_dir = preds_dir / f"{assay_name}_imputed"
                    is_denoised = False
                else:
                    # Leave-one-out imputed
                    assay_name = exp_key
                    assay_dir = preds_dir / f"{assay_name}_imputed"
                    is_denoised = False
                
                # Create assay directory
                assay_dir.mkdir(exist_ok=True)
                
                # Extract predictions
                mu = pred_data['pval_params']['mu'].numpy()
                var = pred_data['pval_params']['var'].numpy()
                peak_scores = pred_data['peak_scores'].numpy()
                
                # Extract observed P and peak data from pre-loaded arrays
                try:
                    exp_idx = experiment_names.index(assay_name)
                    if P_obs_all is not None and exp_idx < P_obs_all.shape[1]:
                        P_obs_flat = P_obs_all[:, exp_idx].numpy()
                    else:
                        P_obs_flat = np.zeros_like(mu)
                    
                    if Peak_obs_all is not None and exp_idx < Peak_obs_all.shape[1]:
                        Peak_obs_flat = Peak_obs_all[:, exp_idx].numpy()
                    else:
                        Peak_obs_flat = np.zeros_like(peak_scores)
                except (ValueError, IndexError) as e:
                    print(f"Warning: Failed to extract observed data for {assay_name}: {e}")
                    P_obs_flat = np.zeros_like(mu)
                    Peak_obs_flat = np.zeros_like(peak_scores)
                
                # Save NPZ files (z is saved once at biosample level, not per assay)
                np.savez_compressed(assay_dir / "mu.npz", mu.flatten())
                np.savez_compressed(assay_dir / "var.npz", var.flatten())
                np.savez_compressed(assay_dir / "peak_scores.npz", peak_scores.flatten())
                np.savez_compressed(assay_dir / "observed_P.npz", P_obs_flat.flatten())
                np.savez_compressed(assay_dir / "observed_peak.npz", Peak_obs_flat.flatten())
                
                print(f"  Saved predictions for {assay_name} ({'denoised' if is_denoised else 'imputed'})")
        
        elif dataset_type == "eic":
            # Iterate through all experiment indices
            for exp_idx, exp_name in enumerate(experiment_names):
                # Determine if denoised or imputed
                is_denoised = exp_idx in list(available_X_indices)
                is_imputed = exp_idx in list(available_Y_indices)
                
                if not (is_denoised or is_imputed):
                    continue  # Skip experiments not in either
                
                # Create assay directory
                assay_dir = preds_dir / exp_name
                assay_dir.mkdir(exist_ok=True)
                
                # Extract predictions
                mu = mu_ups[:, exp_idx].numpy()
                var = var_ups[:, exp_idx].numpy()
                peak_scores = peak_ups[:, exp_idx].numpy()
                
                # Get observed data from pre-loaded P_obs_all and Peak_obs_all (for EIC, these are P and Peak)
                if P_obs_all is not None and exp_idx < P_obs_all.shape[1]:
                    P_obs_flat = P_obs_all[:, exp_idx].numpy()
                else:
                    P_obs_flat = np.zeros_like(mu)
                
                if Peak_obs_all is not None and exp_idx < Peak_obs_all.shape[1]:
                    Peak_obs_flat = Peak_obs_all[:, exp_idx].numpy()
                else:
                    Peak_obs_flat = np.zeros_like(peak_scores)
                
                # Save NPZ files (z is saved once at biosample level, not per assay)
                np.savez_compressed(assay_dir / "mu.npz", mu.flatten())
                np.savez_compressed(assay_dir / "var.npz", var.flatten())
                np.savez_compressed(assay_dir / "peak_scores.npz", peak_scores.flatten())
                np.savez_compressed(assay_dir / "observed_P.npz", P_obs_flat.flatten())
                np.savez_compressed(assay_dir / "observed_peak.npz", Peak_obs_flat.flatten())
                
                print(f"  Saved predictions for {exp_name} ({'denoised' if is_denoised else 'imputed'})")
        
        # Save Z once at biosample level (shared across all assays)
        np.savez_compressed(preds_dir / "z.npz", Z_flat)
        print(f"  Saved latent Z at biosample level")


def load_predictions(model_dir: str, biosample: Optional[str] = None) -> Dict[str, Any]:
    """
    Load predictions from NPZ files in directory structure.
    
    Args:
        model_dir: Path to model directory containing preds/ subdirectory
        biosample: Optional biosample name. If None, load all biosamples.
        
    Returns:
        Dictionary with structure:
        {
            biosample: {
                "assay_imputed" or "assay_denoised": {
                    "mu": np.array,
                    "var": np.array,
                    "z": np.array,
                    "peak_scores": np.array,
                    "observed_P": np.array,
                    "observed_peak": np.array,
                    "base_assay_name": str,
                    "type": str  # "imputed", "denoised", or "unknown"
                }
            }
        }
    """
    model_dir = Path(model_dir)
    preds_dir = model_dir / "preds"
    
    if not preds_dir.exists():
        raise FileNotFoundError(f"Predictions directory not found: {preds_dir}")
    
    results = {}
    
    # Get list of biosamples
    if biosample is None:
        biosamples = [d.name for d in preds_dir.iterdir() if d.is_dir()]
    else:
        biosamples = [biosample]
    
    for bios_name in biosamples:
        bios_dir = preds_dir / bios_name
        if not bios_dir.exists():
            continue
        
        results[bios_name] = {}
        
        # Load Z from biosample level (shared across all assays)
        z_path = bios_dir / "z.npz"
        z_data = None
        if z_path.exists():
            try:
                z_data = np.load(z_path)['arr_0']
            except Exception as e:
                print(f"Warning: Failed to load biosample-level Z for {bios_name}: {e}")
        
        # Get list of assay directories
        assay_dirs = [d.name for d in bios_dir.iterdir() if d.is_dir()]
        
        for assay_dir_name in assay_dirs:
            assay_dir = bios_dir / assay_dir_name
            
            # Determine base assay name and type
            if assay_dir_name.endswith("_denoised"):
                base_assay_name = assay_dir_name.replace("_denoised", "")
                pred_type = "denoised"
            elif assay_dir_name.endswith("_imputed"):
                base_assay_name = assay_dir_name.replace("_imputed", "")
                pred_type = "imputed"
            else:
                # Fallback: assume it's the base name without suffix (for backward compatibility)
                base_assay_name = assay_dir_name
                pred_type = "unknown"
            
            try:
                # Load NPZ files
                mu_data = np.load(assay_dir / "mu.npz")
                var_data = np.load(assay_dir / "var.npz")
                peak_data = np.load(assay_dir / "peak_scores.npz")
                obs_p_data = np.load(assay_dir / "observed_P.npz")
                obs_peak_data = np.load(assay_dir / "observed_peak.npz")
                
                # Try loading z from assay dir for backward compatibility
                assay_z_data = z_data
                assay_z_path = assay_dir / "z.npz"
                if assay_z_path.exists():
                    try:
                        assay_z_data = np.load(assay_z_path)['arr_0']
                    except:
                        pass  # Use biosample-level z
                
                # Extract arrays (NPZ files contain arrays with default key 'arr_0')
                # Store with original directory name as key to preserve type information
                results[bios_name][assay_dir_name] = {
                    "mu": mu_data['arr_0'],
                    "var": var_data['arr_0'],
                    "peak_scores": peak_data['arr_0'],
                    "observed_P": obs_p_data['arr_0'],
                    "observed_peak": obs_peak_data['arr_0'],
                    "z": assay_z_data,
                    "base_assay_name": base_assay_name,
                    "type": pred_type
                }
                
            except Exception as e:
                print(f"Warning: Failed to load predictions for {bios_name}/{assay_dir_name}: {e}")
                continue
        
        return results


def main():
    """CLI interface for CANDI prediction."""
    parser = argparse.ArgumentParser(
        description="CANDI Prediction Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic prediction on chr21
  python pred.py --model-dir models/20251015_013231_CANDI_merged_ccre_5000loci_oct15 \\
                 --data-path /path/to/DATA_CANDI_MERGED \\
                 --bios-name GM12878

  # Prediction with custom metadata
  python pred.py --model-dir models/20251015_013231_CANDI_merged_ccre_5000loci_oct15 \\
                 --data-path /path/to/DATA_CANDI_MERGED \\
                 --bios-name GM12878 \\
                 --y-prompt-spec y_prompt.json \\
                 --output predictions.pkl

  # Extract latent representations
  python pred.py --model-dir models/20251015_013231_CANDI_merged_ccre_5000loci_oct15 \\
                 --data-path /path/to/DATA_CANDI_MERGED \\
                 --bios-name GM12878 \\
                 --get-latent-z \\
                 --output results.pkl
        """
    )
    
    # Required arguments
    parser.add_argument('--model-dir', type=str, required=True,
                       help='Path to model directory containing config JSON and .pt checkpoint')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to dataset directory (e.g., /path/to/DATA_CANDI_MERGED or /path/to/DATA_CANDI_EIC)')
    
    # Optional arguments
    parser.add_argument('--bios-name', type=str, default=None,
                       help='Name of biosample to predict (required if --all-biosamples not set)')
    parser.add_argument('--all-biosamples', action='store_true',
                       help='Run predictions for all biosamples in the dataset')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                       help='Data split to use (default: test)')
    parser.add_argument('--dataset', type=str, default='merged', choices=['merged', 'eic'],
                       help='Dataset type (default: merged)')
    parser.add_argument('--dsf', type=int, default=1,
                       help='Downsampling factor (default: 1)')
    parser.add_argument('--locus', type=str, nargs=3, default=['chr21', '0', '46709983'],
                       help='Genomic locus as chrom start end (default: chr21 0 46709983)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (auto-detect if not specified)')
    
    # Metadata specification
    parser.add_argument('--y-prompt-spec', type=str, default=None,
                       help='JSON file with custom metadata specification')
    parser.add_argument('--fill-prompt-mode', type=str, default='median',
                       choices=['none', 'median', 'mode', 'sample', 'custom'],
                       help='Mode for filling missing metadata (default: median)')
    
    # Output options
    parser.add_argument('--output', type=str, default='predictions.pkl',
                       help='Output file path (default: predictions.pkl, only used for single biosample)')
    parser.add_argument('--get-latent-z', action='store_true',
                       help='Extract latent representations')
    parser.add_argument('--return-raw-predictions', action='store_true',
                       help='Include raw prediction tensors in output')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.all_biosamples and args.bios_name is None:
        parser.error("Either --bios-name or --all-biosamples must be provided")
    
    # Parse locus
    locus = [args.locus[0], int(args.locus[1]), int(args.locus[2])]
    
    # Load Y prompt specification if provided
    fill_y_prompt_spec = None
    if args.y_prompt_spec:
        with open(args.y_prompt_spec, 'r') as f:
            fill_y_prompt_spec = json.load(f)
        print(f"Loaded Y prompt specification from {args.y_prompt_spec}")
    
    try:
        # Initialize predictor
        predictor = CANDIPredictor(args.model_dir, args.device, DNA=True)
        
        # Setup data handler
        predictor.setup_data_handler(args.data_path, args.dataset, split=args.split)
        
        if args.all_biosamples:
            # Run predictions for all biosamples
            predictor.predict_all_biosamples(
                dataset_type=args.dataset,
                split=args.split,
                x_dsf=args.dsf,
                fill_y_prompt_spec=fill_y_prompt_spec,
                fill_prompt_mode=args.fill_prompt_mode,
                locus=locus
            )
            print(f"\n🎉 Completed predictions for all biosamples!")
            print(f"Predictions saved to {args.model_dir}/preds/")
        else:
            # Run single biosample prediction (existing behavior)
            results = predictor.predict_biosample(
                bios_name=args.bios_name,
                x_dsf=args.dsf,
                fill_y_prompt_spec=fill_y_prompt_spec,
                fill_prompt_mode=args.fill_prompt_mode,
                locus=locus,
                get_latent_z=args.get_latent_z,
                return_raw_predictions=args.return_raw_predictions
            )
            
            # Save results
            with open(args.output, 'wb') as f:
                pickle.dump(results, f)
            
            print(f"Predictions saved to {args.output}")
            
            # Print summary for single biosample
            bios_name = args.bios_name
            if bios_name in results:
                n_experiments = len([k for k in results[bios_name].keys() if not k.endswith('_upsampled')])
                n_upsampled = len([k for k in results[bios_name].keys() if k.endswith('_upsampled')])
                print(f"Results summary:")
                print(f"  Biosample: {bios_name}")
                print(f"  Imputed experiments: {n_experiments}")
                print(f"  Denoised experiments: {n_upsampled}")
                if 'latent_z' in results:
                    print(f"  Latent representations: {results['latent_z'].shape}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
