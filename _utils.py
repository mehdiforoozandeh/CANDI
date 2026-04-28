import os, pyBigWig, pybedtools, random, datetime, gzip, pickle, psutil, math
from torch.utils.data import Dataset
from typing import Iterable, Dict, Callable, Union, Tuple
from torch import Tensor
from torch.optim import Optimizer
from tabulate import tabulate
from colorama import Fore, Back, Style

from io import BytesIO
import pandas as pd
import numpy as np
import multiprocessing as mp
import torch
from scipy.stats import nbinom
import torch.distributions as dist
from torch.distributions import Distribution, Gamma, constraints
from torch.distributions import Poisson as PoissonTorch
from torch.distributions.utils import (
    broadcast_all,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)
import matplotlib.pyplot as plt
import torch
from scipy.stats import norm, laplace, t as scipy_t, gamma as scipy_gamma
import psutil

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from scipy.stats import pearsonr, spearmanr, poisson, rankdata
from sklearn.metrics import auc

PROC_GENE_BED_FPATH = "data/gene_bodies.bed"
PROC_PROM_BED_PATH = "data/tss.bed"

class DataMasker:
    """
    DataMasker supports three independent masking strategies that can be combined:
    
    1. FULL LOCI MASKING (p_full_loci): Mask the same loci chunks across ALL available assays.
       - Randomly selects chunks of genomic positions
       - Masks these positions across all assays that have data
       - Does NOT mask metadata
    
    2. FULL ASSAY MASKING (p_full_assay): Completely mask entire assays.
       - For each sample, randomly masks 1 to (num_available-1) assays
       - Masks both data AND metadata for selected assays
       - Ensures at least one assay remains available
    
    3. CHUNK MASKING (p_chunks): Mask different loci chunks independently per assay.
       - Each assay gets its own random set of masked loci positions
       - Does NOT mask metadata
    
    At least one strategy must be applied per batch. Strategies are applied in order:
    full_assay -> full_loci -> chunks
    
    Probabilities are mutable attributes for training-time scheduling.
    """
    
    def __init__(self, mask_value, chunk_size=40, mask_fraction=0.20, 
                 p_full_loci=0.0, p_full_assay=1.0, p_chunks=0.0):
        """
        Args:
            mask_value: Value to use for masking (typically -2 for cloze_mask)
            chunk_size: Size of chunks to mask for full_loci and chunks strategies (default: 40, ~1kb at 25bp)
            mask_fraction: Fraction of loci to mask for full_loci and chunks strategies (default: 0.20)
            p_full_loci: Probability of applying full loci masking (default: 0.0)
            p_full_assay: Probability of applying full assay masking (default: 1.0)
            p_chunks: Probability of applying chunk masking per assay (default: 0.0)
        """
        self.mask_value = mask_value
        self.chunk_size = chunk_size
        self.mask_fraction = mask_fraction
        
        # Mutable probabilities for each strategy (can be modified during training)
        self.p_full_loci = p_full_loci
        self.p_full_assay = p_full_assay
        self.p_chunks = p_chunks
    
    def _mask_full_loci(self, data, metadata, availability):
        """
        Mask the same loci chunks across ALL available assays.
        
        This masks randomly selected chunks of genomic positions across all assays that have data.
        Does NOT mask metadata.
        
        Args:
            data: [B, L, F] signal data tensor
            metadata: [B, 4, F] metadata tensor (not modified)
            availability: [B, F] availability tensor (checked for available assays)
        
        Returns:
            Modified data, unchanged metadata, unchanged availability
        """
        B, L, F = data.shape
        device = data.device
        
        for b in range(B):
            # Get available assays for this sample
            available_assays = torch.where(availability[b] == 1)[0]
            
            if len(available_assays) == 0:
                continue
            
            # Handle edge case: if chunk size >= sequence length
            if self.chunk_size >= L:
                for f_idx in available_assays:
                    data[b, :, f_idx.item()] = self.mask_value
                continue
            
            # Calculate target number of loci to mask
            target_loci_to_mask = L * self.mask_fraction
            num_chunks_needed = max(1, int((target_loci_to_mask + self.chunk_size - 1) // self.chunk_size))
            max_possible_chunks = L // self.chunk_size
            num_chunks = min(num_chunks_needed, max_possible_chunks)
            
            if num_chunks == 0:
                continue
            
            # Generate non-overlapping chunk start positions
            chunk_starts = self._generate_non_overlapping_chunks(L, num_chunks, device)
            
            # Apply masking to selected chunks across ALL available assays
            for start_pos in chunk_starts:
                end_pos = min(start_pos + self.chunk_size, L)
                for f_idx in available_assays:
                    data[b, start_pos:end_pos, f_idx.item()] = self.mask_value
        
        return data, metadata, availability
    
    def _mask_full_assay(self, data, metadata, availability):
        """
        Mask entire assays including their metadata.
        
        For each sample, randomly chooses how many assays to mask (1 to num_available-1),
        ensuring at least one assay remains available.
        
        Args:
            data: [B, L, F] signal data tensor
            metadata: [B, 4, F] metadata tensor
            availability: [B, F] availability tensor
        
        Returns:
            Modified data, metadata, and availability
        """
        B, L, F = data.shape
        
        for b in range(B):
            available_indices = torch.where(availability[b] == 1)[0]
            num_available = len(available_indices)
            
            if num_available <= 1:
                # Can't mask if only 1 or 0 assays available
                continue
            
            # Randomly choose how many assays to mask: between 1 and (num_available - 1)
            num_to_mask = torch.randint(1, num_available, (1,)).item()
            
            # Randomly select which assays to mask
            mask_indices = torch.randperm(num_available)[:num_to_mask]
            actual_indices_to_mask = available_indices[mask_indices]
            
            # Apply full assay masking: mask data, metadata, and update availability
            data[b, :, actual_indices_to_mask] = self.mask_value
            metadata[b, :, actual_indices_to_mask] = self.mask_value
            availability[b, actual_indices_to_mask] = self.mask_value
        
        return data, metadata, availability
    
    def _mask_chunks(self, data, metadata, availability):
        """
        Mask different loci chunks independently for each available assay.
        
        Each assay gets its own random set of masked loci positions.
        Does NOT mask metadata.
        
        Args:
            data: [B, L, F] signal data tensor
            metadata: [B, 4, F] metadata tensor (not modified)
            availability: [B, F] availability tensor (checked for available assays)
        
        Returns:
            Modified data, unchanged metadata, unchanged availability
        """
        B, L, F = data.shape
        device = data.device
        
        for b in range(B):
            # Get available assays for this sample
            available_assays = torch.where(availability[b] == 1)[0]
            
            if len(available_assays) == 0:
                continue
            
            # For each available assay, apply independent chunk masking
            for f_idx in available_assays:
                f = f_idx.item()
                
                # Handle edge case: if chunk size >= sequence length
                if self.chunk_size >= L:
                    data[b, :, f] = self.mask_value
                    continue
                
                # Calculate target number of loci to mask
                target_loci_to_mask = L * self.mask_fraction
                num_chunks_needed = max(1, int((target_loci_to_mask + self.chunk_size - 1) // self.chunk_size))
                max_possible_chunks = L // self.chunk_size
                num_chunks = min(num_chunks_needed, max_possible_chunks)
                
                if num_chunks == 0:
                    continue
                
                # Generate non-overlapping chunk start positions (different for each assay)
                chunk_starts = self._generate_non_overlapping_chunks(L, num_chunks, device)
                
                # Apply masking to selected chunks for this assay only
                for start_pos in chunk_starts:
                    end_pos = min(start_pos + self.chunk_size, L)
                    data[b, start_pos:end_pos, f] = self.mask_value
        
        return data, metadata, availability
    
    def _generate_non_overlapping_chunks(self, L, num_chunks, device):
        """
        Generate non-overlapping chunk start positions.
        
        Args:
            L: Sequence length
            num_chunks: Number of chunks to generate
            device: Device for tensor operations
        
        Returns:
            List of chunk start positions
        """
        chunk_starts = []
        max_start = L - self.chunk_size
        attempts = 0
        max_attempts = 2000
        
        while len(chunk_starts) < num_chunks and attempts < max_attempts:
            start = torch.randint(0, max_start + 1, (1,), device=device).item()
            # Check if this start position overlaps with existing chunks
            overlaps = False
            for existing_start in chunk_starts:
                if not (start + self.chunk_size <= existing_start or 
                        existing_start + self.chunk_size <= start):
                    overlaps = True
                    break
            if not overlaps:
                chunk_starts.append(start)
            attempts += 1
        
        return chunk_starts
    
    def apply_mask(self, data, metadata, availability):
        """
        Apply masking strategies based on their probabilities.
        
        Strategies are applied in order: full_assay -> full_loci -> chunks
        At least one strategy is guaranteed to be applied.
        
        Args:
            data: [B, L, F] signal data tensor
            metadata: [B, 4, F] metadata tensor
            availability: [B, F] availability tensor
        
        Returns:
            Masked data, metadata, and availability tensors
        """
        # Clone tensors to avoid modifying originals
        masked_data = data.clone().float()
        masked_metadata = metadata.clone().float()
        masked_availability = availability.clone().float()
        
        # Track which strategies are applied
        applied_any = False
        
        # Decide which strategies to apply based on probabilities
        apply_full_assay = torch.rand(1).item() < self.p_full_assay
        apply_full_loci = torch.rand(1).item() < self.p_full_loci
        apply_chunks = torch.rand(1).item() < self.p_chunks
        
        # Apply strategies in order: full_assay -> full_loci -> chunks
        if apply_full_assay:
            masked_data, masked_metadata, masked_availability = self._mask_full_assay(
                masked_data, masked_metadata, masked_availability
            )
            applied_any = True
        
        if apply_full_loci:
            masked_data, masked_metadata, masked_availability = self._mask_full_loci(
                masked_data, masked_metadata, masked_availability
            )
            applied_any = True
        
        if apply_chunks:
            masked_data, masked_metadata, masked_availability = self._mask_chunks(
                masked_data, masked_metadata, masked_availability
            )
            applied_any = True
        
        # Ensure at least one strategy is applied
        if not applied_any:
            # Default to full loci masking if nothing was applied
            masked_data, masked_metadata, masked_availability = self._mask_full_loci(
                masked_data, masked_metadata, masked_availability
            )
        
        return masked_data, masked_metadata, masked_availability
    
    def mask_assays(self, data, metadata, availability, num_mask=None):
        """
        Legacy interface - calls apply_mask internally.
        
        Args:
            data: [B, L, F] signal data tensor
            metadata: [B, 4, F] metadata tensor  
            availability: [B, F] availability tensor
            num_mask: Deprecated parameter (ignored)
        
        Returns:
            Masked data, metadata, and availability tensors
        """
        return self.apply_mask(data, metadata, availability)
    
    def set_probabilities(self, p_full_loci=None, p_full_assay=None, p_chunks=None):
        """
        Update masking probabilities (useful for training-time scheduling).
        
        Args:
            p_full_loci: New probability for full loci masking (or None to keep current)
            p_full_assay: New probability for full assay masking (or None to keep current)
            p_chunks: New probability for chunk masking (or None to keep current)
        """
        if p_full_loci is not None:
            self.p_full_loci = p_full_loci
        if p_full_assay is not None:
            self.p_full_assay = p_full_assay
        if p_chunks is not None:
            self.p_chunks = p_chunks
    
    def get_probabilities(self):
        """
        Get current masking probabilities.
        
        Returns:
            dict: Current probabilities for each masking strategy
        """
        return {
            'p_full_loci': self.p_full_loci,
            'p_full_assay': self.p_full_assay,
            'p_chunks': self.p_chunks
        }

def reverse_complement_dna(dna_onehot):
    """
    Reverse complement a one-hot encoded DNA sequence.
    
    Args:
        dna_onehot: Tensor of shape [B, L, 4] where channels are [A, C, G, T]
    
    Returns:
        Reverse complemented DNA tensor [B, L, 4]
    """
    # Step 1: Reverse along sequence dimension (dim=1)
    dna_reversed = torch.flip(dna_onehot, dims=[1])
    
    # Step 2: Complement by swapping channels: A<->T (0<->3), C<->G (1<->2)
    # Channel order: [A, C, G, T] -> [T, G, C, A]
    dna_rc = dna_reversed[:, :, [3, 2, 1, 0]]
    
    return dna_rc

def reverse_signal(signal):
    """
    Reverse signal data along the sequence dimension.
    
    Args:
        signal: Tensor of shape [B, L, ...] where L is sequence length
    
    Returns:
        Reversed signal tensor [B, L, ...]
    """
    return torch.flip(signal, dims=[1])

class METRICS(object):
    def __init__(self, chrom='chr21', bin_size=25):
        self.prom_df = self.get_prom_positions(chrom, bin_size)
        self.gene_df = self.get_gene_positions(chrom, bin_size)

        self.gene_df["strand"] = self.prom_df["strand"]

    def get_gene_positions(self, chrom, bin_size):
        gene_df = pd.read_csv(PROC_GENE_BED_FPATH, sep='\t', header=None,
                              names=['chrom', 'start', 'end', 'gene_id', 'gene_name'])
        chrom_subset = gene_df[gene_df['chrom'] == chrom].copy()

        chrom_subset['start'] = (chrom_subset['start'] / bin_size).apply(lambda s: math.floor(s))
        chrom_subset['end'] = (chrom_subset['end'] / bin_size).apply(lambda s: math.floor(s))
        return chrom_subset

    def get_prom_positions(self, chrom, bin_size):
        prom_df = pd.read_csv(PROC_PROM_BED_PATH, sep='\t', header=None,
                              names=['chrom', 'start', 'end', 'gene_id', 'gene_name', "strand"])
        chrom_subset = prom_df[prom_df['chrom'] == chrom].copy()

        chrom_subset['start'] = (chrom_subset['start'] / bin_size).apply(lambda s: math.floor(s))
        chrom_subset['end'] = (chrom_subset['end'] / bin_size).apply(lambda s: math.floor(s))

        return chrom_subset
        
    def get_signals(self, array, df):
        indices = np.concatenate([np.arange(row['start'], row['end']) for _, row in df.iterrows()])
        valid_indices = indices[indices < len(array)]

        signals = array[valid_indices]
        return signals

    ################################################################################

    def get_gene_signals(self, y_true, y_pred, bin_size=25):
        # assert chrom == 'chr21', f'Got evaluation with unsupported chromosome {chrom}'

        gt_vals = self.get_signals(array=y_true, df=self.gene_df)
        pred_vals = self.get_signals(array=y_pred, df=self.gene_df)

        return gt_vals, pred_vals
    
    def get_prom_signals(self, y_true, y_pred, bin_size=25):
        # assert chrom == 'chr21', f'Got evaluation with unsupported chromosome {chrom}'

        gt_vals = self.get_signals(array=y_true, df=self.prom_df)
        pred_vals = self.get_signals(array=y_pred, df=self.prom_df)

        return gt_vals, pred_vals
    
    def get_1obs_signals(self, y_true, y_pred):
        perc_99 = np.percentile(y_true, 99)
        perc_99_pos = np.where(y_true >= perc_99)[0]

        return y_true[perc_99_pos], y_pred[perc_99_pos]

    def get_1imp_signals(self, y_true, y_pred):
        perc_99 = np.percentile(y_pred, 99)
        perc_99_pos = np.where(y_pred >= perc_99)[0]

        return y_true[perc_99_pos], y_pred[perc_99_pos]
    
    ################################################################################
    def r2(self, y_true, y_pred):
        return r2_score(y_true, y_pred)

    def r2_gene(self, y_true, y_pred, chrom='chr21', bin_size=25):
        # assert chrom == 'chr21', f'Got evaluation with unsupported chromosome {chrom}'

        # gene_df = self.get_gene_positions(chrom, bin_size)
        gt_vals = self.get_signals(array=y_true, df=self.gene_df)
        pred_vals = self.get_signals(array=y_pred, df=self.gene_df)

        return self.r2(y_true=gt_vals, y_pred=pred_vals)
    
    def r2_prom(self, y_true, y_pred, chrom='chr21', bin_size=25):
        # assert chrom == 'chr21', f'Got evaluation with unsupported chromosome {chrom}'

        # prom_df = self.get_prom_positions(chrom, bin_size)
        gt_vals = self.get_signals(array=y_true, df=self.prom_df)
        pred_vals = self.get_signals(array=y_pred, df=self.prom_df)

        return self.r2(y_true=gt_vals, y_pred=pred_vals)

    def r2_1obs(self, y_true, y_pred):
        perc_99 = np.percentile(y_true, 99)
        perc_99_pos = np.where(y_true >= perc_99)[0]

        return self.pearson(y_true[perc_99_pos], y_pred[perc_99_pos])

    def r2_1imp(self, y_true, y_pred):
        perc_99 = np.percentile(y_pred, 99)
        perc_99_pos = np.where(y_pred >= perc_99)[0]

        return self.r2(y_true[perc_99_pos], y_pred[perc_99_pos])

    def mse(self, y_true, y_pred):
        """
        Calculate the genome-wide Mean Squared Error (MSE). This is a measure of the average squared difference 
        between the true and predicted values across the entire genome at a resolution of 25bp.
        """
        return np.mean((np.array(y_true) - np.array(y_pred))**2)
    
    def mse_gene(self, y_true, y_pred, chrom='chr21', bin_size=25):
        # assert chrom == 'chr21', f'Got evaluation with unsupported chromosome {chrom}'

        # gene_df = self.get_gene_positions(chrom, bin_size)
        gt_vals = self.get_signals(array=y_true, df=self.gene_df)
        pred_vals = self.get_signals(array=y_pred, df=self.gene_df)

        return self.mse(y_true=gt_vals, y_pred=pred_vals)

    def pearson_gene(self, y_true, y_pred, chrom='chr21', bin_size=25):
        # assert chrom == 'chr21', f'Got evaluation with unsupported chromosome {chrom}'

        # gene_df = self.get_gene_positions(chrom, bin_size)
        gt_vals = self.get_signals(array=y_true, df=self.gene_df)
        pred_vals = self.get_signals(array=y_pred, df=self.gene_df)

        return self.pearson(y_true=gt_vals, y_pred=pred_vals)

    def spearman_gene(self, y_true, y_pred, chrom='chr21', bin_size=25):
        # assert chrom == 'chr21', f'Got evaluation with unsupported chromosome {chrom}'

        # gene_df = self.get_gene_positions(chrom, bin_size)
        gt_vals = self.get_signals(array=y_true, df=self.gene_df)
        pred_vals = self.get_signals(array=y_pred, df=self.gene_df)

        return self.spearman(y_true=gt_vals, y_pred=pred_vals)

    def pearson(self, y_true, y_pred):
        """
        Calculate the genome-wide Pearson Correlation. This measures the linear relationship between the true 
        and predicted values across the entire genome at a resolution of 25bp.
        """
        return pearsonr(y_pred, y_true)[0]

    def spearman(self, y_true, y_pred):
        """
        Calculate the genome-wide Spearman Correlation. This measures the monotonic relationship between the true 
        and predicted values across the entire genome at a resolution of 25bp.
        """
        return spearmanr(y_pred, y_true)[0]

    def mse_prom(self, y_true, y_pred, chrom='chr21', bin_size=25):
        # assert chrom == 'chr21', f'Got evaluation with unsupported chromosome {chrom}'

        # prom_df = self.get_prom_positions(chrom, bin_size)
        gt_vals = self.get_signals(array=y_true, df=self.prom_df)
        pred_vals = self.get_signals(array=y_pred, df=self.prom_df)

        return self.mse(y_true=gt_vals, y_pred=pred_vals)

    def pearson_prom(self, y_true, y_pred, chrom='chr21', bin_size=25):
        # assert chrom == 'chr21', f'Got evaluation with unsupported chromosome {chrom}'

        # prom_df = self.get_prom_positions(chrom, bin_size)
        gt_vals = self.get_signals(array=y_true, df=self.prom_df)
        pred_vals = self.get_signals(array=y_pred, df=self.prom_df)

        return self.pearson(y_true=gt_vals, y_pred=pred_vals)

    def spearman_prom(self, y_true, y_pred, chrom='chr21', bin_size=25):
        # assert chrom == 'chr21', f'Got evaluation with unsupported chromosome {chrom}'

        # prom_df = self.get_prom_positions(chrom, bin_size)
        gt_vals = self.get_signals(array=y_true, df=self.prom_df)
        pred_vals = self.get_signals(array=y_pred, df=self.prom_df)

        return self.spearman(y_true=gt_vals, y_pred=pred_vals)

    def mse1obs(self, y_true, y_pred):
        """
        Calculate the Mean Squared Error at the top 1% of genomic positions ranked by experimental signal (mse1obs). 
        This is a measure of how well predictions match observations at positions with high experimental signal. 
        It's similar to recall.
        """
        top_1_percent = int(0.01 * len(y_true))
        top_1_percent_indices = np.argsort(y_true)[-top_1_percent:]
        return mean_squared_error(y_true[top_1_percent_indices], y_pred[top_1_percent_indices])

    def mse1imp(self, y_true, y_pred):
        """
        Calculate the Mean Squared Error at the top 1% of genomic positions ranked by predicted signal (mse1imp). 
        This is a measure of how well predictions match observations at positions with high predicted signal. 
        It's similar to precision.
        """
        top_1_percent = int(0.01 * len(y_pred))
        top_1_percent_indices = np.argsort(y_pred)[-top_1_percent:]
        return mean_squared_error(y_true[top_1_percent_indices], y_pred[top_1_percent_indices])

    def pearson1_obs(self, y_true, y_pred):
        perc_99 = np.percentile(y_true, 99)
        perc_99_pos = np.where(y_true >= perc_99)[0]

        return self.pearson(y_true[perc_99_pos], y_pred[perc_99_pos])

    def spearman1_obs(self, y_true, y_pred):
        perc_99 = np.percentile(y_true, 99)
        perc_99_pos = np.where(y_true >= perc_99)[0]

        return self.spearman(y_true[perc_99_pos], y_pred[perc_99_pos])

    def pearson1_imp(self, y_true, y_pred):
        perc_99 = np.percentile(y_pred, 99)
        perc_99_pos = np.where(y_pred >= perc_99)[0]

        return self.pearson(y_true[perc_99_pos], y_pred[perc_99_pos])

    def spearman1_imp(self, y_true, y_pred):
        perc_99 = np.percentile(y_pred, 99)
        perc_99_pos = np.where(y_pred >= perc_99)[0]

        return self.spearman(y_true[perc_99_pos], y_pred[perc_99_pos])

    ################################################################################
    # MAE and MAE-R2 (L1 analog of R^2)
    ################################################################################

    def mae(self, y_true, y_pred):
        """
        Calculate the genome-wide Mean Absolute Error (MAE).
        """
        return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

    def mae_gene(self, y_true, y_pred, chrom='chr21', bin_size=25):
        """Calculate MAE within gene bodies."""
        gt_vals = self.get_signals(array=y_true, df=self.gene_df)
        pred_vals = self.get_signals(array=y_pred, df=self.gene_df)
        return self.mae(y_true=gt_vals, y_pred=pred_vals)

    def mae_prom(self, y_true, y_pred, chrom='chr21', bin_size=25):
        """Calculate MAE within promoter regions."""
        gt_vals = self.get_signals(array=y_true, df=self.prom_df)
        pred_vals = self.get_signals(array=y_pred, df=self.prom_df)
        return self.mae(y_true=gt_vals, y_pred=pred_vals)

    def mae1obs(self, y_true, y_pred):
        """
        Calculate MAE at the top 1% of genomic positions ranked by experimental signal.
        Similar to recall - how well predictions match at high-signal positions.
        """
        perc_99 = np.percentile(y_true, 99)
        perc_99_pos = np.where(y_true >= perc_99)[0]
        return self.mae(y_true[perc_99_pos], y_pred[perc_99_pos])

    def mae1imp(self, y_true, y_pred):
        """
        Calculate MAE at the top 1% of genomic positions ranked by predicted signal.
        Similar to precision - how well predictions match at high-prediction positions.
        """
        perc_99 = np.percentile(y_pred, 99)
        perc_99_pos = np.where(y_pred >= perc_99)[0]
        return self.mae(y_true[perc_99_pos], y_pred[perc_99_pos])

    def mae_r2(self, y_true, y_pred):
        """
        Calculate genome-wide MAE-R2 (L1 analog of R^2 with mean baseline).
        Formula: 1 - sum(|y - yhat|) / sum(|y - mean(y)|)
        Returns 0.0 if denominator is near zero to avoid division errors.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        numerator = np.sum(np.abs(y_true - y_pred))
        denominator = np.sum(np.abs(y_true - np.mean(y_true)))
        if denominator < 1e-10:
            return 0.0
        return 1.0 - (numerator / denominator)

    def mae_r2_gene(self, y_true, y_pred, chrom='chr21', bin_size=25):
        """Calculate MAE-R2 within gene bodies."""
        gt_vals = self.get_signals(array=y_true, df=self.gene_df)
        pred_vals = self.get_signals(array=y_pred, df=self.gene_df)
        return self.mae_r2(y_true=gt_vals, y_pred=pred_vals)

    def mae_r2_prom(self, y_true, y_pred, chrom='chr21', bin_size=25):
        """Calculate MAE-R2 within promoter regions."""
        gt_vals = self.get_signals(array=y_true, df=self.prom_df)
        pred_vals = self.get_signals(array=y_pred, df=self.prom_df)
        return self.mae_r2(y_true=gt_vals, y_pred=pred_vals)

    def mae_r2_1obs(self, y_true, y_pred):
        """Calculate MAE-R2 at the top 1% of genomic positions ranked by experimental signal."""
        perc_99 = np.percentile(y_true, 99)
        perc_99_pos = np.where(y_true >= perc_99)[0]
        return self.mae_r2(y_true[perc_99_pos], y_pred[perc_99_pos])

    def mae_r2_1imp(self, y_true, y_pred):
        """Calculate MAE-R2 at the top 1% of genomic positions ranked by predicted signal."""
        perc_99 = np.percentile(y_pred, 99)
        perc_99_pos = np.where(y_pred >= perc_99)[0]
        return self.mae_r2(y_true[perc_99_pos], y_pred[perc_99_pos])

    ################################################################################

    def aucroc(self, y_true, y_pred):
        """
        Calculate genome-wide AUCROC for peak classification.
        Uses obs_peak as binary ground truth and pred_peak as probabilities.
        """
        return roc_auc_score(y_true, y_pred)

    def aucroc_gene(self, y_true, y_pred, chrom='chr21', bin_size=25):
        """Calculate AUCROC for peak classification within gene bodies."""
        gt_vals = self.get_signals(array=y_true, df=self.gene_df)
        pred_vals = self.get_signals(array=y_pred, df=self.gene_df)
        return self.aucroc(y_true=gt_vals, y_pred=pred_vals)

    def aucroc_prom(self, y_true, y_pred, chrom='chr21', bin_size=25):
        """Calculate AUCROC for peak classification within promoter regions."""
        gt_vals = self.get_signals(array=y_true, df=self.prom_df)
        pred_vals = self.get_signals(array=y_pred, df=self.prom_df)
        return self.aucroc(y_true=gt_vals, y_pred=pred_vals)

    def peak_overlap(self, y_true, y_pred, p=0.01):
        if p == 0:
            return 0

        elif p == 1:
            return 1

        top_p_percent = int(p * len(y_true))

        # Get the indices of the top p percent of the observed (true) values
        top_p_percent_obs_i = np.argsort(y_true)[-top_p_percent:]
        
        # Get the indices of the top p percent of the predicted values
        top_p_percent_pred_i = np.argsort(y_pred)[-top_p_percent:]

        # Calculate the overlap
        overlap = len(np.intersect1d(top_p_percent_obs_i, top_p_percent_pred_i))

        # Calculate the percentage of overlap
        overlap_percent = overlap / top_p_percent 

        return overlap_percent

    def correspondence_curve(self, y_true, y_pred):
        curve = []
        derivatives = []
        steps = [float(p / 100) for p in range(0, 101, 1)]

        obs_rank = np.argsort(y_true)
        pred_rank = np.argsort(y_pred)

        for p in steps:
            if p == 0 or p == 1:
                overlap_percent = p
            else:
                top_p_percent = int(p * len(y_true))
                top_p_percent_obs_i = obs_rank[-top_p_percent:]
                top_p_percent_pred_i = pred_rank[-top_p_percent:]

                overlap = len(np.intersect1d(top_p_percent_obs_i, top_p_percent_pred_i))
                overlap_percent = overlap / len(y_true)

            curve.append((p, overlap_percent))

        # Calculate derivatives using finite differences
        for i in range(1, len(curve)):
            dp = curve[i][0] - curve[i-1][0]
            d_overlap_percent = curve[i][1] - curve[i-1][1]
            derivative = d_overlap_percent / dp
            derivatives.append((curve[i][0], derivative))

        return curve, derivatives

    def confidence_quantile(self, nbinom_p, nbinom_n, y_true):
        nbinom_dist = NegativeBinomial(nbinom_p, nbinom_n)
        return nbinom_dist.cdf(y_true)

    def foreground_vs_background(self, nbinom_p, nbinom_n, y_true):
        """
        inputs: 1) nbinom_p, nbinom_n -> two arrays with length L -> negative binomial dist parameters for each position
                2) y_true -> one array of true observed signal

        task:

            - NEW 2: peak vs. background comparison
            - binarize each observed experiment according to some threshold.
            - measure the two following
                - what fraction of positions outside peaks have overlap (confidence interval) with zero
                - for peaks, for what fraction, the confidence interval overlap with 0. this should ideally be low since otherwise the model is not sure about the peak
            - intervals 90 95 99 percent
        """

        nbinom_dist = NegativeBinomial(nbinom_p, nbinom_n)
        binarized_y = binarize_nbinom(y_true)

        pmf_zero = (nbinom_dist.pmf(0))

        analysis = {}

        background_pmf_zero = pmf_zero[binarized_y == 0].mean() 
        peak_pmf_zero = pmf_zero[binarized_y == 1].mean() 

        analysis["p0_bg"] = background_pmf_zero.item()
        analysis["p0_fg"] = peak_pmf_zero.item()

        return analysis

    def c_index_gauss(self, mus, sigmas, y_true, num_pairs: int = 10000):
        """
        Concordance index for Gaussian predictive marginals,
        estimating over `num_pairs` randomly sampled pairs.
        
        Inputs:
          - mus:       array_like, shape (N,) of predicted means μ_i
          - sigmas:    array_like, shape (N,) of predicted stddevs σ_i
          - y_true:    array_like, shape (N,) of true values y_i
          - num_pairs: number of random (i<j) pairs to sample;
                       if -1, use all possible pairs (i<j)
        Returns:
          - c_index: float in [0,1]
        """
        N = len(y_true)
        labels = []
        scores = []

        if num_pairs == -1:
            # exact over all valid pairs
            for i in range(N):
                for j in range(i+1, N):
                    if y_true[i] == y_true[j]:
                        continue
                    labels.append(int(y_true[i] > y_true[j]))
                    delta = mus[i] - mus[j]
                    sd = np.sqrt(sigmas[i]**2 + sigmas[j]**2)
                    sd = max(sd, 1e-10)  # Clamp to avoid division by zero
                    scores.append(norm.cdf(delta / sd))
        else:
            # Monte Carlo sampling of pairs
            rng = np.random.default_rng()
            count = 0
            while count < num_pairs:
                i, j = rng.integers(0, N, size=2)
                if i == j or y_true[i] == y_true[j]:
                    continue
                # Optional: enforce ordering i<j for consistency
                if i > j:
                    i, j = j, i
                labels.append(int(y_true[i] > y_true[j]))
                delta = mus[i] - mus[j]
                sd = np.sqrt(sigmas[i]**2 + sigmas[j]**2)
                sd = max(sd, 1e-10)  # Clamp to avoid division by zero
                scores.append(norm.cdf(delta / sd))
                count += 1

        return roc_auc_score(labels, scores)

    def c_index_gauss_gene(self, mus, sigmas, y_true, num_pairs=10000):
        indices = np.concatenate([np.arange(row['start'], row['end']) for _, row in self.gene_df.iterrows()])
        valid_indices = indices[indices < len(y_true)]
        
        N = len(valid_indices)
        if (N*(N-1))/2 < num_pairs:
            num_pairs = -1

        c_idx = self.c_index_gauss(mus[valid_indices], sigmas[valid_indices], y_true[valid_indices], num_pairs)
        return c_idx

    def c_index_gauss_prom(self, mus, sigmas, y_true, num_pairs=10000):
        indices = np.concatenate([np.arange(row['start'], row['end']) for _, row in self.prom_df.iterrows()])
        valid_indices = indices[indices < len(y_true)]

        N = len(valid_indices)
        if (N*(N-1))/2 < num_pairs:
            num_pairs = -1

        c_idx = self.c_index_gauss(mus[valid_indices], sigmas[valid_indices], y_true[valid_indices], num_pairs)
        return c_idx

    def c_index_gauss_1obs(self, mus, sigmas, y_true, num_pairs=10000):
        perc_99 = np.percentile(y_true, 99)
        perc_99_pos = np.where(y_true >= perc_99)[0]

        N = len(perc_99_pos)
        if (N*(N-1))/2 < num_pairs:
            num_pairs = -1

        c_idx = self.c_index_gauss(mus[perc_99_pos], sigmas[perc_99_pos], y_true[perc_99_pos], num_pairs)
        return c_idx

    def c_index_laplace(self, mus, log_bs, y_true, M: int = 500, num_pairs: int = 10000, random_state: int = None):
        """
        Monte Carlo Concordance index for Laplace predictive marginals.
        
        Inputs:
          - mus:       array_like, shape (N,) of predicted means
          - log_bs:    array_like, shape (N,) of predicted log-scales
          - y_true:    array_like, shape (N,) of true values
          - M:         number of Monte Carlo samples per pair
          - num_pairs: number of random (i<j) pairs to sample
        """
        rng = np.random.default_rng(random_state)
        N = len(y_true)
        labels = []
        scores = []
        
        # Convert log_b to b (scale)
        bs = np.exp(log_bs)

        def mc_score(i, j):
            # draw M samples from each distribution
            u = laplace.rvs(loc=mus[i], scale=bs[i], size=M, random_state=rng)
            v = laplace.rvs(loc=mus[j], scale=bs[j], size=M, random_state=rng)
            return np.mean(u > v)

        if num_pairs == -1:
            # exact over all pairs
            for i in range(N):
                for j in range(i+1, N):
                    if y_true[i] == y_true[j]:
                        continue
                    labels.append(int(y_true[i] > y_true[j]))
                    scores.append(mc_score(i, j))
        else:
            # sample random pairs
            count = 0
            while count < num_pairs:
                i, j = rng.integers(0, N, size=2)
                if i == j or y_true[i] == y_true[j]:
                    continue
                # ensure i<j for consistency
                if i > j:
                    i, j = j, i
                labels.append(int(y_true[i] > y_true[j]))
                scores.append(mc_score(i, j))
                count += 1

        if not labels:
            return np.nan
        return roc_auc_score(labels, scores)

    def c_index_laplace_gene(self, mus, log_bs, y_true, num_pairs=10000):
        indices = np.concatenate([np.arange(row['start'], row['end']) for _, row in self.gene_df.iterrows()])
        valid_indices = indices[indices < len(y_true)]
        
        N = len(valid_indices)
        if (N*(N-1))/2 < num_pairs:
            num_pairs = -1

        c_idx = self.c_index_laplace(mus[valid_indices], log_bs[valid_indices], y_true[valid_indices], num_pairs=num_pairs)
        return c_idx

    def c_index_laplace_prom(self, mus, log_bs, y_true, num_pairs=10000):
        indices = np.concatenate([np.arange(row['start'], row['end']) for _, row in self.prom_df.iterrows()])
        valid_indices = indices[indices < len(y_true)]

        N = len(valid_indices)
        if (N*(N-1))/2 < num_pairs:
            num_pairs = -1

        c_idx = self.c_index_laplace(mus[valid_indices], log_bs[valid_indices], y_true[valid_indices], num_pairs=num_pairs)
        return c_idx

    def c_index_laplace_1obs(self, mus, log_bs, y_true, num_pairs=10000):
        perc_99 = np.percentile(y_true, 99)
        perc_99_pos = np.where(y_true >= perc_99)[0]

        N = len(perc_99_pos)
        if (N*(N-1))/2 < num_pairs:
            num_pairs = -1

        c_idx = self.c_index_laplace(mus[perc_99_pos], log_bs[perc_99_pos], y_true[perc_99_pos], num_pairs=num_pairs)
        return c_idx

    def c_index_nbinom(self,rs, ps, y_true, M: int = 500, num_pairs: int = 10000, random_state: int = None):
        """
        Monte Carlo Concordance index for Negative‐Binomial predictive marginals.

        For each sampled pair (i,j), draw M samples from NB(ri,pi) and NB(rj,pj),
        then estimate Pr(Y_i>Y_j) by the fraction of draws with u>v.

        Inputs:
          - rs:        (N,) array of NB 'r' parameters
          - ps:        (N,) array of NB 'p' parameters
          - y_true:    (N,) array of true values y_i
          - M:         number of Monte Carlo samples per pair
          - num_pairs: how many random (i<j) pairs to sample;
                       if -1, use all valid pairs (i<j, y_i≠y_j)
          - random_state: seed for reproducibility
        Returns:
          - c_index: float in [0,1]
        """
        rng = np.random.default_rng(random_state)
        N = len(y_true)
        labels = []
        scores = []

        def mc_score(i, j):
            # draw M samples from each distribution
            u = nbinom.rvs(rs[i], ps[i], size=M, random_state=rng)
            v = nbinom.rvs(rs[j], ps[j], size=M, random_state=rng)
            return np.mean(u > v)

        if num_pairs == -1:
            # exact over all pairs
            for i in range(N):
                for j in range(i+1, N):
                    if y_true[i] == y_true[j]:
                        continue
                    labels.append(int(y_true[i] > y_true[j]))
                    scores.append(mc_score(i, j))
        else:
            # sample random pairs
            count = 0
            while count < num_pairs:
                i, j = rng.integers(0, N, size=2)
                if i == j or y_true[i] == y_true[j]:
                    continue
                # ensure i<j for consistency (optional)
                if i > j:
                    i, j = j, i
                labels.append(int(y_true[i] > y_true[j]))
                scores.append(mc_score(i, j))
                count += 1

        if not labels:
            return np.nan
        return roc_auc_score(labels, scores)

    def c_index_nbinom_gene(self, rs, ps, y_true, num_pairs=10000):
        indices = np.concatenate([np.arange(row['start'], row['end']) for _, row in self.gene_df.iterrows()])
        valid_indices = indices[indices < len(y_true)]

        N = len(valid_indices)
        if (N*(N-1))/2 < num_pairs:
            num_pairs = -1

        c_idx = self.c_index_nbinom(rs[valid_indices], ps[valid_indices], y_true[valid_indices], num_pairs)
        return c_idx

    def c_index_nbinom_prom(self, rs, ps, y_true, num_pairs=10000):
        indices = np.concatenate([np.arange(row['start'], row['end']) for _, row in self.prom_df.iterrows()])
        valid_indices = indices[indices < len(y_true)]

        N = len(valid_indices)
        if (N*(N-1))/2 < num_pairs:
            num_pairs = -1

        c_idx = self.c_index_nbinom(rs[valid_indices], ps[valid_indices], y_true[valid_indices], num_pairs)
        return c_idx

    def c_index_nbinom_1obs(self, rs, ps, y_true, num_pairs=10000):
        perc_99 = np.percentile(y_true, 99)
        perc_99_pos = np.where(y_true >= perc_99)[0]

        N = len(perc_99_pos)
        if (N*(N-1))/2 < num_pairs:
            num_pairs = -1
        
        c_idx = self.c_index_nbinom(rs[perc_99_pos], ps[perc_99_pos], y_true[perc_99_pos], num_pairs)
        return c_idx

    def coverage_95ci(self, y_true, lower_bound, upper_bound):
        """
        Calculate the fraction of y_true values that fall within the 95% confidence interval.
        
        Parameters:
        -----------
        y_true : array_like
            True observed values
        lower_bound : array_like
            Lower bound of the 95% confidence interval
        upper_bound : array_like
            Upper bound of the 95% confidence interval
            
        Returns:
        --------
        float : Fraction of y_true values within [lower_bound, upper_bound]
        """
        within_ci = np.logical_and(y_true >= lower_bound, y_true <= upper_bound)
        return np.mean(within_ci)

    def coverage_95ci_gene(self, y_true, lower_bound, upper_bound):
        """Calculate 95% CI coverage for gene body regions."""
        gt_vals = self.get_signals(array=y_true, df=self.gene_df)
        lower_vals = self.get_signals(array=lower_bound, df=self.gene_df)
        upper_vals = self.get_signals(array=upper_bound, df=self.gene_df)
        
        return self.coverage_95ci(y_true=gt_vals, lower_bound=lower_vals, upper_bound=upper_vals)

    def coverage_95ci_prom(self, y_true, lower_bound, upper_bound):
        """Calculate 95% CI coverage for promoter regions."""
        gt_vals = self.get_signals(array=y_true, df=self.prom_df)
        lower_vals = self.get_signals(array=lower_bound, df=self.prom_df)
        upper_vals = self.get_signals(array=upper_bound, df=self.prom_df)
        
        return self.coverage_95ci(y_true=gt_vals, lower_bound=lower_vals, upper_bound=upper_vals)

    def coverage_95ci_1obs(self, y_true, lower_bound, upper_bound):
        """Calculate 95% CI coverage for top 1% of positions by observed signal."""
        perc_99 = np.percentile(y_true, 99)
        perc_99_pos = np.where(y_true >= perc_99)[0]
        
        return self.coverage_95ci(
            y_true[perc_99_pos], 
            lower_bound[perc_99_pos], 
            upper_bound[perc_99_pos]
        )

def get_divisible_heads(dim, target):
    """
    Given a dimension and a target number of heads, returns the largest integer
    <= target that divides dim. If no such number is found, returns 1.
    """
    for n in range(target, 0, -1):
        if dim % n == 0:
            return n
    return 1

def log_resource_usage():
    print(f"CPU Usage: {psutil.cpu_percent()}%")
    print(f"Memory Usage: {psutil.virtual_memory().percent}%")
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.memory_stats()
        print(f"GPU Memory Allocated: {gpu_stats['allocated_bytes.all.current'] / (1024 ** 2)} MB")
        print(f"GPU Memory Reserved: {gpu_stats['reserved_bytes.all.current'] / (1024 ** 2)} MB")
        print(f"GPU Active Memory Allocations: {gpu_stats['active.all.current']}")
        print(f"GPU Memory Allocated (peak): {gpu_stats['allocated_bytes.all.peak'] / (1024 ** 2)} MB")
        print(f"GPU Memory Reserved (peak): {gpu_stats['reserved_bytes.all.peak'] / (1024 ** 2)} MB")

def compute_perplexity(probabilities):
    """
    Computes the perplexity given a list of probabilities.

    Parameters:
    probabilities (list or np.array): A list or array of probabilities assigned by the model to each word in the sequence.

    Returns:
    float: The perplexity of the model on the given sequence.
    """
    N = len(probabilities)
    log_prob_sum = torch.sum(torch.log(probabilities))
    perplexity = torch.exp(-log_prob_sum / N)
    
    return perplexity

class Gaussian:
    def __init__(self, mu, var):
        self.mu = mu
        self.var = var
        # clamp var to avoid sqrt of negative
        if torch.is_tensor(self.var):
             self.var = torch.clamp(self.var, min=1e-10)
        elif isinstance(self.var, np.ndarray):
             self.var = np.clip(self.var, 1e-10, None)
        
        self.sigma = self.var ** (1/2)

    def mean(self):
        return self.mu

    def median(self):
        return self.mu

    def mode(self):
        return self.mu

    def var(self):
        return self.var

    def std(self):
        return self.sigma

    def cdf(self, x):
        x = x.detach().cpu().numpy() if torch.is_tensor(x) else x
        mu = self.mu.detach().cpu().numpy() if torch.is_tensor(self.mu) else self.mu
        sigma = self.sigma.detach().cpu().numpy() if torch.is_tensor(self.sigma) else self.sigma
        # Clamp sigma to avoid division by zero
        if np.isscalar(sigma):
            sigma = max(sigma, 1e-10)
        else:
            sigma = np.clip(sigma, 1e-10, None)
        return torch.tensor(norm.cdf(x, mu, sigma), dtype=torch.float32)

    def pdf(self, x):
        x = x.detach().cpu().numpy() if torch.is_tensor(x) else x
        mu = self.mu.detach().cpu().numpy() if torch.is_tensor(self.mu) else self.mu
        sigma = self.sigma.detach().cpu().numpy() if torch.is_tensor(self.sigma) else self.sigma
        # Clamp sigma to avoid division by zero
        if np.isscalar(sigma):
            sigma = max(sigma, 1e-10)
        else:
            sigma = np.clip(sigma, 1e-10, None)
        return torch.tensor(norm.pdf(x, mu, sigma), dtype=torch.float32)

    def icdf(self, q):
        q = q.detach().cpu().numpy() if torch.is_tensor(q) else q
        mu = self.mu.detach().cpu().numpy() if torch.is_tensor(self.mu) else self.mu
        sigma = self.sigma.detach().cpu().numpy() if torch.is_tensor(self.sigma) else self.sigma
        # Clamp sigma to avoid division by zero
        if np.isscalar(sigma):
            sigma = max(sigma, 1e-10)
        else:
            sigma = np.clip(sigma, 1e-10, None)
        return torch.tensor(norm.ppf(q, mu, sigma), dtype=torch.float32)

    def expect(self):
        return self.mu

    def interval(self, confidence=0.95):
        lower = self.icdf((1 - confidence) / 2)
        upper = self.icdf((1 + confidence) / 2)
        return lower, upper

class Laplace:
    def __init__(self, mu, log_b):
        self.mu = mu
        # log_b can be passed directly; we compute b = exp(log_b)
        # clamp log_b to avoid numerical issues
        if torch.is_tensor(log_b):
             self.log_b = torch.clamp(log_b, min=-10.0, max=10.0)
             self.b = torch.exp(self.log_b)
        elif isinstance(log_b, np.ndarray):
             self.log_b = np.clip(log_b, -10.0, 10.0)
             self.b = np.exp(self.log_b)
        else:
             # scalar case
             self.log_b = max(min(log_b, 10.0), -10.0)
             self.b = math.exp(self.log_b)

    def mean(self):
        return self.mu

    def median(self):
        return self.mu

    def mode(self):
        return self.mu

    def var(self):
        return 2 * (self.b ** 2)

    def std(self):
        return (2 ** 0.5) * self.b

    def cdf(self, x):
        x = x.detach().cpu().numpy() if torch.is_tensor(x) else x
        mu = self.mu.detach().cpu().numpy() if torch.is_tensor(self.mu) else self.mu
        b = self.b.detach().cpu().numpy() if torch.is_tensor(self.b) else self.b
        
        # Clamp scale to avoid division by zero
        if np.isscalar(b):
            b = max(b, 1e-10)
        else:
            b = np.clip(b, 1e-10, None)
            
        return torch.tensor(laplace.cdf(x, loc=mu, scale=b), dtype=torch.float32)

    def pdf(self, x):
        x = x.detach().cpu().numpy() if torch.is_tensor(x) else x
        mu = self.mu.detach().cpu().numpy() if torch.is_tensor(self.mu) else self.mu
        b = self.b.detach().cpu().numpy() if torch.is_tensor(self.b) else self.b
        
        # Clamp scale to avoid division by zero
        if np.isscalar(b):
            b = max(b, 1e-10)
        else:
            b = np.clip(b, 1e-10, None)
            
        return torch.tensor(laplace.pdf(x, loc=mu, scale=b), dtype=torch.float32)

    def icdf(self, q):
        q = q.detach().cpu().numpy() if torch.is_tensor(q) else q
        mu = self.mu.detach().cpu().numpy() if torch.is_tensor(self.mu) else self.mu
        b = self.b.detach().cpu().numpy() if torch.is_tensor(self.b) else self.b
        
        # Clamp scale to avoid division by zero
        if np.isscalar(b):
            b = max(b, 1e-10)
        else:
            b = np.clip(b, 1e-10, None)
            
        return torch.tensor(laplace.ppf(q, loc=mu, scale=b), dtype=torch.float32)

    def expect(self):
        return self.mu

    def interval(self, confidence=0.95):
        lower = self.icdf((1 - confidence) / 2)
        upper = self.icdf((1 + confidence) / 2)
        return lower, upper

class Gamma:
    """
    Gamma distribution wrapper for inference, mirroring Gaussian/Laplace API.

    Args:
        mu: Mean (>0)
        alpha: Concentration/shape (>0)
    """
    def __init__(self, mu, alpha, eps=1e-10):
        self.mu = mu
        self.alpha = alpha
        self.eps = eps

        # Clamp parameters to avoid invalid values
        if torch.is_tensor(self.mu):
            self.mu = torch.clamp(self.mu, min=eps)
        elif isinstance(self.mu, np.ndarray):
            self.mu = np.clip(self.mu, eps, None)
        else:
            self.mu = max(self.mu, eps)

        if torch.is_tensor(self.alpha):
            self.alpha = torch.clamp(self.alpha, min=eps)
        elif isinstance(self.alpha, np.ndarray):
            self.alpha = np.clip(self.alpha, eps, None)
        else:
            self.alpha = max(self.alpha, eps)

        # rate = alpha / mu
        self.beta = self.alpha / self.mu

    def mean(self):
        return self.mu

    def median(self):
        # No closed form; use mean as approximation
        return self.mu

    def mode(self):
        if torch.is_tensor(self.alpha):
            return torch.clamp(self.mu * (self.alpha - 1.0) / self.alpha, min=0.0)
        elif isinstance(self.alpha, np.ndarray):
            return np.clip(self.mu * (self.alpha - 1.0) / self.alpha, 0.0, None)
        else:
            return max(self.mu * (self.alpha - 1.0) / self.alpha, 0.0)

    def var(self):
        return self.alpha / (self.beta ** 2)

    def std(self):
        v = self.var()
        if torch.is_tensor(v):
            return torch.sqrt(v)
        elif isinstance(v, np.ndarray):
            return np.sqrt(v)
        else:
            return math.sqrt(v)

    def pdf(self, x):
        x = x.detach().cpu().numpy() if torch.is_tensor(x) else x
        alpha = self.alpha.detach().cpu().numpy() if torch.is_tensor(self.alpha) else self.alpha
        beta = self.beta.detach().cpu().numpy() if torch.is_tensor(self.beta) else self.beta
        scale = 1.0 / beta
        return torch.tensor(scipy_gamma.pdf(x, a=alpha, scale=scale), dtype=torch.float32)

    def cdf(self, x):
        x = x.detach().cpu().numpy() if torch.is_tensor(x) else x
        alpha = self.alpha.detach().cpu().numpy() if torch.is_tensor(self.alpha) else self.alpha
        beta = self.beta.detach().cpu().numpy() if torch.is_tensor(self.beta) else self.beta
        scale = 1.0 / beta
        return torch.tensor(scipy_gamma.cdf(x, a=alpha, scale=scale), dtype=torch.float32)

    def icdf(self, q):
        q = q.detach().cpu().numpy() if torch.is_tensor(q) else q
        alpha = self.alpha.detach().cpu().numpy() if torch.is_tensor(self.alpha) else self.alpha
        beta = self.beta.detach().cpu().numpy() if torch.is_tensor(self.beta) else self.beta
        scale = 1.0 / beta
        return torch.tensor(scipy_gamma.ppf(q, a=alpha, scale=scale), dtype=torch.float32)

    def interval(self, confidence=0.95):
        lower = self.icdf((1 - confidence) / 2)
        upper = self.icdf((1 + confidence) / 2)
        return lower, upper

class StudentsT:
    """
    Student's t-distribution wrapper for inference, mirroring Gaussian/Laplace API.
    
    Args:
        mu: Location parameter (predicted mean)
        sigma: Scale parameter (predicted scale, >0)
        df: Degrees of freedom (>0, controls tail heaviness)
    """
    def __init__(self, mu, sigma, df):
        self.mu = mu
        
        # Clamp sigma to avoid division by zero
        if torch.is_tensor(sigma):
            self.sigma = torch.clamp(sigma, min=1e-10)
        elif isinstance(sigma, np.ndarray):
            self.sigma = np.clip(sigma, 1e-10, None)
        else:
            self.sigma = max(sigma, 1e-10)
        
        # Clamp df to avoid numerical issues (must be > 0)
        if torch.is_tensor(df):
            self.df = torch.clamp(df, min=1e-6)
        elif isinstance(df, np.ndarray):
            self.df = np.clip(df, 1e-6, None)
        else:
            self.df = max(df, 1e-6)

    def mean(self):
        """Mean is defined for df > 1, otherwise returns mu as approximation."""
        return self.mu

    def median(self):
        """Median equals location for symmetric Student's t."""
        return self.mu

    def mode(self):
        """Mode equals location for symmetric Student's t."""
        return self.mu

    def var(self):
        """
        Variance = sigma^2 * df / (df - 2) for df > 2.
        Returns inf for df <= 2.
        """
        if torch.is_tensor(self.df):
            var = torch.where(
                self.df > 2,
                (self.sigma ** 2) * self.df / (self.df - 2),
                torch.full_like(self.sigma, float('inf'))
            )
        elif isinstance(self.df, np.ndarray):
            var = np.where(
                self.df > 2,
                (self.sigma ** 2) * self.df / (self.df - 2),
                np.inf
            )
        else:
            if self.df > 2:
                var = (self.sigma ** 2) * self.df / (self.df - 2)
            else:
                var = float('inf')
        return var

    def std(self):
        """Standard deviation (sqrt of variance)."""
        v = self.var()
        if torch.is_tensor(v):
            return torch.sqrt(v)
        elif isinstance(v, np.ndarray):
            return np.sqrt(v)
        else:
            return math.sqrt(v) if v != float('inf') else float('inf')

    def cdf(self, x):
        x = x.detach().cpu().numpy() if torch.is_tensor(x) else x
        mu = self.mu.detach().cpu().numpy() if torch.is_tensor(self.mu) else self.mu
        sigma = self.sigma.detach().cpu().numpy() if torch.is_tensor(self.sigma) else self.sigma
        df = self.df.detach().cpu().numpy() if torch.is_tensor(self.df) else self.df
        
        # Clamp parameters
        if np.isscalar(sigma):
            sigma = max(sigma, 1e-10)
        else:
            sigma = np.clip(sigma, 1e-10, None)
        if np.isscalar(df):
            df = max(df, 1e-6)
        else:
            df = np.clip(df, 1e-6, None)
        
        return torch.tensor(scipy_t.cdf(x, df=df, loc=mu, scale=sigma), dtype=torch.float32)

    def pdf(self, x):
        x = x.detach().cpu().numpy() if torch.is_tensor(x) else x
        mu = self.mu.detach().cpu().numpy() if torch.is_tensor(self.mu) else self.mu
        sigma = self.sigma.detach().cpu().numpy() if torch.is_tensor(self.sigma) else self.sigma
        df = self.df.detach().cpu().numpy() if torch.is_tensor(self.df) else self.df
        
        # Clamp parameters
        if np.isscalar(sigma):
            sigma = max(sigma, 1e-10)
        else:
            sigma = np.clip(sigma, 1e-10, None)
        if np.isscalar(df):
            df = max(df, 1e-6)
        else:
            df = np.clip(df, 1e-6, None)
        
        return torch.tensor(scipy_t.pdf(x, df=df, loc=mu, scale=sigma), dtype=torch.float32)

    def icdf(self, q):
        """Inverse CDF (quantile function)."""
        q = q.detach().cpu().numpy() if torch.is_tensor(q) else q
        mu = self.mu.detach().cpu().numpy() if torch.is_tensor(self.mu) else self.mu
        sigma = self.sigma.detach().cpu().numpy() if torch.is_tensor(self.sigma) else self.sigma
        df = self.df.detach().cpu().numpy() if torch.is_tensor(self.df) else self.df
        
        # Clamp parameters
        if np.isscalar(sigma):
            sigma = max(sigma, 1e-10)
        else:
            sigma = np.clip(sigma, 1e-10, None)
        if np.isscalar(df):
            df = max(df, 1e-6)
        else:
            df = np.clip(df, 1e-6, None)
        
        return torch.tensor(scipy_t.ppf(q, df=df, loc=mu, scale=sigma), dtype=torch.float32)

    def expect(self):
        """Expected value (mean)."""
        return self.mu

    def interval(self, confidence=0.95):
        """Return confidence interval bounds."""
        lower = self.icdf((1 - confidence) / 2)
        upper = self.icdf((1 + confidence) / 2)
        return lower, upper

class NegativeBinomial:
    def __init__(self, p, n):
        # Clamp p to (0, 1) to avoid division by zero in scipy
        if torch.is_tensor(p):
            self.p = torch.clamp(p, min=1e-10, max=1.0 - 1e-10)
        elif isinstance(p, np.ndarray):
            self.p = np.clip(p, 1e-10, 1.0 - 1e-10)
        else:
            self.p = max(min(p, 1.0 - 1e-10), 1e-10)
        
        # Clamp n to avoid issues with zero or negative values
        if torch.is_tensor(n):
            self.n = torch.clamp(n, min=1e-10)
        elif isinstance(n, np.ndarray):
            self.n = np.clip(n, 1e-10, None)
        else:
            self.n = max(n, 1e-10)

    def mean(self):
        return (self.n * (1 - self.p)) / self.p

    def median(self):
        return self.icdf(torch.tensor(0.5))

    def mode(self):
        mode = torch.floor(((self.n - 1) * (1 - self.p)) / self.p)
        mode[mode < 0] = 0  # Mode is 0 if the computed value is negative
        return mode

    def var(self):
        return self.n * (1 - self.p) / (self.p ** 2)

    def std(self):
        return self.var().sqrt()

    def cdf(self, k):
        k = k.detach().cpu().numpy() if torch.is_tensor(k) else k
        n = self.n.detach().cpu().numpy() if torch.is_tensor(self.n) else self.n
        p = self.p.detach().cpu().numpy() if torch.is_tensor(self.p) else self.p
        # Clamp p to avoid division by zero
        if np.isscalar(p):
            p = max(min(p, 1.0 - 1e-10), 1e-10)
        else:
            p = np.clip(p, 1e-10, 1.0 - 1e-10)
        return torch.Tensor(nbinom.cdf(k, n, p))

    # def pmf(self, k):
    #     k = torch.tensor(k, dtype=torch.float32)
    #     comb = torch.lgamma(k + self.n) - torch.lgamma(k + 1) - torch.lgamma(self.n)
    #     return torch.exp(comb) * (self.p ** self.n) * ((1 - self.p) ** k)

    def pmf(self, k):
        k = k.detach().cpu().numpy() if torch.is_tensor(k) else k
        n = self.n.detach().cpu().numpy() if torch.is_tensor(self.n) else self.n
        p = self.p.detach().cpu().numpy() if torch.is_tensor(self.p) else self.p
        # Clamp p to avoid division by zero
        if np.isscalar(p):
            p = max(min(p, 1.0 - 1e-10), 1e-10)
        else:
            p = np.clip(p, 1e-10, 1.0 - 1e-10)
        return torch.Tensor(nbinom.pmf(k, n, p))

    def icdf(self, q):
        q = q.detach().cpu().numpy() if torch.is_tensor(q) else q
        n = self.n.detach().cpu().numpy() if torch.is_tensor(self.n) else self.n
        p = self.p.detach().cpu().numpy() if torch.is_tensor(self.p) else self.p
        # Clamp p to avoid division by zero
        if np.isscalar(p):
            p = max(min(p, 1.0 - 1e-10), 1e-10)
        else:
            p = np.clip(p, 1e-10, 1.0 - 1e-10)
        return torch.Tensor(nbinom.ppf(q, n, p))

    def expect(self, stat="mean"):
        if stat == "mean":
            return self.mean()
        elif stat == "mode":
            return self.mode()
        else:
            return self.median()

    def interval(self, confidence=0.95):
        lower = self.icdf(q=(1-confidence)/2)
        upper = self.icdf(q=(1+confidence)/2)
        return lower, upper

def negative_binomial_loss(y_true, n_pred, p_pred, invalid_penalty=1e6):
    """
    Numerically-stable Negative Binomial NLL (closed-form), matching PyTorch's
    NegativeBinomial(total_count=n, probs=q) log_prob when finite.

    Your convention: mean = n * (1 - p) / p
    PyTorch NB convention (with probs=q): mean = n * q / (1 - q)
    => q = 1 - p

    Key differences from negative_binomial_loss():
      - Uses closed-form log_prob (lgamma/log/log1p) instead of torch.distributions.
      - Forces computation in float32 (helpful under AMP) by casting inputs to float().
      - Does NOT replace NaN/Inf with zero (which kills gradients); instead applies a large penalty.
    """
    eps = 1e-8

    # Force FP32 for stability (especially under autocast).
    y = y_true.float()
    n = n_pred.float()
    p = p_pred.float()

    # Parameter clamps (same intent as baseline).
    p = torch.clamp(p, min=eps, max=1.0 - eps)
    n = torch.clamp(n, min=eps)

    # Map to PyTorch NB probs parameter.
    q = 1.0 - p
    q = torch.clamp(q, min=eps, max=1.0 - eps)

    # Counts should be non-negative for a meaningful likelihood.
    # (If you have negatives upstream, they should be masked out earlier.)
    y = torch.clamp(y, min=0.0)

    # Closed-form NB log pmf matching torch.distributions.NegativeBinomial.log_prob:
    # log_prob = lgamma(y+n) - lgamma(y+1) - lgamma(n) + n*log(1-q) + y*log(q)
    # Where q is PyTorch's "probs" (probability of success)
    log_prob = (
        torch.lgamma(y + n)
        - torch.lgamma(y + 1.0)
        - torch.lgamma(n)
        + n * torch.log1p(-q)
        + y * torch.log(q)
    )
    nll = -log_prob

    # Penalize non-finite values (do NOT zero them out).
    bad = ~torch.isfinite(nll)
    if bad.any():
        nll = torch.where(bad, torch.full_like(nll, float(invalid_penalty)), nll)

    return nll

def gamma_nll_loss(y_true, mu_pred, alpha_pred, reduction='mean', eps=1e-6, invalid_penalty=1e6):
    """
    Gamma Negative Log-Likelihood loss.

    Args:
        y_true: Ground truth values (>0)
        mu_pred: Predicted mean (>0)
        alpha_pred: Predicted concentration/shape (>0)
        reduction: 'none', 'mean', or 'sum'
        eps: Small constant for numerical stability
        invalid_penalty: Penalty for non-finite NLL values
    """
    # Force FP32 for stability (especially under autocast)
    y = y_true.float()
    mu = mu_pred.float()
    alpha = alpha_pred.float()

    # Clamp parameters for numerical stability
    y = torch.clamp(y, min=eps)
    mu = torch.clamp(mu, min=eps)
    alpha = torch.clamp(alpha, min=eps)

    # Convert mean/shape to rate
    beta = alpha / mu

    dist = torch.distributions.Gamma(concentration=alpha, rate=beta)
    nll = -dist.log_prob(y)

    # Penalize non-finite values (do NOT zero them out)
    bad = ~torch.isfinite(nll)
    if bad.any():
        nll = torch.where(bad, torch.full_like(nll, float(invalid_penalty)), nll)

    if reduction == 'mean':
        return nll.mean()
    elif reduction == 'sum':
        return nll.sum()
    else:
        return nll

def students_t_nll_loss(y_true, mu_pred, sigma_pred, df_pred, reduction='none', invalid_penalty=1e6):
    """
    Student's t-distribution Negative Log-Likelihood loss.
    
    Uses torch.distributions.StudentT for numerically stable log_prob computation.
    
    Args:
        y_true: Ground truth values [B, L, F] or flattened
        mu_pred: Predicted location (mean) [B, L, F]
        sigma_pred: Predicted scale (>0) [B, L, F]
        df_pred: Predicted degrees of freedom (>0) [B, L, F]
        reduction: 'none', 'mean', or 'sum'
        invalid_penalty: Penalty for non-finite NLL values
        
    Returns:
        NLL loss tensor
    """
    eps = 1e-8
    
    # Force FP32 for stability (especially under autocast)
    y = y_true.float()
    mu = mu_pred.float()
    sigma = sigma_pred.float()
    df = df_pred.float()
    
    # Clamp parameters for numerical stability
    sigma = torch.clamp(sigma, min=eps)
    df = torch.clamp(df, min=eps)
    
    # Create Student's t distribution and compute log_prob
    dist = torch.distributions.StudentT(df=df, loc=mu, scale=sigma)
    log_prob = dist.log_prob(y)
    nll = -log_prob
    
    # Penalize non-finite values (do NOT zero them out)
    bad = ~torch.isfinite(nll)
    if bad.any():
        nll = torch.where(bad, torch.full_like(nll, float(invalid_penalty)), nll)
    
    if reduction == 'mean':
        return nll.mean()
    elif reduction == 'sum':
        return nll.sum()
    else:
        return nll

random.seed(73)
def get_overlap(tup1, tup2):

    x = range(tup1[0], tup1[1])
    y = range(tup2[0], tup2[1])

    return len( range(max(x[0], y[0]), min(x[-1], y[-1])+1))    

def load_gene_coords(file, drop_negative_strand=True, drop_overlapping=True):
    gene_coords = pd.read_csv(file)
    gene_coords = gene_coords.drop(["Unnamed: 0"], axis=1)

    gene_coords["start"] = gene_coords["start"].astype("int")
    gene_coords["end"] = gene_coords["end"].astype("int")

    if drop_negative_strand:
        gene_coords = gene_coords.loc[gene_coords["strand"]=="+", :].reset_index(drop=True)
    
    if drop_overlapping:
        todrop = []
        for i in range(len(gene_coords)-1):
            if get_overlap((gene_coords["start"][i], gene_coords["end"][i]),(gene_coords["start"][i+1], gene_coords["end"][i+1])) >0:
                if (gene_coords["end"][i] - gene_coords["start"][i]) <= gene_coords["end"][i+1] - gene_coords["start"][i+1]:
                    todrop.append(i)
                else:
                    todrop.append(i+1)
        gene_coords = gene_coords.drop(todrop).reset_index(drop=True)

    return gene_coords

def signal_feature_extraction(start, end, strand, chip_seq_signal,
                              bin_size=25, margin=2000, margin_tss=None, margin_tes=None):
    """
    Extracts robust ChIP-seq signal summaries for promoter, gene body, and TES regions.

    Args:
        start: Gene start coordinate
        end: Gene end coordinate
        strand: '+' or '-'
        chip_seq_signal: 1D array of signal values
        bin_size: Resolution in bp per bin
        margin: Default margin for both TSS and TES (used if margin_tss/margin_tes not provided)
        margin_tss: Optional TSS margin (5% upstream + 5% downstream if None, uses margin)
        margin_tes: Optional TES margin (5% upstream + 5% downstream if None, uses margin)

    Returns:
        Dictionary with 12 features (4 per region: promoter, gene body, TES):
          - median signal
          - inter-quartile range (75th – 25th percentile)
          - minimum signal
          - maximum signal
    """

    # 1) Define TSS and TES by strand
    tss = start if strand == '+' else end
    tes = end   if strand == '+' else start

    # 2) Use adaptive margins if provided, otherwise use default
    if margin_tss is None:
        margin_tss = margin
    if margin_tes is None:
        margin_tes = margin

    # 3) Compute bp intervals for each region
    promoter_bp = (tss - margin_tss, tss + margin_tss)
    gene_body_bp = (start, end)
    tes_bp = (tes - margin_tes, tes + margin_tes)

    # 3) Map to bin indices (inclusive of any overlapping bin)
    def to_bins(bp_start, bp_end):
        i0 = max(bp_start // bin_size, 0)
        i1 = min(bp_end   // bin_size + 1, len(chip_seq_signal))
        return i0, i1

    p0, p1 = to_bins(*promoter_bp)
    g0, g1 = to_bins(*gene_body_bp)
    t0, t1 = to_bins(*tes_bp)

    promoter_signal   = chip_seq_signal[p0:p1]
    gene_body_signal  = chip_seq_signal[g0:g1]
    tes_region_signal = chip_seq_signal[t0:t1]

    # 4) Compute robust stats
    def stats(x):
        if x.size == 0:
            return 0.0, 0.0, 0.0, 0.0
        med = np.median(x)
        q75, q25 = np.percentile(x, [75, 25])
        iqr = q75 - q25
        mn = x.min()
        mx = x.max()
        return med, iqr, mn, mx

    prom_med, prom_iqr, prom_min, prom_max = stats(promoter_signal)
    body_med, body_iqr, body_min, body_max = stats(gene_body_signal)
    tes_med, tes_iqr, tes_min, tes_max = stats(tes_region_signal)

    # 5) Return all 12 features
    return {
        'median_sig_promoter':   prom_med,
        'iqr_sig_promoter':      prom_iqr,
        'min_sig_promoter':      prom_min,
        'max_sig_promoter':      prom_max,

        'median_sig_gene_body':  body_med,
        'iqr_sig_gene_body':     body_iqr,
        'min_sig_gene_body':     body_min,
        'max_sig_gene_body':     body_max,

        'median_sig_around_TES': tes_med,
        'iqr_sig_around_TES':    tes_iqr,
        'min_sig_around_TES':    tes_min,
        'max_sig_around_TES':    tes_max,
    }

def capture_gradients_hook(module, grad_input, grad_output):
    if hasattr(module, 'weight') and module.weight is not None:
        if grad_input[0] is not None:
            module.weight.grad_norm = grad_input[0].norm().item()
        else:
            module.weight.grad_norm = 0  # Assign a default value if grad_input[0] is None
    if hasattr(module, 'bias') and module.bias is not None:
        if len(grad_input) > 1 and grad_input[1] is not None:
            module.bias.grad_norm = grad_input[1].norm().item()
        else:
            module.bias.grad_norm = 0  # Assign a default value if grad_input[1] is None

def register_hooks(model):
    for name, module in model.named_modules():
        module.register_full_backward_hook(capture_gradients_hook)

def exponential_linspace_int(start, end, num, divisible_by=1):
    """Exponentially increasing values of integers."""
    def _round(x):
        return int(np.round(x / divisible_by) * divisible_by)
    base = np.exp(np.log(end / start) / (num - 1))
    return [_round(start * base**i) for i in range(num)]

def linear_divisible_linspace(start_size, end_size, layers):
    """Generate channel sizes where each size is divisible by the previous size."""
    sizes = [start_size]
    step = (end_size - start_size) / (layers - 1)

    for i in range(1, layers):
        # Calculate the next size
        next_size = start_size + i * step
        # Ensure the next_size is a multiple of the last size in the list
        last_size = sizes[-1]
        next_size = np.ceil(next_size / last_size) * last_size

        # Ensure not to exceed the end_size on the last step
        if i == layers - 1 and next_size != end_size:
            next_size = end_size

        sizes.append(int(next_size))

    return sizes

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def sequence_pad(data, max_length, pad_value=-1):
    # Get the original dimensions of the data
    original_size = data.size()
    
    # Create a tensor filled with the pad value with the desired size
    padded_data = torch.full((original_size[0], max_length, original_size[2]), pad_value)
    
    # Copy the original data into the padded data tensor
    padded_data[:, :original_size[1], :] = data
    
    # Create a boolean mask indicating whether each value is padded or not
    pad_mask = padded_data == pad_value
    
    return padded_data, pad_mask

def get_bin_value(input_dict):
    if input_dict["bw_obj"] == False:
        input_dict["bw"] = pyBigWig.open(input_dict["bw"])

    bw, chr, start, end, resolution = input_dict["bw"], input_dict["chr"], input_dict["start"], input_dict["end"], input_dict["resolution"]
    bin_value = bw.stats(chr, start, end, type="mean", nBins=(end - start) // resolution)

    if input_dict["bw_obj"] == False:
        bw.close()

    return bin_value

def get_bin_value_dict(input_dict):
    if input_dict["bw_obj"] == False:
        input_dict["bw"] = pyBigWig.open(input_dict["bw"])

    bw, chr, start, end, resolution = input_dict["bw"], input_dict["chr"], input_dict["start"], input_dict["end"], input_dict["resolution"]
    bin_value = bw.stats(chr, start, end, type="mean", nBins=(end - start) // resolution)

    input_dict["signals"] = bin_value

    if input_dict["bw_obj"] == False:
        bw.close()
        del input_dict["bw"]
        
    return input_dict

def add_noise(data, noise_factor):
    noise = torch.from_numpy(np.random.normal(loc=0.0, scale=1.0, size=data.shape))
    noisy_data = data + noise_factor * noise
    noisy_data = torch.clamp(noisy_data, min=0)
    return noisy_data.to(torch.float32)

def peak_overlap(y_true, y_pred, p=0.01):
    top_p_percent = int(p * len(y_true))
    
    # Get the indices of the top p percent of the observed (true) values
    top_p_percent_obs_i = np.argsort(y_true)[-top_p_percent:]
    
    # Get the indices of the top p percent of the predicted values
    top_p_percent_pred_i = np.argsort(y_pred)[-top_p_percent:]

    # Calculate the overlap
    overlap = len(np.intersect1d(top_p_percent_obs_i, top_p_percent_pred_i))

    # Calculate the percentage of overlap
    overlap_percent = overlap / top_p_percent 

    return overlap_percent
        
class COORD(object):
    def __init__(self, Meuleman_file="data/Meuleman.tsv", cCRE_file="data/GRCh38-cCREs.bed", 
                resolution=1000, chr_sizes_file="data/hg38.chrom.sizes", outdir="data/"):
        
        self.resolution = resolution
        self.cCRE_file = cCRE_file
        self.Meuleman_file = Meuleman_file
        self.outdir = outdir
        self.chr_sizes_file = chr_sizes_file    

        main_chrs = ["chr" + str(x) for x in range(1,23)] + ["chrX"]
        main_chrs.remove("chr21") # reserved for validation
        self.chr_sizes = {}

        with open(self.chr_sizes_file, 'r') as f:
            for line in f:
                chr_name, chr_size = line.strip().split('\t')
                if chr_name in main_chrs:
                    self.chr_sizes[chr_name] = int(chr_size)

    def init_bins(self):
        if os.path.exists(f"{self.outdir}/bins_{self.resolution}bp.csv"):
            self.bins = pd.read_csv(f"{self.outdir}/bins_{self.resolution}bp.csv").drop("Unnamed: 0", axis=1)
        else:

            # Create bins
            self.bins = []
            for chr, size in self.chr_sizes.items():
                start_coords = range(0, size, self.resolution)
                end_coords = range(self.resolution, size + self.resolution, self.resolution)
                self.bins.extend([[chr, start, end] for start, end in zip(start_coords, end_coords)][:-1])

            self.bins = pd.DataFrame(self.bins, columns =["chrom", "start", "end"])
            self.bins = self.bins.sort_values(["chrom", "start"]).reset_index(drop=True)
        self.bins.to_csv(f"{self.outdir}/bins_{self.resolution}bp.csv")

    def get_foreground(self):
        if os.path.exists(f'{self.outdir}/foreground_nobin.csv'):
            self.foreground = pd.read_csv(f'{self.outdir}/foreground_nobin.csv').drop("Unnamed: 0", axis=1)
        else:
            ccre = pybedtools.BedTool(self.cCRE_file)
            if self.Meuleman_file == "_":
                self.foreground = ccre.to_dataframe()

            else:
                Meuleman = pd.read_csv(self.Meuleman_file, sep="\t")
                Meuleman.columns = ["chr", "start", "end", "identifier", "mean_signal", "numsamples", "summit", "core_start", "core_end", "component"]
                Meuleman = pybedtools.BedTool.from_dataframe(Meuleman)

                # get the union of ccre and Meuleman
                self.foreground = ccre.cat(Meuleman, postmerge=False)
                self.foreground = self.foreground.to_dataframe()

            self.foreground = self.foreground[["chrom", "start", "end"]]
            self.foreground = self.foreground.sort_values(["chrom", "start"]).reset_index(drop=True)
            self.foreground.to_csv(f'{self.outdir}/foreground_nobin.csv')

    def bin_fg_bg(self):
        self.bins = pybedtools.BedTool.from_dataframe(self.bins)
        self.foreground = pybedtools.BedTool.from_dataframe(self.foreground)

        if os.path.exists(f"{self.outdir}/foreground_bins_{self.resolution}bp.csv") == False:
            # Get the subset of bins that overlap with the foreground
            self.fg_bins = self.bins.intersect(self.foreground, u=True)
            self.fg_bins = self.fg_bins.to_dataframe()
            self.fg_bins.to_csv(f"{self.outdir}/foreground_bins_{self.resolution}bp.csv")
        else:
            self.fg_bins = pd.read_csv(f"{self.outdir}/foreground_bins_{self.resolution}bp.csv").drop("Unnamed: 0", axis=1)

        if os.path.exists(f"{self.outdir}/background_bins_{self.resolution}bp.csv") == False:
            # Get the subset of bins that do not overlap with the foreground
            self.bg_bins = self.bins.intersect(self.foreground, v=True)
            self.bg_bins = self.bg_bins.to_dataframe()
            self.bg_bins.to_csv(f"{self.outdir}/background_bins_{self.resolution}bp.csv")
        else:
            self.bg_bins = pd.read_csv(f"{self.outdir}/background_bins_{self.resolution}bp.csv").drop("Unnamed: 0", axis=1)

        print(f"number of foreground bins: {len(self.fg_bins)} | number of background bins: {len(self.bg_bins)}")

version_higher = ( torch.__version__ >= "1.5.0" )
class AdaBelief(Optimizer):
    r"""Implements AdaBelief algorithm. Modified from Adam in PyTorch
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-16)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        weight_decouple (boolean, optional): ( default: True) If set as True, then
            the optimizer uses decoupled weight decay as in AdamW
        fixed_decay (boolean, optional): (default: False) This is used when weight_decouple
            is set as True.
            When fixed_decay == True, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay$.
            When fixed_decay == False, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay \times lr$. Note that in this case, the
            weight decay ratio decreases with learning rate (lr).
        rectify (boolean, optional): (default: True) If set as True, then perform the rectified
            update similar to RAdam
        degenerated_to_sgd (boolean, optional) (default:True) If set as True, then perform SGD update
            when variance of gradient is high
        print_change_log (boolean, optional) (default: True) If set as True, print the modifcation to
            default hyper-parameters
    reference: AdaBelief Optimizer, adapting stepsizes by the belief in observed gradients, NeurIPS 2020
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-16,
                 weight_decay=0, amsgrad=False, weight_decouple=True, fixed_decay=False, rectify=True,
                 degenerated_to_sgd=True, print_change_log = True):

        # ------------------------------------------------------------------------------
        # Print modifications to default arguments
        if print_change_log:
            print(Fore.RED + 'Please check your arguments if you have upgraded adabelief-pytorch from version 0.0.5.')
            print(Fore.RED + 'Modifications to default arguments:')
            default_table = tabulate([
                ['adabelief-pytorch=0.0.5','1e-8','False','False'],
                ['>=0.1.0 (Current 0.2.0)','1e-16','True','True']],
                headers=['eps','weight_decouple','rectify'])
            print(Fore.RED + default_table)

            recommend_table = tabulate([
                ['Recommended eps = 1e-8', 'Recommended eps = 1e-16'],
                ],
                headers=['SGD better than Adam (e.g. CNN for Image Classification)','Adam better than SGD (e.g. Transformer, GAN)'])
            print(Fore.BLUE + recommend_table)

            print(Fore.BLUE +'For a complete table of recommended hyperparameters, see')
            print(Fore.BLUE + 'https://github.com/juntang-zhuang/Adabelief-Optimizer')

            print(Fore.GREEN + 'You can disable the log message by setting "print_change_log = False", though it is recommended to keep as a reminder.')

            print(Style.RESET_ALL)
        # ------------------------------------------------------------------------------

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, buffer=[[None, None, None] for _ in range(10)])
        super(AdaBelief, self).__init__(params, defaults)

        self.degenerated_to_sgd = degenerated_to_sgd
        self.weight_decouple = weight_decouple
        self.rectify = rectify
        self.fixed_decay = fixed_decay
        if self.weight_decouple:
            print('Weight decoupling enabled in AdaBelief')
            if self.fixed_decay:
                print('Weight decay fixed')
        if self.rectify:
            print('Rectification enabled in AdaBelief')
        if amsgrad:
            print('AMSGrad enabled in AdaBelief')

    def __setstate__(self, state):
        super(AdaBelief, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                amsgrad = group['amsgrad']

                # State initialization
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p.data,memory_format=torch.preserve_format) \
                    if version_higher else torch.zeros_like(p.data)

                # Exponential moving average of squared gradient values
                state['exp_avg_var'] = torch.zeros_like(p.data,memory_format=torch.preserve_format) \
                    if version_higher else torch.zeros_like(p.data)

                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_var'] = torch.zeros_like(p.data,memory_format=torch.preserve_format) \
                        if version_higher else torch.zeros_like(p.data)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # cast data type
                half_precision = False
                if p.data.dtype == torch.float16:
                    half_precision = True
                    p.data = p.data.float()
                    p.grad = p.grad.float()

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'AdaBelief does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                beta1, beta2 = group['betas']

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data,memory_format=torch.preserve_format) \
                        if version_higher else torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_var'] = torch.zeros_like(p.data,memory_format=torch.preserve_format) \
                        if version_higher else torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_var'] = torch.zeros_like(p.data,memory_format=torch.preserve_format) \
                            if version_higher else torch.zeros_like(p.data)
                
                # perform weight decay, check if decoupled weight decay
                if self.weight_decouple:
                    if not self.fixed_decay:
                        p.data.mul_(1.0 - group['lr'] * group['weight_decay'])
                    else:
                        p.data.mul_(1.0 - group['weight_decay'])
                else:
                    if group['weight_decay'] != 0:
                        grad.add_(p.data, alpha=group['weight_decay'])

                # get current state variable
                exp_avg, exp_avg_var = state['exp_avg'], state['exp_avg_var']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Update first and second moment running average
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                grad_residual = grad - exp_avg
                exp_avg_var.mul_(beta2).addcmul_( grad_residual, grad_residual, value=1 - beta2)

                if amsgrad:
                    max_exp_avg_var = state['max_exp_avg_var']
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_var, exp_avg_var.add_(group['eps']), out=max_exp_avg_var)

                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_var.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_var.add_(group['eps']).sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                
                # update
                if not self.rectify:
                    # Default update
                    step_size = group['lr'] / bias_correction1
                    p.data.addcdiv_( exp_avg, denom, value=-step_size)

                else:  # Rectified update, forked from RAdam
                    buffered = group['buffer'][int(state['step'] % 10)]
                    if state['step'] == buffered[0]:
                        N_sma, step_size = buffered[1], buffered[2]
                    else:
                        buffered[0] = state['step']
                        beta2_t = beta2 ** state['step']
                        N_sma_max = 2 / (1 - beta2) - 1
                        N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                        buffered[1] = N_sma

                        # more conservative since it's an approximated value
                        if N_sma >= 5:
                            step_size = math.sqrt(
                                (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                        N_sma_max - 2)) / (1 - beta1 ** state['step'])
                        elif self.degenerated_to_sgd:
                            step_size = 1.0 / (1 - beta1 ** state['step'])
                        else:
                            step_size = -1
                        buffered[2] = step_size

                    if N_sma >= 5:
                        denom = exp_avg_var.sqrt().add_(group['eps'])
                        p.data.addcdiv_(exp_avg, denom, value=-step_size * group['lr'])
                    elif step_size > 0:
                        p.data.add_( exp_avg, alpha=-step_size * group['lr'])
                
                if half_precision:
                    p.data = p.data.half()
                    p.grad = p.grad.half() 

        return loss