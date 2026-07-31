"""Vendored, unmodified third-party-to-this-kit code.

Provenance:
  DataMasker               VERBATIM from EpiDenoise/_utils.py:36-353
  exponential_linspace_int VERBATIM from EpiDenoise/_utils.py:1850-1855
  MISSING / CLOZE          from EpiDenoise/sandbox/batch.py:11-12

Do not edit the bodies below. DataMasker.apply_mask draws from the global torch RNG on
every call, so any change to it shifts the whole masking sequence of a training run.
"""
import numpy as np
import torch


MISSING = -1
CLOZE = -2


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


def exponential_linspace_int(start, end, num, divisible_by=1):
    """Exponentially increasing values of integers."""
    def _round(x):
        return int(np.round(x / divisible_by) * divisible_by)
    base = np.exp(np.log(end / start) / (num - 1))
    return [_round(start * base**i) for i in range(num)]
