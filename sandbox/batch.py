"""Masking + tensor assembly for sandbox training (mirrors train.py::_process_batch core)."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from _utils import DataMasker


MISSING = -1
CLOZE = -2


def make_masker(
    *,
    p_full_loci: float = 0.0,
    p_full_assay: float = 1.0,
    p_chunks: float = 0.0,
    mask_fraction: float = 0.2,
    chunk_size: int = 40,
) -> DataMasker:
    return DataMasker(
        mask_value=CLOZE,
        chunk_size=chunk_size,
        mask_fraction=mask_fraction,
        p_full_loci=p_full_loci,
        p_full_assay=p_full_assay,
        p_chunks=p_chunks,
    )


def prepare_masked_batch(
    batch: Dict[str, torch.Tensor],
    masker: DataMasker,
    device: torch.device,
    *,
    apply_mask: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Returns model inputs on `device`, loss masks, and GT tensors.
    Skips batches with no valid queries (mirrors production k_b==0 handling).
    """
    x_data = batch["x_data"].to(device)
    x_meta = batch["x_meta"].to(device)
    x_avail = batch["x_avail"].to(device)
    x_dna = batch["x_dna"].to(device)
    y_data = batch["y_data"].to(device)
    y_meta = batch["y_meta"].to(device)
    y_avail = batch["y_avail"].to(device)
    y_pval = batch["y_pval"].to(device)
    y_peaks = batch["y_peaks"].to(device)
    control_data = batch["control_data"].to(device)
    control_meta = batch["control_meta"].to(device)
    control_avail = batch["control_avail"].to(device)
    y_dsf = batch.get("y_dsf")
    if y_dsf is not None:
        y_dsf = y_dsf.to(device)

    B, L, F = x_data.shape

    x_data_m = x_data.clone()
    x_meta_m = x_meta.clone()
    x_avail_m = x_avail.clone()
    if apply_mask:
        x_data_m, x_meta_m, x_avail_m = masker.apply_mask(x_data_m, x_meta_m, x_avail_m)

    masked_map = x_data_m == CLOZE
    observed_map = (x_data_m != MISSING) & (x_data_m != CLOZE)
    observed_map = observed_map.clone()
    masked_map = masked_map.clone()

    y_avail_bool = y_avail > 0
    Lm = x_data_m.shape[1]
    y_expand = y_avail_bool.unsqueeze(1).expand(-1, Lm, -1)
    observed_map = observed_map & y_expand
    masked_map = masked_map & y_expand

    sup_any = observed_map.any(dim=1) | masked_map.any(dim=1)
    if not torch.equal(y_avail_bool, sup_any):
        mismatch = int((y_avail_bool != sup_any).sum().item())
        raise ValueError(
            f"Availability/supervision mismatch ({mismatch} assay positions): "
            "(y_avail>0) != (observed_any || masked_any). Data pipeline bug — abort."
        )

    query_mask = y_avail_bool & sup_any
    if y_dsf is None:
        raise ValueError("sandbox batch requires y_dsf tensor (set by SandboxH5Dataset); prod uses None only in legacy paths.")
    query_mask_signal = query_mask & (y_dsf == 1)

    valid_samples = query_mask.any(dim=1)
    if not valid_samples.any():
        return None
    if (~valid_samples).any():
        sel = valid_samples
        x_data_m = x_data_m[sel]
        x_meta_m = x_meta_m[sel]
        x_avail_m = x_avail_m[sel]
        x_dna = x_dna[sel]
        y_data = y_data[sel]
        y_meta = y_meta[sel]
        y_avail = y_avail[sel]
        y_pval = y_pval[sel]
        y_peaks = y_peaks[sel]
        control_data = control_data[sel]
        control_meta = control_meta[sel]
        control_avail = control_avail[sel]
        observed_map = observed_map[sel]
        masked_map = masked_map[sel]
        query_mask = query_mask[sel]
        query_mask_signal = query_mask_signal[sel]
        if y_dsf is not None:
            y_dsf = y_dsf[sel]

    if not observed_map.any():
        return None

    has_masked = masked_map.any()

    x_data_in = torch.cat([x_data_m, control_data], dim=2)
    x_meta_in = torch.cat([x_meta_m, control_meta], dim=2)
    x_avail_in = torch.cat([x_avail_m, control_avail], dim=1)

    signal_observed_map = observed_map
    signal_masked_map = masked_map
    if y_dsf is not None:
        sig_ok = (y_dsf == 1) & (y_avail > 0)
        sig_ok = sig_ok.unsqueeze(1).expand_as(observed_map)
        signal_observed_map = observed_map & sig_ok
        signal_masked_map = masked_map & sig_ok

    return {
        "x_data": x_data_in,
        "x_dna": x_dna,
        "x_meta": x_meta_in,
        "y_meta": y_meta,
        "query_mask": query_mask,
        "query_mask_signal": query_mask_signal,
        "observed_map": observed_map,
        "masked_map": masked_map,
        "signal_observed_map": signal_observed_map,
        "signal_masked_map": signal_masked_map,
        "y_data": y_data,
        "y_pval": y_pval,
        "y_peaks": y_peaks,
        "has_masked_regions": has_masked,
    }
