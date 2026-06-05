"""Synthetic dataset generator for CANDI v2 overfit validation."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from sandbox.batch import CLOZE, MISSING


@dataclass
class SyntheticDataConfig:
    """Controls synthetic ground-truth generation."""

    batch_size: int = 4
    num_assays: int = 4
    context_length: int = 768
    dna_length: int = 19200  # L * 25 bp
    seed: int = 42

    # Per-sample log2 depth multiplier (counts = count_amplitude * 2^depth_log2 * pattern).
    depth_log2: Tuple[float, ...] = (
        math.log2(5.0), math.log2(10.0), math.log2(20.0), math.log2(40.0),
    )
    read_length: float = 100.0
    run_type: int = 0

    # Scales base_pattern [0,1] to realistic count range before depth multiplier.
    count_amplitude: float = 50.0

    # Ground-truth generation
    stochastic: bool = False
    nb_dispersion: float = 50.0  # larger n -> closer to Poisson

    # Masking: number of assays to mask per sample (length = batch_size)
    mask_counts: Tuple[int, ...] = (0, 0, 0, 0)

    # Denoising: DSF=2 binomial subsampling on input
    dsf: int = 1
    stochastic_dsf: bool = False  # Poisson approx for random subsampling

    # DNA: "constant" (zeros) or "motif" (position-dependent)
    dna_mode: str = "constant"

    # If False, counts are constant along L per assay (easier P1 sanity check).
    spatial_pattern: bool = True

    # Control channel scale (always observed)
    control_scale: float = 5.0


def _make_base_patterns(
    num_assays: int, length: int, device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """Per-assay spatial profiles in [0, 1], different sine frequencies."""
    pos = torch.arange(length, device=device, dtype=dtype)
    patterns = []
    for a in range(num_assays):
        freq = 1.0 + 0.5 * a
        wave = 0.5 + 0.5 * torch.sin(2.0 * math.pi * freq * pos / float(length))
        patterns.append(wave)
    return torch.stack(patterns, dim=0)  # [A, L]


def _make_dna(cfg: SyntheticDataConfig, device: torch.device) -> torch.Tensor:
    """One-hot DNA [B, 4, G]."""
    b, g = cfg.batch_size, cfg.dna_length
    if cfg.dna_mode == "constant":
        dna = torch.zeros(b, 4, g, device=device)
        dna[:, 0, :] = 1.0  # all-A stretch
        return dna

    # Motif mode: periodic GC-rich vs AT-rich blocks modulate signal in phase 5
    dna = torch.zeros(b, 4, g, device=device)
    period = 25
    for bidx in range(b):
        for pos in range(g):
            block = (pos // period) % 2
            if block == 0:
                dna[bidx, 1, pos] = 0.5  # C
                dna[bidx, 2, pos] = 0.5  # G
            else:
                dna[bidx, 0, pos] = 0.5  # A
                dna[bidx, 3, pos] = 0.5  # T
    return dna


def _build_metadata(
    cfg: SyntheticDataConfig,
    device: torch.device,
    dtype: torch.dtype,
    *,
    include_control: bool,
) -> torch.Tensor:
    """Metadata tensor [B, 4, A(+1)]. Rows: depth_log2, assay_id, read_length, run_type."""
    b, a = cfg.batch_size, cfg.num_assays
    n_cols = a + 1 if include_control else a
    meta = torch.zeros(b, 4, n_cols, device=device, dtype=dtype)
    depths = list(cfg.depth_log2)
    if len(depths) < b:
        depths = depths + [depths[-1]] * (b - len(depths))
    for bi in range(b):
        d = float(depths[bi])
        for ai in range(n_cols):
            if include_control and ai == a:
                meta[bi, 0, ai] = d
                meta[bi, 1, ai] = float(a)  # control id
            else:
                meta[bi, 0, ai] = d
                meta[bi, 1, ai] = float(ai)
            meta[bi, 2, ai] = cfg.read_length
            meta[bi, 3, ai] = float(cfg.run_type)
    return meta


def _counts_from_depth(
    cfg: SyntheticDataConfig,
    base_pattern: torch.Tensor,
    depth_log2: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Clean counts [B, L, A] = count_amplitude * 2^depth * base_pattern."""
    b, a, l = cfg.batch_size, cfg.num_assays, cfg.context_length
    scale = cfg.count_amplitude * torch.pow(2.0, depth_log2).view(b, 1, 1)
    mu = scale * base_pattern.T.unsqueeze(0)
    if not cfg.stochastic:
        return mu.round().clamp(min=0.0)

    n = torch.full_like(mu, cfg.nb_dispersion)
    p = n / (n + mu + 1e-6)
    p = p.clamp(1e-4, 1.0 - 1e-4)
    dist = torch.distributions.NegativeBinomial(total_count=n, probs=1.0 - p)
    return dist.sample().float()


def _apply_dsf(counts: torch.Tensor, dsf: int, *, stochastic: bool = False) -> torch.Tensor:
    """Subsample counts by DSF (deterministic floor or Poisson approx)."""
    if dsf <= 1:
        return counts
    if stochastic:
        lam = (counts / float(dsf)).clamp(min=0.0)
        return torch.poisson(lam).float()
    return (counts / float(dsf)).floor().clamp(min=0.0)


def _apply_masking(
    x_data: torch.Tensor,
    x_meta: torch.Tensor,
    mask_counts: List[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply full-assay CLOZE masking.

    Returns x_data_m, x_meta_m, observed_map, masked_map (signal assays only).
    """
    b, l, a = x_data.shape
    x_data_m = x_data.clone()
    x_meta_m = x_meta.clone()
    observed = torch.ones(b, l, a, dtype=torch.bool, device=x_data.device)
    masked = torch.zeros(b, l, a, dtype=torch.bool, device=x_data.device)

    for bi in range(b):
        n_mask = int(mask_counts[bi]) if bi < len(mask_counts) else 0
        n_mask = min(n_mask, a)
        if n_mask <= 0:
            continue
        assay_idx = list(range(n_mask))  # mask first n assays
        for ai in assay_idx:
            x_data_m[bi, :, ai] = CLOZE
            x_meta_m[bi, :, ai] = CLOZE
            observed[bi, :, ai] = False
            masked[bi, :, ai] = True
    return x_data_m, x_meta_m, observed, masked


def generate_synthetic_batch(
    cfg: SyntheticDataConfig,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Build a single batch dict compatible with the overfit harness.

    Returns tensors for model input, loss masks, and ground truth.
    """
    torch.manual_seed(cfg.seed)
    b, a, l = cfg.batch_size, cfg.num_assays, cfg.context_length
    dtype = torch.float32

    base = _make_base_patterns(a, l, device, dtype)  # [A, L]
    if not cfg.spatial_pattern:
        # Flat profile per assay (assay-specific level, constant along L)
        assay_level = torch.linspace(0.5, 1.0, a, device=device, dtype=dtype)
        base = assay_level.unsqueeze(1).expand(a, l).clone()
    if cfg.dna_mode == "motif":
        # DNA-modulated amplitude: GC blocks boost counts
        pos = torch.arange(l, device=device, dtype=dtype)
        dna_mod = 1.0 + 0.3 * torch.sin(2.0 * math.pi * pos / 25.0)
        base = base * dna_mod.unsqueeze(0)

    depths = list(cfg.depth_log2)
    if len(depths) < b:
        depths = depths + [depths[-1]] * (b - len(depths))
    depth_t = torch.tensor(depths[:b], device=device, dtype=dtype)

    y_data = _counts_from_depth(cfg, base, depth_t, device, dtype)  # [B,L,A]
    y_meta = _build_metadata(cfg, device, dtype, include_control=False)
    control_data = torch.full(
        (b, l, 1), cfg.control_scale, device=device, dtype=dtype,
    )

    x_data_clean = y_data.clone()
    x_meta_full = _build_metadata(cfg, device, dtype, include_control=True)

    # Input may be DSF-corrupted
    x_data_input = _apply_dsf(x_data_clean, cfg.dsf, stochastic=cfg.stochastic_dsf)

    mask_list = list(cfg.mask_counts)
    if len(mask_list) < b:
        mask_list = mask_list + [0] * (b - len(mask_list))

    x_data_m, x_meta_m, observed, masked = _apply_masking(
        x_data_input, x_meta_full, mask_list,
    )

    x_data_in = torch.cat([x_data_m, control_data], dim=2)
    # x_meta already includes control column from x_meta_full / x_meta_m
    x_meta_in = x_meta_m
    x_dna = _make_dna(cfg, device)

    y_avail = torch.ones(b, a, device=device, dtype=dtype)
    y_pval = torch.zeros(b, l, a, device=device, dtype=dtype)
    y_peaks = torch.zeros(b, l, a, device=device, dtype=dtype)

    signal_observed = observed.clone()
    signal_masked = masked.clone()
    if cfg.dsf > 1:
        sig_ok = torch.ones_like(observed)
        signal_observed = observed & sig_ok
        signal_masked = masked & sig_ok

    query_mask = y_avail > 0

    return {
        "x_data": x_data_in,
        "x_dna": x_dna,
        "x_meta": x_meta_in,
        "y_meta": y_meta,
        "y_data": y_data,
        "y_pval": y_pval,
        "y_peaks": y_peaks,
        "observed_map": observed,
        "masked_map": masked,
        "signal_observed_map": signal_observed,
        "signal_masked_map": signal_masked,
        "query_mask": query_mask,
        "depth_log2": depth_t,
    }


def make_data_config(
    phase: str,
    *,
    context_length: Optional[int] = None,
    spatial_pattern: Optional[bool] = None,
    stochastic: Optional[bool] = None,
    stochastic_dsf: Optional[bool] = None,
    mask_counts: Optional[Tuple[int, ...]] = None,
    production_depth: bool = False,
) -> SyntheticDataConfig:
    """Phase preset with optional overrides."""
    cfg = phase_config(phase)
    if context_length is not None:
        cfg.context_length = context_length
        cfg.dna_length = context_length * 25
    if spatial_pattern is not None:
        cfg.spatial_pattern = spatial_pattern
    if stochastic is not None:
        cfg.stochastic = stochastic
    if stochastic_dsf is not None:
        cfg.stochastic_dsf = stochastic_dsf
    if mask_counts is not None:
        cfg.mask_counts = mask_counts
    if production_depth:
        cfg.depth_log2 = (22.0, 23.0, 24.0, 25.0)
        cfg.count_amplitude = 500.0 / (2.0 ** 22.0)
    return cfg


def phase_config(phase: str) -> SyntheticDataConfig:
    """Preset configs for validation phases P1–P5."""
    if phase == "p1":
        d = math.log2(10.0)
        return SyntheticDataConfig(
            context_length=96,
            dna_length=2400,
            depth_log2=(d, d, d, d),
            mask_counts=(0, 0, 0, 0),
            dsf=1,
            dna_mode="constant",
            spatial_pattern=False,
            stochastic=False,
        )
    if phase == "p2":
        return SyntheticDataConfig(
            context_length=96,
            dna_length=2400,
            spatial_pattern=False,
            depth_log2=(
                math.log2(5.0), math.log2(10.0), math.log2(20.0), math.log2(40.0),
            ),
            mask_counts=(0, 0, 0, 0),
            dsf=1,
            dna_mode="constant",
            stochastic=False,
        )
    if phase == "p3":
        return SyntheticDataConfig(
            context_length=96,
            dna_length=2400,
            spatial_pattern=False,
            depth_log2=(
                math.log2(5.0), math.log2(10.0), math.log2(20.0), math.log2(40.0),
            ),
            mask_counts=(1, 1, 2, 2),
            dsf=1,
            dna_mode="constant",
            stochastic=False,
        )
    if phase == "p4":
        return SyntheticDataConfig(
            context_length=96,
            dna_length=2400,
            spatial_pattern=False,
            depth_log2=(
                math.log2(5.0), math.log2(10.0), math.log2(20.0), math.log2(40.0),
            ),
            mask_counts=(1, 1, 2, 2),
            dsf=2,
            dna_mode="constant",
            stochastic=False,
        )
    if phase == "p5":
        return SyntheticDataConfig(
            context_length=96,
            dna_length=2400,
            spatial_pattern=False,
            depth_log2=(
                math.log2(5.0), math.log2(10.0), math.log2(20.0), math.log2(40.0),
            ),
            mask_counts=(1, 1, 2, 2),
            dsf=2,
            dna_mode="motif",
            stochastic=False,
        )
    raise ValueError(f"Unknown phase={phase}")
