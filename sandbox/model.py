"""Sandbox-sized `CANDI` factory (fixed decoder only)."""
from __future__ import annotations

from model import CANDI


def build_sandbox_candi(
    *,
    context_bins: int = 768,
    signal_dim: int = 8,
    metadata_embedding_dim: int = 32,
    n_cnn_layers: int = 3,
    expansion_factor: int = 2,
    nhead: int = 4,
    n_sab_layers: int = 2,
    pool_size: int = 2,
    dropout: float = 0.1,
    separate_decoders: bool = True,
    mask_stem: bool = True,
    dist_type: str = "gaussian",
    signal_transform: str = "none",
    attention_type: str = "dual",
) -> CANDI:
    """
    `context_bins` is the number of signal bins (= CANDI `context_length`, bin-level).
    The encoder expects DNA at 25x this length (handled upstream by data loaders).
    Must be divisible by `pool_size**n_cnn_layers`.
    """
    pool_factor = int(pool_size) ** int(n_cnn_layers)
    if int(context_bins) % pool_factor != 0:
        raise ValueError(
            f"context_bins ({context_bins}) must be divisible by pool_size**n_cnn_layers={pool_factor}"
        )
    if dist_type == "student_t":
        dist_type = "studentst"
    return CANDI(
        signal_dim=signal_dim,
        metadata_embedding_dim=metadata_embedding_dim,
        conv_kernel_size=3,
        n_cnn_layers=n_cnn_layers,
        nhead=nhead,
        n_sab_layers=n_sab_layers,
        pool_size=pool_size,
        dropout=dropout,
        context_length=int(context_bins),
        expansion_factor=expansion_factor,
        separate_decoders=separate_decoders,
        num_assays=signal_dim,
        num_runtypes=2,
        norm="layer",
        attention_type=attention_type,
        output_ff=False,
        dist_type=dist_type,
        xl_dna=False,
        mask_stem=mask_stem,
        signal_transform=signal_transform,
        decoder_type="fixed",
        enable_latent_kl=False,
    )
