"""CANDI v2 — end-to-end reconstruction model.

Composes V2Encoder + V2Decoder into a single nn.Module.  The forward() method
returns a dict (not a tuple) for clarity and extensibility.

Usage:
    cfg = CANDIv2Config(...)
    model = CANDIv2(cfg).to(device)
    out = model(x_data, x_dna, x_meta, y_meta)
    # out['p'], out['n'], out['peak'], out['z'], ...
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from sandbox.autoresearch.june3.candi_v2.config import CANDIv2Config
from sandbox.autoresearch.june3.candi_v2.encoder import V2Encoder
from sandbox.autoresearch.june3.candi_v2.decoder import V2Decoder


class CANDIv2(nn.Module):
    """First-principles CANDI model: fresh encoder + configurable decoder.

    Inputs:
      x_data:  [B, L, A+1] — signal + control channel (raw counts)
      x_dna:   [B, 4, G]   — one-hot DNA sequence
      x_meta:  [B, 4, A+1] — input metadata (may be masked)
      y_meta:  [B, 4, A]   — target metadata (signal assays only, no control)

    Outputs (dict):
      p:    [B, L, A] — NB probability (or None)
      n:    [B, L, A] — NB dispersion (or None)
      peak: [B, L, A] — peak probability (or None)
      mu:   [B, L, A] — Gaussian mean (or None, only when heads='all')
      var:  [B, L, A] — Gaussian variance (or None, only when heads='all')
      z:    [B, L2, d_model] — encoder latent (detached, for analysis)
    """

    def __init__(self, cfg: CANDIv2Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.signal_dim = int(cfg.encoder.num_assays)

        # Build encoder
        self.encoder = V2Encoder(cfg.encoder)

        # Build decoder
        self.decoder = V2Decoder(
            cfg.decoder,
            encoder_d_model=self.encoder.d_model,
            signal_dim=self.signal_dim,
        )

    def forward(
        self,
        x_data: torch.Tensor,
        x_dna: torch.Tensor,
        x_meta: torch.Tensor,
        y_meta: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        query_mask_signal: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            x_data: [B, L, A+1] — signal + control (may contain MISSING/CLOZE)
            x_dna:  [B, 4, G]   — one-hot DNA
            x_meta: [B, 4, A+1] — input metadata (includes control column)
            y_meta: [B, 4, A]   — target metadata (signal assays only)
            query_mask: unused, kept for train.py interface compatibility
            query_mask_signal: unused, kept for train.py interface compatibility

        Returns:
            dict with keys: p, n, peak, mu, var, z
        """
        del query_mask, query_mask_signal

        # Encode
        z = self.encoder.encode(x_data, x_dna, x_meta, return_meta=False)

        # Decode
        out = self.decoder(z, y_meta)

        # Attach encoder latent for downstream analysis
        out["z"] = z.detach()

        return out

    def forward_tuple(
        self,
        x_data: torch.Tensor,
        x_dna: torch.Tensor,
        x_meta: torch.Tensor,
        y_meta: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        query_mask_signal: Optional[torch.Tensor] = None,
    ):
        """Tuple-based forward for compatibility with existing loss/eval code.

        Returns: (p, n, mu, var, df, peak) — same 6-tuple as production CANDI.
        Inactive heads return zero tensors (not None) for loss compatibility.
        """
        out = self.forward(x_data, x_dna, x_meta, y_meta,
                           query_mask, query_mask_signal)
        B, L = x_data.shape[0], x_data.shape[1]
        A = self.signal_dim
        device = x_data.device
        dtype = x_data.dtype

        def _or_zeros(key: str) -> torch.Tensor:
            v = out.get(key)
            if v is not None:
                return v
            return torch.zeros(B, L, A, device=device, dtype=dtype)

        p = _or_zeros("p")
        n = _or_zeros("n")
        mu = _or_zeros("mu")
        var = _or_zeros("var")
        peak = _or_zeros("peak")
        df = None  # no df in v2
        return p, n, mu, var, df, peak
