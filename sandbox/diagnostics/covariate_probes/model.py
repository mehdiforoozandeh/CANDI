"""Simplified CANDI encoder for covariate probes: count tower (+ optional DNA tower) -> transformer -> head.
Post-conv sequence is downsampled to SEQ=192 so self-attention memory/cost is ~16x lower than at 768."""
import math
import torch
import torch.nn as nn

WIN_BP = 19200      # DNA bp per window
SEQ = 192           # transformer sequence length (768 count bins downsampled x4)
D_MODEL = 128

def _sinusoidal(n, d):
    pe = torch.zeros(n, d)
    pos = torch.arange(n).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
    return pe

class CovariateProbe(nn.Module):
    def __init__(self, use_dna: bool):
        super().__init__()
        self.use_dna = use_dna
        self.count_tower = nn.Sequential(                          # 768 -> 384 -> 192
            nn.Conv1d(1, 64, 7, stride=2, padding=3), nn.GELU(),
            nn.Conv1d(64, 128, 7, stride=2, padding=3), nn.GELU())
        if use_dna:
            self.dna_tower = nn.Sequential(                        # 19200 -> 3840 -> 768 -> 192
                nn.Conv1d(4, 64, 11, stride=5, padding=3), nn.GELU(),
                nn.Conv1d(64, 96, 11, stride=5, padding=3), nn.GELU(),
                nn.Conv1d(96, 128, 7, stride=4, padding=2), nn.GELU())
        self.fuse = nn.Conv1d(256 if use_dna else 128, D_MODEL, 1)
        self.register_buffer("pe", _sinusoidal(SEQ, D_MODEL))
        layer = nn.TransformerEncoderLayer(D_MODEL, nhead=4, dim_feedforward=256,
                                           batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(D_MODEL, 1)

    def forward(self, count, dna=None):
        h = self.count_tower(count)                       # [B,128,192]
        if self.use_dna:
            h = torch.cat([h, self.dna_tower(dna)], dim=1)  # [B,256,192]
        h = self.fuse(h).transpose(1, 2)                  # [B,192,128]
        h = self.transformer(h + self.pe)
        return self.head(h.mean(1)).squeeze(-1)           # [B]
