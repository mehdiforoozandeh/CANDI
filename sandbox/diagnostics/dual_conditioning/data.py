"""Data layer for the dual-conditioning testbed (crux q15).

Reads base counts / DNA / control from `sandbox/data/sandbox.h5` (DSF1 only, denoising-only),
applies per-assay input/output transforms f_x / f_y, and assembles CANDIv2-ready tensors with the
2-row (aug_family, aug_param) metadata. chr19 -> train, chr21 -> test.

Design choices (see plan.md):
- x_data carries RAW transformed counts x' (>=0 int, as float); missing assays -> MISSING(-1).
  The encoder's own signal_transform="arcsinh" handles compression and preserves the -1 sentinel.
- Metadata stores the RAW positive param; normalization (h33: none/zscore/log) happens INSIDE the
  model embedder, so a normalized param can never collide with the -1/-2 availability sentinels.
- `thin` targets are deterministically seeded by (biosample, assay, window_start, family, param, side)
  so they are bit-identical across epochs.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from sandbox import SANDBOX_ASSAYS
from sandbox.diagnostics.dual_conditioning import transforms as T

H5_DEFAULT = "sandbox/data/sandbox.h5"
A = len(SANDBOX_ASSAYS)       # 8 signal assays
L = 768                       # context bins
G = L * 25                    # 19200 DNA bp
MISSING = -1.0
# Winsorize base counts: the raw distribution is heavy-tailed (mean ~3, but max ~10k), and transforms
# like power blow that tail up to astronomically large NB targets that diverge NBNLL. Clip at p99.9-ish
# so the transform algebra is learnable while keeping >99.8% of positions exact.
CLIP = 128
_SEED_SALT = 0x9E3779B1


def _seed(bio: str, assay: int, win_start: int, fam: int, par: float, side: int) -> int:
    """Stable 32-bit seed for deterministic binomial thinning."""
    h = hash((bio, int(assay), int(win_start), int(fam), round(float(par), 6), int(side))) & 0xFFFFFFFF
    return (h ^ _SEED_SALT) & 0xFFFFFFFF


class DualCondData:
    """H5-backed base-count provider + transform/tensor assembler."""

    def __init__(self, h5_path: str = H5_DEFAULT, *, ram_cache: bool = True):
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(self.h5_path)
        self._buf: Optional[bytes] = None
        if ram_cache and self.h5_path.stat().st_size <= 4 * 1024**3:
            self._buf = self.h5_path.read_bytes()
        import h5py
        with self._open() as h5:
            self.bios_order: List[str] = json.loads(h5["biosamples"].attrs["order"])
            chrom = np.array([c.decode() if isinstance(c, bytes) else str(c)
                              for c in h5["windows/chrom"][:]])
            self.win_start = np.array(h5["windows/start"][:], np.int64)
            self.idx_chr19 = np.where(chrom == "chr19")[0]
            self.idx_chr21 = np.where(chrom == "chr21")[0]
            # availability census: meta_dsf1[0, a] != -1 -> assay a present in this biosample
            self.avail: Dict[str, np.ndarray] = {}
            for b in self.bios_order:
                m = np.array(h5["biosamples"][b.replace("/", "_")]["meta_dsf1"])  # [4, A]
                self.avail[b] = (m[0] != -1).astype(np.float32)
        # biosamples with >=1 available assay, and the (bio, assay) instance list
        self.biosamples = [b for b in self.bios_order if self.avail[b].sum() > 0]
        self.instances = [(b, a) for b in self.biosamples
                          for a in range(A) if self.avail[b][a] > 0]

    def _open(self):
        import h5py
        if self._buf is not None:
            return h5py.File(io.BytesIO(self._buf), "r")
        return h5py.File(self.h5_path, "r")

    # ---- raw reads ----
    def _read(self, h5, bio: str, wi: np.ndarray):
        g = h5["biosamples"][bio.replace("/", "_")]
        wi_sorted = np.sort(wi)                      # h5 fancy-indexing needs increasing order
        counts = np.array(g["counts_dsf1"][wi_sorted]).astype(np.int64)   # [B, L, A]
        dna = np.array(g["dna"][wi_sorted]).astype(np.float32)            # [B, G, 4]
        control = np.array(g["control"][wi_sorted]).astype(np.float32)    # [B, L, 1]
        return wi_sorted, counts, dna, control

    def _apply_side(self, base_col: np.ndarray, fam: int, par: float,
                    bio: str, assay: int, win_starts: np.ndarray, side: int) -> np.ndarray:
        """Apply one transform to base_col [B, L] for a single assay (per-window thin seeding)."""
        if T.FAM_NAMES[int(fam)] != "thin":
            return T.apply_transform(base_col, fam, par)
        out = np.empty_like(base_col)
        for i in range(base_col.shape[0]):
            out[i] = T.apply_transform(base_col[i], "thin", par,
                                       seed=_seed(bio, assay, int(win_starts[i]), fam, par, side))
        return out

    def make_batch(self, bio: str, wi: np.ndarray,
                   fam_x: np.ndarray, par_x: np.ndarray,
                   fam_y: np.ndarray, par_y: np.ndarray,
                   device, *,
                   fam_xm: np.ndarray | None = None, par_xm: np.ndarray | None = None,
                   fam_ym: np.ndarray | None = None, par_ym: np.ndarray | None = None,
                   ) -> Dict[str, torch.Tensor]:
        """Assemble one batch. fam_*/par_* are per-assay [A] (unavailable assays ignored).

        The transform APPLIED to the data uses (fam_x,par_x)/(fam_y,par_y); the metadata COVARIATE the
        model reads defaults to the same but can be overridden via fam_xm/par_xm / fam_ym/par_ym. The
        override decouples covariate from target -> used by the shuffle leakage controls (wrong covariate).
        """
        fam_xm = fam_x if fam_xm is None else fam_xm; par_xm = par_x if par_xm is None else par_xm
        fam_ym = fam_y if fam_ym is None else fam_ym; par_ym = par_y if par_ym is None else par_ym
        with self._open() as h5:
            wsorted, counts, dna, control = self._read(h5, bio, wi)
        B = counts.shape[0]
        av = self.avail[bio]                                   # [A]

        x_sig = np.full((B, L, A), MISSING, np.float32)
        y_tgt = np.zeros((B, L, A), np.float32)
        x_meta = np.full((B, 2, A + 1), MISSING, np.float32)
        y_meta = np.full((B, 2, A), MISSING, np.float32)
        avail = np.zeros((B, A), np.float32)

        for a in range(A):
            if av[a] <= 0:
                continue
            base_col = np.clip(counts[:, :, a], 0, CLIP)       # [B, L], winsorize heavy tail
            xp = self._apply_side(base_col, int(fam_x[a]), float(par_x[a]), bio, a, wsorted, side=0)
            yp = self._apply_side(base_col, int(fam_y[a]), float(par_y[a]), bio, a, wsorted, side=1)
            x_sig[:, :, a] = xp.astype(np.float32)
            y_tgt[:, :, a] = yp.astype(np.float32)
            x_meta[:, 0, a] = float(fam_xm[a]); x_meta[:, 1, a] = float(par_xm[a])
            y_meta[:, 0, a] = float(fam_ym[a]); y_meta[:, 1, a] = float(par_ym[a])
            avail[:, a] = 1.0

        # control channel (index A): inert identity metadata where present; the h5 stores -1 for
        # absent control, so mark those samples missing to keep signal/meta availability consistent
        # (candi_v2 mask_token asserts they agree).
        x_data = np.concatenate([x_sig, control], axis=2)      # [B, L, A+1]
        ctrl_missing = (control[:, :, 0] == MISSING).any(axis=1)     # [B]
        x_meta[:, 0, A] = float(T.FAM["identity"]); x_meta[:, 1, A] = 1.0
        x_meta[ctrl_missing, :, A] = MISSING

        return dict(
            x_data=torch.from_numpy(x_data).to(device),
            x_dna=torch.from_numpy(np.transpose(dna, (0, 2, 1))).contiguous().to(device),  # [B,4,G]
            x_meta=torch.from_numpy(x_meta).to(device),
            y_meta=torch.from_numpy(y_meta).to(device),
            y_target=torch.from_numpy(y_tgt).to(device),
            avail=torch.from_numpy(avail).to(device),
        )

    # ---- condition sampling ----
    def sample_conditions(self, bio: str, rng: np.random.Generator,
                          conditions: Sequence[Tuple[int, float]],
                          allowed_cell=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Draw ONE cell (cx, cy) and broadcast to all available assays (uniform per batch).

        Uniform (not per-assay) because the fixed v2 decoder pools y_meta across assays into a single
        global FiLM (meta_embed.mean(dim=1)) — per-assay-independent y_meta would average out and the
        decoder could not steer. So one matrix cell per batch. `allowed_cell(fx,fy)->bool` filters
        family cells (h31 holdout); None = all allowed.
        """
        while True:
            cx = conditions[rng.integers(len(conditions))]
            cy = conditions[rng.integers(len(conditions))]
            if allowed_cell is None or allowed_cell(cx[0], cy[0]):
                break
        fam_x = np.full(A, cx[0], np.int64); par_x = np.full(A, cx[1], np.float32)
        fam_y = np.full(A, cy[0], np.int64); par_y = np.full(A, cy[1], np.float32)
        return fam_x, par_x, fam_y, par_y

    def iter_train(self, batch_size: int, n_batches: int, rng: np.random.Generator,
                   conditions: Sequence[Tuple[int, float]], device, allowed_cell=None):
        """Yield training batches: random biosample + windows + per-assay conditions (chr19)."""
        pool = self.idx_chr19
        for _ in range(n_batches):
            bio = self.biosamples[rng.integers(len(self.biosamples))]
            wi = rng.choice(pool, size=batch_size, replace=False)
            fx, px, fy, py = self.sample_conditions(bio, rng, conditions, allowed_cell)
            yield self.make_batch(bio, wi, fx, px, fy, py, device)

    def eval_windows(self, chrom: str = "chr21") -> np.ndarray:
        return self.idx_chr21 if chrom == "chr21" else self.idx_chr19
