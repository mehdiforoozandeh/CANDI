"""Data layer for the dual-conditioning testbed v2 (crux q15 / q16).

Reads base counts / DNA / control from `sandbox/data/sandbox.h5` (DSF1 only, denoising-only),
applies per-assay input/output transforms f_x / f_y, and assembles model-ready tensors with the
**3-row** metadata (aug_family, aug_param, log2_depth). chr19 -> train, chr21 -> test.

v2 changes from v1 (see plan_v2.md):
- **3-row metadata** `[B, 3, F]`: row0 aug_family (steering), row1 aug_param (steering, raw),
  row2 log2_depth (= meta_dsf1[0, a], the OBSERVED base library size; NON-steerable, h_y-independent,
  transform-independent). FiLM reads rows 0-1; the count-head offset reads row 2.
- **NO winsorize** — v1's CLIP=128 is removed. Raw transformed counts pass straight through; the heavy
  tail is handled by the encoder's arcsinh + the log-link offset head + power<=1.5, not by clipping.
- **Per-assay conditions** (default): each available assay draws (f_x,h_x),(f_y,h_y) independently
  (now possible because v2 conditions per assay). **uniform-per-batch** mode reproduces the v1 regime
  (one matrix cell per batch) as the baseline arm. **forced-identity-x** forces f_x=identity (h35).
- `thin` targets deterministically seeded by (biosample, assay, window_start, family, param, side).
"""
from __future__ import annotations

import io
import json
import zlib
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
N_META_ROWS = 3               # aug_family, aug_param, log2_depth
# Neutral (non-sentinel) placeholder depth for the control column when control signal IS present.
# Control has no meta_dsf1 depth and never enters the count-head offset (decoder emits signal assays
# only), so its exact depth is inert beyond the depth-aware encoder; the dataset-mean depth keeps the
# metadata availability consistent with the (present) control signal (encoder asserts they agree).
CONTROL_DEPTH = 25.1
_SEED_SALT = 0x9E3779B1


def _seed(bio: str, assay: int, win_start: int, fam: int, par: float, side: int) -> int:
    """Stable 32-bit seed for deterministic binomial thinning.

    Uses zlib.crc32 (NOT builtin hash()): Python str hashing is per-process salted (PYTHONHASHSEED
    unset), so a hash()-based seed would make thin targets differ across separate runs/processes.
    crc32 of the encoded key is bit-identical across processes (review: data.py:47)."""
    key = f"{bio}|{int(assay)}|{int(win_start)}|{int(fam)}|{round(float(par), 6)}|{int(side)}".encode()
    return (zlib.crc32(key) ^ _SEED_SALT) & 0xFFFFFFFF


class DualCondData:
    """H5-backed base-count provider + transform/tensor assembler (v2, 3-row metadata)."""

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
            # availability census + per-assay base log2_depth (meta_dsf1[0]).
            self.avail: Dict[str, np.ndarray] = {}
            self.depth: Dict[str, np.ndarray] = {}
            for b in self.bios_order:
                m = np.array(h5["biosamples"][b.replace("/", "_")]["meta_dsf1"])  # [4, A]
                self.avail[b] = (m[0] != -1).astype(np.float32)
                self.depth[b] = m[0].astype(np.float32)                            # [A]; -1 where missing
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
        Row 2 (log2_depth) is ALWAYS the true base library size (never shuffled) — it is non-steerable.
        """
        fam_xm = fam_x if fam_xm is None else fam_xm; par_xm = par_x if par_xm is None else par_xm
        fam_ym = fam_y if fam_ym is None else fam_ym; par_ym = par_y if par_ym is None else par_ym
        with self._open() as h5:
            wsorted, counts, dna, control = self._read(h5, bio, wi)
        B = counts.shape[0]
        av = self.avail[bio]                                   # [A]
        dep = self.depth[bio]                                  # [A]

        x_sig = np.full((B, L, A), MISSING, np.float32)
        y_tgt = np.zeros((B, L, A), np.float32)
        x_meta = np.full((B, N_META_ROWS, A + 1), MISSING, np.float32)
        y_meta = np.full((B, N_META_ROWS, A), MISSING, np.float32)
        avail = np.zeros((B, A), np.float32)

        for a in range(A):
            if av[a] <= 0:
                continue
            base_col = counts[:, :, a]                          # [B, L], RAW (no winsorize)
            xp = self._apply_side(base_col, int(fam_x[a]), float(par_x[a]), bio, a, wsorted, side=0)
            yp = self._apply_side(base_col, int(fam_y[a]), float(par_y[a]), bio, a, wsorted, side=1)
            x_sig[:, :, a] = xp.astype(np.float32)
            y_tgt[:, :, a] = yp.astype(np.float32)
            x_meta[:, 0, a] = float(fam_xm[a]); x_meta[:, 1, a] = float(par_xm[a]); x_meta[:, 2, a] = dep[a]
            y_meta[:, 0, a] = float(fam_ym[a]); y_meta[:, 1, a] = float(par_ym[a]); y_meta[:, 2, a] = dep[a]
            avail[:, a] = 1.0

        # control channel (index A): the h5 stores -1 for absent control across this dataset, so mark
        # those samples missing to keep signal/meta availability consistent (encoder asserts they agree).
        x_data = np.concatenate([x_sig, control], axis=2)      # [B, L, A+1]
        ctrl_missing = (control[:, :, 0] == MISSING).any(axis=1)     # [B]
        x_meta[:, 0, A] = float(T.FAM["identity"]); x_meta[:, 1, A] = 1.0
        x_meta[:, 2, A] = CONTROL_DEPTH                              # neutral, non-sentinel (see const)
        x_meta[ctrl_missing, :, A] = MISSING                        # control absent -> all rows sentinel

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
                          conditions: Sequence[Tuple[int, float]], *,
                          mode: str = "per_assay", allowed_cell=None,
                          force_x_identity: bool = False,
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Draw per-assay (or uniform-per-batch) input/output conditions for one biosample.

        mode="per_assay"  : each available assay draws (cx, cy) INDEPENDENTLY (the v2 default; the
                            off-diagonal per-assay cells are the real steering test).
        mode="uniform"    : one (cx, cy) drawn and broadcast to all assays (the v1 baseline arm).
        force_x_identity  : override f_x=identity for every assay (h35 forced-identity positive control).
        `allowed_cell(fx,fy)->bool` filters family cells (h31 holdout); None = all allowed.
        Unavailable assays carry an identity placeholder (masked out downstream).
        """
        idc = T.FAM["identity"]

        def _draw():
            while True:
                cx = conditions[rng.integers(len(conditions))]
                cy = conditions[rng.integers(len(conditions))]
                if force_x_identity:
                    cx = (idc, 1.0)
                if allowed_cell is None or allowed_cell(cx[0], cy[0]):
                    return cx, cy

        fam_x = np.full(A, idc, np.int64); par_x = np.ones(A, np.float32)
        fam_y = np.full(A, idc, np.int64); par_y = np.ones(A, np.float32)
        if mode == "uniform":
            cx, cy = _draw()
            fam_x[:] = cx[0]; par_x[:] = cx[1]; fam_y[:] = cy[0]; par_y[:] = cy[1]
        elif mode == "per_assay":
            for a in range(A):
                if self.avail[bio][a] <= 0:
                    continue
                cx, cy = _draw()
                fam_x[a] = cx[0]; par_x[a] = cx[1]; fam_y[a] = cy[0]; par_y[a] = cy[1]
        else:
            raise ValueError(f"unknown sampling mode {mode!r}")
        return fam_x, par_x, fam_y, par_y

    def iter_train(self, batch_size: int, n_batches: int, rng: np.random.Generator,
                   conditions: Sequence[Tuple[int, float]], device, *,
                   mode: str = "per_assay", allowed_cell=None, force_x_identity: bool = False,
                   cell_counts=None):
        """Yield training batches: random biosample + windows + conditions (chr19).

        If `cell_counts` (a dict/Counter) is given, increment it per drawn (f_x_family, f_y_family) cell
        over the available assays -- the 2c/h31 per-cell coverage log that separates a genuinely HARD
        family from an UNDERTRAINED one (and confirms the holdout mask actually withholds its cells).
        """
        pool = self.idx_chr19
        for _ in range(n_batches):
            bio = self.biosamples[rng.integers(len(self.biosamples))]
            wi = rng.choice(pool, size=batch_size, replace=False)
            fx, px, fy, py = self.sample_conditions(
                bio, rng, conditions, mode=mode, allowed_cell=allowed_cell,
                force_x_identity=force_x_identity)
            if cell_counts is not None:
                av = self.avail[bio]
                for a in range(A):
                    if av[a] > 0:
                        cell_counts[(int(fx[a]), int(fy[a]))] = cell_counts.get((int(fx[a]), int(fy[a])), 0) + 1
            yield self.make_batch(bio, wi, fx, px, fy, py, device)

    def eval_windows(self, chrom: str = "chr21") -> np.ndarray:
        return self.idx_chr21 if chrom == "chr21" else self.idx_chr19


def stratified_holdout(families: Sequence[str], rho: float, seed: int) -> set:
    """h31 holdout mask: a set of held-out (f_x_family, f_y_family) id pairs.

    Eligible = OFF-diagonal, NON-identity cells (identity row/col + the diagonal are always trained, so
    every transform is seen and only specific PAIRINGS are withheld). ~rho of the eligible cells are held
    out, stratified by a per-family cap so each family keeps >=1 cell as f_x AND >=1 as f_y -> every
    transform still appears on BOTH sides in training (the h31 precondition). Deterministic in `seed`.
    """
    idc = T.FAM["identity"]
    fam_ids = [T.FAM[f] for f in families if T.FAM[f] != idc]
    eligible = [(fx, fy) for fx in fam_ids for fy in fam_ids if fx != fy]
    if not eligible or rho <= 0:
        return set()
    k = int(round(float(rho) * len(eligible)))
    cap = len(fam_ids) - 2          # each family has (len-1) cells per side; hold out at most (len-2) -> keep >=1
    order = np.random.default_rng(seed).permutation(len(eligible))
    out, used_x, used_y = set(), {f: 0 for f in fam_ids}, {f: 0 for f in fam_ids}
    for idx in order:
        if len(out) >= k:
            break
        fx, fy = eligible[int(idx)]
        if used_x[fx] < cap and used_y[fy] < cap:
            out.add((fx, fy)); used_x[fx] += 1; used_y[fy] += 1
    return out
