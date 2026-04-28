"""HDF5-backed iterable dataset for sandbox training (T_* only in train loop)."""
from __future__ import annotations

import io
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import IterableDataset

from sandbox import SANDBOX_ASSAYS

Regime = Literal["type1_chr19", "type2_loci"]

_RAM_CACHE_MAX_DEFAULT = 10 * 1024**3  # 10 GiB


def _decode_ch(x) -> str:
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8")
    return str(x)


def _sample_xy_dsf(
    rng: random.Random,
    dsf_list: Sequence[int],
    mode: str,
    F: int,
) -> Tuple[List[int], List[int]]:
    """Return per-assay x_dsf and y_dsf lists (length F)."""
    dsfs = list(dsf_list)
    if mode == "off":
        return [1] * F, [1] * F
    if mode == "uniform":
        return [rng.choice(dsfs) for _ in range(F)], [rng.choice(dsfs) for _ in range(F)]
    if mode == "x_eq_y":
        v = [rng.choice(dsfs) for _ in range(F)]
        return list(v), list(v)
    if mode == "upsample_only":
        xd = [rng.choice(dsfs) for _ in range(F)]
        yd = []
        for x in xd:
            choices = [d for d in dsfs if d >= x]
            yd.append(rng.choice(choices) if choices else x)
        return xd, yd
    raise ValueError(f"unknown dsf sampling mode {mode}")


def build_canonical_meta(eic_metadata_path: str, sandbox_assays: List[str]) -> torch.Tensor:
    """Build canonical [4, F] metadata tensor from eic_metadata.csv (median stats per assay type).

    Row layout (matches sandbox H5 meta_dsf* convention):
      row 0: depth_log2  = log2(median sequencing depth)
      row 1: assay_idx   = position in sandbox_assays list
      row 2: read_length = median read length (bp)
      row 3: run_type    = 0.0 (single-ended) or 1.0 (paired-ended)

    Missing assay types in the CSV get row values of -1.0.
    """
    import csv, math as _math
    F = len(sandbox_assays)
    meta = [[-1.0] * F for _ in range(4)]
    assay_rows: Dict[str, list] = {a: [] for a in sandbox_assays}
    try:
        with open(eic_metadata_path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                aname = row.get("assay_name", "")
                if aname in assay_rows:
                    try:
                        depth = float(row["depth"])
                        readlen = float(row["read_length"])
                        rt_str = row.get("run_type", "single-ended")
                        run_type = 1.0 if "paired" in rt_str.lower() else 0.0
                        assay_rows[aname].append((depth, readlen, run_type))
                    except (ValueError, KeyError):
                        pass
    except FileNotFoundError:
        pass  # return all-(-1) tensor if file missing
    for fi, assay in enumerate(sandbox_assays):
        rows = assay_rows[assay]
        if not rows:
            continue
        depths = [r[0] for r in rows]
        readlens = [r[1] for r in rows]
        run_types = [r[2] for r in rows]
        depths.sort()
        readlens.sort()
        n = len(depths)
        depth_med = depths[n // 2]
        readlen_med = readlens[n // 2]
        rt_mode = 1.0 if sum(run_types) / n >= 0.5 else 0.0
        meta[0][fi] = _math.log2(max(depth_med, 1.0))
        meta[1][fi] = float(fi)
        meta[2][fi] = readlen_med
        meta[3][fi] = rt_mode
    return torch.tensor(meta, dtype=torch.float32)


class SandboxH5Dataset(IterableDataset):
    """
    Reads `sandbox/prepare_h5.py` layout. Train batches are T_* biosamples only (caller filters).

    Eval (`train=False`): cycles through **all** `T_*` biosamples in sorted `parity.ok` order so
    every epoch's eval covers each biosample (denoising GT = T_*; imputation GT = V_*/B_* when present).

    Yields dicts compatible with production trainer tensor keys (see train.py::_process_batch).
    """

    def __init__(
        self,
        h5_path: Path,
        regime: Regime,
        *,
        train: bool = True,
        batch_size: int = 8,
        biosample_prefix: str = "T_",
        dsf_list: Sequence[int] = (1, 2, 4, 8),
        dsf_sampling: str = "uniform",
        seed: int = 42,
        shuffle: bool = True,
        eval_include_vb_ground_truth: bool = False,
        imp_prefixes: Sequence[str] = ("V_", "B_"),
        h5_cache_ram: bool = True,
        ram_cache_max_bytes: int = _RAM_CACHE_MAX_DEFAULT,
    ):
        super().__init__()
        self.h5_path = Path(h5_path)
        self.regime = regime
        self.train = train
        self.batch_size = batch_size
        self.biosample_prefix = biosample_prefix
        self.dsf_list = tuple(int(x) for x in dsf_list)
        self.dsf_sampling = dsf_sampling
        self.seed = seed
        self.shuffle = shuffle
        self.eval_include_vb_ground_truth = bool(eval_include_vb_ground_truth)
        self.imp_prefixes = tuple(str(p) for p in imp_prefixes)
        self.signal_dim = len(SANDBOX_ASSAYS)
        self._ram_buf: Optional[bytes] = None

        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 not found: {self.h5_path}")

        sz = int(self.h5_path.stat().st_size)
        if h5_cache_ram and 0 < sz <= int(ram_cache_max_bytes):
            with open(self.h5_path, "rb") as rf:
                self._ram_buf = rf.read()

        def _open_read() -> h5py.File:
            if self._ram_buf is not None:
                return h5py.File(io.BytesIO(self._ram_buf), "r")
            return h5py.File(self.h5_path, "r")

        with _open_read() as h5:
            self._bios_order: List[str] = json.loads(h5["biosamples"].attrs["order"])
            chroms = [_decode_ch(x) for x in h5["windows/chrom"][:]]
            rtypes = np.array(h5["windows/region_type"][:])
            starts = np.array(h5["windows/start"][:])
            ends = np.array(h5["windows/end"][:])

        self._windows: List[Tuple[str, int, int, int]] = [
            (chroms[i], int(starts[i]), int(ends[i]), int(rtypes[i])) for i in range(len(chroms))
        ]

        self._train_indices = self._filter_window_indices()
        self._eval_indices = [i for i, w in enumerate(self._windows) if w[0] == "chr21"]

    def _filter_window_indices(self) -> List[int]:
        idxs: List[int] = []
        for i, (c, _s, _e, rt) in enumerate(self._windows):
            if self.train:
                if self.regime == "type1_chr19" and c == "chr19":
                    idxs.append(i)
                elif self.regime == "type2_loci" and rt in (0, 1):
                    idxs.append(i)
            else:
                if c == "chr21":
                    idxs.append(i)
        return idxs

    def estimate_steps_per_epoch(self) -> int:
        """Estimated number of batches per epoch based on training window count."""
        import math
        n = len(self._train_indices) if self.train else len(self._eval_indices)
        return max(1, math.ceil(n / self.batch_size))

    def _bios_candidates(self) -> List[str]:
        return [b for b in self._bios_order if b.startswith(self.biosample_prefix)]

    def _all_imp_biosamples(self, t_bios: str) -> List[str]:
        """Return ALL available imp biosamples (union of every prefix) for a T_* biosample.

        E.g. if T_AAA has both V_AAA and B_AAA, returns ['V_AAA', 'B_AAA'].
        Preserves the order of imp_prefixes.
        """
        if not t_bios.startswith("T_"):
            return []
        base = t_bios[2:]
        return [pref + base for pref in self.imp_prefixes if (pref + base) in self._bios_order]

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        rng = random.Random(self.seed)
        pool = list(self._train_indices if self.train else self._eval_indices)
        if not pool:
            return
        if self.shuffle:
            rng.shuffle(pool)
        else:
            pool.sort()
        bios_pool = self._bios_candidates()
        if not bios_pool:
            return

        bios_eval_order = sorted(bios_pool)
        # Build eval cycle: all (T_biosample, imp_biosample) pairs (union across all imp prefixes).
        # If a T_* has no matching imp biosample, include it with imp=None so it still gets eval.
        eval_pairs: List[Tuple[str, Optional[str]]] = []
        for t_b in bios_eval_order:
            imps = self._all_imp_biosamples(t_b)
            if imps:
                for imp_b in imps:
                    eval_pairs.append((t_b, imp_b))
            else:
                eval_pairs.append((t_b, None))
        if not eval_pairs:
            eval_pairs = [(t_b, None) for t_b in bios_eval_order]
        eval_pair_i = 0

        def _open_read() -> h5py.File:
            if self._ram_buf is not None:
                return h5py.File(io.BytesIO(self._ram_buf), "r")
            return h5py.File(self.h5_path, "r")

        with _open_read() as h5:
            pos = 0
            while pos < len(pool):
                batch_wi = pool[pos : pos + self.batch_size]
                pos += len(batch_wi)
                if self.train and len(batch_wi) < self.batch_size:
                    continue
                B = len(batch_wi)
                if self.train:
                    bios = rng.choice(bios_pool)
                    eval_imp_b = None
                else:
                    bios, eval_imp_b = eval_pairs[eval_pair_i % len(eval_pairs)]
                    eval_pair_i += 1
                gname = bios.replace("/", "_")

                xs_x, xs_y = _sample_xy_dsf(rng, self.dsf_list, self.dsf_sampling, self.signal_dim)
                x_dsf = torch.tensor(xs_x, dtype=torch.int64).unsqueeze(0).expand(B, -1).clone()
                y_dsf = torch.tensor(xs_y, dtype=torch.int64).unsqueeze(0).expand(B, -1).clone()

                g = h5["biosamples"][gname]
                L = g["pval"].shape[1]
                F = self.signal_dim
                Lbp = g["dna"].shape[1]

                x_data = torch.full((B, L, F), -1.0)
                y_data = torch.full((B, L, F), -1.0)
                x_meta = torch.full((B, 4, F), -1.0)
                y_meta = torch.full((B, 4, F), -1.0)
                x_avail = torch.zeros(B, F, dtype=torch.float32)
                y_avail = torch.zeros(B, F, dtype=torch.float32)
                y_pval = torch.zeros(B, L, F)
                y_peaks = torch.zeros(B, L, F)
                x_dna = torch.zeros(B, Lbp, 4)
                control_data = torch.zeros(B, L, 1)
                control_meta = torch.zeros(B, 4, 1)
                control_avail = torch.zeros(B, 1)
                region_type = torch.zeros(B, dtype=torch.uint8)

                for j, wi in enumerate(batch_wi):
                    region_type[j] = int(self._windows[wi][3])
                    y_pval[j] = torch.tensor(np.array(g["pval"][wi]), dtype=torch.float32)
                    y_peaks[j] = torch.tensor(np.array(g["peaks"][wi]), dtype=torch.float32)
                    x_dna[j] = torch.tensor(np.array(g["dna"][wi]), dtype=torch.float32)
                    control_data[j] = torch.tensor(np.array(g["control"][wi]), dtype=torch.float32)
                    control_meta[j] = torch.tensor(np.array(g["control_meta"][wi]), dtype=torch.float32)
                    control_avail[j, 0] = 1.0 if bool((control_data[j] != 0).any().item()) else 0.0
                    for fi in range(F):
                        xd = int(x_dsf[j, fi])
                        yd = int(y_dsf[j, fi])
                        if xd > 0:
                            xm = torch.tensor(np.array(g[f"meta_dsf{xd}"][:, fi]), dtype=torch.float32)
                            x_meta[j, :, fi] = xm
                            if float(xm[0]) != -1.0:
                                x_data[j, :, fi] = torch.tensor(
                                    np.array(g[f"counts_dsf{xd}"][wi, :, fi]), dtype=torch.float32
                                )
                                x_avail[j, fi] = 1.0
                        if yd > 0:
                            ym = torch.tensor(np.array(g[f"meta_dsf{yd}"][:, fi]), dtype=torch.float32)
                            y_meta[j, :, fi] = ym
                            if float(ym[0]) != -1.0:
                                y_data[j, :, fi] = torch.tensor(
                                    np.array(g[f"counts_dsf{yd}"][wi, :, fi]), dtype=torch.float32
                                )
                                y_avail[j, fi] = 1.0

                out: Dict[str, Any] = {
                    "x_data": x_data,
                    "x_meta": x_meta,
                    "x_avail": x_avail,
                    "x_dna": x_dna,
                    "y_data": y_data,
                    "y_meta": y_meta,
                    "y_avail": y_avail,
                    "y_pval": y_pval,
                    "y_peaks": y_peaks,
                    "control_data": control_data,
                    "control_meta": control_meta,
                    "control_avail": control_avail,
                    "x_dsf": x_dsf,
                    "y_dsf": y_dsf,
                    "control_x_dsf": torch.ones(B, dtype=torch.int64),
                    "biosample_name": bios,
                    "region_type": region_type,
                }
                if (not self.train) and self.eval_include_vb_ground_truth and eval_imp_b is not None:
                    imp_gname = eval_imp_b.replace("/", "_")
                    if imp_gname in h5["biosamples"]:
                        g_imp = h5["biosamples"][imp_gname]
                        y_data_imp = torch.full((B, L, F), -1.0)
                        y_pval_imp = torch.zeros(B, L, F)
                        y_peaks_imp = torch.zeros(B, L, F)
                        for j, wi in enumerate(batch_wi):
                            for fi in range(F):
                                yd = int(y_dsf[j, fi])
                                if yd > 0:
                                    ym_imp = torch.tensor(np.array(g_imp[f"meta_dsf{yd}"][:, fi]), dtype=torch.float32)
                                    if float(ym_imp[0]) != -1.0:
                                        y_data_imp[j, :, fi] = torch.tensor(
                                            np.array(g_imp[f"counts_dsf{yd}"][wi, :, fi]), dtype=torch.float32
                                        )
                            y_pval_imp[j] = torch.tensor(np.array(g_imp["pval"][wi]), dtype=torch.float32)
                            y_peaks_imp[j] = torch.tensor(np.array(g_imp["peaks"][wi]), dtype=torch.float32)
                        out["y_data_imp"] = y_data_imp
                        out["y_pval_imp"] = y_pval_imp
                        out["y_peaks_imp"] = y_peaks_imp
                        out["imp_biosample_name"] = eval_imp_b
                        # Also load V_*/B_* metadata (DSF=1) for natural-metadata imputation eval.
                        # Shape [4, F]; broadcast to [B, 4, F] so callers can directly index per-sample.
                        vb_meta_np = np.array(g_imp["meta_dsf1"])  # [4, F]
                        vb_meta = torch.tensor(vb_meta_np, dtype=torch.float32)
                        out["y_meta_imp"] = vb_meta.unsqueeze(0).expand(B, -1, -1).clone()  # [B, 4, F]

                yield out
