"""HDF5-backed iterable dataset for candi_kit training (T_* only in train loop).

Provenance: COPY+EDIT of EpiDenoise/sandbox/data.py:1-353 (class SandboxH5Dataset renamed
CandiKitH5Dataset). `build_canonical_meta` (sandbox/data.py:53-101) is deliberately NOT vendored.

Scale is a property of the baked file, never of the caller: the assay order, F, the DSF ladder, the
context length/resolution and the train/eval chromosomes are READ FROM h5.attrs and exposed as
`.assays / .num_assays / .context_bins / .resolution / .dsf_list / .train_chroms / .eval_chroms`.
The batch dict keys and shapes are byte-identical to sandbox/data.py:306-324.
"""
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

Regime = Literal["type1", "type2_loci"]

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


def h5_depth_center(h5_path, prefix: str = "T_") -> float:
    """Median of the finite `meta_dsf1[0]` (log2 depth) entries over `prefix` biosamples.

    Data-derived replacement for the three inconsistent hardcoded depth_center defaults.
    """
    vals: List[float] = []
    with h5py.File(str(h5_path), "r") as h5:
        for b in json.loads(h5["biosamples"].attrs["order"]):
            if not b.startswith(prefix):
                continue
            row = np.array(h5["biosamples"][b.replace("/", "_")]["meta_dsf1"][0], dtype=np.float64)
            vals.extend(float(v) for v in row if np.isfinite(v) and v != -1.0)
    if not vals:
        raise ValueError(f"no finite meta_dsf1 depths for prefix {prefix!r} in {h5_path}")
    return float(np.median(vals))


class CandiKitH5Dataset(IterableDataset):
    """
    Reads the `candi_kit.prep.bake` layout. Train batches are T_* biosamples only (caller filters).

    Eval (`train=False`): cycles through **all** `T_*` biosamples in sorted `parity.ok` order so
    every epoch's eval covers each biosample (denoising GT = T_*; imputation GT = V_*/B_* when present).
    """

    def __init__(
        self,
        h5_path: Path,
        regime: Regime,
        *,
        train: bool = True,
        batch_size: int = 8,
        biosample_prefix: str = "T_",
        dsf_list: Optional[Sequence[int]] = None,
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
        self.dsf_list: Tuple[int, ...] = tuple(int(x) for x in dsf_list) if dsf_list is not None else ()
        self.dsf_sampling = dsf_sampling
        self.seed = seed
        self.shuffle = shuffle
        self.eval_include_vb_ground_truth = bool(eval_include_vb_ground_truth)
        self.imp_prefixes = tuple(str(p) for p in imp_prefixes)
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
            version = int(h5.attrs.get("version", 1))
            if version < 2:
                raise ValueError(
                    f"{self.h5_path} is schema v{version}; candi_kit requires v2 (self-describing "
                    "attrs + meta_dsf -1 for absent DSF levels). Re-bake with candi_kit.prep.bake.")
            self.assays: List[str] = json.loads(h5.attrs["assays"])
            self.num_assays = len(self.assays)
            self.signal_dim = self.num_assays
            self.context_bins = int(h5.attrs["context_bins"])
            self.resolution = int(h5.attrs["resolution"])
            self.train_chroms: Tuple[str, ...] = tuple(json.loads(h5.attrs["train_chroms"]))
            self.eval_chroms: Tuple[str, ...] = tuple(json.loads(h5.attrs["eval_chroms"]))
            self._train_chroms = set(self.train_chroms)
            self._eval_chroms = set(self.eval_chroms)
            if dsf_list is None:
                self.dsf_list = tuple(int(x) for x in json.loads(h5.attrs["dsf_list"]))
            self._bios_order: List[str] = json.loads(h5["biosamples"].attrs["order"])
            chroms = [_decode_ch(x) for x in h5["windows/chrom"][:]]
            rtypes = np.array(h5["windows/region_type"][:])
            starts = np.array(h5["windows/start"][:])
            ends = np.array(h5["windows/end"][:])

        self._windows: List[Tuple[str, int, int, int]] = [
            (chroms[i], int(starts[i]), int(ends[i]), int(rtypes[i])) for i in range(len(chroms))
        ]

        self._train_indices = self._filter_window_indices()
        self._eval_indices = [i for i, w in enumerate(self._windows) if w[0] in self._eval_chroms]

        if not self._train_indices:
            raise ValueError(
                f"no training windows: regime={regime}, train_chroms={sorted(self._train_chroms)} — "
                "did you bake with --type2-ccre 0 --type2-non 0 under regime type2_loci?")
        if self.dsf_sampling != "off" and len(self.dsf_list) == 1:
            print("[data] WARNING: single-DSF ladder — per-assay independent DSF sampling is INERT, "
                  "and the depth-steering signal the recipe depends on is absent", flush=True)
        with _open_read() as h5:
            self._check_available_columns_nonzero(h5)

    def _check_available_columns_nonzero(self, h5: h5py.File, n_windows: int = 256) -> None:
        """Every column marked available in meta_dsf1 must carry some nonzero count.

        Catches the meta_dsf zero-vs-(-1) poisoning: an assay marked available at log2(depth)=0 with
        all-zero counts trains happily and produces garbage metrics.

        The probe must be WIDE and SPREAD OUT. Sparse broad marks are legitimately empty over any
        short run of adjacent windows -- T_DND-41/H3K9me3 is 17.2% nonzero genome-wide but all-zero
        across the first 4 windows of chr19 (gene-dense, and H3K9me3 is heterochromatin). A contiguous
        4-window probe therefore reports a false "poisoned h5" on perfectly good data. Sample evenly
        across the whole index instead of taking a prefix.
        """
        pool = self._train_indices or self._eval_indices
        wins = pool[:: max(1, len(pool) // n_windows)][:n_windows] if pool else []
        for b in self._bios_candidates():
            g = h5["biosamples"][b.replace("/", "_")]
            depth = np.array(g["meta_dsf1"][0])
            for fi in np.where(depth != -1.0)[0]:
                if not any(int(np.array(g["counts_dsf1"][wi, :, fi]).max()) > 0 for wi in wins):
                    raise ValueError(
                        f"{b}/{self.assays[int(fi)]} is marked available in meta_dsf1 "
                        f"(log2 depth {float(depth[fi])}) but counts_dsf1 is all-zero over "
                        f"{len(wins)} windows — the h5 is poisoned (absent DSF level written as 0 "
                        "instead of -1); re-bake.")

    def _filter_window_indices(self) -> List[int]:
        idxs: List[int] = []
        for i, (c, _s, _e, rt) in enumerate(self._windows):
            if self.train:
                if self.regime == "type1" and c in self._train_chroms:
                    idxs.append(i)
                elif self.regime == "type2_loci" and rt in (0, 1):
                    idxs.append(i)
            else:
                if c in self._eval_chroms:
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
