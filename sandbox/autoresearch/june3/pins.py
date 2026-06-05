"""May31-equivalent H5 pin loader for E34 autoresearch (frozen)."""
from __future__ import annotations

import io
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import torch

from sandbox import SANDBOX_ASSAYS
from sandbox.data import _sample_xy_dsf

ARTIFACT_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = ARTIFACT_DIR / "pin_manifest.json"
BATCH_SIZE = 4
SEED = 42
CHR_FRAC = 0.10
MANIFEST_VERSION = "june3_frac0.10"

T_BIOS = ["T_DND-41", "T_RWPE2", "T_heart_left_ventricle", "T_H1-hESC", "T_H9"]


def _decode_ch(x) -> str:
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8")
    return str(x)


def spread_pick(idxs: List[int], n: int, *, offset: int = 0) -> List[int]:
    if not idxs:
        raise ValueError("empty index pool")
    s = sorted(idxs)
    q = max(1, len(s) // 4)
    quartiles = [s[i * q : (i + 1) * q if i < 3 else len(s)] for i in range(4)]
    out: List[int] = []
    qi, pi = offset % 4, 0
    while len(out) < n:
        pool = quartiles[qi]
        if pool:
            out.append(pool[pi % len(pool)])
            pi += 1
        qi = (qi + 1) % 4
        if pi > len(s) * 2:
            break
    return out[:n]


def spread_pick_unique(idxs: List[int], n: int, *, offset: int = 0) -> List[int]:
    seen: set[int] = set()
    out: List[int] = []
    off = int(offset)
    guard = 0
    while len(out) < n and guard < n * 8:
        for wi in spread_pick(idxs, n, offset=off):
            if wi not in seen:
                seen.add(wi)
                out.append(wi)
                if len(out) >= n:
                    break
        off += 1
        guard += 1
    if len(out) < n:
        raise ValueError(f"spread_pick_unique: got {len(out)} < {n} from pool len {len(idxs)}")
    return out[:n]


def _chr_window_pools(h5_path: Path) -> tuple[List[int], List[int], int, int]:
    with h5py.File(h5_path, "r") as h5:
        chroms = [_decode_ch(x) for x in h5["windows/chrom"][:]]
    windows = list(range(len(chroms)))
    chr19 = [i for i in windows if chroms[i] == "chr19"]
    chr21 = [i for i in windows if chroms[i] == "chr21"]
    return chr19, chr21, len(chr19), len(chr21)


def build_pin_manifest(h5_path: Path, *, frac: float = CHR_FRAC) -> Dict[str, Any]:
    chr19, chr21, n19, n21 = _chr_window_pools(h5_path)
    n_unique = max(BATCH_SIZE, int(round(len(chr19) * frac)))
    unique = spread_pick_unique(chr19, n_unique, offset=0)
    n_batches = max(1, len(unique) // BATCH_SIZE)

    train: List[Dict[str, Any]] = []
    for b in range(n_batches):
        t = T_BIOS[b % len(T_BIOS)]
        wi = [unique[(b * BATCH_SIZE + j) % len(unique)] for j in range(BATCH_SIZE)]
        train.append({"t_bios": t, "window_indices": wi})

    return {
        "version": MANIFEST_VERSION,
        "seed": SEED,
        "h5_path": str(h5_path),
        "chr_frac": float(frac),
        "n_chr19_pool": n19,
        "n_chr21_pool": n21,
        "n_train_unique_windows": len(unique),
        "n_train_batches": len(train),
        "train": train,
    }


def load_manifest(h5_path: Path) -> Dict[str, Any]:
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text())
        if (
            manifest.get("version") == MANIFEST_VERSION
            and float(manifest.get("chr_frac", 0)) == CHR_FRAC
            and Path(str(manifest.get("h5_path", ""))).resolve() == Path(h5_path).resolve()
        ):
            return manifest
    manifest = build_pin_manifest(h5_path, frac=CHR_FRAC)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    return manifest


class H5PinCache:
    def __init__(self, h5_path: Path) -> None:
        self.h5_path = Path(h5_path)
        with open(self.h5_path, "rb") as rf:
            self._buf = rf.read()
        with h5py.File(io.BytesIO(self._buf), "r") as h5:
            chroms = [_decode_ch(x) for x in h5["windows/chrom"][:]]
            rtypes = np.array(h5["windows/region_type"][:])
            starts = np.array(h5["windows/start"][:])
            ends = np.array(h5["windows/end"][:])
        self.windows = [
            (chroms[i], int(starts[i]), int(ends[i]), int(rtypes[i])) for i in range(len(chroms))
        ]
        self.signal_dim = len(SANDBOX_ASSAYS)
        self.dsf_list = (1, 2, 4, 8)

    def _open(self) -> h5py.File:
        return h5py.File(io.BytesIO(self._buf), "r")

    def load_pinned_batch(
        self,
        entry: Dict[str, Any],
        *,
        dsf_sampling: str = "off",
        train: bool = True,
        rng_seed: int = SEED,
    ) -> Dict[str, torch.Tensor]:
        t_bios = str(entry["t_bios"])
        window_indices = list(entry["window_indices"])
        B = len(window_indices)
        rng = random.Random(rng_seed)

        with self._open() as h5:
            g = h5["biosamples"][t_bios.replace("/", "_")]
            L = g["pval"].shape[1]
            F = self.signal_dim
            Lbp = g["dna"].shape[1]

            if train:
                xs_x, xs_y = _sample_xy_dsf(rng, self.dsf_list, dsf_sampling, F)
            else:
                xs_x, xs_y = [1] * F, [1] * F

            x_dsf = torch.tensor(xs_x, dtype=torch.int64).unsqueeze(0).expand(B, -1).clone()
            y_dsf = torch.tensor(xs_y, dtype=torch.int64).unsqueeze(0).expand(B, -1).clone()

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

            for j, wi in enumerate(window_indices):
                region_type[j] = int(self.windows[wi][3])
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

            return {
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
                "biosample_name": t_bios,
                "region_type": region_type,
            }


def load_train_batches(h5_path: Path, manifest: Dict[str, Any]) -> List[Dict[str, torch.Tensor]]:
    cache = H5PinCache(h5_path)
    return [
        cache.load_pinned_batch(e, dsf_sampling="off", train=True, rng_seed=SEED + i)
        for i, e in enumerate(manifest["train"])
    ]


class PinnedTrainDataset:
    def __init__(self, batches: List[Dict[str, torch.Tensor]], seed: int) -> None:
        self._batches = batches
        self._seed = int(seed)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def estimate_steps_per_epoch(self) -> int:
        return max(1, len(self._batches))

    def __iter__(self):
        order = list(range(len(self._batches)))
        rng = random.Random(self._seed + self._epoch)
        rng.shuffle(order)
        for i in order:
            yield self._batches[i]


def eval_max_batches_for_frac(n_chr21_pool: int, frac: float, *, batch_size: int = BATCH_SIZE) -> int:
    n_slots = max(1, int(round(n_chr21_pool * frac)))
    return max(1, int(math.ceil(n_slots / batch_size)))


def verify_type1_chrom_split(h5_path: Path, manifest: Dict[str, Any]) -> None:
    chr19, chr21, _, _ = _chr_window_pools(h5_path)
    for e in manifest["train"]:
        for wi in e["window_indices"]:
            if wi not in chr19:
                raise ValueError(f"train window {wi} not on chr19")


def verify_data_fraction(h5_path: Path, manifest: Dict[str, Any], *, frac: float) -> Dict[str, float]:
    chr19, chr21, n19, n21 = _chr_window_pools(h5_path)
    train_wi = set()
    for e in manifest["train"]:
        train_wi.update(e["window_indices"])
    return {
        "train_unique_frac": len(train_wi) / max(1, n19),
        "n_chr19_pool": float(n19),
        "n_chr21_pool": float(n21),
        "n_train_unique": float(len(train_wi)),
        "n_train_batches": float(len(manifest["train"])),
        "chr_frac_target": float(frac),
    }


def verify_eval_dataloader_chr21(h5_path: Path, regime: str) -> None:
    from sandbox.data import SandboxH5Dataset

    ds = SandboxH5Dataset(h5_path, regime, train=False, batch_size=2, seed=SEED)
    for batch in ds:
        chrom = batch.get("chrom")
        if chrom is not None:
            if isinstance(chrom, (list, tuple)):
                assert all(str(c) == "chr21" for c in chrom)
            else:
                assert str(chrom) == "chr21"
        break
