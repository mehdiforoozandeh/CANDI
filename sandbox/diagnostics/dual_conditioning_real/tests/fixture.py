"""Tiny cached fixture: a 2-3-window-per-chrom slice of `sandbox.h5`, faithful to its layout.

`bake_fixture()` copies a handful of chr19 + chr21 windows for ALL biosample groups into a mini HDF5
that `SandboxH5Dataset` reads unchanged (same group/dataset/dtype schema). Bake once, cache on disk.
All block tests drive off this so they stay CPU-only and second-scale, never touching the 1.6 GB source.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np

from sandbox import SANDBOX_ASSAYS

SRC_H5 = Path(os.environ.get("Q19_SRC_H5", "sandbox/data/sandbox.h5"))
FIX_DIR = Path(os.environ.get(
    "Q19_FIXTURE_DIR", "sandbox/diagnostics/dual_conditioning_real/tests/_fixtures"))
FIX_H5 = FIX_DIR / "mini.h5"

# window-indexed datasets (sliced to the chosen windows) vs full datasets (copied whole).
_WINDOW_DSETS = ("control", "control_meta", "counts_dsf1", "counts_dsf2", "counts_dsf4", "counts_dsf8",
                 "dna", "peaks", "pval")
_FULL_DSETS = ("meta_dsf1", "meta_dsf2", "meta_dsf4", "meta_dsf8")

# Canonical 12 held-out imputation targets, derived from the baked sandbox.h5 census (2026-07-16):
# assay present in the V_/B_ view but ABSENT from the T_ input. run_type is the BAKED label (row 3 of
# the V/B meta_dsf1); the PRD flags that ATAC/DNase/H3K9me3 baked run_type/read_length may have drifted
# vs a later on-disk snapshot -> re-verify against source before trusting flip-test labels.
# tuple = (t_biosample, imp_prefix, assay_name, assay_idx, run_type_label)
HELD_OUT_TARGETS = [
    ("T_DND-41", "V_", "H3K4me1", 3, "single"),
    ("T_DND-41", "B_", "ATAC-seq", 0, "paired"),
    ("T_DND-41", "B_", "H3K9me3", 7, "paired"),
    ("T_H1-hESC", "V_", "H3K27ac", 4, "single"),
    ("T_RWPE2", "B_", "ATAC-seq", 0, "paired"),
    ("T_RWPE2", "B_", "DNase-seq", 1, "paired"),
    ("T_RWPE2", "B_", "H3K4me3", 2, "paired"),
    ("T_RWPE2", "B_", "H3K27ac", 4, "paired"),
    ("T_RWPE2", "B_", "H3K27me3", 5, "paired"),
    ("T_RWPE2", "B_", "H3K36me3", 6, "paired"),
    ("T_RWPE2", "B_", "H3K9me3", 7, "paired"),
    ("T_heart_left_ventricle", "V_", "DNase-seq", 1, "single"),
]


def _decode(x) -> str:
    return x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)


def bake_fixture(dst: Path = FIX_H5, *, n_chr19: int = 4, n_chr21: int = 4, force: bool = False) -> Path:
    """Slice n_chr19 + n_chr21 windows for every biosample group into `dst` (cached; idempotent)."""
    dst = Path(dst)
    if dst.exists() and not force:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(SRC_H5, "r") as s:
        chrom = np.array([_decode(c) for c in s["windows/chrom"][:]])
        order = json.loads(s["biosamples"].attrs["order"])
        g0 = s["biosamples"][order[0].replace("/", "_")]      # reference DNA is shared across groups

        def _top_by_dna(cand, n):
            # pick the most-mappable windows (avoid mostly-N regions like chr21's acrocentric p-arm),
            # striding across the WHOLE chromosome so the candidate pool isn't stuck in an N block.
            if len(cand) > 120:
                cand = cand[:: max(1, len(cand) // 120)]
            sums = np.array([int(np.asarray(g0["dna"][int(i)]).sum()) for i in cand])
            return [int(cand[j]) for j in np.argsort(-sums)[:n]]

        idx19 = _top_by_dna(np.where(chrom == "chr19")[0], n_chr19)
        idx21 = _top_by_dna(np.where(chrom == "chr21")[0], n_chr21)
        wi = np.sort(np.array(idx19 + idx21))                 # h5 fancy-index needs increasing order
        with h5py.File(dst, "w") as d:
            wg = d.create_group("windows")
            wg.create_dataset("chrom", data=chrom[wi].astype(object), dtype=h5py.string_dtype())
            wg.create_dataset("start", data=np.array(s["windows/start"][:])[wi])
            wg.create_dataset("end", data=np.array(s["windows/end"][:])[wi])
            wg.create_dataset("region_type", data=np.array(s["windows/region_type"][:])[wi])
            bg = d.create_group("biosamples")
            bg.attrs["order"] = json.dumps(order)
            for b in order:
                gname = b.replace("/", "_")
                sg = s["biosamples"][gname]
                dg = bg.create_group(gname)
                for k in _WINDOW_DSETS:
                    dg.create_dataset(k, data=sg[k][wi])       # reads only the selected windows
                for k in _FULL_DSETS:
                    dg.create_dataset(k, data=np.array(sg[k]))
    return dst


def biosample_census(h5_path: Path) -> dict:
    """{biosample: avail[8] int} from meta_dsf1 row 0 (depth != -1)."""
    out = {}
    with h5py.File(h5_path, "r") as h:
        order = json.loads(h["biosamples"].attrs["order"])
        for b in order:
            m = np.array(h["biosamples"][b.replace("/", "_")]["meta_dsf1"])   # [4, 8]
            out[b] = (m[0] != -1).astype(int)
    return out


def derive_held_out_targets(h5_path: Path) -> list:
    """Recompute the (t_bios, imp_prefix, assay_name, idx, run_type) inventory from the h5 census."""
    rows = []
    with h5py.File(h5_path, "r") as h:
        order = json.loads(h["biosamples"].attrs["order"])
        census = biosample_census(h5_path)
        for t in [b for b in order if b.startswith("T_")]:
            base = t[2:]
            for pref in ("V_", "B_"):
                imp = pref + base
                if imp not in order:
                    continue
                im = np.array(h["biosamples"][imp.replace("/", "_")]["meta_dsf1"])   # [4, 8]
                for a in range(8):
                    if census[imp][a] == 1 and census[t][a] == 0:
                        rt = int(im[3, a])
                        label = "paired" if rt == 1 else ("single" if rt == 0 else f"rt{rt}")
                        rows.append((t, pref, SANDBOX_ASSAYS[a], a, label))
    return rows


if __name__ == "__main__":
    p = bake_fixture(force=True)
    print("baked", p, "size", p.stat().st_size)
    got = derive_held_out_targets(p)
    assert got == HELD_OUT_TARGETS, ("inventory mismatch", got)
    print("held-out targets:", len(got), "OK")
