"""
Bake sandbox HDF5 from EIC NPZ (via CANDIDataHandler) + parity / overfit gates.

Do not trust legacy prepare_eic_h5 layouts; tensors come from sandbox.reference_sample only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch

from candi_loss import CANDI_LOSS
from sandbox import SANDBOX_ASSAYS
from sandbox.reference_sample import (
    fixed_dsf_pair_maps,
    make_sandbox_handler,
    reference_tensors,
    restrict_navigation_to_biosamples,
)

DSF_PAIRS_FULL = [
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2),
    (1, 4),
    (4, 1),
    (4, 4),
    (1, 8),
    (8, 1),
    (8, 8),
]
DSF_PAIRS = DSF_PAIRS_FULL  # default; validate may slice locally for speed
REGION_CC = 0
REGION_NON = 1
REGION_TILE = 255


def _write_type2_bed(rows: List[List], bed_path: Path) -> None:
    """BED6-ish: chrom start end name for cCRE / non-cCRE training loci (region_type 0/1)."""
    bed_path.parent.mkdir(parents=True, exist_ok=True)
    with bed_path.open("w", encoding="utf-8") as f:
        for r in rows:
            chrom, s, e, rt = r[0], int(r[1]), int(r[2]), int(r[3])
            if rt == REGION_CC:
                name = "cCRE"
            elif rt == REGION_NON:
                name = "non_cCRE"
            else:
                continue
            f.write(f"{chrom}\t{s}\t{e}\t{name}\n")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _collect_bios_from_selection(sel: dict) -> List[str]:
    out: List[str] = []
    for b in sel["biosamples"]:
        name = b["name"]
        for pref, key in (("T_", "T"), ("V_", "V"), ("B_", "B")):
            if len(b.get(key, [])) > 0:
                out.append(pref + name)
    return out


def _filter_bios_on_disk(handler, names: Sequence[str]) -> List[str]:
    return [n for n in names if n in handler.navigation and len(handler.navigation[n]) > 0]


def _tile_windows(chrom: str, chr_len: int, context_bp: int) -> List[Tuple[str, int, int]]:
    out: List[Tuple[str, int, int]] = []
    tiling = (chr_len // context_bp) * context_bp
    for s in range(0, tiling, context_bp):
        e = s + context_bp
        out.append((chrom, s, e))
    return out


def _sample_type2_loci(
    handler,
    n_ccre: int,
    n_non: int,
    context_bp: int,
    seed: int,
    exclude_chroms: Tuple[str, ...] = ("chr19", "chr21"),
) -> List[Tuple[str, int, int, int]]:
    """
    Returns list of (chrom, start, end, region_type).
    """
    random.seed(seed)
    handler._load_ccre_index()
    chrs = [c for c in handler.chr_sizes.keys() if c not in exclude_chroms and c.startswith("chr")]
    if not chrs:
        raise RuntimeError("No chromosomes available for type2 sampling")

    def _allowed(chrom: str, s: int, e: int) -> bool:
        if e > handler.chr_sizes[chrom]:
            return False
        return handler._is_region_allowed(chrom, s, e)

    out: List[Tuple[str, int, int, int]] = []

    # cCRE-centered
    attempts = 0
    while len([x for x in out if x[3] == REGION_CC]) < n_ccre and attempts < n_ccre * 200:
        attempts += 1
        row = handler.ccre_df.sample(n=1).iloc[0]
        chrom = str(row["chrom"])
        if chrom in exclude_chroms or chrom not in handler.chr_sizes:
            continue
        c = (int(row["start"]) + int(row["end"])) // 2
        s = (c - context_bp // 2) // handler.resolution * handler.resolution
        e = s + context_bp
        if s < 0 or e > handler.chr_sizes[chrom]:
            continue
        if not _allowed(chrom, s, e):
            continue
        if (chrom, s, e, REGION_CC) not in out:
            out.append((chrom, s, e, REGION_CC))

    # non-cCRE random
    attempts = 0
    while len([x for x in out if x[3] == REGION_NON]) < n_non and attempts < n_non * 400:
        attempts += 1
        chrom = random.choice(chrs)
        size = handler.chr_sizes[chrom]
        max_start = max(0, (size - context_bp) // handler.resolution)
        s = random.randint(0, max_start) * handler.resolution
        e = s + context_bp
        if handler._overlaps_ccre(chrom, s, e):
            continue
        if not _allowed(chrom, s, e):
            continue
        if (chrom, s, e, REGION_NON) not in out:
            out.append((chrom, s, e, REGION_NON))

    if len([x for x in out if x[3] == REGION_CC]) < n_ccre or len([x for x in out if x[3] == REGION_NON]) < n_non:
        raise RuntimeError(
            f"type2 sampling incomplete: ccre={len([x for x in out if x[3]==REGION_CC])}, "
            f"non={len([x for x in out if x[3]==REGION_NON])}"
        )
    return out


def _all_maps_same_dsf(handler, bios: str, chrom: str, dsf: int) -> Dict[str, int]:
    x, _ = fixed_dsf_pair_maps(handler, bios, chrom, dsf, dsf)
    return x


def _enable_bake_cache(handler) -> Tuple[Any, Any]:
    """Monkeypatch chrom-level loaders on `handler` with size-1 memoization.

    Without this, `reference_tensors` re-reads every .npz file from disk for every
    single window; on sandbox panels (~15 biosamples × ~7k windows × 5 passes)
    that blows up to ~14M NPZ reads and several hours of wall time. With a
    single-entry cache keyed on (bios, locus, dsf/format), consecutive windows
    on the same chromosome for the same biosample hit RAM for counts/BW/peaks/control.

    Returns (clear_all_fn, restore_fn).
    """
    orig_counts = handler.load_bios_Counts
    orig_bw = handler.load_bios_BW
    orig_peaks = handler.load_bios_Peaks
    orig_control = handler.load_bios_Control

    _counts: Dict[tuple, tuple] = {}
    _bw: Dict[tuple, Any] = {}
    _peaks: Dict[tuple, Any] = {}
    _control: Dict[tuple, tuple] = {}

    def _locus_key(locus) -> tuple:
        return tuple(locus) if isinstance(locus, (list, tuple)) else (locus,)

    def _dsf_key(dsf) -> tuple:
        if isinstance(dsf, dict):
            return ("map", tuple(sorted(dsf.items())))
        return ("int", int(dsf))

    def load_counts(bios_name, locus, DSF=1, f_format="npz"):
        key = (bios_name, _locus_key(locus), _dsf_key(DSF), f_format)
        hit = _counts.get(key)
        if hit is None:
            _counts.clear()
            hit = orig_counts(bios_name, locus, DSF, f_format)
            _counts[key] = hit
        return hit

    def load_bw(bios_name, locus, f_format="npz", arcsinh=True):
        key = (bios_name, _locus_key(locus), f_format, bool(arcsinh))
        hit = _bw.get(key)
        if hit is None:
            _bw.clear()
            hit = orig_bw(bios_name, locus, f_format, arcsinh)
            _bw[key] = hit
        return hit

    def load_peaks(bios_name, locus, f_format="npz"):
        key = (bios_name, _locus_key(locus), f_format)
        hit = _peaks.get(key)
        if hit is None:
            _peaks.clear()
            hit = orig_peaks(bios_name, locus, f_format)
            _peaks[key] = hit
        return hit

    def load_control(bios_name, locus, DSF=1, f_format="npz"):
        key = (bios_name, _locus_key(locus), int(DSF), f_format)
        hit = _control.get(key)
        if hit is None:
            _control.clear()
            hit = orig_control(bios_name, locus, DSF, f_format)
            _control[key] = hit
        return hit

    handler.load_bios_Counts = load_counts
    handler.load_bios_BW = load_bw
    handler.load_bios_Peaks = load_peaks
    handler.load_bios_Control = load_control

    def clear_all() -> None:
        _counts.clear()
        _bw.clear()
        _peaks.clear()
        _control.clear()

    def restore() -> None:
        handler.load_bios_Counts = orig_counts
        handler.load_bios_BW = orig_bw
        handler.load_bios_Peaks = orig_peaks
        handler.load_bios_Control = orig_control

    return clear_all, restore


def bake(
    eic_root: Path,
    selection_path: Path,
    out_h5: Path,
    *,
    context_bins: int = 768,
    resolution: int = 25,
    seed: int = 42,
    type2_ccre: int = 1000,
    type2_non: int = 1000,
    max_windows: Optional[int] = None,
    max_tile_per_chrom: Optional[int] = None,
) -> None:
    context_bp = context_bins * resolution
    sel = json.loads(selection_path.read_text())
    wanted = _filter_bios_on_disk(make_sandbox_handler(str(eic_root)), _collect_bios_from_selection(sel))
    handler = make_sandbox_handler(str(eic_root))
    restrict_navigation_to_biosamples(handler, wanted)

    type2 = _sample_type2_loci(handler, type2_ccre, type2_non, context_bp, seed)
    chr19_all = [(c, s, e, REGION_TILE) for c, s, e in _tile_windows("chr19", handler.chr_sizes["chr19"], context_bp)]
    chr21_all = [(c, s, e, REGION_TILE) for c, s, e in _tile_windows("chr21", handler.chr_sizes["chr21"], context_bp)]
    if max_tile_per_chrom is not None:
        k = int(max_tile_per_chrom)
        chr19_all = chr19_all[:k]
        chr21_all = chr21_all[:k]
    rows = [list(x) for x in type2] + [list(x) for x in chr19_all] + [list(x) for x in chr21_all]
    if max_windows is not None:
        rows = rows[: int(max_windows)]
    n = len(rows)
    bios_order = sorted(wanted)
    F = len(SANDBOX_ASSAYS)
    L = context_bins
    Lbp = context_bp

    out_h5.parent.mkdir(parents=True, exist_ok=True)
    if out_h5.exists():
        out_h5.unlink()

    _write_type2_bed(rows, out_h5.with_name(out_h5.stem + "_loci_type2.bed"))

    # Group windows by chromosome so chrom-wide NPZ reads amortize across all windows
    # on that chromosome (see `_enable_bake_cache`).
    rows_by_chrom: Dict[str, List[Tuple[int, List]]] = defaultdict(list)
    for wi, row in enumerate(rows):
        rows_by_chrom[row[0]].append((wi, row))
    chrom_order = sorted(rows_by_chrom.keys())

    clear_cache, restore_cache = _enable_bake_cache(handler)
    t0 = time.time()
    try:
        with h5py.File(out_h5, "w") as h5:
            h5.attrs["version"] = 1
            h5.attrs["context_bins"] = context_bins
            h5.attrs["resolution"] = resolution
            h5.attrs["eic_root"] = str(eic_root)
            h5.attrs["selection_json"] = selection_path.read_text()

            ws = h5.create_group("windows")
            dt = h5py.string_dtype(encoding="utf-8")
            ws.create_dataset("chrom", data=np.array([r[0] for r in rows], dtype=object), dtype=dt)
            ws.create_dataset("start", data=np.array([r[1] for r in rows], dtype=np.int64))
            ws.create_dataset("end", data=np.array([r[2] for r in rows], dtype=np.int64))
            ws.create_dataset("region_type", data=np.array([r[3] for r in rows], dtype=np.uint8))

            bg = h5.create_group("biosamples")
            bg.attrs["order"] = json.dumps(bios_order)

            for bi, bios in enumerate(bios_order):
                g = bg.create_group(bios.replace("/", "_"))
                counts_ds = {
                    dsf: g.create_dataset(
                        f"counts_dsf{dsf}",
                        shape=(n, L, F),
                        dtype=np.int16,
                        chunks=(1, L, F),
                        compression="gzip",
                        compression_opts=1,
                    )
                    for dsf in (1, 2, 4, 8)
                }
                p_ds = g.create_dataset("pval", shape=(n, L, F), dtype=np.float16, chunks=(1, L, F), compression="gzip", compression_opts=1)
                pk_ds = g.create_dataset("peaks", shape=(n, L, F), dtype=np.int64, chunks=(1, L, F), compression="gzip", compression_opts=1)
                cd_ds = g.create_dataset("control", shape=(n, L, 1), dtype=np.float32, chunks=(1, L, 1), compression="gzip", compression_opts=1)
                cm_ds = g.create_dataset("control_meta", shape=(n, 4, 1), dtype=np.float32, chunks=(1, 4, 1), compression="gzip", compression_opts=1)
                dna_ds = g.create_dataset("dna", shape=(n, Lbp, 4), dtype=np.int8, chunks=(1, Lbp, 4), compression="gzip", compression_opts=1)

                md_for_dsf: Dict[int, Optional[np.ndarray]] = {dsf: None for dsf in (1, 2, 4, 8)}

                for chrom in chrom_order:
                    clear_cache()  # bound memory to one (bios, chrom) at a time
                    chrom_rows = rows_by_chrom[chrom]

                    for dsf in (1, 2, 4, 8):
                        cmap = _all_maps_same_dsf(handler, bios, chrom, dsf)
                        if not cmap:
                            for wi, _row in chrom_rows:
                                counts_ds[dsf][wi] = 0
                            continue
                        for wi, row in chrom_rows:
                            _c, s, e, _rt = row
                            locus = [chrom, s, e]
                            ref = reference_tensors(handler, bios, locus, cmap, cmap, y_prompt=True, random_shift=False)
                            counts_ds[dsf][wi] = ref["y_data"].numpy().astype(np.int16).reshape(L, F)
                            if md_for_dsf[dsf] is None:
                                md_for_dsf[dsf] = ref["y_meta"].numpy().astype(np.float32).reshape(4, F)

                    cmap1 = _all_maps_same_dsf(handler, bios, chrom, 1)
                    for wi, row in chrom_rows:
                        _c, s, e, _rt = row
                        locus = [chrom, s, e]
                        if not cmap1:
                            p_ds[wi] = 0
                            pk_ds[wi] = -1
                            cd_ds[wi] = 0
                            cm_ds[wi] = -1
                            dna_ds[wi] = 0
                            continue
                        ref = reference_tensors(handler, bios, locus, cmap1, cmap1, y_prompt=True, random_shift=False)
                        p_ds[wi] = ref["y_pval"].numpy().astype(np.float16).reshape(L, F)
                        pk_ds[wi] = ref["y_peaks"].numpy().astype(np.int64).reshape(L, F)
                        cd_ds[wi] = ref["control_data"].numpy().astype(np.float32).reshape(L, 1)
                        cm_ds[wi] = ref["control_meta"].numpy().astype(np.float32).reshape(4, 1)
                        dna_ds[wi] = ref["x_dna"].numpy().astype(np.int8).reshape(Lbp, 4)

                for dsf in (1, 2, 4, 8):
                    md = md_for_dsf[dsf]
                    if md is None:
                        md = np.zeros((4, F), dtype=np.float32)
                    g.create_dataset(f"meta_dsf{dsf}", data=md)

                elapsed = time.time() - t0
                print(
                    f"[bake] biosample {bi+1}/{len(bios_order)} ({bios}) done "
                    f"({len(chrom_order)} chroms, elapsed={elapsed:.1f}s)",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        restore_cache()

    print(f"Baked {n} windows for {len(bios_order)} biosamples -> {out_h5}", file=sys.stderr)


def _tensors_close(a: torch.Tensor, b: torch.Tensor, rtol=1e-5, atol=1e-6) -> bool:
    if a.shape != b.shape:
        return False
    if a.dtype in (torch.float32, torch.float64, torch.float16) or b.dtype in (
        torch.float32,
        torch.float64,
        torch.float16,
    ):
        return torch.allclose(a.float(), b.float(), rtol=rtol, atol=atol)
    if a.dtype != b.dtype:
        return torch.equal(a.long(), b.long())
    return torch.equal(a, b)


def validate_parity(
    eic_root: Path,
    selection_path: Path,
    h5_path: Path,
    parity_ok: Path,
    *,
    context_bins: int = 768,
    dsf_pairs: Sequence[Tuple[int, int]] = DSF_PAIRS_FULL,
    parity_fast: bool = False,
) -> None:
    sel = json.loads(selection_path.read_text())
    handler = make_sandbox_handler(str(eic_root))
    wanted = _filter_bios_on_disk(handler, _collect_bios_from_selection(sel))
    restrict_navigation_to_biosamples(handler, wanted)

    resolution = 25
    context_bp = context_bins * resolution

    def _dec_ch(x) -> str:
        if isinstance(x, (bytes, bytearray)):
            return x.decode("utf-8")
        return str(x)

    with h5py.File(h5_path, "r") as h5:
        bios_order = json.loads(h5["biosamples"].attrs["order"])
        chroms = np.array([_dec_ch(x) for x in h5["windows/chrom"][:]])
        starts = np.array(h5["windows/start"])
        ends = np.array(h5["windows/end"])

    # probe windows: indices for chr19, chr21, first 10 type2 (first rows are type2)
    idx_chr19 = [i for i in range(len(chroms)) if chroms[i] == "chr19"][:3]
    idx_chr21 = [i for i in range(len(chroms)) if chroms[i] == "chr21"][:3]
    idx_t2 = list(range(min(10, len(chroms))))

    if parity_fast:
        probe_wi = sorted(set((idx_chr19[:1] + idx_chr21[:1] + idx_t2[:4])))
        bios_slice = bios_order[:3]
    else:
        probe_wi = sorted(set(idx_chr19[:3] + idx_chr21[:3] + idx_t2[:10]))
        bios_slice = bios_order[:5]
    probes: List[Tuple[str, int]] = []
    for bios in bios_slice:
        for wi in probe_wi:
            probes.append((bios, wi))

    with h5py.File(h5_path, "r") as h5:
        for bios, wi in probes:
            chrom = chroms[wi]
            locus = [chrom, int(starts[wi]), int(ends[wi])]
            cmap1 = _all_maps_same_dsf(handler, bios, chrom, 1)
            ref1 = (
                reference_tensors(handler, bios, locus, cmap1, cmap1, y_prompt=True, random_shift=False)
                if cmap1
                else None
            )
            g = h5["biosamples"][bios.replace("/", "_")]
            hp = torch.tensor(np.array(g["pval"][wi]), dtype=torch.float32).unsqueeze(0)
            hk = torch.tensor(np.array(g["peaks"][wi]), dtype=torch.int64).unsqueeze(0)
            hdna = torch.tensor(np.array(g["dna"][wi]), dtype=torch.int8).unsqueeze(0)

            if ref1 is not None:
                if not _tensors_close(ref1["y_pval"].float(), hp.float(), rtol=1e-3, atol=1e-3):
                    raise AssertionError(f"parity fail pval bios={bios} wi={wi}")
                if not _tensors_close(ref1["y_peaks"], hk):
                    raise AssertionError(f"parity fail peaks bios={bios} wi={wi}")
                if not _tensors_close(ref1["x_dna"].to(torch.int8), hdna):
                    raise AssertionError(f"parity fail dna bios={bios} wi={wi}")

            for xd, yd in dsf_pairs:
                xm, ym = fixed_dsf_pair_maps(handler, bios, chrom, xd, yd)
                if not xm or not ym:
                    continue
                hxc = torch.tensor(np.array(g[f"counts_dsf{xd}"][wi]), dtype=torch.int64).unsqueeze(0)
                hyc = torch.tensor(np.array(g[f"counts_dsf{yd}"][wi]), dtype=torch.int64).unsqueeze(0)

                rx = reference_tensors(handler, bios, locus, xm, xm, y_prompt=True, random_shift=False)
                ry = reference_tensors(handler, bios, locus, ym, ym, y_prompt=True, random_shift=False)
                if not _tensors_close(rx["y_data"], hxc):
                    raise AssertionError(f"parity fail counts x_dsf={xd} bios={bios} wi={wi}")
                if not _tensors_close(ry["y_data"], hyc):
                    raise AssertionError(f"parity fail counts y_dsf={yd} bios={bios} wi={wi}")

    parity_ok.parent.mkdir(parents=True, exist_ok=True)
    git_sha = ""
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_repo_root()).decode().strip()
    except Exception:
        pass
    src_watch = [
        _repo_root() / "data.py",
        _repo_root() / "data_h5.py",
        _repo_root() / "model.py",
        _repo_root() / "train.py",
        _repo_root() / "sandbox" / "reference_sample.py",
        _repo_root() / "sandbox" / "prepare_h5.py",
        _repo_root() / "sandbox" / "data.py",
    ]
    source_mtimes = {str(p): p.stat().st_mtime for p in src_watch if p.exists()}
    payload = {
        "status": "pass",
        "timestamp": time.time(),
        "git_sha": git_sha,
        "h5_path": str(h5_path),
        "n_probes_executed": len(probes) * len(dsf_pairs),
        "source_mtimes": source_mtimes,
    }
    parity_ok.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {parity_ok}", file=sys.stderr)


def cmd_bake(args: argparse.Namespace) -> int:
    bake(
        Path(args.eic_data),
        Path(args.selection),
        Path(args.out),
        context_bins=args.context_bins,
        seed=args.seed,
        type2_ccre=args.type2_ccre,
        type2_non=args.type2_non,
        max_windows=args.max_windows,
        max_tile_per_chrom=args.max_tile_per_chrom,
    )
    return 0


def cmd_validate_parity(args: argparse.Namespace) -> int:
    pairs = DSF_PAIRS_FULL[:4] if args.parity_fast else DSF_PAIRS_FULL
    validate_parity(
        Path(args.eic_data),
        Path(args.selection),
        Path(args.h5),
        Path(args.parity_ok),
        context_bins=args.context_bins,
        dsf_pairs=pairs,
        parity_fast=args.parity_fast,
    )
    return 0


def cmd_overfit_sanity(args: argparse.Namespace) -> int:
    """
    GATE E — overfit a single batch (plan): default 200 steps, type2_loci, bs=2.
    Asserts (i) total loss drop >= 50%, (ii) each CANDI_LOSS head drops >=25% when its start loss > eps,
    (iii) no nan/inf. Use `--relax-gate-e` for legacy smoke H5s.
    """
    from sandbox.batch import make_masker, prepare_masked_batch
    from sandbox.data import SandboxH5Dataset
    from sandbox.model import build_sandbox_candi

    h5_path = Path(args.h5)
    steps = int(args.steps)
    device = torch.device(args.device)

    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    regime_arg = str(args.regime)
    regime_order = (
        ("type1_chr19", "type2_loci")
        if regime_arg == "auto"
        else (regime_arg,)
    )
    batch = None
    regime_used = None
    for reg in regime_order:
        ds = SandboxH5Dataset(
            h5_path,
            reg,
            train=True,
            batch_size=int(args.batch_size),
            biosample_prefix="T_",
            dsf_list=(1, 2, 4, 8),
            dsf_sampling="off",
            seed=int(args.seed),
            shuffle=False,
            h5_cache_ram=False,
        )
        batch = next(iter(ds), None)
        if batch is not None:
            regime_used = reg
            break
    if batch is None:
        raise RuntimeError(
            "overfit-sanity: no train windows for type1_chr19 or type2_loci (empty or incompatible H5?)"
        )
    print(f"overfit-sanity: using regime={regime_used}", file=sys.stderr)

    masker = make_masker(
        p_full_loci=float(args.p_full_loci),
        p_full_assay=float(args.p_full_assay),
        p_chunks=float(args.p_chunks),
        mask_fraction=float(args.mask_fraction),
        chunk_size=int(args.chunk_size),
    )
    cand = CANDI_LOSS(dist_type="gaussian").to(device)

    model = build_sandbox_candi(context_bins=int(args.context_bins)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    totals: list[float] = []
    heads: list[list[float]] = []
    model.train()
    for step in range(steps):
        prep = prepare_masked_batch(batch, masker, device)
        if prep is None:
            raise RuntimeError("overfit-sanity: masking produced empty batch")
        out = model(
            prep["x_data"],
            prep["x_dna"],
            prep["x_meta"],
            prep["y_meta"],
            query_mask=prep["query_mask"],
            query_mask_signal=prep["query_mask_signal"],
        )
        p, n, mu, scale, df, peak = out
        parts = cand(
            p,
            n,
            mu,
            scale,
            df,
            peak,
            prep["y_data"],
            prep["y_pval"],
            prep["y_peaks"],
            prep["observed_map"],
            prep["masked_map"],
            prep["signal_observed_map"],
            prep["signal_masked_map"],
            global_step=step,
        )
        loss_terms = [float(x.detach().float().cpu().item()) for x in parts]
        total = float(sum(loss_terms))
        if not all(math.isfinite(x) for x in loss_terms) or not math.isfinite(total):
            raise AssertionError(f"overfit-sanity: non-finite loss at step {step}")
        heads.append(loss_terms)
        totals.append(total)
        opt.zero_grad(set_to_none=True)
        total_t = parts[0] + parts[1] + parts[2] + parts[3] + parts[4] + parts[5]
        total_t.backward()
        opt.step()

    t0, t1 = totals[0], totals[-1]
    h0, h1 = heads[0], heads[-1]
    names = ("obs_count", "imp_count", "obs_pval", "imp_pval", "obs_peak", "imp_peak")
    if not args.relax_gate_e:
        if t1 > 0.5 * t0 + 1e-6:
            raise AssertionError(f"Gate E fail: total loss {t0:.6g} -> {t1:.6g} (need <=50% of start)")
        for i, nm in enumerate(names):
            if h0[i] > 1e-6 and h1[i] > 0.75 * h0[i] + 1e-6:
                raise AssertionError(
                    f"Gate E fail head {nm}: {h0[i]:.6g} -> {h1[i]:.6g} (need >=25% relative drop)"
                )
    else:
        if t1 > float(args.rel_improve) * t0 + 1e-6 and t1 > float(args.abs_target):
            raise AssertionError(
                f"overfit-sanity(relaxed): loss_start={t0:.6g} loss_end={t1:.6g}"
            )
    print(
        f"overfit-sanity: ok steps={steps} total {t0:.6g} -> {t1:.6g} heads={dict(zip(names, h1))}",
        file=sys.stderr,
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Sandbox HDF5 bake / gates")
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("bake")
    pb.add_argument("--eic-data", type=Path, required=True)
    pb.add_argument("--selection", type=Path, default=_repo_root() / "sandbox" / "data" / "selection.json")
    pb.add_argument("--out", type=Path, required=True)
    pb.add_argument("--context-bins", type=int, default=768)
    pb.add_argument("--seed", type=int, default=42)
    pb.add_argument("--type2-ccre", type=int, default=1000)
    pb.add_argument("--type2-non", type=int, default=1000)
    pb.add_argument("--max-tile-per-chrom", type=int, default=None, help="Cap chr19/chr21 tile counts (smoke).")
    pb.add_argument("--max-windows", type=int, default=None, help="Truncate to first N windows (debug).")
    pb.set_defaults(func=cmd_bake)

    pv = sub.add_parser("validate-parity")
    pv.add_argument("--eic-data", type=Path, required=True)
    pv.add_argument("--selection", type=Path, default=_repo_root() / "sandbox" / "data" / "selection.json")
    pv.add_argument("--h5", type=Path, required=True)
    pv.add_argument("--parity-ok", type=Path, default=_repo_root() / "sandbox" / "data" / "parity.ok")
    pv.add_argument("--context-bins", type=int, default=768)
    pv.add_argument(
        "--parity-fast",
        action="store_true",
        help="Use fewer DSF pairs + smaller probe grid (faster CI smoke).",
    )
    pv.set_defaults(func=cmd_validate_parity)

    po = sub.add_parser("overfit-sanity")
    po.add_argument("--h5", type=Path, required=True)
    po.add_argument(
        "--regime",
        choices=("auto", "type1_chr19", "type2_loci"),
        default="type2_loci",
        help="Training regime (Gate E default: type2_loci). 'auto' tries chr19 then type2.",
    )
    po.add_argument(
        "--relax-gate-e",
        action="store_true",
        help="Use legacy total-loss-only check (--rel-improve / --abs-target) instead of strict Gate E.",
    )
    po.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    po.add_argument("--steps", type=int, default=200)
    po.add_argument("--batch-size", type=int, default=4)
    po.add_argument("--seed", type=int, default=0)
    po.add_argument("--lr", type=float, default=3e-4)
    po.add_argument("--weight-decay", type=float, default=0.01)
    po.add_argument("--context-bins", type=int, default=768)
    po.add_argument("--p-full-assay", type=float, default=1.0)
    po.add_argument("--p-full-loci", type=float, default=0.0)
    po.add_argument("--p-chunks", type=float, default=0.0)
    po.add_argument("--mask-fraction", type=float, default=0.2)
    po.add_argument("--chunk-size", type=int, default=40)
    po.add_argument("--rel-improve", type=float, default=0.5, help="Pass if loss_end < this * loss_start.")
    po.add_argument("--abs-target", type=float, default=2.0, help="Also pass if loss_end below this.")
    po.add_argument("--contrastive-weight", type=float, default=0.0)
    po.set_defaults(func=cmd_overfit_sanity)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
