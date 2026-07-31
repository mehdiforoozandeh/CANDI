"""
ENCODE-style directory -> HDF5 bake (schema v2, self-describing).

Vendored from /project/6014832/mforooz/EpiDenoise/sandbox/prepare_h5.py lines 1-390 (bake path only;
validate-parity and overfit-sanity are dropped), with edits B1-B8.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

from candi_kit import __version__ as KIT_VERSION
from candi_kit.prep.panel import Panel, load_panel
from candi_kit.prep.paths import SideFiles
from candi_kit.prep.reference_sample import (
    fixed_dsf_pair_maps,
    make_handler,
    reference_tensors,
    resolve_column_order,
    restrict_navigation_to_biosamples,
)

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


def _filter_bios_on_disk(handler, names: Sequence[str]) -> List[str]:
    """Keep the panel biosamples that actually carry panel assays on disk, and SAY which are dropped.

    A panel that silently shrinks is the worst kind of failure here: the bake succeeds, the h5 looks
    fine, and you only notice later that a cell type you asked for is absent. Name every drop and why.
    """
    kept, dropped = [], []
    for n in names:
        if n not in handler.navigation:
            dropped.append((n, "no directory under the data root"))
        elif len(handler.navigation[n]) == 0:
            dropped.append((n, "directory exists but has none of the panel's assays"))
        else:
            kept.append(n)
    for n, why in dropped:
        print(f"[bake] WARNING dropping biosample {n!r}: {why}", file=sys.stderr)
    if dropped:
        print(f"[bake] panel requested {len(names)} biosamples, {len(kept)} usable "
              f"({len(dropped)} dropped -- see warnings above)", file=sys.stderr)
    return kept


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
    exclude_chroms: Tuple[str, ...],  # B4: was hardcoded ("chr19", "chr21")
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
    root: Path,
    panel: Panel,
    out_h5: Path,
    side: SideFiles,
    *,
    seed: int = 42,
    type2_ccre: int = 0,
    type2_non: int = 0,
    max_windows: Optional[int] = None,
    max_tile_per_chrom: Optional[int] = None,
    allow_missing_control: bool = False,
) -> None:
    # B3: scale comes from the panel, never from a CLI flag.
    context_bins = panel.context_bins
    resolution = panel.resolution
    context_bp = context_bins * resolution
    side.validate(need_ccres=(type2_ccre > 0 or type2_non > 0))
    # B1: one handler, built once (the original built two and threw the first away).
    handler = make_handler(root, panel, side)
    wanted = _filter_bios_on_disk(handler, list(panel.biosamples))
    restrict_navigation_to_biosamples(handler, wanted)
    n_train_ok = sum(1 for b in wanted if b.startswith("T_") and len(handler.navigation[b]) >= 2)
    if n_train_ok < 2:
        raise ValueError(
            f"need >=2 T_ biosamples with >=2 panel assays on disk, got {n_train_ok}; "
            "below that whole-assay cloze is skipped and the imputation loss is identically 0")

    exclude = tuple(panel.train_chroms) + tuple(panel.eval_chroms)
    type2: List[Tuple[str, int, int, int]] = []
    if type2_ccre > 0 or type2_non > 0:
        type2 = _sample_type2_loci(handler, type2_ccre, type2_non, context_bp, seed, exclude)
    # B4: tile the panel's chromosomes, not a hardcoded chr19/chr21.
    tiled: List[Tuple[str, int, int, int]] = []
    for chrom in exclude:
        w = [(c, s, e, REGION_TILE) for c, s, e in _tile_windows(chrom, handler.chr_sizes[chrom], context_bp)]
        if max_tile_per_chrom is not None:
            # STRIDE, do not take a prefix. The first tiles of a chromosome are telomeric/N-rich and
            # carry genuinely zero coverage, so a prefix subsample produces an h5 whose every window is
            # empty. That trains to nothing and trips the dataset's all-zero guard with a message saying
            # the h5 is "poisoned" -- a false accusation of data corruption. An even stride spans the
            # whole chromosome and keeps the subsample representative.
            k = int(max_tile_per_chrom)
            if k < len(w):
                step = max(1, len(w) // k)
                w = w[::step][:k]
        tiled.extend(w)
    rows = [list(x) for x in type2] + [list(x) for x in tiled]
    if max_windows is not None:
        rows = rows[: int(max_windows)]
    n = len(rows)
    bios_order = sorted(wanted)
    # B2: the column order is DERIVED from the handler, never declared.
    assay_order = resolve_column_order(handler)
    F = len(assay_order)
    L = context_bins
    Lbp = context_bp

    out_h5.parent.mkdir(parents=True, exist_ok=True)
    tmp_h5 = out_h5.with_name(out_h5.name + ".tmp")
    for p in (out_h5, tmp_h5):
        if p.exists():
            p.unlink()

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
        with h5py.File(tmp_h5, "w") as h5:
            # B7: schema v2 — the file describes its own scale and column order.
            h5.attrs["version"] = 2
            h5.attrs["context_bins"] = context_bins
            h5.attrs["resolution"] = resolution
            h5.attrs["data_root"] = str(root)
            h5.attrs["kit_version"] = KIT_VERSION
            h5.attrs["assays"] = json.dumps(assay_order)
            h5.attrs["assay_ids"] = json.dumps([handler.assay_to_id[a] for a in assay_order])
            h5.attrs["requested_panel"] = json.dumps(list(panel.assays))
            h5.attrs["num_assays"] = F
            h5.attrs["control_assay_id"] = int(handler.control_assay_id)
            h5.attrs["dsf_list"] = json.dumps(list(panel.dsf_list))
            h5.attrs["train_chroms"] = json.dumps(list(panel.train_chroms))
            h5.attrs["eval_chroms"] = json.dumps(list(panel.eval_chroms))
            h5.attrs["panel_json"] = json.dumps(asdict(panel))

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
                        # int32, not int16: real bins exceed 32767 (B_DND-41/DNase-seq hits 52,051 in a
                        # cCRE window). The research pipeline used int16 and clipped silently.
                        dtype=np.int32,
                        chunks=(1, L, F),
                        compression="gzip",
                        compression_opts=1,
                    )
                    for dsf in panel.dsf_list
                }
                p_ds = g.create_dataset("pval", shape=(n, L, F), dtype=np.float16, chunks=(1, L, F), compression="gzip", compression_opts=1)
                pk_ds = g.create_dataset("peaks", shape=(n, L, F), dtype=np.int64, chunks=(1, L, F), compression="gzip", compression_opts=1)
                cd_ds = g.create_dataset("control", shape=(n, L, 1), dtype=np.float32, chunks=(1, L, 1), compression="gzip", compression_opts=1)
                cm_ds = g.create_dataset("control_meta", shape=(n, 4, 1), dtype=np.float32, chunks=(1, 4, 1), compression="gzip", compression_opts=1)
                dna_ds = g.create_dataset("dna", shape=(n, Lbp, 4), dtype=np.int8, chunks=(1, Lbp, 4), compression="gzip", compression_opts=1)

                md_for_dsf: Dict[int, Optional[np.ndarray]] = {dsf: None for dsf in panel.dsf_list}

                for chrom in chrom_order:
                    clear_cache()  # bound memory to one (bios, chrom) at a time
                    chrom_rows = rows_by_chrom[chrom]

                    for dsf in panel.dsf_list:
                        cmap = _all_maps_same_dsf(handler, bios, chrom, dsf)
                        if not cmap:
                            for wi, _row in chrom_rows:
                                counts_ds[dsf][wi] = 0
                            continue
                        for wi, row in chrom_rows:
                            _c, s, e, _rt = row
                            locus = [chrom, s, e]
                            ref = reference_tensors(handler, bios, locus, cmap, cmap, y_prompt=True, random_shift=False)
                            counts_ds[dsf][wi] = ref["y_data"].numpy().astype(np.int32).reshape(L, F)
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

                for dsf in panel.dsf_list:
                    md = md_for_dsf[dsf]
                    if md is None:
                        # B6: -1, NEVER 0. The dataset's availability test is `meta[0] != -1`, so a
                        # zero row marks every assay available at log2(depth)=0 with all-zero counts.
                        md = np.full((4, F), -1.0, dtype=np.float32)
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

    # B8: gate the file BEFORE it is renamed into place, so a poisoned bake never gets a real name.
    _verify(tmp_h5, allow_missing_control=allow_missing_control)
    tmp_h5.rename(out_h5)
    print(f"Baked {n} windows for {len(bios_order)} biosamples -> {out_h5}", file=sys.stderr)


def _verify(h5_path: Path, *, allow_missing_control: bool) -> None:
    """Post-bake gate: F3 (no zero-filled meta row), F4 (the DSF depth ladder), F7 (raw counts),
    F15 (control availability)."""
    with h5py.File(h5_path, "r") as h5:
        dsf_list = json.loads(h5.attrs["dsf_list"])
        assays = json.loads(h5.attrs["assays"])
        bg = h5["biosamples"]
        for bios in json.loads(bg.attrs["order"]):
            g = bg[bios.replace("/", "_")]
            meta = {d: np.asarray(g[f"meta_dsf{d}"]) for d in dsf_list}

            # F3: rows 0-2 (depth, assay_id, read_length) are never all zero for a real column.
            for d, m in meta.items():
                zero = [assays[c] for c in range(m.shape[1]) if not np.any(m[0:3, c])]
                if zero:
                    raise AssertionError(f"F3 {bios}: meta_dsf{d} zero-filled for {zero}")

            # F4: depth halves with the downsampling factor.
            for d in dsf_list:
                for c in range(len(assays)):
                    if meta[1][0, c] == -1 or meta[d][0, c] == -1:
                        continue
                    err = abs(float(meta[d][0, c]) - (float(meta[1][0, c]) - np.log2(d)))
                    if err >= 0.05:
                        raise AssertionError(
                            f"F4 {bios}/{assays[c]}: meta_dsf{d}[0]={meta[d][0, c]:.4f} vs "
                            f"meta_dsf1[0]-log2({d})={meta[1][0, c] - np.log2(d):.4f} (err {err:.4f}) "
                            "-- this column's counts_dsf are not the downsampled data they claim to be")

            # F7: raw non-negative integer counts (arcsinh applied at bake time would collapse the range).
            # The decisive, coverage-independent test is INTEGRALITY: arcsinh/log1p output is non-integer
            # almost everywhere, raw counts are integers by construction.
            # The magnitude test is only a backstop, and it is coverage-dependent: a genuinely sparse
            # biosample can sit under any fixed bar. T_DND-41 (mean 0.384, p99 5.0) maxes at exactly 50
            # over the first 32 windows in the reference sandbox.h5 -- i.e. a 32-window probe against a
            # `<= 50` bar fails on known-good data. Probe wider and treat a low max as a WARNING, letting
            # integrality carry the actual invariant.
            avail = [c for c in range(len(assays)) if meta[1][0, c] != -1]
            n_probe = min(g["counts_dsf1"].shape[0], 1024)
            probe = np.asarray(g["counts_dsf1"][:n_probe])
            if avail:
                sub = probe[:, :, avail]
                if sub.min() < 0:
                    raise AssertionError(f"F7 {bios}: negative counts on an available column")
                if not np.all(sub == np.floor(sub)):
                    raise AssertionError(
                        f"F7 {bios}: non-integer counts -- a signal transform was applied at bake time. "
                        "arcsinh belongs in the model (encoder signal_transform), never in the loader.")
                if sub.max() <= 50:
                    print(f"[verify] WARNING {bios}: max count {sub.max()} over {n_probe} windows is low "
                          "-- expected for a sparse track, but confirm this is raw coverage",
                          file=sys.stderr)

            # F15: control is never masked, so a missing control is a silent capability loss.
            cm = np.asarray(g["control_meta"][:, 0, 0])
            rate = float(np.mean(cm != -1))
            print(f"[verify] {bios}: control available in {100 * rate:.1f}% of windows", file=sys.stderr)
            if rate < 1.0 and not allow_missing_control:
                raise AssertionError(
                    f"F15 {bios}: control available in only {100 * rate:.1f}% of windows; "
                    "pass --allow-missing-control to accept")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="ENCODE-style directory -> CANDI kit HDF5 bake")
    p.add_argument("--root", type=Path, required=True, help="ENCODE-style data directory")
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="output .h5 (point at /scratch)")
    p.add_argument("--fasta", type=Path, required=True)
    p.add_argument("--chrom-sizes", type=Path, required=True)
    p.add_argument("--blacklist", type=Path, default=None)
    p.add_argument("--ccres", type=Path, default=None)
    p.add_argument("--type2-ccre", type=int, default=0)
    p.add_argument("--type2-non", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-tile-per-chrom", type=int, default=None, help="Cap tiles per chromosome (smoke).")
    p.add_argument("--max-windows", type=int, default=None, help="Truncate to first N windows (debug).")
    p.add_argument("--allow-missing-control", action="store_true")
    args = p.parse_args(argv)

    bake(
        args.root,
        load_panel(args.panel),
        args.out,
        SideFiles(args.chrom_sizes, args.fasta, args.blacklist, args.ccres),
        seed=args.seed,
        type2_ccre=args.type2_ccre,
        type2_non=args.type2_non,
        max_windows=args.max_windows,
        max_tile_per_chrom=args.max_tile_per_chrom,
        allow_missing_control=args.allow_missing_control,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
