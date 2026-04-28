#!/usr/bin/env python3

import argparse
import concurrent.futures
import json
import os
import time
from multiprocessing import get_context
from pathlib import Path

import h5py
import numpy as np


def load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def npz_array(path):
    with np.load(path, allow_pickle=True) as payload:
        return np.asarray(payload[payload.files[0]])


def parse_biosamples(value):
    if value is None:
        return None
    bios = [item.strip() for item in value.split(",") if item.strip()]
    return bios or None


def repair_biosample(source_root, prepared_root, bios_name, chunk_bins, compression):
    bios_dir = source_root / bios_name
    meta_path = prepared_root / "h5_metadata" / f"{bios_name}.json"
    h5_path = prepared_root / "biosamples_h5" / f"{bios_name}.h5"
    prepared_meta = load_json(meta_path)
    rewritten = 0

    with h5py.File(h5_path, "r+") as h5f:
        for dsf_key, dsf_saved in prepared_meta.get("counts", {}).items():
            for chrom, idx_map in dsf_saved.get("chrom_assay_to_idx", {}).items():
                assay_items = sorted(idx_map.items(), key=lambda kv: kv[1])
                raw_arrays = []
                raw_dtypes = []
                expected_width = len(idx_map)

                for assay, col_idx in assay_items:
                    npz_path = bios_dir / assay / f"signal_DSF{dsf_key}_res25" / f"{chrom}.npz"
                    raw = npz_array(npz_path)
                    raw_arrays.append((col_idx, raw))
                    raw_dtypes.append(raw.dtype)

                if not raw_arrays:
                    continue

                length = raw_arrays[0][1].shape[0]
                dtype = np.result_type(*raw_dtypes)
                stacked = np.empty((length, expected_width), dtype=dtype)
                for col_idx, raw in raw_arrays:
                    if raw.shape[0] != length:
                        raise ValueError((bios_name, dsf_key, chrom, "shape mismatch", length, raw.shape[0]))
                    stacked[:, col_idx] = raw.astype(dtype, copy=False)

                group = h5f.require_group(f"counts/dsf_{dsf_key}")
                if chrom in group:
                    del group[chrom]

                kwargs = {
                    "data": stacked,
                    "shape": stacked.shape,
                    "dtype": stacked.dtype,
                    "chunks": (min(chunk_bins, stacked.shape[0]), stacked.shape[1]),
                }
                if compression != "none":
                    kwargs["compression"] = compression
                group.create_dataset(chrom, **kwargs)
                rewritten += 1

    return bios_name, rewritten


def main():
    parser = argparse.ArgumentParser(description="Repair HDF5 count datasets to preserve raw count dtype.")
    parser.add_argument("--source-root", type=str, required=True)
    parser.add_argument("--prepared-root", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--biosamples", type=str, default=None, help="Comma-separated biosample names to repair.")
    args = parser.parse_args()

    source_root = Path(args.source_root).expanduser().resolve()
    prepared_root = Path(args.prepared_root).expanduser().resolve()
    manifest = load_json(prepared_root / "manifest.json")
    biosamples = sorted(manifest["biosamples"])
    selected = parse_biosamples(args.biosamples)
    if selected is not None:
        biosamples = [bios for bios in biosamples if bios in set(selected)]

    chunk_bins = int(manifest.get("chunk_bins", 8192))
    compression = str(manifest.get("compression", "none"))
    num_workers = max(1, int(args.num_workers))

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    start = time.time()
    completed = 0
    total_rewritten = 0

    if num_workers == 1:
        for bios_name in biosamples:
            _, rewritten = repair_biosample(source_root, prepared_root, bios_name, chunk_bins, compression)
            completed += 1
            total_rewritten += rewritten
            if completed % 10 == 0 or completed == len(biosamples):
                print(f"repaired {completed}/{len(biosamples)} biosamples total_count_datasets={total_rewritten}", flush=True)
    else:
        mp_context = get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_context) as executor:
            future_to_bios = {
                executor.submit(repair_biosample, source_root, prepared_root, bios_name, chunk_bins, compression): bios_name
                for bios_name in biosamples
            }
            for future in concurrent.futures.as_completed(future_to_bios):
                bios_name = future_to_bios[future]
                try:
                    _, rewritten = future.result()
                except Exception:
                    print(f"[error] failed while repairing {bios_name}", flush=True)
                    raise
                completed += 1
                total_rewritten += rewritten
                if completed % 10 == 0 or completed == len(biosamples):
                    print(f"repaired {completed}/{len(biosamples)} biosamples total_count_datasets={total_rewritten}", flush=True)

    print(f"H5_COUNT_REPAIR_OK total_count_datasets={total_rewritten} elapsed_s={time.time() - start:.2f}", flush=True)


if __name__ == "__main__":
    main()
