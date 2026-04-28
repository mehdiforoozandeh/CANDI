#!/usr/bin/env python3

import argparse
import concurrent.futures
import json
import os
import shutil
import time
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from data_h5 import (
    H5_BIOSAMPLE_STORES_DIRNAME,
    H5_MANIFEST_FILENAME,
    H5_METADATA_DIRNAME,
    H5_SCHEMA_VERSION,
    get_prepared_eic_h5_path,
)

try:
    import h5py
except ImportError as exc:  # pragma: no cover - script-level dependency
    raise SystemExit(
        "The `h5py` package is required to prepare the EIC HDF5 dataset. "
        "Install it in the active environment and re-run this script."
    ) from exc

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional UX dependency
    class tqdm:
        def __init__(self, iterable=None, total=None, desc="", **kwargs):
            self.iterable = iterable
            self.total = total
            self.desc = desc
            self.current = 0
            if desc:
                print(desc)

        def __iter__(self):
            if self.iterable is None:
                return iter(())
            for item in self.iterable:
                yield item
                self.update(1)

        def update(self, n=1):
            self.current += n
            if self.total:
                print(f"{self.desc}: {self.current}/{self.total}")

        def set_postfix(self, **kwargs):
            if kwargs:
                print(f"{self.desc}: {kwargs}")

        def close(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()
            return False


def parse_list_argument(value):
    if value is None:
        return None
    values = [item.strip() for item in value.split(",") if item.strip()]
    return values or None


def load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def init_wandb_run(output_root, args):
    try:
        os.environ.setdefault("WANDB_DISABLE_GIT", "true")
        os.environ.setdefault("WANDB_DISABLE_CODE", "true")
        os.environ.setdefault("WANDB_HTTP_TIMEOUT", "300")
        import wandb
    except ImportError:
        print("Warning: wandb not installed. Skipping W&B initialization.")
        return None

    wandb_dir = output_root / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    return wandb.init(
        project="CANDI",
        name="prepare_h5_eic",
        dir=str(wandb_dir),
        config={
            "input_root": str(args.input_root),
            "output_root": str(args.output_root) if args.output_root is not None else None,
            "biosamples": args.biosamples,
            "chromosomes": args.chromosomes,
            "dsf_list": args.dsf_list,
            "chunk_bins": args.chunk_bins,
            "compression": args.compression,
            "num_workers": getattr(args, "num_workers", 1),
            "overwrite": bool(args.overwrite),
            "resume": bool(args.resume),
        },
    )


def npz_array(path):
    with np.load(path, allow_pickle=True) as payload:
        return payload[payload.files[0]]


def available_chromosomes(directory, selected_chromosomes):
    if not directory.exists():
        return []
    chroms = sorted(path.stem for path in directory.glob("*.npz"))
    if selected_chromosomes is None:
        return chroms
    return [chrom for chrom in chroms if chrom in selected_chromosomes]


def collect_assay_count_data(assay_dir, dsf, selected_chromosomes):
    signal_dir = assay_dir / f"signal_DSF{dsf}_res25"
    metadata_path = signal_dir / "metadata.json"
    if not signal_dir.exists() or not metadata_path.exists():
        return None
    chroms = available_chromosomes(signal_dir, selected_chromosomes)
    if not chroms:
        return None
    return {
        "signal_dir": signal_dir,
        "chromosomes": chroms,
        "metadata": load_json(metadata_path),
    }


def collect_modality_data(assay_dir, modality_dirname, selected_chromosomes):
    modality_dir = assay_dir / modality_dirname
    if not modality_dir.exists():
        return None
    chroms = available_chromosomes(modality_dir, selected_chromosomes)
    if not chroms:
        return None
    return {
        "dir": modality_dir,
        "chromosomes": chroms,
    }


def biosample_store_path(output_root, bios_name):
    return output_root / H5_BIOSAMPLE_STORES_DIRNAME / f"{bios_name}.h5"


def biosample_metadata_path(metadata_dir, bios_name):
    return metadata_dir / f"{bios_name}.json"


def biosample_artifacts_exist(output_root, metadata_dir, bios_name):
    return biosample_store_path(output_root, bios_name).exists() and biosample_metadata_path(metadata_dir, bios_name).exists()


def _stack_columns(entries, dtype=None):
    if not entries:
        return None, {}
    first = np.asarray(npz_array(entries[0][1]))
    stacked_dtype = dtype if dtype is not None else first.dtype
    stacked = np.empty((first.shape[0], len(entries)), dtype=stacked_dtype)
    assay_to_idx = {}
    for idx, (assay, npz_path) in enumerate(entries):
        assay_to_idx[assay] = idx
        values = np.asarray(npz_array(npz_path))
        if values.shape[0] != first.shape[0]:
            raise ValueError(
                f"Shape mismatch while stacking {npz_path}: expected first axis {first.shape[0]}, got {values.shape[0]}"
            )
        if values.ndim != 1:
            raise ValueError(f"Expected 1D chromosome vector in {npz_path}, got shape {values.shape}")
        stacked[:, idx] = values.astype(stacked_dtype, copy=False)
    return stacked, assay_to_idx


def write_dataset(h5_file, dataset_path, data, chunk_bins, compression):
    data = np.asarray(data)
    group_path, dataset_name = dataset_path.rsplit("/", 1)
    group = h5_file.require_group(group_path)

    if dataset_name in group:
        del group[dataset_name]

    if data.ndim == 1:
        chunks = (min(chunk_bins, data.shape[0]),)
    elif data.ndim == 2:
        chunks = (min(chunk_bins, data.shape[0]), data.shape[1])
    else:
        raise ValueError(f"Unsupported dataset rank for {dataset_path}: {data.ndim}")

    kwargs = {
        "data": data,
        "shape": data.shape,
        "dtype": data.dtype,
        "chunks": chunks,
    }
    if compression != "none":
        kwargs["compression"] = compression
    group.create_dataset(dataset_name, **kwargs)


def convert_biosample(source_root, output_root, metadata_dir, bios_name, assay_names, dsf_values, selected_chromosomes, chunk_bins, compression):
    bios_dir = source_root / bios_name
    bios_store = biosample_store_path(output_root, bios_name)
    bios_store.parent.mkdir(parents=True, exist_ok=True)

    bios_output = {
        "biosample_name": bios_name,
        "counts": {},
        "pval": {"chromosomes": [], "chrom_assay_to_idx": {}},
        "peaks": {"chromosomes": [], "chrom_assay_to_idx": {}},
        "control": {},
    }
    stats = {
        "count_arrays_written": 0,
        "pval_arrays_written": 0,
        "peak_arrays_written": 0,
        "control_arrays_written": 0,
        "count_assays": 0,
        "pval_assays": 0,
        "peak_assays": 0,
    }

    count_sources = defaultdict(lambda: defaultdict(list))
    pval_sources = defaultdict(list)
    peaks_sources = defaultdict(list)
    count_metadata = defaultdict(lambda: {"chromosomes": set(), "chrom_assay_to_idx": {}, "metadata": {}})

    with h5py.File(bios_store, "w") as bios_file:
        for assay in assay_names:
            assay_dir = bios_dir / assay
            if assay == "RNA-seq":
                continue

            if assay == "chipseq-control":
                file_meta_path = assay_dir / "file_metadata.json"
                file_meta = load_json(file_meta_path) if file_meta_path.exists() else {}
                for dsf in dsf_values:
                    count_info = collect_assay_count_data(assay_dir, dsf, selected_chromosomes)
                    if count_info is None:
                        continue
                    dsf_key = str(dsf)
                    bios_output["control"][dsf_key] = {
                        "chromosomes": count_info["chromosomes"],
                        "metadata": {
                            "depth": count_info["metadata"]["depth"],
                            "sequencing_platform": file_meta.get("sequencing_platform", {}).get("2", "unknown"),
                            "read_length": file_meta.get("read_length", {}).get("2", None),
                            "run_type": file_meta.get("run_type", {}).get("2", "single-ended"),
                        },
                    }
                    for chrom in count_info["chromosomes"]:
                        counts = np.asarray(npz_array(count_info["signal_dir"] / f"{chrom}.npz"))
                        write_dataset(
                            bios_file,
                            f"control/dsf_{dsf}/{chrom}",
                            counts,
                            chunk_bins=chunk_bins,
                            compression=compression,
                        )
                        stats["control_arrays_written"] += 1
                continue

            file_meta_path = assay_dir / "file_metadata.json"
            if not file_meta_path.exists():
                continue
            file_meta = load_json(file_meta_path)

            for dsf in dsf_values:
                count_info = collect_assay_count_data(assay_dir, dsf, selected_chromosomes)
                if count_info is None:
                    continue
                dsf_key = str(dsf)
                dsf_meta = count_metadata[dsf_key]
                stats["count_assays"] += 1
                dsf_meta["metadata"][assay] = {
                    "depth": count_info["metadata"]["depth"],
                    "sequencing_platform": file_meta.get("sequencing_platform", {}),
                    "read_length": file_meta.get("read_length", {}),
                    "run_type": file_meta.get("run_type", {}),
                }
                dsf_meta["chromosomes"].update(count_info["chromosomes"])

                for chrom in count_info["chromosomes"]:
                    count_sources[dsf_key][chrom].append((assay, count_info["signal_dir"] / f"{chrom}.npz"))

            pval_info = collect_modality_data(assay_dir, "signal_BW_res25", selected_chromosomes)
            if pval_info is not None:
                stats["pval_assays"] += 1
                for chrom in pval_info["chromosomes"]:
                    pval_sources[chrom].append((assay, pval_info["dir"] / f"{chrom}.npz"))

            peaks_info = collect_modality_data(assay_dir, "peaks_res25", selected_chromosomes)
            if peaks_info is not None:
                stats["peak_assays"] += 1
                for chrom in peaks_info["chromosomes"]:
                    peaks_sources[chrom].append((assay, peaks_info["dir"] / f"{chrom}.npz"))

        for dsf_key, chrom_map in count_sources.items():
            bios_output["counts"][dsf_key] = count_metadata[dsf_key]
            bios_output["counts"][dsf_key]["chromosomes"] = sorted(bios_output["counts"][dsf_key]["chromosomes"])
            for chrom, entries in chrom_map.items():
                stacked, assay_to_idx = _stack_columns(entries)
                bios_output["counts"][dsf_key]["chrom_assay_to_idx"][chrom] = assay_to_idx
                write_dataset(
                    bios_file,
                    f"counts/dsf_{dsf_key}/{chrom}",
                    stacked,
                    chunk_bins=chunk_bins,
                    compression=compression,
                )
                stats["count_arrays_written"] += 1

        bios_output["pval"]["chromosomes"] = sorted(pval_sources.keys())
        for chrom, entries in pval_sources.items():
            stacked, assay_to_idx = _stack_columns(entries)
            bios_output["pval"]["chrom_assay_to_idx"][chrom] = assay_to_idx
            write_dataset(
                bios_file,
                f"pval/{chrom}",
                stacked,
                chunk_bins=chunk_bins,
                compression=compression,
            )
            stats["pval_arrays_written"] += 1

        bios_output["peaks"]["chromosomes"] = sorted(peaks_sources.keys())
        for chrom, entries in peaks_sources.items():
            stacked, assay_to_idx = _stack_columns(entries)
            bios_output["peaks"]["chrom_assay_to_idx"][chrom] = assay_to_idx
            write_dataset(
                bios_file,
                f"peaks/{chrom}",
                stacked,
                chunk_bins=chunk_bins,
                compression=compression,
            )
            stats["peak_arrays_written"] += 1

    save_json(biosample_metadata_path(metadata_dir, bios_name), bios_output)
    return bios_name, stats


def build_parser():
    parser = argparse.ArgumentParser(description="Convert the EIC dataset into a HDF5-backed prepared store.")
    parser.add_argument(
        "--input-root",
        type=str,
        default="/project/6014832/mforooz/DATA_CANDI_EIC",
        help="Path to the source EIC dataset root.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Path to the prepared EIC output root. Defaults to a sister DATA_CANDI_EIC_H5 directory.",
    )
    parser.add_argument("--biosamples", type=str, default=None, help="Comma-separated biosample names to convert.")
    parser.add_argument("--chromosomes", type=str, default=None, help="Comma-separated chromosomes to convert.")
    parser.add_argument("--dsf-list", type=str, default="1,2,4", help="Comma-separated DSF values to convert.")
    parser.add_argument("--chunk-bins", type=int, default=8192, help="Chunk size along genomic bins.")
    parser.add_argument(
        "--compression",
        type=str,
        default="none",
        choices=["none", "gzip", "lzf"],
        help="HDF5 compression backend.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing prepared output root.")
    parser.add_argument("--resume", action="store_true", help="Skip biosamples that already have prepared metadata and HDF5 outputs.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of biosamples to convert in parallel (each writes its own .h5 file). Default: 1.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    start_time = time.time()

    source_root = Path(args.input_root).expanduser().resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"Input root not found: {source_root}")

    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else Path(get_prepared_eic_h5_path(source_root))
    selected_biosamples = parse_list_argument(args.biosamples)
    selected_chromosomes = parse_list_argument(args.chromosomes)
    dsf_values = [int(item) for item in parse_list_argument(args.dsf_list)]
    args.num_workers = max(1, int(args.num_workers))

    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)
    metadata_dir = output_root / H5_METADATA_DIRNAME
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (output_root / H5_BIOSAMPLE_STORES_DIRNAME).mkdir(parents=True, exist_ok=True)

    navigation = load_json(source_root / "navigation.json")
    shutil.copy2(source_root / "aliases.json", output_root / "aliases.json")
    biosample_names = sorted(navigation.keys())
    if selected_biosamples is not None:
        biosample_names = [bios for bios in biosample_names if bios in set(selected_biosamples)]
    filtered_navigation = {bios: navigation[bios] for bios in biosample_names}
    save_json(output_root / "navigation.json", filtered_navigation)
    wandb_run = init_wandb_run(output_root, args)

    manifest = {
        "schema_version": H5_SCHEMA_VERSION,
        "layout": "per_biosample_h5",
        "dataset_type": "eic",
        "resolution": 25,
        "source_root": str(source_root),
        "output_root": str(output_root),
        "biosamples": [],
        "dsf_list": dsf_values,
        "chunk_bins": args.chunk_bins,
        "compression": args.compression,
        "num_workers": args.num_workers,
    }
    totals = {
        "converted_biosamples": 0,
        "skipped_biosamples": 0,
        "count_arrays_written": 0,
        "pval_arrays_written": 0,
        "peak_arrays_written": 0,
        "control_arrays_written": 0,
    }

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    pending_jobs = []
    with tqdm(total=len(biosample_names), desc="Preparing EIC HDF5") as pbar:
        for bios_name in biosample_names:
            if args.resume and biosample_artifacts_exist(output_root, metadata_dir, bios_name):
                print(f"[resume] skipping existing biosample: {bios_name}")
                manifest["biosamples"].append(bios_name)
                totals["skipped_biosamples"] += 1
                pbar.update(1)
                pbar.set_postfix(converted=totals["converted_biosamples"], skipped=totals["skipped_biosamples"])
                if wandb_run is not None:
                    wandb_run.log({
                        "progress/processed_biosamples": totals["converted_biosamples"] + totals["skipped_biosamples"],
                        "progress/converted_biosamples": totals["converted_biosamples"],
                        "progress/skipped_biosamples": totals["skipped_biosamples"],
                    })
                continue

            pending_jobs.append(
                {
                    "source_root": source_root,
                    "output_root": output_root,
                    "metadata_dir": metadata_dir,
                    "bios_name": bios_name,
                    "assay_names": list(filtered_navigation[bios_name].keys()),
                    "dsf_values": dsf_values,
                    "selected_chromosomes": selected_chromosomes,
                    "chunk_bins": args.chunk_bins,
                    "compression": args.compression,
                }
            )

        def handle_completed(result_bios_name, bios_stats):
            manifest["biosamples"].append(result_bios_name)
            totals["converted_biosamples"] += 1
            for key in ("count_arrays_written", "pval_arrays_written", "peak_arrays_written", "control_arrays_written"):
                totals[key] += bios_stats[key]
            pbar.update(1)
            pbar.set_postfix(
                converted=totals["converted_biosamples"],
                skipped=totals["skipped_biosamples"],
                counts=totals["count_arrays_written"],
                pval=totals["pval_arrays_written"],
                peaks=totals["peak_arrays_written"],
            )
            if wandb_run is not None:
                wandb_run.log({
                    "progress/processed_biosamples": totals["converted_biosamples"] + totals["skipped_biosamples"],
                    "progress/converted_biosamples": totals["converted_biosamples"],
                    "progress/skipped_biosamples": totals["skipped_biosamples"],
                    "progress/count_arrays_written": totals["count_arrays_written"],
                    "progress/pval_arrays_written": totals["pval_arrays_written"],
                    "progress/peak_arrays_written": totals["peak_arrays_written"],
                    "progress/control_arrays_written": totals["control_arrays_written"],
                    "progress/current_count_assays": bios_stats["count_assays"],
                    "progress/current_pval_assays": bios_stats["pval_assays"],
                    "progress/current_peak_assays": bios_stats["peak_assays"],
                })

        if args.num_workers == 1:
            for job in pending_jobs:
                print(f"[convert] {job['bios_name']}")
                result_bios_name, bios_stats = convert_biosample(**job)
                handle_completed(result_bios_name, bios_stats)
        else:
            mp_context = get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers, mp_context=mp_context) as executor:
                future_to_bios = {
                    executor.submit(convert_biosample, **job): job["bios_name"] for job in pending_jobs
                }
                for future in concurrent.futures.as_completed(future_to_bios):
                    bios = future_to_bios[future]
                    try:
                        result_bios_name, bios_stats = future.result()
                    except Exception:
                        print(f"[error] biosample worker failed: {bios}")
                        raise
                    handle_completed(result_bios_name, bios_stats)

    manifest["biosamples"] = sorted(manifest["biosamples"])
    save_json(output_root / H5_MANIFEST_FILENAME, manifest)
    elapsed_s = time.time() - start_time
    if wandb_run is not None:
        wandb_run.summary["converted_biosamples"] = totals["converted_biosamples"]
        wandb_run.summary["skipped_biosamples"] = totals["skipped_biosamples"]
        wandb_run.summary["count_arrays_written"] = totals["count_arrays_written"]
        wandb_run.summary["pval_arrays_written"] = totals["pval_arrays_written"]
        wandb_run.summary["peak_arrays_written"] = totals["peak_arrays_written"]
        wandb_run.summary["control_arrays_written"] = totals["control_arrays_written"]
        wandb_run.summary["elapsed_s"] = elapsed_s
        wandb_run.finish()
    print(f"Prepared EIC HDF5 dataset written to: {output_root}")


if __name__ == "__main__":
    main()
