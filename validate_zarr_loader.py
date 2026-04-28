#!/usr/bin/env python3

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from data import set_global_seed, CANDIIterableDataset
from data_zarr import ZarrCANDIIterableDataset, get_prepared_eic_path


def build_dataset(dataset_cls, base_path, dsf_list, seed, split="train", num_loci=8, context_bins=128):
    return dataset_cls(
        base_path=base_path,
        dataset_type="eic",
        resolution=25,
        m=num_loci,
        context_length=context_bins * 25,
        split=split,
        loci_gen_strategy="full_chr",
        dsf_list=dsf_list,
        DNA=True,
        shuffle_bios=False,
        balanced_bios_order=False,
        fill_prompt_mode="median",
        signal_transform="log1p",
        seed=seed,
        data_backend="zarr" if dataset_cls is ZarrCANDIIterableDataset else "npz",
    )


def compare_arrays(name, left, right, atol=1e-5):
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        raise AssertionError(f"{name}: shape mismatch {left.shape} != {right.shape}")
    if not np.allclose(left, right, atol=atol, equal_nan=True):
        diff = np.max(np.abs(left - right))
        raise AssertionError(f"{name}: values differ (max abs diff={diff})")


def compare_count_metadata(npz_meta, zarr_meta):
    if npz_meta["depth"] != zarr_meta["depth"]:
        raise AssertionError(f"depth mismatch: {npz_meta['depth']} != {zarr_meta['depth']}")
    for key in ("sequencing_platform", "read_length", "run_type"):
        if npz_meta[key] != zarr_meta[key]:
            raise AssertionError(f"{key} mismatch: {npz_meta[key]} != {zarr_meta[key]}")


def direct_slice_parity(npz_ds, zarr_ds, bios_name, chrom, start_bin, end_bin, dsf):
    locus = [chrom, start_bin * 25, end_bin * 25]

    npz_counts, npz_counts_meta = npz_ds.load_bios_Counts(bios_name, locus, dsf)
    zarr_counts, zarr_counts_meta = zarr_ds.load_bios_Counts(bios_name, locus, dsf)
    if set(npz_counts.keys()) != set(zarr_counts.keys()):
        raise AssertionError("Count assay keys differ between NPZ and Zarr loaders")
    for assay in npz_counts:
        compare_arrays(f"counts:{assay}", npz_counts[assay], zarr_counts[assay], atol=0.0)
        compare_count_metadata(npz_counts_meta[assay], zarr_counts_meta[assay])

    npz_pval = npz_ds.load_bios_BW(bios_name, locus)
    zarr_pval = zarr_ds.load_bios_BW(bios_name, locus)
    if set(npz_pval.keys()) != set(zarr_pval.keys()):
        raise AssertionError("P-value assay keys differ between NPZ and Zarr loaders")
    for assay in npz_pval:
        compare_arrays(f"pval:{assay}", npz_pval[assay], zarr_pval[assay], atol=1e-3)

    npz_peaks = npz_ds.load_bios_Peaks(bios_name, locus)
    zarr_peaks = zarr_ds.load_bios_Peaks(bios_name, locus)
    if set(npz_peaks.keys()) != set(zarr_peaks.keys()):
        raise AssertionError("Peak assay keys differ between NPZ and Zarr loaders")
    for assay in npz_peaks:
        compare_arrays(f"peaks:{assay}", npz_peaks[assay], zarr_peaks[assay], atol=1e-3)

    npz_control, npz_control_meta = npz_ds.load_bios_Control(bios_name, locus, dsf)
    zarr_control, zarr_control_meta = zarr_ds.load_bios_Control(bios_name, locus, dsf)
    if set(npz_control.keys()) != set(zarr_control.keys()):
        raise AssertionError("Control keys differ between NPZ and Zarr loaders")
    if npz_control:
        compare_arrays("control", npz_control["chipseq-control"], zarr_control["chipseq-control"], atol=0.0)
        if npz_control_meta["chipseq-control"] != zarr_control_meta["chipseq-control"]:
            raise AssertionError("Control metadata differs between NPZ and Zarr loaders")


def dataset_smoke(dataset, num_batches=3):
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=0)
    observed = 0
    expected_keys = {
        "sample_id", "x_data", "x_meta", "x_avail", "x_dna",
        "control_data", "control_meta", "control_avail",
        "y_data", "y_meta", "y_avail", "y_pval", "y_peaks", "y_dna",
        "x_dsf", "y_dsf", "dsf_transition_counts", "control_x_dsf",
    }
    for batch in loader:
        observed += 1
        missing = expected_keys.difference(batch.keys())
        if missing:
            raise AssertionError(f"Batch missing keys: {sorted(missing)}")
        if batch["x_data"].ndim != 3:
            raise AssertionError(f"Unexpected x_data rank: {batch['x_data'].shape}")
        if batch["y_pval"].shape != batch["y_data"].shape:
            raise AssertionError("y_pval shape does not match y_data shape")
        if batch["y_peaks"].shape != batch["y_data"].shape:
            raise AssertionError("y_peaks shape does not match y_data shape")
        if observed >= num_batches:
            break
    if observed == 0:
        raise AssertionError("No batches were produced during dataset smoke test")


def benchmark_direct_slice(dataset, bios_name, chrom, start_bin, end_bin, dsf, repeats=5):
    locus = [chrom, start_bin * 25, end_bin * 25]

    def one_pass():
        dataset.load_bios_Counts(bios_name, locus, dsf)
        dataset.load_bios_BW(bios_name, locus)
        dataset.load_bios_Peaks(bios_name, locus)
        dataset.load_bios_Control(bios_name, locus, dsf)

    start = time.perf_counter()
    one_pass()
    first_pass_s = time.perf_counter() - start

    rest = []
    for _ in range(max(0, repeats - 1)):
        t0 = time.perf_counter()
        one_pass()
        rest.append(time.perf_counter() - t0)

    return {
        "first_pass_s": first_pass_s,
        "steady_state_s": float(np.mean(rest)) if rest else None,
        "num_steady_passes": len(rest),
    }


def main():
    parser = argparse.ArgumentParser(description="Validate the prepared Zarr-backed EIC data loader.")
    parser.add_argument("--source-root", type=str, required=True, help="Path to the source DATA_CANDI_EIC root.")
    parser.add_argument("--prepared-root", type=str, default=None, help="Path to the prepared DATA_CANDI_EIC_ZARR root.")
    parser.add_argument("--biosample", type=str, default=None, help="Specific biosample to validate.")
    parser.add_argument("--chromosome", type=str, default="chr19", help="Chromosome to validate.")
    parser.add_argument("--start-bin", type=int, default=0, help="Start bin for direct slice parity checks.")
    parser.add_argument("--end-bin", type=int, default=128, help="End bin for direct slice parity checks.")
    parser.add_argument("--dsf", type=int, default=1, help="DSF to validate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    source_root = str(Path(args.source_root).expanduser().resolve())
    prepared_root = (
        str(Path(args.prepared_root).expanduser().resolve())
        if args.prepared_root is not None
        else get_prepared_eic_path(source_root)
    )

    set_global_seed(args.seed)
    npz_dataset = build_dataset(CANDIIterableDataset, source_root, [args.dsf], args.seed)
    zarr_dataset = build_dataset(ZarrCANDIIterableDataset, prepared_root, [args.dsf], args.seed)

    biosample = args.biosample or sorted(npz_dataset.navigation.keys())[0]
    print(f"[validate] biosample={biosample} chrom={args.chromosome} bins={args.start_bin}:{args.end_bin} dsf={args.dsf}")
    direct_slice_parity(npz_dataset, zarr_dataset, biosample, args.chromosome, args.start_bin, args.end_bin, args.dsf)
    print("[ok] direct slice parity")

    set_global_seed(args.seed)
    zarr_smoke = build_dataset(ZarrCANDIIterableDataset, prepared_root, [args.dsf], args.seed)
    dataset_smoke(zarr_smoke)
    print("[ok] dataset smoke")

    set_global_seed(args.seed)
    npz_bench = build_dataset(CANDIIterableDataset, source_root, [args.dsf], args.seed)
    set_global_seed(args.seed)
    zarr_bench = build_dataset(ZarrCANDIIterableDataset, prepared_root, [args.dsf], args.seed)
    npz_stats = benchmark_direct_slice(npz_bench, biosample, args.chromosome, args.start_bin, args.end_bin, args.dsf)
    zarr_stats = benchmark_direct_slice(zarr_bench, biosample, args.chromosome, args.start_bin, args.end_bin, args.dsf)
    print(f"[bench] npz={npz_stats}")
    print(f"[bench] zarr={zarr_stats}")


if __name__ == "__main__":
    main()
