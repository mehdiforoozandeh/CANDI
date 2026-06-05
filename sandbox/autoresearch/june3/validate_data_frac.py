#!/usr/bin/env python3
"""Sanity-check 10% chr19 train pins and chr21 eval fraction."""
from __future__ import annotations

import sys
from pathlib import Path

from sandbox.autoresearch.june3.pins import CHR_FRAC, load_manifest, verify_data_fraction, verify_type1_chrom_split

H5 = Path(__file__).resolve().parents[2] / "data" / "sandbox.h5"


def main() -> int:
    if not H5.exists():
        print(f"missing {H5}", file=sys.stderr)
        return 1
    manifest = load_manifest(H5)
    verify_type1_chrom_split(H5, manifest)
    fracs = verify_data_fraction(H5, manifest, frac=CHR_FRAC)
    print(f"train_unique_frac: {fracs['train_unique_frac']:.4f}")
    print(f"n_train_batches: {int(fracs['n_train_batches'])}")
    if abs(fracs["train_unique_frac"] - CHR_FRAC) > 0.03:
        print("train fraction out of tolerance", file=sys.stderr)
        return 1
    print("data_frac ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
