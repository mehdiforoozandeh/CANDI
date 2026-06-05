#!/usr/bin/env python3
"""Optional row 0b: compare vb_natural vs canonical imp R² (A1 validation)."""
from __future__ import annotations

import math

from sandbox.autoresearch.may31 import prepare


def main() -> int:
    result = prepare.run_experiment()
    vb = float(result.get("imp_count_r2_gw", float("nan")))
    canonical = float(result.get("imp_count_r2_gw_canonical", float("nan")))
    delta = vb - canonical if math.isfinite(vb) and math.isfinite(canonical) else float("nan")

    print(f"imp_count_r2_gw:           {vb:.6f}" if math.isfinite(vb) else "imp_count_r2_gw:           nan")
    print(
        f"imp_count_r2_gw_canonical: {canonical:.6f}"
        if math.isfinite(canonical) else "imp_count_r2_gw_canonical: nan"
    )
    print(f"delta:                     {delta:.6f}" if math.isfinite(delta) else "delta:                     nan")

    if math.isfinite(delta) and delta <= 0.05:
        print("\nWARNING: delta <= 0.05 — fix prepare.py eval metadata before agent loop.")
        return 1
    if math.isfinite(delta) and delta > 0.10:
        print("\nA1 pass: vb_natural materially better than canonical.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
