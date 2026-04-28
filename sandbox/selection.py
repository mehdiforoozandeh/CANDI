"""
Top-k EIC biosamples by aggregate panel-assay availability across T_/V_/B_.

Golden source: plan §3; biosample rows from data/eic_metadata.csv (same naming as data.py).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
import csv
from typing import Dict, List, Set, Tuple

from sandbox import SANDBOX_ASSAYS

PANEL = frozenset(SANDBOX_ASSAYS)
PREFIXES = ("T_", "V_", "B_")


@dataclass
class BiosampleSelection:
    name: str  # without T_/V_/B_ prefix
    T: List[str]
    V: List[str]
    B: List[str]
    total_score: int


def _strip_prefix(bios_name: str) -> Tuple[str, str | None]:
    for p in PREFIXES:
        if bios_name.startswith(p):
            return bios_name[len(p) :], p
    return bios_name, None


def availability_from_csv(
    eic_csv: Path,
    verify_base_path: Path | None = None,
    resolution: int = 25,
    dsf: int = 1,
) -> Dict[str, Dict[str, Set[str]]]:
    """
    Returns base_name -> {"T": set(assays), "V": set(assays), "B": set(assays)}
    Only assays in SANDBOX_ASSAYS. Optionally require signal_DSF{dsf}_res{res}/{chrom}.npz on disk.
    """
    with open(eic_csv, newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "biosample_name" not in reader.fieldnames or "assay_name" not in reader.fieldnames:
            raise ValueError(f"Expected biosample_name, assay_name in {eic_csv}")
        rows = list(reader)

    out: Dict[str, Dict[str, Set[str]]] = {}
    for row in rows:
        raw = str(row["biosample_name"])
        assay = str(row["assay_name"])
        if assay not in PANEL:
            continue
        base, prefix = _strip_prefix(raw)
        if prefix is None:
            continue
        key = prefix[0]  # T / V / B
        out.setdefault(base, {"T": set(), "V": set(), "B": set()})
        if verify_base_path is not None:
            bios_dir = verify_base_path / raw / assay
            sig = bios_dir / f"signal_DSF{dsf}_res{resolution}"
            # require at least chr19 npz for verification (quick check)
            if not (sig / "chr19.npz").is_file():
                continue
        out[base][key].add(assay)

    return out


def score_biosample(sets: Dict[str, Set[str]]) -> int:
    s = 0
    for k in ("T", "V", "B"):
        s += min(len(sets.get(k, set())), 8)
    return s


def select_top_k(
    eic_csv: Path,
    top_k: int = 5,
    verify_base_path: Path | None = None,
) -> List[BiosampleSelection]:
    avail = availability_from_csv(eic_csv, verify_base_path=verify_base_path)
    scored: List[Tuple[str, int, Dict[str, Set[str]]]] = []
    for name, sets in avail.items():
        sc = score_biosample(sets)
        scored.append((name, sc, sets))
    scored.sort(key=lambda x: (-x[1], x[0]))
    top = scored[:top_k]
    return [
        BiosampleSelection(
            name=n,
            T=sorted(sets["T"]),
            V=sorted(sets["V"]),
            B=sorted(sets["B"]),
            total_score=sc,
        )
        for n, sc, sets in top
    ]


def selection_to_dict(selection: List[BiosampleSelection]) -> dict:
    return {
        "assays": list(SANDBOX_ASSAYS),
        "biosamples": [asdict(b) for b in selection],
    }


def run_gate_a(payload: dict) -> None:
    """Gate A — selection sanity. Raises AssertionError on failure."""
    bios = payload.get("biosamples", [])
    assert len(bios) == 5, f"expected 5 biosamples, got {len(bios)}"
    union_vb: Set[str] = set()
    for b in bios:
        name = b["name"]
        t, v, bb = b["T"], b["V"], b["B"]
        assert len(t) >= 1, f"{name}: need >=1 T_* panel assay for training"
        union_vb.update(v)
        union_vb.update(bb)
    assert len(union_vb) >= 1, "union of V_* and B_* panel assays must be non-empty for imputation eval"


def main(argv: List[str] | None = None) -> int:
    repo = Path(__file__).resolve().parents[1]
    default_csv = repo / "data" / "eic_metadata.csv"
    default_out = repo / "sandbox" / "data" / "selection.json"

    p = argparse.ArgumentParser(description="Select top EIC biosamples for sandbox training.")
    p.add_argument("--eic-metadata", type=Path, default=default_csv)
    p.add_argument("--output", type=Path, default=default_out)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument(
        "--verify-data-path",
        type=Path,
        default=None,
        help="If set, require chr19.npz at DSF=1 for each (biosample, assay) row kept.",
    )
    p.add_argument("--print", action="store_true", help="Print table to stdout.")
    p.add_argument("--gate-a", action="store_true", help="Run Gate A checks on written JSON.")
    args = p.parse_args(argv)

    sel = select_top_k(
        args.eic_metadata,
        top_k=args.top_k,
        verify_base_path=args.verify_data_path,
    )
    payload = selection_to_dict(sel)

    if args.print:
        print(json.dumps(payload, indent=2))
        for b in payload["biosamples"]:
            print(
                f"{b['name']:30s} score={b['total_score']:2d}  "
                f"T={len(b['T'])} V={len(b['V'])} B={len(b['B'])}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {args.output}", file=sys.stderr)

    if args.gate_a:
        run_gate_a(payload)
        print("Gate A: PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
