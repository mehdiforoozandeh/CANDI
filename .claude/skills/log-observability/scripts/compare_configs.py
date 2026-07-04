"""Compare two or more resolved_config.yaml files and print only the keys that differ.

Usage:
    python .cursor/skills/log-observability/scripts/compare_configs.py \
        sandbox/runs/baseline_anchor/resolved_config.yaml \
        sandbox/runs/baseline_dsf1_only/resolved_config.yaml ...

Output:
    For each leaf config key whose value differs across runs, print the key path and the
    per-run values. Suppresses cosmetic differences (run_dir, run_name, tags) by default.

This is the cheapest way to confirm that a sweep is a true controlled experiment and to
surface accidental config drift between baselines.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Tuple


def load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore[import]
    except ImportError as e:  # pragma: no cover - dev env always has pyyaml
        sys.exit(f"PyYAML required: {e}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def flatten(d: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            sub = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten(v, sub))
    elif isinstance(d, list):
        out[prefix] = tuple(d)  # tuples are hashable, easier to compare
    else:
        out[prefix] = d
    return out


COSMETIC_PREFIXES = (
    "training.run_dir",
    "wandb.run_name",
    "wandb.tags",
)


def diff(configs: List[Tuple[str, Dict[str, Any]]], suppress_cosmetic: bool = True) -> Dict[str, Dict[str, Any]]:
    flat = [(name, flatten(cfg)) for name, cfg in configs]
    all_keys = set()
    for _name, fc in flat:
        all_keys.update(fc.keys())
    differing: Dict[str, Dict[str, Any]] = {}
    for k in sorted(all_keys):
        if suppress_cosmetic and any(k.startswith(p) for p in COSMETIC_PREFIXES):
            continue
        values = {name: fc.get(k, "<missing>") for name, fc in flat}
        if len(set(repr(v) for v in values.values())) > 1:
            differing[k] = values
    return differing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("paths", nargs="+", help="Two or more resolved_config.yaml paths.")
    ap.add_argument("--include-cosmetic", action="store_true", help="Also show run_dir/run_name/tags differences.")
    args = ap.parse_args()
    if len(args.paths) < 2:
        sys.exit("Need at least two config paths.")
    configs: List[Tuple[str, Dict[str, Any]]] = []
    for p in args.paths:
        # Use parent dir name as label.
        label = os.path.basename(os.path.dirname(os.path.abspath(p))) or p
        configs.append((label, load_yaml(p)))
    d = diff(configs, suppress_cosmetic=not args.include_cosmetic)
    if not d:
        print("(no differing leaves)")
        return 0
    labels = [name for name, _ in configs]
    width = max(len(k) for k in d)
    print(f"{'key'.ljust(width)}  " + "  ".join(f"{lbl:>22s}" for lbl in labels))
    for k, vals in d.items():
        cells = [str(vals[lbl]) for lbl in labels]
        print(f"{k.ljust(width)}  " + "  ".join(f"{c:>22s}" for c in cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
