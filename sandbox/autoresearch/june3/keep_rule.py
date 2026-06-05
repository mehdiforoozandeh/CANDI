"""Keep/discard for E34 primary_score (higher is better)."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

TSV_COLUMNS = [
    "commit",
    "keep",
    "primary_score",
    "imp_r2",
    "den_r2",
    "count_imp_loss",
    "count_obs_loss",
    "train_imp_loss",
    "train_obs_loss",
    "param_count",
    "vram_mb",
    "vram_ok",
    "status",
    "description",
]


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_best_primary(path: Path) -> float:
    if not path.exists():
        return float("-inf")
    best = float("-inf")
    for line in path.read_text().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        sc = _f(parts[2])
        if math.isfinite(sc):
            best = max(best, sc)
    return best


def should_keep(
    *,
    primary_score: float,
    guards_ok: bool,
    vram_ok: bool,
    status: str,
    best_primary: float,
) -> Tuple[bool, str]:
    if status != "ok":
        return False, "crash"
    if not vram_ok:
        return False, "oom"
    if not guards_ok:
        return False, "guard_fail"
    if not math.isfinite(primary_score):
        return False, "nan_primary"
    if primary_score > best_primary + 1e-6:
        return True, "improved"
    return False, "no_gain"


def format_tsv_row(row: dict) -> str:
    return "\t".join(str(row.get(c, "")) for c in TSV_COLUMNS) + "\n"
