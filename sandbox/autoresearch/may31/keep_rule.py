"""Pareto keep rule for E32 session 3+ (imp_r2 up, den_r2 floor)."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Pareto floor: keep imp gains only if den stays usable (session 3 default).
DEN_KEEP_FLOOR = 0.25
DCR_LO = 3.25
DCR_HI = 4.75
IMP_EPS = 1e-4
DEN_EPS = 1e-4

TSV_COLUMNS = (
    "commit",
    "keep",
    "primary_score",
    "metric_phase",
    "imp_r2",
    "den_r2",
    "imp_r2_canonical",
    "imp_r2_cloze_T",
    "imp_r2_task_gap",
    "dcr",
    "imp_pearson",
    "vram_mb",
    "vram_ok",
    "status",
    "description",
)


def _f(val: Any, default: float = float("nan")) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def guards_ok(
    *,
    dcr: float,
    vram_ok: bool,
    status: str,
) -> bool:
    if status != "ok" or not vram_ok:
        return False
    if not math.isfinite(dcr):
        return False
    return DCR_LO <= dcr <= DCR_HI


def _scan_rows_for_best(
    lines: list[str],
    *,
    require_keep: bool,
) -> Tuple[float, float]:
    if len(lines) < 2:
        return float("-inf"), 0.0
    header = lines[0].split("\t")
    try:
        i_keep = header.index("keep")
    except ValueError:
        i_keep = -1
    try:
        i_imp = header.index("imp_r2")
        i_den = header.index("den_r2")
        i_dcr = header.index("dcr")
        i_vram = header.index("vram_ok")
        i_status = header.index("status")
    except ValueError:
        return float("-inf"), 0.0

    best_imp = float("-inf")
    best_den = 0.0
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) <= max(i_imp, i_den, i_dcr):
            continue
        if require_keep and i_keep >= 0 and parts[i_keep] != "keep":
            continue
        imp = _f(parts[i_imp])
        den = _f(parts[i_den])
        dcr = _f(parts[i_dcr])
        vram_ok = parts[i_vram].lower() in ("true", "1") if len(parts) > i_vram else False
        status = parts[i_status] if len(parts) > i_status else "crash"
        if not guards_ok(dcr=dcr, vram_ok=vram_ok, status=status):
            continue
        if not math.isfinite(imp) or den < DEN_KEEP_FLOOR:
            continue
        if imp > best_imp + IMP_EPS or (
            abs(imp - best_imp) <= IMP_EPS and den > best_den + DEN_EPS
        ):
            best_imp = imp
            best_den = den
    if best_imp == float("-inf"):
        return float("-inf"), 0.0
    return best_imp, best_den


def load_best_pareto(results_path: Path) -> Tuple[float, float]:
    """Return (best_imp_r2, den_r2_at_that_row) among prior keeps; else (-inf, 0)."""
    if not results_path.exists():
        return float("-inf"), 0.0
    lines = results_path.read_text().splitlines()
    best_imp, best_den = _scan_rows_for_best(lines, require_keep=True)
    if best_imp > float("-inf"):
        return best_imp, best_den
    # Session-3 TSV reset: fall back to legacy rows (no keep column).
    legacy = results_path.parent / "results_legacy.tsv"
    if legacy.exists():
        legacy_lines = legacy.read_text().splitlines()
        return _scan_rows_for_best(legacy_lines, require_keep=False)
    return float("-inf"), 0.0


def should_keep(
    *,
    imp_r2: float,
    den_r2: float,
    dcr: float,
    vram_ok: bool,
    status: str,
    best_imp_r2: float,
    best_den_r2: float,
    den_floor: float = DEN_KEEP_FLOOR,
) -> Tuple[bool, str]:
    """Pareto keep: higher imp_r2 with den_r2 >= floor; tie-break on den."""
    if not guards_ok(dcr=dcr, vram_ok=vram_ok, status=status):
        return False, "guard_fail"
    if not math.isfinite(imp_r2) or not math.isfinite(den_r2):
        return False, "nan_metric"
    if den_r2 < den_floor:
        return False, f"den_below_{den_floor}"
    if imp_r2 > best_imp_r2 + IMP_EPS:
        return True, "imp_up"
    if abs(imp_r2 - best_imp_r2) <= IMP_EPS and den_r2 > best_den_r2 + DEN_EPS:
        return True, "tie_den_up"
    return False, "discard"


def format_tsv_row(row: Dict[str, Any]) -> str:
    return "\t".join(str(row.get(col, "")) for col in TSV_COLUMNS) + "\n"
