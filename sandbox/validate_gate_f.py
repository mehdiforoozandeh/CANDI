"""Validate a 3-epoch sandbox.train run against plan Gate F criteria."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PEARSON_DEN_FLOOR = 0.30
AUROC_PEAK_FLOOR = 0.70


def _load_epochs(metrics_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not metrics_path.is_file():
        raise SystemExit(f"gate_f: missing {metrics_path}")
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _epoch_total_loss(row: Dict[str, Any]) -> Optional[float]:
    # Preferred (new schema): explicit eval loss family in metrics.jsonl.
    ev_losses = row.get("eval_losses") or {}
    if "total_loss" in ev_losses:
        return float(ev_losses["total_loss"])
    # Backward compatibility for prefixed dictionaries.
    if "eval_losses/total_loss" in ev_losses:
        return float(ev_losses["eval_losses/total_loss"])
    # Legacy schema fallback.
    ev_legacy = row.get("eval") or {}
    if "den_pval_mse_gw" in ev_legacy:
        return float(ev_legacy["den_pval_mse_gw"])
    return None


def _metric(row: Dict[str, Any], key: str) -> Optional[float]:
    ev_metrics = row.get("eval_metrics") or {}
    if key in ev_metrics:
        return float(ev_metrics[key])
    pref = f"eval_metrics/{key}"
    if pref in ev_metrics:
        return float(ev_metrics[pref])
    ev_legacy = row.get("eval") or {}
    if key in ev_legacy:
        return float(ev_legacy[key])
    return None


def _monotonic_decreasing(xs: List[float], tol: float = 0.10) -> bool:
    for i in range(1, len(xs)):
        if xs[i] > xs[i - 1] * (1.0 + tol):
            return False
    return True


def _monotonic_increasing(xs: List[float], tol: float = 0.10) -> bool:
    for i in range(1, len(xs)):
        if xs[i] < xs[i - 1] * (1.0 - tol):
            return False
    return True


def validate(run_dir: Path) -> Tuple[bool, Dict[str, Any]]:
    report: Dict[str, Any] = {"run_dir": str(run_dir), "checks": {}}
    ok = True

    rows = _load_epochs(run_dir / "metrics.jsonl")
    if len(rows) < 2:
        return False, {**report, "error": f"need >=2 epochs of metrics, got {len(rows)}"}

    total = [x for x in (_epoch_total_loss(r) for r in rows) if x is not None]
    c_total = len(total) >= 2 and _monotonic_decreasing(total)
    report["checks"]["total_decreasing"] = {"ok": c_total, "values": total}
    ok = ok and c_total

    candidates = {
        "den_pval_pearson_gw": "increasing",
        "imp_pval_pearson_gw": "increasing",
        "den_pval_r2_gw": "increasing",
        "den_peak_auroc_gw": "increasing",
    }
    monotonic_any = False
    monotonic_details: Dict[str, Any] = {}
    for key, direction in candidates.items():
        vs = [v for v in (_metric(r, key) for r in rows) if v is not None and math.isfinite(v)]
        if len(vs) < 2:
            continue
        mono = _monotonic_decreasing(vs) if direction == "decreasing" else _monotonic_increasing(vs)
        monotonic_details[key] = {"direction": direction, "ok": mono, "values": vs}
        monotonic_any = monotonic_any or mono
    report["checks"]["monotonic_any"] = {"ok": monotonic_any, "details": monotonic_details}
    ok = ok and monotonic_any

    last = rows[-1]
    pearson_final = _metric(last, "den_pval_pearson_gw") or float("nan")
    auroc_final = _metric(last, "den_peak_auroc_gw") or float("nan")
    c_pearson = math.isfinite(pearson_final) and pearson_final >= PEARSON_DEN_FLOOR
    c_auroc = math.isfinite(auroc_final) and auroc_final >= AUROC_PEAK_FLOOR
    report["checks"]["pearson_floor"] = {"ok": c_pearson, "value": pearson_final, "floor": PEARSON_DEN_FLOOR}
    report["checks"]["auroc_floor"] = {"ok": c_auroc, "value": auroc_final, "floor": AUROC_PEAK_FLOOR}
    ok = ok and c_pearson and c_auroc

    probe = last.get("training_metadata_probes") or last.get("prompt_probe") or {}
    probe_ok = bool(probe) and all(math.isfinite(float(v)) for v in probe.values())
    report["checks"]["probe_finite"] = {"ok": probe_ok, "probe": probe}
    ok = ok and probe_ok

    report["ok"] = ok
    return ok, report


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="Gate F validator: parses <run_dir>/metrics.jsonl + resolved_config.yaml")
    p.add_argument("run_dir", type=Path)
    args = p.parse_args(argv)
    ok, rep = validate(Path(args.run_dir))
    print(json.dumps(rep, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
