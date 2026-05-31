#!/usr/bin/env python3
"""Summarize CANDI v2 validation run metrics.jsonl files for the report."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def summarize_run(run_dir: Path) -> Dict[str, Any]:
    metrics_path = run_dir / "metrics.jsonl"
    records = load_jsonl(metrics_path)
    epochs = [r for r in records if r.get("kind") == "epoch"]
    steps = [r for r in records if r.get("kind") == "training_step"]
    summary: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "exists": metrics_path.exists(),
        "n_epochs": len(epochs),
        "n_steps": len(steps),
    }
    if not epochs:
        return summary
    eval_losses = []
    dcrs = []
    grad_imp = []
    for ep in epochs:
        el = ep.get("eval_losses") or {}
        total = el.get("total_loss")
        if total is not None and math.isfinite(float(total)):
            eval_losses.append(float(total))
        probes = ep.get("training_metadata_probes") or {}
        dcr = probes.get("depth_count_ratio")
        if dcr is not None and math.isfinite(float(dcr)):
            dcrs.append(float(dcr))
    for st in steps:
        tg = st.get("training_grad_norms") or {}
        g = tg.get("training_grad_norms/grad_norm_loss_branch_count_imp")
        if g is None:
            g = tg.get("grad_norm_loss_branch_count_imp")
        if g is not None and math.isfinite(float(g)):
            grad_imp.append(float(g))
    summary["eval_total_loss_by_epoch"] = eval_losses
    if len(eval_losses) >= 2:
        summary["eval_loss_delta_pct"] = 100.0 * (eval_losses[-1] - eval_losses[0]) / max(abs(eval_losses[0]), 1e-9)
    summary["depth_count_ratio_last"] = dcrs[-1] if dcrs else None
    summary["grad_norm_count_imp_max"] = max(grad_imp) if grad_imp else None
    summary["has_nan"] = any(
        not math.isfinite(float(v))
        for ep in epochs
        for d in (ep.get("eval_losses") or {}).values()
        for v in (d if isinstance(d, (int, float)) else [])
    )
    med_keys = []
    for ep in epochs:
        med = ep.get("eval_metrics_median") or {}
        med_keys.extend(med.keys())
    summary["has_median_eval"] = len(med_keys) > 0
    return summary


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sandbox/runs")
    cells = []
    for heads in ("count_only", "count_peak", "all"):
        for ch in ("plain", "depth_offset"):
            cells.append(summarize_run(root / f"validation_{heads}_{ch}"))
    resume = summarize_run(root / "validation_resume")
    print(json.dumps({"matrix": cells, "resume": resume}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
