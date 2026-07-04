"""Inspect `kind: "training_step"` rows in metrics.jsonl for grad-norm / clip / per-loss
trajectories.

Usage:
    python .cursor/skills/log-observability/scripts/inspect_training_steps.py \
        sandbox/runs/baseline_anchor

Outputs (per run):
- count of training_step records
- min/median/p95/max of training_stats/grad_pre_clip_norm
- mean clip fraction (running and windowed)
- min/median/max of each training_losses/<branch> and training_grad_norms/<branch>
- step at which any of {total_loss, grad_pre_clip_norm} first exceeds a divergence threshold

This is the offline source of truth for grad/clip behaviour; W&B has a higher-frequency
copy of the same numbers but is not always reachable.

Note: only runs with `training.training_stats_jsonl_every_n_steps > 0` (default 200 since
2026-04 schema bump) will have training_step rows. Older runs return zeros.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from typing import Any, Dict, List, Optional


def load_step_rows(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("kind") == "training_step":
                rows.append(obj)
    return rows


def extract(rows: List[Dict[str, Any]], family: str, key: str) -> List[float]:
    out: List[float] = []
    for r in rows:
        d = r.get(family, {}) or {}
        v = d.get(key)
        if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
            out.append(float(v))
    return out


def quantile(xs: List[float], q: float) -> Optional[float]:
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def first_exceeds(rows: List[Dict[str, Any]], family: str, key: str, threshold: float) -> Optional[int]:
    for r in rows:
        d = r.get(family, {}) or {}
        v = d.get(key)
        if isinstance(v, (int, float)) and v > threshold:
            return int(r.get("global_step", -1))
    return None


def summarize_run(run_dir: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"run_dir": run_dir}
    rows = load_step_rows(os.path.join(run_dir, "metrics.jsonl"))
    out["training_step_rows"] = len(rows)
    if not rows:
        return out

    out["first_step"] = rows[0].get("global_step")
    out["last_step"] = rows[-1].get("global_step")

    grad_pre = extract(rows, "training_stats", "training_stats/grad_pre_clip_norm")
    out["grad_pre_clip_norm"] = {
        "min": min(grad_pre) if grad_pre else None,
        "median": statistics.median(grad_pre) if grad_pre else None,
        "p95": quantile(grad_pre, 0.95),
        "max": max(grad_pre) if grad_pre else None,
        "n": len(grad_pre),
    }

    clipped_running = extract(rows, "training_stats", "training_stats/grad_clipped_frac_running")
    clipped_window = extract(rows, "training_stats", "training_stats/grad_clipped_frac_window")
    out["clip_fraction"] = {
        "running_mean": (sum(clipped_running) / len(clipped_running)) if clipped_running else None,
        "window_mean": (sum(clipped_window) / len(clipped_window)) if clipped_window else None,
    }

    total_loss = extract(rows, "training_stats", "training_stats/total_loss")
    if total_loss:
        out["total_loss"] = {
            "first": total_loss[0],
            "min": min(total_loss),
            "max": max(total_loss),
            "last": total_loss[-1],
            "n": len(total_loss),
        }
        # divergence: total_loss exceeded 2x its first value
        out["first_step_loss_2x_initial"] = first_exceeds(
            rows, "training_stats", "training_stats/total_loss", 2.0 * total_loss[0]
        )

    branches = ["count_obs", "count_imp", "pval_obs", "pval_imp", "peak_obs", "peak_imp"]
    losses_summary: Dict[str, Any] = {}
    grads_summary: Dict[str, Any] = {}
    for b in branches:
        ll = extract(rows, "training_losses", f"training_losses/{b}")
        gg = extract(rows, "training_grad_norms", f"training_grad_norms/{b}")
        if ll:
            losses_summary[b] = {
                "min": min(ll),
                "median": statistics.median(ll),
                "max": max(ll),
                "last": ll[-1],
            }
        if gg:
            grads_summary[b] = {
                "min": min(gg),
                "median": statistics.median(gg),
                "p95": quantile(gg, 0.95),
                "max": max(gg),
            }
    out["per_branch_losses"] = losses_summary
    out["per_branch_grad_norms"] = grads_summary
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    summaries = [summarize_run(d) for d in args.run_dirs]
    if args.json:
        json.dump({s["run_dir"]: s for s in summaries}, sys.stdout, indent=2, default=float)
        sys.stdout.write("\n")
        return 0
    for s in summaries:
        print(f"\n=== {s['run_dir']} ===")
        if s.get("training_step_rows", 0) == 0:
            print("  (no training_step rows; run predates training_stats_jsonl schema)")
            continue
        print(f"  steps_logged    : {s['training_step_rows']} (step {s['first_step']} -> {s['last_step']})")
        gp = s["grad_pre_clip_norm"]
        print(
            f"  grad_pre_clip   : min={gp['min']!r}  median={gp['median']!r}  "
            f"p95={gp['p95']!r}  max={gp['max']!r}"
        )
        cf = s["clip_fraction"]
        print(f"  clip_fraction   : running_mean={cf['running_mean']!r}  window_mean={cf['window_mean']!r}")
        if "total_loss" in s:
            tl = s["total_loss"]
            print(
                f"  total_loss      : first={tl['first']:.4f}  min={tl['min']:.4f}  "
                f"max={tl['max']:.4f}  last={tl['last']:.4f}"
            )
            if s.get("first_step_loss_2x_initial") is not None:
                print(f"  divergence flag : total_loss exceeded 2x initial at step {s['first_step_loss_2x_initial']}")
        for b, st in s.get("per_branch_grad_norms", {}).items():
            print(
                f"  grad_norm[{b}]   : median={st['median']:.3f}  p95={st['p95']:.3f}  max={st['max']:.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
