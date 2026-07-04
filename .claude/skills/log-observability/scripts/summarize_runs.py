"""Summarize one or more sandbox run directories.

Usage:
    python .cursor/skills/log-observability/scripts/summarize_runs.py \
        sandbox/runs/baseline_anchor sandbox/runs/baseline_dsf1_only ...

Reads `metrics.jsonl` from each directory and prints:
- per-run progress (epochs, steps, sec/epoch, NaN/Inf counts)
- per-run first/last/best for selected eval metrics and losses
- per-run divergence flag (max(total_loss) > 1.5 x best total_loss)
- per-run depth_count_ratio status (target ~4.0; ~1.0 means depth-invariant)

This script reads ONLY metrics.jsonl. It deliberately avoids inference of grad-norm
behaviour because grad-norms only land in metrics.jsonl when the run was launched with
`training.training_stats_jsonl_every_n_steps > 0` (default 200). For older runs that
predate that schema, use scripts/inspect_training_steps.py instead.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional

KEY_METRICS = [
    "eval_metrics_median/den_pval_pearson",
    "eval_metrics_median/den_pval_spearman",
    "eval_metrics_median/den_pval_r2",
    "eval_metrics_median/den_count_pearson",
    "eval_metrics_median/den_count_spearman",
    "eval_metrics_median/den_count_r2",
    "eval_metrics_median/den_peak_auroc",
    "eval_metrics_median/imp_pval_pearson",
    "eval_metrics_median/imp_pval_spearman",
    "eval_metrics_median/imp_pval_r2",
    "eval_metrics_median/imp_count_pearson",
    "eval_metrics_median/imp_count_r2",
    "eval_metrics_median/imp_peak_auroc",
]
LEGACY_METRIC_FALLBACKS = {
    "eval_metrics_median/den_pval_pearson": "eval_metrics/den_pval_pearson_gw",
    "eval_metrics_median/den_pval_spearman": "eval_metrics/den_pval_spearman_gw",
    "eval_metrics_median/den_pval_r2": "eval_metrics/den_pval_r2_gw",
    "eval_metrics_median/den_count_pearson": "eval_metrics/den_count_pearson_gw",
    "eval_metrics_median/den_count_spearman": "eval_metrics/den_count_spearman_gw",
    "eval_metrics_median/den_count_r2": "eval_metrics/den_count_r2_gw",
    "eval_metrics_median/den_peak_auroc": "eval_metrics/den_peak_auroc_gw",
    "eval_metrics_median/imp_pval_pearson": "eval_metrics/imp_pval_pearson_gw",
    "eval_metrics_median/imp_pval_spearman": "eval_metrics/imp_pval_spearman_gw",
    "eval_metrics_median/imp_pval_r2": "eval_metrics/imp_pval_r2_gw",
    "eval_metrics_median/imp_count_pearson": "eval_metrics/imp_count_pearson_gw",
    "eval_metrics_median/imp_count_r2": "eval_metrics/imp_count_r2_gw",
    "eval_metrics_median/imp_peak_auroc": "eval_metrics/imp_peak_auroc_gw",
}
LOSS_KEYS = [
    "eval_losses/total_loss",
    "eval_losses/count_obs_loss",
    "eval_losses/count_imp_loss",
    "eval_losses/pval_obs_loss",
    "eval_losses/pval_imp_loss",
    "eval_losses/peak_obs_loss",
    "eval_losses/peak_imp_loss",
]
PROBE_KEYS = [
    "training_metadata_probes/depth_count_ratio",
    "training_metadata_probes/runtype_mse",
    "training_metadata_probes/readlen_mse",
]

# Per the autoresearch contract: 1obs slices are too sparse for reliable R2 -> excluded
# from the success-criteria short list (still printed individually if needed).
DROP_FROM_SUCCESS = {
    "eval_metrics/den_pval_r2_1obs",
    "eval_metrics/imp_pval_r2_1obs",
}


def load_rows(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                rows.append({"_parse_error": str(e)})
    return rows


def look(row: Dict[str, Any], key: str) -> Optional[float]:
    """Find `family/sub` key whether stored flat or nested.

    Records use either:
      - flat:  row["eval_metrics_median/den_pval_pearson"] = ...
      - nested: row["eval_metrics_median"]["eval_metrics_median/den_pval_pearson"] = ...
      - short-nested: row["eval_metrics_median"]["den_pval_pearson"] = ...
    """
    if key in row:
        v = row[key]
        return float(v) if isinstance(v, (int, float)) else None
    fam, _, sub = key.partition("/")
    family = row.get(fam)
    if isinstance(family, dict):
        if key in family:
            v = family[key]
            return float(v) if isinstance(v, (int, float)) else None
        if sub in family:
            v = family[sub]
            return float(v) if isinstance(v, (int, float)) else None
    fallback = LEGACY_METRIC_FALLBACKS.get(key)
    if fallback is not None:
        return look(row, fallback)
    return None


def summarize(run_dir: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"run_dir": run_dir}
    mpath = os.path.join(run_dir, "metrics.jsonl")
    out["metrics_exists"] = os.path.exists(mpath)
    out["config_exists"] = os.path.exists(os.path.join(run_dir, "resolved_config.yaml"))
    if not out["metrics_exists"]:
        return out

    rows = load_rows(mpath)
    epoch_rows = [r for r in rows if r.get("kind", "epoch") == "epoch"]
    step_rows = [r for r in rows if r.get("kind") == "training_step"]
    out["epoch_rows"] = len(epoch_rows)
    out["training_step_rows"] = len(step_rows)

    if not epoch_rows:
        return out

    out["epoch_first"] = epoch_rows[0].get("epoch")
    out["epoch_last"] = epoch_rows[-1].get("epoch")
    out["global_step_last"] = epoch_rows[-1].get("global_step")
    secs = [r.get("epoch_seconds") for r in epoch_rows if isinstance(r.get("epoch_seconds"), (int, float))]
    out["epoch_seconds_mean"] = sum(secs) / len(secs) if secs else None
    out["epoch_seconds_total"] = sum(secs) if secs else None

    # NaN/Inf scan across eval families
    bad = 0
    for r in epoch_rows:
        for fam in ("eval_metrics", "eval_metrics_median", "eval_losses", "training_metadata_probes"):
            d = r.get(fam, {}) or {}
            if not isinstance(d, dict):
                continue
            for v in d.values():
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    bad += 1
    out["nan_inf_count"] = bad

    metric_summary: Dict[str, Dict[str, Any]] = {}
    for k in KEY_METRICS + LOSS_KEYS + PROBE_KEYS:
        if k in DROP_FROM_SUCCESS:
            continue
        vals = [look(r, k) for r in epoch_rows]
        vals_f = [v for v in vals if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))]
        if not vals_f:
            metric_summary[k] = {"first": None, "last": None, "best": None, "n": 0}
            continue
        is_loss_or_mse = ("loss" in k) or ("_mse" in k)
        metric_summary[k] = {
            "first": vals_f[0],
            "last": vals_f[-1],
            "best": (min if is_loss_or_mse else max)(vals_f),
            "n": len(vals_f),
        }
    out["metrics"] = metric_summary

    # Divergence flag: max total_loss > 1.5 * best total_loss
    total_summary = metric_summary.get("eval_losses/total_loss", {})
    out["diverged"] = (
        total_summary.get("best") is not None
        and total_summary.get("last") is not None
        and total_summary["last"] > 1.5 * total_summary["best"]
    )

    # depth_count_ratio status: target ~4.0
    dcr = metric_summary.get("training_metadata_probes/depth_count_ratio", {})
    last_dcr = dcr.get("last")
    if last_dcr is None:
        out["depth_metadata_status"] = "unknown"
    elif last_dcr < 1.5:
        out["depth_metadata_status"] = "ignored"  # near 1.0 -> depth-invariant
    elif 3.0 <= last_dcr <= 5.0:
        out["depth_metadata_status"] = "healthy"  # near 4.0
    else:
        out["depth_metadata_status"] = f"off_target({last_dcr:.2f})"

    return out


def fmt(v: Any) -> str:
    if v is None:
        return "  -   "
    if isinstance(v, bool):
        return "yes" if v else " no"
    if isinstance(v, float):
        if abs(v) >= 1000 or (v != 0 and abs(v) < 0.0001):
            return f"{v:8.2e}"
        return f"{v:8.4f}"
    return str(v)


def print_table(summaries: List[Dict[str, Any]]) -> None:
    print("\n## Per-run progress\n")
    print(
        f"{'run':40s} {'epochs':>7s} {'last_step':>10s} {'sec/epoch':>10s} "
        f"{'nan/inf':>8s} {'diverged':>10s} {'depth_meta':>14s}"
    )
    for s in summaries:
        print(
            f"{os.path.basename(s['run_dir']):40s} "
            f"{fmt(s.get('epoch_rows'))} "
            f"{fmt(s.get('global_step_last'))} "
            f"{fmt(s.get('epoch_seconds_mean'))} "
            f"{fmt(s.get('nan_inf_count'))} "
            f"{fmt(s.get('diverged')):>10s} "
            f"{str(s.get('depth_metadata_status', '-')):>14s}"
        )

    def cmp_metric(key: str) -> None:
        print(f"\n### {key}\n")
        print(f"{'run':40s} {'first':>10s} {'last':>10s} {'best':>10s} {'n':>4s}")
        for s in summaries:
            m = s.get("metrics", {}).get(key, {})
            print(
                f"{os.path.basename(s['run_dir']):40s} "
                f"{fmt(m.get('first'))} {fmt(m.get('last'))} {fmt(m.get('best'))} "
                f"{fmt(m.get('n'))}"
            )

    for k in [
        "eval_losses/total_loss",
        "eval_metrics_median/den_pval_pearson",
        "eval_metrics_median/den_pval_spearman",
        "eval_metrics_median/den_pval_r2",
        "eval_metrics_median/den_count_pearson",
        "eval_metrics_median/den_peak_auroc",
        "eval_metrics_median/imp_pval_pearson",
        "eval_metrics_median/imp_count_pearson",
        "eval_metrics_median/imp_peak_auroc",
        "eval_losses/pval_obs_loss",
        "eval_losses/peak_obs_loss",
        "training_metadata_probes/depth_count_ratio",
        "training_metadata_probes/runtype_mse",
        "training_metadata_probes/readlen_mse",
    ]:
        cmp_metric(k)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("run_dirs", nargs="+", help="One or more sandbox run directories.")
    ap.add_argument(
        "--json", action="store_true",
        help="Emit a JSON object {run_dir: summary} on stdout instead of a human table.",
    )
    args = ap.parse_args()

    summaries = [summarize(d) for d in args.run_dirs]
    if args.json:
        json.dump({s["run_dir"]: s for s in summaries}, sys.stdout, indent=2, default=float)
        sys.stdout.write("\n")
        return 0
    print_table(summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
