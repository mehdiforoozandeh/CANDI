"""HPO graph log: read/write/upsert of ``sandbox/hpo_graph.json``.

The file is a single JSON document — easy to diff, version-control, and embed
in a publication. All access goes through this module so the schema can be
extended in one place.

Schema (``schema_version=1``)::

    {
      "schema_version": 1,
      "nodes": {
          "<run_id>": {
              "run_id": str,
              "run_dir": str,
              "slurm_job_id": Optional[str],
              "wandb_run_name": Optional[str],
              "created_at": ISO-8601,
              "finished_at": ISO-8601,
              "elapsed_seconds": float,
              "epochs_completed": int,
              "global_step_last": int,
              "diverged": bool,
              "nan_inf_count": int,
              "config_axes": {axis_path: value, ...},  # see hpo/axes.py
              "results_at_best_epoch": {
                  "best_epoch": int,
                  "best_total_loss": float,
                  "last_total_loss": float,
                  "quality_score": float,
                  "eval_losses/<branch>_loss": float, ... (6 branches),
                  "eval_metrics/<key>": float, ... (Tier 1b set),
              },
              "experiment_label": str,
              "parent_run_ids": [run_id, ...],
              "notes": str,
          }
      },
      "edges": [
          {"from": parent_id, "to": child_id, "diff": {axis: [parent_v, child_v]}}
      ]
    }

Updates are atomic (write-temp + rename) and best-effort: if anything goes
wrong while updating the graph, the run still exits 0 — the trainer logs the
failure and moves on.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sandbox.hpo.axes import diff as axes_diff
from sandbox.hpo.axes import extract_axes

GRAPH_SCHEMA_VERSION = 1

# Tier-1 cornerstone losses — kept in lockstep with the log-observability skill.
_IMP_LOSSES = (
    "eval_losses/pval_imp_loss",
    "eval_losses/count_imp_loss",
    "eval_losses/peak_imp_loss",
)
_OBS_LOSSES = (
    "eval_losses/pval_obs_loss",
    "eval_losses/count_obs_loss",
    "eval_losses/peak_obs_loss",
)
_TIER1B_VETO_METRICS = (
    "eval_metrics/imp_pval_pearson_gw",
    "eval_metrics/imp_pval_spearman_gw",
    "eval_metrics/imp_peak_auroc_gw",
)
_TIER1B_FLAG_METRICS = (
    "eval_metrics/den_pval_pearson_gw",
    "eval_metrics/den_pval_spearman_gw",
    "eval_metrics/den_peak_auroc_gw",
    "eval_metrics/imp_count_pearson_gw",
    "eval_metrics/den_count_pearson_gw",
)
_IMP_WEIGHT = 2.0
_OBS_WEIGHT = 1.0


# ── Generic JSON I/O ─────────────────────────────────────────────────────────

def load_graph(path: Path) -> Dict[str, Any]:
    """Load the graph file or return an empty skeleton if it doesn't exist."""
    p = Path(path)
    if not p.exists():
        return {"schema_version": GRAPH_SCHEMA_VERSION, "nodes": {}, "edges": []}
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[hpo] warning: graph file {p} unreadable ({e}); starting fresh", file=sys.stderr)
        return {"schema_version": GRAPH_SCHEMA_VERSION, "nodes": {}, "edges": []}
    data.setdefault("schema_version", GRAPH_SCHEMA_VERSION)
    data.setdefault("nodes", {})
    data.setdefault("edges", [])
    return data


def save_graph(path: Path, data: Dict[str, Any]) -> None:
    """Atomic write: temp file in the same dir + rename."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".hpo_graph.", suffix=".json", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=False, default=float)
            f.write("\n")
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


# ── Pulling results out of metrics.jsonl ─────────────────────────────────────

def _get_nested(row: Dict[str, Any], key: str) -> Optional[float]:
    """Read ``family/sub`` from either flat or nested-dict form (mirrors log-observability)."""
    if key in row:
        v = row[key]
        return float(v) if isinstance(v, (int, float)) and math.isfinite(float(v)) else None
    fam, _, sub = key.partition("/")
    family = row.get(fam)
    if isinstance(family, dict):
        if key in family and isinstance(family[key], (int, float)) and math.isfinite(family[key]):
            return float(family[key])
        if sub in family and isinstance(family[sub], (int, float)) and math.isfinite(family[sub]):
            return float(family[sub])
    return None


def _load_epoch_rows(metrics_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not metrics_path.exists():
        return rows
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("kind", "epoch") == "epoch":
                rows.append(rec)
    return rows


def _summarize_run(metrics_path: Path) -> Dict[str, Any]:
    """Compute best-epoch results + Tier 2 stability flags from ``metrics.jsonl``."""
    rows = _load_epoch_rows(metrics_path)
    out: Dict[str, Any] = {
        "epochs_completed": 0,
        "global_step_last": 0,
        "diverged": None,
        "nan_inf_count": 0,
        "results_at_best_epoch": {},
    }
    if not rows:
        return out
    out["epochs_completed"] = len(rows)
    last = rows[-1]
    out["global_step_last"] = int(last.get("global_step", 0) or 0)

    # NaN/Inf scan over eval families.
    for r in rows:
        for fam in ("eval_metrics", "eval_losses"):
            d = r.get(fam, {}) or {}
            if not isinstance(d, dict):
                continue
            for v in d.values():
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    out["nan_inf_count"] += 1

    # Best epoch by minimum eval_losses/total_loss.
    candidates: List[Tuple[float, Dict[str, Any]]] = []
    for r in rows:
        v = _get_nested(r, "eval_losses/total_loss")
        if v is not None:
            candidates.append((v, r))
    if not candidates:
        out["diverged"] = None
        return out

    best_total, best_row = min(candidates, key=lambda x: x[0])
    last_total = candidates[-1][0]
    out["diverged"] = last_total > 1.5 * best_total if best_total > 0 else False

    res: Dict[str, Any] = {
        "best_epoch": int(best_row.get("epoch", -1)),
        "best_total_loss": float(best_total),
        "last_total_loss": float(last_total),
    }
    imp_vals = [_get_nested(best_row, k) for k in _IMP_LOSSES]
    obs_vals = [_get_nested(best_row, k) for k in _OBS_LOSSES]
    for k, v in zip(_IMP_LOSSES + _OBS_LOSSES, imp_vals + obs_vals):
        res[k] = v
    if all(v is not None for v in imp_vals + obs_vals):
        res["quality_score"] = float(
            _IMP_WEIGHT * sum(imp_vals) + _OBS_WEIGHT * sum(obs_vals)
        )
    else:
        res["quality_score"] = None
    for k in _TIER1B_VETO_METRICS + _TIER1B_FLAG_METRICS:
        res[k] = _get_nested(best_row, k)
    out["results_at_best_epoch"] = res
    return out


# ── Public update entrypoint (called by sandbox.train) ───────────────────────

def diff_axes(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, List[Any]]:
    """Re-export of :func:`sandbox.hpo.axes.diff` for convenience."""
    return axes_diff(a, b)


def _utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def update_graph_for_run(
    *,
    run_id: str,
    run_dir: Path,
    resolved_cfg_dict: Dict[str, Any],
    parent_run_ids: List[str],
    experiment_label: str,
    notes: str,
    elapsed_seconds: float,
    slurm_job_id: Optional[str],
    wandb_run_name: Optional[str],
    graph_path: Path,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Upsert one node + parent edges into the HPO graph and persist atomically.

    Returns the resulting node dict. On any failure the function logs to stderr
    and re-raises (caller is expected to swallow exceptions so a graph error
    never fails the run).
    """
    from sandbox.hpo.axes import extract_axes  # local re-import to make module self-contained

    metrics_path = Path(run_dir) / "metrics.jsonl"
    summary = _summarize_run(metrics_path)
    axes = extract_axes(resolved_cfg_dict)

    node: Dict[str, Any] = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "slurm_job_id": str(slurm_job_id) if slurm_job_id else None,
        "wandb_run_name": wandb_run_name,
        "created_at": created_at or _utc_iso(),
        "finished_at": _utc_iso(),
        "elapsed_seconds": float(elapsed_seconds),
        "epochs_completed": int(summary["epochs_completed"]),
        "global_step_last": int(summary["global_step_last"]),
        "diverged": summary["diverged"],
        "nan_inf_count": int(summary["nan_inf_count"]),
        "config_axes": axes,
        "results_at_best_epoch": summary["results_at_best_epoch"],
        "experiment_label": str(experiment_label or ""),
        "parent_run_ids": list(parent_run_ids or []),
        "notes": str(notes or ""),
    }

    graph = load_graph(graph_path)
    graph["nodes"][run_id] = node

    # Rebuild this run's outgoing edges (i.e. parent → this node).
    # Drop any existing edges pointing into this run, then add fresh ones based
    # on the declared parents — keeps the file canonical when a run is rerun.
    graph["edges"] = [e for e in graph["edges"] if e.get("to") != run_id]
    for parent_id in node["parent_run_ids"]:
        parent_node = graph["nodes"].get(parent_id)
        if parent_node is None:
            edge: Dict[str, Any] = {"from": parent_id, "to": run_id, "diff": None,
                                     "note": "parent not yet in graph"}
        else:
            edge = {
                "from": parent_id,
                "to": run_id,
                "diff": axes_diff(parent_node.get("config_axes", {}), axes),
            }
        graph["edges"].append(edge)

    save_graph(graph_path, graph)
    return node


__all__ = [
    "GRAPH_SCHEMA_VERSION",
    "diff_axes",
    "load_graph",
    "save_graph",
    "update_graph_for_run",
]
