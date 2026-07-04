"""Rank sandbox runs by the Cornerstone Decision Rule.

This is the deterministic implementation of the ranking contract documented in
`.cursor/skills/log-observability/SKILL.md` (section "Cornerstone Decision Rule").

The ranker reads ONLY `metrics.jsonl` (epoch records, kind=="epoch" or legacy rows
without a `kind` field) from each run directory. Any field that is not on the
cornerstone list is ignored on purpose — see SKILL.md for the rationale.

Tiers, in order:
    Tier 2 (eligibility gate):
        - divergence_flag: last(eval_losses/total_loss) > 1.5 * best(eval_losses/total_loss)
        - nan_inf_count over eval_metrics/*, eval_metrics_median/*, and eval_losses/*
        - walltime_killed (caller-provided; we cannot infer this from metrics.jsonl alone)
    Tier 1 (primary quality at run's best epoch):
        quality_score = 2.0 * (pval_imp + count_imp + peak_imp)
                      + 1.0 * (pval_obs + count_obs + peak_obs)
        winner: lower quality_score by > 1% relative AND no individual imp loss
                worse by > 5% relative.
    Tier 1b (veto only): if winner's median imp pearson/spearman/auroc drops by > 10%
        relative vs runner-up, downgrade to "tied — investigate".
    Tier 4 (efficiency tiebreaker, only on Tier-1 ties):
        - global_step at min(eval_losses/total_loss)
        - mean epoch_seconds

Usage:
    python rank_runs.py runA runB [runC ...]
    python rank_runs.py --json runA runB
    python rank_runs.py --walltime-killed runB runA runB  # mark runB ineligible

Exit codes are advisory only; the human-readable report is the primary output.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# --- cornerstone field lists (mirror SKILL.md; keep in sync) -----------------

IMP_LOSSES = [
    "eval_losses/pval_imp_loss",
    "eval_losses/count_imp_loss",
    "eval_losses/peak_imp_loss",
]
OBS_LOSSES = [
    "eval_losses/pval_obs_loss",
    "eval_losses/count_obs_loss",
    "eval_losses/peak_obs_loss",
]
TIER1_LOSSES = IMP_LOSSES + OBS_LOSSES

TIER1B_VETO_METRICS = [  # imp metrics — drop > 10% rel vetoes a Tier-1 win
    "eval_metrics_median/imp_pval_pearson",
    "eval_metrics_median/imp_pval_spearman",
    "eval_metrics_median/imp_peak_auroc",
]
LEGACY_METRIC_FALLBACKS = {
    "eval_metrics_median/imp_pval_pearson": "eval_metrics/imp_pval_pearson_gw",
    "eval_metrics_median/imp_pval_spearman": "eval_metrics/imp_pval_spearman_gw",
    "eval_metrics_median/imp_peak_auroc": "eval_metrics/imp_peak_auroc_gw",
}

TOTAL_LOSS_KEY = "eval_losses/total_loss"

IMP_WEIGHT = 2.0
OBS_WEIGHT = 1.0
QUALITY_WIN_REL = 0.01      # > 1% relative on quality_score
PER_BRANCH_GUARD_REL = 0.05  # imp branch must not regress > 5% relative
VETO_REL = 0.10              # imp veto metric drop > 10% relative


# --- jsonl reading -----------------------------------------------------------

def _load_epoch_rows(run_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"metrics.jsonl not found in {run_dir}")
    rows: List[Dict[str, Any]] = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = rec.get("kind", "epoch")
            if kind == "epoch":
                rows.append(rec)
    return rows


def _is_finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _get(row: Dict[str, Any], key: str) -> Optional[float]:
    """Find `family/sub` key whether stored flat or nested under `family`."""
    if key in row:
        v = row[key]
        return float(v) if _is_finite(v) else None
    fam, _, sub = key.partition("/")
    family = row.get(fam)
    if isinstance(family, dict):
        if key in family and _is_finite(family[key]):
            return float(family[key])
        if sub in family and _is_finite(family[sub]):
            return float(family[sub])
    fallback = LEGACY_METRIC_FALLBACKS.get(key)
    if fallback is not None:
        return _get(row, fallback)
    return None


# --- per-run summary ---------------------------------------------------------

def summarize_run(run_dir: str, walltime_killed: bool = False) -> Dict[str, Any]:
    rows = _load_epoch_rows(run_dir)
    if not rows:
        return {"run_dir": run_dir, "eligible": False, "reason": "no epoch rows"}

    total_losses = [(_get(r, TOTAL_LOSS_KEY), r) for r in rows]
    total_losses_finite = [(v, r) for v, r in total_losses if v is not None]
    if not total_losses_finite:
        return {"run_dir": run_dir, "eligible": False, "reason": "no finite total_loss"}

    best_total, best_row = min(total_losses_finite, key=lambda x: x[0])
    last_total, _last_row = total_losses_finite[-1]

    # Tier 2 — divergence flag.
    diverged = last_total > 1.5 * best_total if best_total > 0 else False

    # Tier 2 — NaN/Inf count over eval_*.
    nan_inf = 0
    for r in rows:
        for k, v in r.items():
            if k in ("eval_metrics", "eval_metrics_median", "eval_losses") and isinstance(v, dict):
                for vv in v.values():
                    if isinstance(vv, (int, float)) and not math.isfinite(float(vv)):
                        nan_inf += 1
                continue
            if k.startswith(("eval_metrics/", "eval_metrics_median/", "eval_losses/")):
                if isinstance(v, (int, float)) and not math.isfinite(float(v)):
                    nan_inf += 1

    # Tier 1 — losses at best-total epoch.
    imp_at_best = [_get(best_row, k) for k in IMP_LOSSES]
    obs_at_best = [_get(best_row, k) for k in OBS_LOSSES]
    if any(v is None for v in imp_at_best + obs_at_best):
        # Insufficient data — flag, but keep run for human inspection.
        quality_score = None
    else:
        quality_score = (
            IMP_WEIGHT * sum(imp_at_best) + OBS_WEIGHT * sum(obs_at_best)
        )

    # Tier 1b — imp veto metrics at best-total epoch.
    imp_veto = {k: _get(best_row, k) for k in TIER1B_VETO_METRICS}

    # Tier 4 — efficiency.
    global_step_at_best = best_row.get("global_step")
    epoch_seconds_vals = [r.get("epoch_seconds") for r in rows if _is_finite(r.get("epoch_seconds"))]
    mean_epoch_seconds = (
        sum(epoch_seconds_vals) / len(epoch_seconds_vals) if epoch_seconds_vals else None
    )

    eligible = True
    reasons: List[str] = []
    if diverged:
        eligible = False
        reasons.append(f"diverged (last={last_total:.4g} > 1.5*best={best_total:.4g})")
    if nan_inf > 0:
        eligible = False
        reasons.append(f"nan_inf_count={nan_inf}")
    if walltime_killed:
        # Mark, but do not auto-disqualify — caller must ensure step-matched comparison.
        reasons.append("walltime_killed (caller asserted)")

    return {
        "run_dir": run_dir,
        "eligible": eligible,
        "reasons": reasons,
        "best_total_loss": best_total,
        "last_total_loss": last_total,
        "best_epoch": best_row.get("epoch"),
        "global_step_at_best": global_step_at_best,
        "mean_epoch_seconds": mean_epoch_seconds,
        "quality_score": quality_score,
        "imp_at_best": dict(zip(IMP_LOSSES, imp_at_best)),
        "obs_at_best": dict(zip(OBS_LOSSES, obs_at_best)),
        "imp_veto_at_best": imp_veto,
        "diverged": diverged,
        "nan_inf_count": nan_inf,
        "walltime_killed": walltime_killed,
    }


# --- pairwise decision -------------------------------------------------------

def _rel_delta(a: float, b: float) -> float:
    """Relative change of a vs b: positive means a is larger."""
    if b == 0:
        return float("inf") if a != 0 else 0.0
    return (a - b) / abs(b)


def decide_pair(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Return verdict for A vs B using the Cornerstone Decision Rule."""
    out = {"a": a["run_dir"], "b": b["run_dir"]}

    if not a["eligible"] or not b["eligible"]:
        out["verdict"] = "ineligible"
        out["why"] = {
            "a_reasons": a.get("reasons", [a.get("reason", "unknown")]),
            "b_reasons": b.get("reasons", [b.get("reason", "unknown")]),
        }
        return out

    qa, qb = a["quality_score"], b["quality_score"]
    if qa is None or qb is None:
        out["verdict"] = "insufficient_data"
        out["why"] = "missing one or more Tier 1 losses"
        return out

    # Tier 1: lower quality_score wins by > 1% rel.
    if abs(_rel_delta(qa, qb)) <= QUALITY_WIN_REL:
        # Tie on quality — defer to Tier 4 efficiency.
        return _efficiency_tiebreak(a, b, out, qa, qb)

    winner, loser = (a, b) if qa < qb else (b, a)
    w_imp = winner["imp_at_best"]
    l_imp = loser["imp_at_best"]

    # Per-branch imp guard: winner must not be > 5% rel worse on any imp loss.
    branch_violations = []
    for k in IMP_LOSSES:
        wv, lv = w_imp[k], l_imp[k]
        if wv is None or lv is None:
            continue
        if _rel_delta(wv, lv) > PER_BRANCH_GUARD_REL:
            branch_violations.append({"branch": k, "winner": wv, "loser": lv,
                                       "rel_delta": _rel_delta(wv, lv)})
    if branch_violations:
        out["verdict"] = "no_winner_branch_guard"
        out["why"] = {
            "tentative_winner": winner["run_dir"],
            "violations": branch_violations,
        }
        return out

    # Tier 1b veto: any imp veto metric drop > 10% rel on winner vs loser.
    veto_drops = []
    for k in TIER1B_VETO_METRICS:
        wv = winner["imp_veto_at_best"].get(k)
        lv = loser["imp_veto_at_best"].get(k)
        if wv is None or lv is None:
            continue
        # Higher is better for these metrics → drop = (winner - loser) / |loser| < -0.10
        if _rel_delta(wv, lv) < -VETO_REL:
            veto_drops.append({"metric": k, "winner": wv, "loser": lv,
                               "rel_delta": _rel_delta(wv, lv)})
    if veto_drops:
        out["verdict"] = "tied_investigate"
        out["why"] = {
            "tentative_winner": winner["run_dir"],
            "veto_drops": veto_drops,
        }
        return out

    out["verdict"] = "winner"
    out["winner"] = winner["run_dir"]
    out["loser"] = loser["run_dir"]
    out["quality_scores"] = {a["run_dir"]: qa, b["run_dir"]: qb}
    return out


def _efficiency_tiebreak(a: Dict[str, Any], b: Dict[str, Any],
                         out: Dict[str, Any], qa: float, qb: float) -> Dict[str, Any]:
    out["verdict"] = "tier1_tie"
    out["quality_scores"] = {a["run_dir"]: qa, b["run_dir"]: qb}
    sa, sb = a.get("global_step_at_best"), b.get("global_step_at_best")
    ea, eb = a.get("mean_epoch_seconds"), b.get("mean_epoch_seconds")
    notes = []
    if sa is not None and sb is not None:
        notes.append(f"global_step_at_best: {a['run_dir']}={sa} vs {b['run_dir']}={sb}")
    if ea is not None and eb is not None:
        notes.append(f"mean_epoch_seconds: {a['run_dir']}={ea:.1f} vs {b['run_dir']}={eb:.1f}")
    out["efficiency_notes"] = notes
    return out


# --- CLI ---------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", nargs="+",
                   help="Sandbox run directories containing metrics.jsonl")
    p.add_argument("--walltime-killed", action="append", default=[],
                   help="Run dir(s) that were killed by SLURM walltime; "
                        "repeat the flag per run.")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of human report")
    args = p.parse_args()

    walltime_killed = set(os.path.normpath(d) for d in args.walltime_killed)
    summaries = []
    for rd in args.run_dirs:
        try:
            s = summarize_run(rd, walltime_killed=os.path.normpath(rd) in walltime_killed)
        except FileNotFoundError as e:
            s = {"run_dir": rd, "eligible": False, "reason": str(e)}
        summaries.append(s)

    pairs: List[Dict[str, Any]] = []
    for i in range(len(summaries)):
        for j in range(i + 1, len(summaries)):
            pairs.append(decide_pair(summaries[i], summaries[j]))

    report = {"summaries": summaries, "pairs": pairs}

    if args.json:
        json.dump(report, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return 0

    for s in summaries:
        print(f"\n=== {s['run_dir']} ===")
        if not s.get("eligible", False):
            print(f"  INELIGIBLE: {s.get('reasons') or s.get('reason')}")
            continue
        qs = s["quality_score"]
        qs_str = f"{qs:.4f}" if qs is not None else "n/a"
        print(f"  quality_score = {qs_str}  "
              f"(best_total_loss={s['best_total_loss']:.4f} @ epoch {s['best_epoch']})")
        for k, v in s["imp_at_best"].items():
            print(f"    {k:38s} = {v}")
        for k, v in s["obs_at_best"].items():
            print(f"    {k:38s} = {v}")
        if s["reasons"]:
            print(f"  notes: {s['reasons']}")

    print("\n--- Pairwise verdicts ---")
    for pv in pairs:
        v = pv["verdict"]
        if v == "winner":
            print(f"  {pv['winner']}  >  {pv['loser']}    "
                  f"({pv['quality_scores']})")
        elif v == "tier1_tie":
            print(f"  {pv['a']}  ~=  {pv['b']}    {pv.get('efficiency_notes')}")
        elif v == "tied_investigate":
            print(f"  {pv['a']}  ?  {pv['b']}    veto: "
                  f"{pv['why']['veto_drops']}")
        elif v == "no_winner_branch_guard":
            print(f"  {pv['a']}  X  {pv['b']}    branch guard: "
                  f"{pv['why']['violations']}")
        else:
            print(f"  {pv['a']} vs {pv['b']}: {v} {pv.get('why', '')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
