#!/usr/bin/env python3
"""Tabulate JEPA metrics with paper-grounded v2 gate and ranking."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MASK_FRAC_THRESHOLD = 0.05

# key, label, higher_is_better, spike_filter_best
METRICS = [
    ("lejepa/combined_loss_scaled", "combined_loss", False, True),
    ("lejepa/pred_loss", "pred_loss", False, True),
    ("lejepa/sigreg_loss", "sigreg_loss", False, False),
    ("lejepa/encoder_eff_rank", "enc_er", True, False),
    ("lejepa/enc_er_delta", "enc_er_delta", None, False),
    ("lejepa/cov_condition_number", "cov_cond", False, False),
    ("lejepa/embedding_mean_norm", "mean_norm", False, False),
    ("lejepa/per_dim_variance_cv", "var_cv", False, False),
    ("lejepa/sigreg_projection_std", "sig_proj_std", False, False),
    ("lejepa/sigreg_converged", "sig_converged", True, False),
    ("lejepa/pred_loss_slope", "pred_slope", None, False),
    # secondary diagnostics kept for backward compatibility
    ("lejepa/cos_sim_ctx_tgt", "cos_sim", False, True),
    ("lejepa/adaLN_gamma_norm", "gamma_norm", True, True),
    ("lejepa/meta_sens_runtype", "runtype_sens", True, False),
    ("lejepa/meta_sens_depth", "depth_sens", True, False),
    ("lejepa/meta_sens_depth_wide", "depth_wide_sens", True, False),
    ("lejepa/meta_sens_readlen", "readlen_sens", True, False),
    ("lejepa/latent_eff_rank", "lat_er", True, False),
    ("lejepa/latent_std_mean", "lat_std_mean", None, False),
    ("lejepa/latent_n_dead", "lat_n_dead", None, False),
]


def load_jepa_steps(
    run_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, List[Dict[str, Any]]]:
    jsonl_path = run_dir / "metrics.jsonl"
    if not jsonl_path.exists():
        return [], [], 0, []
    all_rows: List[Dict[str, Any]] = []
    clean_rows: List[Dict[str, Any]] = []
    epoch_rows: List[Dict[str, Any]] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("kind") == "epoch":
                epoch_rows.append(d)
                continue
            if d.get("kind") != "training_step":
                continue
            lejepa = d.get("lejepa")
            if not isinstance(lejepa, dict):
                continue
            all_rows.append(lejepa)
            if float(lejepa.get("lejepa/mask_frac", 1.0)) >= MASK_FRAC_THRESHOLD:
                clean_rows.append(lejepa)
    return all_rows, clean_rows, len(all_rows) - len(clean_rows), epoch_rows


def get_val(row: Dict[str, Any], key: str) -> Optional[float]:
    v = row.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def summarize_metric(
    all_rows: List[Dict[str, Any]],
    clean_rows: List[Dict[str, Any]],
    key: str,
    higher_is_better: Optional[bool],
    spike_filter_best: bool,
) -> Dict[str, Optional[float]]:
    all_vals = [(i, get_val(r, key)) for i, r in enumerate(all_rows)]
    all_vals = [(i, v) for i, v in all_vals if v is not None]
    if not all_vals:
        return {"first": None, "best": None, "best_step_idx": None, "last": None}
    first_val = all_vals[0][1]
    last_val = all_vals[-1][1]
    if spike_filter_best and clean_rows:
        clean_vals = [(i, get_val(r, key)) for i, r in enumerate(clean_rows)]
        clean_vals = [(i, v) for i, v in clean_vals if v is not None]
        if clean_vals:
            last_val = clean_vals[-1][1]
    best_pool = clean_rows if spike_filter_best else all_rows
    best_vals = [(i, get_val(r, key)) for i, r in enumerate(best_pool)]
    best_vals = [(i, v) for i, v in best_vals if v is not None]
    if not best_vals:
        return {"first": first_val, "best": None, "best_step_idx": None, "last": last_val}
    if higher_is_better is True:
        best_idx, best_val = max(best_vals, key=lambda x: x[1])
    elif higher_is_better is False:
        best_idx, best_val = min(best_vals, key=lambda x: x[1])
    else:
        best_idx, best_val = best_vals[-1]
    return {"first": first_val, "best": best_val, "best_step_idx": best_idx, "last": last_val}


def geometry_gate_v2(stats: Dict[str, Dict[str, Optional[float]]]) -> str:
    issues: List[str] = []
    combined_last = stats.get("lejepa/combined_loss_scaled", {}).get("last")
    pred_slope = stats.get("lejepa/pred_loss_slope", {}).get("last")
    sig_conv = stats.get("lejepa/sigreg_converged", {}).get("last")
    enc_last = stats.get("lejepa/encoder_eff_rank", {}).get("last")
    cov_last = stats.get("lejepa/cov_condition_number", {}).get("last")

    if combined_last is None:
        issues.append("missing combined_loss_scaled")
    if pred_slope is None or pred_slope > 0.0:
        issues.append(f"pred_slope={pred_slope} > 0")
    if sig_conv is None or sig_conv < 0.5:
        issues.append(f"sigreg_converged={sig_conv} < 1")
    if enc_last is None or enc_last < 15.0:
        issues.append(f"enc_er_last={enc_last} < 15")
    if cov_last is None or cov_last >= 50.0:
        issues.append(f"cov_condition_number={cov_last} >= 50")

    return "PASS" if not issues else "FAIL: " + "; ".join(issues)


def geometry_gate_legacy(stats: Dict[str, Dict[str, Optional[float]]]) -> str:
    enc_last = stats.get("lejepa/encoder_eff_rank", {}).get("last")
    cos_best = stats.get("lejepa/cos_sim_ctx_tgt", {}).get("best")
    gamma_last = stats.get("lejepa/adaLN_gamma_norm", {}).get("last")
    if enc_last is None or cos_best is None or gamma_last is None:
        return "UNKNOWN (missing data)"
    issues = []
    if enc_last < 18.0:
        issues.append(f"enc_er_last={enc_last:.1f} < 18")
    if cos_best > 0.15:
        issues.append(f"cos_sim_best={cos_best:.3f} > 0.15")
    if gamma_last < 100.0:
        issues.append(f"gamma_last={gamma_last:.1f} < 100")
    return "PASS" if not issues else "FAIL: " + "; ".join(issues)


def collapse_warnings(stats: Dict[str, Dict[str, Optional[float]]]) -> List[str]:
    warnings: List[str] = []
    sig_last = stats.get("lejepa/sigreg_loss", {}).get("last")
    cov_last = stats.get("lejepa/cov_condition_number", {}).get("last")
    mean_norm_last = stats.get("lejepa/embedding_mean_norm", {}).get("last")
    pred_slope_last = stats.get("lejepa/pred_loss_slope", {}).get("last")
    enc_delta_last = stats.get("lejepa/enc_er_delta", {}).get("last")
    runtype_best = stats.get("lejepa/meta_sens_runtype", {}).get("best")
    runtype_last = stats.get("lejepa/meta_sens_runtype", {}).get("last")

    if sig_last is not None and sig_last > 3.0:
        warnings.append(f"[SIGREG] sigreg_loss_last={sig_last:.3f} > 3.0 (weak isotropy)")
    if cov_last is not None and cov_last > 100.0:
        warnings.append(f"[COLLAPSE] cov_condition_number_last={cov_last:.2f} > 100")
    if mean_norm_last is not None and mean_norm_last > 2.0:
        warnings.append(f"[CENTERING] embedding_mean_norm_last={mean_norm_last:.3f} > 2.0")
    if pred_slope_last is not None and pred_slope_last > 0.0:
        warnings.append(f"[MONOTONICITY] pred_loss_slope_last={pred_slope_last:.3f} > 0")
    if enc_delta_last is not None and enc_delta_last < -0.5:
        warnings.append(f"[COLLAPSE-ONSET] enc_er_delta_last={enc_delta_last:.3f} < -0.5")
    if runtype_best is not None and runtype_last is not None:
        ratio = runtype_last / max(runtype_best, 1e-6)
        if ratio < 0.4:
            warnings.append(f"[BIOLOGY-DEGRADE] runtype_last/best={ratio:.2f} < 0.40")
    return warnings


def analyze_run(run_dir: Path) -> Dict[str, Optional[float]]:
    all_rows, clean_rows, spike_count, epoch_rows = load_jepa_steps(run_dir)
    print(f"\n{'=' * 70}")
    print(f"RUN: {run_dir.name}")
    print(f"  Training steps: {len(all_rows)} total, {spike_count} spike steps")
    if not all_rows:
        print("  ERROR: no valid training_step rows found")
        return {"run": run_dir.name, "rank_score": None}

    stats: Dict[str, Dict[str, Optional[float]]] = {}
    for key, _, hib, sfb in METRICS:
        stats[key] = summarize_metric(all_rows, clean_rows, key, hib, sfb)

    print(f"\n  {'Metric':<22} {'First':>10} {'Best':>10} {'Last':>10}")
    print(f"  {'-' * 54}")
    for key, label, _, _ in METRICS:
        s = stats[key]
        first = f"{s['first']:.4f}" if s["first"] is not None else "  N/A  "
        best = f"{s['best']:.4f}" if s["best"] is not None else "  N/A  "
        last = f"{s['last']:.4f}" if s["last"] is not None else "  N/A  "
        print(f"  {label:<22} {first:>10} {best:>10} {last:>10}")

    print(f"\n  Geometry gate (v2): {geometry_gate_v2(stats)}")
    print(f"  Geometry gate (legacy): {geometry_gate_legacy(stats)}")

    if epoch_rows:
        hi_lo = [r["lejepa/pred_loss_hi_lo_ratio"] for r in epoch_rows if "lejepa/pred_loss_hi_lo_ratio" in r]
        if hi_lo:
            print(f"  pred_loss hi/lo ratio (secondary): {hi_lo[-1]:.3f}")

    warnings = collapse_warnings(stats)
    if warnings:
        print("\n  Warnings:")
        for w in warnings:
            print(f"    {w}")

    # Stage 2 checkpoint by minimum combined_loss_scaled
    step_idx = stats.get("lejepa/combined_loss_scaled", {}).get("best_step_idx")
    best_combined = stats.get("lejepa/combined_loss_scaled", {}).get("best")
    if step_idx is not None and best_combined is not None:
        approx_epoch = step_idx * 200 / 125
        print(
            f"\n  Stage 2 checkpoint (v2): step_idx {step_idx} "
            f"(≈ epoch {approx_epoch:.0f}), combined_loss_scaled={best_combined:.4f}"
        )

    return {
        "run": run_dir.name,
        "rank_score": stats.get("lejepa/combined_loss_scaled", {}).get("best"),
        "cov_last": stats.get("lejepa/cov_condition_number", {}).get("last"),
        "enc_last": stats.get("lejepa/encoder_eff_rank", {}).get("last"),
        "runtype_last": stats.get("lejepa/meta_sens_runtype", {}).get("last"),
    }


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    run_dirs = [Path(p) for p in sys.argv[1:]]
    missing = [p for p in run_dirs if not p.exists()]
    if missing:
        print(f"ERROR: run directories not found: {missing}", file=sys.stderr)
        raise SystemExit(1)

    ranking_rows: List[Dict[str, Optional[float]]] = []
    for rd in run_dirs:
        ranking_rows.append(analyze_run(rd))

    sortable = [r for r in ranking_rows if r.get("rank_score") is not None]
    sortable.sort(key=lambda r: (
        float(r["rank_score"]),  # primary: lower combined_loss_scaled better
        float(r["cov_last"]) if r.get("cov_last") is not None else float("inf"),
        -float(r["enc_last"]) if r.get("enc_last") is not None else float("inf"),
        -float(r["runtype_last"]) if r.get("runtype_last") is not None else float("inf"),
    ))

    if len(sortable) > 1:
        print(f"\n{'=' * 70}")
        print("RANKING (paper-grounded v2)")
        for i, row in enumerate(sortable, start=1):
            cov = float(row["cov_last"]) if row.get("cov_last") is not None else float("inf")
            enc = float(row["enc_last"]) if row.get("enc_last") is not None else float("nan")
            rt = float(row["runtype_last"]) if row.get("runtype_last") is not None else float("nan")
            print(
                f"{i:2d}. {row['run']}: combined_loss_scaled={row['rank_score']:.4f}, "
                f"cov={cov:.2f}, enc_er={enc:.2f}, runtype={rt:.3f}"
            )

    print(f"\n{'=' * 70}")
    print("Done. Use plot_jepa_trajectories.py for trajectory plots.")


if __name__ == "__main__":
    main()
