#!/usr/bin/env python3
"""Run one E32 autoresearch experiment, apply Pareto keep rule, append results.tsv."""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

from sandbox.autoresearch.may31 import keep_rule

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = Path(__file__).resolve().parent
RESULTS_TSV = ARTIFACT_DIR / "results.tsv"
RUN_LOG = ARTIFACT_DIR / "run.log"


def _git_short_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "0000000"


def _parse_summary(log_text: str) -> dict:
    out: dict = {}
    in_block = False
    for line in log_text.splitlines():
        if line.strip() == "---":
            in_block = not in_block
            continue
        if in_block and ":" in line:
            key, _, val = line.partition(":")
            out[key.strip()] = val.strip()
    return out


def _ensure_header() -> None:
    header_line = "\t".join(keep_rule.TSV_COLUMNS) + "\n"
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(header_line)
        return
    first = RESULTS_TSV.read_text().splitlines()[0]
    if first.split("\t") == list(keep_rule.TSV_COLUMNS):
        return
    # Legacy header — archive and start session-3 format (old rows preserved).
    legacy = ARTIFACT_DIR / "results_legacy.tsv"
    if not legacy.exists():
        RESULTS_TSV.rename(legacy)
        RESULTS_TSV.write_text(header_line)


def append_tsv(row: dict) -> None:
    _ensure_header()
    with RESULTS_TSV.open("a") as f:
        f.write(keep_rule.format_tsv_row(row))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--description", type=str, default="agent_step run")
    p.add_argument(
        "--den-floor",
        type=float,
        default=keep_rule.DEN_KEEP_FLOOR,
        help="Minimum den_r2 to keep an imp_r2 improvement",
    )
    args = p.parse_args()

    proc = subprocess.run(
        [sys.executable, "-m", "sandbox.autoresearch.may31.train"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env={**dict(**__import__("os").environ), "PYTHONPATH": str(REPO_ROOT)},
    )
    log = proc.stdout + proc.stderr
    RUN_LOG.write_text(log)

    summary = _parse_summary(log)
    status = summary.get("status", "crash")
    if proc.returncode != 0 and status == "ok":
        status = "crash"
    peak_ok = summary.get("peak_vram_ok", "False").lower() in ("true", "1")

    imp_r2 = keep_rule._f(summary.get("imp_count_r2_gw"))
    den_r2 = keep_rule._f(summary.get("den_count_r2_gw"))
    dcr = keep_rule._f(summary.get("depth_count_ratio"))
    imp_canonical = keep_rule._f(summary.get("imp_count_r2_gw_canonical"))
    imp_cloze = keep_rule._f(summary.get("imp_count_r2_gw_cloze_T"))
    task_gap = keep_rule._f(summary.get("imp_r2_task_gap"))

    best_imp, best_den = keep_rule.load_best_pareto(RESULTS_TSV)
    keep, reason = keep_rule.should_keep(
        imp_r2=imp_r2,
        den_r2=den_r2,
        dcr=dcr,
        vram_ok=peak_ok,
        status=status if status in ("ok", "crash") else "crash",
        best_imp_r2=best_imp,
        best_den_r2=best_den,
        den_floor=args.den_floor,
    )
    keep_label = "keep" if keep else ("crash" if status != "ok" else reason)

    row = {
        "commit": _git_short_hash(),
        "keep": keep_label,
        "primary_score": summary.get("primary_score", "-999"),
        "metric_phase": summary.get("metric_phase", "den"),
        "imp_r2": f"{imp_r2:.6f}" if math.isfinite(imp_r2) else "nan",
        "den_r2": f"{den_r2:.6f}" if math.isfinite(den_r2) else "nan",
        "imp_r2_canonical": (
            f"{imp_canonical:.6f}" if math.isfinite(imp_canonical) else "nan"
        ),
        "imp_r2_cloze_T": f"{imp_cloze:.6f}" if math.isfinite(imp_cloze) else "nan",
        "imp_r2_task_gap": f"{task_gap:.6f}" if math.isfinite(task_gap) else "nan",
        "dcr": f"{dcr:.6f}" if math.isfinite(dcr) else "nan",
        "imp_pearson": summary.get("imp_count_pearson_gw", "nan"),
        "vram_mb": summary.get("peak_vram_mb", "0"),
        "vram_ok": str(peak_ok).lower(),
        "status": status if status in ("ok", "crash") else "crash",
        "description": args.description,
    }
    append_tsv(row)

    print(log, end="" if log.endswith("\n") else "\n")
    print(
        f"KEEP_VERDICT: {keep_label} ({reason}) | "
        f"imp_r2={row['imp_r2']} den_r2={row['den_r2']} | "
        f"best_imp={best_imp:.6f} best_den@best={best_den:.6f}",
        flush=True,
    )
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
