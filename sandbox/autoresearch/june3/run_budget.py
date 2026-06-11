#!/usr/bin/env python3
"""Budget-relaxation runner for E34 june3.

Authorized by user (2026-06-11) to explore a training budget larger than the
frozen prepare.EPOCHS=20. Stays entirely within june3/ (this is a NEW file; it
edits no frozen harness file). It monkeypatches prepare.EPOCHS in-process — the
same sanctioned pattern validate_harness.py uses — then runs one experiment with
the committed get_config() and appends to results.tsv exactly like agent_step.py.

The frozen-training-config guard (optimizer/loss_weights/masking/dsf/batch_size/
amp) is still enforced inside run_experiment; only the epoch budget changes.

Usage:
  python -m sandbox.autoresearch.june3.run_budget --epochs 40 --description "KEEP12 budget=40"
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from sandbox.autoresearch.june3 import keep_rule, prepare, scope

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = Path(__file__).resolve().parent
RESULTS_TSV = ARTIFACT_DIR / "results.tsv"


def _git_short_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "0000000"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--description", default="budget_run")
    args = p.parse_args()

    bad = scope.check_scope(staged_only=False)
    if bad:
        print("SCOPE VIOLATION — refusing to train:", file=sys.stderr)
        for v in bad:
            print(f"  {v}", file=sys.stderr)
        return 2

    prepare.EPOCHS = int(args.epochs)
    result = prepare.run_experiment()
    prepare.print_summary(result)

    status = result.get("status", "crash")
    primary = keep_rule._f(result.get("primary_score", "-999"))
    guards_ok = bool(result.get("guards_ok", False))
    vram_ok = bool(result.get("peak_vram_ok", False))
    best = keep_rule.load_best_primary(RESULTS_TSV)
    keep, reason = keep_rule.should_keep(
        primary_score=primary,
        guards_ok=guards_ok,
        vram_ok=vram_ok,
        status=status,
        best_primary=best,
    )

    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text("\t".join(keep_rule.TSV_COLUMNS) + "\n")
    row = {
        "commit": _git_short_hash(),
        "keep": "keep" if keep else reason,
        "primary_score": result.get("primary_score", "nan"),
        "imp_r2": result.get("imp_count_r2_gw", "nan"),
        "den_r2": result.get("den_count_r2_gw", "nan"),
        "count_imp_loss": result.get("count_imp_loss", "nan"),
        "count_obs_loss": result.get("count_obs_loss", "nan"),
        "train_imp_loss": result.get("train_count_imp_loss", "nan"),
        "train_obs_loss": result.get("train_count_obs_loss", "nan"),
        "param_count": result.get("param_count", "0"),
        "vram_mb": result.get("peak_vram_mb", "0"),
        "vram_ok": str(vram_ok).lower(),
        "status": status,
        "description": f"[{args.epochs}ep] {args.description}",
    }
    with RESULTS_TSV.open("a") as f:
        f.write(keep_rule.format_tsv_row(row))
    print(f"KEEP_VERDICT: {row['keep']} ({reason})", flush=True)
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
