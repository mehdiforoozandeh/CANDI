#!/usr/bin/env python3
"""One E34 AR run + keep verdict + results.tsv append."""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

from sandbox.autoresearch.june3 import keep_rule, scope

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = Path(__file__).resolve().parent
RESULTS_TSV = ARTIFACT_DIR / "results.tsv"
RUN_LOG = ARTIFACT_DIR / "run.log"


def _git_short_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
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


def _enforce_scope_or_exit() -> None:
    bad = scope.check_scope(staged_only=False)
    if bad:
        print("SCOPE VIOLATION — refusing to train:", file=sys.stderr)
        for v in bad:
            print(f"  {v}", file=sys.stderr)
        print(
            "\nRevert forbidden edits, e.g.:\n"
            "  git checkout -- sandbox/train.py\n"
            "  git checkout -- <any path outside sandbox/autoresearch/june3/>",
            file=sys.stderr,
        )
        raise SystemExit(2)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--description", default="agent_step")
    p.add_argument(
        "--skip-scope-check",
        action="store_true",
        help="Human debugging only; never use in agent loop",
    )
    args = p.parse_args()

    if not args.skip_scope_check:
        _enforce_scope_or_exit()

    proc = subprocess.run(
        [sys.executable, "-m", "sandbox.autoresearch.june3.train"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(REPO_ROOT)},
    )
    log = proc.stdout + proc.stderr
    RUN_LOG.write_text(log)
    summary = _parse_summary(log)
    status = summary.get("status", "crash")
    if proc.returncode != 0 and status == "ok":
        status = "crash"

    primary = keep_rule._f(summary.get("primary_score", "-999"))
    guards_ok = summary.get("guards_ok", "False").lower() in ("true", "1")
    vram_ok = summary.get("peak_vram_ok", "False").lower() in ("true", "1")
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
        "primary_score": summary.get("primary_score", "nan"),
        "imp_r2": summary.get("imp_count_r2_gw", "nan"),
        "den_r2": summary.get("den_count_r2_gw", "nan"),
        "count_imp_loss": summary.get("count_imp_loss", "nan"),
        "count_obs_loss": summary.get("count_obs_loss", "nan"),
        "train_imp_loss": summary.get("train_count_imp_loss", "nan"),
        "train_obs_loss": summary.get("train_count_obs_loss", "nan"),
        "param_count": summary.get("param_count", "0"),
        "vram_mb": summary.get("peak_vram_mb", "0"),
        "vram_ok": str(vram_ok).lower(),
        "status": status,
        "description": args.description,
    }
    with RESULTS_TSV.open("a") as f:
        f.write(keep_rule.format_tsv_row(row))

    print(log, end="" if log.endswith("\n") else "\n")
    print(f"KEEP_VERDICT: {row['keep']} ({reason})", flush=True)
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
