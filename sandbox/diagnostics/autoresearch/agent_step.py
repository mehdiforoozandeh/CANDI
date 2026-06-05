#!/usr/bin/env python3
"""Run one autoresearch experiment and append results.tsv.

Used by loop.sh or agent between train.py edits. Does NOT edit train.py.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

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


def append_tsv(row: dict) -> None:
    header = "commit\tcomposite_score\tmemory_gb\tpeak_vram_ok\tstatus\tdescription\n"
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(header)
    line = (
        f"{row.get('commit', '0000000')}\t"
        f"{row.get('composite_score', '9.999999')}\t"
        f"{row.get('memory_gb', '0.0')}\t"
        f"{row.get('peak_vram_ok', 'false')}\t"
        f"{row.get('status', 'crash')}\t"
        f"{row.get('description', '')}\n"
    )
    with RESULTS_TSV.open("a") as f:
        f.write(line)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--description", type=str, default="agent_step run")
    args = p.parse_args()

    proc = subprocess.run(
        [sys.executable, "-m", "sandbox.diagnostics.autoresearch.train"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env={**dict(**__import__("os").environ), "PYTHONPATH": str(REPO_ROOT)},
    )
    log = proc.stdout + proc.stderr
    RUN_LOG.write_text(log)

    summary = _parse_summary(log)
    score_str = summary.get("composite_score", "9.999999")
    try:
        score = float(score_str)
    except ValueError:
        score = 9.999999

    peak_mb_str = summary.get("peak_vram_mb", "0")
    try:
        peak_mb = float(peak_mb_str)
    except ValueError:
        peak_mb = 0.0

    status = summary.get("status", "crash")
    if proc.returncode != 0 and status == "ok":
        status = "crash"
    peak_ok = summary.get("peak_vram_ok", "false").lower() in ("true", "1")

    row = {
        "commit": _git_short_hash(),
        "composite_score": f"{score:.6f}",
        "memory_gb": f"{peak_mb / 1024.0:.1f}",
        "peak_vram_ok": str(peak_ok).lower(),
        "status": status if status in ("ok", "crash") else "crash",
        "description": args.description,
    }
    append_tsv(row)

    print(log)
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
