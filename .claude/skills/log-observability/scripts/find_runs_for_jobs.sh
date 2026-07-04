#!/usr/bin/env bash
# Given one or more SLURM job IDs, print: <jobid> -> <run_dir>  + walltime status (CANCELLED?)
#
# Usage:
#   bash .cursor/skills/log-observability/scripts/find_runs_for_jobs.sh 37341539 37341540 ...
#
# Resolves each jobid via sandbox/slurm_logs/baseline_*_<jobid>.{out,err}.
# Extracts the run-dir name from the .out (BASELINE_NAME pattern) and looks up
# `sandbox/runs/baseline_<name>` (the canonical mapping used by submit_baselines.sh).
set -euo pipefail
LOG_DIR=${LOG_DIR:-sandbox/slurm_logs}

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <jobid> [jobid ...]" >&2
  exit 2
fi

for jid in "$@"; do
  out=$(ls "$LOG_DIR"/baseline_*_${jid}.out 2>/dev/null | head -1 || true)
  err=$(ls "$LOG_DIR"/baseline_*_${jid}.err 2>/dev/null | head -1 || true)
  if [[ -z "$out" ]]; then
    echo "$jid  -> (no slurm log found in $LOG_DIR)"
    continue
  fi
  # Extract `name=<x>` from the [baseline] line
  name=$(grep -m1 -oE 'name=[^ ]+' "$out" | head -1 | cut -d= -f2- || true)
  run_dir="sandbox/runs/baseline_${name}"
  walltime_kill="no"
  if [[ -n "$err" ]] && grep -q "DUE TO TIME LIMIT" "$err" 2>/dev/null; then
    walltime_kill="yes"
  fi
  oom="no"
  if [[ -n "$err" ]] && grep -qE "OutOfMemoryError|out-of-memory" "$err" 2>/dev/null; then
    oom="yes"
  fi
  exists="no"
  if [[ -d "$run_dir" ]]; then exists="yes"; fi
  echo "$jid  name=$name  run_dir=$run_dir  exists=$exists  walltime_kill=$walltime_kill  oom=$oom"
done
