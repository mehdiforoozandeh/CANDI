#!/usr/bin/env bash
# Outer loop for E32 autoresearch — run from compute node inside 4h SLURM session.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

source candi_venv/bin/activate 2>/dev/null || true
module load samtools 2>/dev/null || true

LOG="$ROOT/sandbox/autoresearch/may31/loop.log"
mkdir -p "$(dirname "$LOG")"

echo "=== loop start $(date -Is) ===" >> "$LOG"
while true; do
  echo "--- $(date -Is) ---" >> "$LOG"
  python -m sandbox.autoresearch.may31.agent_step --description "loop.sh" >> "$LOG" 2>&1 || true
  sleep 2
done
