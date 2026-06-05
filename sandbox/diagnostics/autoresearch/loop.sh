#!/usr/bin/env bash
# Outer loop for autoresearch — run from compute node inside 4h SLURM session.
# Primary mode: Cursor agent edits train.py between runs (see program.md).
# This script re-runs agent_step without editing train.py (logging / backup only).

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

source candi_venv/bin/activate 2>/dev/null || true
module load samtools 2>/dev/null || true

LOG="$ROOT/sandbox/diagnostics/autoresearch/loop.log"
mkdir -p "$(dirname "$LOG")"

echo "=== loop start $(date -Is) ===" >> "$LOG"
while true; do
  echo "--- $(date -Is) ---" >> "$LOG"
  python -m sandbox.diagnostics.autoresearch.agent_step --description "loop.sh" >> "$LOG" 2>&1 || true
  sleep 2
done
