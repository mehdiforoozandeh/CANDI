#!/bin/bash
# Stop the ERA search + dashboard and cancel its GPU jobs.
# Kill run.py FIRST so it stops spawning new generations, then reap any in-flight
# `claude -p` / `cursor-agent -p` children it orphaned. Each generation subprocess
# is a heavyweight Node process holding dozens of threads with a long timeout; if
# left orphaned they linger for minutes and exhaust the per-user RLIMIT_NPROC,
# blocking even `fork()` on the login node. ALWAYS reap them on stop.
pkill -u "$USER" -f "run.py --config config_real" 2>/dev/null || true
pkill -u "$USER" -f "dashboard.py" 2>/dev/null || true
sleep 1
pkill -u "$USER" -f "claude -p" 2>/dev/null || true
pkill -u "$USER" -f "cursor-agent -p" 2>/dev/null || true
tmux kill-session -t erav3 2>/dev/null || true
tmux kill-session -t eradash 2>/dev/null || true
squeue -u "$USER" -h -o "%i %j" | awk '$2=="submit.sh"{print $1}' | xargs -r scancel 2>/dev/null || true
echo "Stopped ERA + dashboard, reaped orphaned generators, cancelled submit.sh jobs."
