#!/bin/bash
# Stop the menu autoresearch: set the STOP flag (drivers + orchestrator exit at their next check),
# cancel in-flight jobs, kill the tmux sessions, reap stray CLI subprocesses, write SYNTHESIS.md.
D=/project/6014832/mforooz/EpiDenoise/sandbox/autoresearch/menu
LOOPS="single_lambda crps_calibration factorized axial_longrange repr_first"
touch "$D/STOP"
sleep 2
for t in $LOOPS; do
  scancel --name "menu-$t" 2>/dev/null
  tmux kill-session -t "menu-$t" 2>/dev/null && echo "killed menu-$t"
done
# let the orchestrator write synthesis on its STOP pass, then kill it
( cd "$D" && source /project/6014832/mforooz/EpiDenoise/candi_venv/bin/activate 2>/dev/null && \
  PYTHONPATH=/project/6014832/mforooz/EpiDenoise python _orchestrator/orchestrator.py --once 2>/dev/null )
tmux kill-session -t menu-orch 2>/dev/null && echo "killed menu-orch"
for pat in "claude -p" "cursor-agent -p" "smoke.py" "run_driver.py"; do pkill -9 -u mforooz -f "$pat" 2>/dev/null; done
echo "[stop] STOP flag set, jobs cancelled, sessions killed. SYNTHESIS.md written in $D"
