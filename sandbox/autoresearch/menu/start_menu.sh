#!/bin/bash
# Launch the 5-loop menu autoresearch: build program.md, init loops from the v2 anchor, then start
# one driver tmux per loop + the orchestrator tmux. Safe to close the terminal afterward (tmux
# keeps them running). Stop with ./stop_menu.sh.
#   ./start_menu.sh        # full run (unbounded, until ./stop_menu.sh or ~11h login-node kill)
#   ./start_menu.sh 3      # SMOKE TEST: each loop runs exactly 3 iterations then exits
set -e
D=/project/6014832/mforooz/EpiDenoise/sandbox/autoresearch/menu
V=/project/6014832/mforooz/EpiDenoise/candi_venv/bin/activate
LOOPS="single_lambda crps_calibration factorized axial_longrange repr_first"
MAXIT="${1:-1000000000}"
[ "$MAXIT" -lt 1000000000 ] && echo "[start] SMOKE TEST mode: $MAXIT iterations per loop"

for t in $LOOPS; do
  if tmux has-session -t "menu-$t" 2>/dev/null; then echo "menu-$t already running — ./stop_menu.sh first"; exit 1; fi
done
if tmux has-session -t menu-orch 2>/dev/null; then echo "menu-orch already running — ./stop_menu.sh first"; exit 1; fi

rm -f "$D/STOP"
for t in $LOOPS; do rm -f "$D/$t/DONE"; done          # clear stale done markers from a prior run
cd "$D"; source "$V"; module load samtools 2>/dev/null
export PYTHONPATH=/project/6014832/mforooz/EpiDenoise

echo "[start] building program.md + initialising loops from the v2 anchor…"
python _harness/build_programs.py
python _harness/init_all.py

for t in $LOOPS; do
  tmux new-session -d -s "menu-$t" \
    "cd $D; source $V; module load samtools 2>/dev/null; export PYTHONPATH=/project/6014832/mforooz/EpiDenoise; \
     export MENU_MAX_ITERS=$MAXIT; \
     python -u _harness/run_driver.py $t > $t/run_driver.out 2>&1; echo '[run_driver EXITED]' >> $t/run_driver.out"
  echo "  started menu-$t (max_iters=$MAXIT)"
done
tmux new-session -d -s menu-orch \
  "cd $D; source $V; export PYTHONPATH=/project/6014832/mforooz/EpiDenoise; \
   python -u _orchestrator/orchestrator.py > _orchestrator/orch.out 2>&1"
echo "  started menu-orch (watchdog + dashboard + plateau)"
echo "[start] dashboard: open $D/dashboard.html  ·  stop: $D/stop_menu.sh"
