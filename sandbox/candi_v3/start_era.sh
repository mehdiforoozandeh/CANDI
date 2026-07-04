#!/bin/bash
# Start the CANDI v3 ERA N=100 search (5 concurrent jobs) + live dashboard, detached in tmux.
# Safe to close your terminal/chat afterward — tmux keeps them running.
set -e
D=/project/6014832/mforooz/EpiDenoise/sandbox/candi_v3
V=/project/6014832/mforooz/EpiDenoise/candi_venv/bin/activate

# refuse to double-launch
if tmux has-session -t erav3 2>/dev/null; then echo "erav3 already running — stop it first (./stop_era.sh)"; exit 1; fi

tmux new-session -d -s erav3   "cd $D; source $V; python -u run.py --config config_real.yaml > era_real.out 2>&1; echo '[run.py EXITED]' >> era_real.out"
tmux new-session -d -s eradash "cd $D; source $V; python -u dashboard.py > dashboard.log 2>&1"
echo "Started ERA N=100 (batch_size=5) + dashboard in tmux."
echo "  Dashboard : open $D/dashboard.html in a browser (auto-refresh 5s)"
echo "  Live log  : tmux attach -t erav3      (detach: Ctrl-b then d)"
echo "  Stop all  : $D/stop_era.sh"
