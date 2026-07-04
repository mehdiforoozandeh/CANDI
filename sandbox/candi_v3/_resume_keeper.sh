#!/bin/bash
# Auto-resume keeper for CANDI v3 ERA.
# Relaunches `run.py --resume` until tree.json reaches TARGET nodes, recomputing the
# remaining count each time so it never overshoots. Survives run.py-level deaths (the
# login-node watchdog reaps the heavy driver after ~11h). Stays on the LOGIN node because
# generation (`claude -p` + WebFetch) needs outbound internet, which compute nodes lack.
#
# DOUBLE-SUBMIT SAFETY (critical): before each (re)launch, cancel any leftover submit.sh
# GPU jobs. At that point the previous run.py has already exited (its blocking call
# returned), so any in-flight jobs are ORPHANS that the next `--resume` would otherwise
# resubmit a fresh copy of (the old "10 jobs" bug). Cancelling them first guarantees the
# new round starts from a clean queue -> exactly batch_size jobs per round.
set -u
D=/project/6014832/mforooz/EpiDenoise/sandbox/candi_v3
V=/project/6014832/mforooz/EpiDenoise/candi_venv/bin/activate
cd "$D"; source "$V"; module load samtools 2>/dev/null
TARGET=281                       # cleaned tree = 181 genuine nodes + another 100
MAXLAUNCH=8                      # safety cap on relaunches
LOG="$D/era_resume_keep.out"
treen(){ python -c "import json;print(len(json.load(open('$D/tree.json'))))" 2>/dev/null; }
cancel_orphans(){
  local oj
  oj=$(squeue -u "$USER" -h -o "%i %j" 2>/dev/null | awk '$2=="submit.sh"{print $1}')
  if [ -n "$oj" ]; then scancel $oj; echo "[keeper] cancelled orphan jobs: $oj" >> "$LOG"; sleep 3; fi
}
echo "[keeper] START $(date) target=$TARGET" >> "$LOG"
for i in $(seq 1 $MAXLAUNCH); do
  n=$(treen)
  if [ -z "$n" ]; then echo "[keeper] cannot read tree.json -> abort" >> "$LOG"; break; fi
  if [ "$n" -ge "$TARGET" ]; then echo "[keeper] tree=$n >= $TARGET -> DONE" >> "$LOG"; break; fi
  cancel_orphans                 # clean queue before (re)launch -> no resubmit-double
  rem=$((TARGET - n))
  sed "s/^num_iterations:.*/num_iterations: $rem/" config_real.yaml > config_keeper.yaml
  echo "[keeper] launch $i $(date): tree=$n, adding $rem (target $TARGET)" >> "$LOG"
  python -u run.py --config config_keeper.yaml --resume >> "$LOG" 2>&1
  echo "[keeper] run.py exited (launch $i) $(date): tree now $(treen)" >> "$LOG"
  sleep 20
done
cancel_orphans                   # final tidy if we stop with jobs still queued
echo "[keeper] FINISHED $(date) tree=$(treen)" >> "$LOG"
