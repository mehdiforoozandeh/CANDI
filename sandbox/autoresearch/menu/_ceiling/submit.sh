#!/bin/bash
# Ceiling experiment: base/consol × chr19/augmented, steps-matched (19100) + augmented epochs=5 variant.
# Eval is ALWAYS chr21 (frozen); marginal baseline (Q_imp 0.4857) is chr21-derived so ERA_SCORE stays
# comparable across all cells. Each job releases early; run concurrently on MIG slices.
set -u
M=/project/6014832/mforooz/EpiDenoise/sandbox/autoresearch/menu
WRAP=$M/_harness/sbatch_wrap.sh
CE=$M/_ceiling
BASE=$CE/base_train.py
CONS=$CE/consol_train.py

submit () {  # name cand time  EXTRA_ENV...
  local name=$1 cand=$2 tlim=$3; shift 3
  local out=$CE/${name}.out
  local env="ALL,CAND=$cand,OUT=$out"
  for kv in "$@"; do env="$env,$kv"; done
  local jid=$(sbatch --parsable --time=$tlim --output=$out --export="$env" "$WRAP")
  echo "$jid  $name  ($*)"
  echo "$jid" > $CE/${name}.jobid
}

# --- steps-matched grid (s_max=19100 gradient steps in every cell) ---
submit base_chr19    $BASE 0:35:00  MENU_SMAX=19100
submit consol_chr19  $CONS 0:35:00  MENU_SMAX=19100
submit base_aug      $BASE 0:35:00  MENU_SMAX=19100 MENU_AUGMENT=1
submit consol_aug    $CONS 0:35:00  MENU_SMAX=19100 MENU_AUGMENT=1
# --- augmented epochs=5 variant (compute-uncapped ceiling; ~31600 steps) ---
submit base_aug5ep   $BASE 0:50:00  MENU_MAXEP=5 MENU_TMAX=2400 MENU_AUGMENT=1
submit consol_aug5ep $CONS 0:50:00  MENU_MAXEP=5 MENU_TMAX=2400 MENU_AUGMENT=1
