#!/bin/bash
# candi_kit — the two-arm training array (offset ON / OFF) x 3 seeds.
# #SBATCH skeleton copied from sandbox/diagnostics/dual_conditioning_real/jobs/wd0.sh; only --array,
# the job name and the paths change. --gres is the HARD-RULE MIG slice — never any other spec.
# ~85 min wall per arm (25 epochs x ~2.9 min + ~13 min eval); 6 arms as one array ~= 6 GPU-hours.
# Host RSS 15-18 GiB is the binding constraint, not GPU (the h5 is slurped into one shared buffer).
#   sbatch candi_kit/slurm/train.sh
# Logs go to ./slurm-logs/ RELATIVE TO WHERE YOU SUBMIT FROM. SLURM resolves --output against
# the submitting cwd, not the script's location, so `mkdir -p slurm-logs` first (or pass
# `sbatch --output=... --error=...` to override). Submitting from anywhere is then fine.
#SBATCH --account=def-maxwl
#SBATCH --job-name=candi_kit_arms
#SBATCH --output=slurm-logs/train_%A_%a.out
#SBATCH --error=slurm-logs/train_%A_%a.err
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-5

set -uo pipefail

# --- edit these ---------------------------------------------------------------------------------
KIT="${KIT:-/project/6014832/mforooz/EpiDenoise/candi_kit}"
# VENV: defaults to the environment you are ALREADY in ($VIRTUAL_ENV), which is what README §4.0
# leaves you in. Override with VENV=/path/to/venv. It must NOT default to someone else's venv --
# sourcing a path you do not own fails late, inside python, not at the source line.
VENV="${VENV:-${VIRTUAL_ENV:-}}"
H5="${H5:-/scratch/$USER/candi_kit/q19.h5}"
OUT="${OUT:-/scratch/$USER/candi_kit/runs}"
# -------------------------------------------------------------------------------------------------

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1; unset PYTHONPATH || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLBACKEND=Agg
export WANDB_MODE=disabled
if [ -n "$VENV" ]; then
  source "$VENV/bin/activate"
else
  echo "[error] no environment: set VENV=/path/to/venv, or sbatch from an active venv" >&2; exit 1
fi

cd "$KIT"
mkdir -p "$OUT"

# arm x seed: tasks 0-2 = offset ON seeds 0,1,2; tasks 3-5 = offset OFF seeds 0,1,2.
ARMS=(on on on off off off)
SEEDS=(0 1 2 0 1 2)
ARM="${ARMS[$SLURM_ARRAY_TASK_ID]}"
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

echo "[train] task=$SLURM_ARRAY_TASK_ID arm=$ARM seed=$SEED host=$(hostname)"; nvidia-smi -L || true

# num_assays / context_length / resolution / dsf_list / assay order / chroms are NOT flags — they are
# properties of $H5 and are read from its attrs. --weight-decay 0.0 is the recorded-best setting.
python -m candi_kit.train \
  --h5 "$H5" --out-dir "$OUT" \
  --offset "$ARM" --seed "$SEED" --tag "kit_${ARM}_s${SEED}" \
  --weight-decay 0.0 \
  --dsf-sampling uniform --epochs 25 --batch-size 8 --full-coverage \
  --eval-batch-size 4 --eval-max-batches 0 --eval-budget 50000000 --m3-regions 40 \
  --fg-frac 0.02 --n-boot 1000
rc=$?

echo "[train] DONE task=$SLURM_ARRAY_TASK_ID rc=$rc"
exit $rc
