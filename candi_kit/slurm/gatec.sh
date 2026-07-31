#!/bin/bash
# candi_kit GATE C — knob generalization smoke.
# Proves the panel/scale knobs are REAL, not decoration: 5 assays (not 8) and context_bins 512 (not 768)
# and dsf_list [1,2,4] (not [1,2,4,8]). d_model is auto-derived as (num_assays+1) * 2^n_cnn_layers, so
# this run has d_model 48 where the q19 panel has 72 -- i.e. a different transformer width, which is
# exactly why an existing checkpoint cannot be reused across assay counts.
# Short: 3 epochs, small eval. This is a SMOKE test -- its metrics are not comparable to anything.
# Logs go to ./slurm-logs/ RELATIVE TO WHERE YOU SUBMIT FROM. SLURM resolves --output against
# the submitting cwd, not the script's location, so `mkdir -p slurm-logs` first (or pass
# `sbatch --output=... --error=...` to override). Submitting from anywhere is then fine.
#SBATCH --account=def-maxwl
#SBATCH --job-name=candi_kit_gatec
#SBATCH --output=slurm-logs/gatec_%j.out
#SBATCH --error=slurm-logs/gatec_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=30G
#SBATCH --cpus-per-task=2

set -uo pipefail
KIT="${KIT:-/project/6014832/mforooz/EpiDenoise/candi_kit}"
# VENV: defaults to the environment you are ALREADY in ($VIRTUAL_ENV), which is what README §4.0
# leaves you in. Override with VENV=/path/to/venv. It must NOT default to someone else's venv --
# sourcing a path you do not own fails late, inside python, not at the source line.
VENV="${VENV:-${VIRTUAL_ENV:-}}"
ROOT="${ROOT:-/project/6014832/mforooz/DATA_CANDI_EIC}"
SIDE="${SIDE:-/project/6014832/mforooz/EpiDenoise/data}"
H5="${H5:-/scratch/$USER/candi_kit/gatec.h5}"
OUT="${OUT:-/scratch/$USER/candi_kit/runs_gatec}"

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1; unset PYTHONPATH || true
export MPLBACKEND=Agg WANDB_MODE=disabled
if [ -n "$VENV" ]; then
  source "$VENV/bin/activate"
else
  echo "[error] no environment: set VENV=/path/to/venv, or sbatch from an active venv" >&2; exit 1
fi
cd "$KIT"; mkdir -p "$OUT"

echo "[gatec] bake 5-assay / 512-bin panel"
python -m candi_kit.prep.bake \
  --root "$ROOT" --panel "$KIT/configs/panel.gatec.json" --out "$H5" \
  --fasta "$SIDE/hg38.fa" --chrom-sizes "$SIDE/hg38.chrom.sizes" \
  --type2-ccre 0 --type2-non 0 --allow-missing-control --seed 42 || exit 1

echo "[gatec] train 3 epochs, offset ON"
python -m candi_kit.train --h5 "$H5" --out-dir "$OUT" \
  --offset on --seed 0 --tag gatec_on --weight-decay 0.0 \
  --dsf-sampling uniform --epochs 3 --batch-size 8 \
  --eval-batch-size 4 --eval-max-batches 40 --m3-regions 10 --n-boot 100
rc=$?
echo "[gatec] DONE rc=$rc"
exit $rc
