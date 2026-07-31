#!/bin/bash
# candi_kit — the SMALLEST run that works end to end (README §4.2). Bake + train + report, ~15 min.
# Use this to prove your install and your data before spending real GPU time on the reference panel.
#
#   mkdir -p slurm-logs
#   VENV=~/candi_venv KIT=/path/to/candi_kit ROOT=/path/to/DATA_CANDI_EIC SIDE=~/side \
#     sbatch /path/to/candi_kit/slurm/example.sh
#
# Logs land in ./slurm-logs/ relative to WHERE YOU SUBMIT FROM (SLURM resolves --output against the
# submitting cwd, not the script's location).
#SBATCH --account=def-maxwl
#SBATCH --job-name=candi_kit_example
#SBATCH --output=slurm-logs/example_%j.out
#SBATCH --error=slurm-logs/example_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=30G
#SBATCH --cpus-per-task=2

set -uo pipefail

KIT="${KIT:?set KIT=/path/to/candi_kit}"
ROOT="${ROOT:?set ROOT=/path/to/your ENCODE-style data directory}"
SIDE="${SIDE:?set SIDE=/dir/holding your genome files (README 4.0b)}"
# Override these if your genome files are named differently -- README 4.0b happens to produce
# hg38_subset.*, but nothing requires that name.
FASTA="${FASTA:-$SIDE/hg38_subset.fa}"
CHROM_SIZES="${CHROM_SIZES:-$SIDE/hg38_subset.chrom.sizes}"
VENV="${VENV:-${VIRTUAL_ENV:-}}"
PANEL="${PANEL:-$KIT/configs/panel.example.json}"
H5="${H5:-/scratch/$USER/candi_kit/example.h5}"
OUT="${OUT:-/scratch/$USER/candi_kit/runs_example}"
TAG="${TAG:-example}"

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1; unset PYTHONPATH || true
export MPLBACKEND=Agg WANDB_MODE=disabled
if [ -n "$VENV" ]; then
  source "$VENV/bin/activate"
else
  echo "[error] no environment: set VENV=/path/to/venv, or sbatch from an active venv" >&2; exit 1
fi

mkdir -p "$(dirname "$H5")" "$OUT"

if [ -e "$H5" ]; then
  echo "[example] reusing existing $H5 (delete it to force a re-bake)"
else
  for f in "$FASTA" "$CHROM_SIZES"; do
    [ -r "$f" ] || { echo "[error] not readable: $f  (set FASTA=/CHROM_SIZES=, see README 4.0b)" >&2; exit 1; }
  done
  echo "[example] bake -> $H5"
  python -m candi_kit.prep.bake \
    --root "$ROOT" --panel "$PANEL" --out "$H5" \
    --fasta "$FASTA" --chrom-sizes "$CHROM_SIZES" \
    --type2-ccre 0 --type2-non 0 --allow-missing-control --seed 42 || exit 1
fi

echo "[example] train offset=on"
python -m candi_kit.train --h5 "$H5" --out-dir "$OUT" \
  --offset on --seed 0 --tag "$TAG" --epochs 3 --batch-size 8 --full-coverage \
  --eval-batch-size 4 --eval-max-batches 0 --m3-regions 10 --n-boot 100 || exit 1

echo "[example] report"
python -m candi_kit.report "$OUT/$TAG.json" || exit 1
echo "[example] DONE -> $OUT/$TAG.json"
