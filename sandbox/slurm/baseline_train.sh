#!/bin/bash
# Reusable template for sandbox baseline training runs.
#
# Required env vars (set before sbatch or in submit_baselines.sh):
#   BASELINE_NAME   — short identifier, e.g. "anchor", "dsf1_only", "assay_mask_only"
#   BASELINE_EXTRA  — extra dotted CLI overrides, e.g. "--training.dsf.sampling off"
#
# Optional env vars:
#   SANDBOX_H5         — path to sandbox.h5 (default: sandbox/data/sandbox.h5)
#   BASELINE_EPOCHS    — training epochs (default: 200)
#   BASELINE_SEED      — random seed (default: 42)
#   BASELINE_BATCH     — per-GPU batch size (default: 32). Sandbox CANDI is ~232k params;
#                        VRAM stays low; 32 improves throughput vs 16 on H100 without OOM risk.
#   BASELINE_REGIME_CONFIG — regime overlay config path
#                        (default: sandbox/configs/type1_chr19.yaml)
#   BASELINE_PREFIX  — run-dir / W&B-name prefix (default: "baseline_"). Set to ""
#                        for experiment runs that should land in the HPO graph
#                        without the baseline prefix.
#   BASELINE_TIME     — not read here: walltime is set by submit_baselines.sh via
#                        `sbatch --time=...` (overrides the #SBATCH line below) or
#                        edit #SBATCH --time when running `sbatch` on this file directly.
#
# No --partition: the Alliance job-submit plugin derives it from --time (b1 ≤3h,
# b2 ≤12h, ...). Naming one it would not have chosen fails with the misleading
# "The specified partition does not exist, or the submitted job cannot fit in it."
#SBATCH --job-name=sbx_baseline
#SBATCH --account=def-maxwl
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=sandbox/slurm_logs/baseline_%x_%j.out
#SBATCH --error=sandbox/slurm_logs/baseline_%x_%j.err

set -euo pipefail
cd "${REPO:-/project/6014832/mforooz/EpiDenoise}"
mkdir -p sandbox/slurm_logs sandbox/runs

source candi_venv/bin/activate
module load samtools 2>/dev/null || true

H5="${SANDBOX_H5:-sandbox/data/sandbox.h5}"
NAME="${BASELINE_NAME:?set BASELINE_NAME}"
EPOCHS="${BASELINE_EPOCHS:-200}"
SEED="${BASELINE_SEED:-42}"
BS="${BASELINE_BATCH:-32}"
REGIME_CONFIG="${BASELINE_REGIME_CONFIG:-sandbox/configs/type1_chr19.yaml}"
PREFIX="${BASELINE_PREFIX-baseline_}"   # Default keeps backward-compatible run dir naming
RUN_DIR="sandbox/runs/${PREFIX}${NAME}"
WANDB_RUN="${PREFIX}${NAME}"

echo "[baseline] host=$(hostname) partition=${SLURM_JOB_PARTITION:-?} job=$SLURM_JOB_ID name=$NAME batch=$BS h5=$H5 epochs=$EPOCHS seed=$SEED regime_config=$REGIME_CONFIG"
nvidia-smi -L || true

export WANDB_PROJECT="candi_sandbox"

# shellcheck disable=SC2086
python -m sandbox.train \
  --config sandbox/configs/default.yaml \
  --config "$REGIME_CONFIG" \
  --h5 "$H5" \
  --epochs "$EPOCHS" \
  --batch-size "$BS" \
  --seed "$SEED" \
  --training.run_dir "$RUN_DIR" \
  --run-name "$WANDB_RUN" \
  --wandb.project candi_sandbox \
  --wandb.tags "baseline,${NAME}" \
  ${BASELINE_EXTRA:-}

echo "[baseline] DONE — metrics at ${RUN_DIR}/metrics.jsonl"
