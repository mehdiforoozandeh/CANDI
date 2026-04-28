#!/bin/bash
#SBATCH --job-name=sbx_gate_g
#SBATCH --account=def-maxwl
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=02:00:00
#SBATCH --output=sandbox/slurm_logs/gate_g_%j.out
#SBATCH --error=sandbox/slurm_logs/gate_g_%j.err

set -euo pipefail
cd "${REPO:-/project/6014832/mforooz/EpiDenoise}"
mkdir -p sandbox/slurm_logs sandbox/runs

source candi_venv/bin/activate
module load samtools 2>/dev/null || true

H5="${SANDBOX_H5:-sandbox/data/sandbox.h5}"
REGIME="${SANDBOX_G_REGIME:?set SANDBOX_G_REGIME=type1_chr19|type2_loci}"
SEED="${SANDBOX_G_SEED:-0}"
RUN_DIR="${SANDBOX_G_RUN_DIR:-sandbox/runs/gate_g_${REGIME}}"
RUN_NAME="${SANDBOX_G_RUN_NAME:-gate_g_${REGIME}}"

echo "[gate_g] host=$(hostname) h5=$H5 regime=$REGIME run_dir=$RUN_DIR"
nvidia-smi -L || true

export WANDB_PROJECT="candi_sandbox"
export WANDB_MODE="${WANDB_MODE:-online}"

python -m sandbox.train \
  --config sandbox/configs/default.yaml \
  --config "sandbox/configs/${REGIME}.yaml" \
  --h5 "$H5" \
  --epochs 1 \
  --seed "$SEED" \
  --training.run_dir "$RUN_DIR" \
  --wandb.project candi_sandbox \
  --wandb.run_name "$RUN_NAME" \
  --wandb.mode online \
  --wandb.tags gate_g

python -m sandbox.validate_gate_g "$RUN_DIR"
echo "[gate_g] DONE"
