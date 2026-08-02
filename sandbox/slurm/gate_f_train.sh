#!/bin/bash
# No --partition: the Alliance job-submit plugin derives it from --time (b1 ≤3h,
# b2 ≤12h, ...). Naming one it would not have chosen fails with the misleading
# "The specified partition does not exist, or the submitted job cannot fit in it."
#SBATCH --job-name=sbx_gate_f
#SBATCH --account=def-maxwl
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=02:00:00
#SBATCH --output=sandbox/slurm_logs/gate_f_%j.out
#SBATCH --error=sandbox/slurm_logs/gate_f_%j.err

set -euo pipefail
cd "${REPO:-/project/6014832/mforooz/EpiDenoise}"
mkdir -p sandbox/slurm_logs sandbox/runs

source candi_venv/bin/activate
module load samtools 2>/dev/null || true

H5="${SANDBOX_H5:-sandbox/data/sandbox.h5}"
ATTEMPT="${SANDBOX_F_ATTEMPT:-1}"
EPOCHS="${SANDBOX_F_EPOCHS:-3}"
SEED="${SANDBOX_F_SEED:-0}"
RUN_DIR="${SANDBOX_F_RUN_DIR:-sandbox/runs/gate_f_attempt${ATTEMPT}}"
RUN_NAME="${SANDBOX_F_RUN_NAME:-gate_f_attempt${ATTEMPT}}"

echo "[gate_f] host=$(hostname) h5=$H5 attempt=$ATTEMPT run_dir=$RUN_DIR"
nvidia-smi -L || true

# Wandb: online, project candi_sandbox. Credentials come from ~/.netrc.
export WANDB_PROJECT="candi_sandbox"
export WANDB_MODE="${WANDB_MODE:-online}"

python -m sandbox.train \
  --config sandbox/configs/default.yaml \
  --config sandbox/configs/type2_loci.yaml \
  --h5 "$H5" \
  --epochs "$EPOCHS" \
  --seed "$SEED" \
  --training.run_dir "$RUN_DIR" \
  --wandb.project candi_sandbox \
  --wandb.run_name "$RUN_NAME" \
  --wandb.mode online \
  --wandb.tags gate_f

python -m sandbox.validate_gate_f "$RUN_DIR"
echo "[gate_f] DONE"
