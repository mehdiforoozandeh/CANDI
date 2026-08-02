#!/bin/bash
# No --partition: the Alliance job-submit plugin derives it from --time (b1 ≤3h,
# b2 ≤12h, ...). Naming one it would not have chosen fails with the misleading
# "The specified partition does not exist, or the submitted job cannot fit in it."
#SBATCH --job-name=sbx_gate_e
#SBATCH --account=def-maxwl
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=01:00:00
#SBATCH --output=sandbox/slurm_logs/gate_e_%j.out
#SBATCH --error=sandbox/slurm_logs/gate_e_%j.err

set -euo pipefail
cd "${REPO:-/project/6014832/mforooz/EpiDenoise}"
mkdir -p sandbox/slurm_logs

source candi_venv/bin/activate
module load samtools 2>/dev/null || true

H5="${SANDBOX_H5:-sandbox/data/sandbox.h5}"
REGIME="${SANDBOX_E_REGIME:-type2_loci}"
STEPS="${SANDBOX_E_STEPS:-200}"
BS="${SANDBOX_E_BS:-4}"
LR="${SANDBOX_E_LR:-3e-4}"
RELAX="${SANDBOX_RELAX_GATE_E:-}"

echo "[gate_e] host=$(hostname) h5=$H5 regime=$REGIME"
nvidia-smi -L || true

EXTRA=()
if [ "$RELAX" = "1" ]; then EXTRA+=(--relax-gate-e); fi

python -m sandbox.prepare_h5 overfit-sanity \
  --h5 "$H5" \
  --regime "$REGIME" \
  --steps "$STEPS" \
  --batch-size "$BS" \
  --lr "$LR" \
  "${EXTRA[@]}"

echo "[gate_e] DONE"
