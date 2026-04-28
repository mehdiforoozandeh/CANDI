#!/bin/bash
#SBATCH --job-name=sbx_gate_b_parity
#SBATCH --account=def-maxwl
#SBATCH --partition=cpubase_bycore_b2
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=03:00:00
#SBATCH --output=sandbox/slurm_logs/gate_b_parity_%j.out
#SBATCH --error=sandbox/slurm_logs/gate_b_parity_%j.err

set -euo pipefail
cd "${REPO:-/project/6014832/mforooz/EpiDenoise}"
mkdir -p sandbox/slurm_logs sandbox/data

source candi_venv/bin/activate
module load samtools 2>/dev/null || true

EIC="${SANDBOX_EIC_DATA:-/project/6014832/mforooz/DATA_CANDI_EIC}"
H5="${SANDBOX_H5:-sandbox/data/sandbox.h5}"
OK="${SANDBOX_PARITY_OK:-sandbox/data/parity.ok}"

echo "[gate_b_parity] host=$(hostname) eic=$EIC h5=$H5"
ls -la "$H5"
echo "[gate_b_parity] validate-parity (full grid)"
python -m sandbox.prepare_h5 validate-parity \
  --eic-data "$EIC" \
  --h5 "$H5" \
  --parity-ok "$OK"

echo "[gate_b_parity] parity.ok contents:"
cat "$OK"
echo "[gate_b_parity] DONE"

