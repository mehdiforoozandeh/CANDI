#!/bin/bash
#SBATCH --job-name=sbx_gate_b
#SBATCH --account=def-maxwl
#SBATCH --partition=cpubase_bycore_b2
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=02:00:00
#SBATCH --output=sandbox/slurm_logs/gate_b_%j.out
#SBATCH --error=sandbox/slurm_logs/gate_b_%j.err

set -euo pipefail
cd "${REPO:-/project/6014832/mforooz/EpiDenoise}"
mkdir -p sandbox/slurm_logs sandbox/data

source candi_venv/bin/activate
module load samtools 2>/dev/null || true

EIC="${SANDBOX_EIC_DATA:-/project/6014832/mforooz/DATA_CANDI_EIC}"
H5="sandbox/data/sandbox.h5"
OK="sandbox/data/parity.ok"
TMP="sandbox/data/.sandbox.h5.tmp.${SLURM_JOB_ID:-$$}"

echo "[gate_b] host=$(hostname) eic=$EIC h5=$H5"
python -m sandbox.prepare_h5 bake \
  --eic-data "$EIC" \
  --selection sandbox/data/selection.json \
  --out "$TMP"

mv -f "$TMP" "$H5"

echo "[gate_b] bake size:"
du -h "$H5"

echo "[gate_b] validate-parity (full grid)"
python -m sandbox.prepare_h5 validate-parity \
  --eic-data "$EIC" \
  --h5 "$H5" \
  --parity-ok "$OK"

echo "[gate_b] parity.ok contents:"
cat "$OK"
echo "[gate_b] DONE"
