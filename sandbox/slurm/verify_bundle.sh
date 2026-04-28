#!/bin/bash
# One sbatch job: bundled unit tests for sandbox correctness / config / optimizer / model.
# Intended to reduce queue wait vs many tiny jobs. CPU-only is enough for this pytest set.
#
#SBATCH --job-name=sbx_verify
#SBATCH --account=def-maxwl
#SBATCH --partition=cpubase_interac
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=sandbox/slurm_logs/verify_bundle_%j.out
#SBATCH --error=sandbox/slurm_logs/verify_bundle_%j.err

set -euo pipefail
cd "${REPO:-/project/6014832/mforooz/EpiDenoise}"
mkdir -p sandbox/slurm_logs

source candi_venv/bin/activate
module load samtools 2>/dev/null || true

echo "[verify_bundle] host=$(hostname) cwd=$(pwd)"
./candi_venv/bin/python -m pytest \
  sandbox/tests/test_sandbox_correctness.py \
  sandbox/tests/test_sandbox_model.py \
  sandbox/tests/test_gate_d_optimizer.py \
  sandbox/tests/test_config_yaml.py \
  -q --tb=short

echo "[verify_bundle] DONE"
