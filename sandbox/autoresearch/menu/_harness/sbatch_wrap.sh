#!/bin/bash
#SBATCH --account=def-maxwl_gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=16G
#SBATCH --time=0:35:00
# (harness train cap = 1800s/30min + ~3-4min eval; 35min is the safe minimum. Most candidates
#  reach 5 epochs in ~15-18min and release the allocation early.)
# Run ONE menu-AR candidate under the frozen judge. Args via env:
#   CAND  = absolute path to the candidate train.py
#   OUT   = absolute path for stdout (footer is grepped from here)
# Submit:  sbatch --output=$OUT _harness/sbatch_wrap.sh   (with CAND/OUT exported)
set -u
source /project/6014832/mforooz/EpiDenoise/candi_venv/bin/activate 2>/dev/null
module load samtools 2>/dev/null
export PYTHONPATH=/project/6014832/mforooz/EpiDenoise
export CUBLAS_WORKSPACE_CONFIG=:4096:8     # deterministic cuBLAS (set before cuBLAS init; loop-proof)
cd "$(dirname "$CAND")"
# Run via the determinism launcher (in _harness/, NOT loop-editable) so cuDNN-determinism is forced
# before the candidate imports torch — a loop editing train.py cannot reintroduce nondeterminism.
LAUNCH=/project/6014832/mforooz/EpiDenoise/sandbox/autoresearch/menu/_harness/_det_launch.py
/project/6014832/mforooz/EpiDenoise/candi_venv/bin/python -u "$LAUNCH" "$CAND"
echo "[sbatch_wrap EXITED rc=$?]"
