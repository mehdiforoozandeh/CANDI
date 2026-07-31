#!/bin/bash
# candi_kit — bake: ENCODE-style directory -> HDF5 (schema v2). Pure IO + numpy; no GPU compute.
# Env preamble copied from sandbox/diagnostics/dual_conditioning_real/jobs/wd0.sh (the recorded jobs).
# Directory is slurm/ and NOT jobs/ on purpose — .gitignore:15 has an unanchored `jobs/` rule that
# would silently swallow these scripts (same for models/, logs/).
#
# WHY --gres ON A CPU-ONLY JOB: submitted without it, this routes to the def-maxwl_cpu account, whose
# fairshare is 0.088 vs def-maxwl_gpu's 0.435. Combined with the standing maintenance reservations
# (CDUMaintenance2 alone holds 432 nodes), a 2-core/30G bake lands at rank ~875/876 in
# cpubase_bycore_b1 (876 pending vs 20 running) and effectively never starts. Requesting the smallest
# MIG slice routes it through the GPU account instead, where this project's jobs actually run. The slice
# is idle for the ~20 min the bake takes; that is the intended trade. Drop --gres if the CPU queue
# recovers. Never use any other gres spec (project hard rule).
#   sbatch candi_kit/slurm/bake.sh
# Logs go to ./slurm-logs/ RELATIVE TO WHERE YOU SUBMIT FROM. SLURM resolves --output against
# the submitting cwd, not the script's location, so `mkdir -p slurm-logs` first (or pass
# `sbatch --output=... --error=...` to override). Submitting from anywhere is then fine.
#SBATCH --account=def-maxwl
#SBATCH --job-name=candi_kit_bake
#SBATCH --output=slurm-logs/bake_%j.out
#SBATCH --error=slurm-logs/bake_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=30G
#SBATCH --cpus-per-task=2

set -uo pipefail

# --- edit these four ---------------------------------------------------------------------------
KIT="${KIT:-/project/6014832/mforooz/EpiDenoise/candi_kit}"
# VENV: defaults to the environment you are ALREADY in ($VIRTUAL_ENV), which is what README §4.0
# leaves you in. Override with VENV=/path/to/venv. It must NOT default to someone else's venv --
# sourcing a path you do not own fails late, inside python, not at the source line.
VENV="${VENV:-${VIRTUAL_ENV:-}}"
ROOT="${ROOT:-/project/6014832/mforooz/DATA_CANDI_EIC}"          # ENCODE-style biosample dirs
SIDE="${SIDE:-/project/6014832/mforooz/EpiDenoise/data}"          # hg38.fa, hg38.chrom.sizes, ...
PANEL="${PANEL:-$KIT/configs/panel.q19.json}"
OUT="${OUT:-/scratch/$USER/candi_kit/q19.h5}"                     # /scratch: /project is 82% full
# -----------------------------------------------------------------------------------------------

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1; unset PYTHONPATH || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLBACKEND=Agg
export WANDB_MODE=disabled
if [ -n "$VENV" ]; then
  source "$VENV/bin/activate"
else
  echo "[error] no environment: set VENV=/path/to/venv, or sbatch from an active venv" >&2; exit 1
fi

cd "$KIT"     # the kit is pip-installed (`pip install -e candi_kit/`); cwd is irrelevant
mkdir -p "$(dirname "$OUT")"

echo "[bake] host=$(hostname) panel=$PANEL out=$OUT"

# No --blacklist: the repo's data/hg38_blacklist_v2.bed is 8 bytes containing "# Empty", so passing it
# would only look like filtering. Add `--blacklist <bed>` once you have a real one (the kit warns when
# none is given).
#
# type2 loci: the recorded q19 anchors were measured on an h5 of 7,485 windows = 5,485 type1 (chr19
# 3,053 train + chr21 2,432 eval) PLUS 2,000 type2 (1,000 cCRE + 1,000 non-cCRE sampled across the
# other chromosomes). Baking with --type2-* 0 reproduces the type1 core exactly but drops those 2,000,
# which makes the run non-comparable to the anchors. Keep 1000/1000 to reproduce; set both to 0 for a
# faster chr19-only run, and then do NOT compare the result to the recorded numbers.
# --allow-missing-control: B_DND-41 carries ONLY ATAC-seq and DNase-seq — accessibility assays, which
# need no ChIP input control — so it has no chipseq-control/ directory at all and its control channel is
# the MISSING(-1) sentinel in every window. The F15 gate is correct to stop on this by default (a silently
# absent control IS a capability loss for a ChIP panel); here it is the real structure of the data, and
# the original sandbox.h5 bake stored the identical -1 sentinel. Note this contradicts the "control is
# always available" invariant stated in the project docs. Re-check F15 output whenever the panel changes.
python -m candi_kit.prep.bake \
  --root "$ROOT" \
  --panel "$PANEL" \
  --out "$OUT" \
  --fasta "$SIDE/hg38.fa" \
  --chrom-sizes "$SIDE/hg38.chrom.sizes" \
  --ccres "$SIDE/GRCh38-cCREs.bed" \
  --type2-ccre 1000 --type2-non 1000 \
  --allow-missing-control \
  --seed 42
rc=$?

echo "[bake] DONE rc=$rc"
exit $rc
