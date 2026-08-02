#!/bin/bash
# Submit seven sandbox baseline training runs.
# Run from the repo root:
#   bash sandbox/slurm/submit_baselines.sh
#
# Optional overrides (export before running):
#   BASELINE_PARTITION — escape hatch, normally leave unset. By default no --partition is
#                        passed at all: the Alliance job-submit plugin derives the partition
#                        from --time (b1 ≤3h, b2 ≤12h, b3 ≤1d, ...). Naming one it would not
#                        have chosen fails with the misleading error "The specified partition
#                        does not exist, or the submitted job cannot fit in it." Set this only
#                        to reach a partition the plugin will not pick on its own (e.g.
#                        `gpubackfill`), and expect that error if it disagrees.
#   BASELINE_DRYRUN    — if "1", print the resolved sbatch resource flags and skip sbatch.
#   BASELINE_MEM       — host RAM cap (default: 32G; safe for H5 RAM-cache ≤10G + PyTorch)
#   BASELINE_GRES      — GPU request (default: gpu:nvidia_h100_80gb_hbm3_1g.10gb:1).
#                        CLAUDE.md hard rule: every sandbox job takes the 1g.10gb MIG slice.
#                        Do NOT set this to gpu:h100:1 or any other full-GPU spec.
#   BASELINE_BATCH     — GPU batch size (default: 32; safe on 10/20GB slices for current model)
#   SANDBOX_H5, BASELINE_EPOCHS, BASELINE_SEED — forwarded like baseline_train.sh
#
#   BASELINE_EPOCHS    — training epochs for all baselines (default: 200)
#   BASELINE_TIME      — SLURM walltime (default: 03:00:00), e.g. 04:00:00
#                        Passed as sbatch --time; overrides #SBATCH --time in baseline_train.sh
#   BASELINE_ACCOUNT   — SLURM account (default: def-maxwl_gpu, valid on fir for GPU jobs).
#                        Passed as sbatch --account; overrides #SBATCH --account in baseline_train.sh.
#                        On clusters that use the bare "def-maxwl" account, set this to "def-maxwl".
#
#   BASELINE_REGIME_CONFIG — override regime overlay (default now type1_chr19)
#
# Each job uses baseline_train.sh. Results: sandbox/runs/baseline_<name>/ + W&B.

set -euo pipefail
SCRIPT="$(dirname "$0")/baseline_train.sh"

export BASELINE_BATCH="${BASELINE_BATCH:-32}"
export BASELINE_EPOCHS="${BASELINE_EPOCHS:-200}"
export BASELINE_TIME="${BASELINE_TIME:-03:00:00}"
export BASELINE_ACCOUNT="${BASELINE_ACCOUNT:-def-maxwl_gpu}"
MEM="${BASELINE_MEM:-32G}"
GRES="${BASELINE_GRES:-gpu:nvidia_h100_80gb_hbm3_1g.10gb:1}"

# ── Partition ───────────────────────────────────────────────────────────────
# Deliberately not resolved here. The Alliance job-submit plugin picks the bin
# from --time, so passing --partition can only ever agree with it or break the
# submission. BASELINE_PARTITION stays available as an escape hatch (see header).
SBATCH_RES=(--account="$BASELINE_ACCOUNT" --mem="$MEM" --gres="$GRES" --time="$BASELINE_TIME")
PART_DESC="plugin-derived from --time=$BASELINE_TIME"
if [[ -n "${BASELINE_PARTITION:-}" ]]; then
  SBATCH_RES+=(--partition="$BASELINE_PARTITION")
  PART_DESC="forced via BASELINE_PARTITION=$BASELINE_PARTITION"
fi
echo "[submit_baselines] partition=$PART_DESC  gres=$GRES mem=$MEM time=$BASELINE_TIME"

if [[ "${BASELINE_DRYRUN:-}" == "1" ]]; then
  echo "[submit_baselines] BASELINE_DRYRUN=1 → skipping sbatch."
  printf '[submit_baselines] would submit with:'; printf ' %q' "${SBATCH_RES[@]}"; echo
  exit 0
fi

# ── B1: Anchor (type1 chr19, encode transform=none) ─────────────────────────
B1=$(BASELINE_NAME="anchor" \
     BASELINE_EXTRA="--model.encode_input_transform none" \
     sbatch --job-name=sbx_b1_anchor \
            "${SBATCH_RES[@]}" \
            --parsable \
            "$SCRIPT")
echo "Submitted B1 (anchor)     → SLURM job $B1  (gres=$GRES mem=$MEM batch=$BASELINE_BATCH)"

# ── B2: DSF=1 only (type1 chr19, encode transform=none) ─────────────────────
B2=$(BASELINE_NAME="dsf1_only" \
     BASELINE_EXTRA="--training.dsf.sampling off --model.encode_input_transform none" \
     sbatch --job-name=sbx_b2_dsf1 \
            "${SBATCH_RES[@]}" \
            --parsable \
            "$SCRIPT")
echo "Submitted B2 (dsf1_only)  → SLURM job $B2"

# ── B3: Pure assay masking (type1 chr19, encode transform=none) ─────────────
B3=$(BASELINE_NAME="assay_mask_only" \
     BASELINE_EXTRA="--training.masking.p_full_assay 1.0 --training.masking.p_full_loci 0.0 --model.encode_input_transform none" \
     sbatch --job-name=sbx_b3_assay \
            "${SBATCH_RES[@]}" \
            --parsable \
            "$SCRIPT")
echo "Submitted B3 (assay_mask) → SLURM job $B3"

# ── B4: SGD baseline (lr=1e-4, type1 chr19, encode transform=none) ─────────
B4=$(BASELINE_NAME="sgd_lr1e4" \
     BASELINE_EXTRA="--training.optimizer.name sgd --training.optimizer.sgd.lr 1e-4 --model.encode_input_transform none" \
     sbatch --job-name=sbx_b4_sgd \
            "${SBATCH_RES[@]}" \
            --parsable \
            "$SCRIPT")
echo "Submitted B4 (sgd_lr1e4)  → SLURM job $B4"

# ── B5: Type2 loci only (encode transform=none) ─────────────────────────────
B5=$(BASELINE_NAME="type2_loci_only" \
     BASELINE_REGIME_CONFIG="sandbox/configs/type2_loci.yaml" \
     BASELINE_EXTRA="--model.encode_input_transform none" \
     sbatch --job-name=sbx_b5_type2 \
            "${SBATCH_RES[@]}" \
            --parsable \
            "$SCRIPT")
echo "Submitted B5 (type2_loci) → SLURM job $B5"

# ── B6: Clip-by-value baseline (type1 chr19, encode transform=none) ────────
B6=$(BASELINE_NAME="clip_value_type1" \
     BASELINE_EXTRA="--training.grad.clip_mode value --model.encode_input_transform none" \
     sbatch --job-name=sbx_b6_clipval \
            "${SBATCH_RES[@]}" \
            --parsable \
            "$SCRIPT")
echo "Submitted B6 (clip_value) → SLURM job $B6"

# ── B7: log1p encode transform baseline (type1 chr19) ──────────────────────
B7=$(BASELINE_NAME="log1p_type1" \
     BASELINE_EXTRA="--model.encode_input_transform log1p" \
     sbatch --job-name=sbx_b7_log1p \
            "${SBATCH_RES[@]}" \
            --parsable \
            "$SCRIPT")
echo "Submitted B7 (log1p)      → SLURM job $B7"

echo ""
echo "Monitor:       squeue -u \$USER"
echo "W&B project:   candi_sandbox  (runs: baseline_anchor, baseline_dsf1_only, baseline_assay_mask_only, baseline_sgd_lr1e4, baseline_type2_loci_only, baseline_clip_value_type1, baseline_log1p_type1)"
echo "Logs:          sandbox/slurm_logs/"
echo "Run dirs:      sandbox/runs/baseline_*/"
