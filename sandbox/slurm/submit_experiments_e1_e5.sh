#!/bin/bash
# Submit experiments E1–E5 (post 2026-04-26 baselines) per the HPO plan
# discussed on 2026-04-27.
#
# E1: lrfloor_low      — anchor + min_lr_ratio: 0.1 → 0.01 (tests F3)
# E2: head_count_only  — only count head trains (count_obs/count_imp = 1, others = 0)
# E3: head_pval_only   — only pval head trains (pval_obs/pval_imp = 1, others = 0)
# E4: head_peak_only   — only peak head trains (peak_obs/peak_imp = 1, others = 0)
# E5: head_count_peak  — count + peak heads only (pval_* = 0)
#
# All runs:
#   - type1_chr19 regime (chr19 train / chr21 eval)
#   - 200 epochs, batch 32, walltime 03:00:00
#   - log1p input transform (now the default, no explicit flag)
#   - parent = baseline_anchor in the HPO graph (sandbox/hpo_graph.json)
#
# Run from the repo root:
#   bash sandbox/slurm/submit_experiments_e1_e5.sh
#
# Mostly inherits behaviour from submit_baselines.sh — only the run definitions differ.
set -euo pipefail
SCRIPT="$(dirname "$0")/baseline_train.sh"

export BASELINE_BATCH="${BASELINE_BATCH:-32}"
export BASELINE_EPOCHS="${BASELINE_EPOCHS:-200}"
export BASELINE_TIME="${BASELINE_TIME:-03:00:00}"
export BASELINE_ACCOUNT="${BASELINE_ACCOUNT:-def-maxwl_gpu}"
export BASELINE_PREFIX=""   # Experiments land at sandbox/runs/<NAME>/ (no baseline_ prefix)
MEM="${BASELINE_MEM:-32G}"
GRES="${BASELINE_GRES:-gpu:nvidia_h100_80gb_hbm3_1g.10gb:1}"

# ── Partition (mirrors submit_baselines.sh) ─────────────────────────────────
# Deliberately not resolved here. The Alliance job-submit plugin picks the bin
# from --time, so passing --partition can only ever agree with it or break the
# submission. BASELINE_PARTITION stays available as an escape hatch.
SBATCH_RES=(--account="$BASELINE_ACCOUNT" --mem="$MEM" --gres="$GRES" --time="$BASELINE_TIME")
PART_DESC="plugin-derived from --time=$BASELINE_TIME"
if [[ -n "${BASELINE_PARTITION:-}" ]]; then
  SBATCH_RES+=(--partition="$BASELINE_PARTITION")
  PART_DESC="forced via BASELINE_PARTITION=$BASELINE_PARTITION"
fi
echo "[submit_experiments_e1_e5] partition=$PART_DESC  gres=$GRES mem=$MEM time=$BASELINE_TIME"

if [[ "${BASELINE_DRYRUN:-}" == "1" ]]; then
  echo "[submit_experiments_e1_e5] BASELINE_DRYRUN=1 → skipping sbatch."
  printf '[submit_experiments_e1_e5] would submit with:'; printf ' %q' "${SBATCH_RES[@]}"; echo
  exit 0
fi

PARENT="baseline_anchor"   # All E1-E5 runs derive from the anchor baseline

# ── E1: LR-floor lowered (tests F3 = late-stage divergence under high LR floor) ──
E1=$(BASELINE_NAME="E1_lrfloor_low" \
     BASELINE_EXTRA="--training.schedule.min_lr_ratio 0.01 --hpo.parent ${PARENT} --hpo.experiment_label E1_lrfloor_low" \
     sbatch --job-name=sbx_e1_lrfloor \
            "${SBATCH_RES[@]}" \
            --parsable \
            "$SCRIPT")
echo "Submitted E1 (lrfloor_low)        → SLURM job $E1"

# ── E2: Head isolation — count head only ────────────────────────────────────
E2=$(BASELINE_NAME="E2_head_count_only" \
     BASELINE_EXTRA="--training.loss_weights.pval_weight 0.0 --training.loss_weights.peak_weight 0.0 --hpo.parent ${PARENT} --hpo.experiment_label E2_head_count_only" \
     sbatch --job-name=sbx_e2_count \
            "${SBATCH_RES[@]}" \
            --parsable \
            "$SCRIPT")
echo "Submitted E2 (head_count_only)    → SLURM job $E2"

# ── E3: Head isolation — pval head only ─────────────────────────────────────
E3=$(BASELINE_NAME="E3_head_pval_only" \
     BASELINE_EXTRA="--training.loss_weights.count_weight 0.0 --training.loss_weights.peak_weight 0.0 --hpo.parent ${PARENT} --hpo.experiment_label E3_head_pval_only" \
     sbatch --job-name=sbx_e3_pval \
            "${SBATCH_RES[@]}" \
            --parsable \
            "$SCRIPT")
echo "Submitted E3 (head_pval_only)     → SLURM job $E3"

# ── E4: Head isolation — peak head only ─────────────────────────────────────
E4=$(BASELINE_NAME="E4_head_peak_only" \
     BASELINE_EXTRA="--training.loss_weights.count_weight 0.0 --training.loss_weights.pval_weight 0.0 --hpo.parent ${PARENT} --hpo.experiment_label E4_head_peak_only" \
     sbatch --job-name=sbx_e4_peak \
            "${SBATCH_RES[@]}" \
            --parsable \
            "$SCRIPT")
echo "Submitted E4 (head_peak_only)     → SLURM job $E4"

# ── E5: Head ablation — count + peak (pval muted), the cleanest anchor delta ──
E5=$(BASELINE_NAME="E5_head_count_peak" \
     BASELINE_EXTRA="--training.loss_weights.pval_weight 0.0 --hpo.parent ${PARENT} --hpo.experiment_label E5_head_count_peak" \
     sbatch --job-name=sbx_e5_countpeak \
            "${SBATCH_RES[@]}" \
            --parsable \
            "$SCRIPT")
echo "Submitted E5 (head_count_peak)    → SLURM job $E5"

echo ""
echo "Monitor:       squeue -u \$USER"
echo "W&B project:   candi_sandbox  (runs: E1_lrfloor_low, E2_head_count_only, E3_head_pval_only, E4_head_peak_only, E5_head_count_peak)"
echo "Logs:          sandbox/slurm_logs/"
echo "Run dirs:      sandbox/runs/E[1-5]_*/"
echo "HPO graph:     python -m sandbox.hpo.view_graph --leaderboard"
