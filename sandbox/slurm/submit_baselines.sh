#!/bin/bash
# Submit seven sandbox baseline training runs.
# Run from the repo root:
#   bash sandbox/slurm/submit_baselines.sh
#
# Optional overrides (export before running):
#   BASELINE_PARTITION — SLURM partition list (default: auto-derived from BASELINE_TIME).
#                        Default behaviour queries every `gpubase_bygpu_b*` partition's MaxTime
#                        on the local cluster and submits with a comma-separated list of every
#                        tier that can host BASELINE_TIME, plus `gpubackfill` when its MaxTime
#                        also covers it. Slurm then routes the job to whichever partition can
#                        give an allocation first — minimising queue wait while guaranteeing
#                        the requested walltime is feasible.
#                        Set this env var to force a single partition (e.g. `gpubase_bygpu_b2`)
#                        or to bypass the `scontrol`-based discovery on clusters where it isn't
#                        available.
#   BASELINE_DRYRUN    — if "1", print the resolved partition list and skip sbatch.
#   BASELINE_MEM       — host RAM cap (default: 32G; safe for H5 RAM-cache ≤10G + PyTorch)
#   BASELINE_GRES      — GPU request (default: gpu:nvidia_h100_80gb_hbm3_1g.10gb:1)
#                        Use gpu:nvidia_h100_80gb_hbm3_2g.20gb:1 for larger slice,
#                        or gpu:h100:1 for full GPU.
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

# ── Partition resolution ────────────────────────────────────────────────────
# Convert SLURM-style walltime ([D-]HH:MM:SS or MM:SS) into seconds.
slurm_time_to_seconds() {
  local t="$1" days=0 hms="$1"
  if [[ "$t" == *-* ]]; then
    days="${t%%-*}"
    hms="${t#*-}"
  fi
  IFS=: read -r a b c <<<"$hms"
  if [[ -z "${c:-}" ]]; then
    # MM:SS
    c="$b"; b="$a"; a=0
  fi
  echo $(( days*86400 + 10#${a:-0}*3600 + 10#${b:-0}*60 + 10#${c:-0} ))
}

# Build comma-separated list of every `gpubase_bygpu_b*` partition whose MaxTime
# covers $BASELINE_TIME, plus `gpubackfill` when applicable. Quiet `scontrol`
# failures with a single hardcoded fallback partition so submission always works.
auto_resolve_partitions() {
  local target_secs partitions=() candidate maxtime maxtime_secs
  target_secs="$(slurm_time_to_seconds "$BASELINE_TIME")"
  while read -r candidate; do
    [[ -z "$candidate" ]] && continue
    maxtime="$(scontrol show partition "$candidate" 2>/dev/null \
                 | grep -oE 'MaxTime=[^ ]+' | head -1 | cut -d= -f2)"
    [[ -z "$maxtime" || "$maxtime" == UNLIMITED ]] && { partitions+=("$candidate"); continue; }
    maxtime_secs="$(slurm_time_to_seconds "$maxtime")"
    (( maxtime_secs >= target_secs )) && partitions+=("$candidate")
  done < <(sinfo -h -o "%P" 2>/dev/null \
             | tr -d '*' | sort -u \
             | grep -E '^(gpubase_bygpu_b[0-9]+|gpubackfill)$')
  if [[ ${#partitions[@]} -eq 0 ]]; then
    echo ""  # caller falls back
  else
    local IFS=,
    echo "${partitions[*]}"
  fi
}

if [[ -n "${BASELINE_PARTITION:-}" ]]; then
  PART="$BASELINE_PARTITION"
  PART_SOURCE="explicit BASELINE_PARTITION"
else
  PART="$(auto_resolve_partitions || true)"
  if [[ -z "$PART" ]]; then
    PART="gpubase_bygpu_b2"
    PART_SOURCE="fallback (sinfo/scontrol unavailable)"
  else
    PART_SOURCE="auto from BASELINE_TIME=$BASELINE_TIME"
  fi
fi
echo "[submit_baselines] partition=$PART  ($PART_SOURCE)"

if [[ "${BASELINE_DRYRUN:-}" == "1" ]]; then
  echo "[submit_baselines] BASELINE_DRYRUN=1 → skipping sbatch."
  exit 0
fi

SBATCH_RES=(--account="$BASELINE_ACCOUNT" --partition="$PART" --mem="$MEM" --gres="$GRES" --time="$BASELINE_TIME")

# ── B1: Anchor (type1 chr19, encode transform=none) ─────────────────────────
B1=$(BASELINE_NAME="anchor" \
     BASELINE_EXTRA="--model.encode_input_transform none" \
     sbatch --job-name=sbx_b1_anchor \
            "${SBATCH_RES[@]}" \
            --parsable \
            "$SCRIPT")
echo "Submitted B1 (anchor)     → SLURM job $B1  (partition=$PART gres=$GRES mem=$MEM batch=$BASELINE_BATCH)"

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
