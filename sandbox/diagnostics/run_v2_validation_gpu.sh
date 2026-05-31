#!/usr/bin/env bash
# CANDI v2 validation GPU smoke runner (no wandb).
set -euo pipefail
cd "$(dirname "$0")/../.."
source candi_venv/bin/activate

RUN_ROOT="sandbox/runs"
mkdir -p "$RUN_ROOT"

train_cell() {
  local heads="$1"
  local count_head="$2"
  local run_dir="${RUN_ROOT}/validation_${heads}_${count_head}"
  echo "=== TRAIN ${heads} ${count_head} ==="
  python -m sandbox.train_candi_v2 \
    --no-wandb \
    --run-dir "$run_dir" \
    --set "decoder.heads=${heads}" \
    --set "decoder.count_head=${count_head}" \
    --set training.epochs=2 \
    --set training.max_train_batches=30 \
    --set training.eval_max_batches=10 \
    --set data.regime=type2_loci \
    --set training.training_stats_jsonl_every_n_steps=10 \
    --set training.batch_size=8
}

for heads in count_only count_peak all; do
  for ch in plain depth_offset; do
    train_cell "$heads" "$ch"
  done
done

echo "=== CHECKPOINT RESUME ==="
RESUME_DIR="${RUN_ROOT}/validation_resume"
python -m sandbox.train_candi_v2 --no-wandb --save-checkpoint \
  --run-dir "$RESUME_DIR" \
  --set training.epochs=1 \
  --set training.max_train_batches=20 \
  --set data.regime=type2_loci \
  --set decoder.heads=count_peak

python -m sandbox.train_candi_v2 --no-wandb \
  --run-dir "$RESUME_DIR" \
  --resume "${RESUME_DIR}/checkpoint_last.pt" \
  --set training.epochs=2 \
  --set training.max_train_batches=20 \
  --set data.regime=type2_loci \
  --set decoder.heads=count_peak

echo "=== GRADIENT AUDIT ==="
for heads in count_only count_peak all; do
  for ch in plain depth_offset; do
    python -m sandbox.diagnostics.v2_gradient_audit \
      --device cuda \
      --heads "$heads" \
      --count-head "$ch" \
      --output-dir "${RUN_ROOT}/validation_gradient_audit"
  done
done

echo "ALL GPU VALIDATION STEPS DONE"
