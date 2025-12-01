#!/bin/bash
#
# CANDI Evaluation Pipeline
# Runs predictions and all evaluation scripts on a trained model
#
# Usage: ./run_eval.sh --model-dir /path/to/model [--dataset eic|merged] [--data-path /path/to/data]
#

set -euo pipefail

# Default values
DATASET="merged"
DATA_BASE_PATH="/home/mforooz/projects/def-maxwl/mforooz"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
MODEL_DIR=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --data-path)
            DATA_BASE_PATH="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --model-dir /path/to/model [--dataset eic|merged] [--data-path /base/path]"
            echo ""
            echo "Options:"
            echo "  --model-dir   Path to trained model directory (required)"
            echo "  --dataset     Dataset type: 'merged' or 'eic' (default: merged)"
            echo "  --data-path   Base path for data directories (default: /home/mforooz/projects/def-maxwl/mforooz)"
            echo ""
            echo "Note: RNA-seq analysis always uses 'merged' dataset regardless of --dataset flag."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$MODEL_DIR" ]]; then
    echo "Error: --model-dir is required"
    exit 1
fi

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "Error: Model directory does not exist: $MODEL_DIR"
    exit 1
fi

# Construct data paths
if [[ "$DATASET" == "eic" ]]; then
    DATA_PATH="${DATA_BASE_PATH}/DATA_CANDI_EIC"
else
    DATA_PATH="${DATA_BASE_PATH}/DATA_CANDI_MERGED"
fi

MERGED_DATA_PATH="${DATA_BASE_PATH}/DATA_CANDI_MERGED"

echo "=============================================="
echo "CANDI Evaluation Pipeline"
echo "=============================================="
echo "Model directory: $MODEL_DIR"
echo "Dataset: $DATASET"
echo "Data path: $DATA_PATH"
echo "Merged data path (for RNA-seq): $MERGED_DATA_PATH"
echo "=============================================="
echo ""

# Step 1: Run predictions
echo "[1/5] Running predictions on all biosamples..."
python "${SCRIPT_DIR}/pred.py" \
    --model-dir "$MODEL_DIR" \
    --data-path "$DATA_PATH" \
    --dataset "$DATASET" \
    --all-biosamples \
    --split test \
    --get-latent-z

echo ""
echo "✓ Predictions complete"
echo ""

# Step 2: Compute metrics
echo "[2/5] Computing metrics..."
python "${SCRIPT_DIR}/eval_scripts/compute_metrics.py" \
    --model-dir "$MODEL_DIR" \
    --dataset "$DATASET"

echo ""
echo "✓ Metrics computed"
echo ""

# Step 3: Visualize prediction performance
echo "[3/5] Visualizing prediction performance..."
python "${SCRIPT_DIR}/eval_scripts/viz_pred_perf.py" \
    --model-dir "$MODEL_DIR"

echo ""
echo "✓ Prediction performance plots complete"
echo ""

# Step 4: Visualize confidence calibration
echo "[4/5] Visualizing confidence calibration..."
python "${SCRIPT_DIR}/eval_scripts/viz_conf.py" \
    --model-dir "$MODEL_DIR" \
    --dataset "$DATASET"

echo ""
echo "✓ Confidence calibration plots complete"
echo ""

# Step 5: RNA-seq evaluation (always uses merged dataset)
echo "[5/5] Running RNA-seq evaluation (using merged dataset)..."
python "${SCRIPT_DIR}/eval_scripts/viz_rnaseq.py" \
    --model-dir "$MODEL_DIR" \
    --data-path "$MERGED_DATA_PATH" \
    --run-per-assay

echo ""
echo "✓ RNA-seq evaluation complete"
echo ""

echo "=============================================="
echo "All evaluations complete!"
echo "Results saved to: ${MODEL_DIR}/viz/"
echo "Metrics saved to: ${MODEL_DIR}/preds/metrics.csv"
echo "=============================================="




