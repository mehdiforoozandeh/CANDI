#!/usr/bin/env bash
# wait_for_slot.sh — block until a training slot is free and VRAM is available.
# Usage:  MAX_TRAINING=2 MIN_VRAM_MB=300 bash sandbox/autoresearch/june3/wait_for_slot.sh
# Exits 0 when safe to launch.
MAX_TRAINING="${MAX_TRAINING:-2}"
MIN_VRAM_MB="${MIN_VRAM_MB:-300}"

echo "[wait_for_slot] max_training=${MAX_TRAINING} min_vram_mb=${MIN_VRAM_MB}"

while true; do
    n_train=$(ps aux | grep '[t]rain' | grep mforooz | grep -v grep | wc -l || echo 99)
    n_train=$(echo "$n_train" | tr -d ' ')
    vram_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 99999)
    vram_free=$(echo "$vram_free" | tr -d ' ')

    if [ "$n_train" -lt "$MAX_TRAINING" ] && [ "$vram_free" -gt "$MIN_VRAM_MB" ]; then
        echo "[wait_for_slot] slot available: trainings=${n_train}/${MAX_TRAINING} vram_free=${vram_free}MB"
        exit 0
    fi

    echo "[wait_for_slot] waiting: trainings=${n_train}/${MAX_TRAINING} vram_free=${vram_free}MB"
    sleep 30
done
