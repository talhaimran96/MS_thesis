#!/bin/bash

# Exit on error
set -e

echo "Starting Stream A Data Ablation Experiments..."

# Configuration
SPLITS=(0.1 0.5 1.0)
EPOCHS=20
FREEZE_EPOCHS=5
PRETRAINED_WEIGHTS="Data/models/exp-A-rgb-baseline_videomae/model_last.pth"

# Activate environment if not already activated
if [ -z "$VIRTUAL_ENV" ]; then
    source .venv/bin/activate
fi

for SPLIT in "${SPLITS[@]}"; do
    echo "========================================================="
    echo "Running Ablation for Split: ${SPLIT}"
    echo "========================================================="

    echo "-> 1. Control Model: Baseline 3D CNN (Scratch)"
    python Experiments/evaluate_stream_a.py --model_type baseline_3dcnn --split ${SPLIT} --epochs ${EPOCHS}
    
    echo "-> 2. Control Model: VideoMAE (Scratch - No SSL)"
    python Experiments/evaluate_stream_a.py --model_type videomae --pretrained_weights none --split ${SPLIT} --epochs ${EPOCHS} --freeze_epochs 0
    
    echo "-> 3. Variable Model: VideoMAE (Pretrained with SSL)"
    python Experiments/evaluate_stream_a.py --model_type videomae --pretrained_weights "${PRETRAINED_WEIGHTS}" --split ${SPLIT} --epochs ${EPOCHS} --freeze_epochs ${FREEZE_EPOCHS}

done

echo "All data ablation experiments completed successfully!"
