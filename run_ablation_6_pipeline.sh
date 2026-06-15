#!/bin/bash
# Pretrain STH-MAE with 0.5 and 0.75 mask ratios and run downstream fine-tuning

set -e # Exit on error

echo "=== 1. Pretraining STH-MAE with mask_ratio=0.5 ==="
# python Experiments/pretrain_sth_mae.py --mask_ratio 0.5 --epochs 100

echo "=== 2. Pretraining STH-MAE with mask_ratio=0.75 ==="
# python Experiments/pretrain_sth_mae.py --mask_ratio 0.75 --epochs 100

echo "=== 3. Copying pretrained weights to models/ directory ==="
mkdir -p models
cp Data/models/sth_mae_pretrain_0.5/model_best.pth models/sth_mae_weights_0.5.pth
cp Data/models/sth_mae_pretrain_0.75/model_best.pth models/sth_mae_weights_0.75.pth

echo "=== 4. Running Ablation 6 (Downstream fine-tuning) ==="
python Experiments/evaluate_stream_b.py --model_type sth_mae --pretrained_weights models/sth_mae_weights_0.5.pth --mask_ratio 0.5 --epochs 20
python Experiments/evaluate_stream_b.py --model_type sth_mae --pretrained_weights models/sth_mae_weights_0.75.pth --mask_ratio 0.75 --epochs 20

echo "=== Ablation 6 Pipeline Complete! ==="
