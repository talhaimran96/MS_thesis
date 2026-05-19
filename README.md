# Self-Supervised Fall Detection

This repository contains the codebase for the Self-Supervised Fall Detection thesis project. We explore different streams (RGB and Skeleton) using self-supervised learning for robust, generalizable, and privacy-preserving fall detection.

## Current Focus: Experiment B (Skeleton Stream)
Branch: `experiments/exp-B-skeleton-ssl`

The goal of this branch is to implement and evaluate **Experiment B: The Robustness & Privacy Test**. We are comparing two approaches using skeleton data:
1. **STH-MAE**: A Spatial-Temporal Heatmap Masked Autoencoder (Generative Self-Supervised Learning) on dense 3D Heatmap Volumes.
2. **ST-GCN**: A Spatial-Temporal Graph Convolutional Network (Contrastive Learning) on sparse 3D joints.

## Setup Instructions
1. Ensure the dataset files are populated in `Data/raw/`.
2. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Directory Structure
- `Data/`: Contains `raw`, `processed`, and `models` checkpoints.
- `src/`: Core Python modules for data parsing, heatmaps, and architectures.
- `Experiments/`: Training and evaluation scripts.
- `Planning/`: Master experiment plans and literature.
- `Reports/`: Daily notes and generated results.
- `Results/`: Logged metrics, loss graphs, and hyperparameter configs.

## Execution Guide

All commands should be executed from the root of the repository after activating the virtual environment.

### 1. SSL Pretraining (Self-Supervised Learning)
Pretrain the models on the large unlabelled NTU RGB+D dataset to learn representations.

**STH-MAE (Generative Masked Heatmap Reconstruction):**
```bash
python Experiments/pretrain_sth_mae.py --data_dir Data/raw/nturgbd_skeletons_s001_to_s017 --batch_size 8 --epochs 100 --mask_ratio 0.75
```
*To resume training from a checkpoint, add `--resume Data/models/sth_mae_pretrain/model_last.pth`*

**ST-GCN (Contrastive Learning Baseline):**
```bash
python Experiments/pretrain_st_gcn.py --data_dir Data/raw/nturgbd_skeletons_s001_to_s017 --batch_size 16 --epochs 50
```

### 2. Downstream Fine-Tuning (Non-SSL Direct Trainings)
Once the models are pretrained (or if you want to train them directly on the downstream task from scratch without pretraining), use the evaluation script.

**Fine-tune the pretrained STH-MAE on the Fall Detection Task:**
```bash
python Experiments/evaluate_stream_b.py --model_type sth_mae --pretrained_weights Data/models/sth_mae_pretrain/model_best.pth --epochs 20
```

**Fine-tune the pretrained ST-GCN on the Fall Detection Task:**
```bash
python Experiments/evaluate_stream_b.py --model_type gcn --pretrained_weights Data/models/st_gcn_pretrain/model_best.pth --epochs 20
```

*Note: If you want to run direct supervised training from scratch (no SSL), simply pass a dummy/non-existent path to `--pretrained_weights`, and the script will automatically fallback to training the randomly initialized backbone from scratch.*
