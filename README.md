# Robust activity and behavior recognition system for elderly safety monitoring and fall detection

This repository contains the codebase and experiments for the thesis on self-supervised fall detection.

## Repository Structure

The main content and experiments are organized into two separate branches, focusing on different data streams (RGB and Skeleton).

### 1. RGB Baseline Stream (`experiments/exp-A-rgb-baseline`)
Switch to this branch:
```bash
git checkout experiments/exp-A-rgb-baseline
```

**What it contains:**
This branch implements **Experiment A**, which focuses on evaluating RGB pixel-level video data using traditional supervised baselines and self-supervised Vision Transformers. It includes:
- The VideoMAE V2 architecture utilizing Masked Video Reconstruction on unlabeled datasets (e.g., Kinetics-400).
- Traditional supervised 3D CNN baselines.
- Downstream fine-tuning on the GMDCSA-24 dataset to quantify the reduction in manually labeled data required when using generative SSL versus supervised approaches.
- Analysis of data ablation, domain generalization (testing on OOPS-Fall), and masking ratios.

### 2. Skeleton SSL Stream (`experiments/exp-B-skeleton-ssl`)
Switch to this branch:
```bash
git checkout experiments/exp-B-skeleton-ssl
```

**What it contains:**
This branch implements **Experiment B**, testing the robustness and privacy preservation of skeleton geometric data representations. It includes:
- The Spatial-Temporal Transformer (STH-MAE) utilizing dense 3D Heatmap Volumes for generative Self-Supervised Learning.
- Spatial-Temporal Graph Convolutional Networks (ST-GCN) on sparse 3D joints acting as a contrastive learning baseline.
- Skeleton pose detection processing components used for fine-tuning fall classifiers.
- Pretraining tasks that compare generative heatmap reconstruction against contrastive learning baselines to find the optimal balance of privacy and noise robustness.
