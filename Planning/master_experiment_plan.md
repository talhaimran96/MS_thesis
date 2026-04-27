# Master Experiment Plan: Self-Supervised Fall Detection

## Overview
This document serves as the master reference for all experiments related to the Self-Supervised Fall Detection thesis. It outlines the experimental designs, architectures, data processing pipelines, and standardized procedures for conducting and recording experiments.

## Standard Operating Procedures (Instructions for Agents & User)

1. **Branching Strategy**: Each distinct experiment must be conducted on its own dedicated Git branch (e.g., `exp-A-rgb-baseline`, `exp-B-skeleton-ssl`). Do not commit experimental code directly to `main`.
2. **Daily Reporting**: Record each day's experiment plan, progress, and results in a Markdown file located in the `Reports/` directory. These files will aggregate reports across all branches (e.g., `Reports/2026-04-19-experiment-A-setup.md`).
3. **Continuous Notes Logging**: All raw data, daily observations, terminal outputs, and collaborative comments between the user and agents must be logged in `Reports/daily_notes.txt`. This single file will serve as the primary source material for drafting the final thesis report.
4. **Coding Standards**: Always adhere to PEP8 formatting and coding standards for all Python scripts.
5. **Virtual Environment**: All Python code must be executed within the project's virtual environment. Always activate it using `source .venv/bin/activate` before running scripts or installing packages.
6. **Dependency Management**: Any time a new Python dependency is installed, the agent must immediately update `requirements.txt`.
7. **Documentation Maintenance**: The agent must continuously update the project `README.md` with any new setup instructions, newly established branches, or major updates to the experiments.
8. **Experiment Tracking**: Use the `ExperimentLogger` (`src/utils/logger.py`) for all training scripts. It automatically saves hyperparams, training/validation metrics, and loss graphs to the `Results/{branch_name}_{model_type}/` folder.
9. **Model Saving Conventions**: All models should be saved in `Data/models/{branch_name}/`. Always save the best performing model based on validation loss as `{model_type}_best.pth` and the final epoch model as `{model_type}_last.pth`.
10. **Resumable Training**: All training scripts must be written to be resumable from a checkpoint. Include an `argparse` flag like `--resume` that loads the model's weights, the optimizer's state, and the starting epoch from the saved `.pth` file, minimizing data loss if a run is interrupted.

## Data to Record
For every experiment, ensure the following metrics and configurations are explicitly documented in the respective daily report:

- **Experiment Metadata**: Branch name, date, and specific dataset splits used.
- **Architectures & Hyperparameters**: Model utilized (e.g., ViT, STH-MAE, GCN, 3D CNN), masking ratio, learning rate, batch size, and epochs.
- **Evaluation Metrics (Clinical & Technical)**:
  - **Sensitivity & Specificity**: To evaluate true fall detection vs. missed alarms.
  - **F1-Score**: The primary performance metric to handle severe class imbalance (99% non-fall activities).
  - **False Alarm Rate (FAR)**: Critical clinical metric to measure and minimize the potential for caregiver "alert fatigue".
  - **Computational Latency (FPS)**: To evaluate real-time feasibility and deployment on edge devices.

## Data Directory Structure
The `Data/` folder has been structured to separate raw data from processed datasets:
- `Data/raw/`: Original, unmodified datasets (`GMDCSA-24`, `CompleteDataSet.csv`, etc.).
- `Data/processed/`: Preprocessed data ready for training (e.g., extracted $224 \times 224$ RGB frames, generated 3D Spatial-Temporal Heatmap Volumes).
- `Data/models/`: Saved model weights, checkpoints, and logs.

### Current Datasets Inventory (Data/raw)

| Dataset Name | Path | Structure & How to Read | Status |
| :--- | :--- | :--- | :--- |
| **GMDCSA-24** | `Data/raw/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-master/` | Subfolders for each `Subject` (1-4), split into `Fall/` and `ADL/` containing `.csv` files. | ✅ Present |
| **UP-Fall (Sensor Data)** | `Data/raw/CompleteDataSet.csv` | 47-column CSV (294k rows). Use `pandas.read_csv(path, low_memory=False)`. Contains Accelerometer, Angular Velocity, Luminosity, IR, BrainSensor, Subject, Activity, Tag. | ✅ Present (Sensors only) |
| **UP-Fall (Video Data)** | *Missing* | Multi-camera RGB video clips for Streams A & B. | ❌ Missing (Could not find images/videos) |
| **Kinetics-400** | `Data/raw/kinetics400_5per/` | Video dataset for ViT pretraining. Structure: `train/` directory containing subfolders for each action class. | ✅ Present (5% subset) |
| **NTU RGB+D (120)** | `Data/raw/nturgbd_skeletons_s001_to_s017/` | Large-scale skeleton dataset for STH-MAE pretraining. | ✅ Present |
| **OOPS-Fall** | `Data/raw/oops_dataset/` | Unpredictable "wild" real-world accident videos. Structure: `video/oops_video/` contains clips, `annotations/` contains json/txt labels and splits. | ✅ Present |

---

## Experiment Descriptions & Data Pipelines

### Data Pipelines
1. **RGB Stream (Stream A)**: Process raw RGB video clips into sequences of 16 frames at a $224 \times 224$ resolution.
2. **Skeleton Stream (Stream B)**: Extract 17 body keypoints per frame using YOLO-Pose or MediaPipe, converting these raw coordinate sequences into dense 3D Spatial-Temporal Heatmap Volumes to reduce sensitivity to sensor noise and poor lighting.

### Experiment A: RGB Stream vs. Traditional Supervised Baselines
- **Proposed Architecture**: Vision Transformer (ViT-Base and ViT-Large) acting as VideoMAE V2.
- **Baseline Architecture**: Traditional supervised 3D CNNs.
- **Pretraining Task**: Train on unlabeled Kinetics-400 or NTU RGB+D using Masked Video Reconstruction with a 90% Tube Masking ratio.
- **Comparison Goal**: Quantify the reduction in manually labeled training data required when utilizing generative SSL compared to supervised methods.

### Experiment B: Skeleton Stream vs. Contrastive Learning Baselines
- **Proposed Architecture**: Spatial-Temporal Transformer (STH-MAE) utilizing dense 3D Heatmap Volumes.
- **Baseline Architecture**: Graph Convolutional Networks (GCNs) on sparse, raw 3D joints/graphs.
- **Pretraining Task**: Train ST-Transformer on NTU RGB+D 120 (Skeleton data) using Masked Heatmap Reconstruction with random 3D masking.
- **Baseline Pretraining Task**: Contrastive Learning using InfoNCE optimization.
- **Comparison Goal**: Test if the generative heatmap representation offers the optimal "middle ground" of privacy and noise robustness compared to the contrastive baseline.

### Experiment C: Cross-Modality Baseline Comparison
- **Goal**: Evaluate the performance trade-offs directly between Stream A (RGB pixel-level video data) and Stream B (Skeleton geometric data). Assess accuracy, computational latency, and privacy preservation.

### Experiment D: Anomaly Detection Proxy Tasks & Domain Generalization
- **Proxy Tasks (Fine-Tuning)**: Integrate multi-task proxy learning to distinguish the "Arrow of Time" (forward vs. backward sequences) and "Motion Irregularity" (smooth vs. shuffled frames).
- **Domain Generalization**: Train models on "staged" datasets (UP-Fall) and evaluate their ability to bridge the domain gap by testing on "wild" real-world sets (GMDCSA-24 and OOPS-Fall).

### Experiment E: Reference Stream Baseline
- **Proposed Architecture**: Reference stream utilizing MediaPipe or open-source object detection for pose estimation, followed by classification via GNNs, LSTMs, and traditional ML classifiers.
- **Comparison Goal**: Establish a baseline for pose-based detection and compute downward/fall velocity against complex generative models.
