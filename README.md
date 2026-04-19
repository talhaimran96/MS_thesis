# Elderly Fall Detection (Self-Supervised Learning)

This repository contains the code and experiments for the thesis on "Self-Supervised Fall Detection". The goal of this research is to evaluate generative self-supervised learning (SSL) backbones against traditional supervised methods for fall detection in the elderly.

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd elderly_fall_detection
   ```
2. **Setup & Activate Virtual Environment**:
   ```bash
   # Create a virtual environment (if not already created)
   python3 -m venv .venv
   
   # Activate it
   source .venv/bin/activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Directory Structure

- `Data/`: Contains all datasets. Separated into `raw/`, `processed/`, and `models/`.
- `Experiments/`: Miscellaneous scripts and exploratory code.
- `Planning/`: Project plans, proposals, and the `master_experiment_plan.md`.
- `Reports/`: Daily experiment logs and `daily_notes.txt` for thesis formulation.

## Core Experiments & Branching

Each major experiment is isolated into its own git branch to maintain code integrity. As experiments are started, their respective branches will be listed below:

- **Experiment A**: `exp-A-rgb-baseline` (RGB Stream vs. Traditional Supervised Baselines)
- **Experiment B**: `exp-B-skeleton-ssl` (Skeleton Stream vs. Contrastive Learning Baselines)
- **Experiment C**: `exp-C-cross-modality` (Cross-Modality Baseline Comparison)
- **Experiment D**: `exp-D-anomaly-detection` (Anomaly Detection Proxy Tasks & Domain Generalization)
- **Experiment E**: `exp-E-reference-stream` (Reference Stream Baseline)

*(Note: These branches will be created as the respective experiments begin.)*

Please refer to `Planning/master_experiment_plan.md` for detailed procedures and experimental designs.
