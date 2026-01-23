# CCT-LSTM Reproduce

A complete, end-to-end pipeline for reproducing the two-stage **CCT-LSTM** workflow on the [UBFC-Phys](https://search-data.ubfc.fr/FR-18008901306731-2022-05-05_UBFC-Phys-A-Multimodal-Dataset-For.html) dataset.

- **Stage 1**: Single-modality CCT pre-training (facial landmark and rPPG branches, separately).
- **Stage 2**: Multimodal CCT-LSTM fine-tuning for stress classification tasks.

Tested on Python 3.10 / CUDA 12.1 with conda. WandB logging is optional.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Dataset Preparation (UBFC-Phys)](#2-dataset-preparation-ubfc-phys)
3. [Repository Structure](#3-repository-structure)
4. [Stage 1: CCT Pre-training](#4-stage-1-cct-pre-training)
5. [Stage 2: CCT-LSTM Fine-tuning](#5-stage-2-cct-lstm-fine-tuning)
6. [One-shot Pipeline Runner](#6-one-shot-pipeline-runner)
7. [EDA Visualization Scripts](#7-eda-visualization-scripts-optional)
8. [WandB Logging](#8-wandb-logging)
9. [Known Issues & Tips](#9-known-issues--tips)
10. [References and Citation](#10-references-and-citation)
11. [License](#11-license)

---

## 1. Environment Setup

### 1.1 Create Conda Environment

```bash
conda env create -f environment.yml
conda activate cctlstm
```

### 1.2 Verify Installation

```bash
python - <<'PY'
import sys, torch, torchvision, cv2, mediapipe as mp, numpy as np, PIL
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
print("TorchVision:", torchvision.__version__)
print("OpenCV:", cv2.__version__)
print("MediaPipe:", mp.__version__)
print("NumPy:", np.__version__, "| Pillow:", PIL.__version__)
PY
```

Expected output (versions may vary):
```
Python: 3.10.x
PyTorch: 2.5.1+cu121 | CUDA available: True
TorchVision: 0.20.1
OpenCV: 4.11.0
MediaPipe: 0.10.21
NumPy: 1.26.x | Pillow: 11.2.1
```

---

## 2. Dataset Preparation (UBFC-Phys)

### 2.1 Download the Dataset

Download UBFC-Phys from [dat@UBFC](https://search-data.ubfc.fr/FR-18008901306731-2022-05-05_UBFC-Phys-A-Multimodal-Dataset-For.html) and extract it.

### 2.2 Expected Dataset Structure

The dataset should be organized as follows:

```
<your_download_location>/
└── UBFC-Phys/
    └── Data/
        ├── s1/
        │   ├── vid_s1_T1.avi
        │   ├── vid_s1_T2.avi
        │   ├── vid_s1_T3.avi
        │   ├── bvp_s1_T1.csv
        │   ├── bvp_s1_T2.csv
        │   ├── bvp_s1_T3.csv
        │   ├── eda_s1_T1.csv
        │   ├── eda_s1_T2.csv
        │   ├── eda_s1_T3.csv
        │   ├── info_s1.txt
        │   └── selfReportedAnx_s1.csv
        ├── s2/
        │   └── ...
        └── s56/
            └── ...
```

![Dataset Structure](imgs/structure.png)

### 2.3 Pipeline Workflow (Data Processing)

The `main.py` script orchestrates all data preprocessing steps. Run them **in order**:

#### Step 1: Integrity Check & Manifest Creation (Mandatory)

```bash
python main.py check
```

This will:
- Verify all 56 subject folders exist with required files.
- Create `UBFC_data_path.txt` in the repo root (stores the path to your dataset).
- Create `master_manifest.csv` under `UBFC-Phys/` (maps subjects to ctrl/test groups).

The integrity check validates:
1. `UBFC-Phys/Data/` folder exists.
2. 56 subject folders (`s1` to `s56`) are present.
3. Each subject folder contains:
   - 3 video files: `vid_sXX_T1.avi`, `vid_sXX_T2.avi`, `vid_sXX_T3.avi`
   - 3 BVP files: `bvp_sXX_T1.csv`, `bvp_sXX_T2.csv`, `bvp_sXX_T3.csv`
   - 3 EDA files: `eda_sXX_T1.csv`, `eda_sXX_T2.csv`, `eda_sXX_T3.csv`
   - 1 info file: `info_sXX.txt`
   - 1 self-report file: `selfReportedAnx_sXX.csv`

#### Step 2: Facial Landmark Extraction

```bash
python main.py extract
```

Extracts facial landmarks from all videos using MediaPipe Face Landmarker. Outputs JSON files under each subject's `landmarks/` folder.

#### Step 3: MTF Image Generation

```bash
python main.py process
```

Converts landmark JSON and rPPG JSON files into Markov Transition Field (MTF) images. Outputs PNG files under each subject's `mtf_images/` folder.

> **Note**: If you already have pre-generated MTF images, you can skip Steps 2 and 3.

### 2.4 Known Dataset Discrepancies

> [!WARNING]
> The official UBFC-Phys dataset has the following issues:
> - **`s40`**: The file `vid_s40_T3.avi` is **missing**. This is an upstream issue with the dataset.
> - **`s56`**: The file `selfReportedAnx_s56.csv` is **missing**. Additionally, `s55` and `s56` have identical self-report data. Since self-report files are not used in this pipeline, this is ignored.

---

## 3. Repository Structure

```
CCT-LSTM_reproduce/
├── main.py                     # Data processing pipeline orchestrator
├── train_cct.py                # Stage 1: CCT pre-training (single modality)
├── train_cct_lstm.py           # Stage 2: CCT-LSTM for T1/T2/T3 classification (7-fold CV)
├── train_cct_lstm_levels.py    # Stage 2: CCT-LSTM for T1/T3-ctrl/T3-test (5-fold stratified CV)
├── verify_landmarks_vs_rppg.py # Utility to verify landmark/rPPG alignment
│
├── model/
│   └── model.py                # CCTForPreTraining & CCT_LSTM_Model definitions
│
├── preprocessing/
│   ├── dataset.py              # PyTorch Datasets for Stage 1 & Stage 2
│   ├── file_path_gen.py        # Dynamic path generation utility
│   ├── integrity_and_masterManifest.py  # Dataset integrity checker
│   ├── landmark_extractor.py   # MediaPipe-based landmark extraction
│   ├── pca_and_mtf.py          # PCA dimensionality reduction & MTF generation
│   └── rppg_extractor.py       # rPPG signal extraction (reference implementation)
│
├── scripts/
│   ├── run_final_report.sh              # One-shot pipeline runner
│   └── dataset_visualization/           # EDA/HRV exploratory analysis tools
│       ├── data_mining_common.py        # Shared utilities and constants
│       ├── eda_mining.py                # Feature extraction (EDA tonic, HR/HRV)
│       ├── eda_visual_subject_line.py   # Per-subject signal visualization
│       └── eda_visual_task_scatter.py   # Cross-subject task comparison
│
├── utils/
│   └── summarize_reports.py    # Report summarization utility
│
├── weights/                    # Best model checkpoints per fold (git-ignored)
├── reports/                    # Confusion matrices & classification reports (git-ignored)
├── logs/                       # Training logs (git-ignored)
├── wandb/                      # WandB local run data (git-ignored)
│
├── environment.yml             # Conda environment specification
├── UBFC_data_path.txt          # Auto-generated dataset path config (git-ignored)
├── THIRD_PARTY_LICENSES.md     # Third-party dependency licenses
├── LICENSE                     # MIT License
└── README.md                   # This file
```

---

## 4. Stage 1: CCT Pre-training

Train the CCT (Compact Convolutional Transformer) backbone on single-modality MTF images with 7-fold cross-validation.

```bash
# Train landmark branch (replace 'landmark' with 'rppg' for rPPG branch)
python train_cct.py \
  --modality landmark \
  --epochs 100 \
  --lr 5e-6 \
  --wd 2e-3 \
  --dropout 0.3 \
  --emb-dropout 0.3 \
  --early-stop-patience 60 \
  --batch-size 16 \
  --device cuda \
  --use-cutout \
  --use-noise \
  --use-wandb \
  --wandb-project cct_pretraining
```

**Outputs**: `weights/cct_{modality}_fold_{k}_best.pth` (k = 1 to 7)

> For all available options: `python train_cct.py --help`

---

## 5. Stage 2: CCT-LSTM Fine-tuning

Fine-tune the full CCT-LSTM model on multimodal (landmark + rPPG) sequences. Stage 1 weights are automatically loaded.

### 5.1 Task Classification (T1/T2/T3) — 7-fold CV

```bash
python train_cct_lstm.py \
  --epochs 100 \
  --lr 5e-5 \
  --lstm-hidden-size 512 \
  --dropout 0.3 \
  --early-stop-patience 30 \
  --report-dir reports/tasks \
  --device cuda \
  --use-wandb
```

**Outputs**: `weights/cctlstm_tasks_fold_{k}_best.pth`, `reports/tasks/fold_{k}/`

### 5.2 Level Classification (T1 / T3-ctrl / T3-test) — 5-fold Stratified CV

```bash
python train_cct_lstm_levels.py \
  --epochs 100 \
  --lr 5e-5 \
  --lstm-hidden-size 512 \
  --dropout 0.3 \
  --early-stop-patience 20 \
  --report-dir reports/levels \
  --device cuda \
  --use-wandb
```

**Outputs**: `weights/cctlstm_levels_fold_{k}_best.pth`, `reports/levels/fold_{k}/`

> For all available options: `python train_cct_lstm.py --help` or `python train_cct_lstm_levels.py --help`

---

## 6. One-shot Pipeline Runner

For convenience, `scripts/run_final_report.sh` chains all stages together:

```bash
bash scripts/run_final_report.sh
```

**What it runs** (in order):
1. Stage 1: CCT pre-training (Landmark)
2. Stage 1: CCT pre-training (rPPG)
3. Stage 2: CCT-LSTM Tasks (T1/T2/T3)
4. Stage 2: CCT-LSTM Levels (T1/T3-ctrl/T3-test)

> **Note**: Review and customize the script before running. You may want to:
> - Comment out Stage 1 if you already have trained weights.
> - Adjust hyperparameters based on your hardware and experimental needs.
> - Modify the conda activation path if your setup differs.

---

## 7. EDA Visualization Scripts (Optional)

Scripts under `scripts/dataset_visualization/` provide exploratory data analysis tools for the UBFC-Phys physiological signals (EDA and BVP/PPG). These are **optional** and not required for model training.

### 7.1 Setup

Create a dataset path file in the scripts directory:

```bash
echo "/path/to/UBFC-Phys/Data" > scripts/dataset_visualization/UBFC_path.txt
```

### 7.2 Available Scripts

| Script | Description |
|--------|-------------|
| `eda_mining.py` | Extract EDA tonic/phasic and HR/HRV features, generate scatter plots and merged CSV |
| `eda_visual_subject_line.py` | Visualize single subject's EDA signals (Raw, Tonic, Phasic) across tasks |
| `eda_visual_task_scatter.py` | Cross-subject task comparison using SCR features |

### 7.3 Usage

```bash
cd scripts/dataset_visualization
python eda_mining.py                  # Feature extraction and correlation plots
python eda_visual_subject_line.py     # Per-subject signal visualization
python eda_visual_task_scatter.py     # Task comparison scatter plots
```

**Outputs**:
- `eda_results/` — Per-subject visualizations
- `data_mining/` — Merged feature CSV and correlation plots

---

## 8. WandB Logging

Enable experiment tracking with Weights & Biases:

```bash
python train_cct.py --use-wandb --wandb-project my_project --wandb-run-name my_run
```

All training scripts support:
- `--use-wandb`: Enable WandB logging.
- `--wandb-project`: Project name in WandB.
- `--wandb-run-name`: Optional run name.

**Current behavior**: All folds are logged within a single WandB run. If you prefer per-fold runs, modify the training scripts to call `wandb.init()` inside the cross-validation loop with distinct run names.

---

## 9. Known Issues & Tips

### 9.1 Class Collapse in Levels Task

The Levels classification task (T1 vs T3-ctrl vs T3-test) is more challenging. If you observe:
- `val_f1` stuck at ~0.22 (random baseline for 3 classes)
- All predictions falling into a single class

**Try**:
- Lower learning rate (e.g., `1e-5` instead of `2e-4`)
- Larger LSTM hidden size (e.g., `512` instead of `256`)
- Reduce regularization (lower dropout, lower weight decay)

### 9.2 Path Configuration

- `UBFC_data_path.txt` must exist in the repo root.
- Generated automatically by `python main.py check`.
- If you move the dataset, re-run the integrity check.

### 9.3 Missing Data

- `s40`: Missing `vid_s40_T3.avi` — This appears to be a naming error within the dataset itself.
- `s56`: Missing self-report file, s55 & s56 are identical. — ignored since self-reports are not used.

### 9.4 GPU Memory

- Stage 2 processes variable-length sequences with batch size = 1.
- If you encounter OOM errors, reduce `--num-workers` or sequence length.

---

## 10. References and Citation

### 10.1 Original Paper

If you use this code, please cite the original paper:

```bibtex
@inproceedings{ziaratnia2024multimodal,
  title={Multimodal Deep Learning for Remote Stress Estimation Using CCT-LSTM},
  author={Ziaratnia, Sayyedjavad and Laohakangvalvit, Tipporn and Sugaya, Midori and Sripian, Peeraya},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  pages={8321--8329},
  year={2024},
  doi={10.1109/WACV57701.2024.00815}
}
```

### 10.2 Dataset

- **UBFC-Phys**: [Download Link](https://search-data.ubfc.fr/FR-18008901306731-2022-05-05_UBFC-Phys-A-Multimodal-Dataset-For.html)

### 10.3 Implementation Notes

**rPPG Extraction**: Although the paper references "Face2PPG" methodology, personal communication with the authors clarified that they utilized the [pyVHR](https://github.com/phuselab/pyVHR) library directly for rPPG extraction, including its OMIT algorithm and ROI handling, without additional modifications. This repository therefore provides a reference implementation based on pyVHR to maintain fidelity to the authors' actual workflow. See `preprocessing/rppg_extractor.py` for details.

---

## 11. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Please cite the original UBFC-Phys dataset and CCT-LSTM paper if you use this code in your work.
