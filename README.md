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
7. [WandB Logging](#7-wandb-logging)
8. [Known Issues & Tips](#8-known-issues--tips)
9. [References](#9-references)

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
│   └── run_final_report.sh     # One-shot pipeline runner
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
└──  README.md                   # This file
```

---

## 4. Stage 1: CCT Pre-training

Train the CCT (Compact Convolutional Transformer) backbone on single-modality MTF images with 7-fold cross-validation.

### 4.1 Train Landmark Branch

```bash
python train_cct.py \
  --modality landmark \
  --epochs 100 \
  --lr 5e-6 \
  --wd 2e-3 \
  --dropout 0.3 \
  --emb-dropout 0.3 \
  --early-stop-patience 60 \
  --batch-size 16 \
  --num_workers 12 \
  --device cuda \
  --use-cutout \
  --cutout-n-holes 2 \
  --cutout-length 32 \
  --use-noise \
  --noise-std 0.02 \
  --use-wandb \
  --wandb-project cct_pretraining_landmark \
  --wandb-run-name landmark_run_1
```

### 4.2 Train rPPG Branch

```bash
python train_cct.py \
  --modality rppg \
  --epochs 100 \
  --lr 5e-6 \
  --wd 2e-3 \
  --dropout 0.3 \
  --emb-dropout 0.3 \
  --early-stop-patience 60 \
  --batch-size 16 \
  --num_workers 12 \
  --device cuda \
  --use-cutout \
  --cutout-n-holes 2 \
  --cutout-length 32 \
  --use-noise \
  --noise-std 0.02 \
  --use-wandb \
  --wandb-project cct_pretraining_rppg \
  --wandb-run-name rppg_run_1
```

### 4.3 Outputs

- **Weights**: `weights/cct_{modality}_fold_{k}_best.pth` (k = 1 to 7)
- **Logs**: Training progress printed to console and optionally logged to WandB.

### 4.4 Key Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--modality` | (required) | `landmark` or `rppg` |
| `--data-root` | None | Path to dataset root; if omitted, reads from `UBFC_data_path.txt` |
| `--epochs` | 100 | Maximum training epochs per fold |
| `--lr` | 1e-4 | Learning rate |
| `--wd` | 1e-4 | Weight decay |
| `--dropout` | 0.1 | CCT dropout rate |
| `--emb-dropout` | 0.1 | CCT embedding dropout rate |
| `--early-stop-patience` | 30 | Early stopping patience (0 to disable) |
| `--use-cutout` | False | Enable random cutout augmentation |
| `--use-noise` | False | Enable Gaussian noise augmentation |
| `--use-mixup` | False | Enable Mixup augmentation |
| `--use-wandb` | False | Enable WandB logging |

---

## 5. Stage 2: CCT-LSTM Fine-tuning

Fine-tune the full CCT-LSTM model on multimodal (landmark + rPPG) sequences. Stage 1 weights are automatically loaded.

### 5.1 Task Classification (T1/T2/T3) — 7-fold CV

Classifies stress levels across all three experimental tasks.

```bash
python train_cct_lstm.py \
  --epochs 100 \
  --lr 5e-5 \
  --weight-decay 1e-4 \
  --lstm-hidden-size 512 \
  --lstm-num-layers 2 \
  --dropout 0.3 \
  --emb-dropout 0.3 \
  --early-stop-patience 30 \
  --stage1-weights-dir weights \
  --output-dir weights \
  --report-dir reports/tasks \
  --device cuda \
  --num-workers 12 \
  --use-wandb \
  --wandb-project cct_lstm_tasks \
  --wandb-run-name tasks_run_1
```

**Outputs**:
- `weights/cctlstm_tasks_fold_{k}_best.pth`
- `reports/tasks/fold_{k}/confusion_matrix.npy`
- `reports/tasks/fold_{k}/classification_report.txt`

### 5.2 Level Classification (T1 / T3-ctrl / T3-test) — 5-fold Stratified CV

Classifies stress levels distinguishing control vs. test group responses.

```bash
python train_cct_lstm_levels.py \
  --epochs 100 \
  --lr 5e-5 \
  --weight-decay 1e-4 \
  --lstm-hidden-size 512 \
  --lstm-num-layers 2 \
  --dropout 0.3 \
  --emb-dropout 0.3 \
  --early-stop-patience 20 \
  --stage1-weights-dir weights \
  --output-dir weights \
  --report-dir reports/levels \
  --device cuda \
  --num-workers 12 \
  --use-wandb \
  --wandb-project cct_lstm_levels \
  --wandb-run-name levels_run_1
```

**Outputs**:
- `weights/cctlstm_levels_fold_{k}_best.pth`
- `reports/levels/fold_{k}/confusion_matrix.npy`
- `reports/levels/fold_{k}/classification_report.txt`

### 5.3 Key Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--stage1-weights-dir` | `weights` | Directory containing Stage 1 weights |
| `--lstm-hidden-size` | 512 | LSTM hidden dimension |
| `--lstm-num-layers` | 2 | Number of LSTM layers |
| `--freeze-cct` | False | Freeze CCT backbones (train LSTM + classifier only) |

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

## 7. WandB Logging

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

## 8. Known Issues & Tips

### 8.1 Class Collapse in Levels Task

The Levels classification task (T1 vs T3-ctrl vs T3-test) is more challenging. If you observe:
- `val_f1` stuck at ~0.22 (random baseline for 3 classes)
- All predictions falling into a single class

**Try**:
- Lower learning rate (e.g., `1e-5` instead of `2e-4`)
- Larger LSTM hidden size (e.g., `512` instead of `256`)
- Reduce regularization (lower dropout, lower weight decay)

### 8.2 Path Configuration

- `UBFC_data_path.txt` must exist in the repo root.
- Generated automatically by `python main.py check`.
- If you move the dataset, re-run the integrity check.

### 8.3 Missing Data

- `s40`: Missing `vid_s40_T3.avi` — this subject's T3 data will be skipped.
- `s56`: Missing self-report file — ignored since self-reports are not used.

### 8.4 GPU Memory

- Stage 2 processes variable-length sequences with batch size = 1.
- If you encounter OOM errors, reduce `--num-workers` or sequence length.

---

## 9. References

- **Dataset**: [UBFC-Phys](https://search-data.ubfc.fr/FR-18008901306731-2022-05-05_UBFC-Phys-A-Multimodal-Dataset-For.html)
- **Original Paper**: Ziaratnia et al., "Multimodal Deep Learning for Remote Stress Estimation Using CCT-LSTM", 2024.

---

## License

This repository is for research and educational purposes. Please cite the original UBFC-Phys dataset and CCT-LSTM paper if you use this code in your work.
