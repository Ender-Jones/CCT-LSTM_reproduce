# CCT-LSTM Reproduce

End-to-end pipeline for reproducing the two-stage CCT-LSTM workflow on the UBFC-Phys dataset:

- **Stage 1**: Single-modality CCT pre-training (landmark-only, rPPG-only).
- **Stage 2**: Multimodal CCT-LSTM fine-tuning for two tasks: (a) T1/T2/T3 classification, (b) T1 vs T3-ctrl vs T3-test.

The repo is structured and tested on Python 3.10 / CUDA with conda. WandB logging is optional.

---

## 1. Environment

```bash
conda env create -f environment.yml
conda activate cctlstm

python - <<'PY'
import sys, torch, torchvision, cv2, mediapipe as mp, numpy as np, PIL
print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("torchvision", torchvision.__version__)
print("cv2", cv2.__version__)
print("mediapipe", mp.__version__)
print("numpy", np.__version__, "pillow", PIL.__version__)
PY
```

---

## 2. Data preparation (UBFC-Phys)

1) **Download** UBFC-Phys and keep the structure:
```
<your_root>/UBFC-Phys/Data/s1/...
```

2) **Integrity check & manifest (mandatory)**  
This will create `UBFC_data_path.txt` in repo root and `master_manifest.csv` under `UBFC-Phys/`.
```bash
python main.py check
```

Known issues in the official dataset:
- `s40`: `vid_s40_T3.avi` missing (upstream issue).
- `s56`: `selfReportedAnx_s56.csv` missing; `s55` & `s56` self-reports identical. We do not use these self-report files.

3) **Optional: landmark extraction & MTF generation**  
If you need to regenerate intermediates:
```bash
python main.py extract   # facial landmarks -> JSON
python main.py process   # landmarks/rPPG JSON -> MTF images
```

Path handling:
- `preprocessing/file_path_gen.py` reads `UBFC_data_path.txt` from repo root by default.  
  You can override with `FilePathGen(config_path=...)` if you keep data elsewhere.

---

## 3. Repository layout

```
train_cct.py                # Stage 1 pre-training (single modality)
train_cct_lstm.py           # Stage 2 tasks (T1/T2/T3), 7-fold CV
train_cct_lstm_levels.py    # Stage 2 levels (T1 / T3-ctrl / T3-test), 5-fold stratified CV
model/model.py              # CCT and CCT-LSTM definitions
preprocessing/dataset.py    # Datasets for Stage1/Stage2
preprocessing/*.py          # Landmark extraction, MTF generation, integrity check, etc.
scripts/run_final_report.sh # Pipeline runner (configurable)
reports/                    # Fold-wise confusion matrices & reports
weights/                    # Best checkpoints per fold
logs/                       # Training logs
wandb/                      # Local WandB run data (ignored by git)
```

---

## 4. Stage 1 — CCT pre-training

Train landmark or rPPG branches separately with 7-fold CV. Outputs:
`weights/cct_{modality}_fold_{k}_best.pth`

Example (landmark):
```bash
python train_cct.py \
  --modality landmark \
  --lr 5e-6 --wd 2e-3 \
  --dropout 0.3 --emb-dropout 0.3 \
  --early-stop-patience 60 \
  --num-workers 12 --device cuda \
  --use-wandb --wandb-project cct_pretraining_landmark \
  --wandb-run-name final_landmark_cutout_noise_lr5e-6
```
Key options: `--use-cutout`, `--use-noise`, `--mixup`, `--data-root` (else read from `UBFC_data_path.txt`).

---

## 5. Stage 2 — CCT-LSTM

Both scripts auto-load Stage 1 weights by fold index:
`weights/cct_landmark_fold_k_best.pth`, `weights/cct_rppg_fold_k_best.pth`.

### 5.1 Tasks (T1/T2/T3), 7-fold
Outputs:
- `weights/cctlstm_tasks_fold_{k}_best.pth`
- `reports/tasks/fold_{k}/confusion_matrix.npy`, `classification_report.txt`

Current tuned run (dropout 0.5, wd 1e-3, hidden 256, lr 2e-5) achieved:
- Avg best val F1 (7-fold): **0.6622**

Run example:
```bash
python train_cct_lstm.py \
  --lr 2e-5 --weight-decay 1e-3 \
  --lstm-hidden-size 256 --lstm-num-layers 2 \
  --dropout 0.5 --emb-dropout 0.5 \
  --early-stop-patience 30 \
  --stage1-weights-dir weights \
  --output-dir weights \
  --report-dir reports/tasks_tuned \
  --device cuda --num-workers 12 \
  --use-wandb --wandb-project cct_lstm_tasks \
  --wandb-run-name final_tasks_reg_dropout05_wd1e3_hid256_lr2e5
```

### 5.2 Levels (T1 / T3-ctrl / T3-test), 5-fold stratified
Outputs:
- `weights/cctlstm_levels_fold_{k}_best.pth`
- `reports/levels*/fold_{k}/confusion_matrix.npy`, `classification_report.txt`

Warning: the latest tuned run (lr 2e-4, hidden 256) collapsed to a single-class prediction (macro-F1 ≈ 0.22, best epoch=1).  
Try more conservative settings (e.g., hidden 512, lr 1e-4 or lower) if you need usable level results.

Run example:
```bash
python train_cct_lstm_levels.py \
  --lr 2e-4 --weight-decay 1e-4 \
  --lstm-hidden-size 256 --lstm-num-layers 2 \
  --dropout 0.3 --emb-dropout 0.3 \
  --early-stop-patience 20 \
  --stage1-weights-dir weights \
  --output-dir weights \
  --report-dir reports/levels_tuned \
  --device cuda --num-workers 12 \
  --use-wandb --wandb-project cct_lstm_level \
  --wandb-run-name final_levels_boost_lr2e4_hid256
```

---

## 6. One-shot runner (configurable)

`scripts/run_final_report.sh` can chain the stages. The current version:
- Runs Stage 1 Landmark (to refresh fold-1 weight), skips Stage 1 rPPG.
- Runs Stage 2 Tasks (tuned).
- Runs Stage 2 Levels (tuned; currently collapses).

Adjust the script before use (e.g., comment out Stage 1 if you already have weights, tweak Stage 2 hyperparams).

---

## 7. WandB logging

Enable with `--use-wandb --wandb-project <name> --wandb-run-name <name>`.  
Current scripts log all folds in a single run; if you prefer per-fold runs, wrap `wandb.init()` inside the CV loop with distinct names/groups.

---

## 8. Known issues / tips

- **Class collapse in Levels**: Precision warnings and flat F1=0.22 indicate all predictions fall into one class. Lower LR and/or larger hidden size (512) can help; monitor per-class support.
- **Dataset quirks**: Missing `s40` T3 video, missing `s56` self-report; ignored by this pipeline.
- **Path config**: `UBFC_data_path.txt` must live in repo root for FilePathGen defaults.
- **Validation collapse**: If Tasks also collapse (macro-F1 ~0.33), reduce LR or regularization.

---

## 9. References

- Pipeline and tensor flow details: see `README_research.md`.
- Dataset: [UBFC-Phys](https://search-data.ubfc.fr/FR-18008901306731-2022-05-05_UBFC-Phys-A-Multimodal-Dataset-For.html).

