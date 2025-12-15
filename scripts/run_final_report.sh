#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status.

# Ensure conda environment is active
source ~/Applications/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate cctlstm

mkdir -p logs
mkdir -p weights
mkdir -p reports

echo "============================================================"
echo "🚀 Starting FINAL PIPELINE for Report Generation"
echo "============================================================"

# 1. Stage 1: CCT Pre-training (Landmark) - Best Params: Cutout + Noise
echo "------------------------------------------------------------"
echo "▶️  Stage 1: Landmark Pre-training (Best Params: Cutout + Noise)"
echo "------------------------------------------------------------"
python -u train_cct.py \
  --modality landmark \
  --epochs 100 \
  --lr 5e-6 \
  --wd 2e-3 \
  --device cuda \
  --num_workers 12 \
  --seed -1 \
  --dropout 0.3 \
  --emb-dropout 0.3 \
  --early-stop-patience 60 \
  --use-cutout \
  --cutout-n-holes 2 \
  --cutout-length 32 \
  --use-noise \
  --noise-std 0.02 \
  --use-wandb \
  --wandb-project cct_pretraining_landmark \
  --wandb-run-name final_landmark_cutout_noise_lr5e-6 \
  2>&1 | tee logs/final_stage1_landmark.log

# 2. Stage 1: CCT Pre-training (rPPG) - Using same robust strategy
echo "------------------------------------------------------------"
echo "▶️  Stage 1: rPPG Pre-training (Transferring Best Strategy)"
echo "------------------------------------------------------------"
python -u train_cct.py \
  --modality rppg \
  --epochs 100 \
  --lr 5e-6 \
  --wd 2e-3 \
  --device cuda \
  --num_workers 20 \
  --seed -1 \
  --dropout 0.3 \
  --emb-dropout 0.3 \
  --early-stop-patience 60 \
  --use-cutout \
  --cutout-n-holes 2 \
  --cutout-length 32 \
  --use-noise \
  --noise-std 0.02 \
  --use-wandb \
  --wandb-project cct_pretraining_rPPG \
  --wandb-run-name final_rppg_cutout_noise_lr5e-6 \
  2>&1 | tee logs/final_stage1_rppg.log

# 3. Stage 2: CCT-LSTM Tasks (T1/T2/T3) - Fixing Overfitting
echo "------------------------------------------------------------"
echo "▶️  Stage 2: Tasks (T1/T2/T3) - Tuned for Regularization"
echo "------------------------------------------------------------"
# Changes:
# 1. dropout / emb-dropout: 0.3 -> 0.5 (Stronger regularization)
# 2. weight-decay: 1e-4 -> 1e-3 (Stronger L2 penalty)
# 3. lstm-hidden-size: 512 -> 256 (Reduce model capacity)
# 4. lr: 5e-5 -> 2e-5 (Slower, more stable updates)
python -u train_cct_lstm.py \
  --epochs 100 \
  --lr 2e-5 \
  --weight-decay 1e-3 \
  --lstm-hidden-size 256 \
  --lstm-num-layers 2 \
  --device cuda \
  --num-workers 12 \
  --seed -1 \
  --dropout 0.5 \
  --emb-dropout 0.5 \
  --early-stop-patience 30 \
  --stage1-weights-dir weights \
  --output-dir weights \
  --report-dir reports/tasks_tuned \
  --use-wandb \
  --wandb-project cct_lstm_tasks \
  --wandb-run-name final_tasks_reg_dropout05_wd1e3_hid256_lr2e5 \
  2>&1 | tee logs/final_stage2_tasks_tuned.log

# 4. Stage 2: CCT-LSTM Levels (T1/Ctrl/Test) - Fixing Underfitting/Stagnation
echo "------------------------------------------------------------"
echo "▶️  Stage 2: Levels (T1/Ctrl/Test) - Tuned for Convergence"
echo "------------------------------------------------------------"
# Changes:
# 1. lr: 5e-5 -> 2e-4 (Increase LR to escape plateaus)
# 2. weight-decay: 1e-4 (Keep low)
# 3. lstm-hidden-size: 512 -> 256 (Reduce complexity to help convergence)
# 4. dropout: 0.3 (Keep moderate)
python -u train_cct_lstm_levels.py \
  --epochs 100 \
  --lr 2e-4 \
  --weight-decay 1e-4 \
  --lstm-hidden-size 256 \
  --lstm-num-layers 2 \
  --device cuda \
  --num-workers 12 \
  --seed -1 \
  --dropout 0.3 \
  --emb-dropout 0.3 \
  --early-stop-patience 20 \
  --stage1-weights-dir weights \
  --output-dir weights \
  --report-dir reports/levels_tuned \
  --use-wandb \
  --wandb-project cct_lstm_level \
  --wandb-run-name final_levels_boost_lr2e4_hid256 \
  2>&1 | tee logs/final_stage2_levels_tuned.log

echo "============================================================"
echo "✅ FINAL PIPELINE COMPLETED SUCCESSFULLY"
echo "============================================================"
