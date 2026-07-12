#!/bin/bash
# =============================================================================
# Medformer PTB-XL Full Reproduction — Run All Experiments
# =============================================================================
# Usage: bash run_all.sh
#
# This script runs ALL experiments in sequence:
#   1. Medformer baseline (100 epochs, patience=10)
#   2. MedformerFFT concat (100 epochs, patience=10)
#   3. FrequencyOnly (100 epochs, patience=10)
#
# Total estimated time on RTX 4090: ~15-30 hours
# Use screen/tmux so it survives SSH disconnect.
#
# BEFORE RUNNING:
#   tmux new -s exp
#   bash run_all.sh
#   Ctrl+b, d  (to detach)
#   tmux attach -t exp  (to reattach)
# =============================================================================

set -e
source $(conda info --base)/etc/profile.d/conda.sh
conda activate medformer

cd "$(dirname "$0")/Medformer-main"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="results/server_run_${TIMESTAMP}"
mkdir -p "$RESULT_DIR" logs

echo "============================================"
echo " MEDFORMER PTB-XL FULL REPRODUCTION"
echo " Start time: $(date)"
echo " Result dir: $RESULT_DIR"
echo " GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "============================================"

# ============================================================================
# Experiment 1: Medformer Baseline
# ============================================================================
echo ""
echo "##############################################"
echo " #1 Medformer Baseline — Full PTB-XL"
echo "##############################################"
echo ""

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/PTB-XL/ \
  --model_id PTB-XL-Baseline \
  --model Medformer \
  --data PTB-XL \
  --e_layers 6 \
  --batch_size 64 \
  --d_model 128 \
  --d_ff 256 \
  --patch_len_list 2,4,8,8,16,16,16,16,32,32,32,32,32,32,32,32 \
  --augmentations none \
  --swa \
  --des ServerBaseline \
  --itr 1 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10 \
  2>&1 | tee "logs/PTB-XL-Baseline.log"

echo ""
echo ">>> Experiment 1 (Baseline) done at $(date)"

# Copy results
cp -r results/classification/PTB-XL-Baseline "$RESULT_DIR/" 2>/dev/null || true
cp -r checkpoints/classification/PTB-XL-Baseline "$RESULT_DIR/checkpoints_Baseline" 2>/dev/null || true

# Show key metrics
echo ""
echo "--- Medformer Baseline Results ---"
grep -E "Validation results|Test results" "logs/PTB-XL-Baseline.log" | tail -2

# ============================================================================
# Experiment 2: MedformerFFT Concat
# ============================================================================
echo ""
echo "##############################################"
echo " #2 MedformerFFT — Time-Frequency Concat"
echo "##############################################"
echo ""

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/PTB-XL/ \
  --model_id PTB-XL-FFT-Concat \
  --model MedformerFFT \
  --data PTB-XL \
  --e_layers 6 \
  --batch_size 64 \
  --d_model 128 \
  --d_ff 256 \
  --patch_len_list 2,4,8,8,16,16,16,16,32,32,32,32,32,32,32,32 \
  --augmentations none \
  --swa \
  --des ServerFFTConcat \
  --itr 1 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10 \
  2>&1 | tee "logs/PTB-XL-FFT-Concat.log"

echo ""
echo ">>> Experiment 2 (FFT Concat) done at $(date)"

cp -r results/classification/PTB-XL-FFT-Concat "$RESULT_DIR/" 2>/dev/null || true
cp -r checkpoints/classification/PTB-XL-FFT-Concat "$RESULT_DIR/checkpoints_FFTConcat" 2>/dev/null || true

echo ""
echo "--- MedformerFFT Concat Results ---"
grep -E "Validation results|Test results" "logs/PTB-XL-FFT-Concat.log" | tail -2

# ============================================================================
# Experiment 3: FrequencyOnly (Ablation Control)
# ============================================================================
echo ""
echo "##############################################"
echo " #3 FrequencyOnly — Ablation Control"
echo "##############################################"
echo ""

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/PTB-XL/ \
  --model_id PTB-XL-FrequencyOnly \
  --model FrequencyOnly \
  --data PTB-XL \
  --e_layers 6 \
  --batch_size 64 \
  --d_model 128 \
  --d_ff 256 \
  --patch_len_list 2,4,8,8,16,16,16,16,32,32,32,32,32,32,32,32 \
  --augmentations none \
  --swa \
  --des ServerFreqOnly \
  --itr 1 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10 \
  2>&1 | tee "logs/PTB-XL-FrequencyOnly.log"

echo ""
echo ">>> Experiment 3 (FrequencyOnly) done at $(date)"

cp -r results/classification/PTB-XL-FrequencyOnly "$RESULT_DIR/" 2>/dev/null || true
cp -r checkpoints/classification/PTB-XL-FrequencyOnly "$RESULT_DIR/checkpoints_FreqOnly" 2>/dev/null || true

echo ""
echo "--- FrequencyOnly Results ---"
grep -E "Validation results|Test results" "logs/PTB-XL-FrequencyOnly.log" | tail -2

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================"
echo " ALL EXPERIMENTS COMPLETE"
echo " End time: $(date)"
echo "============================================"
echo ""
echo "=== FINAL RESULTS SUMMARY ==="
echo ""
echo "--- Medformer Baseline ---"
grep -E "Validation results|Test results" "logs/PTB-XL-Baseline.log" | tail -2
echo ""
echo "--- MedformerFFT Concat ---"
grep -E "Validation results|Test results" "logs/PTB-XL-FFT-Concat.log" | tail -2
echo ""
echo "--- FrequencyOnly ---"
grep -E "Validation results|Test results" "logs/PTB-XL-FrequencyOnly.log" | tail -2
echo ""
echo "============================================"
echo "Results saved to: $RESULT_DIR/"
echo "All logs saved to: logs/"
echo ""
echo "TO DOWNLOAD RESULTS TO YOUR LOCAL MACHINE:"
echo "  On your Windows machine, run:"
echo "  scp -r user@server_ip:~/server_deploy/Medformer-main/results/ ./results/"
echo "  scp -r user@server_ip:~/server_deploy/Medformer-main/logs/ ./logs/"
echo "  scp -r user@server_ip:~/server_deploy/Medformer-main/checkpoints/ ./checkpoints/"
echo "============================================"
