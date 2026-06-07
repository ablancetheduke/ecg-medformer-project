#!/bin/bash
# =============================================================================
# Server Timing Test — Estimate per-epoch time on your rented GPU
# =============================================================================
# Run this first to calibrate. 1 epoch only, ~15-30 min on RTX 4090.
# =============================================================================

set -e
source $(conda info --base)/etc/profile.d/conda.sh
conda activate medformer

cd "$(dirname "$0")/Medformer-main"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/timing_${TIMESTAMP}.log"

echo "Starting 1-epoch timing test at $(date)"
echo "Log: $LOG"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"

START_TIME=$(date +%s)

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/PTB-XL/ \
  --model_id PTB-XL-Timing-server \
  --model Medformer \
  --data PTB-XL \
  --e_layers 6 \
  --batch_size 64 \
  --d_model 128 \
  --d_ff 256 \
  --patch_len_list 2,4,8,8,16,16,16,16,32,32,32,32,32,32,32,32 \
  --augmentations none \
  --swa \
  --des TimingServer \
  --itr 1 \
  --learning_rate 0.0001 \
  --train_epochs 1 \
  --patience 1 \
  2>&1 | tee "$LOG"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))

echo ""
echo "============================================"
echo " TIMING TEST COMPLETE"
echo "============================================"
echo "Total time: ${ELAPSED}s (${MINUTES} min)"
echo ""

# Extract and show results
grep -E "Epoch.*cost time|Validation results|Test results" "$LOG"

echo ""
echo "============================================"
echo " TIME ESTIMATE FOR FULL RUNS"
echo "============================================"
echo "Based on this timing test:"
echo "  Per epoch: ~${MINUTES} min"
echo "  100 epochs (max): ~$((MINUTES * 100 / 60)) hours"
echo "  With early stopping (~30 epochs): ~$((MINUTES * 30 / 60)) hours"
echo "  3 experiments (Medformer + MedformerFFT + FrequencyOnly): ~$((MINUTES * 90 / 60)) hours total"
echo ""
echo "All experiments will be run with: bash run_all.sh"
