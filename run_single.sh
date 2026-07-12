#!/bin/bash
# =============================================================================
# Single Experiment Runner — for manual one-by-one execution
# =============================================================================
# Usage: bash run_single.sh <model_name> [batch_size]
#   model_name: Medformer | MedformerFFT | FrequencyOnly
#   batch_size: 64 (default), 128, 256, etc.
#
# Examples:
#   bash run_single.sh Medformer
#   bash run_single.sh MedformerFFT 128
# =============================================================================

set -e
source $(conda info --base)/etc/profile.d/conda.sh
conda activate medformer

MODEL=${1:-Medformer}
BATCH_SIZE=${2:-64}
cd "$(dirname "$0")/Medformer-main"

echo "Running: model=$MODEL, batch_size=$BATCH_SIZE"
echo "Start: $(date)"

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/PTB-XL/ \
  --model_id "PTB-XL-${MODEL}-bs${BATCH_SIZE}" \
  --model "$MODEL" \
  --data PTB-XL \
  --e_layers 6 \
  --batch_size "$BATCH_SIZE" \
  --d_model 128 \
  --d_ff 256 \
  --patch_len_list 2,4,8,8,16,16,16,16,32,32,32,32,32,32,32,32 \
  --augmentations none \
  --swa \
  --des "Server${MODEL}" \
  --itr 1 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10 \
  2>&1 | tee "logs/PTB-XL-${MODEL}-bs${BATCH_SIZE}.log"

echo ""
echo "Done: $(date)"
echo "--- Results ---"
grep -E "Validation results|Test results" "logs/PTB-XL-${MODEL}-bs${BATCH_SIZE}.log" | tail -2
