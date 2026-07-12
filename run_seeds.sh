#!/bin/bash
# ============================================================
# 正确版多seed队列 — 每个实验用不同 --itr
# itr=2 → seed42, itr=3 → seed43
# ============================================================
set -e
cd "$(dirname "$0")/Medformer-main"

BASE="--task_name classification --is_training 1 \
  --root_path ./dataset/PTB-XL/ --data PTB-XL \
  --e_layers 6 --batch_size 64 --d_model 128 --d_ff 256 \
  --patch_len_list 2,4,8,8,16,16,16,16,32,32,32,32,32,32,32,32 \
  --augmentations none --swa \
  --learning_rate 0.0001 --train_epochs 100 --patience 10"

echo "===== 多Seed正确版 开始: $(date) ====="

# ── Baseline ──
echo "[1/4] Baseline s42 (--itr 2)..."
python -u run.py $BASE --model Medformer --itr 2 --model_id PTB-XL-Baseline-s42 --des BaselineS42 2>&1 | tee ../logs/PTB-XL-Baseline-s42.log
echo "✅ [1/4] done @ $(date)"

echo "[2/4] Baseline s43 (--itr 3)..."
python -u run.py $BASE --model Medformer --itr 3 --model_id PTB-XL-Baseline-s43 --des BaselineS43 2>&1 | tee ../logs/PTB-XL-Baseline-s43.log
echo "✅ [2/4] done @ $(date)"

# ── FFT-Concat ──
echo "[3/4] FFT s42 (--itr 2)..."
python -u run.py $BASE --model MedformerFFT --itr 2 --model_id PTB-XL-FFT-s42 --des FFTS42 2>&1 | tee ../logs/PTB-XL-FFT-s42.log
echo "✅ [3/4] done @ $(date)"

echo "[4/4] FFT s43 (--itr 3)..."
python -u run.py $BASE --model MedformerFFT --itr 3 --model_id PTB-XL-FFT-s43 --des FFTS43 2>&1 | tee ../logs/PTB-XL-FFT-s43.log
echo "✅ [4/4] done @ $(date)"

echo "===== 🎉 全完成: $(date) ====="
