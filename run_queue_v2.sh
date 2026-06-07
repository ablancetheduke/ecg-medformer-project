#!/bin/bash
# 队列2号 — 跳过已完成的(Baseline s44, FFT s44)和垃圾Gate
set -e
cd /root/autodl-tmp/server_deploy/Medformer-main

ARGS="--task_name classification --is_training 1 \
  --root_path ./dataset/PTB-XL/ --data PTB-XL \
  --e_layers 6 --batch_size 64 --d_model 128 --d_ff 256 \
  --patch_len_list 2,4,8,8,16,16,16,16,32,32,32,32,32,32,32,32 \
  --augmentations none --swa --itr 1 \
  --learning_rate 0.0001 --train_epochs 100 --patience 10"

echo "===== 队列 V2 开始: $(date) ====="

# [1] Bilinear s42
echo "[1/3] Bilinear s42..."
python -u run.py $ARGS --model MedformerFFT_bilinear --model_id PTB-XL-Bilinear-s42 --des BiS42 2>&1 | tee ../logs/PTB-XL-Bilinear-s42.log
echo "✅ [1/3] Bilinear s42 done @ $(date)"

# [2] CrossAttnV2 s42
echo "[2/3] CrossAttnV2 s42..."
python -u run.py $ARGS --model MedformerFFT_crossattn --model_id PTB-XL-CrossAttnV2-s42 --des CrS42 2>&1 | tee ../logs/PTB-XL-CrossAttnV2-s42.log
echo "✅ [2/3] CrossAttnV2 s42 done @ $(date)"

# [3] Wavelet s42
echo "[3/3] Wavelet s42..."
python -u run.py $ARGS --model MedformerWavelet --model_id PTB-XL-Wavelet-s42 --des WavS42 2>&1 | tee ../logs/PTB-XL-Wavelet-s42.log
echo "✅ [3/3] Wavelet s42 done @ $(date)"

echo "===== 🎉 全部完成: $(date) ====="
