#!/bin/bash
# ============================================================
# 自动训练队列 — 跑完一个自动开始下一个
# 用法: tmux new -s queue bash run_queue.sh
#       tmux attach -t queue   (查看进度)
#       Ctrl+B D               (退出但不中断)
# ============================================================
set -e

ROOT_DIR="$(cd "$(dirname "$0")/Medformer-main" && pwd)"
LOG_DIR="$(cd "$(dirname "$0")" && pwd)/logs"
cd "$ROOT_DIR"

# 公共参数
COMMON_ARGS="--task_name classification --is_training 1 \
  --root_path ./dataset/PTB-XL/ --data PTB-XL \
  --e_layers 6 --batch_size 64 --d_model 128 --d_ff 256 \
  --patch_len_list 2,4,8,8,16,16,16,16,32,32,32,32,32,32,32,32 \
  --augmentations none --swa --itr 1 \
  --learning_rate 0.0001 --train_epochs 100 --patience 10"

# ============================================================
# 队列: 按顺序跑，跑完一个自动下一个
# 注释掉不需要的实验行即可
# ============================================================

echo "=========================================="
echo " 队列开始: $(date)"
echo "=========================================="

# --- 第1组: 核心实验补seed ---
# Baseline seed=44 (补第3个seed)
echo "[1/6] Baseline s44 开始..."
python -u run.py $COMMON_ARGS \
  --model Medformer --model_id PTB-XL-Baseline-s44 \
  --des BaselineS44 2>&1 | tee "$LOG_DIR/PTB-XL-Baseline-s44.log"
echo "✅ [1/6] Baseline s44 完成 @ $(date)"

# FFT-Concat seed=44 (补第3个seed)
echo "[2/6] FFT s44 开始..."
python -u run.py $COMMON_ARGS \
  --model MedformerFFT --model_id PTB-XL-FFT-s44 \
  --des FFTS44 2>&1 | tee "$LOG_DIR/PTB-XL-FFT-s44.log"
echo "✅ [2/6] FFT s44 完成 @ $(date)"

# --- 第2组: 融合实验补seed ---
# Gate seed=42
echo "[3/6] Gate s42 开始..."
python -u run.py $COMMON_ARGS \
  --model MedformerFFT_gate --model_id PTB-XL-Gate-s42 \
  --des GateS42 2>&1 | tee "$LOG_DIR/PTB-XL-Gate-s42.log"
echo "✅ [3/6] Gate s42 完成 @ $(date)"

# Bilinear seed=42
echo "[4/6] Bilinear s42 开始..."
python -u run.py $COMMON_ARGS \
  --model MedformerFFT_bilinear --model_id PTB-XL-Bilinear-s42 \
  --des BilinearS42 2>&1 | tee "$LOG_DIR/PTB-XL-Bilinear-s42.log"
echo "✅ [4/6] Bilinear s42 完成 @ $(date)"

# CrossAttnV2 seed=42
echo "[5/6] CrossAttnV2 s42 开始..."
python -u run.py $COMMON_ARGS \
  --model MedformerFFT_crossattn --model_id PTB-XL-CrossAttnV2-s42 \
  --des CrossAttnS42 2>&1 | tee "$LOG_DIR/PTB-XL-CrossAttnV2-s42.log"
echo "✅ [5/6] CrossAttnV2 s42 完成 @ $(date)"

# --- 第3组: 频域实验补seed ---
# Wavelet seed=42
echo "[6/6] Wavelet s42 开始..."
python -u run.py $COMMON_ARGS \
  --model MedformerWavelet --model_id PTB-XL-Wavelet-s42 \
  --des WaveletS42 2>&1 | tee "$LOG_DIR/PTB-XL-Wavelet-s42.log"
echo "✅ [6/6] Wavelet s42 完成 @ $(date)"

# DCT seed=42
# echo "[7/7] DCT s42 开始..."
# python -u run.py $COMMON_ARGS \
#   --model MedformerDCT --model_id PTB-XL-DCT-s42 \
#   --des DCTS42 2>&1 | tee "$LOG_DIR/PTB-XL-DCT-s42.log"
# echo "✅ [7/7] DCT s42 完成 @ $(date)"

echo "=========================================="
echo " 🎉 全部完成! $(date)"
echo "=========================================="

# 自动关机 (确认要关机再取消下面这行注释)
# shutdown -h now
