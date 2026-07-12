#!/bin/bash
# =============================================================================
# Medformer PTB-XL Reproduction — Server Setup Script
# =============================================================================
# Usage: bash setup.sh
# This script sets up the conda environment and verifies everything.
# Run this ONCE after logging into your server.
# =============================================================================

set -e

echo "============================================"
echo " Step 1: Create conda environment"
echo "============================================"
conda create -n medformer python=3.8 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate medformer

echo ""
echo "============================================"
echo " Step 2: Install PyTorch with CUDA"
echo "============================================"
pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "============================================"
echo " Step 3: Install project dependencies"
echo "============================================"
pip install numpy pandas scikit-learn natsort sktime==0.16.1 reformer-pytorch==1.4.4 patool

echo ""
echo "============================================"
echo " Step 4: Verify GPU and PyTorch"
echo "============================================"
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'  Memory: {torch.cuda.get_device_properties(i).total_mem / 1024**3:.1f} GB')
"

echo ""
echo "============================================"
echo " Step 5: Extract PTB-XL dataset"
echo "============================================"
cd Medformer-main
mkdir -p dataset
if [ -f "../PTB-XL.zip" ]; then
    echo "Extracting PTB-XL.zip from parent directory..."
    unzip -o ../PTB-XL.zip -d dataset/
else
    echo "WARNING: PTB-XL.zip not found at ../PTB-XL.zip"
    echo "Please place PTB-XL.zip in the server_deploy/ directory and re-run."
    exit 1
fi

echo ""
echo "============================================"
echo " Step 6: Verify dataset"
echo "============================================"
python -c "
import numpy as np
from pathlib import Path
root = Path('dataset/PTB-XL')
y = np.load(root / 'Label' / 'label.npy')
features = sorted((root / 'Feature').glob('*.npy'))
x = np.load(features[0])
print(f'Label shape: {y.shape}')
print(f'Feature file count: {len(features)}')
print(f'Sample feature shape: {x.shape}')
print(f'First 5 labels: {y[:5].tolist()}')
# Verify expected values
assert y.shape == (17596, 2), f'Label shape mismatch: {y.shape} != (17596, 2)'
assert x.shape == (20, 250, 12), f'Feature shape mismatch: {x.shape} != (20, 250, 12)'
print('Dataset VERIFIED OK ✓')
"

echo ""
echo "============================================"
echo " Step 7: Create log directory"
echo "============================================"
mkdir -p logs

echo ""
echo "============================================"
echo " SETUP COMPLETE!"
echo "============================================"
echo ""
echo "Next: Run the timing test to estimate server speed:"
echo "  bash timing_test.sh"
echo ""
echo "Or run all experiments directly:"
echo "  bash run_all.sh"
