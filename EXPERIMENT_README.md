# Medformer ECG Time-Frequency Dual-Branch Experiment

## 1. Project Overview

This repository contains source code for ECG classification experiments based on Medformer.

The project first reproduces the Medformer baseline on the PTB-XL ECG dataset, then adds frequency-domain branches such as FFT, DCT, and wavelet variants to explore time-frequency fusion.

The main goal is to test whether frequency-domain information can improve ECG time-series classification performance.

---

## 2. Main Research Question

The original Medformer mainly models medical time-series signals in the time domain.

This project asks:

Can frequency-domain modeling, especially FFT-based representation, improve Medformer on ECG classification tasks?

---

## 3. Dataset

Dataset:

- PTB-XL ECG dataset

Expected dataset path:

```text
Medformer-main/dataset/PTB-XL/
The dataset is not included in this GitHub repository because of size and license restrictions.

4. Included Files

This repository includes:

Original Medformer code
Modified FFT / DCT / Wavelet model variants
Frequency-only ablation models
Time-frequency fusion variants
Training scripts
Queue scripts
Setup scripts
Experiment records

Important model files:

Medformer-main/models/Medformer.py
Medformer-main/models/MedformerFFT.py
Medformer-main/models/MedformerDCT.py
Medformer-main/models/MedformerWavelet.py
Medformer-main/models/FrequencyOnly.py
Medformer-main/models/DCTOnly.py
Medformer-main/models/WaveletOnly.py
Medformer-main/models/MedformerFFT_bilinear.py
Medformer-main/models/MedformerFFT_crossattn.py
Medformer-main/models/MedformerFFT_gate.py

Important scripts:

run_all.sh
run_single.sh
run_queue.sh
run_queue_v2.sh
run_seeds.sh
timing_test.sh
setup.sh
Medformer-main/run.py
5. Excluded Files

The following are intentionally excluded from GitHub:

Medformer-main/dataset/
Medformer-main/checkpoints/
Medformer-main/logs/
Medformer-main/results/
logs/
*.pth
*.pt
*.ckpt
*.tar.gz
*.zip
*.npy
*.npz
*.csv
*.pkl
*.h5
*.mat
*.dat
*.hea

Reasons:

Dataset files are large and may have license restrictions.
Checkpoints are trained model outputs.
Logs and result files can become large.
GitHub should mainly store source code, scripts, and documentation.
6. About Model Weights

Model weights are generated after training.

They may differ between runs because of:

Random initialization
Random seeds
Data shuffling
CUDA nondeterminism
GPU differences
Library version differences
Early stopping

Therefore, weights should be treated as experiment outputs, not source code.

Recommended separate backup command:

tar -czf checkpoints_backup.tar.gz Medformer-main/checkpoints

Do not commit checkpoint files directly to GitHub.

7. Environment

Install dependencies:

cd Medformer-main
pip install -r requirements.txt

Some experiments may also require:

natsort
reformer-pytorch
patool
sktime
8. Typical Usage

After cloning this repository on a new server:

git clone https://github.com/ablancetheduke/ecg-medformer-project.git
cd ecg-medformer-project

Then place the PTB-XL dataset under:

Medformer-main/dataset/PTB-XL/

Run experiments, for example:

bash run_queue_v2.sh

or:

bash run_seeds.sh
9. Experiment Groups

Main experiment groups:

Medformer baseline
Medformer + FFT fusion
Frequency-only model
DCT variant
Wavelet variant
Bilinear fusion variant
Cross-attention fusion variant
Gated fusion variant

The purpose is to compare:

Time-domain modeling
Frequency-domain modeling
Time-frequency fusion
Different fusion strategies
10. Reproducibility Notes

For each experiment, record:

model_name
random_seed
training_command
batch_size
learning_rate
epochs
patience
best_epoch
validation_metric
test_metric
checkpoint_path
log_path
GPU_type
CUDA_version
PyTorch_version

Large outputs should be backed up separately, not committed to GitHub.

11. Backup Strategy

GitHub stores:

Source code
Scripts
Documentation
Experiment notes

Separate local/cloud backup stores:

PTB-XL dataset
Checkpoints
Logs
Result tables
Compressed experiment outputs

Suggested backup commands:

tar -czf experiment_logs_and_results.tar.gz logs Medformer-main/logs Medformer-main/results RECORD.md README_FIRST.txt
tar -czf checkpoints_backup.tar.gz Medformer-main/checkpoints
12. Project Status

The source code has been backed up to GitHub.

Further work:

Finish baseline training
Finish FFT fusion experiments
Run ablation studies
Record metrics
Compare all models
Prepare tables and figures for the final report or paper
