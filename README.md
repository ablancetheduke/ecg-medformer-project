# Medformer ECG Time-Frequency Dual-Branch Classification

基于 Medformer (NeurIPS 2024) 在 PTB-XL 心电图数据集上的五分类实验项目。包含 Medformer 基线复现及 FFT/DCT/Wavelet 多种时频双分支融合变体。

> GitHub 仓库仅包含**源代码与脚本**，训练数据、checkpoint、日志等大文件不在此仓库中。

---

## 目录说明

```
ecg-medformer-project/
│
├── Medformer-main/                   ← 核心代码目录
│   ├── models/                       # 模型定义（PyTorch）
│   │   ├── Medformer.py              # 基线模型（纯时域）
│   │   ├── MedformerFFT.py           # 时域 + FFT 幅值分支 (Concat 融合)
│   │   ├── MedformerFFT_bilinear.py  # 双线性融合变体
│   │   ├── MedformerFFT_crossattn.py # 交叉注意力融合变体
│   │   ├── MedformerFFT_gate.py      # 门控融合变体
│   │   ├── MedformerDCT.py           # 时域 + DCT 频域分支
│   │   ├── MedformerWavelet.py       # 时域 + Wavelet 频域分支
│   │   ├── FrequencyOnly.py          # 消融：纯 FFT 频谱 + MLP（无时域）
│   │   ├── DCTOnly.py                # 消融：纯 DCT + MLP
│   │   ├── WaveletOnly.py            # 消融：纯 Wavelet + MLP
│   │   └── (Autoformer, Informer, PatchTST 等对比模型)
│   │
│   ├── data_provider/                # 数据加载与预处理
│   │   ├── data_loader.py            # PTBXLLoader：PTB-XL 数据读取、标准化、划分
│   │   └── data_factory.py           # 数据集注册与 DataLoader 创建
│   │
│   ├── exp/                          # 实验引擎
│   │   └── exp_classification.py     # 训练循环、验证、测试、早停、SWA、指标计算
│   │
│   ├── layers/                       # 神经网络层
│   │   ├── Embed.py                  # 多粒度分片嵌入 (Multi-granularity Patching)
│   │   ├── Medformer_EncDec.py       # Medformer 编码器
│   │   ├── SelfAttention_Family.py   # 自注意力变体
│   │   └── ...
│   │
│   ├── utils/                        # 工具函数
│   │   ├── metrics.py                # 分类指标（Acc, F1, AUROC, AUPRC）
│   │   ├── losses.py                 # 损失函数
│   │   └── tools.py                  # 通用工具
│   │
│   ├── data_preprocessing/           # 数据集预处理 Notebook
│   │   ├── PTB-XL_preprocessing.ipynb
│   │   ├── PTB_preprocessing.ipynb
│   │   └── ...
│   │
│   ├── scripts/classification/       # 各模型训练 Shell 脚本
│   ├── experiments.ipynb             # 交互式实验 Notebook
│   ├── supplement_experiments.ipynb  # 补充实验
│   ├── run.py                        # 命令行训练入口
│   ├── meta-run.py                   # 批量实验调度
│   ├── requirements.txt              # Python 依赖
│   └── figs/                         # 模型架构图
│
├── run_all.sh                        # 一键运行全部实验（Baseline → FFT → FrequencyOnly）
├── run_single.sh                     # 逐个实验手动执行
├── run_queue.sh / run_queue_v2.sh    # 实验队列调度
├── run_seeds.sh                      # 多种子重复实验
├── timing_test.sh                    # GPU 测速脚本（预估训练时长）
├── setup.sh                          # 服务器环境一键配置（conda + PyTorch + 数据解压）
├── pack_for_server.bat               # Windows 打包上传脚本
│
├── README_FIRST.txt                  # 服务器部署详细说明（上传→配置→运行→下载）
├── RECORD.md                         # 实验记录模板（填写指标用）
├── EXPERIMENT_README.md              # 英文项目说明
├── .gitignore                        # Git 忽略规则（排除数据集/checkpoint/日志/CSV/npy等）
└── LICENSE                           # 许可证
```

---

## 快速开始

```bash
git clone https://github.com/ablancetheduke/ecg-medformer-project.git
cd ecg-medformer-project
pip install -r Medformer-main/requirements.txt
```

### 服务器训练

1. 上传到服务器，先配置环境：
   ```bash
   bash setup.sh
   ```
2. （可选）测速预估时长：
   ```bash
   bash timing_test.sh
   ```
3. 启动实验（推荐 tmux 防止断连中断）：
   ```bash
   tmux new -s exp
   bash run_all.sh
   ```
   详细步骤见 `README_FIRST.txt`。

---

## 数据准备

PTB-XL 数据集从 [PhysioNet](https://physionet.org/content/ptb-xl/) 下载，预处理后放置：

```
Medformer-main/dataset/PTB-XL/
├── Feature/
│   ├── feature_00001.npy
│   └── ...
└── Label/
    └── label.npy
```

---

## 实验结果

| 模型 | Test Acc | Test F1 | Test AUROC | Test AUPRC |
|---|---|---|---|---|
| Medformer (Baseline) | 73.04% | 61.75% | 89.46% | 65.94% |
| MedformerFFT (Concat) | 73.20% | 61.88% | 89.53% | 66.27% |
| FrequencyOnly | 65.46% | 51.89% | 84.53% | 56.46% |

---

## 模型权重

Checkpoint 归档为 `checkpoints_backup.tar.gz`（约 1.5 GB），通过 [GitHub Release](https://github.com/ablancetheduke/ecg-medformer-project/releases) 发布。
