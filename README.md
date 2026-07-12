# Medformer ECG 时频双分支分类实验

基于 Medformer (NeurIPS 2024) 在 PTB-XL 心电图数据集上的五分类复现与改进实验。核心思路：在 Medformer 时域建模基础上引入 FFT/DCT/Wavelet 频域分支，探索时频融合是否提升 ECG 分类性能。

- 项目仓库：[ablancetheduke/ecg-medformer-project](https://github.com/ablancetheduke/ecg-medformer-project)
- 主实验 Notebook：`MedformerFFT_PTBXL_Complete_Experiment.executed.ipynb`

---

## 主要结论

| 模型 | 种子数 | Test Acc | Test F1 | Test AUROC | Test AUPRC |
|---|---|---|---|---|---|
| **Medformer (Baseline)** | 3 | 73.04% ± 0.20 | 61.75% ± 0.13 | 89.46% ± 0.18 | 65.94% ± 0.22 |
| **MedformerFFT (Concat)** | 3 | **73.20%** ± 0.11 | **61.88%** ± 0.05 | **89.53%** ± 0.19 | **66.27%** ± 0.44 |
| FrequencyOnly (消融) | 1 | 65.46% | 51.89% | 84.53% | 56.46% |

**FFT 频域分支带来一致但微弱提升**（+0.16% Acc, +0.13% F1, +0.33% AUPRC），纯频域模型大幅落后，说明时域建模对 ECG 分类不可替代。

多种融合策略对比（单次运行最佳结果）：

| 融合策略 | Test Acc | Test F1 | Test AUROC |
|---|---|---|---|
| Gate Fusion | **73.40%** | 61.70% | **89.57%** |
| Baseline (纯时域) | 73.10% | 61.90% | 89.38% |
| FFT Concat | 73.27% | 61.82% | 89.52% |
| Wavelet | 73.33% | 61.72% | 89.28% |
| Bilinear | 72.84% | 60.83% | 89.42% |
| DCT | 72.26% | 61.08% | 89.17% |
| Cross-Attention | 64.90% | 50.02% | 84.12% |

---

## 目录结构

```
深度学习/
├── MedformerFFT_PTBXL_Complete_Experiment.executed.ipynb  ← 主实验 Notebook
├── 期末大作业.docx / 期末大作业.pdf    ← 期末报告
├── README.md                         ← 本文件
├── MANIFEST.json                     ← 文件清单
│
├── formal_code/                      ← 训练代码与部署脚本
│   ├── Medformer-main/               # 模型代码（完整版）
│   │   ├── models/                   # Medformer、MedformerFFT、MedformerDCT 等
│   │   ├── data_provider/            # 数据加载器 (PTBXLLoader)
│   │   ├── exp/                      # 训练/评估引擎
│   │   ├── layers/                   # Attention、Embedding 等
│   │   ├── data_preprocessing/       # PTB-XL 等数据集预处理
│   │   ├── scripts/classification/   # 各模型训练脚本
│   │   ├── run.py / meta-run.py      # 训练入口
│   │   └── figs/                     # 模型架构图
│   ├── run_all.sh / run_single.sh    # 训练调度脚本
│   ├── run_queue.sh / run_queue_v2.sh
│   ├── run_seeds.sh / timing_test.sh
│   ├── setup.sh / pack_for_server.bat
│   ├── README_FIRST.txt              # 服务器部署说明
│   ├── RECORD.md                     # 实验记录模板
│   └── EXPERIMENT_README.md          # 仓库说明（英文）
│
├── data/ptb_xl/raw_partial/          ← PTB-XL 元数据 + 10 条 ECG 样例
│
├── results/                          ← 训练日志 + 实验汇总 + 鲁棒性结果
├── outputs/notebook/                 ← Notebook 预测数组和中间结果
├── notebooks/notebook_figures/       ← 可视化图片
└── scripts/                          ← 工具脚本
```

---

## 模型变体

| 模型文件 | 类型 | 说明 |
|---|---|---|
| `Medformer.py` | Baseline | 原始 Medformer（纯时域） |
| `MedformerFFT.py` | 时频融合 | Medformer + FFT 频域分支 (Concat) |
| `MedformerFFT_bilinear.py` | 融合变体 | Medformer + FFT 双线性融合 |
| `MedformerFFT_crossattn.py` | 融合变体 | Medformer + FFT 交叉注意力融合 |
| `MedformerFFT_gate.py` | 融合变体 | Medformer + FFT 门控融合 |
| `MedformerDCT.py` | 频域变体 | Medformer + DCT 分支 |
| `MedformerWavelet.py` | 频域变体 | Medformer + Wavelet 分支 |
| `FrequencyOnly.py` | 消融实验 | 纯 FFT 频谱 + MLP（无时域） |
| `DCTOnly.py` | 消融实验 | 纯 DCT + MLP |
| `WaveletOnly.py` | 消融实验 | 纯 Wavelet + MLP |

---

## 环境与运行

```bash
# 安装依赖
pip install -r formal_code/Medformer-main/requirements.txt

# 启动 Notebook
jupyter notebook MedformerFFT_PTBXL_Complete_Experiment.executed.ipynb
```

完整训练（需准备 PTB-XL 预处理后的 `.npy` 数据）：

```bash
cd formal_code

# 单个实验
bash run_single.sh MedformerFFT 64

# 多实验队列
bash run_queue_v2.sh

# 多种子
bash run_seeds.sh
```

---

## 数据说明

- `data/ptb_xl/raw_partial/` 包含 PTB-XL 公开数据集的 10 条低采样率 12 导联记录（`.hea` + `.dat`）及 `ptbxl_database.csv` 元数据，可用于数据探索和格式验证
- 完整训练数据需从 [PhysioNet](https://physionet.org/content/ptb-xl/) 下载并预处理，放置在 `formal_code/Medformer-main/dataset/PTB-XL/` 下

## 模型权重

主实验 checkpoint 约 1.5 GB，已单独归档为 `checkpoints_backup.tar.gz`，计划作为 [GitHub Release](https://github.com/ablancetheduke/ecg-medformer-project/releases) 附件发布。权重未就绪时，Notebook 中的数据探索、日志解析、曲线绘制和已保存预测结果分析仍可运行。

## 特殊说明

本项目的完整实验采用 Python 脚本在 RTX4090服务器用时一周完成。由于主实验包含 Baseline、MedformerFFT 及多个消融模型，并对主模型使用随机种子 42、43、44 进行重复训练，单次训练耗时较长，因此训练任务、日志记录和 checkpoint 保存均通过项目中的命令行训练入口执行。

本 Notebook 依据同一项目代码整理而成，用于集中展示实验流程与结果，包括数据读取与预处理、模型结构、训练配置、评估方法、训练曲线、测试集结果和消融分析。Notebook 中展示的指标、图像、预测数组和结果表均来自项目实际运行过程中保存的日志、checkpoint 或评估输出，不使用人工构造的实验结果。当前 Notebook 可直接完成数据样例检查、日志解析、结果可视化及已保存预测结果的分析；在配置完整 PTB-XL 预处理数据和 checkpoint 后，也可调用项目评估流程重新加载模型并在测试集上计算指标。完整训练可通过项目训练脚本在具备相应 GPU 环境的设备上执行。

本人主页以及本文件夹也放置相应的实验代码。最后论文准备word以及pdf，其中word版本由pdf转换而来，按照发表论文格式书写，后附有实验报告，故均为图片内容，望老师理解。
