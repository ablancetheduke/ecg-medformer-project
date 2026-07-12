# MedformerFFT on PTB-XL

MedformerFFT 是一个面向 PTB-XL 五分类 ECG 任务的时频双分支实验项目：以 Medformer 建模时域波形，并通过轻量级 FFT 分支编码幅度谱，随后进行特征拼接分类。

- 项目仓库：[ablancetheduke/ecg-medformer-project](https://github.com/ablancetheduke/ecg-medformer-project)
- Notebook：`MedformerFFT_PTBXL_Complete_Experiment.executed.ipynb`

本目录包含 Notebook、核心 Python 源码、PTB-XL 原始样例、训练日志、实验汇总、预测数组和可视化结果。

## 目录说明

```
深度学习/
├── MedformerFFT_PTBXL_Complete_Experiment.executed.ipynb  # 实验流程与结果分析
├── Medformer/Medformer-main/       # 训练入口、数据加载、模型、训练与评估代码
├── data/ptb_xl/raw_partial/        # PTB-XL 元数据与 10 条 12 导联原始样例
├── results/                        # 全部训练日志、实验汇总和鲁棒性结果
├── outputs/notebook/               # Notebook 使用的预测概率、标签与预测类别
├── notebooks/notebook_figures/     # 训练曲线、ROC、频谱等已有图片
└── scripts/                        # 结果汇总和鲁棒性分析脚本
```

## 代码入口

- `Medformer/Medformer-main/run.py`：训练与测试入口。
- `Medformer/Medformer-main/data_provider/data_loader.py`：`PTBXLLoader` 和标准化预处理。
- `Medformer/Medformer-main/data_provider/data_factory.py`：数据集与 DataLoader 注册。
- `Medformer/Medformer-main/models/MedformerFFT.py`：时频双分支模型。
- `Medformer/Medformer-main/exp/exp_classification.py`：训练、验证、测试、早停与指标计算。
- `scripts/collect_experiment_results.py`：解析日志并生成实验汇总。
- `scripts/evaluate_robustness.py`：鲁棒性评估。

## 数据子集

`data/ptb_xl/raw_partial/.../records100/00000/` 含 00001--00010 的 `.hea` 和 `.dat` 文件，每条为 PTB-XL 低采样率 12 导联记录；`ptbxl_database.csv` 为对应的原始元数据。该子集可用于检查数据格式、读取波形和运行 Notebook 的数据探索部分。

完整训练使用预处理后的 `Feature/*.npy` 与 `Label/label.npy`，应放到 `Medformer/Medformer-main/dataset/PTB-XL/`。本整理包不包含完整训练数据。

## 环境与运行

```powershell
cd 深度学习
python -m pip install -r Medformer/Medformer-main/requirements.txt
jupyter notebook MedformerFFT_PTBXL_Complete_Experiment.executed.ipynb
```

完整训练命令见 Notebook 与 `Medformer/Medformer-main/run.py`。测试集预测数组、训练日志和结果汇总保存在 `outputs/notebook/` 与 `results/`。

## 模型权重与 Release

为避免将大文件写入 Git 历史，训练数据与模型权重不随常规仓库提交。主实验 checkpoint 已单独归档为 `checkpoints_backup.tar.gz`（约 1.5 GB），计划作为 GitHub Release 附件发布：

<https://github.com/ablancetheduke/ecg-medformer-project/releases>

发布后，下载并解压该归档，将其中的 checkpoint 目录恢复到：

```text
Medformer/Medformer-main/checkpoints/
```

随后可通过 `Exp_Classification.test(..., test=1)` 加载验证集 Macro F1 最优权重并运行测试评估。权重未就绪时，Notebook 中的数据探索、日志解析、曲线绘制和已保存预测结果分析仍可运行。
