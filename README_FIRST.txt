================================================================================
  Medformer PTB-XL 复现实验 — 服务器部署包
  论文: Medformer (NeurIPS 2024)
  目标: 在 PTB-XL 全量数据上复现 Medformer baseline 并测试 FFT 改进
================================================================================

📦 你需要上传到服务器的文件
================================================================================

把以下内容打包上传到你租的服务器的同一目录下 (建议放在 ~/medformer_exp/)：

  server_deploy/
  ├── setup.sh              ← 环境配置（先跑这个）
  ├── timing_test.sh         ← 测速脚本（可选，确认 GPU 速度）
  ├── run_all.sh             ← 一键跑全部 3 个实验（推荐）
  ├── run_single.sh          ← 逐个实验手动执行
  ├── README_FIRST.txt       ← 本文件
  ├── RECORD.md              ← 实验记录模板（填空即可）
  ├── PTB-XL.zip             ← 你本地的 PTB-XL 处理后数据（必须！）
  └── Medformer-main/        ← 你的 Medformer 代码目录（必须！）

================================================================================
🚀 操作步骤（按顺序执行）
================================================================================

第1步: 登录服务器，上传文件
--------------------------------------------------------------------------------
  用你习惯的方式（scp/rsync/WinSCP/FileZilla）把整个 server_deploy/ 文件夹
  传到服务器上。

  如果是自己打包上传：
    在你的 Windows 机器上:
      cd E:\deeplearning\ecg_time_frequency_dual_branch_classification
      tar -czf medformer_deploy.tar.gz server_deploy/
    然后 scp medformer_deploy.tar.gz 到服务器
    在服务器上:
      tar -xzf medformer_deploy.tar.gz
      cd server_deploy

第2步: 安装环境（只跑一次）
--------------------------------------------------------------------------------
  bash setup.sh

  这一步会: 创建 conda 环境 → 装 PyTorch → 装依赖 → 验证 GPU → 解压数据
  预计 10-15 分钟。

第3步: （可选）测速
--------------------------------------------------------------------------------
  bash timing_test.sh

  跑 1 个 epoch 估算服务器 GPU 速度。RTX 4090 大约 15-20 分钟。
  跑完之后你会看到预估的总时间。

第4步: 启动实验（用 tmux 防止 SSH 断开导致中断）
--------------------------------------------------------------------------------
  tmux new -s exp
  bash run_all.sh

  按 Ctrl+b 然后 d 来 detach（退出 tmux 但程序继续跑）
  重新连接: tmux attach -t exp
  查看 tmux 列表: tmux ls

  如果不想用 tmux（不推荐）:
  bash run_all.sh

第5步: 下载结果
--------------------------------------------------------------------------------
  实验跑完后，在你的 Windows 机器上运行:
    scp -r your_user@server_ip:~/server_deploy/Medformer-main/results/ ./results/
    scp -r your_user@server_ip:~/server_deploy/Medformer-main/logs/ ./logs/

  然后把结果发给我（Claude Code），我来分析。

================================================================================
📊 这个包会跑哪些实验
================================================================================

  顺序执行 3 个实验:

  实验1: Medformer baseline
    模型: Medformer
    轮数: 100 epochs, patience=10（验证 F1 不涨就早停）
    目的: 拿到 PTB-XL 上的 baseline 指标

  实验2: MedformerFFT concat
    模型: MedformerFFT（时域 Medformer + FFT 频域分支 + concat 融合）
    轮数: 同上
    目的: 证明 FFT 频域分支有没有效果

  实验3: FrequencyOnly
    模型: FrequencyOnly（纯 FFT 频谱 + MLP）
    轮数: 同上
    目的: 消融实验——看纯频域信息本身有多少分类能力

================================================================================
⏱️ 时间预估
================================================================================

  基于本地 RTX 4060 实测: 1 epoch ≈ 60 分钟, batch_size=32

  假设租 RTX 4090 (约 3-4x 速度快于 4060):
    1 epoch ≈ 15-20 分钟
    每个实验 (early stop ~30 epochs) ≈ 7.5-10 小时
    3 个实验总计 ≈ 22-30 小时

  假设租 A5000/A6000:
    更快，可能 15-20 小时全部完成

================================================================================
📝 结果记录
================================================================================

  打开 RECORD.md，按模板填写每个实验的结果。
  主要记录: Val/Test 的 Accuracy, F1, AUROC, AUPRC

  论文参考值 (PTB-XL 5-class):
    Medformer: Accuracy 72.51%, F1 61.72%, AUROC 88.93%, AUPRC 66.13%

  我们的本地 1-epoch 测试值（供对比）:
    1 epoch:   Accuracy 70.89%, F1 57.64%, AUROC 87.79%, AUPRC 63.89%

================================================================================
⚠️ 常见问题
================================================================================

Q: OOM (CUDA out of memory) 怎么办？
A: 把 batch_size 改小。编辑 run_all.sh，把 --batch_size 64 改成 32 或 16。
   或者单跑: bash run_single.sh Medformer 32

Q: 想先跑一个看看效果？
A: bash run_single.sh Medformer 64
   只跑 baseline 这一个实验。

Q: 想只用部分数据快速测试？
A: 我们已经有 PTB-XL-Partial 数据集（200条），但在全量数据上没必要。

Q: 想用论文的官方设置（augmentation, 5 seeds）？
A: 编辑 run_all.sh，加上 --augmentations jitter0.2,scale0.2,drop0.5 --itr 5
   时间会变成原来的 5 倍！仅当时间充裕时做。

================================================================================
📁 目录结构（跑完之后）
================================================================================

  server_deploy/
  ├── Medformer-main/
  │   ├── logs/                   ← 所有训练日志
  │   │   ├── PTB-XL-Baseline.log
  │   │   ├── PTB-XL-FFT-Concat.log
  │   │   └── PTB-XL-FrequencyOnly.log
  │   ├── results/
  │   │   ├── classification/
  │   │   │   ├── PTB-XL-Baseline/
  │   │   │   ├── PTB-XL-FFT-Concat/
  │   │   │   └── PTB-XL-FrequencyOnly/
  │   │   └── server_run_YYYYMMDD_HHMMSS/  ← 汇总备份
  │   └── checkpoints/
  └── RECORD.md                  ← 你自己填的实验记录
================================================================================
