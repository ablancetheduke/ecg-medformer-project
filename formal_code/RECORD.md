# Server Experiment Record

> **Instructions:** After each experiment finishes, fill in the table below.
> Find the metrics in the log file or result_classification.txt.
>
> grep "Validation results\|Test results" logs/PTB-XL-Baseline.log

---

## Experiment 1: Medformer Baseline

| Item | Value |
|---|---|
| Start time | |
| End time | |
| GPU | |
| Batch size | |
| Stopped epoch | |
| **Val Loss** | |
| **Val Accuracy** | |
| **Val Precision** | |
| **Val Recall** | |
| **Val F1** | |
| **Val AUROC** | |
| **Val AUPRC** | |
| **Test Loss** | |
| **Test Accuracy** | |
| **Test Precision** | |
| **Test Recall** | |
| **Test F1** | |
| **Test AUROC** | |
| **Test AUPRC** | |
| Notes | |

### Paper comparison

| Metric | Paper (5-class) | Ours |
|---|---|---|
| Accuracy | 72.51% | |
| F1 | 61.72% | |
| AUROC | 88.93% | |
| AUPRC | 66.13% | |

---

## Experiment 2: MedformerFFT Concat

| Item | Value |
|---|---|
| Start time | |
| End time | |
| Stopped epoch | |
| **Val Accuracy** | |
| **Val F1** | |
| **Val AUROC** | |
| **Val AUPRC** | |
| **Test Accuracy** | |
| **Test F1** | |
| **Test AUROC** | |
| **Test AUPRC** | |
| Notes | |

### Comparison vs Baseline

| Metric | Baseline | FFT Concat | Delta |
|---|---|---|---|
| Test Accuracy | | | |
| Test F1 | | | |
| Test AUROC | | | |
| Test AUPRC | | | |

---

## Experiment 3: FrequencyOnly (Ablation)

| Item | Value |
|---|---|
| Start time | |
| End time | |
| **Test Accuracy** | |
| **Test F1** | |
| **Test AUROC** | |
| **Test AUPRC** | |
| Notes | |

---

## Final Summary Table

| Model | Acc | F1 | AUROC | AUPRC |
|---|---|---|---|---|
| Medformer (Baseline) | | | | |
| MedformerFFT (Concat) | | | | |
| FrequencyOnly | | | | |

## Things to send back to Claude Code

1. All log files (logs/*.log)
2. This completed RECORD.md
3. result_classification.txt from each experiment
4. Checkpoint directories (optional, but good for resume)
