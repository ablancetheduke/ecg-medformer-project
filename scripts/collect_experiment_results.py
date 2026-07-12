#!/usr/bin/env python3
"""
collect_experiment_results.py
=============================
Read all training logs from results/logs/, parse the metrics, detect seeds,
classify model types, and produce a unified experiment summary table.

Output:
  1. Console: formatted summary table
  2. experiments_summary.csv: machine-readable results
  3. experiments_summary.md: markdown table for paper/report

Usage:
  python scripts/collect_experiment_results.py

Does NOT modify any training code or model files — read-only.
"""

import re
import os
import csv
import json
from pathlib import Path
from collections import defaultdict
from typing import Optional, Dict, List, Tuple

# ── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "results" / "logs"
CHECKPOINT_BACKUP = PROJECT_ROOT / "checkpoints_backup.tar.gz"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_CSV = PROJECT_ROOT / "results" / "experiments_summary.csv"
OUTPUT_MD = PROJECT_ROOT / "results" / "experiments_summary.md"
RESULTS_TXT = PROJECT_ROOT / "results" / "PTB-XL-Baseline.txt"


# ── Model type classification ────────────────────────────────────────
MODEL_TYPE_MAP: Dict[str, str] = {
    "Medformer":              "Baseline",
    "MedformerFFT":           "FFT",
    "FrequencyOnly":          "FFT-only",
    "MedformerFFT_gate":      "Fusion",
    "MedformerFFT_crossattn": "Fusion",
    "MedformerFFT_bilinear":  "Fusion",
    "MedformerWavelet":       "Wavelet",
    "MedformerDCT":           "DCT",
    "WaveletOnly":            "Wavelet-only",
    "DCTOnly":                "DCT-only",
}

# Sort order for model types (determines table order)
MODEL_TYPE_ORDER = {
    "Baseline": 0,
    "FFT": 1,
    "FFT-only": 2,
    "Wavelet": 3,
    "Wavelet-only": 4,
    "DCT": 5,
    "DCT-only": 6,
    "Fusion": 7,
}


# ── Regex patterns ────────────────────────────────────────────────────
RE_ARGS_LINE = re.compile(r"^Args in experiment:\s*$")
RE_NAMESPACE = re.compile(r"Namespace\((.+)\)$")
RE_MODEL_ID = re.compile(r"model_id='([^']+)'")
RE_MODEL = re.compile(r"model='([^']+)'")
RE_DES = re.compile(r"des='([^']+)'")
RE_ITR = re.compile(r"itr=(\d+)")

RE_EPOCH_TRAIN = re.compile(r"^Epoch: (\d+),\s*Steps:.+Train Loss: ([\d.]+)")
RE_EPOCH_TIME = re.compile(r"^Epoch: (\d+) cost time: ([\d.]+)")

# Validation / Test results — note: Recall then space then F1 (no comma for Recall→F1 in test)
RE_VAL = re.compile(
    r"^Validation results --- "
    r"Loss: ([\d.eE+-]+), "
    r"Accuracy: ([\d.]+), "
    r"Precision: ([\d.]+), "
    r"Recall: ([\d.]+), "
    r"F1: ([\d.]+), "
    r"AUROC: ([\d.]+), "
    r"AUPRC: ([\d.]+)"
)
RE_TEST = re.compile(
    r"^Test results --- "
    r"Loss: ([\d.eE+-]+), "
    r"Accuracy: ([\d.]+), "
    r"Precision: ([\d.]+), "
    r"Recall: ([\d.]+) "
    r"F1: ([\d.]+), "
    r"AUROC: ([\d.]+), "
    r"AUPRC: ([\d.]+)"
)
RE_EARLY = re.compile(r"^Early stopping")

# Seed extraction from filename:  PTB-XL-Baseline-s42.log → 42
RE_SEED_FILENAME = re.compile(r"-s(\d+)\.log$", re.IGNORECASE)
# Seed extraction from model_id:  PTB-XL-Baseline-seed42 → 42
RE_SEED_MODELID = re.compile(r"[_-]s(?:eed)?(\d+)", re.IGNORECASE)


# ── Parse a single log file ──────────────────────────────────────────

class ExperimentResult:
    """Hold parsed results for a single experiment."""
    def __init__(self):
        self.log_path: str = ""
        self.model_id: str = ""
        self.model_name: str = ""          # e.g. "Medformer"
        self.model_type: str = ""          # e.g. "Baseline"
        self.seed: Optional[int] = None
        self.des: str = ""
        self.itr: Optional[int] = None
        self.total_epochs: int = 0         # last epoch that completed training
        self.early_stopped: bool = False
        self.early_stop_epoch: Optional[int] = None
        self.is_complete: bool = False     # did training finish (early stop or max epochs)
        self.notes: List[str] = []

        # Per-epoch records: epoch -> {train_loss, cost_time, val_*, test_*}
        self.epochs: Dict[int, dict] = defaultdict(dict)

        # Best-by-validation metrics
        self.best_by: str = ""             # "val_f1" or "val_auroc"
        self.best_epoch: Optional[int] = None
        self.best_val: dict = {}
        self.best_test: dict = {}

        # Final epoch metrics (last available)
        self.final_epoch: Optional[int] = None
        self.final_val: dict = {}
        self.final_test: dict = {}

        # Last evaluation after early-stopping (SWA reload, if present)
        self.post_es_val: dict = {}
        self.post_es_test: dict = {}


def extract_seed(filename: str, model_id: str, itr_val: Optional[int]) -> Optional[int]:
    """Extract seed from filename first, then model_id, then itr arg."""
    m = RE_SEED_FILENAME.search(filename)
    if m:
        return int(m.group(1))
    m = RE_SEED_MODELID.search(model_id)
    if m:
        return int(m.group(1))
    if itr_val is not None:
        # In the original Medformer codebase, itr controls the seed
        # but itr=1 doesn't mean seed=1 — it's an index into a fixed seed list.
        # We use itr as a fallback identifier, not the actual seed value.
        return None  # don't guess
    return None


def parse_log(filepath: Path) -> Optional[ExperimentResult]:
    """Parse one training log. Returns None if unparseable."""
    result = ExperimentResult()
    result.log_path = str(filepath.relative_to(PROJECT_ROOT))

    lines = filepath.read_text(errors="replace").splitlines()
    if not lines:
        result.notes.append("empty file")
        return result

    # ── Parse Args line ──────────────────────────────────────────
    # The first line should be "Args in experiment:" followed by Namespace(...)
    # But in practice line 1 is the "Args in experiment:" header, line 2 is the Namespace

    args_str = None
    for i, line in enumerate(lines):
        if line.strip() == "Args in experiment:" and i + 1 < len(lines):
            m = RE_NAMESPACE.match(lines[i + 1].strip())
            if m:
                args_str = lines[i + 1].strip()
                break
        elif "Namespace(" in line and "model_id" in line:
            m = RE_NAMESPACE.search(line)
            if m:
                args_str = line.strip()
                break

    if args_str is None:
        result.notes.append("cannot parse Args — no Namespace found")
        # Still try to get model from filename
        result.model_name = _guess_model_from_filename(filepath.stem)
        result.model_type = MODEL_TYPE_MAP.get(result.model_name, "Unknown")
        result.seed = extract_seed(filepath.name, "", None)
        return result

    # Extract key fields
    m_id = RE_MODEL_ID.search(args_str)
    if m_id:
        result.model_id = m_id.group(1)
    m_model = RE_MODEL.search(args_str)
    if m_model:
        result.model_name = m_model.group(1)
    m_des = RE_DES.search(args_str)
    if m_des:
        result.des = m_des.group(1)
    m_itr = RE_ITR.search(args_str)
    result.itr = int(m_itr.group(1)) if m_itr else None

    # Classify
    result.model_type = MODEL_TYPE_MAP.get(result.model_name, "Unknown")
    result.seed = extract_seed(filepath.name, result.model_id, result.itr)

    # If no seed from filename/model_id, try to infer from itr
    if result.seed is None and result.itr is not None:
        # In server deployment, itr was set to 1 for all runs;
        # seed was manually changed in the code. So itr alone is not reliable.
        # We flag this as a note.
        result.notes.append(f"seed unknown; itr={result.itr}")

    # ── Parse epoch-by-epoch metrics ─────────────────────────────
    current_epoch: Optional[int] = None
    max_epoch_seen = 0

    for i, line in enumerate(lines):
        # Epoch header: "Epoch: N, Steps: ..."
        m_ep = RE_EPOCH_TRAIN.match(line.strip())
        if m_ep:
            current_epoch = int(m_ep.group(1))
            max_epoch_seen = max(max_epoch_seen, current_epoch)
            result.epochs[current_epoch]["train_loss"] = float(m_ep.group(2))
            continue

        # Epoch cost time: "Epoch: N cost time: ..."
        m_time = RE_EPOCH_TIME.match(line.strip())
        if m_time:
            ep = int(m_time.group(1))
            result.epochs[ep]["cost_time"] = float(m_time.group(2))
            continue

        # Validation
        m_val = RE_VAL.match(line.strip())
        if m_val and current_epoch is not None:
            result.epochs[current_epoch]["val"] = {
                "loss": float(m_val.group(1)),
                "acc": float(m_val.group(2)),
                "prec": float(m_val.group(3)),
                "recall": float(m_val.group(4)),
                "f1": float(m_val.group(5)),
                "auroc": float(m_val.group(6)),
                "auprc": float(m_val.group(7)),
            }
            continue

        # Test
        m_test = RE_TEST.match(line.strip())
        if m_test and current_epoch is not None:
            result.epochs[current_epoch]["test"] = {
                "loss": float(m_test.group(1)),
                "acc": float(m_test.group(2)),
                "prec": float(m_test.group(3)),
                "recall": float(m_test.group(4)),
                "f1": float(m_test.group(5)),
                "auroc": float(m_test.group(6)),
                "auprc": float(m_test.group(7)),
            }
            continue

        # Early stopping
        if RE_EARLY.match(line.strip()):
            result.early_stopped = True
            # record the epoch before this early stop
            if result.early_stop_epoch is None:
                result.early_stop_epoch = max_epoch_seen
            # After early stopping, SWA checkpoints get loaded and evaluated.
            # The next val/test lines are post-ES evaluations.
            continue

    result.total_epochs = max_epoch_seen

    # ── Determine best epoch ─────────────────────────────────────
    # Best by validation F1 (standard choice)
    best_f1_epoch = None
    best_f1_val = -1.0
    best_auroc_epoch = None
    best_auroc_val = -1.0

    for ep, data in result.epochs.items():
        if "val" not in data:
            continue
        vf1 = data["val"]["f1"]
        vauc = data["val"]["auroc"]
        if vf1 > best_f1_val:
            best_f1_val = vf1
            best_f1_epoch = ep
        if vauc > best_auroc_val:
            best_auroc_val = vauc
            best_auroc_epoch = ep

    # Default: best by val F1
    if best_f1_epoch is not None:
        result.best_by = "val_f1"
        result.best_epoch = best_f1_epoch
        result.best_val = result.epochs[best_f1_epoch].get("val", {})
        result.best_test = result.epochs[best_f1_epoch].get("test", {})

    # ── Final epoch ──────────────────────────────────────────────
    if max_epoch_seen > 0 and "val" in result.epochs.get(max_epoch_seen, {}):
        result.final_epoch = max_epoch_seen
        result.final_val = result.epochs[max_epoch_seen].get("val", {})
        result.final_test = result.epochs[max_epoch_seen].get("test", {})

    # ── Post-early-stopping evaluation (SWA) ─────────────────────
    # If early stopped, the val/test lines after the last "Early stopping"
    # are the SWA-reloaded best-checkpoint evaluation.
    if result.early_stopped:
        # Find the LAST val/test pair (post-ES SWA eval)
        last_val_ep = None
        last_test_ep = None
        for ep in sorted(result.epochs.keys(), reverse=True):
            if "val" in result.epochs[ep]:
                last_val_ep = ep
                break
        if last_val_ep and last_val_ep > (result.early_stop_epoch or 0):
            result.post_es_val = result.epochs[last_val_ep].get("val", {})
            result.post_es_test = result.epochs[last_val_ep].get("test", {})
        else:
            result.post_es_val = result.best_val
            result.post_es_test = result.best_test

    # ── Completion status ────────────────────────────────────────
    if result.early_stopped:
        result.is_complete = True
        result.notes.append(f"early stop @ ep {result.early_stop_epoch}")
    elif result.total_epochs >= 100:
        result.is_complete = True
        result.notes.append("ran full 100 epochs")
    elif result.total_epochs >= 10:
        result.is_complete = False
        result.notes.append(f"interrupted @ ep {result.total_epochs}")
    else:
        result.is_complete = False
        result.notes.append(f"incomplete ({result.total_epochs} epochs)")

    return result


def _guess_model_from_filename(stem: str) -> str:
    """Rough guess of model name from log filename."""
    # Map known filename patterns to model names
    KNOWN = {
        "PTB-XL-Baseline": "Medformer",
        "PTB-XL-FFT": "MedformerFFT",
        "PTB-XL-FrequencyOnly": "FrequencyOnly",
        "PTB-XL-Gate": "MedformerFFT_gate",
        "PTB-XL-CrossAttn": "MedformerFFT_crossattn",
        "PTB-XL-CrossAttnV2": "MedformerFFT_crossattn",
        "PTB-XL-Bilinear": "MedformerFFT_bilinear",
        "PTB-XL-Wavelet": "MedformerWavelet",
        "PTB-XL-DCT": "MedformerDCT",
        "PTB-XL-WaveletOnly": "WaveletOnly",
        "PTB-XL-DCTOnly": "DCTOnly",
    }
    # Strip seed suffix
    base = re.sub(r"-s\d+$", "", stem)
    return KNOWN.get(base, "Unknown")


# ── Checkpoint path inference ────────────────────────────────────────

def infer_checkpoint_path(result: ExperimentResult) -> str:
    """Infer the expected checkpoint path from model_id + des."""
    # Standard Medformer checkpoint naming pattern:
    # checkpoints/{setting}/checkpoint.pth
    # where setting = classification_{model_id}_{model}_{data}_{...}_{des}_seed{XX}
    # Since we don't know the exact seed used in the setting string,
    # we list the expected directory pattern.
    setting_pattern = f"classification_{result.model_id}_{result.model_name}_PTB-XL_"
    dirs = []
    if CHECKPOINTS_DIR.exists():
        for d in CHECKPOINTS_DIR.iterdir():
            if d.is_dir() and setting_pattern in d.name:
                ckpt = d / "checkpoint.pth"
                if ckpt.exists():
                    dirs.append(str(d.relative_to(PROJECT_ROOT)))
    if dirs:
        return dirs[0]
    # Fallback: show expected pattern
    seed_str = f"_seed{result.seed}" if result.seed else "_seed??"
    return (f"checkpoints/classification_{result.model_id}_"
            f"{result.model_name}_PTB-XL_ftM_sl96_*_{result.des}{seed_str}/")


# ── Formatters ────────────────────────────────────────────────────────

def pct(x: float) -> str:
    """Format as percentage string."""
    return f"{x * 100:.2f}%"


def fmt_metric(d: dict, key: str) -> str:
    """Safely format a metric from a dict."""
    if not d or key not in d:
        return "—"
    return pct(d[key])


def fmt_val_test(result: ExperimentResult) -> Tuple[str, str, str, str, str, str]:
    """Return (val_f1, val_auroc, test_acc, test_f1, test_auroc, test_auprc)."""
    v = result.best_val
    t = result.best_test
    return (
        fmt_metric(v, "f1"),
        fmt_metric(v, "auroc"),
        fmt_metric(t, "acc"),
        fmt_metric(t, "f1"),
        fmt_metric(t, "auroc"),
        fmt_metric(t, "auprc"),
    )


# ── Main ──────────────────────────────────────────────────────────────

def main():
    log_files = sorted(LOGS_DIR.glob("*.log"))
    if not log_files:
        print(f"[ERROR] No log files found in {LOGS_DIR}")
        return

    print(f"[INFO] Found {len(log_files)} log files")
    print(f"[INFO] Parsing...")

    results: List[ExperimentResult] = []
    unparseable: List[str] = []

    for lf in log_files:
        r = parse_log(lf)
        if r is None:
            unparseable.append(lf.name)
        else:
            results.append(r)

    if unparseable:
        print(f"[WARN] {len(unparseable)} logs could not be parsed:")
        for u in unparseable:
            print(f"       - {u}")

    print(f"[INFO] Successfully parsed {len(results)} experiments\n")

    # ── Sort: model type order → seed → model_id ─────────────────
    results.sort(key=lambda r: (
        MODEL_TYPE_ORDER.get(r.model_type, 99),
        r.seed if r.seed is not None else 999,
        r.model_id,
    ))

    # ── Console table ────────────────────────────────────────────
    header = (
        f"{'Experiment':<28} {'Type':<14} {'Seed':>5} {'Eps':>4} "
        f"{'V-F1':>8} {'V-AUC':>8} {'T-Acc':>8} {'T-F1':>8} {'T-AUC':>8} {'T-AUPRC':>8} "
        f"{'Done':>5} {'Log File':<38} {'Notes'}"
    )
    sep = "─" * len(header)

    print(sep)
    print(header)
    print(sep)

    for r in results:
        seed_str = str(r.seed) if r.seed is not None else "?"
        done = "[Y]" if r.is_complete else "[N]"
        vf1, vauc, tacc, tf1, tauc, tauprc = fmt_val_test(r)

        exp_name = r.model_id if r.model_id else r.log_path
        log_name = Path(r.log_path).name

        line = (
            f"{exp_name:<28} {r.model_type:<14} {seed_str:>5} {r.best_epoch or r.total_epochs:>4} "
            f"{vf1:>8} {vauc:>8} {tacc:>8} {tf1:>8} {tauc:>8} {tauprc:>8} "
            f"{done:>5} {log_name:<38} "
            f"{'; '.join(r.notes) if r.notes else ''}"
        )
        print(line)
    print(sep)

    # ── CSV output ───────────────────────────────────────────────
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "experiment_name", "model_name", "model_type", "seed",
            "total_epochs", "best_epoch", "early_stopped", "is_complete",
            "val_loss", "val_acc", "val_prec", "val_recall", "val_f1", "val_auroc", "val_auprc",
            "test_loss", "test_acc", "test_prec", "test_recall", "test_f1", "test_auroc", "test_auprc",
            "log_path", "checkpoint_path", "notes"
        ])
        for r in results:
            v = r.best_val
            t = r.best_test
            ckpt = infer_checkpoint_path(r)
            writer.writerow([
                r.model_id or Path(r.log_path).stem,
                r.model_name,
                r.model_type,
                r.seed if r.seed is not None else "",
                r.total_epochs,
                r.best_epoch or "",
                str(r.early_stopped),
                str(r.is_complete),
                v.get("loss", ""), v.get("acc", ""), v.get("prec", ""), v.get("recall", ""),
                v.get("f1", ""), v.get("auroc", ""), v.get("auprc", ""),
                t.get("loss", ""), t.get("acc", ""), t.get("prec", ""), t.get("recall", ""),
                t.get("f1", ""), t.get("auroc", ""), t.get("auprc", ""),
                r.log_path,
                ckpt,
                "; ".join(r.notes),
            ])

    print(f"\n[INFO] CSV written to: {OUTPUT_CSV}")

    # ── Markdown table ───────────────────────────────────────────
    md_lines = []
    md_lines.append("# Experiment Results Summary\n")
    md_lines.append(f"> Auto-generated by `scripts/collect_experiment_results.py`\n")
    md_lines.append(f"> Total experiments: {len(results)}  |  Complete: "
                    f"{sum(1 for r in results if r.is_complete)}  |  "
                    f"Incomplete: {sum(1 for r in results if not r.is_complete)}\n")
    md_lines.append("")

    # Full table
    md_lines.append("## Full Experiment Table\n")
    md_lines.append(
        "| Experiment | Type | Seed | Epochs | Val F1 | Val AUROC | "
        "Test Acc | Test F1 | Test AUROC | Test AUPRC | Done | Log | Notes |"
    )
    md_lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )

    for r in results:
        seed_str = str(r.seed) if r.seed is not None else "?"
        done = "[Y]" if r.is_complete else "[N]"
        vf1, vauc, tacc, tf1, tauc, tauprc = fmt_val_test(r)
        log_name = Path(r.log_path).name
        notes = "; ".join(r.notes) if r.notes else ""
        md_lines.append(
            f"| {r.model_id or '?'} | {r.model_type} | {seed_str} | "
            f"{r.best_epoch or r.total_epochs} | {vf1} | {vauc} | "
            f"{tacc} | {tf1} | {tauc} | {tauprc} | {done} | "
            f"`{log_name}` | {notes} |"
        )

    md_lines.append("")

    # ── Model type summary tables ────────────────────────────────
    # Group by model_type
    groups = defaultdict(list)
    for r in results:
        groups[r.model_type].append(r)

    md_lines.append("## Summary by Model Type\n")

    for mtype in sorted(groups.keys(), key=lambda t: MODEL_TYPE_ORDER.get(t, 99)):
        group = groups[mtype]
        md_lines.append(f"### {mtype}\n")
        md_lines.append(
            "| Experiment | Seed | Epochs | Val F1 | Val AUROC | "
            "Test Acc | Test F1 | Test AUROC | Test AUPRC | Done |"
        )
        md_lines.append(
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
        )
        for r in sorted(group, key=lambda x: (x.seed or 999, x.model_id)):
            seed_str = str(r.seed) if r.seed is not None else "?"
            done = "[Y]" if r.is_complete else "[N]"
            vf1, vauc, tacc, tf1, tauc, tauprc = fmt_val_test(r)
            md_lines.append(
                f"| {r.model_id} | {seed_str} | {r.best_epoch or r.total_epochs} | "
                f"{vf1} | {vauc} | {tacc} | {tf1} | {tauc} | {tauprc} | {done} |"
            )
        md_lines.append("")

    # ── Multi-seed summary ────────────────────────────────────────
    md_lines.append("## Multi-Seed Aggregation (mean±std)\n")
    multi_seed = defaultdict(list)
    for r in results:
        if r.seed is not None and r.is_complete:
            key = (r.model_name, r.model_type)
            multi_seed[key].append(r)

    md_lines.append(
        "| Model | Seeds | Test AUROC (mean±std) | Test F1 (mean±std) | "
        "Test Acc (mean±std) | Test AUPRC (mean±std) |"
    )
    md_lines.append(
        "|---|---:|---:|---:|---:|"
    )

    for (mname, mtype), group in sorted(multi_seed.items(),
                                         key=lambda x: MODEL_TYPE_ORDER.get(x[0][1], 99)):
        if len(group) < 2:
            continue  # skip single-seed
        seeds = sorted([r.seed for r in group if r.seed is not None])
        seed_str = ",".join(str(s) for s in seeds)

        def _mean_std(key: str) -> str:
            vals = [r.best_test.get(key, 0) * 100 for r in group if r.best_test]
            if len(vals) < 2:
                return f"{vals[0]:.2f}%" if vals else "—"
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
            std = var ** 0.5
            return f"{mean:.2f}±{std:.2f}%"

        md_lines.append(
            f"| {mname} ({mtype}) | {seed_str} | "
            f"{_mean_std('auroc')} | {_mean_std('f1')} | "
            f"{_mean_std('acc')} | {_mean_std('auprc')} |"
        )
    md_lines.append("")

    # ── Incomplete experiments ────────────────────────────────────
    incomplete = [r for r in results if not r.is_complete]
    if incomplete:
        md_lines.append("## ⚠️ Incomplete Experiments\n")
        for r in incomplete:
            md_lines.append(
                f"- **{r.model_id or Path(r.log_path).stem}** (seed={r.seed}): "
                f"stopped at epoch {r.total_epochs}. "
                f"{'; '.join(r.notes)}"
            )
        md_lines.append("")

    # ── Checkpoint availability ───────────────────────────────────
    md_lines.append("## Checkpoint Availability\n")
    if CHECKPOINTS_DIR.exists() and any(CHECKPOINTS_DIR.iterdir()):
        for d in sorted(CHECKPOINTS_DIR.iterdir()):
            if d.is_dir():
                ckpt = d / "checkpoint.pth"
                status = "[OK] exists" if ckpt.exists() else "[MISSING]"
                md_lines.append(f"- `{d.name}/` — {status}")
    else:
        md_lines.append(
            f"- ⚠️ `checkpoints/` directory is empty. "
            f"All checkpoints are in `checkpoints_backup.tar.gz` "
            f"({CHECKPOINT_BACKUP.stat().st_size / 1e9:.1f} GB). "
            f"Extract with: `tar -xzf checkpoints_backup.tar.gz`"
        )
    md_lines.append("")

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"[INFO] Markdown written to: {OUTPUT_MD}")
    print(f"[INFO] Done.")


if __name__ == "__main__":
    main()
