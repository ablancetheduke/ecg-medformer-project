import argparse
import csv
import os
import random
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


GAUSSIAN_STRENGTHS = [0.00, 0.01, 0.03, 0.05, 0.10]
TIME_MASK_STRENGTHS = [0.00, 0.05, 0.10, 0.20]


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def repo_root():
    return Path(__file__).resolve().parents[1]


def medformer_root(root):
    return root / "Medformer" / "Medformer-main"


def resolve_path(path, root):
    if path is None:
        return None
    path = Path(path)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def checkpoint_candidates(root, model_name, data_name=None, model_id=None):
    checkpoints_dir = medformer_root(root) / "checkpoints" / "classification"
    if not checkpoints_dir.exists():
        return []

    candidates = []
    for checkpoint in checkpoints_dir.rglob("checkpoint.pth"):
        if checkpoint.parent.parent.name != model_name:
            continue
        text = str(checkpoint)
        if data_name and data_name not in text:
            continue
        if model_id and model_id not in text:
            continue
        candidates.append(checkpoint)
    return sorted(candidates)


def choose_checkpoint(root, model_name, explicit_path, data_name, model_id):
    if explicit_path:
        checkpoint = resolve_path(explicit_path, root)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint

    candidates = checkpoint_candidates(root, model_name, data_name=data_name, model_id=model_id)
    if len(candidates) == 1:
        return candidates[0]

    label = model_id or data_name or "all"
    print(f"\nCheckpoint path is not unique for {model_name} ({label}). Candidates:")
    if candidates:
        for path in candidates:
            print(f"  {path}")
    else:
        print("  <none found>")
    print("\nPass an explicit checkpoint path with --baseline_checkpoint or --fft_checkpoint.")
    raise SystemExit(2)


def build_args(cli_args, model_name):
    use_gpu = cli_args.use_gpu and torch.cuda.is_available()
    return SimpleNamespace(
        task_name="classification",
        is_training=0,
        model_id="robustness_eval",
        model=model_name,
        data=cli_args.data,
        root_path=cli_args.root_path,
        data_path="",
        features="M",
        target="OT",
        freq="h",
        seq_len=cli_args.seq_len,
        label_len=cli_args.label_len,
        pred_len=cli_args.pred_len,
        seasonal_patterns="Monthly",
        inverse=False,
        mask_rate=0.25,
        anomaly_ratio=0.25,
        expand=2,
        d_conv=4,
        top_k=5,
        num_kernels=6,
        enc_in=7,
        dec_in=7,
        c_out=7,
        d_model=cli_args.d_model,
        n_heads=cli_args.n_heads,
        e_layers=cli_args.e_layers,
        d_layers=cli_args.d_layers,
        d_ff=cli_args.d_ff,
        moving_avg=25,
        factor=1,
        distil=True,
        dropout=cli_args.dropout,
        embed="timeF",
        activation="gelu",
        output_attention=False,
        no_inter_attn=False,
        chunk_size=16,
        patch_len=16,
        stride=8,
        sampling_rate=256,
        patch_len_list=cli_args.patch_len_list,
        single_channel=False,
        augmentations=cli_args.augmentations,
        num_workers=cli_args.num_workers,
        itr=1,
        train_epochs=0,
        batch_size=cli_args.batch_size,
        patience=0,
        learning_rate=0.0,
        des="robustness_eval",
        loss="CE",
        lradj="type1",
        use_amp=False,
        swa=cli_args.swa,
        use_gpu=use_gpu,
        gpu=cli_args.gpu,
        use_multi_gpu=False,
        devices=str(cli_args.gpu),
        device_ids=[cli_args.gpu],
        p_hidden_dims=[128, 128],
        p_hidden_layers=2,
        seed=cli_args.seed,
    )


def apply_perturbation(batch_x, perturbation, strength, generator):
    if strength == 0:
        return batch_x

    if perturbation == "gaussian_noise":
        noise = torch.randn(
            batch_x.shape,
            generator=generator,
            device=batch_x.device,
            dtype=batch_x.dtype,
        )
        return batch_x + strength * noise

    if perturbation == "time_mask":
        perturbed = batch_x.clone()
        seq_len = perturbed.shape[1]
        mask_len = int(round(seq_len * strength))
        if mask_len <= 0:
            return perturbed
        mask_len = min(mask_len, seq_len)
        max_start = seq_len - mask_len
        starts = torch.randint(
            0,
            max_start + 1,
            (perturbed.shape[0],),
            generator=generator,
            device=perturbed.device,
        )
        for sample_idx, start in enumerate(starts.tolist()):
            perturbed[sample_idx, start : start + mask_len, :] = 0
        return perturbed

    raise ValueError(f"Unknown perturbation: {perturbation}")


def safe_metric(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except ValueError:
        return float("nan")


def evaluate(exp, data_loader, perturbation, strength, seed):
    model = exp.swa_model if exp.swa else exp.model
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    device = exp.device
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    total_loss = 0.0
    total_samples = 0
    preds = []
    trues = []

    with torch.no_grad():
        for batch_x, label, padding_mask in data_loader:
            batch_x = batch_x.float().to(device)
            padding_mask = padding_mask.float().to(device)
            label = label.to(device).long()

            batch_x = apply_perturbation(batch_x, perturbation, strength, generator)
            outputs = model(batch_x, padding_mask, None, None)
            loss = criterion(outputs, label)

            batch_size = label.shape[0]
            total_loss += loss.item()
            total_samples += batch_size
            preds.append(outputs.detach().cpu())
            trues.append(label.detach().cpu())

    preds = torch.cat(preds, dim=0)
    trues = torch.cat(trues, dim=0)
    probs_tensor = F.softmax(preds, dim=1)
    predictions = torch.argmax(probs_tensor, dim=1).numpy()
    probs = probs_tensor.numpy()
    trues_np = trues.flatten().numpy()
    trues_onehot = F.one_hot(
        trues.reshape(-1).to(torch.long),
        num_classes=exp.args.num_class,
    ).float().numpy()

    return {
        "loss": total_loss / max(total_samples, 1),
        "Accuracy": accuracy_score(trues_np, predictions),
        "Precision": precision_score(trues_np, predictions, average="macro", zero_division=0),
        "Recall": recall_score(trues_np, predictions, average="macro", zero_division=0),
        "F1": f1_score(trues_np, predictions, average="macro", zero_division=0),
        "AUROC": safe_metric(roc_auc_score, trues_onehot, probs, multi_class="ovr"),
        "AUPRC": safe_metric(average_precision_score, trues_onehot, probs, average="macro"),
        "num_samples": total_samples,
    }


def load_experiment(cli_args, model_name, checkpoint_path):
    # data_provider imports the M4 forecasting module unconditionally; PTB-XL
    # classification does not use patoolib, but the import must still resolve.
    sys.modules.setdefault("patoolib", types.ModuleType("patoolib"))
    if "sktime.datasets" not in sys.modules:
        sktime_module = types.ModuleType("sktime")
        sktime_datasets_module = types.ModuleType("sktime.datasets")

        def _unused_tsfile_loader(*_args, **_kwargs):
            raise RuntimeError("sktime is only required for UEA .ts datasets, not PTB-XL.")

        sktime_datasets_module.load_from_tsfile_to_dataframe = _unused_tsfile_loader
        sktime_module.datasets = sktime_datasets_module
        sys.modules.setdefault("sktime", sktime_module)
        sys.modules.setdefault("sktime.datasets", sktime_datasets_module)
    if "natsort" not in sys.modules:
        natsort_module = types.ModuleType("natsort")
        natsort_module.natsorted = sorted
        sys.modules.setdefault("natsort", natsort_module)
    if "reformer_pytorch" not in sys.modules:
        reformer_module = types.ModuleType("reformer_pytorch")

        class _UnusedLSHSelfAttention(nn.Module):
            def __init__(self, *_args, **_kwargs):
                super().__init__()
                raise RuntimeError("reformer_pytorch is not required for Medformer robustness evaluation.")

        reformer_module.LSHSelfAttention = _UnusedLSHSelfAttention
        sys.modules.setdefault("reformer_pytorch", reformer_module)
    from exp.exp_classification import Exp_Classification

    args = build_args(cli_args, model_name)
    exp = Exp_Classification(args)
    checkpoint = torch.load(checkpoint_path, map_location=exp.device)
    if exp.swa:
        exp.swa_model.load_state_dict(checkpoint)
    else:
        exp.model.load_state_dict(checkpoint)
    _, test_loader = exp._get_data(flag="TEST")
    return exp, test_loader


def write_results(output_path, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_alias",
        "model",
        "checkpoint",
        "data",
        "root_path",
        "perturbation",
        "strength",
        "loss",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "AUROC",
        "AUPRC",
        "num_samples",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def result_fieldnames():
    return [
        "model_alias",
        "model",
        "checkpoint",
        "data",
        "root_path",
        "perturbation",
        "strength",
        "loss",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "AUROC",
        "AUPRC",
        "num_samples",
    ]


def existing_result_keys(output_path):
    if not output_path.exists():
        return set()
    with output_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {
            (row["model_alias"], row["perturbation"], row["strength"])
            for row in reader
        }


def append_result(output_path, row):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists() and output_path.stat().st_size > 0
    with output_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=result_fieldnames())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Medformer and MedformerFFT robustness on existing checkpoints."
    )
    parser.add_argument("--data", default="PTB-XL-Partial")
    parser.add_argument("--root_path", default="./dataset/PTB-XL-Partial/")
    parser.add_argument("--baseline_checkpoint", default=None)
    parser.add_argument("--fft_checkpoint", default=None)
    parser.add_argument("--baseline_model_id", default=None)
    parser.add_argument("--fft_model_id", default=None)
    parser.add_argument("--output", default="results/robustness_results.csv")
    parser.add_argument(
        "--model_subset",
        choices=["both", "baseline", "fft"],
        default="both",
        help="Evaluate both models or only one model. Useful for resuming long full-dataset runs.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--e_layers", type=int, default=6)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--patch_len_list",
        default="2,4,8,8,16,16,16,16,32,32,32,32,32,32,32,32",
    )
    parser.add_argument("--augmentations", default="none")
    parser.add_argument("--swa", type=str_to_bool, default=False)
    parser.add_argument("--use_gpu", type=str_to_bool, default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--list_checkpoints",
        action="store_true",
        help="List discovered Medformer and MedformerFFT checkpoints, then exit.",
    )
    return parser.parse_args()


def main():
    cli_args = parse_args()
    root = repo_root()
    med_root = medformer_root(root)
    sys.path.insert(0, str(med_root))

    random.seed(cli_args.seed)
    np.random.seed(cli_args.seed)
    torch.manual_seed(cli_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cli_args.seed)
        torch.cuda.manual_seed_all(cli_args.seed)

    if cli_args.list_checkpoints:
        for model_name in ("Medformer", "MedformerFFT"):
            print(f"\n{model_name}:")
            candidates = checkpoint_candidates(root, model_name, data_name=cli_args.data)
            if candidates:
                for path in candidates:
                    print(f"  {path}")
            else:
                print("  <none found>")
        return 0

    baseline_checkpoint = choose_checkpoint(
        root,
        "Medformer",
        cli_args.baseline_checkpoint,
        cli_args.data,
        cli_args.baseline_model_id,
    )
    fft_checkpoint = choose_checkpoint(
        root,
        "MedformerFFT",
        cli_args.fft_checkpoint,
        cli_args.data,
        cli_args.fft_model_id,
    )

    os.chdir(med_root)
    output_path = resolve_path(cli_args.output, root)
    completed_keys = existing_result_keys(output_path)
    model_specs = [
        ("baseline", "Medformer", baseline_checkpoint),
        ("fft", "MedformerFFT", fft_checkpoint),
    ]
    if cli_args.model_subset != "both":
        model_specs = [spec for spec in model_specs if spec[0] == cli_args.model_subset]
    perturbation_specs = [
        ("gaussian_noise", GAUSSIAN_STRENGTHS),
        ("time_mask", TIME_MASK_STRENGTHS),
    ]

    for model_alias, model_name, checkpoint_path in model_specs:
        print(f"Loading {model_alias}: {checkpoint_path}", flush=True)
        exp, test_loader = load_experiment(cli_args, model_name, checkpoint_path)
        for perturbation, strengths in perturbation_specs:
            for strength in strengths:
                row_key = (model_alias, perturbation, f"{strength:.2f}")
                if row_key in completed_keys:
                    print(
                        f"skip     {model_alias:8s} {perturbation:14s} {strength:.2f}",
                        flush=True,
                    )
                    continue
                metrics = evaluate(
                    exp,
                    test_loader,
                    perturbation=perturbation,
                    strength=strength,
                    seed=cli_args.seed,
                )
                row = {
                    "model_alias": model_alias,
                    "model": model_name,
                    "checkpoint": str(checkpoint_path),
                    "data": cli_args.data,
                    "root_path": cli_args.root_path,
                    "perturbation": perturbation,
                    "strength": f"{strength:.2f}",
                    **metrics,
                }
                append_result(output_path, row)
                completed_keys.add(row_key)
                print(
                    f"{model_alias:8s} {perturbation:14s} {strength:.2f} "
                    f"Acc={metrics['Accuracy']:.5f} F1={metrics['F1']:.5f}",
                    flush=True,
                )

    print(f"\nSaved: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
