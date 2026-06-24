#!/usr/bin/env python3
"""Train a small CNN on an image-folder dataset described by metadata.csv."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def choose_device(force_cpu: bool = False) -> str:
    if force_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FolderDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, image_root: Path) -> None:
        self.rows = rows.reset_index(drop=True)
        self.image_root = image_root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows.iloc[idx]
        image = Image.open(self.image_root / row["image_path"]).convert("RGB")
        arr = np.asarray(image, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr.transpose(2, 0, 1))
        y = int(row["label"])
        return x, y


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z1 = F.relu(self.conv1(x))
        p1 = self.pool(z1)
        layer1_gap = p1.mean(dim=(2, 3))

        z2 = F.relu(self.conv2(p1))
        p2 = self.pool(z2)
        layer2_gap = p2.mean(dim=(2, 3))

        final_fc128 = F.relu(self.fc1(layer2_gap))
        return {
            "layer1_gap": layer1_gap,
            "layer2_gap": layer2_gap,
            "final_fc128": final_fc128,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        return self.fc2(feats["final_fc128"])


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            running_loss += float(loss.item()) * int(y.numel())
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    return {"loss": running_loss / max(total, 1), "acc": correct / max(total, 1)}


def append_csv_row(path: Path, row: dict[str, object], fieldnames: Iterable[str]) -> None:
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def plot_training_curves(history_path: Path, output_path: Path) -> None:
    df = pd.read_csv(history_path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=180)
    axes[0].plot(df["epoch"], df["train_loss"], label="train")
    axes[0].plot(df["epoch"], df["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].legend()
    axes[1].plot(df["epoch"], df["train_acc"], label="train")
    axes[1].plot(df["epoch"], df["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, linestyle=":", alpha=0.5)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    mat = np.zeros((n_classes, n_classes), dtype=np.int64)
    for a, b in zip(y_true, y_pred, strict=True):
        mat[int(a), int(b)] += 1
    return mat


def save_confusion_outputs(matrix: np.ndarray, labels: list[str], csv_path: Path, png_path: Path) -> None:
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(csv_path)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=180)
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Test Confusion Matrix")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)


def train_model(
    metadata_path: Path,
    image_root: Path,
    models_dir: Path,
    logs_dir: Path,
    batch_size: int,
    epochs: int,
    lr: float,
    seed: int,
    scheduler_name: str,
    plateau_factor: float,
    plateau_patience: int,
    min_lr: float,
    early_stop_patience: int,
    force_retrain: bool,
    force_cpu: bool,
) -> dict[str, object]:
    device = choose_device(force_cpu)
    checkpoint_path = models_dir / "imagefolder_cnn_checkpoint.pt"
    history_path = models_dir / "train_history.csv"
    metrics_path = models_dir / "classification_metrics.csv"
    curves_path = models_dir / "training_curves.png"
    confusion_csv_path = models_dir / "test_confusion_matrix.csv"
    confusion_png_path = models_dir / "test_confusion_matrix.png"
    train_log_path = logs_dir / "train.log"

    if checkpoint_path.exists() and not force_retrain:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        payload["skipped_existing_checkpoint"] = True
        return payload

    set_seed(seed)
    meta = pd.read_csv(metadata_path)
    if "split" not in meta.columns:
        raise SystemExit("metadata.csv must contain a split column")
    class_labels = (
        meta[["label", "class_name"]]
        .drop_duplicates()
        .sort_values("label")
    )
    labels = class_labels["class_name"].tolist()
    n_classes = len(labels)

    train_rows = meta[meta["split"] == "train"].copy()
    val_rows = meta[meta["split"] == "val"].copy()
    test_rows = meta[meta["split"] == "test"].copy()
    if min(len(train_rows), len(val_rows), len(test_rows)) == 0:
        raise SystemExit("train/val/test splits must all be non-empty")

    train_loader = DataLoader(FolderDataset(train_rows, image_root), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(FolderDataset(val_rows, image_root), batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(FolderDataset(test_rows, image_root), batch_size=batch_size, shuffle=False, num_workers=0)

    for path in [models_dir, logs_dir]:
        path.mkdir(parents=True, exist_ok=True)
    for path in [history_path, train_log_path, curves_path, metrics_path, confusion_csv_path, confusion_png_path, checkpoint_path]:
        if path.exists():
            path.unlink()

    config = {
        "dataset": image_root.name,
        "metadata_path": str(metadata_path),
        "image_root": str(image_root),
        "architecture": "Conv2d(3,32,3,pad=1)-ReLU-MaxPool; Conv2d(32,64,3,pad=1)-ReLU-MaxPool; GAP; Linear(64,128)-ReLU; Linear(128,C)",
        "train_shape": [len(train_rows), 3, 224, 224],
        "val_shape": [len(val_rows), 3, 224, 224],
        "test_shape": [len(test_rows), 3, 224, 224],
        "class_counts_train": dict(Counter(train_rows["class_name"])),
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": lr,
        "optimizer": "Adam",
        "scheduler": scheduler_name,
        "plateau_factor": plateau_factor,
        "plateau_patience": plateau_patience,
        "min_lr": min_lr,
        "early_stop_patience": early_stop_patience,
        "loss": "CrossEntropyLoss",
        "pixel_scale": "RGB float in [0, 1]",
        "normalization": "none beyond division by 255",
        "augmentation": "none",
        "checkpoint_selection": "best validation accuracy",
        "seed": seed,
        "device": device,
        "embedding_names": ["layer1_gap", "layer2_gap", "final_fc128"],
    }
    (models_dir / "train_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
    with train_log_path.open("w") as f:
        f.write("Image-folder CNN training started\n")
        f.write(json.dumps(config, ensure_ascii=False) + "\n")

    model = SmallCNN(n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    if scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=plateau_factor, patience=plateau_patience, min_lr=min_lr
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs // 3, 1), gamma=0.5)

    header = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "device"]
    best_val_acc = -1.0
    best_epoch = 0
    best_state = None
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0
        correct = 0
        running_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * int(y.numel())
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())

        train_metrics = {"loss": running_loss / max(total, 1), "acc": correct / max(total, 1)}
        val_metrics = evaluate(model, val_loader, device)
        lr_now = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "lr": lr_now,
            "device": device,
        }
        append_csv_row(history_path, row, header)
        msg = (
            f"[train] epoch={epoch:02d} train_loss={train_metrics['loss']:.6f} "
            f"train_acc={train_metrics['acc']:.4f} val_loss={val_metrics['loss']:.6f} "
            f"val_acc={val_metrics['acc']:.4f} lr={lr_now:.6g} device={device}"
        )
        with train_log_path.open("a") as f:
            f.write(msg + "\n")
        print(msg, flush=True)

        improved = val_metrics["acc"] > best_val_acc
        if improved:
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        if scheduler_name == "plateau":
            scheduler.step(val_metrics["acc"])
        else:
            scheduler.step()

        if early_stop_patience > 0 and stale_epochs >= early_stop_patience:
            break

    if best_state is None:
        raise RuntimeError("training did not produce a best checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    test_metrics = evaluate(model, test_loader, device)

    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x.to(device))
            pred = logits.argmax(dim=1).cpu().numpy()
            y_true.extend(y.numpy().tolist())
            y_pred.extend(pred.tolist())
    conf = confusion_matrix(np.asarray(y_true), np.asarray(y_pred), n_classes)
    save_confusion_outputs(conf, labels, confusion_csv_path, confusion_png_path)

    payload = {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "test_acc": test_metrics["acc"],
        "test_loss": test_metrics["loss"],
        "seed": seed,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "device": device,
        "config_path": str(models_dir / "train_config.json"),
        "history_path": str(history_path),
        "train_log_path": str(train_log_path),
        "training_curves_path": str(curves_path),
        "metrics_path": str(metrics_path),
        "confusion_matrix_csv_path": str(confusion_csv_path),
        "confusion_matrix_png_path": str(confusion_png_path),
        "model_state_dict": best_state,
        "labels": labels,
        "num_classes": n_classes,
    }
    torch.save(payload, checkpoint_path)
    plot_training_curves(history_path, curves_path)

    append_csv_row(metrics_path, {
        "split": "validation",
        "loss": "",
        "accuracy": best_val_acc,
        "selected_by": "best validation accuracy",
        "best_epoch": best_epoch,
        "seed": seed,
        "device": device,
    }, ["split", "loss", "accuracy", "selected_by", "best_epoch", "seed", "device"])
    append_csv_row(metrics_path, {
        "split": "test",
        "loss": test_metrics["loss"],
        "accuracy": test_metrics["acc"],
        "selected_by": "reported after checkpoint selection only",
        "best_epoch": best_epoch,
        "seed": seed,
        "device": device,
    }, ["split", "loss", "accuracy", "selected_by", "best_epoch", "seed", "device"])

    done_msg = f"[done] best_epoch={best_epoch} best_val_acc={best_val_acc:.4f} test_acc={test_metrics['acc']:.4f}"
    with train_log_path.open("a") as f:
        f.write(done_msg + "\n")
    print(done_msg, flush=True)
    return payload


_MODEL_CACHE: dict[str, tuple[nn.Module, str]] = {}


def load_model_for_inference(checkpoint_path: Path, force_cpu: bool = False) -> tuple[nn.Module, str]:
    device = choose_device(force_cpu)
    cache_key = f"{checkpoint_path}:{device}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = SmallCNN(int(payload["num_classes"]))
    model.load_state_dict(payload["model_state_dict"])
    model.eval().to(device)
    _MODEL_CACHE[cache_key] = (model, device)
    return model, device


def extract_embeddings(
    checkpoint_path: Path,
    data_loader: DataLoader,
    layer_name: str,
    force_cpu: bool = False,
) -> np.ndarray:
    if layer_name not in {"layer1_gap", "layer2_gap", "final_fc128"}:
        raise ValueError(f"Unsupported layer_name: {layer_name}")
    model, device = load_model_for_inference(checkpoint_path, force_cpu=force_cpu)
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in data_loader:
            x = batch[0]
            feats = model.forward_features(x.to(device))[layer_name].cpu().numpy()
            outputs.append(feats.astype(np.float32, copy=False))
    return np.concatenate(outputs, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--models-dir", default=str(PROJECT_ROOT / "yao" / "models" / "nctcrc_cnn"))
    parser.add_argument("--logs-dir", default=str(PROJECT_ROOT / "yao" / "logs" / "nctcrc_cnn"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--scheduler", choices=["step", "plateau"], default="step")
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-patience", type=int, default=3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = train_model(
        metadata_path=Path(args.metadata).resolve(),
        image_root=Path(args.image_root).resolve(),
        models_dir=Path(args.models_dir).resolve(),
        logs_dir=Path(args.logs_dir).resolve(),
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        scheduler_name=args.scheduler,
        plateau_factor=args.plateau_factor,
        plateau_patience=args.plateau_patience,
        min_lr=args.min_lr,
        early_stop_patience=args.early_stop_patience,
        force_retrain=args.force_retrain,
        force_cpu=args.cpu,
    )
    slim = {k: v for k, v in payload.items() if k != "model_state_dict"}
    print(json.dumps(slim, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
