#!/usr/bin/env python3
"""Train a small CNN on BloodMNIST-224 and extract frozen embeddings."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "mnist" / "bloodmnist_224.npz"
DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[1] / "models" / "bloodmnist_cnn"
DEFAULT_LOG_DIR = Path(__file__).resolve().parents[1] / "logs" / "bloodmnist_cnn"

CLASS_NAMES = {
    0: "basophil",
    1: "eosinophil",
    2: "erythroblast",
    3: "immature granulocytes",
    4: "lymphocyte",
    5: "monocyte",
    6: "neutrophil",
    7: "platelet",
}


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(force_cpu: bool = False) -> str:
    if force_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def append_csv_row(path: Path, row: Dict[str, object], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(message + "\n")


def load_bloodmnist_npz(data_path: Path) -> Dict[str, np.ndarray]:
    data = np.load(data_path)
    out: Dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        images = data[f"{split}_images"].astype(np.float32) / 255.0
        labels = data[f"{split}_labels"].reshape(-1).astype(np.int64)
        images = np.transpose(images, (0, 3, 1, 2))
        out[f"{split}_x"] = images
        out[f"{split}_y"] = labels
    return out


class SmallBloodMNISTCNN(nn.Module):
    def __init__(self, n_classes: int = 8) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        layer1_map = self.pool(self.relu(self.conv1(x)))
        layer2_map = self.pool(self.relu(self.conv2(layer1_map)))
        layer1_gap = self.gap(layer1_map).flatten(1)
        layer2_gap = self.gap(layer2_map).flatten(1)
        final_fc128 = self.relu(self.fc1(layer2_gap))
        logits = self.fc2(final_fc128)
        return {
            "layer1_gap": layer1_gap,
            "layer2_gap": layer2_gap,
            "final_fc128": final_fc128,
            "logits": logits,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)["logits"]


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, device: str) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total = 0
    correct = 0
    running_loss = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            preds = logits.argmax(dim=1)
            total += yb.numel()
            correct += (preds == yb).sum().item()
            running_loss += loss.item() * yb.size(0)
    return {"loss": running_loss / max(total, 1), "acc": correct / max(total, 1)}


def confusion_matrix(model: nn.Module, loader: DataLoader, device: str, n_classes: int = 8) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            truth = yb.numpy()
            for t, p in zip(truth, preds):
                matrix[int(t), int(p)] += 1
    return matrix


def plot_training_curves(history_path: Path, output_path: Path) -> None:
    epochs, train_loss, val_loss, train_acc, val_acc = [], [], [], [], []
    with history_path.open(newline="") as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            train_acc.append(float(row["train_acc"]))
            val_acc.append(float(row["val_acc"]))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_loss, marker="o", label="train")
    axes[0].plot(epochs, val_loss, marker="o", label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("BloodMNIST CNN Loss")
    axes[0].legend()

    axes[1].plot(epochs, train_acc, marker="o", label="train")
    axes[1].plot(epochs, val_acc, marker="o", label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("BloodMNIST CNN Accuracy")
    axes[1].legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_confusion_matrix(matrix: np.ndarray, csv_path: Path, png_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_label", "true_class"] + [f"pred_{i}" for i in range(matrix.shape[1])])
        for label, row in enumerate(matrix):
            writer.writerow([label, CLASS_NAMES[label]] + row.tolist())

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_title("BloodMNIST Test Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=160)
    plt.close(fig)


def train_cnn(
    data_path: Path,
    models_dir: Path,
    log_dir: Path,
    batch_size: int,
    epochs: int,
    lr: float,
    seed: int,
    force_cpu: bool,
    force_retrain: bool,
    scheduler_type: str = "step",
    early_stop_patience: int = 0,
    plateau_patience: int = 3,
    plateau_factor: float = 0.5,
    min_lr: float = 1e-5,
) -> Dict[str, object]:
    set_reproducibility(seed)
    device = get_device(force_cpu=force_cpu)
    bundle = load_bloodmnist_npz(data_path)

    checkpoint_path = models_dir / "bloodmnist_cnn_checkpoint.pt"
    config_path = models_dir / "train_config.json"
    history_path = models_dir / "train_history.csv"
    train_log_path = log_dir / "train.log"
    curves_path = models_dir / "training_curves.png"
    metrics_path = models_dir / "classification_metrics.csv"
    confusion_csv_path = models_dir / "test_confusion_matrix.csv"
    confusion_png_path = models_dir / "test_confusion_matrix.png"

    if checkpoint_path.exists() and not force_retrain:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        payload["skipped_existing_checkpoint"] = True
        return payload

    for path in [history_path, train_log_path, curves_path, metrics_path, confusion_csv_path, confusion_png_path, checkpoint_path]:
        if force_retrain and path.exists():
            path.unlink()

    config = {
        "dataset": "BloodMNIST-224",
        "data_path": str(data_path),
        "pixel_scale": "RGB float in [0, 1]",
        "augmentation": "none",
        "normalization": "none beyond division by 255",
        "architecture": "Conv2d(3,32,3,pad=1)-ReLU-MaxPool; Conv2d(32,64,3,pad=1)-ReLU-MaxPool; GAP; Linear(64,128)-ReLU; Linear(128,8)",
        "embedding_names": ["layer1_gap", "layer2_gap", "final_fc128"],
        "checkpoint_selection": "best validation accuracy",
        "batch_size": batch_size,
        "epochs": epochs,
        "optimizer": "Adam",
        "learning_rate": lr,
        "scheduler": scheduler_type,
        "early_stop_patience": early_stop_patience,
        "plateau_patience": plateau_patience,
        "plateau_factor": plateau_factor,
        "min_lr": min_lr,
        "loss": "CrossEntropyLoss",
        "seed": seed,
        "device": device,
        "train_shape": list(bundle["train_x"].shape),
        "val_shape": list(bundle["val_x"].shape),
        "test_shape": list(bundle["test_x"].shape),
    }
    write_json(config_path, config)

    train_loader = make_loader(bundle["train_x"], bundle["train_y"], batch_size, shuffle=True, device=device)
    val_loader = make_loader(bundle["val_x"], bundle["val_y"], batch_size, shuffle=False, device=device)
    test_loader = make_loader(bundle["test_x"], bundle["test_y"], batch_size, shuffle=False, device=device)

    model = SmallBloodMNISTCNN().to(device)
    best_val_acc = -1.0
    best_epoch = 0
    best_state = None
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
    elif scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=plateau_factor, patience=plateau_patience, min_lr=min_lr)
    elif scheduler_type == "none":
        scheduler = None
    else:
        raise ValueError(f"unknown scheduler_type: {scheduler_type}")
    epochs_since_best = 0

    header = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "device"]
    append_log(train_log_path, "BloodMNIST CNN training started")
    append_log(train_log_path, json.dumps(config, sort_keys=True))

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0
        correct = 0
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            preds = logits.argmax(dim=1)
            total += yb.numel()
            correct += (preds == yb).sum().item()
            running_loss += loss.item() * yb.size(0)

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
        print(msg, flush=True)
        append_log(train_log_path, msg)

        improved = val_metrics["acc"] > best_val_acc
        if improved:
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        if scheduler is not None:
            if scheduler_type == "plateau":
                scheduler.step(val_metrics["acc"])
            else:
                scheduler.step()

        if early_stop_patience > 0 and epochs_since_best >= early_stop_patience:
            stop_msg = (
                f"[early-stop] epoch={epoch:02d} no_val_acc_improvement_epochs={epochs_since_best} "
                f"best_epoch={best_epoch} best_val_acc={best_val_acc:.4f}"
            )
            print(stop_msg, flush=True)
            append_log(train_log_path, stop_msg)
            break

    if best_state is None:
        raise RuntimeError("training did not produce a best checkpoint")

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    conf = confusion_matrix(model, test_loader, device)

    payload = {
        "state_dict": best_state,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "test_acc": test_metrics["acc"],
        "test_loss": test_metrics["loss"],
        "seed": seed,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "device": device,
        "config_path": str(config_path),
        "history_path": str(history_path),
        "train_log_path": str(train_log_path),
        "training_curves_path": str(curves_path),
        "metrics_path": str(metrics_path),
        "confusion_matrix_csv_path": str(confusion_csv_path),
        "confusion_matrix_png_path": str(confusion_png_path),
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)
    plot_training_curves(history_path, curves_path)
    save_confusion_matrix(conf, confusion_csv_path, confusion_png_path)

    metric_fields = ["split", "loss", "accuracy", "selected_by", "best_epoch", "seed", "device"]
    if metrics_path.exists():
        metrics_path.unlink()
    append_csv_row(metrics_path, {
        "split": "validation",
        "loss": "",
        "accuracy": best_val_acc,
        "selected_by": "best validation accuracy",
        "best_epoch": best_epoch,
        "seed": seed,
        "device": device,
    }, metric_fields)
    append_csv_row(metrics_path, {
        "split": "test",
        "loss": test_metrics["loss"],
        "accuracy": test_metrics["acc"],
        "selected_by": "reported after checkpoint selection only",
        "best_epoch": best_epoch,
        "seed": seed,
        "device": device,
    }, metric_fields)

    done_msg = f"[done] best_epoch={best_epoch} best_val_acc={best_val_acc:.4f} test_acc={test_metrics['acc']:.4f}"
    print(done_msg, flush=True)
    append_log(train_log_path, done_msg)
    return payload


_MODEL_CACHE: Dict[str, Tuple[nn.Module, str]] = {}


def load_model_for_inference(checkpoint_path: Path, force_cpu: bool = False) -> Tuple[nn.Module, str]:
    device = get_device(force_cpu=force_cpu)
    cache_key = f"{checkpoint_path}:{device}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = SmallBloodMNISTCNN().to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    _MODEL_CACHE[cache_key] = (model, device)
    return model, device


def extract_embeddings(
    checkpoint_path: Path,
    images: np.ndarray,
    batch_size: int = 512,
    layers: list[str] | None = None,
    force_cpu: bool = False,
) -> Dict[str, np.ndarray]:
    model, device = load_model_for_inference(checkpoint_path, force_cpu=force_cpu)
    arr = np.asarray(images)
    if arr.ndim == 4 and arr.shape[-1] == 3:
        arr = np.transpose(arr, (0, 3, 1, 2))
    arr = arr.astype(np.float32)
    if arr.max(initial=0) > 1.0:
        arr = arr / 255.0

    loader = DataLoader(TensorDataset(torch.from_numpy(arr).float()), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device == "cuda"))
    requested = tuple(layers or ["layer1_gap", "layer2_gap", "final_fc128"])
    chunks: Dict[str, list[np.ndarray]] = {name: [] for name in requested}

    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            outputs = model.forward_features(xb)
            for name in requested:
                chunks[name].append(outputs[name].detach().cpu().numpy())
    return {name: np.concatenate(parts, axis=0) for name, parts in chunks.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the BloodMNIST-224 CNN model.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--scheduler", choices=["step", "plateau", "none"], default="step")
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--plateau-patience", type=int, default=3)
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = train_cnn(
        data_path=args.data_path,
        models_dir=args.models_dir,
        log_dir=args.log_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        force_cpu=args.cpu,
        force_retrain=args.force_retrain,
        scheduler_type=args.scheduler,
        early_stop_patience=args.early_stop_patience,
        plateau_patience=args.plateau_patience,
        plateau_factor=args.plateau_factor,
        min_lr=args.min_lr,
    )
    summary = {k: payload[k] for k in payload if k != "state_dict"}
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
