#!/usr/bin/env python3
"""Center-shift PathMNIST CNN training and embedding pipeline.

This CNN is trained only on the source-domain auxiliary pool carved from the
PathMNIST official train split. The source holdout pool and official test split
are reserved for MMMD two-sample testing.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
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

CLASS_NAMES = {
    0: "adipose",
    1: "background",
    2: "debris",
    3: "lymphocytes",
    4: "mucus",
    5: "smooth muscle",
    6: "normal colon mucosa",
    7: "cancer-associated stroma",
    8: "colorectal adenocarcinoma epithelium",
}

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = DEFAULT_ROOT / "data" / "pathmnist.npz"
DEFAULT_CONFIG_DIR = DEFAULT_ROOT / "configs"
DEFAULT_MODELS_DIR = DEFAULT_WORKSPACE / "models"
DEFAULT_LOG_DIR = DEFAULT_WORKSPACE / "logs"


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False


def get_device(prefer_cuda: bool = True) -> str:
    if prefer_cuda and torch.cuda.is_available():
        # This shared GPU environment has shown cuDNN initialization issues in earlier runs.
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        return "cuda"
    return "cpu"


def append_csv_row(path: Path, row: Dict[str, object], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(message + "\n")


def load_pathmnist_npz(data_path: Path) -> Dict[str, np.ndarray]:
    data = np.load(data_path)
    out: Dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        images = data[f"{split}_images"].astype(np.float32) / 255.0
        labels = data[f"{split}_labels"].reshape(-1).astype(np.int64)
        images = np.transpose(images, (0, 3, 1, 2))
        out[f"{split}_x"] = images
        out[f"{split}_y"] = labels
    return out


def make_center_shift_split(labels: np.ndarray, holdout_fraction: float = 0.20, seed: int = 2026) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    cnn_train_parts = []
    source_holdout_parts = []
    labels = np.asarray(labels).reshape(-1)
    for label in sorted(np.unique(labels).tolist()):
        idx = np.where(labels == label)[0]
        idx = rng.permutation(idx)
        n_holdout = int(round(len(idx) * holdout_fraction))
        source_holdout_parts.append(idx[:n_holdout])
        cnn_train_parts.append(idx[n_holdout:])
    return {
        "cnn_train_indices": np.sort(np.concatenate(cnn_train_parts).astype(np.int64)),
        "source_holdout_indices": np.sort(np.concatenate(source_holdout_parts).astype(np.int64)),
    }


def save_split_indices(config_dir: Path, split: Dict[str, np.ndarray], holdout_fraction: float, seed: int) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    out_path = config_dir / f"center_shift_split_seed{seed}_holdout{int(holdout_fraction * 100):02d}.npz"
    np.savez(
        out_path,
        cnn_train_indices=split["cnn_train_indices"],
        source_holdout_indices=split["source_holdout_indices"],
        seed=np.array([seed], dtype=np.int64),
        holdout_fraction=np.array([holdout_fraction], dtype=np.float64),
    )
    return out_path


def class_count_rows(labels_by_pool: Dict[str, np.ndarray]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split_name, labels in labels_by_pool.items():
        unique, counts = np.unique(labels, return_counts=True)
        count_by_label = dict(zip(unique.tolist(), counts.tolist()))
        for label in range(9):
            rows.append(
                {
                    "split": split_name,
                    "label": label,
                    "label_name": CLASS_NAMES[label],
                    "count": int(count_by_label.get(label, 0)),
                }
            )
    return rows


class SmallPathMNISTCNN(nn.Module):
    def __init__(self, n_classes: int = 9) -> None:
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


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, device: str, seed: int) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=(device == "cuda"),
        persistent_workers=True,
        generator=generator,
    )


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total = 0
    correct = 0
    running_loss = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            preds = logits.argmax(dim=1)
            total += yb.numel()
            correct += (preds == yb).sum().item()
            running_loss += loss.item() * yb.size(0)
    return {"loss": running_loss / max(total, 1), "acc": correct / max(total, 1)}


def confusion_matrix(model: nn.Module, loader: DataLoader, device: str, n_classes: int = 9) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            truth = yb.numpy()
            for t, p in zip(truth, preds):
                matrix[int(t), int(p)] += 1
    return matrix


def plot_training_curves(history_path: Path, output_path: Path) -> None:
    epochs, train_loss, val_loss, train_acc, val_acc, lrs = [], [], [], [], [], []
    with history_path.open(newline="") as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            train_acc.append(float(row["train_acc"]))
            val_acc.append(float(row["val_acc"]))
            lrs.append(float(row["lr"]))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(epochs, train_loss, marker="o", label="train")
    axes[0].plot(epochs, val_loss, marker="o", label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Center-Shift CNN Loss")
    axes[0].legend()

    axes[1].plot(epochs, train_acc, marker="o", label="train")
    axes[1].plot(epochs, val_acc, marker="o", label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Center-Shift CNN Accuracy")
    axes[1].legend()

    axes[2].plot(epochs, lrs, marker="o")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning rate")
    axes[2].set_title("Learning Rate")

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
    ax.set_title("PathMNIST Validation Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=160)
    plt.close(fig)


def remove_existing_training_outputs(paths: Iterable[Path]) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def train_cnn(
    data_path: Path,
    config_dir: Path,
    models_dir: Path,
    log_dir: Path,
    batch_size: int,
    epochs: int,
    lr: float,
    seed: int,
    holdout_fraction: float,
    prefer_cuda: bool,
    force_retrain: bool,
    resume_from_latest: bool,
    scheduler_type: str = "none",
    plateau_patience: int = 4,
    plateau_factor: float = 0.5,
    min_lr: float = 1e-5,
    target_val_acc: float = 0.0,
) -> Dict[str, object]:
    if force_retrain and resume_from_latest:
        raise ValueError("force_retrain and resume_from_latest are mutually exclusive")

    set_reproducibility(seed)
    device = get_device(prefer_cuda=prefer_cuda)
    bundle = load_pathmnist_npz(data_path)
    split = make_center_shift_split(bundle["train_y"], holdout_fraction=holdout_fraction, seed=seed)
    split_indices_path = save_split_indices(config_dir, split, holdout_fraction, seed)

    train_idx = split["cnn_train_indices"]
    source_holdout_idx = split["source_holdout_indices"]
    cnn_train_x = bundle["train_x"][train_idx]
    cnn_train_y = bundle["train_y"][train_idx]
    val_x = bundle["val_x"]
    val_y = bundle["val_y"]

    checkpoint_path = models_dir / "pathmnist_cnn_checkpoint.pt"
    latest_path = models_dir / "pathmnist_cnn_latest.pt"
    config_path = models_dir / "train_config.json"
    run_history_path = models_dir / "train_runs.csv"
    history_path = models_dir / "train_history.csv"
    train_log_path = log_dir / "pathmnist_train.log"
    curves_path = models_dir / "training_curves.png"
    metrics_path = models_dir / "classification_metrics.csv"
    confusion_csv_path = models_dir / "val_confusion_matrix.csv"
    confusion_png_path = models_dir / "val_confusion_matrix.png"

    removable = [
        checkpoint_path,
        latest_path,
        config_path,
        run_history_path,
        history_path,
        train_log_path,
        curves_path,
        metrics_path,
        confusion_csv_path,
        confusion_png_path,
    ]
    if force_retrain:
        remove_existing_training_outputs(removable)

    existing_payload = None
    start_epoch = 0
    model = SmallPathMNISTCNN().to(device)
    best_val_acc = -1.0
    best_epoch = 0
    best_state = None

    if resume_from_latest:
        if not latest_path.exists():
            raise FileNotFoundError(f"cannot resume; latest checkpoint not found: {latest_path}")
        existing_payload = torch.load(latest_path, map_location="cpu", weights_only=False)
        model.load_state_dict(existing_payload["latest_state_dict"])
        best_val_acc = float(existing_payload.get("best_val_acc", -1.0))
        best_epoch = int(existing_payload.get("best_epoch", 0))
        best_state = existing_payload.get("best_state_dict")
        start_epoch = int(existing_payload.get("latest_epoch", 0))

    end_epoch = start_epoch + epochs
    config = {
        "dataset": "PathMNIST-28",
        "experiment": "within-class source-to-external center shift",
        "data_path": str(data_path),
        "split_indices_path": str(split_indices_path),
        "source_split_rule": "official train stratified into cnn_train_pool and source_holdout_pool",
        "holdout_fraction": holdout_fraction,
        "cnn_training_pool": "cnn_train_pool only",
        "validation_pool": "official val only",
        "external_pool_role": "reserved for MMMD testing; not used in CNN training or checkpoint selection",
        "pixel_scale": "RGB float in [0, 1]",
        "augmentation": "none",
        "normalization": "none beyond division by 255",
        "architecture": "Conv2d(3,32,3,pad=1)-ReLU-MaxPool; Conv2d(32,64,3,pad=1)-ReLU-MaxPool; GAP; Linear(64,128)-ReLU; Linear(128,9)",
        "embedding_names": ["layer1_gap", "layer2_gap", "final_fc128"],
        "checkpoint_selection": "best official validation accuracy",
        "batch_size": batch_size,
        "epochs_requested_this_run": epochs,
        "start_epoch": start_epoch,
        "end_epoch": end_epoch,
        "resume_from_latest": resume_from_latest,
        "optimizer": "Adam",
        "learning_rate": lr,
        "scheduler": scheduler_type,
        "plateau_patience": plateau_patience,
        "plateau_factor": plateau_factor,
        "min_lr": min_lr,
        "target_val_acc": target_val_acc,
        "loss": "CrossEntropyLoss",
        "seed": seed,
        "device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "official_train_shape": list(bundle["train_x"].shape),
        "cnn_train_shape": list(cnn_train_x.shape),
        "source_holdout_shape": [int(source_holdout_idx.size), 3, 28, 28],
        "val_shape": list(val_x.shape),
        "external_pool_shape": list(bundle["test_x"].shape),
    }
    write_json(config_path, config)

    label_pools = {
        "cnn_train_pool": cnn_train_y,
        "source_holdout_pool": bundle["train_y"][source_holdout_idx],
        "val": bundle["val_y"],
        "external_pool": bundle["test_y"],
    }
    write_csv(
        config_dir / "pathmnist_center_shift_class_counts.csv",
        ["split", "label", "label_name", "count"],
        class_count_rows(label_pools),
    )

    train_loader = make_loader(cnn_train_x, cnn_train_y, batch_size, shuffle=True, device=device, seed=seed + start_epoch)
    train_eval_loader = make_loader(cnn_train_x, cnn_train_y, batch_size, shuffle=False, device=device, seed=seed)
    val_loader = make_loader(val_x, val_y, batch_size, shuffle=False, device=device, seed=seed)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if scheduler_type == "none":
        scheduler = None
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
    elif scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=plateau_factor,
            patience=plateau_patience,
            min_lr=min_lr,
        )
    else:
        raise ValueError(f"unknown scheduler_type: {scheduler_type}")

    history_header = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "device"]
    run_header = ["run_start_epoch", "run_end_epoch", "epochs", "lr", "scheduler", "resume_from_latest", "seed", "device"]
    append_csv_row(run_history_path, {
        "run_start_epoch": start_epoch + 1,
        "run_end_epoch": end_epoch,
        "epochs": epochs,
        "lr": lr,
        "scheduler": scheduler_type,
        "resume_from_latest": int(resume_from_latest),
        "seed": seed,
        "device": device,
    }, run_header)

    append_log(train_log_path, "PathMNIST center-shift CNN training resumed" if resume_from_latest else "PathMNIST center-shift CNN training started")
    append_log(train_log_path, json.dumps(config, sort_keys=True))

    final_epoch = start_epoch
    for epoch in range(start_epoch + 1, end_epoch + 1):
        final_epoch = epoch
        model.train()
        total = 0
        correct = 0
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
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
        append_csv_row(history_path, row, history_header)
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
            best_payload = {
                "state_dict": best_state,
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
                "seed": seed,
                "holdout_fraction": holdout_fraction,
                "batch_size": batch_size,
                "latest_epoch": epoch,
                "lr": lr,
                "device": device,
                "config_path": str(config_path),
                "history_path": str(history_path),
                "train_log_path": str(train_log_path),
            }
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_payload, checkpoint_path)

        latest_payload = {
            "latest_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "best_state_dict": best_state,
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "latest_epoch": epoch,
            "seed": seed,
            "holdout_fraction": holdout_fraction,
            "batch_size": batch_size,
            "lr": lr,
            "device": device,
            "config_path": str(config_path),
            "history_path": str(history_path),
            "train_log_path": str(train_log_path),
        }
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(latest_payload, latest_path)

        if scheduler is not None:
            if scheduler_type == "plateau":
                scheduler.step(val_metrics["acc"])
            else:
                scheduler.step()

        if target_val_acc > 0 and best_val_acc >= target_val_acc:
            stop_msg = (
                f"[target-stop] epoch={epoch:02d} best_epoch={best_epoch} "
                f"best_val_acc={best_val_acc:.4f} target_val_acc={target_val_acc:.4f}"
            )
            print(stop_msg, flush=True)
            append_log(train_log_path, stop_msg)
            break

    if best_state is None:
        if existing_payload is not None and existing_payload.get("best_state_dict") is not None:
            best_state = existing_payload["best_state_dict"]
        else:
            raise RuntimeError("training did not produce a best checkpoint")

    best_model = SmallPathMNISTCNN().to(device)
    best_model.load_state_dict(best_state)
    train_best = evaluate(best_model, train_eval_loader, device)
    val_best = evaluate(best_model, val_loader, device)
    conf = confusion_matrix(best_model, val_loader, device)

    plot_training_curves(history_path, curves_path)
    save_confusion_matrix(conf, confusion_csv_path, confusion_png_path)

    metric_fields = ["split", "loss", "accuracy", "selected_by", "best_epoch", "seed", "device"]
    if metrics_path.exists():
        metrics_path.unlink()
    append_csv_row(metrics_path, {
        "split": "cnn_train_pool",
        "loss": train_best["loss"],
        "accuracy": train_best["acc"],
        "selected_by": "best validation accuracy checkpoint",
        "best_epoch": best_epoch,
        "seed": seed,
        "device": device,
    }, metric_fields)
    append_csv_row(metrics_path, {
        "split": "validation",
        "loss": val_best["loss"],
        "accuracy": val_best["acc"],
        "selected_by": "best validation accuracy",
        "best_epoch": best_epoch,
        "seed": seed,
        "device": device,
    }, metric_fields)

    done_msg = f"[done] best_epoch={best_epoch} best_val_acc={best_val_acc:.4f} train_acc_at_best={train_best['acc']:.4f}"
    print(done_msg, flush=True)
    append_log(train_log_path, done_msg)
    return {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "train_acc_at_best": train_best["acc"],
        "train_loss_at_best": train_best["loss"],
        "val_loss_at_best": val_best["loss"],
        "latest_epoch": final_epoch,
        "checkpoint_path": str(checkpoint_path),
        "latest_checkpoint_path": str(latest_path),
        "history_path": str(history_path),
        "metrics_path": str(metrics_path),
        "training_curves_path": str(curves_path),
        "log_path": str(train_log_path),
    }


_MODEL_CACHE: Dict[str, Tuple[nn.Module, str]] = {}


def load_model_for_inference(checkpoint_path: Path, prefer_cuda: bool = True) -> Tuple[nn.Module, str]:
    device = get_device(prefer_cuda=prefer_cuda)
    cache_key = f"{checkpoint_path}:{device}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = SmallPathMNISTCNN().to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    _MODEL_CACHE[cache_key] = (model, device)
    return model, device


def extract_embeddings(
    checkpoint_path: Path,
    images: np.ndarray,
    batch_size: int = 1024,
    layers: list[str] | None = None,
    prefer_cuda: bool = True,
) -> Dict[str, np.ndarray]:
    """Extract frozen CNN representations from RGB images in uint8 or [0,1] float form."""
    model, device = load_model_for_inference(checkpoint_path, prefer_cuda=prefer_cuda)
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
            xb = xb.to(device, non_blocking=True)
            outputs = model.forward_features(xb)
            for name in requested:
                chunks[name].append(outputs[name].detach().cpu().numpy())
    return {name: np.concatenate(parts, axis=0) for name, parts in chunks.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the center-shift PathMNIST-28 CNN model.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--resume-from-latest", action="store_true")
    parser.add_argument("--scheduler", choices=["none", "step", "plateau"], default="none")
    parser.add_argument("--plateau-patience", type=int, default=4)
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--target-val-acc", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = train_cnn(
        data_path=args.data_path,
        config_dir=args.config_dir,
        models_dir=args.models_dir,
        log_dir=args.log_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        holdout_fraction=args.holdout_fraction,
        prefer_cuda=not args.cpu,
        force_retrain=args.force_retrain,
        resume_from_latest=args.resume_from_latest,
        scheduler_type=args.scheduler,
        plateau_patience=args.plateau_patience,
        plateau_factor=args.plateau_factor,
        min_lr=args.min_lr,
        target_val_acc=args.target_val_acc,
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
