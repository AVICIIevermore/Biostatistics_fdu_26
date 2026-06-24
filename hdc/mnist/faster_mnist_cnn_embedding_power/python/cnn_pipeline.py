import csv
import gzip
import json
import os
import urllib.request
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


MNIST_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist"
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def append_history_row(history_path: str, row: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    file_exists = os.path.exists(history_path)
    fieldnames = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "device"]
    with open(history_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def append_train_log(log_path: str, message: str) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:
        f.write(message + "\n")


def write_train_config(config_path: str, payload: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(payload, f, indent=2)


def plot_training_curves(history_path: str, output_path: str) -> None:
    epochs = []
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    with open(history_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            train_acc.append(float(row["train_acc"]))
            val_acc.append(float(row["val_acc"]))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_loss, marker="o", label="train loss")
    axes[0].plot(epochs, val_loss, marker="o", label="val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("CNN Training Loss")
    axes[0].legend()

    axes[1].plot(epochs, train_acc, marker="o", label="train acc")
    axes[1].plot(epochs, val_acc, marker="o", label="val acc")
    axes[1].axhline(0.98, color="red", linestyle="--", linewidth=1, label="0.98 target")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("CNN Training Accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def ensure_mnist_file(url: str, dest: str) -> None:
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        urllib.request.urlretrieve(url, dest)


def read_mnist_images(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        _ = int.from_bytes(f.read(4), "big")
        n = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        cols = int.from_bytes(f.read(4), "big")
        data = np.frombuffer(f.read(n * rows * cols), dtype=np.uint8)
    return data.reshape(n, rows * cols).astype(np.float32) / 255.0


def read_mnist_labels(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        _ = int.from_bytes(f.read(4), "big")
        n = int.from_bytes(f.read(4), "big")
        data = np.frombuffer(f.read(n), dtype=np.uint8)
    return data.astype(np.int64)


def ensure_mnist_data(data_dir: str) -> None:
    for filename in (
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ):
        ensure_mnist_file(f"{MNIST_BASE_URL}/{filename}", os.path.join(data_dir, filename))


def load_mnist_train_test(data_dir: str) -> Dict[str, np.ndarray]:
    ensure_mnist_data(data_dir)
    return {
        "train_x": read_mnist_images(os.path.join(data_dir, "train-images-idx3-ubyte.gz")),
        "train_y": read_mnist_labels(os.path.join(data_dir, "train-labels-idx1-ubyte.gz")),
        "test_x": read_mnist_images(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")),
        "test_y": read_mnist_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")),
    }


class SmallMNISTCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = x.view(-1, 1, 28, 28)
        layer1_map = self.pool(self.relu(self.conv1(x)))
        layer2_map = self.pool(self.relu(self.conv2(layer1_map)))
        layer1_gap = self.gap(layer1_map).flatten(1)
        layer2_gap = self.gap(layer2_map).flatten(1)
        final_embedding = self.relu(self.fc1(layer2_gap))
        logits = self.fc2(final_embedding)
        return {
            "layer1": layer1_gap,
            "layer2": layer2_gap,
            "final": final_embedding,
            "logits": logits,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)["logits"]


def get_device() -> str:
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        return "cuda"
    return "cpu"


def evaluate_metrics(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.numel()
            running_loss += loss.item() * yb.size(0)
    return {"loss": running_loss / max(total, 1), "acc": correct / max(total, 1)}


def train_cnn_model(
    data_dir: str,
    checkpoint_path: str,
    batch_size: int = 128,
    epochs: int = 20,
    lr: float = 1e-3,
    seed: int = 20260525,
    val_fraction: float = 5000 / 60000,
    target_val_accuracy: float = 0.98,
    history_path: str | None = None,
    train_log_path: str | None = None,
    train_plot_path: str | None = None,
    config_dump_path: str | None = None,
    force_retrain: bool = False,
    resume_from_checkpoint: bool = False,
    resume_lr: float | None = None,
) -> Dict[str, object]:
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    artifacts_dir = os.path.dirname(checkpoint_path)
    history_path = history_path or os.path.join(artifacts_dir, "train_history.csv")
    train_log_path = train_log_path or os.path.join(artifacts_dir, "train.log")
    train_plot_path = train_plot_path or os.path.join(artifacts_dir, "training_curves.png")
    config_dump_path = config_dump_path or os.path.join(artifacts_dir, "train_config.json")

    if (not force_retrain) and os.path.exists(checkpoint_path) and (not resume_from_checkpoint):
        payload = load_checkpoint(checkpoint_path)
        payload["device"] = get_device()
        payload["history_path"] = history_path
        payload["train_log_path"] = train_log_path
        payload["train_plot_path"] = train_plot_path
        payload["config_dump_path"] = config_dump_path
        return payload

    if force_retrain:
        for path in (history_path, train_log_path, train_plot_path):
            if os.path.exists(path):
                os.remove(path)

    device = get_device()
    write_train_config(config_dump_path, {
        "data_dir": data_dir,
        "checkpoint_path": checkpoint_path,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "seed": seed,
        "val_fraction": val_fraction,
        "target_val_accuracy": target_val_accuracy,
        "device": device,
        "normalization_mean": MNIST_MEAN,
        "normalization_std": MNIST_STD,
        "resume_from_checkpoint": resume_from_checkpoint,
        "resume_lr": resume_lr,
    })

    bundle = load_mnist_train_test(data_dir)
    train_x = torch.from_numpy(bundle["train_x"]).float()
    train_y = torch.from_numpy(bundle["train_y"])
    train_x = (train_x - MNIST_MEAN) / MNIST_STD
    dataset = TensorDataset(train_x, train_y)
    val_size = int(round(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device == "cuda"))

    model = SmallMNISTCNN().to(device)
    start_epoch = 0
    best_val_acc = -1.0
    best_state = None

    if resume_from_checkpoint and os.path.exists(checkpoint_path):
        existing_payload = load_checkpoint(checkpoint_path)
        model.load_state_dict(existing_payload["state_dict"])
        start_epoch = int(existing_payload.get("epochs", 0))
        best_val_acc = float(existing_payload.get("best_val_acc", -1.0))
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    effective_lr = resume_lr if (resume_from_checkpoint and resume_lr is not None) else lr
    optimizer = torch.optim.Adam(model.parameters(), lr=effective_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    torch.manual_seed(seed)
    np.random.seed(seed)

    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * yb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.numel()

        train_loss = running_loss / train_size
        train_acc = correct / max(total, 1)
        val_metrics = evaluate_metrics(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "lr": optimizer.param_groups[0]["lr"],
            "device": device,
        }
        append_history_row(history_path, row)
        message = (
            f"[train] epoch={epoch} train_loss={train_loss:.6f} train_acc={train_acc:.4f} "
            f"val_loss={val_metrics['loss']:.6f} val_acc={val_metrics['acc']:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.6g} device={device}"
        )
        print(message, flush=True)
        append_train_log(train_log_path, message)

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        scheduler.step()

        if val_metrics["acc"] >= target_val_accuracy:
            break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")

    payload = {
        "state_dict": best_state,
        "best_val_acc": best_val_acc,
        "device": device,
        "seed": seed,
        "batch_size": batch_size,
        "epochs": epoch,
        "lr": lr,
        "val_fraction": val_fraction,
        "history_path": history_path,
        "train_log_path": train_log_path,
        "train_plot_path": train_plot_path,
        "config_dump_path": config_dump_path,
        "resume_from_checkpoint": resume_from_checkpoint,
        "resume_lr": resume_lr,
    }
    torch.save(payload, checkpoint_path)
    plot_training_curves(history_path, train_plot_path)
    return payload


def load_checkpoint(checkpoint_path: str) -> Dict[str, object]:
    return torch.load(checkpoint_path, map_location="cpu")


_MODEL_CACHE: Dict[str, Tuple[nn.Module, str]] = {}


def load_model_for_inference(checkpoint_path: str) -> Tuple[nn.Module, str]:
    cached = _MODEL_CACHE.get(checkpoint_path)
    device = get_device()
    if cached is not None and cached[1] == device:
        return cached

    payload = load_checkpoint(checkpoint_path)
    model = SmallMNISTCNN().to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    cached = (model, device)
    _MODEL_CACHE[checkpoint_path] = cached
    return cached


def extract_embeddings(
    checkpoint_path: str,
    flat_images: np.ndarray,
    batch_size: int = 1024,
) -> Dict[str, np.ndarray]:
    model, device = load_model_for_inference(checkpoint_path)
    x = torch.from_numpy(np.asarray(flat_images, dtype=np.float32))
    x = (x - MNIST_MEAN) / MNIST_STD
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device == "cuda"))

    layer1_chunks = []
    layer2_chunks = []
    final_chunks = []

    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            out = model.forward_features(xb)
            layer1_chunks.append(out["layer1"].detach().cpu().numpy())
            layer2_chunks.append(out["layer2"].detach().cpu().numpy())
            final_chunks.append(out["final"].detach().cpu().numpy())

    return {
        "layer1": np.concatenate(layer1_chunks, axis=0),
        "layer2": np.concatenate(layer2_chunks, axis=0),
        "final": np.concatenate(final_chunks, axis=0),
    }
