#!/usr/bin/env python3
"""Extract CLIP/DINOv2 embeddings for MNIST additive-noise experiments.

This mirrors the structure of Dechao's CNN experiments: for each outer
replication and each noise level, add Gaussian noise to a fixed digit pool,
then extract frozen visual embeddings.
"""

from __future__ import annotations

import argparse
import gzip
import csv
import os
import urllib.request

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from extract_medmnist_embeddings import FrozenEncoder


MNIST_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist"


def ensure_file(url: str, dest: str) -> None:
    if os.path.exists(dest):
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def maybe_download_mnist(data_dir: str, split: str) -> tuple[str, str]:
    if split == "train":
        image_name = "train-images-idx3-ubyte.gz"
        label_name = "train-labels-idx1-ubyte.gz"
    else:
        image_name = "t10k-images-idx3-ubyte.gz"
        label_name = "t10k-labels-idx1-ubyte.gz"
    image_path = os.path.join(data_dir, image_name)
    label_path = os.path.join(data_dir, label_name)
    ensure_file(f"{MNIST_BASE_URL}/{image_name}", image_path)
    ensure_file(f"{MNIST_BASE_URL}/{label_name}", label_path)
    return image_path, label_path


def read_mnist_images(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        _ = int.from_bytes(f.read(4), "big")
        n = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        cols = int.from_bytes(f.read(4), "big")
        data = np.frombuffer(f.read(n * rows * cols), dtype=np.uint8)
    return data.reshape(n, rows, cols).astype(np.float32) / 255.0


def read_mnist_labels(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        _ = int.from_bytes(f.read(4), "big")
        n = int.from_bytes(f.read(4), "big")
        data = np.frombuffer(f.read(n), dtype=np.uint8)
    return data.astype(np.int64)


def parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip() != ""]


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip() != ""]


class NoisyImageDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, ids: np.ndarray) -> None:
        self.images = images
        self.labels = labels
        self.ids = ids

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        arr = np.clip(np.rint(self.images[idx] * 255.0), 0, 255).astype(np.uint8)
        image = Image.fromarray(arr, mode="L").convert("RGB")
        return image, int(self.labels[idx]), int(self.ids[idx])


def collate_noisy(batch):
    images, labels, ids = zip(*batch)
    return list(images), np.asarray(labels, dtype=np.int64), np.asarray(ids, dtype=np.int64)


def select_digit_pool(images: np.ndarray, labels: np.ndarray, keep_labels: list[int], max_per_label: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.where(np.isin(labels, np.asarray(keep_labels, dtype=np.int64)))[0]
    if max_per_label is not None:
        chosen: list[int] = []
        for label in sorted(set(labels[indices].tolist())):
            label_indices = indices[labels[indices] == label][:max_per_label]
            chosen.extend(label_indices.tolist())
        indices = np.asarray(chosen, dtype=np.int64)
    return images[indices], labels[indices], indices


def write_header(path: str, embedding_dim: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        features = ",".join(f"feat_{i:04d}" for i in range(embedding_dim))
        f.write(f"id,split,group,label,outer_iter,noise_sigma,{features}\n")


def append_rows(path: str, split: str, group: str, outer_iter: int, noise_sigma: float, ids: np.ndarray, labels: np.ndarray, feats: np.ndarray) -> None:
    with open(path, "a") as f:
        for idx, label, feat in zip(ids, labels, feats, strict=True):
            feat_text = ",".join(f"{x:.8g}" for x in feat.tolist())
            f.write(f"{int(idx)},{split},{group},{int(label)},{outer_iter},{noise_sigma:.8g},{feat_text}\n")


class EmbeddingWriter:
    def __init__(self, output: str) -> None:
        self.output = output
        self.use_dir = output.endswith(os.sep) or os.path.splitext(output)[1] == ""
        self.csv_header_written = False
        self.embedding_chunks: list[np.ndarray] = []
        self.metadata_rows: list[dict[str, object]] = []
        if self.use_dir:
            os.makedirs(output, exist_ok=True)

    def append(self, split: str, group: str, outer_iter: int, noise_sigma: float, ids: np.ndarray, labels: np.ndarray, feats: np.ndarray) -> None:
        if self.use_dir:
            self.embedding_chunks.append(feats.astype(np.float32, copy=False))
            for idx, label in zip(ids, labels, strict=True):
                self.metadata_rows.append({
                    "id": int(idx),
                    "split": split,
                    "group": group,
                    "label": int(label),
                    "outer_iter": int(outer_iter),
                    "noise_sigma": float(noise_sigma),
                })
        else:
            if not self.csv_header_written:
                write_header(self.output, feats.shape[1])
                self.csv_header_written = True
            append_rows(self.output, split, group, outer_iter, noise_sigma, ids, labels, feats)

    def close(self) -> None:
        if not self.use_dir:
            return
        if not self.embedding_chunks:
            raise SystemExit("No embeddings were generated.")
        embeddings = np.concatenate(self.embedding_chunks, axis=0).astype(np.float32, copy=False)
        np.save(os.path.join(self.output, "embeddings.npy"), embeddings)
        metadata_path = os.path.join(self.output, "metadata.csv")
        with open(metadata_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "split", "group", "label", "outer_iter", "noise_sigma"])
            writer.writeheader()
            writer.writerows(self.metadata_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="yao/data/mnist")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--images", default=None, help="Optional explicit MNIST images gzip path.")
    parser.add_argument("--labels", default=None, help="Optional explicit MNIST labels gzip path.")
    parser.add_argument("--keep-labels", default=None, help="Optional pooled labels. If omitted, x/y label union is used.")
    parser.add_argument("--x-labels", default="1,2,3")
    parser.add_argument("--y-labels", default="1,2,8")
    parser.add_argument("--max-per-label", type=int, default=None)
    parser.add_argument("--noise-levels", default="0,0.2,0.4,0.6,0.8,1")
    parser.add_argument("--n-rep", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--encoder", default="dinov2", choices=["dinov2", "clip"])
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-l2-normalize", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--output", required=True, help="CSV file path, or directory path for embeddings.npy + metadata.csv.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.images and args.labels:
        image_path, label_path = args.images, args.labels
    else:
        image_path, label_path = maybe_download_mnist(args.data_dir, args.split)

    images = read_mnist_images(image_path)
    labels = read_mnist_labels(label_path)
    x_labels = parse_int_list(args.x_labels)
    y_labels = parse_int_list(args.y_labels)
    keep_labels = parse_int_list(args.keep_labels) if args.keep_labels else sorted(set(x_labels + y_labels))
    all_images, all_labels, all_ids = select_digit_pool(
        images,
        labels,
        keep_labels=keep_labels,
        max_per_label=args.max_per_label,
    )
    x_mask = np.isin(all_labels, np.asarray(x_labels, dtype=np.int64))
    y_mask = np.isin(all_labels, np.asarray(y_labels, dtype=np.int64))
    group_payloads = [
        ("X", all_images[x_mask], all_labels[x_mask], all_ids[x_mask]),
        ("Y", all_images[y_mask], all_labels[y_mask], all_ids[y_mask]),
    ]

    encoder = FrozenEncoder(args.encoder, args.model_name, args.device, local_files_only=args.local_files_only)
    noise_levels = parse_float_list(args.noise_levels)
    writer = EmbeddingWriter(args.output)

    for outer_iter in range(1, args.n_rep + 1):
        rng = np.random.default_rng(args.seed + 1000 * outer_iter)
        for sigma in noise_levels:
            for group, group_images, group_labels, group_ids in group_payloads:
                noisy = np.clip(group_images + rng.normal(0.0, sigma, size=group_images.shape).astype(np.float32), 0.0, 1.0)
                dataset = NoisyImageDataset(noisy, group_labels, group_ids)
                loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_noisy)
                for batch_id, (batch_images, batch_labels, batch_ids) in enumerate(loader, start=1):
                    feats = encoder.encode(batch_images, normalize=not args.no_l2_normalize)
                    writer.append(args.split, group, outer_iter, sigma, batch_ids, batch_labels, feats)
                print(f"[embed] outer_iter={outer_iter}/{args.n_rep} sigma={sigma:g} group={group} rows={len(group_labels)}", flush=True)

    writer.close()
    print(f"Wrote noisy embeddings to {args.output}")


if __name__ == "__main__":
    main()
