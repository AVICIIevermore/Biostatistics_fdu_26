#!/usr/bin/env python3
"""Extract CLIP/DINOv2 embeddings from local MNIST idx gzip files.

This script is intended as a fast smoke-test data source for the full
embedding -> MMMD pipeline.
"""

from __future__ import annotations

import argparse
import gzip
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from extract_medmnist_embeddings import FrozenEncoder, write_embedding_csv


def read_mnist_images(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        _ = int.from_bytes(f.read(4), "big")
        n = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        cols = int.from_bytes(f.read(4), "big")
        data = np.frombuffer(f.read(n * rows * cols), dtype=np.uint8)
    return data.reshape(n, rows, cols)


def read_mnist_labels(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        _ = int.from_bytes(f.read(4), "big")
        n = int.from_bytes(f.read(4), "big")
        data = np.frombuffer(f.read(n), dtype=np.uint8)
    return data.astype(np.int64)


class LocalMNISTDataset(Dataset):
    def __init__(self, images_path: str, labels_path: str, labels_keep: list[int] | None, max_per_label: int | None) -> None:
        images = read_mnist_images(images_path)
        labels = read_mnist_labels(labels_path)

        if labels_keep is not None:
            keep = np.isin(labels, np.asarray(labels_keep, dtype=np.int64))
        else:
            keep = np.ones_like(labels, dtype=bool)
        indices = np.where(keep)[0]

        if max_per_label is not None:
            selected: list[int] = []
            for label in sorted(set(labels[indices].tolist())):
                label_indices = indices[labels[indices] == label][:max_per_label]
                selected.extend(label_indices.tolist())
            indices = np.asarray(selected, dtype=np.int64)

        self.images = images[indices]
        self.labels = labels[indices]
        self.original_ids = indices

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        image = Image.fromarray(self.images[idx], mode="L").convert("RGB")
        return image, int(self.labels[idx]), int(self.original_ids[idx])


def collate_mnist(batch):
    images, labels, ids = zip(*batch)
    return list(images), np.asarray(labels, dtype=np.int64), np.asarray(ids, dtype=np.int64)


def parse_label_list(text: str | None) -> list[int] | None:
    if text is None or text.strip() == "":
        return None
    return [int(x) for x in text.split(",")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--keep-labels", default="1,2,3,8")
    parser.add_argument("--max-per-label", type=int, default=120)
    parser.add_argument("--encoder", default="dinov2", choices=["dinov2", "clip"])
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-l2-normalize", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = LocalMNISTDataset(
        images_path=args.images,
        labels_path=args.labels,
        labels_keep=parse_label_list(args.keep_labels),
        max_per_label=args.max_per_label,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_mnist)
    encoder = FrozenEncoder(args.encoder, args.model_name, args.device, local_files_only=args.local_files_only)

    rows = []
    for batch_id, (images, labels, ids) in enumerate(loader, start=1):
        feats = encoder.encode(images, normalize=not args.no_l2_normalize)
        for idx, label, feat in zip(ids, labels, feats, strict=True):
            rows.append((int(idx), "train", int(label), feat))
        if batch_id % 10 == 0:
            print(f"[embed] batches={batch_id} rows={len(rows)} dim={feats.shape[1]}", flush=True)

    write_embedding_csv(args.output, rows)
    print(f"Wrote {len(rows)} embeddings to {args.output}")


if __name__ == "__main__":
    main()
