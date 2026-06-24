#!/usr/bin/env python3
"""Scan a class-per-folder image dataset and build a stratified metadata CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Root directory with one subfolder per class.")
    parser.add_argument("--output", required=True, help="Output metadata.csv path.")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260602)
    return parser.parse_args()


def assign_splits(n_items: int, train_frac: float, val_frac: float) -> list[str]:
    n_train = int(round(n_items * train_frac))
    n_val = int(round(n_items * val_frac))
    if n_train + n_val >= n_items:
        n_val = max(0, min(n_val, n_items - 1))
        n_train = max(0, min(n_train, n_items - n_val - 1))
    n_test = n_items - n_train - n_val
    return ["train"] * n_train + ["val"] * n_val + ["test"] * n_test


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if not data_root.is_dir():
        raise SystemExit(f"Dataset root not found: {data_root}")
    if not (0.0 < args.train_frac < 1.0 and 0.0 <= args.val_frac < 1.0 and args.train_frac + args.val_frac < 1.0):
        raise SystemExit("Need 0 < train_frac < 1, 0 <= val_frac < 1, and train_frac + val_frac < 1.")

    rng = np.random.default_rng(args.seed)
    class_dirs = sorted([path for path in data_root.iterdir() if path.is_dir()])
    if not class_dirs:
        raise SystemExit(f"No class subfolders found under: {data_root}")

    rows: list[dict[str, object]] = []
    sample_id = 0
    for label, class_dir in enumerate(class_dirs):
        files = sorted(
            [
                path
                for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ]
        )
        if not files:
            continue
        perm = rng.permutation(len(files))
        shuffled = [files[i] for i in perm]
        splits = assign_splits(len(shuffled), args.train_frac, args.val_frac)
        for split, path in zip(splits, shuffled, strict=True):
            rows.append(
                {
                    "id": sample_id,
                    "image_path": str(path.relative_to(data_root)),
                    "class_name": class_dir.name,
                    "label": label,
                    "split": split,
                }
            )
            sample_id += 1

    if not rows:
        raise SystemExit(f"No images found under: {data_root}")

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "image_path", "class_name", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output}")
    for label, class_dir in enumerate(class_dirs):
        count = sum(1 for row in rows if row["label"] == label)
        print(f"{label},{class_dir.name},{count}")


if __name__ == "__main__":
    main()
