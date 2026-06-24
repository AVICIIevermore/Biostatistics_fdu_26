#!/usr/bin/env python3
"""Extract a shared raw-pixel pool from a local image-folder dataset."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def select_rows(meta: pd.DataFrame, keep_labels: list[int], max_per_label: int | None) -> pd.DataFrame:
    sub = meta[meta["label"].isin(keep_labels)].copy()
    if max_per_label is not None:
        sub = (
            sub.sort_values(["label", "id"])
            .groupby("label", group_keys=False)
            .head(max_per_label)
            .reset_index(drop=True)
        )
    return sub.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--x-labels", required=True)
    parser.add_argument("--y-labels", required=True)
    parser.add_argument("--keep-labels", default=None)
    parser.add_argument("--max-per-label", type=int, default=None)
    parser.add_argument("--pool-only", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta = pd.read_csv(args.metadata)
    if "split" in meta.columns:
        meta = meta[meta["split"] == args.split].copy()
    x_labels = parse_int_list(args.x_labels)
    y_labels = parse_int_list(args.y_labels)
    keep_labels = parse_int_list(args.keep_labels) if args.keep_labels else sorted(set(x_labels + y_labels))
    meta = select_rows(meta, keep_labels, args.max_per_label)
    if "id" not in meta.columns:
        meta["id"] = np.arange(len(meta), dtype=np.int64)

    image_root = Path(args.image_root).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    feats = []
    rows = []
    for _, row in meta.iterrows():
        image = Image.open(image_root / row["image_path"]).convert("RGB")
        arr = np.asarray(image, dtype=np.float32) / 255.0
        feats.append(arr.reshape(-1).astype(np.float32, copy=False))
        rows.append({
            "id": int(row["id"]),
            "split": str(row["split"]),
            "label": int(row["label"]),
            "outer_iter": 1,
            "noise_sigma": 0.0,
        })

    embeddings = np.stack(feats, axis=0).astype(np.float32, copy=False)
    np.save(output / "embeddings.npy", embeddings)
    with (output / "metadata.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "split", "label", "outer_iter", "noise_sigma"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote raw-pixel pool to {output}")


if __name__ == "__main__":
    main()
