#!/usr/bin/env python3
"""Extract shared CLIP/DINOv2 embedding pools from a local image-folder dataset."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from extract_medmnist_embeddings import FrozenEncoder


def parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


class FolderImageDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, image_root: Path, noise_sigma: float, rng_seed: int) -> None:
        self.rows = rows.reset_index(drop=True)
        self.image_root = image_root
        self.noise_sigma = float(noise_sigma)
        self.rng_seed = int(rng_seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows.iloc[idx]
        image = Image.open(self.image_root / row["image_path"]).convert("RGB")
        arr = np.asarray(image, dtype=np.float32) / 255.0
        if self.noise_sigma > 0:
            rng = np.random.default_rng(self.rng_seed + int(row["id"]))
            arr = np.clip(arr + rng.normal(0.0, self.noise_sigma, size=arr.shape).astype(np.float32), 0.0, 1.0)
        arr_uint8 = np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)
        return Image.fromarray(arr_uint8, mode="RGB"), int(row["label"]), int(row["id"]), str(row["split"])


def collate_batch(batch):
    images, labels, ids, splits = zip(*batch)
    return list(images), np.asarray(labels, dtype=np.int64), np.asarray(ids, dtype=np.int64), list(splits)


class EmbeddingWriter:
    def __init__(self, output: str, include_group: bool) -> None:
        self.output = output
        self.include_group = include_group
        self.embedding_chunks: list[np.ndarray] = []
        self.metadata_rows: list[dict[str, object]] = []
        os.makedirs(output, exist_ok=True)

    def append(
        self,
        split: list[str],
        group: str | None,
        outer_iter: int,
        noise_sigma: float,
        ids: np.ndarray,
        labels: np.ndarray,
        feats: np.ndarray,
    ) -> None:
        self.embedding_chunks.append(feats.astype(np.float32, copy=False))
        for one_split, idx, label in zip(split, ids, labels, strict=True):
            row = {
                "id": int(idx),
                "split": one_split,
                "label": int(label),
                "outer_iter": int(outer_iter),
                "noise_sigma": float(noise_sigma),
            }
            if self.include_group:
                row["group"] = group
            self.metadata_rows.append(row)

    def close(self) -> None:
        if not self.embedding_chunks:
            raise SystemExit("No embeddings were generated.")
        embeddings = np.concatenate(self.embedding_chunks, axis=0).astype(np.float32, copy=False)
        np.save(os.path.join(self.output, "embeddings.npy"), embeddings)
        with open(os.path.join(self.output, "metadata.csv"), "w", newline="") as f:
            fieldnames = ["id", "split"]
            if self.include_group:
                fieldnames.append("group")
            fieldnames.extend(["label", "outer_iter", "noise_sigma"])
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.metadata_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--x-labels", required=True)
    parser.add_argument("--y-labels", required=True)
    parser.add_argument("--keep-labels", default=None)
    parser.add_argument("--max-per-label", type=int, default=None)
    parser.add_argument("--noise-levels", default="0")
    parser.add_argument("--n-rep", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--pool-only", action="store_true")
    parser.add_argument("--encoder", default="dinov2", choices=["dinov2", "clip"])
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-l2-normalize", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--device",
        default="mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    meta = pd.read_csv(args.metadata)
    if "image_path" not in meta.columns or "label" not in meta.columns:
        raise SystemExit("Metadata must contain image_path and label columns.")
    if "split" in meta.columns:
        meta = meta[meta["split"] == args.split].copy()
    x_labels = parse_int_list(args.x_labels)
    y_labels = parse_int_list(args.y_labels)
    keep_labels = parse_int_list(args.keep_labels) if args.keep_labels else sorted(set(x_labels + y_labels))
    meta = select_rows(meta, keep_labels, args.max_per_label)
    if "id" not in meta.columns:
        meta["id"] = np.arange(len(meta), dtype=np.int64)

    if args.pool_only:
        group_payloads = [(None, meta)]
    else:
        group_payloads = [
            ("X", meta[meta["label"].isin(x_labels)].copy()),
            ("Y", meta[meta["label"].isin(y_labels)].copy()),
        ]

    encoder = FrozenEncoder(args.encoder, args.model_name, args.device, local_files_only=args.local_files_only)
    writer = EmbeddingWriter(args.output, include_group=not args.pool_only)
    noise_levels = parse_float_list(args.noise_levels)
    image_root = Path(args.image_root).resolve()

    for outer_iter in range(1, args.n_rep + 1):
        for sigma in noise_levels:
            for group, group_meta in group_payloads:
                dataset = FolderImageDataset(group_meta, image_root=image_root, noise_sigma=sigma, rng_seed=args.seed + 1000 * outer_iter)
                loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)
                for batch_images, batch_labels, batch_ids, batch_splits in loader:
                    feats = encoder.encode(batch_images, normalize=not args.no_l2_normalize)
                    writer.append(batch_splits, group, outer_iter, sigma, batch_ids, batch_labels, feats)
                print(
                    f"[embed] outer_iter={outer_iter}/{args.n_rep} sigma={sigma:g} "
                    f"{'pool=ALL' if group is None else f'group={group}'} rows={len(group_meta)}",
                    flush=True,
                )

    writer.close()
    print(f"Wrote noisy embeddings to {args.output}")


if __name__ == "__main__":
    main()
