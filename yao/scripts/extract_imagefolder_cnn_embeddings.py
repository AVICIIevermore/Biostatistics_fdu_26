#!/usr/bin/env python3
"""Extract shared CNN embedding pools from a local image-folder dataset."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from imagefolder_cnn_pipeline import extract_embeddings


def parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


class FolderNoisyDataset(Dataset):
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
        x = torch.from_numpy(arr.transpose(2, 0, 1))
        return x, int(row["label"]), int(row["id"]), str(row["split"])


def collate_batch(batch):
    xs, labels, ids, splits = zip(*batch)
    return torch.stack(xs, dim=0), np.asarray(labels, dtype=np.int64), np.asarray(ids, dtype=np.int64), list(splits)


class EmbeddingWriter:
    def __init__(self, output: str, include_group: bool) -> None:
        self.output = Path(output)
        self.include_group = include_group
        self.embedding_chunks: list[np.ndarray] = []
        self.metadata_rows: list[dict[str, object]] = []
        self.output.mkdir(parents=True, exist_ok=True)

    def append(self, splits: list[str], group: str | None, outer_iter: int, noise_sigma: float, ids: np.ndarray, labels: np.ndarray, feats: np.ndarray) -> None:
        self.embedding_chunks.append(feats.astype(np.float32, copy=False))
        for one_split, idx, label in zip(splits, ids, labels, strict=True):
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
        embeddings = np.concatenate(self.embedding_chunks, axis=0).astype(np.float32, copy=False)
        np.save(self.output / "embeddings.npy", embeddings)
        with (self.output / "metadata.csv").open("w", newline="") as f:
            fieldnames = ["id", "split"]
            if self.include_group:
                fieldnames.append("group")
            fieldnames.extend(["label", "outer_iter", "noise_sigma"])
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.metadata_rows)


def select_rows(meta: pd.DataFrame, keep_labels: list[int], max_per_label: int | None) -> pd.DataFrame:
    sub = meta[meta["label"].isin(keep_labels)].copy()
    if max_per_label is not None:
        sub = sub.sort_values(["label", "id"]).groupby("label", group_keys=False).head(max_per_label).reset_index(drop=True)
    return sub.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--x-labels", required=True)
    parser.add_argument("--y-labels", required=True)
    parser.add_argument("--keep-labels", default=None)
    parser.add_argument("--max-per-label", type=int, default=None)
    parser.add_argument("--noise-levels", default="0")
    parser.add_argument("--n-rep", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--pool-only", action="store_true")
    parser.add_argument("--layer", default="final_fc128", choices=["layer1_gap", "layer2_gap", "final_fc128"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cpu", action="store_true")
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

    if args.pool_only:
        group_payloads = [(None, meta)]
    else:
        group_payloads = [
            ("X", meta[meta["label"].isin(x_labels)].copy()),
            ("Y", meta[meta["label"].isin(y_labels)].copy()),
        ]

    writer = EmbeddingWriter(args.output, include_group=not args.pool_only)
    noise_levels = [float(x) for x in args.noise_levels.split(",") if x.strip()]
    image_root = Path(args.image_root).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()

    for outer_iter in range(1, args.n_rep + 1):
        for sigma in noise_levels:
            for group, group_meta in group_payloads:
                dataset = FolderNoisyDataset(group_meta, image_root=image_root, noise_sigma=sigma, rng_seed=args.seed + 1000 * outer_iter)
                loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)
                feats = extract_embeddings(checkpoint_path, loader, layer_name=args.layer, force_cpu=args.cpu)
                ids = group_meta["id"].to_numpy(dtype=np.int64)
                labels = group_meta["label"].to_numpy(dtype=np.int64)
                splits = group_meta["split"].astype(str).tolist()
                writer.append(splits, group, outer_iter, sigma, ids, labels, feats)
                print(
                    f"[cnn-embed] outer_iter={outer_iter}/{args.n_rep} sigma={sigma:g} "
                    f"{'pool=ALL' if group is None else f'group={group}'} rows={len(group_meta)}",
                    flush=True,
                )

    writer.close()
    print(f"Wrote noisy CNN embeddings to {args.output}")


if __name__ == "__main__":
    main()
