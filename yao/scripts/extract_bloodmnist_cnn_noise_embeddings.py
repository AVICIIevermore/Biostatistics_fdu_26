#!/usr/bin/env python3
"""Extract noisy BloodMNIST CNN embedding pools for MMMD testing."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np

import bloodmnist_cnn_pipeline as cnn


def parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


class EmbeddingWriter:
    def __init__(self, output: str, include_group: bool) -> None:
        self.output = output
        self.include_group = include_group
        self.embedding_chunks: list[np.ndarray] = []
        self.metadata_rows: list[dict[str, object]] = []
        os.makedirs(output, exist_ok=True)

    def append(
        self,
        split: str,
        group: str | None,
        outer_iter: int,
        noise_sigma: float,
        ids: np.ndarray,
        labels: np.ndarray,
        feats: np.ndarray,
    ) -> None:
        self.embedding_chunks.append(feats.astype(np.float32, copy=False))
        for idx, label in zip(ids, labels, strict=True):
            row = {
                "id": int(idx),
                "split": split,
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


def load_npz_split(path: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    images = data[f"{split}_images"]
    labels = data[f"{split}_labels"].reshape(-1).astype(np.int64)
    return images, labels


def select_label_pool(images: np.ndarray, labels: np.ndarray, keep_labels: list[int], max_per_label: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.where(np.isin(labels, np.asarray(keep_labels, dtype=np.int64)))[0]
    if max_per_label is not None:
        chosen: list[int] = []
        for label in sorted(set(labels[indices].tolist())):
            label_indices = indices[labels[indices] == label][:max_per_label]
            chosen.extend(label_indices.tolist())
        indices = np.asarray(chosen, dtype=np.int64)
    return images[indices], labels[indices], indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--x-labels", default="0")
    parser.add_argument("--y-labels", default="4")
    parser.add_argument("--keep-labels", default=None)
    parser.add_argument("--max-per-label", type=int, default=None)
    parser.add_argument("--noise-levels", default="0,0.2,0.4,0.6,0.8,1")
    parser.add_argument("--n-rep", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--pool-only", action="store_true")
    parser.add_argument("--layer", default="final_fc128", choices=["layer1_gap", "layer2_gap", "final_fc128"])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images, labels = load_npz_split(args.npz, args.split)
    x_labels = parse_int_list(args.x_labels)
    y_labels = parse_int_list(args.y_labels)
    keep_labels = parse_int_list(args.keep_labels) if args.keep_labels else sorted(set(x_labels + y_labels))
    all_images, all_labels, all_ids = select_label_pool(images, labels, keep_labels, args.max_per_label)

    if args.pool_only:
        group_payloads = [(None, all_images, all_labels, all_ids)]
    else:
        x_mask = np.isin(all_labels, np.asarray(x_labels, dtype=np.int64))
        y_mask = np.isin(all_labels, np.asarray(y_labels, dtype=np.int64))
        group_payloads = [
            ("X", all_images[x_mask], all_labels[x_mask], all_ids[x_mask]),
            ("Y", all_images[y_mask], all_labels[y_mask], all_ids[y_mask]),
        ]

    writer = EmbeddingWriter(args.output, include_group=not args.pool_only)
    noise_levels = parse_float_list(args.noise_levels)
    checkpoint = Path(args.checkpoint)

    for outer_iter in range(1, args.n_rep + 1):
        rng = np.random.default_rng(args.seed + 1000 * outer_iter)
        for sigma in noise_levels:
            for group, group_images, group_labels, group_ids in group_payloads:
                noisy = np.clip(
                    group_images.astype(np.float32) / 255.0
                    + rng.normal(0.0, sigma, size=group_images.shape).astype(np.float32),
                    0.0,
                    1.0,
                )
                noisy_uint8 = np.clip(np.rint(noisy * 255.0), 0, 255).astype(np.uint8)
                emb_map = cnn.extract_embeddings(
                    checkpoint_path=checkpoint,
                    images=noisy_uint8,
                    batch_size=args.batch_size,
                    layers=[args.layer],
                    force_cpu=args.cpu,
                )
                feats = emb_map[args.layer]
                writer.append(args.split, group, outer_iter, sigma, group_ids, group_labels, feats)
                print(
                    f"[cnn-embed] outer_iter={outer_iter}/{args.n_rep} sigma={sigma:g} "
                    f"{'pool=ALL' if group is None else f'group={group}'} rows={len(group_labels)}",
                    flush=True,
                )

    writer.close()
    print(f"Wrote noisy CNN embeddings to {args.output}")


if __name__ == "__main__":
    main()
