#!/usr/bin/env python3
"""Extract CLIP or DINOv2 embeddings from an image manifest CSV.

The manifest should contain at least:
image_path,label

Optional:
id,split

This is useful for BBBC021 after preparing a metadata table that points to
merged RGB/TIFF images or pre-rendered channel composites.
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from extract_medmnist_embeddings import FrozenEncoder, write_embedding_csv


class ManifestImageDataset(Dataset):
    def __init__(self, manifest: str, image_root: str | None, split: str | None) -> None:
        self.manifest = manifest
        self.image_root = image_root
        with open(manifest, newline="") as f:
            reader = csv.DictReader(f)
            if "image_path" not in reader.fieldnames or "label" not in reader.fieldnames:
                raise SystemExit("Manifest must contain image_path and label columns.")
            rows = list(reader)

        if split is not None and "split" in (reader.fieldnames or []):
            rows = [row for row in rows if row.get("split") == split]
        self.rows = rows
        self.split = split or "all"

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        path = row["image_path"]
        if self.image_root and not os.path.isabs(path):
            path = os.path.join(self.image_root, path)
        image = Image.open(path).convert("RGB")
        label = int(row["label"])
        sample_id = int(row["id"]) if row.get("id", "").strip() else idx
        split = row.get("split", self.split)
        return image, label, sample_id, split


def collate_manifest(batch):
    images, labels, ids, splits = zip(*batch)
    return list(images), np.asarray(labels, dtype=np.int64), np.asarray(ids, dtype=np.int64), list(splits)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--encoder", default="dinov2", choices=["dinov2", "clip"])
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--no-l2-normalize", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = ManifestImageDataset(args.manifest, args.image_root, args.split)
    if args.max_images is not None:
        dataset.rows = dataset.rows[:args.max_images]
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_manifest)
    encoder = FrozenEncoder(args.encoder, args.model_name, args.device, local_files_only=args.local_files_only)

    rows = []
    for batch_id, (images, labels, ids, splits) in enumerate(loader, start=1):
        feats = encoder.encode(images, normalize=not args.no_l2_normalize)
        for idx, split, label, feat in zip(ids, splits, labels, feats, strict=True):
            rows.append((int(idx), split, int(label), feat))
        if batch_id % 10 == 0:
            print(f"[embed] batches={batch_id} rows={len(rows)} dim={feats.shape[1]}", flush=True)

    write_embedding_csv(args.output, rows)
    print(f"Wrote {len(rows)} embeddings to {args.output}")


if __name__ == "__main__":
    main()
