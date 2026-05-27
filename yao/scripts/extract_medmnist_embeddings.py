#!/usr/bin/env python3
"""Extract frozen CLIP or DINOv2 embeddings for a MedMNIST 2D dataset.

The output is a flat CSV with columns:
id, split, label, feat_0000, feat_0001, ...

Example:
python scripts/extract_medmnist_embeddings.py \
  --dataset pathmnist --size 224 --split test \
  --encoder dinov2 --output data/embeddings/pathmnist_dinov2.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


@dataclass
class ImageRecord:
    image: Image.Image
    label: int
    idx: int


class MedMNISTImageDataset(Dataset):
    def __init__(self, dataset_name: str, split: str, size: int, data_root: str) -> None:
        try:
            import medmnist
            from medmnist import INFO
        except ImportError as exc:
            raise SystemExit("Missing package 'medmnist'. Install with: pip install medmnist") from exc

        key = dataset_name.lower()
        if key not in INFO:
            raise SystemExit(f"Unknown MedMNIST dataset '{dataset_name}'. Available keys include: {', '.join(sorted(INFO)[:8])} ...")

        info = INFO[key]
        dataset_class = getattr(medmnist, info["python_class"])
        os.makedirs(data_root, exist_ok=True)
        self.dataset = dataset_class(split=split, root=data_root, size=size, download=True)
        self.split = split

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> ImageRecord:
        image, target = self.dataset[idx]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        image = image.convert("RGB")
        label_arr = np.asarray(target).reshape(-1)
        label = int(label_arr[0])
        return ImageRecord(image=image, label=label, idx=idx)


def collate_records(batch: list[ImageRecord]) -> tuple[list[Image.Image], np.ndarray, np.ndarray]:
    return (
        [item.image for item in batch],
        np.asarray([item.label for item in batch], dtype=np.int64),
        np.asarray([item.idx for item in batch], dtype=np.int64),
    )


class FrozenEncoder:
    def __init__(self, encoder: str, model_name: str | None, device: str, local_files_only: bool = False) -> None:
        try:
            from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise SystemExit("Missing package 'transformers'. Install with: pip install transformers") from exc

        self.encoder = encoder
        self.device = torch.device(device)

        if encoder == "clip":
            self.model_name = model_name or "openai/clip-vit-base-patch32"
            self.processor = CLIPProcessor.from_pretrained(self.model_name, local_files_only=local_files_only)
            self.model = CLIPModel.from_pretrained(self.model_name, local_files_only=local_files_only).to(self.device).eval()
        elif encoder == "dinov2":
            self.model_name = model_name or "facebook/dinov2-base"
            self.processor = AutoImageProcessor.from_pretrained(self.model_name, local_files_only=local_files_only)
            self.model = AutoModel.from_pretrained(self.model_name, local_files_only=local_files_only).to(self.device).eval()
        else:
            raise ValueError(f"Unsupported encoder: {encoder}")

    @torch.no_grad()
    def encode(self, images: list[Image.Image], normalize: bool = True) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        if self.encoder == "clip":
            feats = self.model.get_image_features(**inputs)
        else:
            out = self.model(**inputs)
            feats = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0]

        if normalize:
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        return feats.detach().cpu().numpy().astype(np.float32)


def write_embedding_csv(path: str, rows: Iterable[tuple[int, str, int, np.ndarray]]) -> None:
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    rows = list(rows)
    if not rows:
        raise SystemExit("No rows to write.")
    dim = int(rows[0][3].shape[0])
    header = ["id", "split", "label"] + [f"feat_{i:04d}" for i in range(dim)]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for idx, split, label, feat in rows:
            writer.writerow([idx, split, label, *feat.tolist()])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="pathmnist", help="MedMNIST dataset key, e.g. pathmnist, bloodmnist, dermamnist.")
    parser.add_argument("--size", type=int, default=224, choices=[28, 64, 128, 224], help="MedMNIST image size.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split to embed.")
    parser.add_argument("--data-root", default="data/medmnist", help="Where MedMNIST downloads are cached.")
    parser.add_argument("--encoder", default="dinov2", choices=["dinov2", "clip"], help="Frozen image encoder.")
    parser.add_argument("--model-name", default=None, help="Optional Hugging Face model id override.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument("--no-l2-normalize", action="store_true", help="Disable L2 normalization of embeddings.")
    parser.add_argument("--local-files-only", action="store_true", help="Use only locally cached Hugging Face model files.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = MedMNISTImageDataset(args.dataset, args.split, args.size, args.data_root)
    if args.max_images is not None:
        indices = list(range(min(args.max_images, len(dataset))))
        subset = torch.utils.data.Subset(dataset, indices)
    else:
        subset = dataset

    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_records)
    encoder = FrozenEncoder(args.encoder, args.model_name, args.device, local_files_only=args.local_files_only)

    rows: list[tuple[int, str, int, np.ndarray]] = []
    for batch_id, (images, labels, ids) in enumerate(loader, start=1):
        feats = encoder.encode(images, normalize=not args.no_l2_normalize)
        for idx, label, feat in zip(ids, labels, feats, strict=True):
            rows.append((int(idx), args.split, int(label), feat))
        if batch_id % 10 == 0:
            print(f"[embed] batches={batch_id} rows={len(rows)} dim={feats.shape[1]}", flush=True)

    write_embedding_csv(args.output, rows)
    print(f"Wrote {len(rows)} embeddings to {args.output}")


if __name__ == "__main__":
    main()
