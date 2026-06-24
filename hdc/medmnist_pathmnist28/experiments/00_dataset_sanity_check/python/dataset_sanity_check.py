#!/usr/bin/env python3
"""Phase 1 sanity check for PathMNIST-28.

This script only inspects the official PathMNIST npz file. It does not train a
CNN and does not run any two-sample test.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = DATASET_ROOT / "data" / "pathmnist.npz"
RESULTS_DIR = EXPERIMENT_ROOT / "Results"
LOG_DIR = EXPERIMENT_ROOT / "logs"

CLASS_NAMES = {
    0: "adipose",
    1: "background",
    2: "debris",
    3: "lymphocytes",
    4: "mucus",
    5: "smooth muscle",
    6: "normal colon mucosa",
    7: "cancer-associated stroma",
    8: "colorectal adenocarcinoma epithelium",
}

SPLITS = ["train", "val", "test"]
H1_SAMPLE_SIZES = [30, 60, 90, 120, 150]
H0_SAMPLE_SIZES = [60, 120]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_arrays(data: np.lib.npyio.NpzFile, split: str) -> tuple[np.ndarray, np.ndarray]:
    images = data[f"{split}_images"]
    labels = data[f"{split}_labels"].reshape(-1).astype(int)
    return images, labels


def summarize_data(data: np.lib.npyio.NpzFile) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary_rows: list[dict[str, object]] = []
    count_rows: list[dict[str, object]] = []

    for split in SPLITS:
        images, labels = split_arrays(data, split)
        summary_rows.append(
            {
                "split": split,
                "image_shape": "x".join(map(str, images.shape)),
                "label_shape": "x".join(map(str, data[f"{split}_labels"].shape)),
                "n_samples": images.shape[0],
                "height": images.shape[1],
                "width": images.shape[2],
                "channels": images.shape[3],
                "image_dtype": str(images.dtype),
                "label_dtype": str(data[f"{split}_labels"].dtype),
                "pixel_min": int(images.min()),
                "pixel_max": int(images.max()),
            }
        )

        unique, counts = np.unique(labels, return_counts=True)
        count_by_label = dict(zip(unique.tolist(), counts.tolist()))
        for label in range(9):
            count_rows.append(
                {
                    "split": split,
                    "label": label,
                    "class_name": CLASS_NAMES[label],
                    "count": count_by_label.get(label, 0),
                }
            )

    return summary_rows, count_rows


def feasibility_rows(test_counts: dict[int, int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for n in H1_SAMPLE_SIZES:
        per_class = n // 3
        requirements = {3: 2 * per_class, 5: 2 * per_class, 6: per_class, 8: per_class}
        feasible = all(test_counts.get(label, 0) >= required for label, required in requirements.items())
        limiting_margin = min(test_counts.get(label, 0) - required for label, required in requirements.items())
        rows.append(
            {
                "scenario": "mix635_vs_mix835",
                "null_or_alternative": "H1",
                "sample_size": n,
                "per_class_per_group": per_class,
                "required_class_3": requirements[3],
                "required_class_5": requirements[5],
                "required_class_6": requirements[6],
                "required_class_8": requirements[8],
                "available_class_3": test_counts.get(3, 0),
                "available_class_5": test_counts.get(5, 0),
                "available_class_6": test_counts.get(6, 0),
                "available_class_8": test_counts.get(8, 0),
                "feasible_disjoint": int(feasible),
                "limiting_margin": limiting_margin,
            }
        )

    for n in H0_SAMPLE_SIZES:
        per_class = n // 3
        requirements = {3: 2 * per_class, 5: 2 * per_class, 6: 2 * per_class, 8: 0}
        feasible = all(test_counts.get(label, 0) >= required for label, required in requirements.items())
        limiting_margin = min(test_counts.get(label, 0) - required for label, required in requirements.items())
        rows.append(
            {
                "scenario": "mix635_vs_mix635_null",
                "null_or_alternative": "H0",
                "sample_size": n,
                "per_class_per_group": per_class,
                "required_class_3": requirements[3],
                "required_class_5": requirements[5],
                "required_class_6": requirements[6],
                "required_class_8": requirements[8],
                "available_class_3": test_counts.get(3, 0),
                "available_class_5": test_counts.get(5, 0),
                "available_class_6": test_counts.get(6, 0),
                "available_class_8": test_counts.get(8, 0),
                "feasible_disjoint": int(feasible),
                "limiting_margin": limiting_margin,
            }
        )

    return rows


def save_montage(images: np.ndarray, labels: np.ndarray, out_path: Path, examples_per_class: int = 5) -> None:
    rng = np.random.default_rng(2026)
    fig, axes = plt.subplots(9, examples_per_class, figsize=(examples_per_class * 1.5, 12))
    for label in range(9):
        idx = np.where(labels == label)[0]
        selected = rng.choice(idx, size=min(examples_per_class, len(idx)), replace=False)
        for col in range(examples_per_class):
            ax = axes[label, col]
            ax.axis("off")
            if col < len(selected):
                ax.imshow(images[selected[col]])
            if col == 0:
                ax.set_title(f"{label}: {CLASS_NAMES[label]}", fontsize=8, loc="left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    data = np.load(DATA_PATH)
    summary_rows, count_rows = summarize_data(data)

    write_csv(
        RESULTS_DIR / "data_summary.csv",
        [
            "split",
            "image_shape",
            "label_shape",
            "n_samples",
            "height",
            "width",
            "channels",
            "image_dtype",
            "label_dtype",
            "pixel_min",
            "pixel_max",
        ],
        summary_rows,
    )
    write_csv(
        RESULTS_DIR / "pathmnist_class_counts.csv",
        ["split", "label", "class_name", "count"],
        count_rows,
    )

    test_images, test_labels = split_arrays(data, "test")
    test_counts = {row["label"]: row["count"] for row in count_rows if row["split"] == "test"}
    feasibility = feasibility_rows(test_counts)
    write_csv(
        RESULTS_DIR / "sampling_feasibility.csv",
        [
            "scenario",
            "null_or_alternative",
            "sample_size",
            "per_class_per_group",
            "required_class_3",
            "required_class_5",
            "required_class_6",
            "required_class_8",
            "available_class_3",
            "available_class_5",
            "available_class_6",
            "available_class_8",
            "feasible_disjoint",
            "limiting_margin",
        ],
        feasibility,
    )
    save_montage(test_images, test_labels, RESULTS_DIR / "pathmnist_examples.png")

    all_feasible = all(row["feasible_disjoint"] == 1 for row in feasibility)
    lines = [
        "PathMNIST-28 Phase 1 sanity check",
        f"data_path: {DATA_PATH}",
        "",
        "Split shapes:",
    ]
    for row in summary_rows:
        lines.append(f"- {row['split']}: images={row['image_shape']}, labels={row['label_shape']}, dtype={row['image_dtype']}")
    lines.extend(["", "Target test class counts:"])
    for label in [3, 5, 6, 8]:
        lines.append(f"- {label} ({CLASS_NAMES[label]}): {test_counts.get(label, 0)}")
    lines.extend(["", f"All planned disjoint sample sizes feasible: {all_feasible}"])
    for row in feasibility:
        lines.append(
            f"- {row['scenario']} n={row['sample_size']}: feasible={bool(row['feasible_disjoint'])}, "
            f"limiting_margin={row['limiting_margin']}"
        )

    log_text = "\n".join(lines) + "\n"
    (LOG_DIR / "dataset_sanity_check.log").write_text(log_text)
    print(log_text)


if __name__ == "__main__":
    main()
