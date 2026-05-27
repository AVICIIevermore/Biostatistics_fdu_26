#!/usr/bin/env python3
"""Dataset split sanity check for PathMNIST within-class center-shift MMMD."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SHARED_PY = PROJECT_ROOT / "shared_model" / "python"
sys.path.insert(0, str(SHARED_PY))

from pathmnist_center_shift_cnn_pipeline import (  # noqa: E402
    CLASS_NAMES,
    DEFAULT_CONFIG_DIR,
    DEFAULT_DATA_PATH,
    class_count_rows,
    make_center_shift_split,
    save_split_indices,
)

RESULTS_DIR = EXPERIMENT_ROOT / "Results"
LOG_DIR = EXPERIMENT_ROOT / "logs"
TARGET_LABELS = [6, 8]
N_GRID = [20, 30, 50, 80, 100]
SETTINGS = ["H1_source_vs_external", "H0_source", "H0_external"]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_raw_npz(data_path: Path) -> dict[str, np.ndarray]:
    data = np.load(data_path)
    return {
        "train_images": data["train_images"],
        "train_labels": data["train_labels"].reshape(-1).astype(int),
        "val_images": data["val_images"],
        "val_labels": data["val_labels"].reshape(-1).astype(int),
        "test_images": data["test_images"],
        "test_labels": data["test_labels"].reshape(-1).astype(int),
    }


def data_summary_rows(raw: dict[str, np.ndarray], split: dict[str, np.ndarray]) -> list[dict[str, object]]:
    pools = {
        "cnn_train_pool": (raw["train_images"][split["cnn_train_indices"]], raw["train_labels"][split["cnn_train_indices"]]),
        "source_holdout_pool": (raw["train_images"][split["source_holdout_indices"]], raw["train_labels"][split["source_holdout_indices"]]),
        "val": (raw["val_images"], raw["val_labels"]),
        "external_pool": (raw["test_images"], raw["test_labels"]),
    }
    rows = []
    for name, (images, labels) in pools.items():
        rows.append(
            {
                "split": name,
                "image_shape": "x".join(map(str, images.shape)),
                "label_shape": "x".join(map(str, labels.shape)),
                "n_samples": int(images.shape[0]),
                "height": int(images.shape[1]),
                "width": int(images.shape[2]),
                "channels": int(images.shape[3]),
                "image_dtype": str(images.dtype),
                "pixel_min": int(images.min()),
                "pixel_max": int(images.max()),
            }
        )
    return rows


def feasibility_rows(counts: dict[tuple[str, int], int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label in TARGET_LABELS:
        source_count = counts[("source_holdout_pool", label)]
        external_count = counts[("external_pool", label)]
        for setting in SETTINGS:
            for n in N_GRID:
                if setting == "H1_source_vs_external":
                    feasible = n <= min(source_count, external_count)
                    required_source = n
                    required_external = n
                    limiting_margin = min(source_count - n, external_count - n)
                elif setting == "H0_source":
                    feasible = 2 * n <= source_count
                    required_source = 2 * n
                    required_external = 0
                    limiting_margin = source_count - 2 * n
                elif setting == "H0_external":
                    feasible = 2 * n <= external_count
                    required_source = 0
                    required_external = 2 * n
                    limiting_margin = external_count - 2 * n
                else:
                    raise ValueError(setting)
                rows.append(
                    {
                        "label": label,
                        "label_name": CLASS_NAMES[label],
                        "setting": setting,
                        "n": n,
                        "source_count_available": source_count,
                        "external_count_available": external_count,
                        "required_source_count": required_source,
                        "required_external_count": required_external,
                        "feasible_disjoint": int(feasible),
                        "limiting_margin": int(limiting_margin),
                        "skip_reason": "" if feasible else "insufficient disjoint samples",
                    }
                )
    return rows


def save_examples(raw: dict[str, np.ndarray], split: dict[str, np.ndarray], out_path: Path) -> None:
    rng = np.random.default_rng(2026)
    source_images = raw["train_images"][split["source_holdout_indices"]]
    source_labels = raw["train_labels"][split["source_holdout_indices"]]
    external_images = raw["test_images"]
    external_labels = raw["test_labels"]

    fig, axes = plt.subplots(len(TARGET_LABELS), 8, figsize=(12, 4.8))
    for row, label in enumerate(TARGET_LABELS):
        source_idx = np.where(source_labels == label)[0]
        external_idx = np.where(external_labels == label)[0]
        source_sel = rng.choice(source_idx, size=4, replace=False)
        external_sel = rng.choice(external_idx, size=4, replace=False)
        for col in range(8):
            ax = axes[row, col]
            ax.axis("off")
            if col < 4:
                ax.imshow(source_images[source_sel[col]])
                title = "source" if col == 0 else ""
            else:
                ax.imshow(external_images[external_sel[col - 4]])
                title = "external" if col == 4 else ""
            if title:
                ax.set_title(f"{label} {title}", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    seed = 2026
    holdout_fraction = 0.20
    data_path = DEFAULT_DATA_PATH
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw_npz(data_path)
    split = make_center_shift_split(raw["train_labels"], holdout_fraction=holdout_fraction, seed=seed)
    split_indices_path = save_split_indices(DEFAULT_CONFIG_DIR, split, holdout_fraction, seed)

    labels_by_pool = {
        "cnn_train_pool": raw["train_labels"][split["cnn_train_indices"]],
        "source_holdout_pool": raw["train_labels"][split["source_holdout_indices"]],
        "val": raw["val_labels"],
        "external_pool": raw["test_labels"],
    }
    count_rows = class_count_rows(labels_by_pool)
    counts = {(row["split"], int(row["label"])): int(row["count"]) for row in count_rows}
    feasibility = feasibility_rows(counts)

    write_csv(
        RESULTS_DIR / "data_summary.csv",
        ["split", "image_shape", "label_shape", "n_samples", "height", "width", "channels", "image_dtype", "pixel_min", "pixel_max"],
        data_summary_rows(raw, split),
    )
    write_csv(
        RESULTS_DIR / "pathmnist_class_counts.csv",
        ["split", "label", "label_name", "count"],
        count_rows,
    )
    write_csv(
        RESULTS_DIR / "sampling_feasibility.csv",
        [
            "label",
            "label_name",
            "setting",
            "n",
            "source_count_available",
            "external_count_available",
            "required_source_count",
            "required_external_count",
            "feasible_disjoint",
            "limiting_margin",
            "skip_reason",
        ],
        feasibility,
    )
    save_examples(raw, split, RESULTS_DIR / "pathmnist_center_shift_examples.png")

    all_feasible = all(row["feasible_disjoint"] == 1 for row in feasibility)
    lines = [
        "PathMNIST center-shift dataset split sanity check",
        f"data_path: {data_path}",
        f"split_indices_path: {split_indices_path}",
        f"seed: {seed}",
        f"holdout_fraction: {holdout_fraction}",
        "",
        "Target counts:",
    ]
    for label in TARGET_LABELS:
        lines.append(
            f"- label {label} ({CLASS_NAMES[label]}): source_holdout={counts[('source_holdout_pool', label)]}, "
            f"external={counts[('external_pool', label)]}"
        )
    lines.extend(["", f"All planned label/setting/n cells feasible: {all_feasible}", ""])
    for row in feasibility:
        if row["label"] in TARGET_LABELS and row["n"] in [20, 100]:
            lines.append(
                f"- label={row['label']} setting={row['setting']} n={row['n']}: "
                f"feasible={bool(row['feasible_disjoint'])}, margin={row['limiting_margin']}"
            )
    log_text = "\n".join(lines) + "\n"
    (LOG_DIR / "dataset_split_sanity_check.log").write_text(log_text)
    print(log_text)


if __name__ == "__main__":
    main()
