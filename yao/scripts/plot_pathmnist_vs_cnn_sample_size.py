#!/usr/bin/env python3
"""Plot PathMNIST sample-size comparisons: CNN baselines vs DINOv2/CLIP."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


POWER_LABELS = {
    "raw_pixel_gaussian5": "Raw Pixel + GAUSS5",
    "cnn_final_fc128_gaussian5": "CNN Final FC128 + GAUSS5",
    "cnn_multilayer_single_gaussian": "CNN Multilayer Single",
    "cnn_multilayer_gaussian15": "CNN Multilayer Gaussian15",
}

COLORS = {
    "Raw Pixel + GAUSS5": "#1f4e79",
    "CNN Final FC128 + GAUSS5": "#d35400",
    "CNN Multilayer Single": "#2e8b57",
    "CNN Multilayer Gaussian15": "#7b5ea7",
    "DINOV2 + GAUSS5": "#c0392b",
    "CLIP + GAUSS5": "#8e44ad",
}

MARKERS = {
    "Raw Pixel + GAUSS5": "o",
    "CNN Final FC128 + GAUSS5": "s",
    "CNN Multilayer Single": "^",
    "CNN Multilayer Gaussian15": "D",
    "DINOV2 + GAUSS5": "X",
    "CLIP + GAUSS5": "P",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn-power", required=True, type=Path)
    parser.add_argument("--cnn-type1", required=True, type=Path)
    parser.add_argument("--dinov2", required=True, type=Path)
    parser.add_argument("--clip", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def prep_cnn_power(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    df["method_label"] = df["method"].map(POWER_LABELS)
    df["scenario"] = "power"
    df["metric_mean"] = df["rejection_rate"]
    return df[["scenario", "sample_size", "method_label", "metric_mean"]]


def prep_cnn_type1(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    df["method_label"] = df["method"].map(POWER_LABELS)
    df["scenario"] = "type1"
    df["metric_mean"] = df["type1_error"]
    return df[["scenario", "sample_size", "method_label", "metric_mean"]]


def prep_embed(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    return df[["scenario", "sample_size", "method_label", "metric_mean"]]


def plot_panel(ax, df: pd.DataFrame, scenario: str, title: str, ylabel: str) -> None:
    part = df[df["scenario"] == scenario].copy()
    for method_label in [
        "Raw Pixel + GAUSS5",
        "CNN Final FC128 + GAUSS5",
        "CNN Multilayer Single",
        "CNN Multilayer Gaussian15",
        "DINOV2 + GAUSS5",
        "CLIP + GAUSS5",
    ]:
        sub = part[part["method_label"] == method_label].sort_values("sample_size")
        if sub.empty:
            continue
        ax.plot(
            sub["sample_size"],
            sub["metric_mean"],
            label=method_label,
            color=COLORS[method_label],
            marker=MARKERS[method_label],
            linewidth=2.0,
            markersize=6.5,
        )
    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xlabel("Sample size per sample", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_ylim(-0.02, 1.02)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cnn_power = prep_cnn_power(args.cnn_power)
    cnn_type1 = prep_cnn_type1(args.cnn_type1)
    dino = prep_embed(args.dinov2)
    clip = prep_embed(args.clip)
    df = pd.concat([cnn_power, cnn_type1, dino, clip], ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.0), dpi=180)
    plot_panel(axes[0], df, "power", "Power: mix635 vs mix835", "Power")
    plot_panel(axes[1], df, "type1", "Type-I: mix635 vs mix635", "Type-I error")
    axes[1].axhline(0.05, color="black", linestyle="--", linewidth=1, alpha=0.7)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("PathMNIST-28: CNN Baselines vs DINOv2 / CLIP", fontsize=16, weight="bold", y=1.12)
    fig.tight_layout()

    png = args.output_dir / "pathmnist_sample_size_vs_cnn.png"
    pdf = args.output_dir / "pathmnist_sample_size_vs_cnn.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
