#!/usr/bin/env python3
"""Plot MNIST DINOv2 MMMD power against Dechao's CNN embedding baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn-summary", required=True, type=Path)
    parser.add_argument("--dinov2-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cnn = pd.read_csv(args.cnn_summary)
    cnn = cnn[["method", "method_label", "noise_sigma", "power_mean", "power_se", "n_rep"]]

    dino = pd.read_csv(args.dinov2_summary)
    dino = dino.assign(
        method="dinov2_base_gaussian5",
        method_label="DINOv2 Base Gaussian-5 (768-d)",
    )
    dino = dino[["method", "method_label", "noise_sigma", "power_mean", "power_se", "n_rep"]]

    combined = pd.concat([cnn, dino], ignore_index=True)
    combined.to_csv(args.output_dir / "mnist_dinov2_vs_cnn_power_summary.csv", index=False)

    labels = list(cnn["method_label"].drop_duplicates()) + ["DINOv2 Base Gaussian-5 (768-d)"]
    colors = {
        "Raw Pixel Gaussian-5 (784-d)": "#111111",
        "Layer1 GAP Gaussian-5 (32-d)": "#4C78A8",
        "Layer1 2x2 Gaussian-5 (128-d)": "#C9762B",
        "Layer2 GAP Gaussian-5 (64-d)": "#58A48E",
        "Final FC Gaussian-5 (128-d)": "#B87A9B",
        "Multilayer Single Gaussian (3 comps)": "#D4A72C",
        "Multilayer Gaussian-15 (15 comps)": "#79AEDA",
        "DINOv2 Base Gaussian-5 (768-d)": "#D43F3A",
    }
    markers = ["o", "^", "s", "D", "o", "*", "+", "X"]

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=180)
    for marker, label in zip(markers, labels):
        part = combined[combined["method_label"] == label].sort_values("noise_sigma")
        if part.empty:
            continue
        ax.plot(
            part["noise_sigma"],
            part["power_mean"],
            label=label,
            color=colors.get(label),
            marker=marker,
            linewidth=2.0 if label.startswith("DINOv2") else 1.7,
            markersize=7.5 if label.startswith("DINOv2") else 6,
        )

    ax.set_title("MNIST DINOv2 vs CNN Embedding MMMD Power Comparison", fontsize=17, weight="bold")
    ax.set_xlabel("Noise sigma", fontsize=13)
    ax.set_ylabel("Power / rejection rate", fontsize=13)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(-0.04, 1.04)
    ax.grid(True, linestyle=":", linewidth=1.0, alpha=0.55)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=10.5)
    fig.tight_layout()

    png_path = args.output_dir / "mnist_dinov2_vs_cnn_power_comparison.png"
    pdf_path = args.output_dir / "mnist_dinov2_vs_cnn_power_comparison.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
