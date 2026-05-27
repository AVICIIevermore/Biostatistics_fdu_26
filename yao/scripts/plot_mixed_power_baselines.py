#!/usr/bin/env python3
"""Plot mixed-population BloodMNIST power comparison across paper baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


LABELS = {
    "GAUSS1": "Gaussian MMD (single)",
    "LAP1": "Laplace MMD (single)",
    "GAUSS5": "Gaussian MMMD (5 kernels)",
    "LAP5": "Laplace MMMD (5 kernels)",
    "MIXED": "Mixed MMMD",
}

COLORS = {
    "GAUSS1": "#c03a2b",
    "LAP1": "#2e8b57",
    "GAUSS5": "#1f4e79",
    "LAP5": "#d17c00",
    "MIXED": "#7b5ea7",
}

MARKERS = {
    "GAUSS1": "o",
    "LAP1": "^",
    "GAUSS5": "X",
    "LAP5": "s",
    "MIXED": "D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.summary)
    df = df.sort_values(["method", "noise_sigma"])

    fig, ax = plt.subplots(figsize=(10.5, 6.4), dpi=180)
    for method in ["GAUSS1", "LAP1", "GAUSS5", "LAP5", "MIXED"]:
        part = df[df["method"] == method].sort_values("noise_sigma")
        if part.empty:
            continue
        ax.plot(
            part["noise_sigma"],
            part["power_mean"],
            label=LABELS[method],
            color=COLORS[method],
            marker=MARKERS[method],
            linewidth=2.2 if method == "GAUSS5" else 1.8,
            markersize=7 if method == "GAUSS5" else 6,
        )

    ax.set_title("BloodMNIST Mixed-Population Power: DINOv2 + Paper Baselines", fontsize=16, weight="bold")
    ax.set_xlabel("Noise sigma", fontsize=13)
    ax.set_ylabel("Power / rejection rate", fontsize=13)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=10.5)
    fig.tight_layout()

    png = args.output_dir / "bloodmnist_mixed_power_baselines.png"
    pdf = args.output_dir / "bloodmnist_mixed_power_baselines.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
