#!/usr/bin/env python3
"""Plot raw pixel vs DINOv2 GAUSS5 comparisons for mixed BloodMNIST experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_summary(path: Path, metric_col: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    df["value"] = df[metric_col]
    return df[["noise_sigma", "value"]]


def plot_panel(ax, dino: pd.DataFrame, raw: pd.DataFrame, title: str, ylabel: str) -> None:
    ax.plot(dino["noise_sigma"], dino["value"], color="#c0392b", marker="X", linewidth=2.2, markersize=7, label="DINOv2 + GAUSS5")
    ax.plot(raw["noise_sigma"], raw["value"], color="#1f4e79", marker="o", linewidth=2.0, markersize=6, label="Raw Pixel + GAUSS5")
    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xlabel("Noise sigma", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=":", alpha=0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dino-power", required=True, type=Path)
    parser.add_argument("--raw-power", required=True, type=Path)
    parser.add_argument("--dino-type1-a", required=True, type=Path)
    parser.add_argument("--raw-type1-a", required=True, type=Path)
    parser.add_argument("--dino-type1-b", required=True, type=Path)
    parser.add_argument("--raw-type1-b", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), dpi=180, sharex=True)

    plot_panel(
        axes[0],
        load_summary(args.dino_power, "power_mean"),
        load_summary(args.raw_power, "power_mean"),
        "Power: A=(0,1,3) vs B=(0,1,6)",
        "Power",
    )
    plot_panel(
        axes[1],
        load_summary(args.dino_type1_a, "type1_mean"),
        load_summary(args.raw_type1_a, "type1_mean"),
        "Type-I: A vs A",
        "Type-I error",
    )
    plot_panel(
        axes[2],
        load_summary(args.dino_type1_b, "type1_mean"),
        load_summary(args.raw_type1_b, "type1_mean"),
        "Type-I: B vs B",
        "Type-I error",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("BloodMNIST Mixed-Population: Raw Pixel vs DINOv2 (GAUSS5)", fontsize=16, weight="bold", y=1.08)
    fig.tight_layout()

    png = args.output_dir / "bloodmnist_mixed_rawpixel_vs_dinov2_gauss5.png"
    pdf = args.output_dir / "bloodmnist_mixed_rawpixel_vs_dinov2_gauss5.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
