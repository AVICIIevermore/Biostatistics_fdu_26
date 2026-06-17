#!/usr/bin/env python3
"""Plot power and Type-I error against sample size for raw pixel, DINOv2, and CLIP."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


STYLE = {
    "CLIP + GAUSS5": ("#8e44ad", "D"),
    "DINOv2 + GAUSS5": ("#c0392b", "X"),
    "Raw Pixel + GAUSS5": ("#1f4e79", "o"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--title-suffix", default="sigma=0.6")
    return parser.parse_args()


def plot_panel(ax, df: pd.DataFrame, scenario: str, ylabel: str, title: str) -> None:
    sub = df[df["scenario"] == scenario].sort_values(["method_label", "sample_size"])
    for method_label in ["CLIP + GAUSS5", "DINOv2 + GAUSS5", "Raw Pixel + GAUSS5"]:
        part = sub[sub["method_label"] == method_label]
        if part.empty:
            continue
        color, marker = STYLE[method_label]
        ax.plot(part["sample_size"], part["metric_mean"], color=color, marker=marker, linewidth=2.0, markersize=6.5, label=method_label)
    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xlabel("Sample size per sample", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(sorted(df["sample_size"].unique()))
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=":", alpha=0.5)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.summary_csv)

    fig, axes = plt.subplots(1, 3, figsize=(16.4, 4.8), dpi=180, sharex=True)
    plot_panel(axes[0], df, "power", "Power", "Power: A=(0,1,3) vs B=(0,1,6)")
    plot_panel(axes[1], df, "type1_a", "Type-I error", "Type-I: A vs A")
    plot_panel(axes[2], df, "type1_b", "Type-I error", "Type-I: B vs B")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle(f"BloodMNIST Mixed-Population Sample-Size Comparison ({args.title_suffix})", fontsize=16, weight="bold", y=1.08)
    fig.tight_layout()

    png = args.output_dir / "bloodmnist_mixed_sample_size_comparison.png"
    pdf = args.output_dir / "bloodmnist_mixed_sample_size_comparison.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
