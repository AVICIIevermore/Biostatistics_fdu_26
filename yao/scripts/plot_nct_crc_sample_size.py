#!/usr/bin/env python3
"""Plot sample-size comparisons for NCT-CRC-HE-100K."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


STYLE = {
    "CLIP + GAUSS5": ("#8e44ad", "D"),
    "DINOv2 + GAUSS5": ("#c0392b", "X"),
    "Raw Pixel + GAUSS5": ("#1f4e79", "o"),
    "CNN Final FC128 + GAUSS5": ("#2e8b57", "s"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def plot_panel(ax, df: pd.DataFrame, scenario: str, ylabel: str, title: str) -> None:
    sub = df[df["scenario"] == scenario].sort_values(["method_label", "sample_size"])
    order = ["Raw Pixel + GAUSS5", "CLIP + GAUSS5", "DINOv2 + GAUSS5", "CNN Final FC128 + GAUSS5"]
    for method_label in order:
        part = sub[sub["method_label"] == method_label]
        if part.empty:
            continue
        color, marker = STYLE[method_label]
        ax.plot(part["sample_size"], part["metric_mean"], color=color, marker=marker, linewidth=2.0, markersize=6.5, label=method_label)
    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xlabel("Sample size per sample", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(sorted(df["sample_size"].unique()))
    ax.set_ylim(0.0, 0.2 if scenario.startswith("type1") else 1.02)
    ax.grid(True, linestyle=":", alpha=0.5)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.summary_csv)

    fig, axes = plt.subplots(1, 3, figsize=(16.4, 4.8), dpi=180, sharex=True)
    plot_panel(axes[0], df, "power", "Power", "Power: A=(NORM, STR, LYM) vs B=(TUM, STR, LYM)")
    plot_panel(axes[1], df, "type1_a", "Type-I error", "Type-I: A vs A")
    plot_panel(axes[2], df, "type1_b", "Type-I error", "Type-I: B vs B")
    axes[1].axhline(0.05, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[2].axhline(0.05, color="black", linestyle="--", linewidth=1.0, alpha=0.7)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("NCT-CRC-HE-100K Sample-Size Comparison (No Noise)", fontsize=16, weight="bold", y=1.08)
    fig.tight_layout()

    png = args.output_dir / "nctcrc_sample_size_comparison.png"
    pdf = args.output_dir / "nctcrc_sample_size_comparison.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
