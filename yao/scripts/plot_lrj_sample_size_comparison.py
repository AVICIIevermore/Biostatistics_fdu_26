#!/usr/bin/env python3
"""Plot formal lrj sample-size comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


STYLE = {
    "GEXP-5": ("#1f4e79", "o"),
    "NEW-MMMD": ("#c0392b", "X"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--title", default="NCT-CRC + DINOv2: GEXP-5 vs NEW-MMMD")
    return parser.parse_args()


def plot_panel(ax, df: pd.DataFrame, scenario: str, ylabel: str, title: str, value_col: str) -> None:
    sub = df[df["scenario"] == scenario].sort_values(["method", "sample_size"])
    for method in ["GEXP-5", "NEW-MMMD"]:
        part = sub[sub["method"] == method]
        if part.empty:
            continue
        color, marker = STYLE[method]
        ax.plot(part["sample_size"], part[value_col], color=color, marker=marker, linewidth=2.0, markersize=6.5, label=method)
    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xlabel("Sample size per sample", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(sorted(df["sample_size"].unique()))
    ax.grid(True, linestyle=":", alpha=0.5)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.summary_csv)

    fig, axes = plt.subplots(1, 3, figsize=(16.4, 4.8), dpi=180, sharex=True)
    plot_panel(axes[0], df, "power", "Rejection rate", "Power", "reject_rate")
    plot_panel(axes[1], df, "type1_a", "Rejection rate", "Type-I: A vs A", "reject_rate")
    plot_panel(axes[2], df, "type1_b", "Rejection rate", "Type-I: B vs B", "reject_rate")
    axes[1].axhline(0.05, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[2].axhline(0.05, color="black", linestyle="--", linewidth=1.0, alpha=0.7)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle(args.title, fontsize=16, weight="bold", y=1.08)
    fig.tight_layout()

    png = args.output_dir / "lrj_sample_size_comparison.png"
    pdf = args.output_dir / "lrj_sample_size_comparison.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
