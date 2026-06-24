#!/usr/bin/env python3
"""Plot NCT-CRC lrj comparisons across multiple representations."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REP_ORDER = [
    "Raw Pixel",
    "CLIP",
    "DINOv2",
    "CNN Final FC128",
]

REP_COLORS = {
    "Raw Pixel": "#1f4e79",
    "CLIP": "#8e44ad",
    "DINOv2": "#c0392b",
    "CNN Final FC128": "#2e8b57",
}

METHOD_STYLES = {
    "GEXP-5": ("o", "-"),
    "NEW-MMMD": ("s", "--"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def plot_panel(ax, df: pd.DataFrame, scenario: str, title: str, ylabel: str) -> None:
    sub = df[df["scenario"] == scenario].copy()
    for rep in REP_ORDER:
        rep_df = sub[sub["representation"] == rep].sort_values(["method", "sample_size"])
        if rep_df.empty:
            continue
        color = REP_COLORS[rep]
        for method_name in ["GEXP-5", "NEW-MMMD"]:
            part = rep_df[rep_df["method"] == method_name].sort_values("sample_size")
            if part.empty:
                continue
            marker, linestyle = METHOD_STYLES[method_name]
            ax.plot(
                part["sample_size"],
                part["reject_rate"],
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=2.0,
                markersize=6.0,
                label=f"{rep} / {method_name}",
            )
    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xlabel("Sample size per sample", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(sorted(sub["sample_size"].unique()))
    ax.set_ylim(0.0, 0.2 if scenario.startswith("type1") else 1.02)
    ax.grid(True, linestyle=":", alpha=0.5)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.summary_csv)

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.2), dpi=180, sharex=True)
    plot_panel(axes[0], df, "power", "Power: A=(NORM, STR, LYM) vs B=(TUM, STR, LYM)", "Reject rate / power")
    plot_panel(axes[1], df, "type1_a", "Type-I: A vs A", "Type-I error")
    plot_panel(axes[2], df, "type1_b", "Type-I: B vs B", "Type-I error")
    axes[1].axhline(0.05, color="black", linestyle=":", linewidth=1.1, alpha=0.75)
    axes[2].axhline(0.05, color="black", linestyle=":", linewidth=1.1, alpha=0.75)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.12), fontsize=9)
    fig.suptitle("NCT-CRC: Representation Layer vs Testing Layer (GEXP-5 vs NEW-MMMD)", fontsize=16, weight="bold", y=1.16)
    fig.tight_layout()

    png = args.output_dir / "nctcrc_lrj_all_representations.png"
    pdf = args.output_dir / "nctcrc_lrj_all_representations.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
