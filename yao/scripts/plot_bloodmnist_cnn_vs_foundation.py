#!/usr/bin/env python3
"""Plot BloodMNIST mixed-population comparison for Raw Pixel, DINOv2, CLIP, and CNN."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ORDER = [
    "Raw Pixel + GAUSS5",
    "CLIP + GAUSS5",
    "DINOv2 + GAUSS5",
    "CNN Final FC128 + GAUSS5",
]

COLORS = {
    "Raw Pixel + GAUSS5": "#1f4e79",
    "CLIP + GAUSS5": "#8e44ad",
    "DINOv2 + GAUSS5": "#c0392b",
    "CNN Final FC128 + GAUSS5": "#2e8b57",
}

MARKERS = {
    "Raw Pixel + GAUSS5": "o",
    "CLIP + GAUSS5": "D",
    "DINOv2 + GAUSS5": "X",
    "CNN Final FC128 + GAUSS5": "s",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-power", required=True, type=Path)
    parser.add_argument("--raw-type1a", required=True, type=Path)
    parser.add_argument("--raw-type1b", required=True, type=Path)
    parser.add_argument("--dino-power", required=True, type=Path)
    parser.add_argument("--dino-type1a", required=True, type=Path)
    parser.add_argument("--dino-type1b", required=True, type=Path)
    parser.add_argument("--clip-power", required=True, type=Path)
    parser.add_argument("--clip-type1a", required=True, type=Path)
    parser.add_argument("--clip-type1b", required=True, type=Path)
    parser.add_argument("--cnn-power", required=True, type=Path)
    parser.add_argument("--cnn-type1a", required=True, type=Path)
    parser.add_argument("--cnn-type1b", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def load_embed(path: Path, label: str, scenario_name: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    metric_col = "power_mean" if "power_mean" in df.columns else "type1_mean"
    return pd.DataFrame(
        {
            "noise_sigma": df["noise_sigma"].astype(float),
            "metric_mean": df[metric_col].astype(float),
            "method_label": label,
            "scenario": scenario_name,
        }
    )


def load_raw(path: Path, label: str, scenario_name: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    metric_col = "power_mean" if "power_mean" in df.columns else "type1_mean"
    return pd.DataFrame(
        {
            "noise_sigma": df["noise_sigma"].astype(float),
            "metric_mean": df[metric_col].astype(float),
            "method_label": label,
            "scenario": scenario_name,
        }
    )


def plot_panel(ax, df: pd.DataFrame, scenario: str, title: str, ylabel: str) -> None:
    part = df[df["scenario"] == scenario].copy()
    for method_label in ORDER:
        sub = part[part["method_label"] == method_label].sort_values("noise_sigma")
        if sub.empty:
            continue
        ax.plot(
            sub["noise_sigma"],
            sub["metric_mean"],
            label=method_label,
            color=COLORS[method_label],
            marker=MARKERS[method_label],
            linewidth=2.0,
            markersize=6.5,
        )
    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xlabel("Noise sigma", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_ylim(0.0, 0.2 if scenario.startswith("type1") else 1.02)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.concat(
        [
            load_raw(args.raw_power, "Raw Pixel + GAUSS5", "power"),
            load_raw(args.raw_type1a, "Raw Pixel + GAUSS5", "type1_a"),
            load_raw(args.raw_type1b, "Raw Pixel + GAUSS5", "type1_b"),
            load_embed(args.clip_power, "CLIP + GAUSS5", "power"),
            load_embed(args.clip_type1a, "CLIP + GAUSS5", "type1_a"),
            load_embed(args.clip_type1b, "CLIP + GAUSS5", "type1_b"),
            load_embed(args.dino_power, "DINOv2 + GAUSS5", "power"),
            load_embed(args.dino_type1a, "DINOv2 + GAUSS5", "type1_a"),
            load_embed(args.dino_type1b, "DINOv2 + GAUSS5", "type1_b"),
            load_embed(args.cnn_power, "CNN Final FC128 + GAUSS5", "power"),
            load_embed(args.cnn_type1a, "CNN Final FC128 + GAUSS5", "type1_a"),
            load_embed(args.cnn_type1b, "CNN Final FC128 + GAUSS5", "type1_b"),
        ],
        ignore_index=True,
    )

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.8), dpi=180, sharex=True)
    plot_panel(axes[0], df, "power", "Power: A=(0,1,3) vs B=(0,1,6)", "Power")
    plot_panel(axes[1], df, "type1_a", "Type-I: A vs A", "Type-I error")
    plot_panel(axes[2], df, "type1_b", "Type-I: B vs B", "Type-I error")
    axes[1].axhline(0.05, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[2].axhline(0.05, color="black", linestyle="--", linewidth=1.0, alpha=0.7)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("BloodMNIST-224 Mixed-Population: Raw vs CLIP vs DINOv2 vs CNN", fontsize=16, weight="bold", y=1.12)
    fig.tight_layout()

    png = args.output_dir / "bloodmnist_mixed_raw_clip_dino_cnn_gauss5.png"
    pdf = args.output_dir / "bloodmnist_mixed_raw_clip_dino_cnn_gauss5.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
