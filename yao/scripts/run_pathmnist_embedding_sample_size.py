#!/usr/bin/env python3
"""Run PathMNIST sample-size MMMD experiments with frozen embeddings.

This script aligns with dechao's PathMNIST-28 setup:
- H1 power: mix635_vs_mix835 with sample sizes 30,60,90,120,150
- H0 type-I: mix635_vs_mix635_null with sample sizes 60,120
- test split only
- class-balanced sampling
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
YAO_ROOT = REPO_ROOT / "yao"
CONFIG_DIR = YAO_ROOT / "configs"

POWER_X = [6, 3, 5]
POWER_Y = [8, 3, 5]
TYPE1 = [6, 3, 5]


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def make_config_text(
    embedding_file: str,
    output_dir: str,
    sample_size: int,
    n_outer: int,
    n_inner: int,
    b_boot: int,
    alpha: float,
    seed: int,
    n_cores: int,
    scenario: str,
) -> str:
    lines = [
        f'embedding_file <- "{embedding_file}"',
        f'output_dir <- "{output_dir}"',
        "",
        f"sample_size <- {sample_size}L",
        f"n_outer <- {n_outer}L",
        f"n_inner <- {n_inner}L",
        f"B_boot <- {b_boot}L",
        f"alpha <- {alpha}",
        f"seed <- {seed}L",
        "ridge_scale <- 1e-5",
        f"n_cores <- {n_cores}L",
        "sample_replace <- FALSE",
        "balanced_sampling <- TRUE",
        "noise_subset <- c(0)",
        "",
        'methods <- c("GAUSS5")',
        "",
    ]
    if scenario == "power":
        lines.extend(
            [
                'scenario <- "power"',
                "alternative_x_labels <- c(6, 3, 5)",
                "alternative_y_labels <- c(8, 3, 5)",
            ]
        )
    elif scenario == "type1":
        lines.extend(
            [
                'scenario <- "type1"',
                "alternative_x_labels <- c(6, 3, 5)",
                "alternative_y_labels <- c(6, 3, 5)",
                "null_x_labels <- c(6, 3, 5)",
                "null_y_labels <- c(6, 3, 5)",
            ]
        )
    else:
        raise ValueError(f"Unsupported scenario: {scenario}")
    lines.append("")
    return "\n".join(lines)


def read_metric(summary_path: Path, scenario_name: str, sample_size: int, method_label: str) -> dict[str, object]:
    df = pd.read_csv(summary_path)
    if "power_mean" in df.columns:
        mean = float(df.loc[0, "power_mean"])
        se = float(df.loc[0, "power_se"])
    else:
        mean = float(df.loc[0, "type1_mean"])
        se = float(df.loc[0, "type1_se"])
    return {
        "scenario": scenario_name,
        "sample_size": sample_size,
        "method_label": method_label,
        "noise_sigma": float(df.loc[0, "noise_sigma"]),
        "metric_mean": mean,
        "metric_se": se,
        "n_rep": int(df.loc[0, "n_rep"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", required=True, choices=["dinov2", "clip"])
    parser.add_argument("--npz", default="yao/data/medmnist/pathmnist.npz")
    parser.add_argument("--embedding-dir", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--n-outer", type=int, default=10)
    parser.add_argument("--n-inner", type=int, default=500)
    parser.add_argument("--b-boot", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--n-cores", type=int, default=4)
    parser.add_argument("--power-sizes", default="30,60,90,120,150")
    parser.add_argument("--type1-sizes", default="60,120")
    parser.add_argument("--force-embed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embedding_dir = (
        REPO_ROOT / args.embedding_dir
        if args.embedding_dir
        else REPO_ROOT / "yao" / "data" / "embeddings" / f"pathmnist28_{args.encoder}_mix635_835_sharedpool"
    )
    results_dir = (
        REPO_ROOT / args.results_dir
        if args.results_dir
        else REPO_ROOT / "yao" / "results" / f"pathmnist28_{args.encoder}_sample_size_alignment"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    embeddings_npy = embedding_dir / "embeddings.npy"
    metadata_csv = embedding_dir / "metadata.csv"
    if args.force_embed or not (embeddings_npy.exists() and metadata_csv.exists()):
        run(
            [
                "uv",
                "run",
                "python",
                "yao/scripts/extract_medmnist_noise_embeddings.py",
                "--npz",
                args.npz,
                "--split",
                args.split,
                "--x-labels",
                "6,3,5",
                "--y-labels",
                "8,3,5",
                "--keep-labels",
                "3,5,6,8",
                "--noise-levels",
                "0",
                "--n-rep",
                "1",
                "--seed",
                str(args.seed),
                "--pool-only",
                "--encoder",
                args.encoder,
                "--batch-size",
                str(args.batch_size),
                "--device",
                args.device,
                "--output",
                str(embedding_dir),
            ]
        )

    power_sizes = [int(x) for x in args.power_sizes.split(",") if x.strip()]
    type1_sizes = [int(x) for x in args.type1_sizes.split(",") if x.strip()]
    rows: list[dict[str, object]] = []

    for sample_size in power_sizes:
        config_path = CONFIG_DIR / f"_tmp_pathmnist_{args.encoder}_power_n{sample_size}.R"
        config_path.write_text(
            make_config_text(
                embedding_file=str(embedding_dir.relative_to(YAO_ROOT)),
                output_dir=str((results_dir / f"power_n{sample_size}").relative_to(YAO_ROOT)),
                sample_size=sample_size,
                n_outer=args.n_outer,
                n_inner=args.n_inner,
                b_boot=args.b_boot,
                alpha=args.alpha,
                seed=args.seed,
                n_cores=args.n_cores,
                scenario="power",
            )
        )
        run(["Rscript", "yao/scripts/run_noisy_embedding_testing.R", str(config_path)])
        rows.append(
            read_metric(results_dir / f"power_n{sample_size}" / "noisy_embedding_summary.csv", "power", sample_size, f"{args.encoder.upper()} + GAUSS5")
        )
        config_path.unlink(missing_ok=True)

    for sample_size in type1_sizes:
        config_path = CONFIG_DIR / f"_tmp_pathmnist_{args.encoder}_type1_n{sample_size}.R"
        config_path.write_text(
            make_config_text(
                embedding_file=str(embedding_dir.relative_to(YAO_ROOT)),
                output_dir=str((results_dir / f"type1_n{sample_size}").relative_to(YAO_ROOT)),
                sample_size=sample_size,
                n_outer=args.n_outer,
                n_inner=args.n_inner,
                b_boot=args.b_boot,
                alpha=args.alpha,
                seed=args.seed + 1,
                n_cores=args.n_cores,
                scenario="type1",
            )
        )
        run(["Rscript", "yao/scripts/run_noisy_embedding_testing.R", str(config_path)])
        rows.append(
            read_metric(results_dir / f"type1_n{sample_size}" / "noisy_embedding_summary.csv", "type1", sample_size, f"{args.encoder.upper()} + GAUSS5")
        )
        config_path.unlink(missing_ok=True)

    summary = pd.DataFrame(rows).sort_values(["scenario", "sample_size"]).reset_index(drop=True)
    summary.to_csv(results_dir / "pathmnist_embedding_summary.csv", index=False)
    print(results_dir / "pathmnist_embedding_summary.csv")


if __name__ == "__main__":
    main()
