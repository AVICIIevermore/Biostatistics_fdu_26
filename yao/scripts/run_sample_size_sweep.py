#!/usr/bin/env python3
"""Run a lightweight sample-size sweep for BloodMNIST mixed-population experiments."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def write_r_config(
    path: Path,
    embedding_file: str,
    output_dir: str,
    sample_size: int,
    n_inner: int,
    n_outer: int,
    b_boot: int,
    noise_sigma: float,
    scenario: str,
    labels_x: list[int],
    labels_y: list[int],
    seed: int,
    n_cores: int,
) -> None:
    if scenario == "power":
        scenario_block = (
            'scenario <- "power"\n'
            'alternative_x_labels <- c(%s)\n'
            'alternative_y_labels <- c(%s)\n'
        ) % (", ".join(map(str, labels_x)), ", ".join(map(str, labels_y)))
    else:
        scenario_block = (
            'scenario <- "type1"\n'
            'alternative_x_labels <- c(%s)\n'
            'alternative_y_labels <- c(%s)\n'
            'null_x_labels <- c(%s)\n'
            'null_y_labels <- c(%s)\n'
        ) % (
            ", ".join(map(str, labels_x)),
            ", ".join(map(str, labels_y)),
            ", ".join(map(str, labels_x)),
            ", ".join(map(str, labels_y)),
        )

    text = f"""embedding_file <- "{embedding_file}"
output_dir <- "{output_dir}"

sample_size <- {sample_size}L
n_inner <- {n_inner}L
n_outer <- {n_outer}L
B_boot <- {b_boot}L
alpha <- 0.05
seed <- {seed}L
ridge_scale <- 1e-5
n_cores <- {n_cores}L
sample_replace <- FALSE
balanced_sampling <- TRUE
noise_subset <- c({noise_sigma})

methods <- c("GAUSS5")

{scenario_block}
"""
    path.write_text(text)


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def scenario_name(scenario: str, labels_x: list[int], labels_y: list[int]) -> str:
    if scenario == "power":
        return "power"
    return "type1_a" if labels_x == [0, 1, 3] else "type1_b"


def collect_summary(summary_path: Path, method_label: str, sample_size: int, scenario_name_value: str) -> dict[str, object]:
    df = pd.read_csv(summary_path)
    if "power_mean" in df.columns:
        metric_mean = float(df.loc[0, "power_mean"])
        metric_se = float(df.loc[0, "power_se"])
    else:
        metric_mean = float(df.loc[0, "type1_mean"])
        metric_se = float(df.loc[0, "type1_se"])
    return {
        "method_label": method_label,
        "sample_size": sample_size,
        "scenario": scenario_name_value,
        "noise_sigma": float(df.loc[0, "noise_sigma"]),
        "metric_mean": metric_mean,
        "metric_se": metric_se,
        "n_rep": int(df.loc[0, "n_rep"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-sizes", default="30,60,90,120,150")
    parser.add_argument("--noise-sigma", type=float, default=0.6)
    parser.add_argument("--n-inner", type=int, default=30)
    parser.add_argument("--n-rep", type=int, default=6)
    parser.add_argument("--b-boot", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--n-cores", type=int, default=8)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--npz", default="yao/data/mnist/bloodmnist_224.npz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = (PROJECT_ROOT / args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    sample_sizes = [int(x) for x in args.sample_sizes.split(",") if x.strip()]

    rows: list[dict[str, object]] = []
    scenarios = [
        ("power", [0, 1, 3], [0, 1, 6]),
        ("type1", [0, 1, 3], [0, 1, 3]),
        ("type1", [0, 1, 6], [0, 1, 6]),
    ]
    embedding_methods = [
        ("DINOv2 + GAUSS5", "data/embeddings/bloodmnist224_dinov2_mix013_016_sharedpool"),
        ("CLIP + GAUSS5", "data/embeddings/bloodmnist224_clip_mix013_016_sharedpool"),
    ]

    tmpdir_path = PROJECT_ROOT / "yao" / "configs"
    tmpdir_path.mkdir(parents=True, exist_ok=True)

    for method_label, embedding_file in embedding_methods:
        for sample_size in sample_sizes:
            for scenario, labels_x, labels_y in scenarios:
                scenario_id = scenario_name(scenario, labels_x, labels_y)
                run_outdir = outdir / f"{method_label.split()[0].lower()}_{scenario_id}_n{sample_size}"
                config_path = tmpdir_path / f"{method_label.split()[0].lower()}_{scenario_id}_n{sample_size}.R"
                write_r_config(
                    path=config_path,
                    embedding_file=embedding_file,
                    output_dir=str(run_outdir.relative_to(PROJECT_ROOT / "yao")),
                    sample_size=sample_size,
                    n_inner=args.n_inner,
                    n_outer=args.n_rep,
                    b_boot=args.b_boot,
                    noise_sigma=args.noise_sigma,
                    scenario=scenario,
                    labels_x=labels_x,
                    labels_y=labels_y,
                    seed=args.seed,
                    n_cores=args.n_cores,
                )
                run(["Rscript", "yao/scripts/run_noisy_embedding_testing.R", str(config_path)])
                rows.append(
                    collect_summary(
                        run_outdir / "noisy_embedding_summary.csv",
                        method_label=method_label,
                        sample_size=sample_size,
                        scenario_name_value=scenario_id,
                    )
                )
                config_path.unlink(missing_ok=True)

    for sample_size in sample_sizes:
        for scenario, labels_x, labels_y in scenarios:
            scenario_id = scenario_name(scenario, labels_x, labels_y)
            run_outdir = outdir / f"raw_{scenario_id}_n{sample_size}"
            run(
                [
                    "uv",
                    "run",
                    "python",
                    "yao/scripts/run_rawpixel_gauss5_mixed.py",
                    "--npz",
                    args.npz,
                    "--split",
                    "test",
                    "--scenario",
                    scenario,
                    "--x-labels",
                    ",".join(map(str, labels_x)),
                    "--y-labels",
                    ",".join(map(str, labels_y)),
                    "--noise-levels",
                    str(args.noise_sigma),
                    "--sample-size",
                    str(sample_size),
                    "--n-inner",
                    str(args.n_inner),
                    "--n-rep",
                    str(args.n_rep),
                    "--b-boot",
                    str(args.b_boot),
                    "--seed",
                    str(args.seed),
                    "--output-dir",
                    str(run_outdir),
                ]
            )
            rows.append(
                collect_summary(
                    run_outdir / "rawpixel_gauss5_summary.csv",
                    method_label="Raw Pixel + GAUSS5",
                    sample_size=sample_size,
                    scenario_name_value=scenario_id,
                )
            )

    summary = pd.DataFrame(rows).sort_values(["scenario", "sample_size", "method_label"]).reset_index(drop=True)
    summary.to_csv(outdir / "sample_size_summary.csv", index=False)
    print(outdir / "sample_size_summary.csv")


if __name__ == "__main__":
    main()
