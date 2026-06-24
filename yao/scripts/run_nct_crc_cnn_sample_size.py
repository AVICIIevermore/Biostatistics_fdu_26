#!/usr/bin/env python3
"""Run NCT-CRC sample-size study for a trained CNN embedding."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
YAO_ROOT = PROJECT_ROOT / "yao"
CONFIG_DIR = YAO_ROOT / "configs"


def run(cmd: list[str]) -> None:
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", str(PROJECT_ROOT / ".uv-cache"))
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env=env)


def write_r_config(
    path: Path,
    embedding_file: str,
    output_dir: str,
    sample_size: int,
    n_inner: int,
    n_outer: int,
    b_boot: int,
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
noise_subset <- c(0)

methods <- c("GAUSS5")

{scenario_block}
"""
    path.write_text(text)


def scenario_name(scenario: str, labels_x: list[int], labels_y: list[int]) -> str:
    if scenario == "power":
        return "power"
    return "type1_a" if labels_x == [6, 7, 3] else "type1_b"


def layer_method_label(layer: str) -> str:
    mapping = {
        "layer1_gap": "CNN Layer1 GAP + GAUSS5",
        "layer2_gap": "CNN Layer2 GAP + GAUSS5",
        "final_fc128": "CNN Final FC128 + GAUSS5",
    }
    return mapping[layer]


def layer_slug(layer: str) -> str:
    mapping = {
        "layer1_gap": "layer1gap",
        "layer2_gap": "layer2gap",
        "final_fc128": "finalfc",
    }
    return mapping[layer]


def collect_summary(summary_path: Path, sample_size: int, scenario_name_value: str, layer: str) -> dict[str, object]:
    df = pd.read_csv(summary_path)
    if "power_mean" in df.columns:
        metric_mean = float(df.loc[0, "power_mean"])
        metric_se = float(df.loc[0, "power_se"])
    else:
        metric_mean = float(df.loc[0, "type1_mean"])
        metric_se = float(df.loc[0, "type1_se"])
    return {
        "method_label": layer_method_label(layer),
        "sample_size": sample_size,
        "scenario": scenario_name_value,
        "noise_sigma": float(df.loc[0, "noise_sigma"]),
        "metric_mean": metric_mean,
        "metric_se": metric_se,
        "n_rep": int(df.loc[0, "n_rep"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample-sizes", default="30,60,90,120,150")
    parser.add_argument("--n-inner", type=int, default=20)
    parser.add_argument("--n-rep", type=int, default=4)
    parser.add_argument("--b-boot", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--n-cores", type=int, default=8)
    parser.add_argument("--max-per-label", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--layer", default="final_fc128", choices=["layer1_gap", "layer2_gap", "final_fc128"])
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = (PROJECT_ROOT / args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    sample_sizes = [int(x) for x in args.sample_sizes.split(",") if x.strip()]
    labels_a = [6, 7, 3]
    labels_b = [8, 7, 3]
    keep_labels = [3, 6, 7, 8]

    embedding_dir = PROJECT_ROOT / "yao" / "data" / "embeddings" / f"nctcrc_cnn_{layer_slug(args.layer)}_mix_norm_str_lym_tum_sharedpool"
    if not (embedding_dir / "embeddings.npy").exists():
        run(
            [
                "uv", "run", "python", "yao/scripts/extract_imagefolder_cnn_embeddings.py",
                "--metadata", args.metadata,
                "--image-root", args.image_root,
                "--checkpoint", args.checkpoint,
                "--split", args.split,
                "--x-labels", ",".join(map(str, labels_a)),
                "--y-labels", ",".join(map(str, labels_b)),
                "--keep-labels", ",".join(map(str, keep_labels)),
                "--max-per-label", str(args.max_per_label),
                "--noise-levels", "0",
                "--n-rep", "1",
                "--pool-only",
                "--layer", args.layer,
                "--batch-size", str(args.batch_size),
                "--output", str(embedding_dir),
            ]
        )

    rows: list[dict[str, object]] = []
    scenarios = [
        ("power", labels_a, labels_b),
        ("type1", labels_a, labels_a),
        ("type1", labels_b, labels_b),
    ]
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for sample_size in sample_sizes:
        for scenario, labels_x, labels_y in scenarios:
            scenario_id = scenario_name(scenario, labels_x, labels_y)
            run_outdir = outdir / f"{layer_slug(args.layer)}_{scenario_id}_n{sample_size}"
            config_path = CONFIG_DIR / f"_tmp_nctcrc_cnn_{layer_slug(args.layer)}_{scenario_id}_n{sample_size}.R"
            write_r_config(
                path=config_path,
                embedding_file=str(embedding_dir.relative_to(YAO_ROOT)),
                output_dir=str(run_outdir.relative_to(YAO_ROOT)),
                sample_size=sample_size,
                n_inner=args.n_inner,
                n_outer=args.n_rep,
                b_boot=args.b_boot,
                scenario=scenario,
                labels_x=labels_x,
                labels_y=labels_y,
                seed=args.seed,
                n_cores=args.n_cores,
            )
            run(["Rscript", "yao/scripts/run_noisy_embedding_testing.R", str(config_path)])
            rows.append(collect_summary(run_outdir / "noisy_embedding_summary.csv", sample_size, scenario_id, args.layer))
            config_path.unlink(missing_ok=True)

    pd.DataFrame(rows).to_csv(outdir / "cnn_sample_size_summary.csv", index=False)
    print(outdir / "cnn_sample_size_summary.csv")


if __name__ == "__main__":
    main()
