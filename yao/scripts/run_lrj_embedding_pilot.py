#!/usr/bin/env python3
"""Small-scale pilot: compare GEXP-5 and NEW-MMMD on shared embedding pools."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from lrj_mmmd_utils import gexp5_one, new_one


def read_npy_matrix(path: Path) -> np.ndarray:
    return np.load(path)


def sample_balanced_matrix(meta_labels: np.ndarray, target_labels: list[int], draws: int, sample_size: int, seed: int) -> np.ndarray:
    labels_unique = sorted(set(int(x) for x in target_labels))
    n_labels = len(labels_unique)
    if sample_size % n_labels != 0:
        raise ValueError("sample_size must be divisible by the number of target labels.")
    per_label = sample_size // n_labels
    rng = np.random.default_rng(seed)
    pools = [np.where(meta_labels == lbl)[0] for lbl in labels_unique]
    out = np.zeros((draws, sample_size), dtype=np.int64)
    for i in range(draws):
        draw_idx = []
        for pool in pools:
            draw_idx.extend(rng.choice(pool, size=per_label, replace=False).tolist())
        draw_idx = np.asarray(draw_idx, dtype=np.int64)
        rng.shuffle(draw_idx)
        out[i] = draw_idx
    return out


def sample_type1_pairs_balanced(pool_labels: np.ndarray, target_labels: list[int], draws: int, sample_size: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    labels_unique = sorted(set(int(x) for x in target_labels))
    n_labels = len(labels_unique)
    if sample_size % n_labels != 0:
        raise ValueError("sample_size must be divisible by the number of target labels.")
    per_label = sample_size // n_labels
    rng = np.random.default_rng(seed)
    pools = [np.where(pool_labels == lbl)[0] for lbl in labels_unique]
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(draws):
        x_idx, y_idx = [], []
        for pool in pools:
            chosen = rng.choice(pool, size=2 * per_label, replace=False)
            x_idx.extend(chosen[:per_label].tolist())
            y_idx.extend(chosen[per_label:].tolist())
        x_idx = np.asarray(x_idx, dtype=np.int64)
        y_idx = np.asarray(y_idx, dtype=np.int64)
        rng.shuffle(x_idx)
        rng.shuffle(y_idx)
        out.append((x_idx, y_idx))
    return out


def parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--sample-size", type=int, default=90)
    parser.add_argument("--n-inner", type=int, default=10)
    parser.add_argument("--n-outer", type=int, default=2)
    parser.add_argument("--b-boot", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--power-x-labels", required=True)
    parser.add_argument("--power-y-labels", required=True)
    parser.add_argument("--type1-a-labels", required=True)
    parser.add_argument("--type1-b-labels", required=True)
    parser.add_argument("--noise-sigma", type=float, default=0.0)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embedding_dir = Path(args.embedding_dir).resolve()
    meta = pd.read_csv(embedding_dir / "metadata.csv")
    emb = read_npy_matrix(embedding_dir / "embeddings.npy").astype(np.float64, copy=False)
    meta = meta.copy()
    meta["row_index"] = np.arange(len(meta), dtype=np.int64)

    if "noise_sigma" in meta.columns:
        meta = meta[np.isclose(meta["noise_sigma"].astype(float), args.noise_sigma)].copy()
    if "outer_iter" in meta.columns:
        meta = meta[meta["outer_iter"] == int(meta["outer_iter"].min())].copy()
    rows = meta["row_index"].to_numpy(dtype=np.int64)
    emb = emb[rows]
    labels = meta["label"].to_numpy(dtype=np.int64)

    power_x = parse_int_list(args.power_x_labels)
    power_y = parse_int_list(args.power_y_labels)
    type1_a = parse_int_list(args.type1_a_labels)
    type1_b = parse_int_list(args.type1_b_labels)

    scenarios = [
        ("power", power_x, power_y),
        ("type1_a", type1_a, type1_a),
        ("type1_b", type1_b, type1_b),
    ]
    methods = [
        ("GEXP-5", gexp5_one),
        ("NEW-MMMD", new_one),
    ]

    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    results = []

    for outer in range(1, args.n_outer + 1):
        for scenario_name, x_labels, y_labels in scenarios:
            if scenario_name == "power":
                x_pool = np.where(np.isin(labels, np.asarray(x_labels)))[0]
                y_pool = np.where(np.isin(labels, np.asarray(y_labels)))[0]
                x_labels_pool = labels[x_pool]
                y_labels_pool = labels[y_pool]
                resamp_x = sample_balanced_matrix(x_labels_pool, x_labels, args.n_inner, args.sample_size, args.seed + outer * 100000 + 11)
                resamp_y = sample_balanced_matrix(y_labels_pool, y_labels, args.n_inner, args.sample_size, args.seed + outer * 100000 + 29)
            else:
                pool = np.where(np.isin(labels, np.asarray(x_labels)))[0]
                pool_labels = labels[pool]
                null_pairs = sample_type1_pairs_balanced(pool_labels, x_labels, args.n_inner, args.sample_size, args.seed + outer * 100000 + 71)

            for inner in range(args.n_inner):
                if scenario_name == "power":
                    X = emb[x_pool[resamp_x[inner]]]
                    Y = emb[y_pool[resamp_y[inner]]]
                else:
                    x_idx, y_idx = null_pairs[inner]
                    X = emb[pool[x_idx]]
                    Y = emb[pool[y_idx]]
                for method_name, runner in methods:
                    rng = np.random.default_rng(args.seed + outer * 1000000 + inner * 100 + (0 if method_name == "GEXP-5" else 1))
                    out = runner(X, Y, n_boot=args.b_boot, rng=rng)
                    results.append(
                        {
                            "outer_iter": outer,
                            "inner_iter": inner + 1,
                            "scenario": scenario_name,
                            "method": method_name,
                            "reject": int(out["reject"]),
                            "q": int(out["q"]),
                            "cond_raw": float(out["cond_raw"]),
                            "cond_reg": float(out["cond_reg"]),
                            "ridge_lambda": float(out["ridge_lambda"]),
                            "stat": float(out["stat"]),
                            "cutoff": float(out["cutoff"]),
                        }
                    )
            print(f"[pilot] outer={outer}/{args.n_outer} scenario={scenario_name} done", flush=True)

    df = pd.DataFrame(results)
    summary = (
        df.groupby(["scenario", "method"], as_index=False)
        .agg(
            reject_rate=("reject", "mean"),
            q_median=("q", "median"),
            cond_raw_median=("cond_raw", "median"),
            cond_reg_median=("cond_reg", "median"),
            ridge_lambda_median=("ridge_lambda", "median"),
        )
    )
    df.to_csv(outdir / "lrj_pilot_results.csv", index=False)
    summary.to_csv(outdir / "lrj_pilot_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(outdir / "lrj_pilot_summary.csv")


if __name__ == "__main__":
    main()
