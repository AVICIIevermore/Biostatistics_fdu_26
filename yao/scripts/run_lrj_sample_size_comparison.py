#!/usr/bin/env python3
"""Formal sample-size comparison for lrj GEXP-5 vs NEW-MMMD on one embedding pool."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lrj_mmmd_utils import gexp5_one, new_one


def parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--sample-sizes", default="30,60,90,120,150")
    parser.add_argument("--n-inner", type=int, default=30)
    parser.add_argument("--n-outer", type=int, default=6)
    parser.add_argument("--b-boot", type=int, default=150)
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
    sample_sizes = [int(x) for x in args.sample_sizes.split(",") if x.strip()]
    embedding_dir = Path(args.embedding_dir).resolve()
    meta = pd.read_csv(embedding_dir / "metadata.csv")
    emb = np.load(embedding_dir / "embeddings.npy").astype(np.float64, copy=False)
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
    summary_rows = []

    for sample_size in sample_sizes:
        print(f"[formal] sample_size={sample_size}", flush=True)
        for scenario_name, x_labels, y_labels in scenarios:
            for method_name, runner in methods:
                t0 = time.time()
                rejects, qs, cond_raws, cond_regs, ridges = [], [], [], [], []
                for outer in range(1, args.n_outer + 1):
                    if scenario_name == "power":
                        x_pool = np.where(np.isin(labels, np.asarray(x_labels)))[0]
                        y_pool = np.where(np.isin(labels, np.asarray(y_labels)))[0]
                        x_labels_pool = labels[x_pool]
                        y_labels_pool = labels[y_pool]
                        resamp_x = sample_balanced_matrix(
                            x_labels_pool, x_labels, args.n_inner, sample_size,
                            args.seed + sample_size * 100000 + outer * 1000 + 11
                        )
                        resamp_y = sample_balanced_matrix(
                            y_labels_pool, y_labels, args.n_inner, sample_size,
                            args.seed + sample_size * 100000 + outer * 1000 + 29
                        )
                    else:
                        pool = np.where(np.isin(labels, np.asarray(x_labels)))[0]
                        pool_labels = labels[pool]
                        null_pairs = sample_type1_pairs_balanced(
                            pool_labels, x_labels, args.n_inner, sample_size,
                            args.seed + sample_size * 100000 + outer * 1000 + 71
                        )

                    for inner in range(args.n_inner):
                        if scenario_name == "power":
                            X = emb[x_pool[resamp_x[inner]]]
                            Y = emb[y_pool[resamp_y[inner]]]
                        else:
                            x_idx, y_idx = null_pairs[inner]
                            X = emb[pool[x_idx]]
                            Y = emb[pool[y_idx]]
                        rng = np.random.default_rng(
                            args.seed + sample_size * 1000000 + outer * 10000 + inner * 100 + (0 if method_name == "GEXP-5" else 1)
                        )
                        out = runner(X, Y, n_boot=args.b_boot, rng=rng)
                        row = {
                            "sample_size": sample_size,
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
                        results.append(row)
                        rejects.append(row["reject"])
                        qs.append(row["q"])
                        cond_raws.append(row["cond_raw"])
                        cond_regs.append(row["cond_reg"])
                        ridges.append(row["ridge_lambda"])
                runtime = time.time() - t0
                summary_rows.append(
                    {
                        "sample_size": sample_size,
                        "scenario": scenario_name,
                        "method": method_name,
                        "reject_rate": float(np.mean(rejects)),
                        "reject_se": float(np.sqrt(np.var(rejects, ddof=1) / len(rejects))) if len(rejects) > 1 else np.nan,
                        "q_median": float(np.median(qs)),
                        "cond_raw_median": float(np.median(cond_raws)),
                        "cond_reg_median": float(np.median(cond_regs)),
                        "ridge_lambda_median": float(np.median(ridges)),
                        "runtime_sec": runtime,
                        "n_outer": args.n_outer,
                        "n_inner": args.n_inner,
                        "b_boot": args.b_boot,
                    }
                )
                print(
                    f"[formal] n={sample_size} scenario={scenario_name} method={method_name} "
                    f"reject_rate={np.mean(rejects):.3f} q_med={np.median(qs):.1f} time={runtime:.1f}s",
                    flush=True,
                )

    results_df = pd.DataFrame(results)
    summary_df = pd.DataFrame(summary_rows).sort_values(["scenario", "sample_size", "method"]).reset_index(drop=True)
    results_df.to_csv(outdir / "lrj_formal_results.csv", index=False)
    summary_df.to_csv(outdir / "lrj_formal_summary.csv", index=False)
    print(outdir / "lrj_formal_summary.csv")


if __name__ == "__main__":
    main()
