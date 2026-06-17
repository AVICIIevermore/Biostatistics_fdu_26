#!/usr/bin/env python3
"""Run raw-pixel GAUSS5 MMMD experiments on BloodMNIST-style noisy image mixtures."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def pairwise_sq_dists(Z: np.ndarray) -> np.ndarray:
    G = Z @ Z.T
    norms = np.sum(Z * Z, axis=1)
    D = norms[:, None] + norms[None, :] - 2.0 * G
    np.maximum(D, 0.0, out=D)
    return D


def median_gamma(X: np.ndarray, Y: np.ndarray) -> float:
    Z = np.vstack([X, Y]).astype(np.float64, copy=False)
    D = pairwise_sq_dists(Z)
    upper = D[np.triu_indices(D.shape[0], k=1)]
    med = float(np.median(upper))
    if not np.isfinite(med) or med <= 0:
        raise ValueError("Median squared distance is not positive.")
    return 1.0 / med


def gauss5_test(X: np.ndarray, Y: np.ndarray, B_boot: int, alpha: float, ridge_scale: float, rng: np.random.Generator) -> int:
    n = X.shape[0]
    X = X.astype(np.float64, copy=False)
    Y = Y.astype(np.float64, copy=False)
    Z = np.vstack([X, Y])
    D = pairwise_sq_dists(Z)
    upper = D[np.triu_indices(D.shape[0], k=1)]
    med = float(np.median(upper))
    if not np.isfinite(med) or med <= 0:
        raise ValueError("Median squared distance is not positive.")
    base_gamma = 1.0 / med
    gammas = [(2.0**k) * base_gamma for k in range(-2, 3)]
    Dxx = D[:n, :n]
    Dyy = D[n:, n:]
    Dxy = D[:n, n:]
    C = np.eye(n) - np.ones((n, n)) / n
    offdiag_mask = ~np.eye(n, dtype=bool)

    centered = []
    mmd_vec = []
    for gamma in gammas:
        Kxx = np.exp(-gamma * Dxx, dtype=np.float64)
        Kyy = np.exp(-gamma * Dyy, dtype=np.float64)
        Kxy = np.exp(-gamma * Dxy, dtype=np.float64)
        centered.append(C @ Kxx @ C)
        mmd = Kxx[offdiag_mask].mean() + Kyy[offdiag_mask].mean() - 2.0 * Kxy.mean()
        mmd_vec.append(n * float(mmd))
    k_count = len(centered)

    sigma_hat = np.zeros((k_count, k_count), dtype=np.float64)
    for i in range(k_count):
        for j in range(k_count):
            sigma_hat[i, j] = (8.0 / (n * n)) * np.trace(centered[i] @ centered[j])

    diag_ref = np.min(np.diag(sigma_hat))
    if not np.isfinite(diag_ref) or diag_ref <= 0:
        diag_ref = float(np.mean(np.diag(sigma_hat)))
    if not np.isfinite(diag_ref) or diag_ref <= 0:
        diag_ref = 1.0
    sigma_reg = sigma_hat + (ridge_scale * diag_ref) * np.eye(k_count)
    inv_cov = np.linalg.inv(sigma_reg)

    mmd_vec = np.asarray(mmd_vec, dtype=np.float64)
    stat = float(mmd_vec @ inv_cov @ mmd_vec)

    U = rng.normal(loc=0.0, scale=np.sqrt(2.0), size=(B_boot, n))
    boot_stats = np.zeros((B_boot, k_count), dtype=np.float64)
    for j, Kc in enumerate(centered):
        K = Kc / n
        KU = K @ U.T
        boot_stats[:, j] = np.sum(U.T * KU, axis=0) - 2.0 * np.trace(K)

    quad = np.einsum("bi,ij,bj->b", boot_stats, inv_cov, boot_stats)
    cutoff = float(np.quantile(quad, 1.0 - alpha, method="median_unbiased"))
    return int(stat > cutoff)


def sample_balanced(labels: np.ndarray, target_labels: list[int], per_label: int, rng: np.random.Generator) -> np.ndarray:
    idx = []
    for lbl in target_labels:
        pool = np.where(labels == lbl)[0]
        idx.extend(rng.choice(pool, size=per_label, replace=False).tolist())
    idx = np.asarray(idx, dtype=np.int64)
    rng.shuffle(idx)
    return idx


def sample_type1_balanced(labels: np.ndarray, target_labels: list[int], per_label: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    x_idx, y_idx = [], []
    for lbl in target_labels:
        pool = np.where(labels == lbl)[0]
        chosen = rng.choice(pool, size=2 * per_label, replace=False)
        x_idx.extend(chosen[:per_label].tolist())
        y_idx.extend(chosen[per_label:].tolist())
    x_idx = np.asarray(x_idx, dtype=np.int64)
    y_idx = np.asarray(y_idx, dtype=np.int64)
    rng.shuffle(x_idx)
    rng.shuffle(y_idx)
    return x_idx, y_idx


def add_noise_and_flatten(images: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noisy = np.clip(images.astype(np.float32) / 255.0 + rng.normal(0.0, sigma, size=images.shape).astype(np.float32), 0.0, 1.0)
    return noisy.reshape(noisy.shape[0], -1).astype(np.float64, copy=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--scenario", required=True, choices=["power", "type1"])
    parser.add_argument("--x-labels", required=True)
    parser.add_argument("--y-labels", required=True)
    parser.add_argument("--noise-levels", default="0,0.2,0.4,0.6,0.8,1")
    parser.add_argument("--sample-size", type=int, default=90)
    parser.add_argument("--n-inner", type=int, default=100)
    parser.add_argument("--n-rep", type=int, default=10)
    parser.add_argument("--b-boot", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--ridge-scale", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.npz)
    images = data[f"{args.split}_images"]
    labels = data[f"{args.split}_labels"].reshape(-1).astype(np.int64)

    x_labels = parse_int_list(args.x_labels)
    y_labels = parse_int_list(args.y_labels)
    noise_levels = parse_float_list(args.noise_levels)
    per_label = args.sample_size // len(set(x_labels))
    if args.sample_size % len(set(x_labels)) != 0:
        raise SystemExit("sample-size must be divisible by the number of labels in x-labels.")
    if args.scenario == "power" and args.sample_size % len(set(y_labels)) != 0:
        raise SystemExit("sample-size must be divisible by the number of labels in y-labels.")

    rows = []
    for outer_iter in range(1, args.n_rep + 1):
        for sigma in noise_levels:
            rejects = []
            for inner in range(1, args.n_inner + 1):
                rng = np.random.default_rng(args.seed + outer_iter * 100000 + round(sigma * 1000) * 100 + inner)
                if args.scenario == "power":
                    x_idx = sample_balanced(labels, x_labels, per_label, rng)
                    y_idx = sample_balanced(labels, y_labels, args.sample_size // len(set(y_labels)), rng)
                else:
                    x_idx, y_idx = sample_type1_balanced(labels, x_labels, per_label, rng)
                X = add_noise_and_flatten(images[x_idx], sigma, rng)
                Y = add_noise_and_flatten(images[y_idx], sigma, rng)
                rejects.append(gauss5_test(X, Y, B_boot=args.b_boot, alpha=args.alpha, ridge_scale=args.ridge_scale, rng=rng))

            rows.append({
                "outer_iter": outer_iter,
                "noise_sigma": sigma,
                "method": "GAUSS5",
                "reject_rate": float(np.mean(rejects)),
                "scenario": args.scenario,
            })
            print(
                f"outer={outer_iter}/{args.n_rep} sigma={sigma:g} scenario={args.scenario} reject_rate={np.mean(rejects):.3f}",
                flush=True,
            )

    results = pd.DataFrame(rows)
    summary = (
        results.groupby(["noise_sigma", "method", "scenario"])["reject_rate"]
        .agg(["mean", "sem"])
        .reset_index()
    )
    summary.columns = [
        "noise_sigma",
        "method",
        "scenario",
        "power_mean" if args.scenario == "power" else "type1_mean",
        "power_se" if args.scenario == "power" else "type1_se",
    ]
    summary["n_rep"] = args.n_rep

    results.to_csv(outdir / "rawpixel_gauss5_results.csv", index=False)
    summary.to_csv(outdir / "rawpixel_gauss5_summary.csv", index=False)
    print(summary)
    print(f"Wrote results to: {outdir}")


if __name__ == "__main__":
    main()
