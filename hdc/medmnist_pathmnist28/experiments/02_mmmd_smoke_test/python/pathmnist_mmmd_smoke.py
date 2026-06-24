#!/usr/bin/env python3
"""PathMNIST-28 MMMD smoke test.

This intentionally small run validates data sampling, raw-pixel MMMD,
CNN-final-embedding MMMD, diagnostics, and plots before the full power run.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = DATASET_ROOT / "data" / "pathmnist.npz"
CHECKPOINT_PATH = DATASET_ROOT / "shared_model" / "models" / "pathmnist_cnn_checkpoint.pt"
CNN_MODULE_PATH = DATASET_ROOT / "shared_model" / "python" / "pathmnist_cnn_pipeline.py"
RESULTS_DIR = EXPERIMENT_ROOT / "Results"
LOG_DIR = EXPERIMENT_ROOT / "logs"

CLASS_NAMES = {
    3: "lymphocytes",
    5: "smooth muscle",
    6: "normal colon mucosa",
    8: "colorectal adenocarcinoma epithelium",
}
METHODS = ["raw_pixel_gaussian5", "cnn_final_fc128_gaussian5"]


def log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    with (LOG_DIR / "smoke_test.log").open("a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def write_csv(path: Path, fieldnames: Iterable[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def load_cnn_module():
    spec = importlib.util.spec_from_file_location("pathmnist_cnn_pipeline", CNN_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_test_data() -> tuple[np.ndarray, np.ndarray]:
    data = np.load(DATA_PATH)
    images = data["test_images"]
    labels = data["test_labels"].reshape(-1).astype(int)
    return images, labels


def sample_balanced_h1(labels: np.ndarray, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if n % 3 != 0:
        raise ValueError("sample size must be divisible by 3 for three-class balanced mixtures")
    per_class = n // 3
    by_class = {label: np.where(labels == label)[0] for label in [3, 5, 6, 8]}
    for label, idx in by_class.items():
        needed = 2 * per_class if label in [3, 5] else per_class
        if len(idx) < needed:
            raise ValueError(f"class {label} has {len(idx)} test images, need {needed}")

    x_parts = [rng.choice(by_class[6], per_class, replace=False)]
    y_parts = [rng.choice(by_class[8], per_class, replace=False)]
    for shared_label in [3, 5]:
        chosen = rng.choice(by_class[shared_label], 2 * per_class, replace=False)
        x_parts.append(chosen[:per_class])
        y_parts.append(chosen[per_class:])

    x_idx = np.concatenate(x_parts)
    y_idx = np.concatenate(y_parts)
    rng.shuffle(x_idx)
    rng.shuffle(y_idx)
    return x_idx, y_idx


def pairwise_sq_dists(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_norm = np.sum(x * x, axis=1, keepdims=True)
    y_norm = np.sum(y * y, axis=1, keepdims=True).T
    out = x_norm + y_norm - 2.0 * (x @ y.T)
    return np.maximum(out, 0.0)


def gaussian5_gammas(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    z = np.vstack([x, y])
    d2 = pairwise_sq_dists(z, z)
    upper = d2[np.triu_indices_from(d2, k=1)]
    median_d2 = np.median(upper[upper > 0])
    if not np.isfinite(median_d2) or median_d2 <= 0:
        median_d2 = 1.0
    med_gamma = 1.0 / median_d2
    return (2.0 ** np.arange(-2, 3)) * med_gamma


def kernel_mats(x: np.ndarray, y: np.ndarray, gammas: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    dxx = pairwise_sq_dists(x, x)
    dyy = pairwise_sq_dists(y, y)
    dxy = pairwise_sq_dists(x, y)
    return ([np.exp(-g * dxx) for g in gammas], [np.exp(-g * dyy) for g in gammas], [np.exp(-g * dxy) for g in gammas])


def condition_number(mat: np.ndarray) -> float:
    values = np.linalg.svd(mat, compute_uv=False)
    if len(values) == 0 or values[-1] <= 0 or not np.all(np.isfinite(values)):
        return float("nan")
    return float(values[0] / values[-1])


def run_mmmd_gaussian5(x: np.ndarray, y: np.ndarray, b_boot: int, alpha: float, rng: np.random.Generator) -> dict[str, object]:
    n = x.shape[0]
    gammas = gaussian5_gammas(x, y)
    kxx_list, kyy_list, kxy_list = kernel_mats(x, y, gammas)
    offdiag = ~np.eye(n, dtype=bool)
    mmd_vec = np.array([
        n * np.mean(kxx[offdiag] + kyy[offdiag] - 2.0 * kxy[offdiag])
        for kxx, kyy, kxy in zip(kxx_list, kyy_list, kxy_list)
    ])

    c = np.eye(n) - np.ones((n, n)) / n
    centered = [c @ kxx @ c for kxx in kxx_list]
    r = len(centered)
    sigma_hat = np.empty((r, r), dtype=float)
    for i in range(r):
        for j in range(r):
            sigma_hat[i, j] = (8.0 / (n ** 2)) * np.trace(centered[i] @ centered[j])

    diag_mean = float(np.mean(np.diag(sigma_hat)))
    if not np.isfinite(diag_mean) or diag_mean <= 0:
        diag_mean = 1.0
    ridge_lambda = 1e-4 * diag_mean
    sigma_reg = sigma_hat + ridge_lambda * np.eye(r)
    inv_cov = np.linalg.solve(sigma_reg, np.eye(r))

    u = rng.normal(loc=0.0, scale=np.sqrt(2.0), size=(b_boot, n))
    boot_stats = np.empty((b_boot, r), dtype=float)
    for i, k_centered in enumerate(centered):
        kboot = k_centered / n
        ku = kboot @ u.T
        boot_stats[:, i] = np.sum(u * ku.T, axis=1) - 2.0 * np.trace(kboot)
    boot_quad = np.sum((boot_stats @ inv_cov) * boot_stats, axis=1)
    cutoff = float(np.quantile(boot_quad, 1.0 - alpha))
    stat = float(mmd_vec.T @ inv_cov @ mmd_vec)
    return {
        "stat": stat,
        "cutoff": cutoff,
        "reject": int(stat > cutoff),
        "kernel_count": r,
        "lambda": ridge_lambda,
        "cond_sigma_hat": condition_number(sigma_hat),
        "cond_sigma_reg": condition_number(sigma_reg),
        "bandwidth_gammas": ";".join(f"{g:.8g}" for g in gammas),
    }


def plot_summary(summary_rows: list[dict[str, object]]) -> None:
    methods = [row["method"] for row in summary_rows]
    rates = [float(row["rejection_rate"]) for row in summary_rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(methods, rates, color=["#4C78A8", "#F58518"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rejection rate")
    ax.set_title("PathMNIST MMMD Smoke Test")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "smoke_rejection_rates.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--outer-repetitions", type=int, default=2)
    parser.add_argument("--b-bootstrap", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260526)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "scenario": "mix635_vs_mix835",
        "sample_size": args.sample_size,
        "outer_repetitions": args.outer_repetitions,
        "B_boot": args.b_bootstrap,
        "alpha": args.alpha,
        "seed": args.seed,
        "methods": METHODS,
        "checkpoint_path": str(CHECKPOINT_PATH),
    }
    (RESULTS_DIR / "smoke_config.json").write_text(json.dumps(config, indent=2))
    log(f"smoke config: {json.dumps(config, sort_keys=True)}")

    images, labels = load_test_data()
    flat_pixels = images.reshape(images.shape[0], -1).astype(np.float32) / 255.0
    rng = np.random.default_rng(args.seed)

    cnn = load_cnn_module()
    rows: list[dict[str, object]] = []
    for outer_iter in range(1, args.outer_repetitions + 1):
        start = time.time()
        x_idx, y_idx = sample_balanced_h1(labels, args.sample_size, rng)
        x_raw = flat_pixels[x_idx]
        y_raw = flat_pixels[y_idx]
        emb = cnn.extract_embeddings(CHECKPOINT_PATH, images[np.concatenate([x_idx, y_idx])], batch_size=512, layers=["final_fc128"])
        final = emb["final_fc128"]
        x_final = final[:args.sample_size]
        y_final = final[args.sample_size:]

        for method, x_feat, y_feat in [
            ("raw_pixel_gaussian5", x_raw, y_raw),
            ("cnn_final_fc128_gaussian5", x_final, y_final),
        ]:
            method_rng = np.random.default_rng(args.seed + 10000 * outer_iter + len(rows))
            out = run_mmmd_gaussian5(x_feat, y_feat, args.b_bootstrap, args.alpha, method_rng)
            rows.append({
                "scenario": "mix635_vs_mix835",
                "outer_iter": outer_iter,
                "sample_size": args.sample_size,
                "per_class_per_group": args.sample_size // 3,
                "method": method,
                "alpha": args.alpha,
                "B_boot": args.b_bootstrap,
                "kernel_count": out["kernel_count"],
                "stat": out["stat"],
                "cutoff": out["cutoff"],
                "reject": out["reject"],
                "lambda": out["lambda"],
                "cond_sigma_hat": out["cond_sigma_hat"],
                "cond_sigma_reg": out["cond_sigma_reg"],
                "bandwidth_gammas": out["bandwidth_gammas"],
                "runtime_sec": time.time() - start,
                "seed": args.seed,
            })
        log(f"outer_iter {outer_iter}/{args.outer_repetitions} finished in {time.time() - start:.2f}s")

    result_fields = [
        "scenario", "outer_iter", "sample_size", "per_class_per_group", "method", "alpha", "B_boot", "kernel_count",
        "stat", "cutoff", "reject", "lambda", "cond_sigma_hat", "cond_sigma_reg", "bandwidth_gammas", "runtime_sec", "seed",
    ]
    write_csv(RESULTS_DIR / "smoke_results.csv", result_fields, rows)
    write_csv(RESULTS_DIR / "smoke_sigma_diagnostics.csv", result_fields, rows)

    summary_rows = []
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        summary_rows.append({
            "scenario": "mix635_vs_mix835",
            "sample_size": args.sample_size,
            "method": method,
            "outer_repetitions": len(method_rows),
            "rejection_rate": float(np.mean([int(row["reject"]) for row in method_rows])),
            "mean_stat": float(np.mean([float(row["stat"]) for row in method_rows])),
            "mean_cutoff": float(np.mean([float(row["cutoff"]) for row in method_rows])),
            "mean_cond_sigma_reg": float(np.mean([float(row["cond_sigma_reg"]) for row in method_rows])),
        })
    write_csv(
        RESULTS_DIR / "smoke_summary.csv",
        ["scenario", "sample_size", "method", "outer_repetitions", "rejection_rate", "mean_stat", "mean_cutoff", "mean_cond_sigma_reg"],
        summary_rows,
    )
    plot_summary(summary_rows)
    log("smoke outputs written")


if __name__ == "__main__":
    main()
