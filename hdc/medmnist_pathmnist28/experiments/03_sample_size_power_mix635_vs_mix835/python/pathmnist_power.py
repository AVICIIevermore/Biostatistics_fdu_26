#!/usr/bin/env python3
"""PathMNIST-28 sample-size power experiment for MMMD.

This is the Phase 4 main H1 run. It uses the frozen shared CNN checkpoint and
writes incremental CSV outputs so the run can resume after interruption.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path
from typing import Iterable

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

METHODS = [
    "raw_pixel_gaussian5",
    "cnn_final_fc128_gaussian5",
    "cnn_multilayer_single_gaussian",
    "cnn_multilayer_gaussian15",
]
RESULT_FIELDS = [
    "scenario", "outer_iter", "inner_iter", "sample_size", "per_class_per_group", "method", "alpha", "B_boot", "kernel_count",
    "stat", "cutoff", "reject", "lambda", "cond_sigma_hat", "cond_sigma_reg", "runtime_sec", "seed",
]
SUMMARY_FIELDS = [
    "scenario", "sample_size", "method", "outer_repetitions", "inner_repetitions", "independent_tests",
    "rejection_rate", "mean_stat", "mean_cutoff", "mean_cond_sigma_reg",
]


def log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    with (LOG_DIR / "power_run.log").open("a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def append_csv(path: Path, fieldnames: Iterable[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


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
    return data["test_images"], data["test_labels"].reshape(-1).astype(int)


def completed_outer_keys(results_path: Path, inner_repetitions: int) -> set[tuple[int, int]]:
    if not results_path.exists():
        return set()
    rows_by_key: dict[tuple[int, int], set[tuple[int, str]]] = {}
    with results_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if "inner_iter" not in row or row["inner_iter"] == "":
                continue
            key = (int(row["sample_size"]), int(row["outer_iter"]))
            rows_by_key.setdefault(key, set()).add((int(row["inner_iter"]), row["method"]))
    expected = {(inner_iter, method) for inner_iter in range(1, inner_repetitions + 1) for method in METHODS}
    return {key for key, rows in rows_by_key.items() if expected.issubset(rows)}


def read_result_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def sample_balanced_h1(labels: np.ndarray, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if n % 3 != 0:
        raise ValueError("sample size must be divisible by 3")
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
    return np.maximum(x_norm + y_norm - 2.0 * (x @ y.T), 0.0)


def gaussian_gammas(x: np.ndarray, y: np.ndarray, mode: str) -> np.ndarray:
    z = np.vstack([x, y])
    d2 = pairwise_sq_dists(z, z)
    upper = d2[np.triu_indices_from(d2, k=1)]
    positive = upper[upper > 0]
    median_d2 = np.median(positive) if positive.size else 1.0
    if not np.isfinite(median_d2) or median_d2 <= 0:
        median_d2 = 1.0
    med_gamma = 1.0 / median_d2
    if mode == "single":
        return np.array([med_gamma], dtype=float)
    if mode == "gaussian5":
        return (2.0 ** np.arange(-2, 3)) * med_gamma
    raise ValueError(f"unknown gamma mode: {mode}")


def kernel_mats_for_features(x: np.ndarray, y: np.ndarray, mode: str) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    gammas = gaussian_gammas(x, y, mode)
    dxx = pairwise_sq_dists(x, x)
    dyy = pairwise_sq_dists(y, y)
    dxy = pairwise_sq_dists(x, y)
    return ([np.exp(-g * dxx) for g in gammas], [np.exp(-g * dyy) for g in gammas], [np.exp(-g * dxy) for g in gammas])


def condition_number(mat: np.ndarray) -> float:
    values = np.linalg.svd(mat, compute_uv=False)
    if len(values) == 0 or values[-1] <= 0 or not np.all(np.isfinite(values)):
        return float("nan")
    return float(values[0] / values[-1])


def run_mmmd_from_kernel_lists(kxx_list: list[np.ndarray], kyy_list: list[np.ndarray], kxy_list: list[np.ndarray], b_boot: int, alpha: float, rng: np.random.Generator) -> dict[str, object]:
    n = kxx_list[0].shape[0]
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
    }


def run_single_rep_methods(raw_x: np.ndarray, raw_y: np.ndarray, emb_x: dict[str, np.ndarray], emb_y: dict[str, np.ndarray], b_boot: int, alpha: float, seed: int) -> dict[str, dict[str, object]]:
    specs = {
        "raw_pixel_gaussian5": [(raw_x, raw_y, "gaussian5")],
        "cnn_final_fc128_gaussian5": [(emb_x["final_fc128"], emb_y["final_fc128"], "gaussian5")],
        "cnn_multilayer_single_gaussian": [
            (emb_x["layer1_gap"], emb_y["layer1_gap"], "single"),
            (emb_x["layer2_gap"], emb_y["layer2_gap"], "single"),
            (emb_x["final_fc128"], emb_y["final_fc128"], "single"),
        ],
        "cnn_multilayer_gaussian15": [
            (emb_x["layer1_gap"], emb_y["layer1_gap"], "gaussian5"),
            (emb_x["layer2_gap"], emb_y["layer2_gap"], "gaussian5"),
            (emb_x["final_fc128"], emb_y["final_fc128"], "gaussian5"),
        ],
    }
    out = {}
    for method_index, method in enumerate(METHODS):
        kxx_all, kyy_all, kxy_all = [], [], []
        for x_feat, y_feat, mode in specs[method]:
            kxx, kyy, kxy = kernel_mats_for_features(x_feat, y_feat, mode)
            kxx_all.extend(kxx); kyy_all.extend(kyy); kxy_all.extend(kxy)
        out[method] = run_mmmd_from_kernel_lists(kxx_all, kyy_all, kxy_all, b_boot, alpha, np.random.default_rng(seed + method_index))
    return out


def extract_test_embeddings(cnn, images: np.ndarray, batch_size: int) -> dict[str, np.ndarray]:
    log("extracting frozen CNN embeddings for full test split")
    start = time.time()
    embeddings = cnn.extract_embeddings(
        CHECKPOINT_PATH,
        images,
        batch_size=batch_size,
        layers=["layer1_gap", "layer2_gap", "final_fc128"],
    )
    log(f"full test embedding extraction complete in {time.time() - start:.2f}s")
    return embeddings


def summarize_and_plot(results_path: Path) -> None:
    raw_rows = read_result_rows(results_path)
    summary_rows: list[dict[str, object]] = []
    for sample_size in sorted({int(row["sample_size"]) for row in raw_rows}):
        for method in METHODS:
            rows = [row for row in raw_rows if int(row["sample_size"]) == sample_size and row["method"] == method]
            if not rows:
                continue
            outer_count = len({int(row["outer_iter"]) for row in rows})
            inner_count = max(int(row.get("inner_iter", 1)) for row in rows)
            summary_rows.append({
                "scenario": "mix635_vs_mix835",
                "sample_size": sample_size,
                "method": method,
                "outer_repetitions": outer_count,
                "inner_repetitions": inner_count,
                "independent_tests": len(rows),
                "rejection_rate": float(np.mean([int(row["reject"]) for row in rows])),
                "mean_stat": float(np.mean([float(row["stat"]) for row in rows])),
                "mean_cutoff": float(np.mean([float(row["cutoff"]) for row in rows])),
                "mean_cond_sigma_reg": float(np.mean([float(row["cond_sigma_reg"]) for row in rows])),
            })
    write_csv(RESULTS_DIR / "sample_size_power_summary.csv", SUMMARY_FIELDS, summary_rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in METHODS:
        rows = [row for row in summary_rows if row["method"] == method]
        if rows:
            ax.plot([row["sample_size"] for row in rows], [row["rejection_rate"] for row in rows], marker="o", label=method)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Group size n")
    ax.set_ylabel("Rejection rate")
    ax.set_title("PathMNIST Power: mix635 vs mix835")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "sample_size_power_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in METHODS:
        rows = [row for row in raw_rows if row["method"] == method]
        xs = sorted({int(row["sample_size"]) for row in rows})
        ys = [np.mean([float(row["cond_sigma_reg"]) for row in rows if int(row["sample_size"]) == x]) for x in xs]
        if xs:
            ax.plot(xs, ys, marker="o", label=method)
    ax.set_yscale("log")
    ax.set_xlabel("Group size n")
    ax.set_ylabel("Mean regularized condition number")
    ax.set_title("PathMNIST MMMD Covariance Diagnostics")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "sigma_condition_number.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[30, 60, 90, 120, 150])
    parser.add_argument("--outer-repetitions", type=int, default=10)
    parser.add_argument("--inner-repetitions", type=int, default=500)
    parser.add_argument("--b-bootstrap", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "sample_size_power_results.csv"
    diag_path = RESULTS_DIR / "sigma_diagnostics.csv"
    if args.force:
        for path in [results_path, diag_path, RESULTS_DIR / "sample_size_power_summary.csv", RESULTS_DIR / "sample_size_power_curve.png", RESULTS_DIR / "sigma_condition_number.png"]:
            if path.exists():
                path.unlink()

    config = {
        "scenario": "mix635_vs_mix835",
        "sample_sizes": args.sample_sizes,
        "outer_repetitions": args.outer_repetitions,
        "inner_repetitions": args.inner_repetitions,
        "B_boot": args.b_bootstrap,
        "alpha": args.alpha,
        "seed": args.seed,
        "methods": METHODS,
        "checkpoint_path": str(CHECKPOINT_PATH),
    }
    (RESULTS_DIR / "power_config.json").write_text(json.dumps(config, indent=2))
    log(f"power config: {json.dumps(config, sort_keys=True)}")

    images, labels = load_test_data()
    flat_pixels = images.reshape(images.shape[0], -1).astype(np.float32) / 255.0
    cnn = load_cnn_module()
    all_embeddings = extract_test_embeddings(cnn, images, args.eval_batch_size)
    done = completed_outer_keys(results_path, args.inner_repetitions)
    log(f"completed sample_size/outer_iter keys at start: {len(done)}")

    for sample_size in args.sample_sizes:
        for outer_iter in range(1, args.outer_repetitions + 1):
            key = (sample_size, outer_iter)
            if key in done:
                log(f"skip completed sample_size={sample_size} outer_iter={outer_iter}")
                continue
            outer_start = time.time()
            rows: list[dict[str, object]] = []
            for inner_iter in range(1, args.inner_repetitions + 1):
                if inner_iter == 1 or inner_iter % 50 == 0 or inner_iter == args.inner_repetitions:
                    log(
                        f"progress sample_size={sample_size} outer_iter={outer_iter}/{args.outer_repetitions} "
                        f"inner_iter={inner_iter}/{args.inner_repetitions}"
                    )
                inner_start = time.time()
                rep_seed = args.seed + 1_000_000 * sample_size + 10_000 * outer_iter + inner_iter
                rng = np.random.default_rng(rep_seed)
                x_idx, y_idx = sample_balanced_h1(labels, sample_size, rng)
                raw_x = flat_pixels[x_idx]
                raw_y = flat_pixels[y_idx]
                emb_x = {name: value[x_idx] for name, value in all_embeddings.items()}
                emb_y = {name: value[y_idx] for name, value in all_embeddings.items()}
                method_outputs = run_single_rep_methods(raw_x, raw_y, emb_x, emb_y, args.b_bootstrap, args.alpha, rep_seed)

                runtime = time.time() - inner_start
                for method in METHODS:
                    out = method_outputs[method]
                    rows.append({
                        "scenario": "mix635_vs_mix835",
                        "outer_iter": outer_iter,
                        "inner_iter": inner_iter,
                        "sample_size": sample_size,
                        "per_class_per_group": sample_size // 3,
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
                        "runtime_sec": runtime,
                        "seed": rep_seed,
                    })
            append_csv(results_path, RESULT_FIELDS, rows)
            append_csv(diag_path, RESULT_FIELDS, rows)
            summarize_and_plot(results_path)
            log(
                f"finished sample_size={sample_size} outer_iter={outer_iter}/{args.outer_repetitions} "
                f"inner_repetitions={args.inner_repetitions} in {time.time() - outer_start:.2f}s"
            )

    summarize_and_plot(results_path)
    log("Phase 4 power run complete")


if __name__ == "__main__":
    main()
