#!/usr/bin/env python3
"""PathMNIST within-class center-shift MMMD runner.

Runs H1 source-vs-external and within-domain H0 checks using frozen CNN
representations trained on the independent cnn_train_pool.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "pathmnist.npz"
CONFIG_DIR = PROJECT_ROOT / "configs"
SPLIT_PATH = CONFIG_DIR / "center_shift_split_seed2026_holdout20.npz"
CHECKPOINT_PATH = PROJECT_ROOT / "shared_model" / "models" / "pathmnist_cnn_checkpoint.pt"
CNN_MODULE_PATH = PROJECT_ROOT / "shared_model" / "python" / "pathmnist_center_shift_cnn_pipeline.py"

CLASS_NAMES = {
    0: "adipose",
    1: "background",
    2: "debris",
    3: "lymphocytes",
    4: "mucus",
    5: "smooth muscle",
    6: "normal colon mucosa",
    7: "cancer-associated stroma",
    8: "colorectal adenocarcinoma epithelium",
}

ALL_METHODS = [
    "raw_pixel_gaussian5",
    "cnn_final_fc128_gaussian5",
    "cnn_multilayer_single_gaussian",
    "cnn_multilayer_gaussian15",
]
ALL_SETTINGS = ["H1_source_vs_external", "H0_source", "H0_external"]
RESULT_FIELDS = [
    "dataset",
    "label",
    "label_name",
    "setting",
    "n",
    "method",
    "outer_iter",
    "inner_iter",
    "rep_id",
    "seed",
    "stat",
    "cutoff",
    "reject",
    "kernel_count",
    "lambda",
    "cond_sigma_hat",
    "cond_sigma_reg",
    "runtime_sec",
    "source_count_available",
    "external_count_available",
    "sample_x_pool",
    "sample_y_pool",
    "sample_x_indices",
    "sample_y_indices",
    "n_tests_per_cell_actual",
    "B_boot_actual",
    "alpha",
]
SUMMARY_FIELDS = [
    "label",
    "label_name",
    "setting",
    "n",
    "method",
    "rejection_rate",
    "binomial_se",
    "ci_lower",
    "ci_upper",
    "mean_stat",
    "mean_cutoff",
    "median_cond_sigma_hat",
    "median_cond_sigma_reg",
    "median_lambda",
    "n_tests_per_cell_actual",
    "B_boot_actual",
    "alpha",
]
SKIPPED_FIELDS = [
    "label",
    "label_name",
    "setting",
    "n",
    "source_count_available",
    "external_count_available",
    "required_source_count",
    "required_external_count",
    "skip_reason",
]


class Paths:
    def __init__(self, experiment_root: Path) -> None:
        self.root = experiment_root
        self.results_dir = experiment_root / "Results"
        self.logs_dir = experiment_root / "logs"
        self.results_path = self.results_dir / "pathmnist_center_shift_results.csv"
        self.summary_path = self.results_dir / "pathmnist_center_shift_summary.csv"
        self.diag_path = self.results_dir / "pathmnist_sigma_diagnostics.csv"
        self.skipped_path = self.results_dir / "pathmnist_center_shift_skipped.csv"
        self.config_path = self.results_dir / "pathmnist_center_shift_config.json"
        self.power_plot_path = self.results_dir / "pathmnist_center_shift_power_curve.png"
        self.type1_plot_path = self.results_dir / "pathmnist_type1_check.png"
        self.gap_plot_path = self.results_dir / "pathmnist_center_shift_gap.png"
        self.cond_plot_path = self.results_dir / "pathmnist_condition_number.png"
        self.log_path = self.logs_dir / "center_shift_mmmd_run.log"


def log(paths: Paths, message: str) -> None:
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    with paths.log_path.open("a") as f:
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


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def import_cnn_module():
    spec = importlib.util.spec_from_file_location("pathmnist_center_shift_cnn_pipeline", CNN_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_pools() -> dict[str, object]:
    data = np.load(DATA_PATH)
    split = np.load(SPLIT_PATH)
    source_idx = split["source_holdout_indices"].astype(int)
    source_images = data["train_images"][source_idx]
    source_labels = data["train_labels"].reshape(-1).astype(int)[source_idx]
    external_images = data["test_images"]
    external_labels = data["test_labels"].reshape(-1).astype(int)
    return {
        "source_images": source_images,
        "source_labels": source_labels,
        "source_official_indices": source_idx,
        "external_images": external_images,
        "external_labels": external_labels,
        "external_official_indices": np.arange(external_images.shape[0], dtype=int),
    }


def flatten_pixels(images: np.ndarray) -> np.ndarray:
    return images.reshape(images.shape[0], -1).astype(np.float32) / 255.0


def count_label(labels: np.ndarray, label: int) -> int:
    return int(np.sum(labels == label))


def feasibility(label: int, setting: str, n: int, pools: dict[str, object]) -> tuple[bool, int, int, str]:
    source_count = count_label(pools["source_labels"], label)  # type: ignore[arg-type]
    external_count = count_label(pools["external_labels"], label)  # type: ignore[arg-type]
    if setting == "H1_source_vs_external":
        req_source, req_external = n, n
    elif setting == "H0_source":
        req_source, req_external = 2 * n, 0
    elif setting == "H0_external":
        req_source, req_external = 0, 2 * n
    else:
        raise ValueError(f"unknown setting: {setting}")
    ok = source_count >= req_source and external_count >= req_external
    reason = "" if ok else "insufficient disjoint samples"
    return ok, req_source, req_external, reason


def choose_without_replacement(pool: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(pool) < n:
        raise ValueError(f"pool has {len(pool)} elements, need {n}")
    return rng.choice(pool, size=n, replace=False)


def sample_indices(label: int, setting: str, n: int, pools: dict[str, object], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, str, str]:
    source_labels = pools["source_labels"]  # type: ignore[assignment]
    external_labels = pools["external_labels"]  # type: ignore[assignment]
    source_pool = np.where(source_labels == label)[0]
    external_pool = np.where(external_labels == label)[0]
    if setting == "H1_source_vs_external":
        x_pos = choose_without_replacement(source_pool, n, rng)
        y_pos = choose_without_replacement(external_pool, n, rng)
        return x_pos, y_pos, "source_holdout_pool", "external_pool"
    if setting == "H0_source":
        chosen = choose_without_replacement(source_pool, 2 * n, rng)
        return chosen[:n], chosen[n:], "source_holdout_pool", "source_holdout_pool"
    if setting == "H0_external":
        chosen = choose_without_replacement(external_pool, 2 * n, rng)
        return chosen[:n], chosen[n:], "external_pool", "external_pool"
    raise ValueError(f"unknown setting: {setting}")


def official_indices(pos: np.ndarray, pool_name: str, pools: dict[str, object]) -> np.ndarray:
    if pool_name == "source_holdout_pool":
        return pools["source_official_indices"][pos]  # type: ignore[index]
    if pool_name == "external_pool":
        return pools["external_official_indices"][pos]  # type: ignore[index]
    raise ValueError(pool_name)


def features_for_sample(
    x_pos: np.ndarray,
    y_pos: np.ndarray,
    x_pool: str,
    y_pool: str,
    raw_source: np.ndarray,
    raw_external: np.ndarray,
    emb_source: dict[str, np.ndarray],
    emb_external: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    raw_by_pool = {"source_holdout_pool": raw_source, "external_pool": raw_external}
    emb_by_pool = {"source_holdout_pool": emb_source, "external_pool": emb_external}
    raw_x = raw_by_pool[x_pool][x_pos]
    raw_y = raw_by_pool[y_pool][y_pos]
    emb_x = {name: values[x_pos] for name, values in emb_by_pool[x_pool].items()}
    emb_y = {name: values[y_pos] for name, values in emb_by_pool[y_pool].items()}
    return raw_x, raw_y, emb_x, emb_y


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


def run_methods(raw_x: np.ndarray, raw_y: np.ndarray, emb_x: dict[str, np.ndarray], emb_y: dict[str, np.ndarray], methods: list[str], b_boot: int, alpha: float, seed: int) -> dict[str, dict[str, object]]:
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
    out: dict[str, dict[str, object]] = {}
    for method_index, method in enumerate(methods):
        kxx_all: list[np.ndarray] = []
        kyy_all: list[np.ndarray] = []
        kxy_all: list[np.ndarray] = []
        for x_feat, y_feat, mode in specs[method]:
            kxx, kyy, kxy = kernel_mats_for_features(x_feat, y_feat, mode)
            kxx_all.extend(kxx)
            kyy_all.extend(kyy)
            kxy_all.extend(kxy)
        out[method] = run_mmmd_from_kernel_lists(kxx_all, kyy_all, kxy_all, b_boot, alpha, np.random.default_rng(seed + method_index))
    return out


def completed_cell_keys(results_path: Path, labels: list[int], settings: list[str], n_grid: list[int], methods: list[str], inner_repetitions: int) -> set[tuple[int, str, int, int]]:
    if not results_path.exists():
        return set()
    rows_by_key: dict[tuple[int, str, int, int], set[tuple[int, str]]] = {}
    with results_path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["label"]), row["setting"], int(row["n"]), int(row["outer_iter"]))
            rows_by_key.setdefault(key, set()).add((int(row["inner_iter"]), row["method"]))
    expected = {(inner_iter, method) for inner_iter in range(1, inner_repetitions + 1) for method in methods}
    valid_prefixes = {(label, setting, n) for label in labels for setting in settings for n in n_grid}
    return {key for key, rows in rows_by_key.items() if key[:3] in valid_prefixes and expected.issubset(rows)}


def summarize(paths: Paths) -> list[dict[str, object]]:
    raw_rows = read_rows(paths.results_path)
    summary_rows: list[dict[str, object]] = []
    groups = sorted({(int(r["label"]), r["setting"], int(r["n"]), r["method"]) for r in raw_rows})
    for label, setting, n, method in groups:
        rows = [r for r in raw_rows if int(r["label"]) == label and r["setting"] == setting and int(r["n"]) == n and r["method"] == method]
        rejects = np.array([int(r["reject"]) for r in rows], dtype=float)
        m = len(rows)
        if m == 0:
            continue
        rate = float(np.mean(rejects))
        se = float(math.sqrt(rate * (1.0 - rate) / m)) if m else float("nan")
        summary_rows.append({
            "label": label,
            "label_name": CLASS_NAMES[label],
            "setting": setting,
            "n": n,
            "method": method,
            "rejection_rate": rate,
            "binomial_se": se,
            "ci_lower": max(0.0, rate - 1.96 * se),
            "ci_upper": min(1.0, rate + 1.96 * se),
            "mean_stat": float(np.mean([float(r["stat"]) for r in rows])),
            "mean_cutoff": float(np.mean([float(r["cutoff"]) for r in rows])),
            "median_cond_sigma_hat": float(np.median([float(r["cond_sigma_hat"]) for r in rows])),
            "median_cond_sigma_reg": float(np.median([float(r["cond_sigma_reg"]) for r in rows])),
            "median_lambda": float(np.median([float(r["lambda"]) for r in rows])),
            "n_tests_per_cell_actual": m,
            "B_boot_actual": int(rows[0]["B_boot_actual"]),
            "alpha": float(rows[0]["alpha"]),
        })
    write_csv(paths.summary_path, SUMMARY_FIELDS, summary_rows)
    return summary_rows


def label_axes(labels: list[int], figsize: tuple[int, int]) -> tuple[plt.Figure, list[plt.Axes]]:
    fig, axes_obj = plt.subplots(1, len(labels), figsize=figsize, squeeze=False)
    return fig, list(axes_obj[0])


def plot_outputs(paths: Paths, summary_rows: list[dict[str, object]]) -> None:
    if not summary_rows:
        return
    labels = sorted({int(r["label"]) for r in summary_rows})
    methods = [m for m in ALL_METHODS if any(r["method"] == m for r in summary_rows)]
    alpha = float(summary_rows[0]["alpha"])

    h1_rows = [r for r in summary_rows if r["setting"] == "H1_source_vs_external"]
    if h1_rows:
        fig, axes = label_axes(labels, (6 * len(labels), 4))
        for ax, label in zip(axes, labels):
            for method in methods:
                rows = sorted([r for r in h1_rows if int(r["label"]) == label and r["method"] == method], key=lambda x: int(x["n"]))
                if rows:
                    ax.plot([int(r["n"]) for r in rows], [float(r["rejection_rate"]) for r in rows], marker="o", label=method)
            ax.axhline(alpha, color="black", linestyle="--", linewidth=1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Group size n")
            ax.set_ylabel("Rejection rate")
            ax.set_title(f"H1 source vs external: label {label}")
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(paths.power_plot_path, dpi=160)
        plt.close(fig)

    h0_rows = [r for r in summary_rows if r["setting"] in {"H0_source", "H0_external"}]
    if h0_rows:
        h0_settings = [setting for setting in ["H0_source", "H0_external"] if any(r["setting"] == setting for r in h0_rows)]
        fig, axes_obj = plt.subplots(
            len(h0_settings),
            len(labels),
            figsize=(6 * len(labels), 3.6 * len(h0_settings)),
            squeeze=False,
            sharey=True,
        )
        setting_titles = {
            "H0_source": "source internal null",
            "H0_external": "external internal null",
        }
        for row_idx, setting in enumerate(h0_settings):
            for col_idx, label in enumerate(labels):
                ax = axes_obj[row_idx, col_idx]
                for method in methods:
                    rows = sorted(
                        [r for r in h0_rows if int(r["label"]) == label and r["setting"] == setting and r["method"] == method],
                        key=lambda x: int(x["n"]),
                    )
                    if rows:
                        ax.plot([int(r["n"]) for r in rows], [float(r["rejection_rate"]) for r in rows], marker="o", label=method)
                ax.axhline(alpha, color="black", linestyle=":", linewidth=1, label="alpha=0.05")
                ax.set_ylim(0, 0.2)
                ax.set_xlabel("Group size n")
                ax.set_ylabel("Empirical Type-I error")
                ax.set_title(f"{setting_titles[setting]}: label {label}")
                ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(paths.type1_plot_path, dpi=160)
        plt.close(fig)

    if h1_rows and h0_rows:
        gap_rows = []
        for h1 in h1_rows:
            label = int(h1["label"])
            n = int(h1["n"])
            method = h1["method"]
            matched_h0 = [r for r in h0_rows if int(r["label"]) == label and int(r["n"]) == n and r["method"] == method]
            if matched_h0:
                gap_rows.append({"label": label, "n": n, "method": method, "gap": float(h1["rejection_rate"]) - float(np.mean([float(r["rejection_rate"]) for r in matched_h0]))})
        if gap_rows:
            fig, axes = label_axes(labels, (6 * len(labels), 4))
            for ax, label in zip(axes, labels):
                for method in methods:
                    rows = sorted([r for r in gap_rows if int(r["label"]) == label and r["method"] == method], key=lambda x: int(x["n"]))
                    if rows:
                        ax.plot([int(r["n"]) for r in rows], [float(r["gap"]) for r in rows], marker="o", label=method)
                ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
                ax.set_xlabel("Group size n")
                ax.set_ylabel("H1 rejection - average H0 rejection")
                ax.set_title(f"Center-shift gap: label {label}")
                ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(paths.gap_plot_path, dpi=160)
            plt.close(fig)

    diag_rows = summary_rows
    fig, axes = label_axes(labels, (6 * len(labels), 4))
    for ax, label in zip(axes, labels):
        for method in methods:
            rows = sorted([r for r in diag_rows if int(r["label"]) == label and r["method"] == method], key=lambda x: int(x["n"]))
            if rows:
                by_n: dict[int, list[float]] = {}
                for row in rows:
                    by_n.setdefault(int(row["n"]), []).append(float(row["median_cond_sigma_reg"]))
                xs = sorted(by_n)
                ys = [float(np.median(by_n[x])) for x in xs]
                ax.plot(xs, ys, marker="o", label=method)
        ax.set_yscale("log")
        ax.set_xlabel("Group size n")
        ax.set_ylabel("Median regularized condition number")
        ax.set_title(f"Covariance diagnostics: label {label}")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(paths.cond_plot_path, dpi=160)
    plt.close(fig)


def stringify_indices(values: np.ndarray) -> str:
    return ";".join(str(int(v)) for v in values.tolist())


def run(args: argparse.Namespace) -> None:
    paths = Paths(args.experiment_root)
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    cleanup = [
        paths.results_path,
        paths.summary_path,
        paths.diag_path,
        paths.skipped_path,
        paths.config_path,
        paths.power_plot_path,
        paths.type1_plot_path,
        paths.gap_plot_path,
        paths.cond_plot_path,
    ]
    if args.force:
        for path in cleanup:
            if path.exists():
                path.unlink()

    methods = args.methods
    labels = args.labels
    settings = args.settings
    n_grid = args.n_grid
    n_tests = args.outer_repetitions * args.inner_repetitions
    config = {
        "dataset": "PathMNIST-28",
        "labels": labels,
        "settings": settings,
        "n_grid": n_grid,
        "methods": methods,
        "outer_repetitions": args.outer_repetitions,
        "inner_repetitions": args.inner_repetitions,
        "n_tests_per_cell_actual": n_tests,
        "B_boot": args.b_bootstrap,
        "alpha": args.alpha,
        "seed": args.seed,
        "checkpoint_path": str(CHECKPOINT_PATH),
        "split_path": str(SPLIT_PATH),
    }
    paths.config_path.write_text(json.dumps(config, indent=2))
    log(paths, f"center-shift MMMD config: {json.dumps(config, sort_keys=True)}")

    pools = load_pools()
    raw_source = flatten_pixels(pools["source_images"])  # type: ignore[arg-type]
    raw_external = flatten_pixels(pools["external_images"])  # type: ignore[arg-type]
    cnn = import_cnn_module()
    log(paths, "extracting frozen CNN embeddings for source_holdout_pool")
    t0 = time.time()
    emb_source = cnn.extract_embeddings(CHECKPOINT_PATH, pools["source_images"], batch_size=args.eval_batch_size, layers=["layer1_gap", "layer2_gap", "final_fc128"])
    log(paths, f"source_holdout embedding extraction complete in {time.time() - t0:.2f}s")
    log(paths, "extracting frozen CNN embeddings for external_pool")
    t0 = time.time()
    emb_external = cnn.extract_embeddings(CHECKPOINT_PATH, pools["external_images"], batch_size=args.eval_batch_size, layers=["layer1_gap", "layer2_gap", "final_fc128"])
    log(paths, f"external embedding extraction complete in {time.time() - t0:.2f}s")

    skipped: list[dict[str, object]] = []
    for label in labels:
        for setting in settings:
            for n in n_grid:
                ok, req_source, req_external, reason = feasibility(label, setting, n, pools)
                if not ok:
                    skipped.append({
                        "label": label,
                        "label_name": CLASS_NAMES[label],
                        "setting": setting,
                        "n": n,
                        "source_count_available": count_label(pools["source_labels"], label),  # type: ignore[arg-type]
                        "external_count_available": count_label(pools["external_labels"], label),  # type: ignore[arg-type]
                        "required_source_count": req_source,
                        "required_external_count": req_external,
                        "skip_reason": reason,
                    })
    write_csv(paths.skipped_path, SKIPPED_FIELDS, skipped)
    if skipped:
        log(paths, f"skipping {len(skipped)} infeasible cells; see {paths.skipped_path}")

    done = completed_cell_keys(paths.results_path, labels, settings, n_grid, methods, args.inner_repetitions)
    log(paths, f"completed label/setting/n/outer keys at start: {len(done)}")

    for label in labels:
        source_count = count_label(pools["source_labels"], label)  # type: ignore[arg-type]
        external_count = count_label(pools["external_labels"], label)  # type: ignore[arg-type]
        for setting in settings:
            for n in n_grid:
                ok, _, _, _ = feasibility(label, setting, n, pools)
                if not ok:
                    continue
                for outer_iter in range(1, args.outer_repetitions + 1):
                    key = (label, setting, n, outer_iter)
                    if key in done:
                        log(paths, f"skip completed label={label} setting={setting} n={n} outer_iter={outer_iter}")
                        continue
                    outer_start = time.time()
                    rows: list[dict[str, object]] = []
                    for inner_iter in range(1, args.inner_repetitions + 1):
                        if inner_iter == 1 or inner_iter % args.progress_every == 0 or inner_iter == args.inner_repetitions:
                            log(paths, f"progress label={label} setting={setting} n={n} outer_iter={outer_iter}/{args.outer_repetitions} inner_iter={inner_iter}/{args.inner_repetitions}")
                        inner_start = time.time()
                        rep_seed = args.seed + 10_000_000 * label + 1_000_000 * n + 100_000 * outer_iter + inner_iter
                        setting_offset = {"H1_source_vs_external": 11, "H0_source": 23, "H0_external": 37}[setting]
                        rng = np.random.default_rng(rep_seed + setting_offset)
                        x_pos, y_pos, x_pool, y_pool = sample_indices(label, setting, n, pools, rng)
                        raw_x, raw_y, emb_x, emb_y = features_for_sample(x_pos, y_pos, x_pool, y_pool, raw_source, raw_external, emb_source, emb_external)
                        method_outputs = run_methods(raw_x, raw_y, emb_x, emb_y, methods, args.b_bootstrap, args.alpha, rep_seed)
                        runtime = time.time() - inner_start
                        x_official = official_indices(x_pos, x_pool, pools)
                        y_official = official_indices(y_pos, y_pool, pools)
                        for method in methods:
                            out = method_outputs[method]
                            rows.append({
                                "dataset": "PathMNIST-28",
                                "label": label,
                                "label_name": CLASS_NAMES[label],
                                "setting": setting,
                                "n": n,
                                "method": method,
                                "outer_iter": outer_iter,
                                "inner_iter": inner_iter,
                                "rep_id": f"{label}:{setting}:{n}:{outer_iter}:{inner_iter}",
                                "seed": rep_seed,
                                "stat": out["stat"],
                                "cutoff": out["cutoff"],
                                "reject": out["reject"],
                                "kernel_count": out["kernel_count"],
                                "lambda": out["lambda"],
                                "cond_sigma_hat": out["cond_sigma_hat"],
                                "cond_sigma_reg": out["cond_sigma_reg"],
                                "runtime_sec": runtime,
                                "source_count_available": source_count,
                                "external_count_available": external_count,
                                "sample_x_pool": x_pool,
                                "sample_y_pool": y_pool,
                                "sample_x_indices": stringify_indices(x_official),
                                "sample_y_indices": stringify_indices(y_official),
                                "n_tests_per_cell_actual": n_tests,
                                "B_boot_actual": args.b_bootstrap,
                                "alpha": args.alpha,
                            })
                    append_csv(paths.results_path, RESULT_FIELDS, rows)
                    append_csv(paths.diag_path, RESULT_FIELDS, rows)
                    summary_rows = summarize(paths)
                    plot_outputs(paths, summary_rows)
                    log(paths, f"finished label={label} setting={setting} n={n} outer_iter={outer_iter}/{args.outer_repetitions} inner_repetitions={args.inner_repetitions} in {time.time() - outer_start:.2f}s")

    summary_rows = summarize(paths)
    plot_outputs(paths, summary_rows)
    log(paths, "center-shift MMMD run complete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PathMNIST center-shift MMMD experiments.")
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--labels", type=int, nargs="+", default=[6, 8])
    parser.add_argument("--settings", nargs="+", choices=ALL_SETTINGS, default=ALL_SETTINGS)
    parser.add_argument("--n-grid", type=int, nargs="+", default=[20, 30, 50, 80, 100])
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=ALL_METHODS)
    parser.add_argument("--outer-repetitions", type=int, default=10)
    parser.add_argument("--inner-repetitions", type=int, default=500)
    parser.add_argument("--b-bootstrap", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
