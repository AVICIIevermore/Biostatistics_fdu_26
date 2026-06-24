#!/usr/bin/env python3
"""Semantic/residual decomposition for PathMNIST CNN-MMMD.

The decomposition is based on the frozen classifier head:

    logits = W h + b
    h_sem = projection of h onto row(W)
    h_res = h - h_sem

The formal loops match the completed PathMNIST formal experiments:
outer_repetitions=10, inner_repetitions=500, B_boot=500.
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
import torch
from torch.utils.data import DataLoader, TensorDataset


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DECHAO_ROOT = EXPERIMENT_ROOT.parent

CLASS_ROOT = DECHAO_ROOT / "medmnist_pathmnist28"
CENTER_ROOT = DECHAO_ROOT / "pathmnist_center_shift_mmmd"

DATA_PATH = CLASS_ROOT / "data" / "pathmnist.npz"
CLASS_CHECKPOINT = CLASS_ROOT / "shared_model" / "models" / "pathmnist_cnn_checkpoint.pt"
CLASS_CNN_MODULE = CLASS_ROOT / "shared_model" / "python" / "pathmnist_cnn_pipeline.py"
CENTER_CHECKPOINT = CENTER_ROOT / "shared_model" / "models" / "pathmnist_cnn_checkpoint.pt"
CENTER_CNN_MODULE = CENTER_ROOT / "shared_model" / "python" / "pathmnist_center_shift_cnn_pipeline.py"
CENTER_SPLIT_PATH = CENTER_ROOT / "configs" / "center_shift_split_seed2026_holdout20.npz"

EXISTING_CLASS_POWER_SUMMARY = CLASS_ROOT / "experiments" / "03_sample_size_power_mix635_vs_mix835" / "Results" / "sample_size_power_summary.csv"
EXISTING_CLASS_TYPE1_SUMMARY = CLASS_ROOT / "experiments" / "04_type1_mix635_vs_mix635_null" / "Results" / "type1_summary.csv"
EXISTING_CENTER_POWER_SUMMARY = CENTER_ROOT / "experiments" / "03_center_shift_power" / "Results" / "pathmnist_center_shift_summary.csv"
EXISTING_CENTER_TYPE1_SUMMARY = CENTER_ROOT / "experiments" / "04_center_shift_type1_checks" / "Results" / "pathmnist_center_shift_summary.csv"

RESULTS_DIR = EXPERIMENT_ROOT / "Results"
LOG_DIR = EXPERIMENT_ROOT / "logs"
LOG_PATH = LOG_DIR / "semantic_residual_run.log"

POWER_RAW_PATH = RESULTS_DIR / "semantic_residual_power_results.csv"
TYPE1_RAW_PATH = RESULTS_DIR / "semantic_residual_type1_results.csv"
POWER_SUMMARY_PATH = RESULTS_DIR / "semantic_residual_mmmd_summary.csv"
TYPE1_SUMMARY_PATH = RESULTS_DIR / "semantic_residual_type1_summary.csv"
POWER_PLOT_PATH = RESULTS_DIR / "semantic_vs_residual_power_plot.png"
TYPE1_PLOT_PATH = RESULTS_DIR / "semantic_vs_residual_type1_plot.png"
CONFIG_PATH = RESULTS_DIR / "semantic_residual_config.json"
DIAG_PATH = RESULTS_DIR / "projection_diagnostics.json"
INTERPRETATION_PATH = RESULTS_DIR / "short_interpretation.md"

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

ALL_REPRESENTATIONS = ["final_full", "logits", "final_semantic", "final_residual"]
COMPUTED_REPRESENTATIONS = ["logits", "final_semantic", "final_residual"]
OPTIONAL_REPRESENTATIONS = ["final_semantic", "final_residual"]

RAW_FIELDS = [
    "result_kind",
    "experiment",
    "scenario",
    "setting",
    "label",
    "label_name",
    "n",
    "representation",
    "preprocess",
    "outer_iter",
    "inner_iter",
    "seed",
    "stat",
    "cutoff",
    "reject",
    "kernel_count",
    "lambda",
    "cond_sigma_hat",
    "cond_sigma_reg",
    "runtime_sec",
    "source",
]

SUMMARY_POWER_FIELDS = [
    "experiment",
    "scenario",
    "setting",
    "label",
    "label_name",
    "n",
    "representation",
    "preprocess",
    "outer_repetitions",
    "inner_repetitions",
    "independent_tests",
    "power",
    "binomial_se",
    "ci_lower",
    "ci_upper",
    "mean_stat",
    "mean_cutoff",
    "cond_sigma_reg_summary",
    "cond_sigma_reg_stat",
    "B_boot",
    "alpha",
    "source",
    "reused_from",
]

SUMMARY_TYPE1_FIELDS = [
    "experiment",
    "scenario",
    "setting",
    "label",
    "label_name",
    "n",
    "representation",
    "preprocess",
    "outer_repetitions",
    "inner_repetitions",
    "independent_tests",
    "type1_error",
    "binomial_se",
    "ci_lower",
    "ci_upper",
    "mean_stat",
    "mean_cutoff",
    "cond_sigma_reg_summary",
    "cond_sigma_reg_stat",
    "B_boot",
    "alpha",
    "source",
    "reused_from",
]


def log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    with LOG_PATH.open("a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


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


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_pathmnist_raw() -> dict[str, np.ndarray]:
    data = np.load(DATA_PATH)
    return {
        "train_images": data["train_images"],
        "train_labels": data["train_labels"].reshape(-1).astype(int),
        "val_images": data["val_images"],
        "val_labels": data["val_labels"].reshape(-1).astype(int),
        "test_images": data["test_images"],
        "test_labels": data["test_labels"].reshape(-1).astype(int),
    }


def prepare_images(images: np.ndarray, preprocess: str) -> np.ndarray:
    arr = np.asarray(images).astype(np.float32)
    if arr.ndim != 4:
        raise ValueError(f"expected 4D image tensor, got shape {arr.shape}")
    if arr.shape[-1] == 3:
        if arr.max(initial=0) > 1.0:
            arr = arr / 255.0
        arr = np.transpose(arr, (0, 3, 1, 2))
    elif arr.shape[1] == 3:
        if arr.max(initial=0) > 1.0:
            arr = arr / 255.0
    else:
        raise ValueError(f"expected RGB images, got shape {arr.shape}")

    if preprocess == "none":
        return arr
    if preprocess == "channel_standardized":
        mean = arr.mean(axis=(2, 3), keepdims=True)
        std = arr.std(axis=(2, 3), keepdims=True)
        return (arr - mean) / np.maximum(std, 1e-6)
    raise ValueError(f"unknown preprocess: {preprocess}")


def row_space_basis(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    w64 = np.asarray(weight, dtype=np.float64)
    _, s, vh = np.linalg.svd(w64, full_matrices=False)
    if s.size == 0:
        return np.zeros((0, w64.shape[1]), dtype=np.float64), s, 0.0
    tol = max(w64.shape) * np.finfo(np.float64).eps * float(s[0])
    rank = int(np.sum(s > tol))
    return vh[:rank], s, tol


def extract_representations(
    cnn_module,
    checkpoint_path: Path,
    images: np.ndarray,
    batch_size: int,
    preprocess: str,
    tag: str,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    model, device = cnn_module.load_model_for_inference(checkpoint_path, prefer_cuda=True)
    arr = prepare_images(images, preprocess)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(arr).float()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            outputs = model.forward_features(xb)
            chunks.append(outputs["final_fc128"].detach().cpu().numpy())
    h = np.concatenate(chunks, axis=0).astype(np.float32)

    weight = model.fc2.weight.detach().cpu().numpy().astype(np.float64)
    bias = model.fc2.bias.detach().cpu().numpy().astype(np.float64)
    basis, singular_values, tol = row_space_basis(weight)
    h64 = h.astype(np.float64)
    h_sem = (h64 @ basis.T) @ basis if basis.size else np.zeros_like(h64)
    h_res = h64 - h_sem
    logits = h64 @ weight.T + bias

    full_norm = np.linalg.norm(h64, axis=1)
    sem_norm = np.linalg.norm(h_sem, axis=1)
    res_norm = np.linalg.norm(h_res, axis=1)
    diag = {
        "tag": tag,
        "checkpoint_path": str(checkpoint_path),
        "preprocess": preprocess,
        "classifier_weight_shape": list(weight.shape),
        "svd_tol": tol,
        "singular_values": singular_values.tolist(),
        "row_space_rank": int(basis.shape[0]),
        "mean_norm_final_full": float(np.mean(full_norm)),
        "mean_norm_final_semantic": float(np.mean(sem_norm)),
        "mean_norm_final_residual": float(np.mean(res_norm)),
        "mean_semantic_norm_fraction": float(np.mean(sem_norm / np.maximum(full_norm, 1e-12))),
        "mean_residual_norm_fraction": float(np.mean(res_norm / np.maximum(full_norm, 1e-12))),
    }
    log(
        f"extracted {tag} preprocess={preprocess}: n={h.shape[0]} "
        f"rank(row(W))={basis.shape[0]} mean_norm full/sem/res="
        f"{diag['mean_norm_final_full']:.4f}/{diag['mean_norm_final_semantic']:.4f}/{diag['mean_norm_final_residual']:.4f}"
    )
    reps = {
        "final_full": h,
        "logits": logits.astype(np.float32),
        "final_semantic": h_sem.astype(np.float32),
        "final_residual": h_res.astype(np.float32),
    }
    return reps, diag


def pairwise_sq_dists(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_norm = np.sum(x * x, axis=1, keepdims=True)
    y_norm = np.sum(y * y, axis=1, keepdims=True).T
    return np.maximum(x_norm + y_norm - 2.0 * (x @ y.T), 0.0)


def gaussian_gammas(x: np.ndarray, y: np.ndarray, mode: str = "gaussian5") -> np.ndarray:
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


def kernel_mats_for_features(x: np.ndarray, y: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    gammas = gaussian_gammas(x, y, "gaussian5")
    dxx = pairwise_sq_dists(x, x)
    dyy = pairwise_sq_dists(y, y)
    dxy = pairwise_sq_dists(x, y)
    return ([np.exp(-g * dxx) for g in gammas], [np.exp(-g * dyy) for g in gammas], [np.exp(-g * dxy) for g in gammas])


def condition_number(mat: np.ndarray) -> float:
    values = np.linalg.svd(mat, compute_uv=False)
    if len(values) == 0 or values[-1] <= 0 or not np.all(np.isfinite(values)):
        return float("nan")
    return float(values[0] / values[-1])


def run_mmmd_from_kernel_lists(
    kxx_list: list[np.ndarray],
    kyy_list: list[np.ndarray],
    kxy_list: list[np.ndarray],
    b_boot: int,
    alpha: float,
    rng: np.random.Generator,
) -> dict[str, object]:
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


def run_gaussian5_mmmd(x: np.ndarray, y: np.ndarray, b_boot: int, alpha: float, seed: int) -> dict[str, object]:
    kxx, kyy, kxy = kernel_mats_for_features(x, y)
    return run_mmmd_from_kernel_lists(kxx, kyy, kxy, b_boot, alpha, np.random.default_rng(seed))


def sample_balanced_h1_class(labels: np.ndarray, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if n % 3 != 0:
        raise ValueError("class-mixture sample size must be divisible by 3")
    per_class = n // 3
    by_class = {label: np.where(labels == label)[0] for label in [3, 5, 6, 8]}
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


def sample_balanced_h0_class(labels: np.ndarray, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if n % 3 != 0:
        raise ValueError("class-mixture null sample size must be divisible by 3")
    per_class = n // 3
    x_parts, y_parts = [], []
    for label in [6, 3, 5]:
        pool = np.where(labels == label)[0]
        chosen = rng.choice(pool, 2 * per_class, replace=False)
        x_parts.append(chosen[:per_class])
        y_parts.append(chosen[per_class:])
    x_idx = np.concatenate(x_parts)
    y_idx = np.concatenate(y_parts)
    rng.shuffle(x_idx)
    rng.shuffle(y_idx)
    return x_idx, y_idx


def load_center_pools(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    split = np.load(CENTER_SPLIT_PATH)
    source_idx = split["source_holdout_indices"].astype(int)
    source_images = raw["train_images"][source_idx]
    source_labels = raw["train_labels"][source_idx]
    external_images = raw["test_images"]
    external_labels = raw["test_labels"]
    return {
        "source_images": source_images,
        "source_labels": source_labels,
        "source_official_indices": source_idx,
        "external_images": external_images,
        "external_labels": external_labels,
        "external_official_indices": np.arange(external_images.shape[0], dtype=int),
    }


def sample_center_indices(
    label: int,
    setting: str,
    n: int,
    pools: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    source_pool = np.where(pools["source_labels"] == label)[0]
    external_pool = np.where(pools["external_labels"] == label)[0]
    if setting == "H1_source_vs_external":
        return (
            rng.choice(source_pool, size=n, replace=False),
            rng.choice(external_pool, size=n, replace=False),
            "source_holdout_pool",
            "external_pool",
        )
    if setting == "H0_source":
        chosen = rng.choice(source_pool, size=2 * n, replace=False)
        return chosen[:n], chosen[n:], "source_holdout_pool", "source_holdout_pool"
    if setting == "H0_external":
        chosen = rng.choice(external_pool, size=2 * n, replace=False)
        return chosen[:n], chosen[n:], "external_pool", "external_pool"
    raise ValueError(f"unknown setting: {setting}")


def features_for_center_sample(
    x_pos: np.ndarray,
    y_pos: np.ndarray,
    x_pool: str,
    y_pool: str,
    reps_source: dict[str, np.ndarray],
    reps_external: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    reps_by_pool = {"source_holdout_pool": reps_source, "external_pool": reps_external}
    return (
        {name: values[x_pos] for name, values in reps_by_pool[x_pool].items()},
        {name: values[y_pos] for name, values in reps_by_pool[y_pool].items()},
    )


def completed_outer_keys(raw_path: Path, representations: list[str], inner_repetitions: int) -> set[tuple[str, str, str, str, str, int, str, int]]:
    expected = {(inner_iter, rep) for inner_iter in range(1, inner_repetitions + 1) for rep in representations}
    rows_by_key: dict[tuple[str, str, str, str, str, int, str, int], set[tuple[int, str]]] = {}
    if not raw_path.exists():
        return set()
    with raw_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["representation"] not in representations:
                continue
            key = (
                row["result_kind"],
                row["experiment"],
                row["scenario"],
                row["setting"],
                row["label"],
                int(row["n"]),
                row["preprocess"],
                int(row["outer_iter"]),
            )
            rows_by_key.setdefault(key, set()).add((int(row["inner_iter"]), row["representation"]))
    return {key for key, seen in rows_by_key.items() if expected.issubset(seen)}


def rep_seed_offset(representation: str) -> int:
    return {
        "final_full": 0,
        "logits": 101,
        "final_semantic": 211,
        "final_residual": 307,
    }[representation]


def run_class_mixture(
    reps_test: dict[str, np.ndarray],
    labels: np.ndarray,
    sample_sizes_h1: list[int],
    sample_sizes_h0: list[int],
    outer_repetitions: int,
    inner_repetitions: int,
    b_boot: int,
    alpha: float,
    progress_every: int,
) -> None:
    done_power = completed_outer_keys(POWER_RAW_PATH, COMPUTED_REPRESENTATIONS, inner_repetitions)
    done_type1 = completed_outer_keys(TYPE1_RAW_PATH, COMPUTED_REPRESENTATIONS, inner_repetitions)
    log(f"class-mixture completed power outer keys at start={len(done_power)} type1={len(done_type1)}")

    for n in sample_sizes_h1:
        for outer_iter in range(1, outer_repetitions + 1):
            key = ("power", "class_mixture", "mix635_vs_mix835", "H1_mix635_vs_mix835", "", n, "none", outer_iter)
            if key in done_power:
                log(f"skip completed class H1 n={n} outer_iter={outer_iter}")
                continue
            rows: list[dict[str, object]] = []
            t_outer = time.time()
            for inner_iter in range(1, inner_repetitions + 1):
                if inner_iter == 1 or inner_iter % progress_every == 0 or inner_iter == inner_repetitions:
                    log(f"progress class H1 n={n} outer={outer_iter}/{outer_repetitions} inner={inner_iter}/{inner_repetitions}")
                rep_seed = 20260526 + 1_000_000 * n + 10_000 * outer_iter + inner_iter
                rng = np.random.default_rng(rep_seed)
                x_idx, y_idx = sample_balanced_h1_class(labels, n, rng)
                t0 = time.time()
                for rep in COMPUTED_REPRESENTATIONS:
                    out = run_gaussian5_mmmd(
                        reps_test[rep][x_idx],
                        reps_test[rep][y_idx],
                        b_boot,
                        alpha,
                        rep_seed + rep_seed_offset(rep),
                    )
                    rows.append(make_raw_row("power", "class_mixture", "mix635_vs_mix835", "H1_mix635_vs_mix835", "", "", n, rep, "none", outer_iter, inner_iter, rep_seed, out, time.time() - t0))
            append_csv(POWER_RAW_PATH, RAW_FIELDS, rows)
            log(f"finished class H1 n={n} outer={outer_iter}/{outer_repetitions} in {time.time() - t_outer:.2f}s")

    for n in sample_sizes_h0:
        for outer_iter in range(1, outer_repetitions + 1):
            key = ("type1", "class_mixture", "mix635_vs_mix635_null", "H0_mix635_vs_mix635", "", n, "none", outer_iter)
            if key in done_type1:
                log(f"skip completed class H0 n={n} outer_iter={outer_iter}")
                continue
            rows = []
            t_outer = time.time()
            for inner_iter in range(1, inner_repetitions + 1):
                if inner_iter == 1 or inner_iter % progress_every == 0 or inner_iter == inner_repetitions:
                    log(f"progress class H0 n={n} outer={outer_iter}/{outer_repetitions} inner={inner_iter}/{inner_repetitions}")
                rep_seed = 20260527 + 1_000_000 * n + 10_000 * outer_iter + inner_iter
                rng = np.random.default_rng(rep_seed)
                x_idx, y_idx = sample_balanced_h0_class(labels, n, rng)
                t0 = time.time()
                for rep in COMPUTED_REPRESENTATIONS:
                    out = run_gaussian5_mmmd(
                        reps_test[rep][x_idx],
                        reps_test[rep][y_idx],
                        b_boot,
                        alpha,
                        rep_seed + rep_seed_offset(rep),
                    )
                    rows.append(make_raw_row("type1", "class_mixture", "mix635_vs_mix635_null", "H0_mix635_vs_mix635", "", "", n, rep, "none", outer_iter, inner_iter, rep_seed, out, time.time() - t0))
            append_csv(TYPE1_RAW_PATH, RAW_FIELDS, rows)
            log(f"finished class H0 n={n} outer={outer_iter}/{outer_repetitions} in {time.time() - t_outer:.2f}s")


def run_center_shift(
    reps_source: dict[str, np.ndarray],
    reps_external: dict[str, np.ndarray],
    pools: dict[str, np.ndarray],
    labels: list[int],
    n_grid: list[int],
    settings: list[str],
    representations: list[str],
    preprocess: str,
    outer_repetitions: int,
    inner_repetitions: int,
    b_boot: int,
    alpha: float,
    progress_every: int,
) -> None:
    power_settings = {"H1_source_vs_external"}
    type1_settings = {"H0_source", "H0_external"}
    done_power = completed_outer_keys(POWER_RAW_PATH, representations, inner_repetitions)
    done_type1 = completed_outer_keys(TYPE1_RAW_PATH, representations, inner_repetitions)
    log(f"center-shift preprocess={preprocess} completed power outer keys={len(done_power)} type1={len(done_type1)}")

    for label in labels:
        for setting in settings:
            raw_path = POWER_RAW_PATH if setting in power_settings else TYPE1_RAW_PATH
            result_kind = "power" if setting in power_settings else "type1"
            done = done_power if setting in power_settings else done_type1
            for n in n_grid:
                for outer_iter in range(1, outer_repetitions + 1):
                    key = (result_kind, "center_shift", f"label{label}_source_vs_external", setting, str(label), n, preprocess, outer_iter)
                    if key in done:
                        log(f"skip completed center {setting} label={label} n={n} outer_iter={outer_iter} preprocess={preprocess}")
                        continue
                    rows: list[dict[str, object]] = []
                    t_outer = time.time()
                    for inner_iter in range(1, inner_repetitions + 1):
                        if inner_iter == 1 or inner_iter % progress_every == 0 or inner_iter == inner_repetitions:
                            log(
                                f"progress center {setting} label={label} n={n} preprocess={preprocess} "
                                f"outer={outer_iter}/{outer_repetitions} inner={inner_iter}/{inner_repetitions}"
                            )
                        rep_seed = 20260527 + 10_000_000 * label + 1_000_000 * n + 100_000 * outer_iter + inner_iter
                        setting_offset = {"H1_source_vs_external": 11, "H0_source": 23, "H0_external": 37}[setting]
                        rng = np.random.default_rng(rep_seed + setting_offset)
                        x_pos, y_pos, x_pool, y_pool = sample_center_indices(label, setting, n, pools, rng)
                        emb_x, emb_y = features_for_center_sample(x_pos, y_pos, x_pool, y_pool, reps_source, reps_external)
                        t0 = time.time()
                        for rep in representations:
                            out = run_gaussian5_mmmd(
                                emb_x[rep],
                                emb_y[rep],
                                b_boot,
                                alpha,
                                rep_seed + rep_seed_offset(rep),
                            )
                            rows.append(make_raw_row(result_kind, "center_shift", f"label{label}_source_vs_external", setting, str(label), CLASS_NAMES[label], n, rep, preprocess, outer_iter, inner_iter, rep_seed, out, time.time() - t0))
                    append_csv(raw_path, RAW_FIELDS, rows)
                    log(f"finished center {setting} label={label} n={n} outer={outer_iter}/{outer_repetitions} preprocess={preprocess} in {time.time() - t_outer:.2f}s")


def make_raw_row(
    result_kind: str,
    experiment: str,
    scenario: str,
    setting: str,
    label: str,
    label_name: str,
    n: int,
    representation: str,
    preprocess: str,
    outer_iter: int,
    inner_iter: int,
    seed: int,
    out: dict[str, object],
    runtime_sec: float,
) -> dict[str, object]:
    return {
        "result_kind": result_kind,
        "experiment": experiment,
        "scenario": scenario,
        "setting": setting,
        "label": label,
        "label_name": label_name,
        "n": n,
        "representation": representation,
        "preprocess": preprocess,
        "outer_iter": outer_iter,
        "inner_iter": inner_iter,
        "seed": seed,
        "stat": out["stat"],
        "cutoff": out["cutoff"],
        "reject": out["reject"],
        "kernel_count": out["kernel_count"],
        "lambda": out["lambda"],
        "cond_sigma_hat": out["cond_sigma_hat"],
        "cond_sigma_reg": out["cond_sigma_reg"],
        "runtime_sec": runtime_sec,
        "source": "computed",
    }


def binomial_interval(rate: float, m: int) -> tuple[float, float, float]:
    if m <= 0:
        return float("nan"), float("nan"), float("nan")
    se = math.sqrt(rate * (1.0 - rate) / m)
    return se, max(0.0, rate - 1.96 * se), min(1.0, rate + 1.96 * se)


def summarize_computed(raw_path: Path, result_kind: str, b_boot: int, alpha: float) -> list[dict[str, object]]:
    rows = [row for row in read_rows(raw_path) if row["result_kind"] == result_kind]
    groups = sorted({
        (
            row["experiment"],
            row["scenario"],
            row["setting"],
            row["label"],
            row["label_name"],
            int(row["n"]),
            row["representation"],
            row["preprocess"],
        )
        for row in rows
    })
    out_rows: list[dict[str, object]] = []
    for experiment, scenario, setting, label, label_name, n, representation, preprocess in groups:
        subset = [
            row for row in rows
            if row["experiment"] == experiment
            and row["scenario"] == scenario
            and row["setting"] == setting
            and row["label"] == label
            and int(row["n"]) == n
            and row["representation"] == representation
            and row["preprocess"] == preprocess
        ]
        rejects = np.array([int(row["reject"]) for row in subset], dtype=float)
        m = len(subset)
        rate = float(np.mean(rejects)) if m else float("nan")
        se, lo, hi = binomial_interval(rate, m)
        common = {
            "experiment": experiment,
            "scenario": scenario,
            "setting": setting,
            "label": label,
            "label_name": label_name,
            "n": n,
            "representation": representation,
            "preprocess": preprocess,
            "outer_repetitions": len({int(row["outer_iter"]) for row in subset}),
            "inner_repetitions": max(int(row["inner_iter"]) for row in subset) if subset else 0,
            "independent_tests": m,
            "binomial_se": se,
            "ci_lower": lo,
            "ci_upper": hi,
            "mean_stat": float(np.mean([float(row["stat"]) for row in subset])) if subset else float("nan"),
            "mean_cutoff": float(np.mean([float(row["cutoff"]) for row in subset])) if subset else float("nan"),
            "cond_sigma_reg_summary": float(np.mean([float(row["cond_sigma_reg"]) for row in subset])) if subset else float("nan"),
            "cond_sigma_reg_stat": "mean",
            "B_boot": b_boot,
            "alpha": alpha,
            "source": "computed",
            "reused_from": "",
        }
        if result_kind == "power":
            common["power"] = rate
        else:
            common["type1_error"] = rate
        out_rows.append(common)
    return out_rows


def reused_summary_row(
    result_kind: str,
    experiment: str,
    scenario: str,
    setting: str,
    label: str,
    label_name: str,
    n: int,
    rate: float,
    independent_tests: int,
    mean_stat: float,
    mean_cutoff: float,
    cond_value: float,
    cond_stat: str,
    reused_from: Path,
    b_boot: int,
    alpha: float,
    outer_repetitions: int = 10,
    inner_repetitions: int = 500,
) -> dict[str, object]:
    se, lo, hi = binomial_interval(rate, independent_tests)
    row = {
        "experiment": experiment,
        "scenario": scenario,
        "setting": setting,
        "label": label,
        "label_name": label_name,
        "n": n,
        "representation": "final_full",
        "preprocess": "none",
        "outer_repetitions": outer_repetitions,
        "inner_repetitions": inner_repetitions,
        "independent_tests": independent_tests,
        "binomial_se": se,
        "ci_lower": lo,
        "ci_upper": hi,
        "mean_stat": mean_stat,
        "mean_cutoff": mean_cutoff,
        "cond_sigma_reg_summary": cond_value,
        "cond_sigma_reg_stat": cond_stat,
        "B_boot": b_boot,
        "alpha": alpha,
        "source": "reused_final_fc128_gaussian5",
        "reused_from": str(reused_from),
    }
    if result_kind == "power":
        row["power"] = rate
    else:
        row["type1_error"] = rate
    return row


def collect_reused_final_full(sample_sizes_h1: list[int], sample_sizes_h0: list[int], center_n_grid: list[int], b_boot: int, alpha: float) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    power_rows: list[dict[str, object]] = []
    type1_rows: list[dict[str, object]] = []

    for row in read_rows(EXISTING_CLASS_POWER_SUMMARY):
        if row["method"] != "cnn_final_fc128_gaussian5" or int(row["sample_size"]) not in sample_sizes_h1:
            continue
        n = int(row["sample_size"])
        power_rows.append(reused_summary_row(
            "power", "class_mixture", "mix635_vs_mix835", "H1_mix635_vs_mix835", "", "", n,
            float(row["rejection_rate"]), int(row["independent_tests"]), float(row["mean_stat"]),
            float(row["mean_cutoff"]), float(row["mean_cond_sigma_reg"]), "mean",
            EXISTING_CLASS_POWER_SUMMARY, b_boot, alpha, int(row["outer_repetitions"]), int(row["inner_repetitions"]),
        ))

    for row in read_rows(EXISTING_CLASS_TYPE1_SUMMARY):
        if row["method"] != "cnn_final_fc128_gaussian5" or int(row["sample_size"]) not in sample_sizes_h0:
            continue
        n = int(row["sample_size"])
        type1_rows.append(reused_summary_row(
            "type1", "class_mixture", "mix635_vs_mix635_null", "H0_mix635_vs_mix635", "", "", n,
            float(row["type1_error"]), int(row["independent_tests"]), float(row["mean_stat"]),
            float(row["mean_cutoff"]), float(row["mean_cond_sigma_reg"]), "mean",
            EXISTING_CLASS_TYPE1_SUMMARY, b_boot, alpha, int(row["outer_repetitions"]), int(row["inner_repetitions"]),
        ))

    for row in read_rows(EXISTING_CENTER_POWER_SUMMARY):
        if row["method"] != "cnn_final_fc128_gaussian5" or row["setting"] != "H1_source_vs_external" or int(row["n"]) not in center_n_grid:
            continue
        label = int(row["label"])
        if label not in [6, 8]:
            continue
        power_rows.append(reused_summary_row(
            "power", "center_shift", f"label{label}_source_vs_external", row["setting"], str(label), row["label_name"], int(row["n"]),
            float(row["rejection_rate"]), int(row["n_tests_per_cell_actual"]), float(row["mean_stat"]),
            float(row["mean_cutoff"]), float(row["median_cond_sigma_reg"]), "median",
            EXISTING_CENTER_POWER_SUMMARY, int(row["B_boot_actual"]), float(row["alpha"]),
        ))

    for row in read_rows(EXISTING_CENTER_TYPE1_SUMMARY):
        if row["method"] != "cnn_final_fc128_gaussian5" or row["setting"] not in {"H0_source", "H0_external"} or int(row["n"]) not in center_n_grid:
            continue
        label = int(row["label"])
        if label not in [6, 8]:
            continue
        type1_rows.append(reused_summary_row(
            "type1", "center_shift", f"label{label}_source_vs_external", row["setting"], str(label), row["label_name"], int(row["n"]),
            float(row["rejection_rate"]), int(row["n_tests_per_cell_actual"]), float(row["mean_stat"]),
            float(row["mean_cutoff"]), float(row["median_cond_sigma_reg"]), "median",
            EXISTING_CENTER_TYPE1_SUMMARY, int(row["B_boot_actual"]), float(row["alpha"]),
        ))
    return power_rows, type1_rows


def rebuild_summaries(sample_sizes_h1: list[int], sample_sizes_h0: list[int], center_n_grid: list[int], b_boot: int, alpha: float) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    reused_power, reused_type1 = collect_reused_final_full(sample_sizes_h1, sample_sizes_h0, center_n_grid, b_boot, alpha)
    computed_power = summarize_computed(POWER_RAW_PATH, "power", b_boot, alpha)
    computed_type1 = summarize_computed(TYPE1_RAW_PATH, "type1", b_boot, alpha)
    power_rows = sorted(reused_power + computed_power, key=lambda r: (str(r["experiment"]), str(r["setting"]), str(r["label"]), int(r["n"]), str(r["preprocess"]), ALL_REPRESENTATIONS.index(str(r["representation"])) if str(r["representation"]) in ALL_REPRESENTATIONS else 99))
    type1_rows = sorted(reused_type1 + computed_type1, key=lambda r: (str(r["experiment"]), str(r["setting"]), str(r["label"]), int(r["n"]), str(r["preprocess"]), ALL_REPRESENTATIONS.index(str(r["representation"])) if str(r["representation"]) in ALL_REPRESENTATIONS else 99))
    write_csv(POWER_SUMMARY_PATH, SUMMARY_POWER_FIELDS, power_rows)
    write_csv(TYPE1_SUMMARY_PATH, SUMMARY_TYPE1_FIELDS, type1_rows)
    return power_rows, type1_rows


def display_name(row: dict[str, object]) -> str:
    rep = str(row["representation"])
    preprocess = str(row["preprocess"])
    if preprocess == "none":
        return rep
    if preprocess == "channel_standardized":
        return f"{rep} + channel std"
    return f"{rep} + {preprocess}"


def plot_power(power_rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    panels = [
        ("Class mixture: mix635 vs mix835", lambda r: r["experiment"] == "class_mixture"),
        ("Center shift: label 6", lambda r: r["experiment"] == "center_shift" and str(r["label"]) == "6"),
        ("Center shift: label 8", lambda r: r["experiment"] == "center_shift" and str(r["label"]) == "8"),
    ]
    colors = {
        "final_full": "#1f77b4",
        "logits": "#ff7f0e",
        "final_semantic": "#2ca02c",
        "final_residual": "#d62728",
    }
    for ax, (title, pred) in zip(axes, panels):
        subset = [r for r in power_rows if pred(r)]
        names = sorted({display_name(r) for r in subset})
        for name in names:
            rows = sorted([r for r in subset if display_name(r) == name], key=lambda x: int(x["n"]))
            if not rows:
                continue
            rep = str(rows[0]["representation"])
            linestyle = "--" if str(rows[0]["preprocess"]) != "none" else "-"
            ax.plot([int(r["n"]) for r in rows], [float(r["power"]) for r in rows], marker="o", label=name, color=colors.get(rep), linestyle=linestyle)
        ax.axhline(0.05, color="black", linestyle=":", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Group size n")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Empirical rejection rate / power")
    axes[-1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(POWER_PLOT_PATH, dpi=180)
    plt.close(fig)


def plot_type1(type1_rows: list[dict[str, object]]) -> None:
    fig, axes_obj = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    axes = list(axes_obj.flat)
    panels = [
        ("Class null: mix635 vs mix635", lambda r: r["experiment"] == "class_mixture"),
        ("Label 6 H0 source", lambda r: r["experiment"] == "center_shift" and str(r["label"]) == "6" and r["setting"] == "H0_source"),
        ("Label 6 H0 external", lambda r: r["experiment"] == "center_shift" and str(r["label"]) == "6" and r["setting"] == "H0_external"),
        ("Label 8 H0 source", lambda r: r["experiment"] == "center_shift" and str(r["label"]) == "8" and r["setting"] == "H0_source"),
        ("Label 8 H0 external", lambda r: r["experiment"] == "center_shift" and str(r["label"]) == "8" and r["setting"] == "H0_external"),
    ]
    colors = {
        "final_full": "#1f77b4",
        "logits": "#ff7f0e",
        "final_semantic": "#2ca02c",
        "final_residual": "#d62728",
    }
    for ax, (title, pred) in zip(axes, panels):
        subset = [r for r in type1_rows if pred(r)]
        names = sorted({display_name(r) for r in subset})
        for name in names:
            rows = sorted([r for r in subset if display_name(r) == name], key=lambda x: int(x["n"]))
            if not rows:
                continue
            rep = str(rows[0]["representation"])
            linestyle = "--" if str(rows[0]["preprocess"]) != "none" else "-"
            ax.plot([int(r["n"]) for r in rows], [float(r["type1_error"]) for r in rows], marker="o", label=name, color=colors.get(rep), linestyle=linestyle)
        ax.axhline(0.05, color="black", linestyle=":", linewidth=1)
        ax.set_ylim(0, 0.2)
        ax.set_title(title)
        ax.set_xlabel("Group size n")
        ax.set_ylabel("Empirical Type-I error")
        ax.grid(alpha=0.2)
    axes[-1].axis("off")
    axes[2].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(TYPE1_PLOT_PATH, dpi=180)
    plt.close(fig)


def witness_scores(x_ref: np.ndarray, y_ref: np.ndarray, z: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    gammas = gaussian_gammas(x_ref, y_ref, "gaussian5")
    scores = np.zeros(z.shape[0], dtype=np.float64)
    x_ref64 = x_ref.astype(np.float64)
    y_ref64 = y_ref.astype(np.float64)
    for start in range(0, z.shape[0], batch_size):
        stop = min(z.shape[0], start + batch_size)
        z_batch = z[start:stop].astype(np.float64)
        block = np.zeros(stop - start, dtype=np.float64)
        d_zx = pairwise_sq_dists(z_batch, x_ref64)
        d_zy = pairwise_sq_dists(z_batch, y_ref64)
        for gamma in gammas:
            block += np.exp(-gamma * d_zx).mean(axis=1) - np.exp(-gamma * d_zy).mean(axis=1)
        scores[start:stop] = block / len(gammas)
    return scores


def plot_grouped_images(groups: list[tuple[str, list[dict[str, object]]]], output_path: Path, suptitle: str, cols: int = 8) -> None:
    rows = len(groups)
    fig, axes_obj = plt.subplots(rows, cols, figsize=(1.75 * cols, 2.1 * rows), squeeze=False)
    for row_idx, (group_title, records) in enumerate(groups):
        for col_idx in range(cols):
            ax = axes_obj[row_idx, col_idx]
            ax.axis("off")
            if col_idx >= len(records):
                continue
            rec = records[col_idx]
            img = rec["image"]
            ax.imshow(img)
            ax.set_title(str(rec["title"]), fontsize=7)
        axes_obj[row_idx, 0].set_ylabel(group_title, fontsize=9)
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_class_witness(raw: dict[str, np.ndarray], reps_test: dict[str, np.ndarray]) -> None:
    labels = raw["test_labels"]
    images = raw["test_images"]
    rng = np.random.default_rng(20260604)
    x_idx, y_idx = sample_balanced_h1_class(labels, 120, rng)
    candidate_idx = np.concatenate([x_idx, y_idx])
    group_by_idx = {int(i): "X" for i in x_idx}
    group_by_idx.update({int(i): "Y" for i in y_idx})
    outputs = {
        "final_semantic": RESULTS_DIR / "witness_examples_class_mixture_semantic.png",
        "final_residual": RESULTS_DIR / "witness_examples_class_mixture_residual.png",
    }
    for rep, output in outputs.items():
        scores = witness_scores(reps_test[rep][x_idx], reps_test[rep][y_idx], reps_test[rep][candidate_idx])
        order_pos = np.argsort(-scores)[:8]
        order_neg = np.argsort(scores)[:8]
        groups = []
        for title, order in [("positive: X-like", order_pos), ("negative: Y-like", order_neg)]:
            records = []
            for pos in order:
                idx = int(candidate_idx[pos])
                records.append({
                    "image": images[idx],
                    "title": f"{group_by_idx[idx]} y={labels[idx]} s={scores[pos]:.3f}",
                })
            groups.append((title, records))
        plot_grouped_images(groups, output, f"Class-mixture Gaussian5 MMD witness examples: {rep}", cols=8)
        log(f"wrote {output}")


def make_center_witness(pools: dict[str, np.ndarray], reps_source: dict[str, np.ndarray], reps_external: dict[str, np.ndarray]) -> None:
    outputs = {
        "final_semantic": RESULTS_DIR / "witness_examples_center_shift_semantic.png",
        "final_residual": RESULTS_DIR / "witness_examples_center_shift_residual.png",
    }
    for rep, output in outputs.items():
        groups: list[tuple[str, list[dict[str, object]]]] = []
        for label in [6, 8]:
            rng = np.random.default_rng(20260604 + label)
            x_ref, y_ref, _, _ = sample_center_indices(label, "H1_source_vs_external", 50, pools, rng)
            source_pool = np.where(pools["source_labels"] == label)[0]
            external_pool = np.where(pools["external_labels"] == label)[0]
            z_feat = np.vstack([reps_source[rep][source_pool], reps_external[rep][external_pool]])
            scores = witness_scores(reps_source[rep][x_ref], reps_external[rep][y_ref], z_feat)
            z_images = np.concatenate([pools["source_images"][source_pool], pools["external_images"][external_pool]], axis=0)
            z_pool = np.array(["source"] * len(source_pool) + ["external"] * len(external_pool))
            pos_order = np.argsort(-scores)[:6]
            neg_order = np.argsort(scores)[:6]
            for group_title, order in [(f"label {label} positive", pos_order), (f"label {label} negative", neg_order)]:
                records = []
                for pos in order:
                    records.append({
                        "image": z_images[pos],
                        "title": f"{z_pool[pos]} s={scores[pos]:.3f}",
                    })
                groups.append((group_title, records))
        plot_grouped_images(groups, output, f"Center-shift Gaussian5 MMD witness examples: {rep}", cols=6)
        log(f"wrote {output}")


def find_summary_row(rows: list[dict[str, object]], **criteria: object) -> dict[str, object] | None:
    for row in rows:
        if all(str(row.get(k, "")) == str(v) for k, v in criteria.items()):
            return row
    return None


def write_interpretation(power_rows: list[dict[str, object]], type1_rows: list[dict[str, object]], diagnostics: list[dict[str, object]]) -> None:
    def value(row: dict[str, object] | None, field: str) -> str:
        if row is None:
            return "NA"
        return f"{float(row[field]):.4f}"

    class_n120 = [
        (rep, value(find_summary_row(power_rows, experiment="class_mixture", n=120, representation=rep, preprocess="none"), "power"))
        for rep in ALL_REPRESENTATIONS
    ]
    center_l6_n50 = [
        (rep, value(find_summary_row(power_rows, experiment="center_shift", setting="H1_source_vs_external", label="6", n=50, representation=rep, preprocess="none"), "power"))
        for rep in ALL_REPRESENTATIONS
    ]
    center_l8_n50 = [
        (rep, value(find_summary_row(power_rows, experiment="center_shift", setting="H1_source_vs_external", label="8", n=50, representation=rep, preprocess="none"), "power"))
        for rep in ALL_REPRESENTATIONS
    ]
    center_l6_n50_channel_std = [
        (rep, value(find_summary_row(power_rows, experiment="center_shift", setting="H1_source_vs_external", label="6", n=50, representation=rep, preprocess="channel_standardized"), "power"))
        for rep in OPTIONAL_REPRESENTATIONS
    ]
    center_l8_n50_channel_std = [
        (rep, value(find_summary_row(power_rows, experiment="center_shift", setting="H1_source_vs_external", label="8", n=50, representation=rep, preprocess="channel_standardized"), "power"))
        for rep in OPTIONAL_REPRESENTATIONS
    ]
    max_type1_by_rep: dict[str, float] = {}
    for row in type1_rows:
        if row["preprocess"] != "none":
            continue
        rep = str(row["representation"])
        max_type1_by_rep[rep] = max(max_type1_by_rep.get(rep, 0.0), float(row["type1_error"]))

    rank_lines = [
        f"- `{d['tag']}` / `{d['preprocess']}`: row-space rank {d['row_space_rank']} of {d['classifier_weight_shape'][1]}, "
        f"mean norm fractions semantic={d['mean_semantic_norm_fraction']:.3f}, residual={d['mean_residual_norm_fraction']:.3f}."
        for d in diagnostics
        if d["preprocess"] == "none"
    ]
    optional_rows = [r for r in power_rows if r["preprocess"] == "channel_standardized"]
    optional_note = "Not run." if not optional_rows else "Included in the summary CSVs and plotted as dashed `+ channel std` lines for center-shift semantic/residual rows."

    lines = [
        "# Semantic vs Residual MMMD Interpretation",
        "",
        "## Setup",
        "",
        "- `final_full` reuses the completed `cnn_final_fc128_gaussian5` formal PathMNIST MMMD summaries.",
        "- `logits`, `final_semantic`, and `final_residual` are newly evaluated with Gaussian5 MMMD.",
        "- The projection uses SVD of the classifier weight `W`; no CNN checkpoint is retrained.",
        "- Formal loops: outer repetitions = 10, inner independent tests = 500, multiplier bootstrap B = 500.",
        "",
        "## Projection Diagnostics",
        "",
        *rank_lines,
        "",
        "## Key Readout",
        "",
        "Class mixture power at n=120:",
        "",
        *[f"- `{rep}`: {val}" for rep, val in class_n120],
        "",
        "Center-shift power at n=50, label 6:",
        "",
        *[f"- `{rep}`: {val}" for rep, val in center_l6_n50],
        "",
        "Center-shift power at n=50, label 8:",
        "",
        *[f"- `{rep}`: {val}" for rep, val in center_l8_n50],
        "",
        "Maximum Type-I error observed across the matched null checks, by representation:",
        "",
        *[f"- `{rep}`: {max_type1_by_rep.get(rep, float('nan')):.4f}" for rep in ALL_REPRESENTATIONS],
        "",
        "## Interpretation",
        "",
        "- If `final_semantic` tracks class-mixture power and `final_residual` is weaker, the class-mixture MMMD signal is mostly classifier-used class semantics.",
        "- If `final_residual` remains strong for center-shift while `logits` or `final_semantic` are weaker, the center-shift signal includes classifier-ignored morphology, staining, scanner, or domain residual variation.",
        "- `logits` is a 9-dimensional compressed semantic representation. Similar behavior between `logits` and `final_semantic` means the classifier row space is largely enough for the detected signal; divergence means extra geometry inside the row-space projection matters.",
        "- The center-shift H0-source/H0-external rows should be read as calibration checks. High H1 power is only compelling when Type-I error for the matching representation and n is near alpha=0.05.",
        "",
        "## Optional Channel Standardization",
        "",
        optional_note,
        "",
        "Center-shift power at n=50 after per-image channel standardization:",
        "",
        "Label 6:",
        "",
        *[f"- `{rep}`: {val}" for rep, val in center_l6_n50_channel_std],
        "",
        "Label 8:",
        "",
        *[f"- `{rep}`: {val}" for rep, val in center_l8_n50_channel_std],
        "",
        "## Outputs",
        "",
        f"- Power summary: `{POWER_SUMMARY_PATH.relative_to(EXPERIMENT_ROOT)}`",
        f"- Type-I summary: `{TYPE1_SUMMARY_PATH.relative_to(EXPERIMENT_ROOT)}`",
        f"- Power plot: `{POWER_PLOT_PATH.relative_to(EXPERIMENT_ROOT)}`",
        f"- Type-I plot: `{TYPE1_PLOT_PATH.relative_to(EXPERIMENT_ROOT)}`",
        "- Witness grids: `witness_examples_*_{semantic,residual}.png`",
    ]
    INTERPRETATION_PATH.write_text("\n".join(lines) + "\n")
    log(f"wrote {INTERPRETATION_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-repetitions", type=int, default=10)
    parser.add_argument("--inner-repetitions", type=int, default=500)
    parser.add_argument("--b-bootstrap", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--class-h1-sample-sizes", type=int, nargs="+", default=[60, 90, 120])
    parser.add_argument("--class-h0-sample-sizes", type=int, nargs="+", default=[60, 120])
    parser.add_argument("--center-n-grid", type=int, nargs="+", default=[20, 30, 50])
    parser.add_argument("--center-labels", type=int, nargs="+", default=[6, 8])
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--run-channel-standardization", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-main-loops", action="store_true")
    parser.add_argument("--skip-witness", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if args.force:
        for path in [
            POWER_RAW_PATH,
            TYPE1_RAW_PATH,
            POWER_SUMMARY_PATH,
            TYPE1_SUMMARY_PATH,
            POWER_PLOT_PATH,
            TYPE1_PLOT_PATH,
            DIAG_PATH,
            INTERPRETATION_PATH,
            RESULTS_DIR / "witness_examples_class_mixture_semantic.png",
            RESULTS_DIR / "witness_examples_class_mixture_residual.png",
            RESULTS_DIR / "witness_examples_center_shift_semantic.png",
            RESULTS_DIR / "witness_examples_center_shift_residual.png",
        ]:
            if path.exists():
                path.unlink()

    config = {
        "dataset": "PathMNIST-28",
        "outer_repetitions": args.outer_repetitions,
        "inner_repetitions": args.inner_repetitions,
        "B_boot": args.b_bootstrap,
        "alpha": args.alpha,
        "class_mixture_h1": "mix635_vs_mix835",
        "class_mixture_h1_sample_sizes": args.class_h1_sample_sizes,
        "class_mixture_h0": "mix635_vs_mix635",
        "class_mixture_h0_sample_sizes": args.class_h0_sample_sizes,
        "center_shift_labels": args.center_labels,
        "center_shift_settings": ["H1_source_vs_external", "H0_source", "H0_external"],
        "center_shift_n_grid": args.center_n_grid,
        "representations": ALL_REPRESENTATIONS,
        "newly_computed_representations": COMPUTED_REPRESENTATIONS,
        "final_full_reused_from": {
            "class_power": str(EXISTING_CLASS_POWER_SUMMARY),
            "class_type1": str(EXISTING_CLASS_TYPE1_SUMMARY),
            "center_power": str(EXISTING_CENTER_POWER_SUMMARY),
            "center_type1": str(EXISTING_CENTER_TYPE1_SUMMARY),
        },
        "class_checkpoint": str(CLASS_CHECKPOINT),
        "center_checkpoint": str(CENTER_CHECKPOINT),
        "center_split_path": str(CENTER_SPLIT_PATH),
        "optional_channel_standardization": bool(args.run_channel_standardization),
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2))
    log(f"config: {json.dumps(config, sort_keys=True)}")

    raw = load_pathmnist_raw()
    class_cnn = import_module(CLASS_CNN_MODULE, "pathmnist_class_cnn_pipeline")
    center_cnn = import_module(CENTER_CNN_MODULE, "pathmnist_center_cnn_pipeline")

    diagnostics: list[dict[str, object]] = []
    t0 = time.time()
    reps_test, diag = extract_representations(class_cnn, CLASS_CHECKPOINT, raw["test_images"], args.eval_batch_size, "none", "class_mixture_test")
    diagnostics.append(diag)
    log(f"class test representation extraction total {time.time() - t0:.2f}s")

    pools = load_center_pools(raw)
    t0 = time.time()
    reps_source, diag = extract_representations(center_cnn, CENTER_CHECKPOINT, pools["source_images"], args.eval_batch_size, "none", "center_source_holdout")
    diagnostics.append(diag)
    reps_external, diag = extract_representations(center_cnn, CENTER_CHECKPOINT, pools["external_images"], args.eval_batch_size, "none", "center_external")
    diagnostics.append(diag)
    log(f"center representation extraction total {time.time() - t0:.2f}s")

    if not args.skip_main_loops:
        run_class_mixture(
            reps_test,
            raw["test_labels"],
            args.class_h1_sample_sizes,
            args.class_h0_sample_sizes,
            args.outer_repetitions,
            args.inner_repetitions,
            args.b_bootstrap,
            args.alpha,
            args.progress_every,
        )
        run_center_shift(
            reps_source,
            reps_external,
            pools,
            args.center_labels,
            args.center_n_grid,
            ["H1_source_vs_external", "H0_source", "H0_external"],
            COMPUTED_REPRESENTATIONS,
            "none",
            args.outer_repetitions,
            args.inner_repetitions,
            args.b_bootstrap,
            args.alpha,
            args.progress_every,
        )

    if args.run_channel_standardization and not args.skip_main_loops:
        t0 = time.time()
        reps_source_std, diag = extract_representations(center_cnn, CENTER_CHECKPOINT, pools["source_images"], args.eval_batch_size, "channel_standardized", "center_source_holdout")
        diagnostics.append(diag)
        reps_external_std, diag = extract_representations(center_cnn, CENTER_CHECKPOINT, pools["external_images"], args.eval_batch_size, "channel_standardized", "center_external")
        diagnostics.append(diag)
        log(f"center channel-standardized representation extraction total {time.time() - t0:.2f}s")
        run_center_shift(
            reps_source_std,
            reps_external_std,
            pools,
            args.center_labels,
            args.center_n_grid,
            ["H1_source_vs_external", "H0_source", "H0_external"],
            OPTIONAL_REPRESENTATIONS,
            "channel_standardized",
            args.outer_repetitions,
            args.inner_repetitions,
            args.b_bootstrap,
            args.alpha,
            args.progress_every,
        )

    DIAG_PATH.write_text(json.dumps(diagnostics, indent=2))
    power_rows, type1_rows = rebuild_summaries(args.class_h1_sample_sizes, args.class_h0_sample_sizes, args.center_n_grid, args.b_bootstrap, args.alpha)
    plot_power(power_rows)
    plot_type1(type1_rows)
    if not args.skip_witness:
        make_class_witness(raw, reps_test)
        make_center_witness(pools, reps_source, reps_external)
    write_interpretation(power_rows, type1_rows, diagnostics)
    log("semantic/residual MMMD experiment complete")


if __name__ == "__main__":
    main()
