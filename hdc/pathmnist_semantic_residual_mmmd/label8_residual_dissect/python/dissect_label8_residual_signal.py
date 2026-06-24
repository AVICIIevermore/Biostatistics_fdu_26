#!/usr/bin/env python3
"""Dissect why PathMNIST label-8 center-shift is residual-dominated.

This script reuses the existing PathMNIST center-shift CNN checkpoint and
centered-W semantic/residual decomposition. It does not retrain CNNs.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.stats import rankdata
from torch.utils.data import DataLoader, TensorDataset


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_ROOT = EXPERIMENT_ROOT.parent
VALIDITY_SCRIPT = SEMANTIC_ROOT / "validity_controls" / "python" / "semantic_residual_validity_controls.py"

RESULTS_DIR = EXPERIMENT_ROOT / "Results"
FEATURE_DIR = EXPERIMENT_ROOT / "Features"
LOG_DIR = EXPERIMENT_ROOT / "logs"
LOG_PATH = LOG_DIR / "label8_residual_dissect_run.log"

FEATURE_CACHE = FEATURE_DIR / "centered_center_pools_features.npz"
CONFIG_PATH = RESULTS_DIR / "label8_residual_dissect_config.json"
DIAG_PATH = RESULTS_DIR / "centered_feature_diagnostics.csv"

CROSS_LABEL_RAW = RESULTS_DIR / "cross_label_domain_axis_transfer_repeats.csv"
CROSS_LABEL_SUMMARY = RESULTS_DIR / "cross_label_domain_axis_transfer.csv"
CROSS_LABEL_PLOT = RESULTS_DIR / "cross_label_domain_axis_transfer.png"

COLOR_AUC_RAW = RESULTS_DIR / "color_domain_probe_auc_repeats.csv"
COLOR_AUC_SUMMARY = RESULTS_DIR / "color_domain_probe_auc.csv"

COLOR_ONLY_RAW = RESULTS_DIR / "color_only_mmmd_results.csv"
COLOR_ONLY_SUMMARY = RESULTS_DIR / "color_only_mmmd_summary.csv"

COLOR_ADJ_RAW = RESULTS_DIR / "color_adjusted_embedding_mmmd_results.csv"
COLOR_ADJ_SUMMARY = RESULTS_DIR / "color_adjusted_embedding_mmmd_summary.csv"
COLOR_ADJ_PLOT = RESULTS_DIR / "color_adjusted_center_shift_plot.png"

INTRINSIC_RAW = RESULTS_DIR / "residual_intrinsic_dimension_results.csv"
INTRINSIC_SUMMARY = RESULTS_DIR / "residual_intrinsic_dimension_summary.csv"
INTRINSIC_PLOT = RESULTS_DIR / "residual_intrinsic_dimension_curve.png"

REPORT_PATH = RESULTS_DIR / "label8_residual_dissect_short_report.md"

CLASS_NAMES = {
    6: "normal colon mucosa",
    8: "colorectal adenocarcinoma epithelium",
}

MMMD_RAW_FIELDS = [
    "result_kind",
    "experiment",
    "scenario",
    "setting",
    "label",
    "label_name",
    "n",
    "representation",
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
    "B_boot",
    "alpha",
]

MMMD_SUMMARY_FIELDS = [
    "result_kind",
    "experiment",
    "scenario",
    "setting",
    "label",
    "label_name",
    "n",
    "representation",
    "outer_repetitions",
    "inner_repetitions",
    "independent_tests",
    "rejection_rate",
    "power",
    "type1_error",
    "binomial_se",
    "ci_lower",
    "ci_upper",
    "mean_stat",
    "mean_cutoff",
    "mean_cond_sigma_reg",
    "B_boot",
    "alpha",
]

CROSS_LABEL_RAW_FIELDS = [
    "train_label",
    "train_label_name",
    "test_label",
    "test_label_name",
    "representation",
    "axis_method",
    "repeat",
    "train_source_n",
    "train_external_n",
    "test_source_n",
    "test_external_n",
    "auc",
    "accuracy",
    "effect_size",
]

CROSS_LABEL_SUMMARY_FIELDS = [
    "train_label",
    "train_label_name",
    "test_label",
    "test_label_name",
    "representation",
    "axis_method",
    "repeats",
    "mean_auc",
    "sd_auc",
    "mean_accuracy",
    "sd_accuracy",
    "mean_effect_size",
    "sd_effect_size",
    "mean_train_source_n",
    "mean_train_external_n",
    "mean_test_source_n",
    "mean_test_external_n",
]


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


VALID = import_module(VALIDITY_SCRIPT, "semantic_residual_validity_controls")
BASE = VALID.BASE


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


def write_csv(path: Path, fieldnames: Iterable[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, fieldnames: Iterable[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def binomial_interval(rate: float, n: int) -> tuple[float, float, float]:
    if n <= 0 or not np.isfinite(rate):
        return float("nan"), float("nan"), float("nan")
    se = math.sqrt(rate * (1.0 - rate) / n)
    return se, max(0.0, rate - 1.96 * se), min(1.0, rate + 1.96 * se)


def stable_seed_offset(text: str) -> int:
    return int(sum((idx + 1) * ord(ch) for idx, ch in enumerate(text)) % 1_000_000)


def fit_pca_basis(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    x64 = np.asarray(x, dtype=np.float64)
    mean = x64.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(x64 - mean, full_matrices=False)
    return mean.astype(np.float32), vh[:k].astype(np.float32)


def apply_pca(x: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return ((x.astype(np.float32) - mean) @ basis.T).astype(np.float32)


def color_features(images: np.ndarray) -> np.ndarray:
    arr = np.asarray(images, dtype=np.float32)
    if arr.max(initial=0) > 1.0:
        arr = arr / 255.0
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"expected NHWC RGB images, got {arr.shape}")
    means = arr.mean(axis=(1, 2))
    stds = arr.std(axis=(1, 2))
    quant = np.quantile(arr, [0.10, 0.50, 0.90], axis=(1, 2))
    quant = np.transpose(quant, (1, 0, 2)).reshape(arr.shape[0], 9)
    return np.concatenate([means, stds, quant], axis=1).astype(np.float32)


def extract_centered_reps(cnn_module, checkpoint_path: Path, images: np.ndarray, batch_size: int, pool_tag: str) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    model, device = cnn_module.load_model_for_inference(checkpoint_path, prefer_cuda=True)
    arr = BASE.prepare_images(images, "none")
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
    h = np.concatenate(chunks, axis=0).astype(np.float64)

    weight = model.fc2.weight.detach().cpu().numpy().astype(np.float64)
    weight_centered = weight - weight.mean(axis=0, keepdims=True)
    basis_w, singular_w, _ = BASE.row_space_basis(weight)
    basis_wc, singular_wc, _ = BASE.row_space_basis(weight_centered)
    semantic_centered = (h @ basis_wc.T) @ basis_wc if basis_wc.size else np.zeros_like(h)
    residual_full = h - semantic_centered

    full_norm = np.linalg.norm(h, axis=1)
    sem_norm = np.linalg.norm(semantic_centered, axis=1)
    res_norm = np.linalg.norm(residual_full, axis=1)
    diag = {
        "pool_tag": pool_tag,
        "checkpoint_path": str(checkpoint_path),
        "rank_W": int(basis_w.shape[0]),
        "rank_Wc": int(basis_wc.shape[0]),
        "singular_values_W": ";".join(f"{v:.10g}" for v in singular_w),
        "singular_values_Wc": ";".join(f"{v:.10g}" for v in singular_wc),
        "mean_norm_final_full": float(np.mean(full_norm)),
        "mean_norm_semantic_centered": float(np.mean(sem_norm)),
        "mean_norm_residual_full": float(np.mean(res_norm)),
        "mean_semantic_centered_norm_fraction": float(np.mean(sem_norm / np.maximum(full_norm, 1e-12))),
        "mean_residual_full_norm_fraction": float(np.mean(res_norm / np.maximum(full_norm, 1e-12))),
    }
    reps = {
        "semantic_centered": semantic_centered.astype(np.float32),
        "residual_full": residual_full.astype(np.float32),
    }
    log(
        f"extracted {pool_tag}: n={h.shape[0]} rank(W)={diag['rank_W']} rank(Wc)={diag['rank_Wc']} "
        f"mean_norm sem/res={diag['mean_norm_semantic_centered']:.4f}/{diag['mean_norm_residual_full']:.4f}"
    )
    return reps, diag


def load_or_build_features(eval_batch_size: int, force_features: bool) -> dict[str, np.ndarray]:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    if FEATURE_CACHE.exists() and not force_features:
        log(f"load feature cache: {FEATURE_CACHE}")
        loaded = np.load(FEATURE_CACHE)
        return {key: loaded[key] for key in loaded.files}

    raw = BASE.load_pathmnist_raw()
    pools = BASE.load_center_pools(raw)
    cnn_module = import_module(BASE.CENTER_CNN_MODULE, "pathmnist_center_shift_cnn_pipeline")
    reps_source, diag_source = extract_centered_reps(cnn_module, BASE.CENTER_CHECKPOINT, pools["source_images"], eval_batch_size, "center_source_holdout")
    reps_external, diag_external = extract_centered_reps(cnn_module, BASE.CENTER_CHECKPOINT, pools["external_images"], eval_batch_size, "center_external")

    residual_combined = np.vstack([reps_source["residual_full"], reps_external["residual_full"]])
    mean8, basis8 = fit_pca_basis(residual_combined, 8)
    source_top8 = apply_pca(reps_source["residual_full"], mean8, basis8)
    external_top8 = apply_pca(reps_external["residual_full"], mean8, basis8)

    features = {
        "source_labels": pools["source_labels"].astype(np.int64),
        "external_labels": pools["external_labels"].astype(np.int64),
        "source_semantic_centered": reps_source["semantic_centered"],
        "external_semantic_centered": reps_external["semantic_centered"],
        "source_residual_full": reps_source["residual_full"],
        "external_residual_full": reps_external["residual_full"],
        "source_residual_top8_pca": source_top8,
        "external_residual_top8_pca": external_top8,
        "source_color15": color_features(pools["source_images"]),
        "external_color15": color_features(pools["external_images"]),
    }
    np.savez_compressed(FEATURE_CACHE, **features)
    write_csv(DIAG_PATH, list(diag_source.keys()), [diag_source, diag_external])
    log(f"wrote feature cache: {FEATURE_CACHE}")
    return features


def standardize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((x_train - mean) / std).astype(np.float64), ((x_test - mean) / std).astype(np.float64), mean, std


def auc_from_scores(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    rank_sum_pos = float(np.sum(ranks[y == 1]))
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def effect_size_from_scores(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    src = scores[y == 0]
    ext = scores[y == 1]
    if len(src) < 2 or len(ext) < 2:
        return float("nan")
    pooled = math.sqrt(((len(src) - 1) * np.var(src, ddof=1) + (len(ext) - 1) * np.var(ext, ddof=1)) / (len(src) + len(ext) - 2))
    return float((np.mean(ext) - np.mean(src)) / max(pooled, 1e-12))


def fit_logistic_axis(x_train: np.ndarray, y_train: np.ndarray, l2: float) -> tuple[np.ndarray, float]:
    x = np.asarray(x_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    d = x.shape[1]

    def loss_grad(params: np.ndarray) -> tuple[float, np.ndarray]:
        w = params[:d]
        b = params[d]
        z = x @ w + b
        loss_terms = np.logaddexp(0.0, z) - y * z
        prob = 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))
        diff = prob - y
        loss = float(np.mean(loss_terms) + 0.5 * l2 * np.dot(w, w))
        grad_w = (x.T @ diff) / x.shape[0] + l2 * w
        grad_b = np.array([np.mean(diff)])
        return loss, np.concatenate([grad_w, grad_b])

    result = minimize(
        fun=lambda p: loss_grad(p)[0],
        x0=np.zeros(d + 1, dtype=np.float64),
        jac=lambda p: loss_grad(p)[1],
        method="L-BFGS-B",
        options={"maxiter": 200, "ftol": 1e-9},
    )
    params = result.x
    return params[:d], float(params[d])


def split_domain_data(
    source_x: np.ndarray,
    external_x: np.ndarray,
    train_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_bal = min(len(source_x), len(external_x))
    n_train = max(2, int(round(train_fraction * n_bal)))
    n_train = min(n_train, n_bal - 2)
    source_perm = rng.permutation(len(source_x))[:n_bal]
    external_perm = rng.permutation(len(external_x))[:n_bal]
    src_train = source_x[source_perm[:n_train]]
    ext_train = external_x[external_perm[:n_train]]
    src_test = source_x[source_perm[n_train:]]
    ext_test = external_x[external_perm[n_train:]]
    return src_train, ext_train, src_test, ext_test


def domain_axis_metrics(
    train_source: np.ndarray,
    train_external: np.ndarray,
    test_source: np.ndarray,
    test_external: np.ndarray,
    axis_method: str,
    l2: float,
) -> dict[str, float]:
    x_train = np.vstack([train_source, train_external])
    y_train = np.concatenate([np.zeros(len(train_source)), np.ones(len(train_external))]).astype(int)
    x_test = np.vstack([test_source, test_external])
    y_test = np.concatenate([np.zeros(len(test_source)), np.ones(len(test_external))]).astype(int)
    x_train_z, x_test_z, _, _ = standardize_train_test(x_train, x_test)

    if axis_method == "logistic_regression":
        w, b = fit_logistic_axis(x_train_z, y_train, l2=l2)
        scores_train = x_train_z @ w + b
        scores_test = x_test_z @ w + b
        threshold = 0.0
    elif axis_method == "mean_difference":
        source_mean = x_train_z[y_train == 0].mean(axis=0)
        external_mean = x_train_z[y_train == 1].mean(axis=0)
        w = external_mean - source_mean
        scores_train = x_train_z @ w
        scores_test = x_test_z @ w
        threshold = 0.5 * (scores_train[y_train == 0].mean() + scores_train[y_train == 1].mean())
    else:
        raise ValueError(f"unknown axis method: {axis_method}")

    preds = (scores_test >= threshold).astype(int)
    return {
        "auc": float(auc_from_scores(y_test, scores_test)),
        "accuracy": float(np.mean(preds == y_test)),
        "effect_size": effect_size_from_scores(y_test, scores_test),
    }


def run_cross_label_domain_axis(features: dict[str, np.ndarray], repeats: int, train_fraction: float, l2: float, force: bool) -> list[dict[str, object]]:
    if CROSS_LABEL_SUMMARY.exists() and not force:
        log("skip cross-label domain-axis transfer; summary already exists")
        return read_rows(CROSS_LABEL_SUMMARY)
    if force:
        for path in [CROSS_LABEL_RAW, CROSS_LABEL_SUMMARY, CROSS_LABEL_PLOT]:
            if path.exists():
                path.unlink()

    label_masks = {
        "source": {label: features["source_labels"] == label for label in [6, 8]},
        "external": {label: features["external_labels"] == label for label in [6, 8]},
    }
    cases = [(8, 8), (8, 6), (6, 6), (6, 8)]
    reps = ["residual_top8_pca", "residual_full"]
    methods = ["logistic_regression", "mean_difference"]
    rows: list[dict[str, object]] = []
    for train_label, test_label in cases:
        for rep in reps:
            source_key = f"source_{rep}"
            external_key = f"external_{rep}"
            train_source_all = features[source_key][label_masks["source"][train_label]]
            train_external_all = features[external_key][label_masks["external"][train_label]]
            test_source_all = features[source_key][label_masks["source"][test_label]]
            test_external_all = features[external_key][label_masks["external"][test_label]]
            for repeat in range(1, repeats + 1):
                rng = np.random.default_rng(20260609 + 100_000 * train_label + 10_000 * test_label + 100 * repeat + stable_seed_offset(rep))
                src_train, ext_train, src_hold, ext_hold = split_domain_data(train_source_all, train_external_all, train_fraction, rng)
                if train_label == test_label:
                    test_source = src_hold
                    test_external = ext_hold
                else:
                    test_source = test_source_all
                    test_external = test_external_all
                for method in methods:
                    metrics = domain_axis_metrics(src_train, ext_train, test_source, test_external, method, l2)
                    rows.append(
                        {
                            "train_label": train_label,
                            "train_label_name": CLASS_NAMES[train_label],
                            "test_label": test_label,
                            "test_label_name": CLASS_NAMES[test_label],
                            "representation": rep,
                            "axis_method": method,
                            "repeat": repeat,
                            "train_source_n": len(src_train),
                            "train_external_n": len(ext_train),
                            "test_source_n": len(test_source),
                            "test_external_n": len(test_external),
                            **metrics,
                        }
                    )
        log(f"finished domain-axis cases train_label={train_label}")
    write_csv(CROSS_LABEL_RAW, CROSS_LABEL_RAW_FIELDS, rows)
    summary = summarize_axis_rows(rows, CROSS_LABEL_SUMMARY)
    plot_cross_label_transfer(summary)
    log(f"wrote cross-label transfer outputs: {CROSS_LABEL_SUMMARY}")
    return summary


def summarize_axis_rows(rows: list[dict[str, object]], output_path: Path) -> list[dict[str, object]]:
    groups = sorted(
        {
            (
                int(row["train_label"]),
                int(row["test_label"]),
                str(row["representation"]),
                str(row["axis_method"]),
            )
            for row in rows
        }
    )
    out: list[dict[str, object]] = []
    for train_label, test_label, rep, method in groups:
        subset = [
            row
            for row in rows
            if int(row["train_label"]) == train_label
            and int(row["test_label"]) == test_label
            and str(row["representation"]) == rep
            and str(row["axis_method"]) == method
        ]
        out.append(
            {
                "train_label": train_label,
                "train_label_name": CLASS_NAMES[train_label],
                "test_label": test_label,
                "test_label_name": CLASS_NAMES[test_label],
                "representation": rep,
                "axis_method": method,
                "repeats": len(subset),
                "mean_auc": float(np.mean([float(row["auc"]) for row in subset])),
                "sd_auc": float(np.std([float(row["auc"]) for row in subset], ddof=1)) if len(subset) > 1 else 0.0,
                "mean_accuracy": float(np.mean([float(row["accuracy"]) for row in subset])),
                "sd_accuracy": float(np.std([float(row["accuracy"]) for row in subset], ddof=1)) if len(subset) > 1 else 0.0,
                "mean_effect_size": float(np.mean([float(row["effect_size"]) for row in subset])),
                "sd_effect_size": float(np.std([float(row["effect_size"]) for row in subset], ddof=1)) if len(subset) > 1 else 0.0,
                "mean_train_source_n": float(np.mean([float(row["train_source_n"]) for row in subset])),
                "mean_train_external_n": float(np.mean([float(row["train_external_n"]) for row in subset])),
                "mean_test_source_n": float(np.mean([float(row["test_source_n"]) for row in subset])),
                "mean_test_external_n": float(np.mean([float(row["test_external_n"]) for row in subset])),
            }
        )
    write_csv(output_path, CROSS_LABEL_SUMMARY_FIELDS, out)
    return out


def run_color_domain_probe(features: dict[str, np.ndarray], repeats: int, train_fraction: float, l2: float, force: bool) -> list[dict[str, object]]:
    if COLOR_AUC_SUMMARY.exists() and not force:
        log("skip color domain probe; summary already exists")
        return read_rows(COLOR_AUC_SUMMARY)
    if force:
        for path in [COLOR_AUC_RAW, COLOR_AUC_SUMMARY]:
            if path.exists():
                path.unlink()
    rows: list[dict[str, object]] = []
    for label in [6, 8]:
        source_all = features["source_color15"][features["source_labels"] == label]
        external_all = features["external_color15"][features["external_labels"] == label]
        for repeat in range(1, repeats + 1):
            rng = np.random.default_rng(20260610 + 100_000 * label + repeat)
            src_train, ext_train, src_test, ext_test = split_domain_data(source_all, external_all, train_fraction, rng)
            for method in ["logistic_regression", "mean_difference"]:
                metrics = domain_axis_metrics(src_train, ext_train, src_test, ext_test, method, l2)
                rows.append(
                    {
                        "train_label": label,
                        "train_label_name": CLASS_NAMES[label],
                        "test_label": label,
                        "test_label_name": CLASS_NAMES[label],
                        "representation": "color15",
                        "axis_method": method,
                        "repeat": repeat,
                        "train_source_n": len(src_train),
                        "train_external_n": len(ext_train),
                        "test_source_n": len(src_test),
                        "test_external_n": len(ext_test),
                        **metrics,
                    }
                )
        log(f"finished color domain probe label={label}")
    write_csv(COLOR_AUC_RAW, CROSS_LABEL_RAW_FIELDS, rows)
    return summarize_axis_rows(rows, COLOR_AUC_SUMMARY)


def residualize_by_color(source_embedding: np.ndarray, external_embedding: np.ndarray, source_color: np.ndarray, external_color: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    color = np.vstack([source_color, external_color]).astype(np.float64)
    c_mean = color.mean(axis=0, keepdims=True)
    c_std = color.std(axis=0, keepdims=True)
    c_std = np.where(c_std < 1e-6, 1.0, c_std)
    color_z = (color - c_mean) / c_std
    design = np.concatenate([np.ones((color_z.shape[0], 1)), color_z], axis=1)
    emb = np.vstack([source_embedding, external_embedding]).astype(np.float64)
    beta, *_ = np.linalg.lstsq(design, emb, rcond=None)
    resid = emb - design @ beta
    n_source = len(source_embedding)
    return resid[:n_source].astype(np.float32), resid[n_source:].astype(np.float32)


def build_color_adjusted_reps(features: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    # Fit color adjustment within each target label. This is stricter for the
    # label-specific center-shift question than one global color regression.
    semantic_src_adj = np.zeros_like(features["source_semantic_centered"], dtype=np.float32)
    semantic_ext_adj = np.zeros_like(features["external_semantic_centered"], dtype=np.float32)
    residual_src_adj = np.zeros_like(features["source_residual_full"], dtype=np.float32)
    residual_ext_adj = np.zeros_like(features["external_residual_full"], dtype=np.float32)
    residual_src_top8 = np.zeros((features["source_residual_full"].shape[0], 8), dtype=np.float32)
    residual_ext_top8 = np.zeros((features["external_residual_full"].shape[0], 8), dtype=np.float32)

    for label in [6, 8]:
        src_mask = features["source_labels"] == label
        ext_mask = features["external_labels"] == label
        sem_src_l, sem_ext_l = residualize_by_color(
            features["source_semantic_centered"][src_mask],
            features["external_semantic_centered"][ext_mask],
            features["source_color15"][src_mask],
            features["external_color15"][ext_mask],
        )
        res_src_l, res_ext_l = residualize_by_color(
            features["source_residual_full"][src_mask],
            features["external_residual_full"][ext_mask],
            features["source_color15"][src_mask],
            features["external_color15"][ext_mask],
        )
        mean8, basis8 = fit_pca_basis(np.vstack([res_src_l, res_ext_l]), 8)
        semantic_src_adj[src_mask] = sem_src_l
        semantic_ext_adj[ext_mask] = sem_ext_l
        residual_src_adj[src_mask] = res_src_l
        residual_ext_adj[ext_mask] = res_ext_l
        residual_src_top8[src_mask] = apply_pca(res_src_l, mean8, basis8)
        residual_ext_top8[ext_mask] = apply_pca(res_ext_l, mean8, basis8)

    source_reps = {
        "color15": features["source_color15"],
        "semantic_centered_color_adjusted": semantic_src_adj,
        "residual_full_color_adjusted": residual_src_adj,
        "residual_top8_pca_color_adjusted": residual_src_top8,
    }
    external_reps = {
        "color15": features["external_color15"],
        "semantic_centered_color_adjusted": semantic_ext_adj,
        "residual_full_color_adjusted": residual_ext_adj,
        "residual_top8_pca_color_adjusted": residual_ext_top8,
    }
    return source_reps, external_reps


def build_intrinsic_dimension_reps(features: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    mask_source8 = features["source_labels"] == 8
    mask_external8 = features["external_labels"] == 8
    combined_label8 = np.vstack([features["source_residual_full"][mask_source8], features["external_residual_full"][mask_external8]])
    mean32, basis32 = fit_pca_basis(combined_label8, 32)
    source_reps: dict[str, np.ndarray] = {}
    external_reps: dict[str, np.ndarray] = {}
    for k in [1, 2, 4, 8, 16, 32]:
        key = f"residual_pca_k{k}"
        source_reps[key] = apply_pca(features["source_residual_full"], mean32, basis32[:k])
        external_reps[key] = apply_pca(features["external_residual_full"], mean32, basis32[:k])
    return source_reps, external_reps


def completed_mmmd_outer_keys(path: Path, representations: list[str], inner_repetitions: int) -> set[tuple[str, str, str, str, int, int]]:
    expected = {(inner, rep) for inner in range(1, inner_repetitions + 1) for rep in representations}
    seen_by_key: dict[tuple[str, str, str, str, int, int], set[tuple[int, str]]] = {}
    for row in read_rows(path):
        key = (
            row["result_kind"],
            row["scenario"],
            row["setting"],
            row["label"],
            int(row["n"]),
            int(row["outer_iter"]),
        )
        seen_by_key.setdefault(key, set()).add((int(row["inner_iter"]), row["representation"]))
    return {key for key, seen in seen_by_key.items() if expected.issubset(seen)}


def features_for_sample(
    x_pos: np.ndarray,
    y_pos: np.ndarray,
    x_pool: str,
    y_pool: str,
    reps_source: dict[str, np.ndarray],
    reps_external: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    by_pool = {"source_holdout_pool": reps_source, "external_pool": reps_external}
    return {key: val[x_pos] for key, val in by_pool[x_pool].items()}, {key: val[y_pos] for key, val in by_pool[y_pool].items()}


def run_mmmd_grid(
    raw_path: Path,
    experiment: str,
    reps_source: dict[str, np.ndarray],
    reps_external: dict[str, np.ndarray],
    pools: dict[str, np.ndarray],
    labels: list[int],
    n_grid: list[int],
    settings: list[str],
    representations: list[str],
    outer_repetitions: int,
    inner_repetitions: int,
    b_boot: int,
    alpha: float,
    progress_every: int,
    seed_base: int,
) -> None:
    done = completed_mmmd_outer_keys(raw_path, representations, inner_repetitions)
    log(f"{experiment} completed outer keys={len(done)}")
    for label in labels:
        for setting in settings:
            result_kind = "power" if setting == "H1_source_vs_external" else "type1"
            for n in n_grid:
                for outer_iter in range(1, outer_repetitions + 1):
                    scenario = f"label{label}_source_vs_external"
                    key = (result_kind, scenario, setting, str(label), n, outer_iter)
                    if key in done:
                        log(f"skip {experiment} {setting} label={label} n={n} outer={outer_iter}")
                        continue
                    rows: list[dict[str, object]] = []
                    t_outer = time.time()
                    for inner_iter in range(1, inner_repetitions + 1):
                        if inner_iter == 1 or inner_iter % progress_every == 0 or inner_iter == inner_repetitions:
                            log(
                                f"progress {experiment} {setting} label={label} n={n} "
                                f"outer={outer_iter}/{outer_repetitions} inner={inner_iter}/{inner_repetitions}"
                            )
                        base_seed = seed_base + 10_000_000 * label + 1_000_000 * n + 10_000 * outer_iter + inner_iter
                        setting_offset = {"H1_source_vs_external": 11, "H0_source": 23, "H0_external": 37}[setting]
                        rng = np.random.default_rng(base_seed + setting_offset)
                        x_pos, y_pos, x_pool, y_pool = BASE.sample_center_indices(label, setting, n, pools, rng)
                        emb_x, emb_y = features_for_sample(x_pos, y_pos, x_pool, y_pool, reps_source, reps_external)
                        for rep in representations:
                            seed = base_seed + stable_seed_offset(rep)
                            t0 = time.time()
                            out = VALID.run_gaussian5_bootstrap(emb_x[rep], emb_y[rep], b_boot, alpha, seed)
                            rows.append(
                                {
                                    "result_kind": result_kind,
                                    "experiment": experiment,
                                    "scenario": scenario,
                                    "setting": setting,
                                    "label": str(label),
                                    "label_name": CLASS_NAMES[label],
                                    "n": n,
                                    "representation": rep,
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
                                    "runtime_sec": time.time() - t0,
                                    "B_boot": b_boot,
                                    "alpha": alpha,
                                }
                            )
                    append_csv(raw_path, MMMD_RAW_FIELDS, rows)
                    log(f"finished {experiment} {setting} label={label} n={n} outer={outer_iter}/{outer_repetitions} in {time.time() - t_outer:.2f}s")


def summarize_mmmd(raw_path: Path, summary_path: Path, b_boot: int, alpha: float) -> list[dict[str, object]]:
    rows = read_rows(raw_path)
    groups = sorted(
        {
            (
                row["result_kind"],
                row["experiment"],
                row["scenario"],
                row["setting"],
                row["label"],
                row["label_name"],
                int(row["n"]),
                row["representation"],
            )
            for row in rows
        }
    )
    out: list[dict[str, object]] = []
    for result_kind, experiment, scenario, setting, label, label_name, n, rep in groups:
        subset = [
            row
            for row in rows
            if row["result_kind"] == result_kind
            and row["experiment"] == experiment
            and row["scenario"] == scenario
            and row["setting"] == setting
            and row["label"] == label
            and int(row["n"]) == n
            and row["representation"] == rep
        ]
        rejects = np.array([int(row["reject"]) for row in subset], dtype=float)
        rate = float(np.mean(rejects)) if len(subset) else float("nan")
        se, lo, hi = binomial_interval(rate, len(subset))
        out.append(
            {
                "result_kind": result_kind,
                "experiment": experiment,
                "scenario": scenario,
                "setting": setting,
                "label": label,
                "label_name": label_name,
                "n": n,
                "representation": rep,
                "outer_repetitions": len({int(row["outer_iter"]) for row in subset}),
                "inner_repetitions": max(int(row["inner_iter"]) for row in subset) if subset else 0,
                "independent_tests": len(subset),
                "rejection_rate": rate,
                "power": rate if result_kind == "power" else "",
                "type1_error": rate if result_kind == "type1" else "",
                "binomial_se": se,
                "ci_lower": lo,
                "ci_upper": hi,
                "mean_stat": float(np.mean([float(row["stat"]) for row in subset])) if subset else float("nan"),
                "mean_cutoff": float(np.mean([float(row["cutoff"]) for row in subset])) if subset else float("nan"),
                "mean_cond_sigma_reg": float(np.mean([float(row["cond_sigma_reg"]) for row in subset])) if subset else float("nan"),
                "B_boot": b_boot,
                "alpha": alpha,
            }
        )
    write_csv(summary_path, MMMD_SUMMARY_FIELDS, out)
    return out


def lookup(rows: list[dict[str, object]] | list[dict[str, str]], field: str, **criteria: object) -> float | None:
    for row in rows:
        if all(str(row.get(k, "")) == str(v) for k, v in criteria.items()):
            return float(row[field])
    return None


def max_lookup(rows: list[dict[str, object]] | list[dict[str, str]], field: str, **criteria: object) -> float | None:
    vals = [float(row[field]) for row in rows if all(str(row.get(k, "")) == str(v) for k, v in criteria.items()) and row.get(field, "") != ""]
    return max(vals) if vals else None


def plot_cross_label_transfer(summary: list[dict[str, object]]) -> None:
    cases = [(8, 8), (8, 6), (6, 6), (6, 8)]
    case_labels = [f"{a}->{b}" for a, b in cases]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, rep in zip(axes, ["residual_top8_pca", "residual_full"]):
        for method, marker in [("logistic_regression", "o"), ("mean_difference", "s")]:
            vals = []
            for train_label, test_label in cases:
                val = lookup(summary, "mean_auc", train_label=train_label, test_label=test_label, representation=rep, axis_method=method)
                vals.append(np.nan if val is None else val)
            ax.plot(case_labels, vals, marker=marker, linewidth=2, label=method)
        ax.axhline(0.5, color="black", linestyle=":", linewidth=1)
        ax.set_title(rep)
        ax.set_xlabel("Train label -> test label")
        ax.set_ylabel("AUC")
        ax.set_ylim(0.45, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(CROSS_LABEL_PLOT, dpi=180)
    plt.close(fig)


def plot_color_adjusted(color_only: list[dict[str, object]], color_adj: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    reps = [
        ("color15", "color-only", "#555555", color_only),
        ("semantic_centered_color_adjusted", "semantic color-adjusted", "#2ca02c", color_adj),
        ("residual_top8_pca_color_adjusted", "residual top8 color-adjusted", "#9467bd", color_adj),
        ("residual_full_color_adjusted", "residual full color-adjusted", "#d62728", color_adj),
    ]
    for ax, label in zip(axes, [6, 8]):
        for rep, name, color, rows in reps:
            vals = []
            ns = [20, 30, 50]
            for n in ns:
                vals.append(lookup(rows, "rejection_rate", result_kind="power", setting="H1_source_vs_external", label=label, n=n, representation=rep))
            ax.plot(ns, vals, marker="o", linewidth=2, label=name, color=color)
        ax.axhline(0.05, color="black", linestyle=":", linewidth=1)
        ax.set_title(f"Label {label} source vs external")
        ax.set_xlabel("Group size n")
        ax.set_ylabel("Rejection rate")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
    axes[-1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(COLOR_ADJ_PLOT, dpi=180)
    plt.close(fig)


def plot_intrinsic_dimension(summary: list[dict[str, object]]) -> None:
    ks = [1, 2, 4, 8, 16, 32]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    panels = [
        ("H1 source vs external", "power", "H1_source_vs_external", "rejection_rate"),
        ("H0 source", "type1", "H0_source", "rejection_rate"),
        ("H0 external", "type1", "H0_external", "rejection_rate"),
    ]
    for ax, (title, kind, setting, field) in zip(axes, panels):
        for n in [20, 30, 50]:
            vals = []
            for k in ks:
                vals.append(lookup(summary, field, result_kind=kind, setting=setting, label=8, n=n, representation=f"residual_pca_k{k}"))
            ax.plot(ks, vals, marker="o", linewidth=2, label=f"n={n}")
        ax.axhline(0.05, color="black", linestyle=":", linewidth=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_title(title)
        ax.set_xlabel("Residual PCA dimension k")
        ax.set_ylim(0, 1.02 if kind == "power" else 0.25)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Rejection rate")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(INTRINSIC_PLOT, dpi=180)
    plt.close(fig)


def fmt(x: float | None) -> str:
    return "NA" if x is None else f"{x:.3f}"


def write_short_report(
    cross_summary: list[dict[str, object]] | list[dict[str, str]],
    color_auc: list[dict[str, object]] | list[dict[str, str]],
    color_only: list[dict[str, object]] | list[dict[str, str]],
    color_adj: list[dict[str, object]] | list[dict[str, str]],
    intrinsic: list[dict[str, object]] | list[dict[str, str]],
) -> None:
    label8_self = lookup(cross_summary, "mean_auc", train_label=8, test_label=8, representation="residual_full", axis_method="logistic_regression")
    label8_to_6 = lookup(cross_summary, "mean_auc", train_label=8, test_label=6, representation="residual_full", axis_method="logistic_regression")
    label6_to_8 = lookup(cross_summary, "mean_auc", train_label=6, test_label=8, representation="residual_full", axis_method="logistic_regression")
    color8 = lookup(color_auc, "mean_auc", train_label=8, test_label=8, representation="color15", axis_method="logistic_regression")
    color6 = lookup(color_auc, "mean_auc", train_label=6, test_label=6, representation="color15", axis_method="logistic_regression")
    color_only8 = lookup(color_only, "rejection_rate", result_kind="power", label=8, n=50, representation="color15")
    adj8 = lookup(color_adj, "rejection_rate", result_kind="power", label=8, n=50, representation="residual_full_color_adjusted")
    adj6 = lookup(color_adj, "rejection_rate", result_kind="power", label=6, n=50, representation="residual_full_color_adjusted")
    k1 = lookup(intrinsic, "rejection_rate", result_kind="power", setting="H1_source_vs_external", label=8, n=50, representation="residual_pca_k1")
    k8 = lookup(intrinsic, "rejection_rate", result_kind="power", setting="H1_source_vs_external", label=8, n=50, representation="residual_pca_k8")
    k32 = lookup(intrinsic, "rejection_rate", result_kind="power", setting="H1_source_vs_external", label=8, n=50, representation="residual_pca_k32")
    h0max = max_lookup(intrinsic, "rejection_rate", result_kind="type1")

    def yes_transfer() -> str:
        if label8_self is None or label8_to_6 is None:
            return "无法判断"
        if label8_to_6 >= 0.85 or label8_to_6 >= label8_self - 0.08:
            return "更像共享的 source/external domain axis"
        if label8_to_6 <= 0.70:
            return "更像 label-specific residual/domain interaction"
        return "介于两者之间，存在共享 domain axis 但 label-specific 成分也明显"

    lines = [
        "# Label-8 Residual Signal Dissection: Short Report",
        "",
        "This report uses existing PathMNIST center-shift CNN checkpoints and centered-W residual features. No CNN was retrained.",
        "",
        "## 1. Is label-8 residual shift shared with label 6, or label-specific?",
        "",
        f"- residual_full logistic AUC: train 8 -> test 8 = {fmt(label8_self)}, train 8 -> test 6 = {fmt(label8_to_6)}, train 6 -> test 8 = {fmt(label6_to_8)}.",
        f"- Interpretation: {yes_transfer()}.",
        "",
        "## 2. How much is explained by color/stain features?",
        "",
        f"- color-only logistic AUC: label 6 = {fmt(color6)}, label 8 = {fmt(color8)}.",
        f"- color-only MMMD power at n=50 for label 8 = {fmt(color_only8)}.",
        f"- after color adjustment, residual_full MMMD power at n=50: label 6 = {fmt(adj6)}, label 8 = {fmt(adj8)}.",
        "- If color-only is high and adjusted residual drops, the residual shift is substantially stain/site/channel driven. If adjusted residual remains high, non-color texture/morphology/acquisition structure remains.",
        "",
        "## 3. Is residual shift low-dimensional or distributed?",
        "",
        f"- label 8 residual PCA power at n=50: k=1 = {fmt(k1)}, k=8 = {fmt(k8)}, k=32 = {fmt(k32)}.",
        f"- max intrinsic-dimension H0 rejection = {fmt(h0max)}.",
        "- If k=1/2 is already high, the shift is close to a low-dimensional domain axis. If power rises gradually through k=16/32, it is distributed across residual texture/morphology directions.",
        "",
        "## 4. Safest interpretation for final presentation",
        "",
        "- The label-8 center-shift residual signal is robust, but it should be presented as a residual domain-shift signal unless color-adjusted residual power and transfer results clearly support a stronger morphology-domain interaction claim.",
        "- Avoid claiming a purely biological morphology difference unless color/stain adjustment still leaves high residual power and cross-label transfer is weak.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-repetitions", type=int, default=10)
    parser.add_argument("--inner-repetitions", type=int, default=20)
    parser.add_argument("--b-bootstrap", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-grid", type=int, nargs="+", default=[20, 30, 50])
    parser.add_argument("--domain-repeats", type=int, default=20)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--logistic-l2", type=float, default=1e-3)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "outer_repetitions": args.outer_repetitions,
        "inner_repetitions": args.inner_repetitions,
        "B_boot": args.b_bootstrap,
        "alpha": args.alpha,
        "n_grid": args.n_grid,
        "domain_repeats": args.domain_repeats,
        "train_fraction": args.train_fraction,
        "logistic_l2": args.logistic_l2,
        "center_checkpoint": str(BASE.CENTER_CHECKPOINT),
        "center_split_path": str(BASE.CENTER_SPLIT_PATH),
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2))
    log(f"config: {json.dumps(config, sort_keys=True)}")

    if args.summarize_only:
        cross_summary = read_rows(CROSS_LABEL_SUMMARY)
        color_auc = read_rows(COLOR_AUC_SUMMARY)
        color_only = summarize_mmmd(COLOR_ONLY_RAW, COLOR_ONLY_SUMMARY, args.b_bootstrap, args.alpha) if COLOR_ONLY_RAW.exists() else []
        color_adj = summarize_mmmd(COLOR_ADJ_RAW, COLOR_ADJ_SUMMARY, args.b_bootstrap, args.alpha) if COLOR_ADJ_RAW.exists() else []
        intrinsic = summarize_mmmd(INTRINSIC_RAW, INTRINSIC_SUMMARY, args.b_bootstrap, args.alpha) if INTRINSIC_RAW.exists() else []
        if cross_summary:
            plot_cross_label_transfer(cross_summary)
        if color_only and color_adj:
            plot_color_adjusted(color_only, color_adj)
        if intrinsic:
            plot_intrinsic_dimension(intrinsic)
        write_short_report(cross_summary, color_auc, color_only, color_adj, intrinsic)
        log("summarize-only complete")
        return

    if args.force:
        for path in [
            CROSS_LABEL_RAW,
            CROSS_LABEL_SUMMARY,
            CROSS_LABEL_PLOT,
            COLOR_AUC_RAW,
            COLOR_AUC_SUMMARY,
            COLOR_ONLY_RAW,
            COLOR_ONLY_SUMMARY,
            COLOR_ADJ_RAW,
            COLOR_ADJ_SUMMARY,
            COLOR_ADJ_PLOT,
            INTRINSIC_RAW,
            INTRINSIC_SUMMARY,
            INTRINSIC_PLOT,
            REPORT_PATH,
        ]:
            if path.exists():
                path.unlink()

    features = load_or_build_features(args.eval_batch_size, args.force_features)
    raw = BASE.load_pathmnist_raw()
    pools = BASE.load_center_pools(raw)

    cross_summary = run_cross_label_domain_axis(features, args.domain_repeats, args.train_fraction, args.logistic_l2, args.force)
    color_auc = run_color_domain_probe(features, args.domain_repeats, args.train_fraction, args.logistic_l2, args.force)

    color_source_reps, color_external_reps = build_color_adjusted_reps(features)
    run_mmmd_grid(
        COLOR_ONLY_RAW,
        "color_only_center_shift",
        {"color15": color_source_reps["color15"]},
        {"color15": color_external_reps["color15"]},
        pools,
        [6, 8],
        args.n_grid,
        ["H1_source_vs_external", "H0_source", "H0_external"],
        ["color15"],
        args.outer_repetitions,
        args.inner_repetitions,
        args.b_bootstrap,
        args.alpha,
        args.progress_every,
        20260611,
    )
    color_only_summary = summarize_mmmd(COLOR_ONLY_RAW, COLOR_ONLY_SUMMARY, args.b_bootstrap, args.alpha)

    adjusted_reps = [
        "semantic_centered_color_adjusted",
        "residual_top8_pca_color_adjusted",
        "residual_full_color_adjusted",
    ]
    run_mmmd_grid(
        COLOR_ADJ_RAW,
        "color_adjusted_center_shift",
        color_source_reps,
        color_external_reps,
        pools,
        [6, 8],
        args.n_grid,
        ["H1_source_vs_external", "H0_source", "H0_external"],
        adjusted_reps,
        args.outer_repetitions,
        args.inner_repetitions,
        args.b_bootstrap,
        args.alpha,
        args.progress_every,
        20260612,
    )
    color_adj_summary = summarize_mmmd(COLOR_ADJ_RAW, COLOR_ADJ_SUMMARY, args.b_bootstrap, args.alpha)
    plot_color_adjusted(color_only_summary, color_adj_summary)

    intrinsic_source_reps, intrinsic_external_reps = build_intrinsic_dimension_reps(features)
    intrinsic_reps = [f"residual_pca_k{k}" for k in [1, 2, 4, 8, 16, 32]]
    run_mmmd_grid(
        INTRINSIC_RAW,
        "residual_intrinsic_dimension",
        intrinsic_source_reps,
        intrinsic_external_reps,
        pools,
        [8],
        args.n_grid,
        ["H1_source_vs_external", "H0_source", "H0_external"],
        intrinsic_reps,
        args.outer_repetitions,
        args.inner_repetitions,
        args.b_bootstrap,
        args.alpha,
        args.progress_every,
        20260613,
    )
    intrinsic_summary = summarize_mmmd(INTRINSIC_RAW, INTRINSIC_SUMMARY, args.b_bootstrap, args.alpha)
    plot_intrinsic_dimension(intrinsic_summary)

    write_short_report(cross_summary, color_auc, color_only_summary, color_adj_summary, intrinsic_summary)
    log(f"wrote report: {REPORT_PATH}")
    log("label-8 residual dissection complete")


if __name__ == "__main__":
    main()
