#!/usr/bin/env python3
"""Explain the PathMNIST label-8 residual center-shift signal by color.

This script continues the label-8 residual dissection without retraining CNNs
or adding datasets. It reuses the centered feature cache created by
dissect_label8_residual_signal.py.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = EXPERIMENT_ROOT / "python" / "dissect_label8_residual_signal.py"
RESULTS_DIR = EXPERIMENT_ROOT / "Results"
LOG_DIR = EXPERIMENT_ROOT / "logs"
LOG_PATH = LOG_DIR / "label8_residual_color_explain_run.log"

DOMAIN_R2_CSV = RESULTS_DIR / "domain_axis_color_r2.csv"
DOMAIN_R2_PLOT = RESULTS_DIR / "domain_axis_color_r2_plot.png"
MATCHED_RAW = RESULTS_DIR / "color_matched_residual_mmmd_results.csv"
MATCHED_SUMMARY = RESULTS_DIR / "color_matched_residual_mmmd_summary.csv"
MATCHED_PLOT = RESULTS_DIR / "color_matched_residual_mmmd_plot.png"
MATCHED_QUALITY = RESULTS_DIR / "color_matched_residual_match_quality.csv"
REPORT_PATH = RESULTS_DIR / "label8_residual_color_explain_short_report.md"
CONFIG_PATH = RESULTS_DIR / "label8_residual_color_explain_config.json"

COLOR_FEATURE_NAMES = [
    "mean_R",
    "mean_G",
    "mean_B",
    "std_R",
    "std_G",
    "std_B",
    "q10_R",
    "q10_G",
    "q10_B",
    "q50_R",
    "q50_G",
    "q50_B",
    "q90_R",
    "q90_G",
    "q90_B",
]

DOMAIN_R2_FIELDS = [
    "label",
    "label_name",
    "representation",
    "axis_method",
    "repeats",
    "mean_r2",
    "sd_r2",
    "mean_auc_before_color_residualization",
    "sd_auc_before_color_residualization",
    "mean_auc_after_color_residualization",
    "sd_auc_after_color_residualization",
    "mean_accuracy_before_color_residualization",
    "mean_accuracy_after_color_residualization",
    "top_correlated_color_features",
    "mean_abs_corr_by_feature",
]

MATCHED_QUALITY_FIELDS = [
    "label",
    "label_name",
    "matching_method",
    "propensity_logit_caliper",
    "source_label8_n",
    "external_label8_n",
    "source_overlap_n",
    "external_overlap_n",
    "matched_pair_n",
    "mean_abs_propensity_logit_gap",
    "median_abs_propensity_logit_gap",
    "p90_abs_propensity_logit_gap",
    "matched_color_only_auc",
]


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BASE_EXP = import_module(SCRIPT_PATH, "label8_residual_dissect_base")


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


def stable_seed_offset(text: str) -> int:
    return int(sum((idx + 1) * ord(ch) for idx, ch in enumerate(text)) % 1_000_000)


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


def fit_linear_residualizer(x_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1, 1)
    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True)
    x_std = np.where(x_std < 1e-6, 1.0, x_std)
    design = np.concatenate([np.ones((x_train.shape[0], 1)), (x_train - x_mean) / x_std], axis=1)
    beta, *_ = np.linalg.lstsq(design, y_train, rcond=None)
    return beta, x_mean, x_std


def apply_linear_model(x: np.ndarray, beta: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray) -> np.ndarray:
    design = np.concatenate([np.ones((x.shape[0], 1)), (x - x_mean) / x_std], axis=1)
    return (design @ beta).ravel()


def split_domain_arrays(
    source_x: np.ndarray,
    external_x: np.ndarray,
    source_color: np.ndarray,
    external_color: np.ndarray,
    train_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_bal = min(len(source_x), len(external_x))
    n_train = max(2, int(round(train_fraction * n_bal)))
    n_train = min(n_train, n_bal - 2)
    src_perm = rng.permutation(len(source_x))[:n_bal]
    ext_perm = rng.permutation(len(external_x))[:n_bal]
    return (
        source_x[src_perm[:n_train]],
        external_x[ext_perm[:n_train]],
        source_x[src_perm[n_train:]],
        external_x[ext_perm[n_train:]],
        source_color[src_perm[:n_train]],
        external_color[ext_perm[:n_train]],
        source_color[src_perm[n_train:]],
        external_color[ext_perm[n_train:]],
    )


def fit_domain_axis_scores(
    train_source: np.ndarray,
    train_external: np.ndarray,
    test_source: np.ndarray,
    test_external: np.ndarray,
    l2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train = np.vstack([train_source, train_external])
    y_train = np.concatenate([np.zeros(len(train_source)), np.ones(len(train_external))]).astype(int)
    x_test = np.vstack([test_source, test_external])
    y_test = np.concatenate([np.zeros(len(test_source)), np.ones(len(test_external))]).astype(int)
    x_train_z, x_test_z, _, _ = standardize_train_test(x_train, x_test)
    w, b = BASE_EXP.fit_logistic_axis(x_train_z, y_train, l2=l2)
    return x_train_z @ w + b, x_test_z @ w + b, y_train, y_test


def top_color_correlations(scores: np.ndarray, colors: np.ndarray) -> np.ndarray:
    out = []
    for j in range(colors.shape[1]):
        if np.std(colors[:, j]) < 1e-12 or np.std(scores) < 1e-12:
            out.append(0.0)
        else:
            out.append(float(np.corrcoef(scores, colors[:, j])[0, 1]))
    return np.array(out, dtype=float)


def run_domain_axis_color_r2(
    features: dict[str, np.ndarray],
    repeats: int,
    train_fraction: float,
    logistic_l2: float,
    force: bool,
) -> list[dict[str, object]]:
    if DOMAIN_R2_CSV.exists() and not force:
        log("skip domain-axis color R2; output already exists")
        return read_rows(DOMAIN_R2_CSV)

    label = 8
    label_name = BASE_EXP.CLASS_NAMES[label]
    src_mask = features["source_labels"] == label
    ext_mask = features["external_labels"] == label
    rows = []
    for rep in ["residual_top8_pca", "residual_full"]:
        repeat_metrics = []
        corr_by_repeat = []
        src_x = features[f"source_{rep}"][src_mask]
        ext_x = features[f"external_{rep}"][ext_mask]
        src_color = features["source_color15"][src_mask]
        ext_color = features["external_color15"][ext_mask]
        for repeat in range(1, repeats + 1):
            rng = np.random.default_rng(20260615 + 1000 * repeat + stable_seed_offset(rep))
            src_train, ext_train, src_test, ext_test, src_color_train, ext_color_train, src_color_test, ext_color_test = split_domain_arrays(
                src_x,
                ext_x,
                src_color,
                ext_color,
                train_fraction,
                rng,
            )
            scores_train, scores_test, y_train, y_test = fit_domain_axis_scores(
                src_train,
                ext_train,
                src_test,
                ext_test,
                logistic_l2,
            )
            color_train = np.vstack([src_color_train, ext_color_train])
            color_test = np.vstack([src_color_test, ext_color_test])
            beta, c_mean, c_std = fit_linear_residualizer(color_train, scores_train)
            pred_test = apply_linear_model(color_test, beta, c_mean, c_std)
            residual_scores_test = scores_test - pred_test
            ss_res = float(np.sum((scores_test - pred_test) ** 2))
            ss_tot = float(np.sum((scores_test - scores_test.mean()) ** 2))
            r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
            threshold_before = 0.0
            threshold_after = float(np.median(scores_train - apply_linear_model(color_train, beta, c_mean, c_std)))
            repeat_metrics.append(
                {
                    "r2": r2,
                    "auc_before": auc_from_scores(y_test, scores_test),
                    "auc_after": auc_from_scores(y_test, residual_scores_test),
                    "accuracy_before": float(np.mean((scores_test >= threshold_before).astype(int) == y_test)),
                    "accuracy_after": float(np.mean((residual_scores_test >= threshold_after).astype(int) == y_test)),
                }
            )
            corr_by_repeat.append(np.abs(top_color_correlations(scores_test, color_test)))
        corr_mean = np.mean(np.vstack(corr_by_repeat), axis=0)
        top_idx = np.argsort(corr_mean)[::-1][:5]
        rows.append(
            {
                "label": label,
                "label_name": label_name,
                "representation": rep,
                "axis_method": "logistic_regression",
                "repeats": repeats,
                "mean_r2": float(np.mean([m["r2"] for m in repeat_metrics])),
                "sd_r2": float(np.std([m["r2"] for m in repeat_metrics], ddof=1)),
                "mean_auc_before_color_residualization": float(np.mean([m["auc_before"] for m in repeat_metrics])),
                "sd_auc_before_color_residualization": float(np.std([m["auc_before"] for m in repeat_metrics], ddof=1)),
                "mean_auc_after_color_residualization": float(np.mean([m["auc_after"] for m in repeat_metrics])),
                "sd_auc_after_color_residualization": float(np.std([m["auc_after"] for m in repeat_metrics], ddof=1)),
                "mean_accuracy_before_color_residualization": float(np.mean([m["accuracy_before"] for m in repeat_metrics])),
                "mean_accuracy_after_color_residualization": float(np.mean([m["accuracy_after"] for m in repeat_metrics])),
                "top_correlated_color_features": ";".join(f"{COLOR_FEATURE_NAMES[i]}:{corr_mean[i]:.4f}" for i in top_idx),
                "mean_abs_corr_by_feature": ";".join(f"{name}:{corr_mean[i]:.4f}" for i, name in enumerate(COLOR_FEATURE_NAMES)),
            }
        )
        log(f"finished domain-axis color R2 rep={rep}")
    write_csv(DOMAIN_R2_CSV, DOMAIN_R2_FIELDS, rows)
    plot_domain_axis_color_r2(rows)
    log(f"wrote {DOMAIN_R2_CSV}")
    return rows


def plot_domain_axis_color_r2(rows: list[dict[str, object]]) -> None:
    reps = [str(row["representation"]) for row in rows]
    r2 = [float(row["mean_r2"]) for row in rows]
    auc_before = [float(row["mean_auc_before_color_residualization"]) for row in rows]
    auc_after = [float(row["mean_auc_after_color_residualization"]) for row in rows]
    x = np.arange(len(reps))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(x, r2, color=["#9467bd", "#d62728"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(reps, rotation=20, ha="right")
    axes[0].set_ylabel("Holdout R2")
    axes[0].set_title("Domain-axis score explained by color15")
    axes[0].set_ylim(0, max(0.05, min(1.0, max(r2) * 1.25)))
    axes[0].grid(axis="y", alpha=0.25)
    width = 0.35
    axes[1].bar(x - width / 2, auc_before, width=width, label="before", color="#555555")
    axes[1].bar(x + width / 2, auc_after, width=width, label="after score residualization", color="#2ca02c")
    axes[1].axhline(0.5, color="black", linestyle=":", linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(reps, rotation=20, ha="right")
    axes[1].set_ylabel("Domain AUC")
    axes[1].set_ylim(0.45, 1.0)
    axes[1].set_title("Domain AUC before/after color residualization")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(DOMAIN_R2_PLOT, dpi=180)
    plt.close(fig)


def fit_color_propensity(features: dict[str, np.ndarray], label: int, l2: float) -> tuple[np.ndarray, np.ndarray]:
    src_mask = features["source_labels"] == label
    ext_mask = features["external_labels"] == label
    source_color = features["source_color15"][src_mask]
    external_color = features["external_color15"][ext_mask]
    x = np.vstack([source_color, external_color])
    y = np.concatenate([np.zeros(len(source_color)), np.ones(len(external_color))]).astype(int)
    x_z, _, mean, std = standardize_train_test(x, x)
    w, b = BASE_EXP.fit_logistic_axis(x_z, y, l2=l2)
    source_scores = ((source_color - mean) / std) @ w + b
    external_scores = ((external_color - mean) / std) @ w + b
    return source_scores, external_scores


def build_color_matched_pairs(
    features: dict[str, np.ndarray],
    label: int,
    l2: float,
    caliper: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    src_mask = features["source_labels"] == label
    ext_mask = features["external_labels"] == label
    source_positions = np.where(src_mask)[0]
    external_positions = np.where(ext_mask)[0]
    source_scores, external_scores = fit_color_propensity(features, label, l2)
    overlap_lo = max(float(np.min(source_scores)), float(np.min(external_scores)))
    overlap_hi = min(float(np.max(source_scores)), float(np.max(external_scores)))
    source_overlap = np.where((source_scores >= overlap_lo) & (source_scores <= overlap_hi))[0]
    external_overlap = np.where((external_scores >= overlap_lo) & (external_scores <= overlap_hi))[0]

    source_order = source_overlap[np.argsort(source_scores[source_overlap])]
    source_vals = [float(v) for v in source_scores[source_order]]
    source_idx = [int(v) for v in source_order]
    matched_source = []
    matched_external = []
    matched_gaps = []
    for external_idx in external_overlap[np.argsort(external_scores[external_overlap])]:
        target = float(external_scores[external_idx])
        insert_pos = bisect.bisect_left(source_vals, target)
        candidates = []
        if insert_pos < len(source_vals):
            candidates.append(insert_pos)
        if insert_pos > 0:
            candidates.append(insert_pos - 1)
        if not candidates:
            continue
        best_pos = min(candidates, key=lambda idx: abs(source_vals[idx] - target))
        gap = abs(source_vals[best_pos] - target)
        if gap > caliper:
            continue
        matched_source.append(source_idx.pop(best_pos))
        matched_external.append(int(external_idx))
        matched_gaps.append(gap)
        source_vals.pop(best_pos)

    if not matched_source:
        raise RuntimeError("color matching produced zero pairs; relax the propensity caliper")

    matched_source = np.array(matched_source, dtype=int)
    matched_external = np.array(matched_external, dtype=int)
    matched_gaps_arr = np.array(matched_gaps, dtype=float)
    csrc = features["source_color15"][source_positions[matched_source]]
    cext = features["external_color15"][external_positions[matched_external]]
    y = np.concatenate([np.zeros(len(csrc)), np.ones(len(cext))]).astype(int)
    x = np.vstack([csrc, cext])
    x_z, _, _, _ = standardize_train_test(x, x)
    w, b = BASE_EXP.fit_logistic_axis(x_z, y, l2=l2)
    color_auc = auc_from_scores(y, x_z @ w + b)
    quality = {
        "label": label,
        "label_name": BASE_EXP.CLASS_NAMES[label],
        "matching_method": "common_support_nearest_neighbor_color_propensity",
        "propensity_logit_caliper": caliper,
        "source_label8_n": int(len(source_scores)),
        "external_label8_n": int(len(external_scores)),
        "source_overlap_n": int(len(source_overlap)),
        "external_overlap_n": int(len(external_overlap)),
        "matched_pair_n": int(len(matched_source)),
        "mean_abs_propensity_logit_gap": float(np.mean(matched_gaps_arr)),
        "median_abs_propensity_logit_gap": float(np.median(matched_gaps_arr)),
        "p90_abs_propensity_logit_gap": float(np.quantile(matched_gaps_arr, 0.90)),
        "matched_color_only_auc": float(color_auc),
    }
    return source_positions[matched_source], external_positions[matched_external], quality


def completed_matched_outer_keys(raw_path: Path, representations: list[str], inner_repetitions: int) -> set[tuple[str, str, str, str, int, int]]:
    expected = {(inner, rep) for inner in range(1, inner_repetitions + 1) for rep in representations}
    seen_by_key: dict[tuple[str, str, str, str, int, int], set[tuple[int, str]]] = {}
    for row in read_rows(raw_path):
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


def sample_color_matched_h1(
    pair_source_pos: np.ndarray,
    pair_external_pos: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    pair_idx = rng.choice(np.arange(len(pair_source_pos)), size=n, replace=False)
    return pair_source_pos[pair_idx], pair_external_pos[pair_idx], "source_holdout_pool", "external_pool"


def run_color_matched_mmmd(
    features: dict[str, np.ndarray],
    n_grid: list[int],
    outer_repetitions: int,
    inner_repetitions: int,
    b_boot: int,
    alpha: float,
    progress_every: int,
    logistic_l2: float,
    propensity_caliper: float,
    force: bool,
) -> list[dict[str, object]]:
    representations = ["residual_top8_pca", "residual_full"]
    if MATCHED_SUMMARY.exists() and not force:
        log("skip color-matched residual MMMD; summary already exists")
        return read_rows(MATCHED_SUMMARY)
    if force:
        for path in [MATCHED_RAW, MATCHED_SUMMARY, MATCHED_PLOT, MATCHED_QUALITY]:
            if path.exists():
                path.unlink()

    raw = BASE_EXP.BASE.load_pathmnist_raw()
    pools = BASE_EXP.BASE.load_center_pools(raw)
    reps_source = {rep: features[f"source_{rep}"] for rep in representations}
    reps_external = {rep: features[f"external_{rep}"] for rep in representations}
    pair_source_pos, pair_external_pos, quality = build_color_matched_pairs(features, 8, logistic_l2, propensity_caliper)
    write_csv(MATCHED_QUALITY, MATCHED_QUALITY_FIELDS, [quality])
    log(
        "built color-propensity matched pairs label=8 "
        f"pair_n={len(pair_source_pos)} mean_abs_logit_gap={float(quality['mean_abs_propensity_logit_gap']):.4f} "
        f"matched_color_auc={float(quality['matched_color_only_auc']):.4f}"
    )

    done = completed_matched_outer_keys(MATCHED_RAW, representations, inner_repetitions)
    for setting in ["H1_source_vs_external_color_matched", "H0_source", "H0_external"]:
        result_kind = "power" if setting == "H1_source_vs_external_color_matched" else "type1"
        for n in n_grid:
            for outer_iter in range(1, outer_repetitions + 1):
                scenario = "label8_source_vs_external_color_matched"
                key = (result_kind, scenario, setting, "8", n, outer_iter)
                if key in done:
                    log(f"skip color-matched MMMD {setting} n={n} outer={outer_iter}")
                    continue
                rows = []
                t_outer = time.time()
                for inner_iter in range(1, inner_repetitions + 1):
                    if inner_iter == 1 or inner_iter % progress_every == 0 or inner_iter == inner_repetitions:
                        log(
                            f"progress color-matched residual MMMD {setting} label=8 n={n} "
                            f"outer={outer_iter}/{outer_repetitions} inner={inner_iter}/{inner_repetitions}"
                        )
                    base_seed = 20260616 + 1_000_000 * n + 10_000 * outer_iter + inner_iter
                    rng = np.random.default_rng(base_seed + stable_seed_offset(setting))
                    if setting == "H1_source_vs_external_color_matched":
                        x_pos, y_pos, x_pool, y_pool = sample_color_matched_h1(pair_source_pos, pair_external_pos, n, rng)
                    else:
                        x_pos, y_pos, x_pool, y_pool = BASE_EXP.BASE.sample_center_indices(8, setting, n, pools, rng)
                    emb_x, emb_y = BASE_EXP.features_for_sample(x_pos, y_pos, x_pool, y_pool, reps_source, reps_external)
                    for rep in representations:
                        seed = base_seed + stable_seed_offset(rep)
                        t0 = time.time()
                        out = BASE_EXP.VALID.run_gaussian5_bootstrap(emb_x[rep], emb_y[rep], b_boot, alpha, seed)
                        rows.append(
                            {
                                "result_kind": result_kind,
                                "experiment": "color_matched_residual_center_shift",
                                "scenario": scenario,
                                "setting": setting,
                                "label": "8",
                                "label_name": BASE_EXP.CLASS_NAMES[8],
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
                append_csv(MATCHED_RAW, BASE_EXP.MMMD_RAW_FIELDS, rows)
                log(f"finished color-matched residual MMMD {setting} n={n} outer={outer_iter}/{outer_repetitions} in {time.time() - t_outer:.2f}s")
    summary = BASE_EXP.summarize_mmmd(MATCHED_RAW, MATCHED_SUMMARY, b_boot, alpha)
    plot_color_matched_mmmd(summary)
    log(f"wrote {MATCHED_SUMMARY}")
    return summary


def lookup(rows: list[dict[str, object]] | list[dict[str, str]], field: str, **criteria: object) -> float | None:
    for row in rows:
        if all(str(row.get(k, "")) == str(v) for k, v in criteria.items()):
            return float(row[field])
    return None


def max_lookup(rows: list[dict[str, object]] | list[dict[str, str]], field: str, **criteria: object) -> float | None:
    vals = [float(row[field]) for row in rows if row.get(field, "") != "" and all(str(row.get(k, "")) == str(v) for k, v in criteria.items())]
    return max(vals) if vals else None


def plot_color_matched_mmmd(summary: list[dict[str, object]] | list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    reps = [("residual_top8_pca", "#9467bd"), ("residual_full", "#d62728")]
    panels = [
        ("H1 color-matched source vs external", "power", "H1_source_vs_external_color_matched", 1.0),
        ("H0 checks", "type1", None, 0.25),
    ]
    for ax, (title, kind, setting, ymax) in zip(axes, panels):
        for rep, color in reps:
            if setting is not None:
                vals = [lookup(summary, "rejection_rate", result_kind=kind, setting=setting, n=n, representation=rep) for n in [20, 30, 50]]
                ax.plot([20, 30, 50], vals, marker="o", linewidth=2, label=rep, color=color)
            else:
                vals_source = [lookup(summary, "rejection_rate", result_kind=kind, setting="H0_source", n=n, representation=rep) for n in [20, 30, 50]]
                vals_external = [lookup(summary, "rejection_rate", result_kind=kind, setting="H0_external", n=n, representation=rep) for n in [20, 30, 50]]
                ax.plot([20, 30, 50], vals_source, marker="o", linewidth=2, label=f"{rep} H0-source", color=color)
                ax.plot([20, 30, 50], vals_external, marker="s", linestyle="--", linewidth=2, label=f"{rep} H0-external", color=color)
        ax.axhline(0.05, color="black", linestyle=":", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Group size n")
        ax.set_ylim(0, ymax)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Rejection rate")
    axes[1].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(MATCHED_PLOT, dpi=180)
    plt.close(fig)


def fmt(x: float | None) -> str:
    return "NA" if x is None or not np.isfinite(x) else f"{x:.3f}"


def write_report(domain_rows: list[dict[str, object]] | list[dict[str, str]], matched_summary: list[dict[str, object]] | list[dict[str, str]]) -> None:
    r2_top8 = lookup(domain_rows, "mean_r2", representation="residual_top8_pca")
    r2_full = lookup(domain_rows, "mean_r2", representation="residual_full")
    auc_top8_before = lookup(domain_rows, "mean_auc_before_color_residualization", representation="residual_top8_pca")
    auc_top8_after = lookup(domain_rows, "mean_auc_after_color_residualization", representation="residual_top8_pca")
    auc_full_before = lookup(domain_rows, "mean_auc_before_color_residualization", representation="residual_full")
    auc_full_after = lookup(domain_rows, "mean_auc_after_color_residualization", representation="residual_full")
    matched_top8_n50 = lookup(
        matched_summary,
        "rejection_rate",
        result_kind="power",
        setting="H1_source_vs_external_color_matched",
        n=50,
        representation="residual_top8_pca",
    )
    matched_full_n50 = lookup(
        matched_summary,
        "rejection_rate",
        result_kind="power",
        setting="H1_source_vs_external_color_matched",
        n=50,
        representation="residual_full",
    )
    h0_max = max_lookup(matched_summary, "rejection_rate", result_kind="type1")
    quality_rows = read_rows(MATCHED_QUALITY)
    quality = quality_rows[0] if quality_rows else {}
    top_features = {
        str(row["representation"]): str(row["top_correlated_color_features"])
        for row in domain_rows
        if "top_correlated_color_features" in row
    }
    lines = [
        "# Label-8 Residual Signal by Color: Short Report",
        "",
        "This run reused the existing PathMNIST center-shift CNN checkpoint and cached centered residual features. No model was retrained and no new dataset was added.",
        "",
        "## Domain-axis color R2",
        "",
        f"- residual_top8_pca: color15 R2 = {fmt(r2_top8)}, domain AUC before/after score color residualization = {fmt(auc_top8_before)} / {fmt(auc_top8_after)}.",
        f"- residual_full logistic axis: color15 R2 = {fmt(r2_full)}, domain AUC before/after score color residualization = {fmt(auc_full_before)} / {fmt(auc_full_after)}.",
        f"- top correlated color features for residual_top8_pca: {top_features.get('residual_top8_pca', 'NA')}.",
        f"- top correlated color features for residual_full: {top_features.get('residual_full', 'NA')}.",
        "",
        "## Color-matched residual MMMD",
        "",
        f"- matching quality: matched pairs = {quality.get('matched_pair_n', 'NA')}, mean propensity logit gap = {fmt(float(quality['mean_abs_propensity_logit_gap'])) if quality else 'NA'}, matched color-only AUC = {fmt(float(quality['matched_color_only_auc'])) if quality else 'NA'}.",
        f"- label 8 color-matched H1 power at n=50: residual_top8_pca = {fmt(matched_top8_n50)}, residual_full = {fmt(matched_full_n50)}.",
        f"- max H0 rejection rate across color-matched residual checks = {fmt(h0_max)}.",
        "",
        "## Safe conclusion",
        "",
        f"- Color/stain explains a substantial part of the label-8 residual domain signal: the residual_top8_pca domain score has color R2 {fmt(r2_top8)} and its AUC drops from {fmt(auc_top8_before)} to {fmt(auc_top8_after)} after score-level color residualization.",
        f"- Some residual structure remains after stricter color-propensity matching: at n=50, both residual_top8_pca and residual_full reject at {fmt(matched_top8_n50)} / {fmt(matched_full_n50)}, while the largest matched H0 rejection is {fmt(h0_max)}.",
        "- Because matched color-only AUC is still above 0.5, this is best described as color-reduced evidence rather than a perfectly color-randomized comparison.",
        "- If color R2 is high and the domain AUC drops strongly after score residualization, a large part of the label-8 residual signal should be described as color/stain-associated residual domain shift.",
        "- If color-matched residual MMMD remains clearly above H0, then some color-adjusted residual structure remains, but it should be presented conservatively as residual texture/morphology/acquisition structure rather than a pure biological morphology claim.",
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
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--propensity-caliper", type=float, default=0.1)
    parser.add_argument("--skip-color-matched", action="store_true")
    parser.add_argument("--force", action="store_true")
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
        "propensity_caliper": args.propensity_caliper,
        "color_matched": not args.skip_color_matched,
        "feature_cache": str(BASE_EXP.FEATURE_CACHE),
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2))
    log(f"config: {json.dumps(config, sort_keys=True)}")

    if args.summarize_only:
        domain_rows = read_rows(DOMAIN_R2_CSV)
        matched_summary = BASE_EXP.summarize_mmmd(MATCHED_RAW, MATCHED_SUMMARY, args.b_bootstrap, args.alpha) if MATCHED_RAW.exists() else []
        if domain_rows:
            plot_domain_axis_color_r2(domain_rows)
        if matched_summary:
            plot_color_matched_mmmd(matched_summary)
        write_report(domain_rows, matched_summary)
        log("summarize-only complete")
        return

    features = BASE_EXP.load_or_build_features(eval_batch_size=1024, force_features=False)
    domain_rows = run_domain_axis_color_r2(features, args.domain_repeats, args.train_fraction, args.logistic_l2, args.force)
    matched_summary: list[dict[str, object]] | list[dict[str, str]] = []
    if not args.skip_color_matched:
        matched_summary = run_color_matched_mmmd(
            features,
            args.n_grid,
            args.outer_repetitions,
            args.inner_repetitions,
            args.b_bootstrap,
            args.alpha,
            args.progress_every,
            args.logistic_l2,
            args.propensity_caliper,
            args.force,
        )
    write_report(domain_rows, matched_summary)
    log(f"complete; report={REPORT_PATH}")


if __name__ == "__main__":
    main()
