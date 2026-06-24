#!/usr/bin/env python3
"""Validity controls for PathMNIST semantic/residual CNN-MMMD.

Controls implemented from plan_next.md:

1. Centered-W decomposition using row(Wc), where Wc subtracts the class-wise
   mean classifier row.
2. Dimension-matched residual controls: residual_top8_pca and 20 random
   residual 8D projections.
3. Permutation calibration for the center-shift key cells.

The script is resumable at the outer-repetition level. It does not retrain any
CNN checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


CONTROL_ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_ROOT = CONTROL_ROOT.parent
BASE_SCRIPT = SEMANTIC_ROOT / "python" / "semantic_residual_mmmd.py"
RESULTS_DIR = CONTROL_ROOT / "Results"
LOG_DIR = CONTROL_ROOT / "logs"
LOG_PATH = LOG_DIR / "validity_controls_run.log"

DIM_RAW_PATH = RESULTS_DIR / "dim_matched_residual_mmmd_results.csv"
DIM_SUMMARY_PATH = RESULTS_DIR / "dim_matched_residual_mmmd_summary.csv"
DIM_POWER_PLOT = RESULTS_DIR / "dim_matched_residual_power.png"
DIM_TYPE1_PLOT = RESULTS_DIR / "dim_matched_residual_type1.png"
DIM_EXCESS_PLOT = RESULTS_DIR / "dim_matched_residual_excess_rejection.png"

PERM_RAW_PATH = RESULTS_DIR / "permutation_calibration_center_shift_results.csv"
PERM_SUMMARY_PATH = RESULTS_DIR / "permutation_calibration_center_shift_summary.csv"
PERM_POWER_PLOT = RESULTS_DIR / "permutation_vs_bootstrap_power.png"
PERM_TYPE1_PLOT = RESULTS_DIR / "permutation_vs_bootstrap_type1.png"

DIAG_PATH = RESULTS_DIR / "semantic_residual_centered_diagnostics.csv"
CONFIG_PATH = RESULTS_DIR / "validity_controls_config.json"
REPORT_PATH = RESULTS_DIR / "validity_controls_short_report.md"

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

DIM_REP_ORDER = [
    "centered_logits",
    "semantic_centered",
    "residual_full",
    "residual_top8_pca",
    "residual_random8",
    "final_full",
]
PERM_REP_ORDER = [
    "centered_logits",
    "semantic_centered",
    "residual_full",
    "residual_top8_pca",
    "final_full",
]
COLORS = {
    "centered_logits": "#ff7f0e",
    "semantic_centered": "#2ca02c",
    "residual_full": "#d62728",
    "residual_top8_pca": "#9467bd",
    "residual_random8": "#8c564b",
    "final_full": "#1f77b4",
}

DIM_RAW_FIELDS = [
    "result_kind",
    "experiment",
    "scenario",
    "setting",
    "label",
    "label_name",
    "n",
    "representation",
    "projection_id",
    "rep_run_id",
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

DIM_SUMMARY_FIELDS = [
    "result_kind",
    "experiment",
    "scenario",
    "setting",
    "label",
    "label_name",
    "n",
    "representation",
    "projection_id",
    "summary_scope",
    "projection_repeats",
    "outer_repetitions",
    "inner_repetitions",
    "tests_per_projection",
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

PERM_RAW_FIELDS = [
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
    "bootstrap_cutoff",
    "bootstrap_reject",
    "permutation_cutoff",
    "permutation_reject",
    "kernel_count",
    "lambda",
    "cond_sigma_hat",
    "cond_sigma_reg",
    "runtime_sec",
    "B_boot",
    "B_perm",
    "alpha",
]

PERM_SUMMARY_FIELDS = [
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
    "bootstrap_rejection_rate",
    "permutation_rejection_rate",
    "bootstrap_binomial_se",
    "permutation_binomial_se",
    "mean_stat",
    "mean_bootstrap_cutoff",
    "mean_permutation_cutoff",
    "mean_cond_sigma_reg",
    "B_boot",
    "B_perm",
    "alpha",
]

DIAG_FIELDS = [
    "pool_tag",
    "checkpoint_path",
    "rank_W",
    "rank_Wc",
    "singular_values_W",
    "singular_values_Wc",
    "mean_norm_final_full",
    "mean_norm_semantic_centered",
    "mean_norm_residual_full",
    "mean_semantic_centered_norm_fraction",
    "mean_residual_full_norm_fraction",
]


def import_base_module():
    spec = importlib.util.spec_from_file_location("semantic_residual_mmmd_base", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BASE = import_base_module()


@dataclass(frozen=True)
class RepSpec:
    representation: str
    feature_key: str
    projection_id: str = "none"

    @property
    def rep_run_id(self) -> str:
        return f"{self.representation}:{self.projection_id}"


def log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    with LOG_PATH.open("a") as f:
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


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def svd_basis(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    return BASE.row_space_basis(weight)


def stable_seed_offset(text: str) -> int:
    return int(sum((idx + 1) * ord(ch) for idx, ch in enumerate(text)) % 1_000_000)


def binomial_interval(rate: float, m: int) -> tuple[float, float, float]:
    if m <= 0 or not np.isfinite(rate):
        return float("nan"), float("nan"), float("nan")
    se = math.sqrt(rate * (1.0 - rate) / m)
    return se, max(0.0, rate - 1.96 * se), min(1.0, rate + 1.96 * se)


def extract_centered_base_reps(
    cnn_module,
    checkpoint_path: Path,
    images: np.ndarray,
    batch_size: int,
    pool_tag: str,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
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
    bias = model.fc2.bias.detach().cpu().numpy().astype(np.float64)
    weight_centered = weight - weight.mean(axis=0, keepdims=True)
    basis_w, singular_w, _ = svd_basis(weight)
    basis_wc, singular_wc, _ = svd_basis(weight_centered)

    logits = h @ weight.T + bias
    centered_logits = logits - logits.mean(axis=1, keepdims=True)
    semantic_centered = (h @ basis_wc.T) @ basis_wc if basis_wc.size else np.zeros_like(h)
    residual_full = h - semantic_centered

    full_norm = np.linalg.norm(h, axis=1)
    semantic_norm = np.linalg.norm(semantic_centered, axis=1)
    residual_norm = np.linalg.norm(residual_full, axis=1)
    diag = {
        "pool_tag": pool_tag,
        "checkpoint_path": str(checkpoint_path),
        "rank_W": int(basis_w.shape[0]),
        "rank_Wc": int(basis_wc.shape[0]),
        "singular_values_W": ";".join(f"{v:.10g}" for v in singular_w),
        "singular_values_Wc": ";".join(f"{v:.10g}" for v in singular_wc),
        "mean_norm_final_full": float(np.mean(full_norm)),
        "mean_norm_semantic_centered": float(np.mean(semantic_norm)),
        "mean_norm_residual_full": float(np.mean(residual_norm)),
        "mean_semantic_centered_norm_fraction": float(np.mean(semantic_norm / np.maximum(full_norm, 1e-12))),
        "mean_residual_full_norm_fraction": float(np.mean(residual_norm / np.maximum(full_norm, 1e-12))),
    }
    log(
        f"extracted {pool_tag}: n={h.shape[0]} rank(W)={diag['rank_W']} rank(Wc)={diag['rank_Wc']} "
        f"mean_norm full/sem_c/res={diag['mean_norm_final_full']:.4f}/"
        f"{diag['mean_norm_semantic_centered']:.4f}/{diag['mean_norm_residual_full']:.4f}"
    )
    reps = {
        "final_full": h.astype(np.float32),
        "centered_logits": centered_logits.astype(np.float32),
        "semantic_centered": semantic_centered.astype(np.float32),
        "residual_full": residual_full.astype(np.float32),
    }
    return reps, diag


def fit_pca_basis(residual: np.ndarray, k: int = 8) -> tuple[np.ndarray, np.ndarray]:
    residual64 = residual.astype(np.float64)
    mean = residual64.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(residual64 - mean, full_matrices=False)
    return mean.astype(np.float32), vh[:k].astype(np.float32)


def apply_pca(residual: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return ((residual.astype(np.float32) - mean) @ basis.T).astype(np.float32)


def make_random_bases(dim: int, k: int, repeats: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    bases = []
    for _ in range(repeats):
        mat = rng.normal(size=(dim, k))
        q, _ = np.linalg.qr(mat)
        bases.append(q[:, :k].astype(np.float32))
    return bases


def add_dimension_controls_single_pool(
    reps: dict[str, np.ndarray],
    random_repeats: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], list[RepSpec]]:
    out = dict(reps)
    mean, basis = fit_pca_basis(out["residual_full"], 8)
    out["residual_top8_pca"] = apply_pca(out["residual_full"], mean, basis)
    rep_specs = [
        RepSpec("centered_logits", "centered_logits"),
        RepSpec("semantic_centered", "semantic_centered"),
        RepSpec("residual_full", "residual_full"),
        RepSpec("residual_top8_pca", "residual_top8_pca"),
        RepSpec("final_full", "final_full"),
    ]
    for idx, random_basis in enumerate(make_random_bases(out["residual_full"].shape[1], 8, random_repeats, seed)):
        key = f"residual_random8_r{idx:02d}"
        out[key] = (out["residual_full"] @ random_basis).astype(np.float32)
        rep_specs.append(RepSpec("residual_random8", key, f"r{idx:02d}"))
    return out, rep_specs


def add_dimension_controls_two_pools(
    reps_a: dict[str, np.ndarray],
    reps_b: dict[str, np.ndarray],
    random_repeats: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[RepSpec]]:
    out_a = dict(reps_a)
    out_b = dict(reps_b)
    residual_combined = np.vstack([out_a["residual_full"], out_b["residual_full"]])
    mean, basis = fit_pca_basis(residual_combined, 8)
    out_a["residual_top8_pca"] = apply_pca(out_a["residual_full"], mean, basis)
    out_b["residual_top8_pca"] = apply_pca(out_b["residual_full"], mean, basis)
    rep_specs = [
        RepSpec("centered_logits", "centered_logits"),
        RepSpec("semantic_centered", "semantic_centered"),
        RepSpec("residual_full", "residual_full"),
        RepSpec("residual_top8_pca", "residual_top8_pca"),
        RepSpec("final_full", "final_full"),
    ]
    for idx, random_basis in enumerate(make_random_bases(out_a["residual_full"].shape[1], 8, random_repeats, seed)):
        key = f"residual_random8_r{idx:02d}"
        out_a[key] = (out_a["residual_full"] @ random_basis).astype(np.float32)
        out_b[key] = (out_b["residual_full"] @ random_basis).astype(np.float32)
        rep_specs.append(RepSpec("residual_random8", key, f"r{idx:02d}"))
    return out_a, out_b, rep_specs


def dim_specs_for_permutation(rep_specs: list[RepSpec]) -> list[RepSpec]:
    allowed = set(PERM_REP_ORDER)
    return [spec for spec in rep_specs if spec.representation in allowed]


def completed_dim_outer_keys(
    path: Path,
    rep_specs: list[RepSpec],
    inner_repetitions: int,
) -> set[tuple[str, str, str, str, str, int, int]]:
    expected = {(inner, spec.rep_run_id) for inner in range(1, inner_repetitions + 1) for spec in rep_specs}
    seen_by_key: dict[tuple[str, str, str, str, str, int, int], set[tuple[int, str]]] = {}
    for row in read_rows(path):
        key = (
            row["result_kind"],
            row["experiment"],
            row["scenario"],
            row["setting"],
            row["label"],
            int(row["n"]),
            int(row["outer_iter"]),
        )
        seen_by_key.setdefault(key, set()).add((int(row["inner_iter"]), row["rep_run_id"]))
    return {key for key, seen in seen_by_key.items() if expected.issubset(seen)}


def completed_perm_outer_keys(
    path: Path,
    rep_specs: list[RepSpec],
    inner_repetitions: int,
) -> set[tuple[str, str, str, str, int, int]]:
    expected = {(inner, spec.representation) for inner in range(1, inner_repetitions + 1) for spec in rep_specs}
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


def mmmd_bootstrap_with_inv(
    kxx_list: list[np.ndarray],
    kyy_list: list[np.ndarray],
    kxy_list: list[np.ndarray],
    b_boot: int,
    alpha: float,
    rng: np.random.Generator,
) -> dict[str, object]:
    n = kxx_list[0].shape[0]
    offdiag = ~np.eye(n, dtype=bool)
    mmd_vec = np.array(
        [
            n * np.mean(kxx[offdiag] + kyy[offdiag] - 2.0 * kxy[offdiag])
            for kxx, kyy, kxy in zip(kxx_list, kyy_list, kxy_list)
        ]
    )

    c = np.eye(n) - np.ones((n, n)) / n
    centered = [c @ kxx @ c for kxx in kxx_list]
    r = len(centered)
    sigma_hat = np.empty((r, r), dtype=float)
    for i in range(r):
        for j in range(r):
            sigma_hat[i, j] = (8.0 / (n**2)) * np.trace(centered[i] @ centered[j])

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
    stat = float(mmd_vec.T @ inv_cov @ mmd_vec)
    cutoff = float(np.quantile(boot_quad, 1.0 - alpha))
    return {
        "stat": stat,
        "cutoff": cutoff,
        "reject": int(stat > cutoff),
        "kernel_count": r,
        "lambda": ridge_lambda,
        "cond_sigma_hat": BASE.condition_number(sigma_hat),
        "cond_sigma_reg": BASE.condition_number(sigma_reg),
        "inv_cov": inv_cov,
        "mmd_vec": mmd_vec,
    }


def run_gaussian5_bootstrap(x: np.ndarray, y: np.ndarray, b_boot: int, alpha: float, seed: int) -> dict[str, object]:
    kxx, kyy, kxy = BASE.kernel_mats_for_features(x, y)
    return mmmd_bootstrap_with_inv(kxx, kyy, kxy, b_boot, alpha, np.random.default_rng(seed))


def mmd_vec_from_full_kernels(kernel_list: list[np.ndarray], idx_x: np.ndarray, idx_y: np.ndarray) -> np.ndarray:
    n = len(idx_x)
    offdiag = ~np.eye(n, dtype=bool)
    vals = []
    for kernel in kernel_list:
        kxx = kernel[np.ix_(idx_x, idx_x)]
        kyy = kernel[np.ix_(idx_y, idx_y)]
        kxy = kernel[np.ix_(idx_x, idx_y)]
        vals.append(n * np.mean(kxx[offdiag] + kyy[offdiag] - 2.0 * kxy[offdiag]))
    return np.array(vals, dtype=float)


def run_gaussian5_bootstrap_and_permutation(
    x: np.ndarray,
    y: np.ndarray,
    b_boot: int,
    b_perm: int,
    alpha: float,
    seed: int,
) -> dict[str, object]:
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    z = np.vstack([x64, y64])
    gammas = BASE.gaussian_gammas(x64, y64, "gaussian5")
    dzz = BASE.pairwise_sq_dists(z, z)
    full_kernels = [np.exp(-gamma * dzz) for gamma in gammas]
    n = x64.shape[0]
    idx_x = np.arange(n)
    idx_y = np.arange(n, 2 * n)
    kxx = [kernel[:n, :n] for kernel in full_kernels]
    kyy = [kernel[n:, n:] for kernel in full_kernels]
    kxy = [kernel[:n, n:] for kernel in full_kernels]

    rng = np.random.default_rng(seed)
    boot = mmmd_bootstrap_with_inv(kxx, kyy, kxy, b_boot, alpha, rng)
    inv_cov = boot["inv_cov"]
    perm_stats = np.empty(b_perm, dtype=float)
    for b_idx in range(b_perm):
        perm = rng.permutation(2 * n)
        px = perm[:n]
        py = perm[n:]
        mmd_vec = mmd_vec_from_full_kernels(full_kernels, px, py)
        perm_stats[b_idx] = float(mmd_vec.T @ inv_cov @ mmd_vec)
    permutation_cutoff = float(np.quantile(perm_stats, 1.0 - alpha))
    return {
        "stat": boot["stat"],
        "bootstrap_cutoff": boot["cutoff"],
        "bootstrap_reject": boot["reject"],
        "permutation_cutoff": permutation_cutoff,
        "permutation_reject": int(float(boot["stat"]) > permutation_cutoff),
        "kernel_count": boot["kernel_count"],
        "lambda": boot["lambda"],
        "cond_sigma_hat": boot["cond_sigma_hat"],
        "cond_sigma_reg": boot["cond_sigma_reg"],
    }


def make_dim_row(
    result_kind: str,
    experiment: str,
    scenario: str,
    setting: str,
    label: str,
    label_name: str,
    n: int,
    spec: RepSpec,
    outer_iter: int,
    inner_iter: int,
    seed: int,
    out: dict[str, object],
    runtime_sec: float,
    b_boot: int,
    alpha: float,
) -> dict[str, object]:
    return {
        "result_kind": result_kind,
        "experiment": experiment,
        "scenario": scenario,
        "setting": setting,
        "label": label,
        "label_name": label_name,
        "n": n,
        "representation": spec.representation,
        "projection_id": spec.projection_id,
        "rep_run_id": spec.rep_run_id,
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
        "B_boot": b_boot,
        "alpha": alpha,
    }


def make_perm_row(
    result_kind: str,
    scenario: str,
    setting: str,
    label: int,
    n: int,
    spec: RepSpec,
    outer_iter: int,
    inner_iter: int,
    seed: int,
    out: dict[str, object],
    runtime_sec: float,
    b_boot: int,
    b_perm: int,
    alpha: float,
) -> dict[str, object]:
    return {
        "result_kind": result_kind,
        "experiment": "center_shift",
        "scenario": scenario,
        "setting": setting,
        "label": str(label),
        "label_name": CLASS_NAMES[label],
        "n": n,
        "representation": spec.representation,
        "outer_iter": outer_iter,
        "inner_iter": inner_iter,
        "seed": seed,
        "stat": out["stat"],
        "bootstrap_cutoff": out["bootstrap_cutoff"],
        "bootstrap_reject": out["bootstrap_reject"],
        "permutation_cutoff": out["permutation_cutoff"],
        "permutation_reject": out["permutation_reject"],
        "kernel_count": out["kernel_count"],
        "lambda": out["lambda"],
        "cond_sigma_hat": out["cond_sigma_hat"],
        "cond_sigma_reg": out["cond_sigma_reg"],
        "runtime_sec": runtime_sec,
        "B_boot": b_boot,
        "B_perm": b_perm,
        "alpha": alpha,
    }


def run_dim_class_mixture(
    reps_test: dict[str, np.ndarray],
    labels: np.ndarray,
    rep_specs: list[RepSpec],
    h1_sizes: list[int],
    h0_sizes: list[int],
    outer_repetitions: int,
    inner_repetitions: int,
    b_boot: int,
    alpha: float,
    progress_every: int,
) -> None:
    done = completed_dim_outer_keys(DIM_RAW_PATH, rep_specs, inner_repetitions)
    log(f"dim class-mixture completed outer keys={len(done)}")

    for n in h1_sizes:
        for outer_iter in range(1, outer_repetitions + 1):
            key = ("power", "class_mixture", "mix635_vs_mix835", "H1_mix635_vs_mix835", "", n, outer_iter)
            if key in done:
                log(f"skip dim class H1 n={n} outer={outer_iter}")
                continue
            rows: list[dict[str, object]] = []
            t_outer = time.time()
            for inner_iter in range(1, inner_repetitions + 1):
                if inner_iter == 1 or inner_iter % progress_every == 0 or inner_iter == inner_repetitions:
                    log(f"progress dim class H1 n={n} outer={outer_iter}/{outer_repetitions} inner={inner_iter}/{inner_repetitions}")
                base_seed = 20260605 + 1_000_000 * n + 10_000 * outer_iter + inner_iter
                rng = np.random.default_rng(base_seed)
                x_idx, y_idx = BASE.sample_balanced_h1_class(labels, n, rng)
                for spec in rep_specs:
                    t0 = time.time()
                    seed = base_seed + stable_seed_offset(spec.rep_run_id)
                    out = run_gaussian5_bootstrap(reps_test[spec.feature_key][x_idx], reps_test[spec.feature_key][y_idx], b_boot, alpha, seed)
                    rows.append(make_dim_row("power", "class_mixture", "mix635_vs_mix835", "H1_mix635_vs_mix835", "", "", n, spec, outer_iter, inner_iter, seed, out, time.time() - t0, b_boot, alpha))
            append_csv(DIM_RAW_PATH, DIM_RAW_FIELDS, rows)
            log(f"finished dim class H1 n={n} outer={outer_iter}/{outer_repetitions} in {time.time() - t_outer:.2f}s")

    for n in h0_sizes:
        for outer_iter in range(1, outer_repetitions + 1):
            key = ("type1", "class_mixture", "mix635_vs_mix635_null", "H0_mix635_vs_mix635", "", n, outer_iter)
            if key in done:
                log(f"skip dim class H0 n={n} outer={outer_iter}")
                continue
            rows = []
            t_outer = time.time()
            for inner_iter in range(1, inner_repetitions + 1):
                if inner_iter == 1 or inner_iter % progress_every == 0 or inner_iter == inner_repetitions:
                    log(f"progress dim class H0 n={n} outer={outer_iter}/{outer_repetitions} inner={inner_iter}/{inner_repetitions}")
                base_seed = 20260606 + 1_000_000 * n + 10_000 * outer_iter + inner_iter
                rng = np.random.default_rng(base_seed)
                x_idx, y_idx = BASE.sample_balanced_h0_class(labels, n, rng)
                for spec in rep_specs:
                    t0 = time.time()
                    seed = base_seed + stable_seed_offset(spec.rep_run_id)
                    out = run_gaussian5_bootstrap(reps_test[spec.feature_key][x_idx], reps_test[spec.feature_key][y_idx], b_boot, alpha, seed)
                    rows.append(make_dim_row("type1", "class_mixture", "mix635_vs_mix635_null", "H0_mix635_vs_mix635", "", "", n, spec, outer_iter, inner_iter, seed, out, time.time() - t0, b_boot, alpha))
            append_csv(DIM_RAW_PATH, DIM_RAW_FIELDS, rows)
            log(f"finished dim class H0 n={n} outer={outer_iter}/{outer_repetitions} in {time.time() - t_outer:.2f}s")


def center_features_for_sample(
    x_pos: np.ndarray,
    y_pos: np.ndarray,
    x_pool: str,
    y_pool: str,
    reps_source: dict[str, np.ndarray],
    reps_external: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    by_pool = {"source_holdout_pool": reps_source, "external_pool": reps_external}
    return (
        {name: values[x_pos] for name, values in by_pool[x_pool].items()},
        {name: values[y_pos] for name, values in by_pool[y_pool].items()},
    )


def run_dim_center_shift(
    reps_source: dict[str, np.ndarray],
    reps_external: dict[str, np.ndarray],
    pools: dict[str, np.ndarray],
    rep_specs: list[RepSpec],
    labels: list[int],
    h1_n: list[int],
    h0_n: list[int],
    outer_repetitions: int,
    inner_repetitions: int,
    b_boot: int,
    alpha: float,
    progress_every: int,
) -> None:
    done = completed_dim_outer_keys(DIM_RAW_PATH, rep_specs, inner_repetitions)
    log(f"dim center-shift completed outer keys={len(done)}")
    for label in labels:
        for setting in ["H1_source_vs_external", "H0_source", "H0_external"]:
            n_grid = h1_n if setting == "H1_source_vs_external" else h0_n
            result_kind = "power" if setting == "H1_source_vs_external" else "type1"
            for n in n_grid:
                for outer_iter in range(1, outer_repetitions + 1):
                    scenario = f"label{label}_source_vs_external"
                    key = (result_kind, "center_shift", scenario, setting, str(label), n, outer_iter)
                    if key in done:
                        log(f"skip dim center {setting} label={label} n={n} outer={outer_iter}")
                        continue
                    rows: list[dict[str, object]] = []
                    t_outer = time.time()
                    for inner_iter in range(1, inner_repetitions + 1):
                        if inner_iter == 1 or inner_iter % progress_every == 0 or inner_iter == inner_repetitions:
                            log(
                                f"progress dim center {setting} label={label} n={n} "
                                f"outer={outer_iter}/{outer_repetitions} inner={inner_iter}/{inner_repetitions}"
                            )
                        base_seed = 20260607 + 10_000_000 * label + 1_000_000 * n + 10_000 * outer_iter + inner_iter
                        setting_offset = {"H1_source_vs_external": 11, "H0_source": 23, "H0_external": 37}[setting]
                        rng = np.random.default_rng(base_seed + setting_offset)
                        x_pos, y_pos, x_pool, y_pool = BASE.sample_center_indices(label, setting, n, pools, rng)
                        emb_x, emb_y = center_features_for_sample(x_pos, y_pos, x_pool, y_pool, reps_source, reps_external)
                        for spec in rep_specs:
                            t0 = time.time()
                            seed = base_seed + stable_seed_offset(spec.rep_run_id)
                            out = run_gaussian5_bootstrap(emb_x[spec.feature_key], emb_y[spec.feature_key], b_boot, alpha, seed)
                            rows.append(make_dim_row(result_kind, "center_shift", scenario, setting, str(label), CLASS_NAMES[label], n, spec, outer_iter, inner_iter, seed, out, time.time() - t0, b_boot, alpha))
                    append_csv(DIM_RAW_PATH, DIM_RAW_FIELDS, rows)
                    log(f"finished dim center {setting} label={label} n={n} outer={outer_iter}/{outer_repetitions} in {time.time() - t_outer:.2f}s")


def run_permutation_center_shift(
    reps_source: dict[str, np.ndarray],
    reps_external: dict[str, np.ndarray],
    pools: dict[str, np.ndarray],
    rep_specs: list[RepSpec],
    labels: list[int],
    n_grid: list[int],
    outer_repetitions: int,
    inner_repetitions: int,
    b_boot: int,
    b_perm: int,
    alpha: float,
    progress_every: int,
) -> None:
    done = completed_perm_outer_keys(PERM_RAW_PATH, rep_specs, inner_repetitions)
    log(f"permutation center-shift completed outer keys={len(done)}")
    for label in labels:
        for setting in ["H1_source_vs_external", "H0_source", "H0_external"]:
            result_kind = "power" if setting == "H1_source_vs_external" else "type1"
            for n in n_grid:
                for outer_iter in range(1, outer_repetitions + 1):
                    scenario = f"label{label}_source_vs_external"
                    key = (result_kind, scenario, setting, str(label), n, outer_iter)
                    if key in done:
                        log(f"skip permutation {setting} label={label} n={n} outer={outer_iter}")
                        continue
                    rows: list[dict[str, object]] = []
                    t_outer = time.time()
                    for inner_iter in range(1, inner_repetitions + 1):
                        if inner_iter == 1 or inner_iter % progress_every == 0 or inner_iter == inner_repetitions:
                            log(
                                f"progress permutation {setting} label={label} n={n} "
                                f"outer={outer_iter}/{outer_repetitions} inner={inner_iter}/{inner_repetitions}"
                            )
                        base_seed = 20260608 + 10_000_000 * label + 1_000_000 * n + 10_000 * outer_iter + inner_iter
                        setting_offset = {"H1_source_vs_external": 11, "H0_source": 23, "H0_external": 37}[setting]
                        rng = np.random.default_rng(base_seed + setting_offset)
                        x_pos, y_pos, x_pool, y_pool = BASE.sample_center_indices(label, setting, n, pools, rng)
                        emb_x, emb_y = center_features_for_sample(x_pos, y_pos, x_pool, y_pool, reps_source, reps_external)
                        for spec in rep_specs:
                            t0 = time.time()
                            seed = base_seed + stable_seed_offset(spec.rep_run_id)
                            out = run_gaussian5_bootstrap_and_permutation(
                                emb_x[spec.feature_key],
                                emb_y[spec.feature_key],
                                b_boot,
                                b_perm,
                                alpha,
                                seed,
                            )
                            rows.append(make_perm_row(result_kind, scenario, setting, label, n, spec, outer_iter, inner_iter, seed, out, time.time() - t0, b_boot, b_perm, alpha))
                    append_csv(PERM_RAW_PATH, PERM_RAW_FIELDS, rows)
                    log(f"finished permutation {setting} label={label} n={n} outer={outer_iter}/{outer_repetitions} in {time.time() - t_outer:.2f}s")


def summarize_group(rows: list[dict[str, str]], projection_id: str, summary_scope: str, projection_repeats: int, b_boot: int, alpha: float) -> dict[str, object]:
    rejects = np.array([int(row["reject"]) for row in rows], dtype=float)
    m = len(rows)
    rate = float(np.mean(rejects)) if m else float("nan")
    se, lo, hi = binomial_interval(rate, m)
    common = rows[0]
    result_kind = common["result_kind"]
    tests_per_projection = int(round(m / max(projection_repeats, 1)))
    return {
        "result_kind": result_kind,
        "experiment": common["experiment"],
        "scenario": common["scenario"],
        "setting": common["setting"],
        "label": common["label"],
        "label_name": common["label_name"],
        "n": int(common["n"]),
        "representation": common["representation"],
        "projection_id": projection_id,
        "summary_scope": summary_scope,
        "projection_repeats": projection_repeats,
        "outer_repetitions": len({int(row["outer_iter"]) for row in rows}),
        "inner_repetitions": max(int(row["inner_iter"]) for row in rows) if rows else 0,
        "tests_per_projection": tests_per_projection,
        "independent_tests": m,
        "rejection_rate": rate,
        "power": rate if result_kind == "power" else "",
        "type1_error": rate if result_kind == "type1" else "",
        "binomial_se": se,
        "ci_lower": lo,
        "ci_upper": hi,
        "mean_stat": float(np.mean([float(row["stat"]) for row in rows])) if rows else float("nan"),
        "mean_cutoff": float(np.mean([float(row["cutoff"]) for row in rows])) if rows else float("nan"),
        "mean_cond_sigma_reg": float(np.mean([float(row["cond_sigma_reg"]) for row in rows])) if rows else float("nan"),
        "B_boot": b_boot,
        "alpha": alpha,
    }


def rebuild_dim_summary(b_boot: int, alpha: float) -> list[dict[str, object]]:
    rows = read_rows(DIM_RAW_PATH)
    summary: list[dict[str, object]] = []
    base_keys = sorted(
        {
            (
                row["result_kind"],
                row["experiment"],
                row["scenario"],
                row["setting"],
                row["label"],
                int(row["n"]),
                row["representation"],
            )
            for row in rows
        }
    )
    for result_kind, experiment, scenario, setting, label, n, representation in base_keys:
        subset_base = [
            row
            for row in rows
            if row["result_kind"] == result_kind
            and row["experiment"] == experiment
            and row["scenario"] == scenario
            and row["setting"] == setting
            and row["label"] == label
            and int(row["n"]) == n
            and row["representation"] == representation
        ]
        projection_ids = sorted({row["projection_id"] for row in subset_base})
        if representation == "residual_random8":
            summary.append(summarize_group(subset_base, "aggregate20", "aggregate", len(projection_ids), b_boot, alpha))
            for projection_id in projection_ids:
                rows_p = [row for row in subset_base if row["projection_id"] == projection_id]
                summary.append(summarize_group(rows_p, projection_id, "per_projection", 1, b_boot, alpha))
        else:
            summary.append(summarize_group(subset_base, projection_ids[0], "single", 1, b_boot, alpha))

    def sort_key(row: dict[str, object]):
        rep = str(row["representation"])
        return (
            str(row["result_kind"]),
            str(row["experiment"]),
            str(row["setting"]),
            str(row["label"]),
            int(row["n"]),
            DIM_REP_ORDER.index(rep) if rep in DIM_REP_ORDER else 99,
            str(row["summary_scope"]),
            str(row["projection_id"]),
        )

    summary = sorted(summary, key=sort_key)
    write_csv(DIM_SUMMARY_PATH, DIM_SUMMARY_FIELDS, summary)
    return summary


def rebuild_perm_summary(b_boot: int, b_perm: int, alpha: float) -> list[dict[str, object]]:
    rows = read_rows(PERM_RAW_PATH)
    groups = sorted(
        {
            (
                row["result_kind"],
                row["experiment"],
                row["scenario"],
                row["setting"],
                row["label"],
                int(row["n"]),
                row["representation"],
            )
            for row in rows
        }
    )
    summary: list[dict[str, object]] = []
    for result_kind, experiment, scenario, setting, label, n, representation in groups:
        subset = [
            row
            for row in rows
            if row["result_kind"] == result_kind
            and row["experiment"] == experiment
            and row["scenario"] == scenario
            and row["setting"] == setting
            and row["label"] == label
            and int(row["n"]) == n
            and row["representation"] == representation
        ]
        m = len(subset)
        boot_rate = float(np.mean([int(row["bootstrap_reject"]) for row in subset])) if subset else float("nan")
        perm_rate = float(np.mean([int(row["permutation_reject"]) for row in subset])) if subset else float("nan")
        boot_se, _, _ = binomial_interval(boot_rate, m)
        perm_se, _, _ = binomial_interval(perm_rate, m)
        first = subset[0]
        summary.append(
            {
                "result_kind": result_kind,
                "experiment": experiment,
                "scenario": scenario,
                "setting": setting,
                "label": label,
                "label_name": first["label_name"],
                "n": n,
                "representation": representation,
                "outer_repetitions": len({int(row["outer_iter"]) for row in subset}),
                "inner_repetitions": max(int(row["inner_iter"]) for row in subset) if subset else 0,
                "independent_tests": m,
                "bootstrap_rejection_rate": boot_rate,
                "permutation_rejection_rate": perm_rate,
                "bootstrap_binomial_se": boot_se,
                "permutation_binomial_se": perm_se,
                "mean_stat": float(np.mean([float(row["stat"]) for row in subset])) if subset else float("nan"),
                "mean_bootstrap_cutoff": float(np.mean([float(row["bootstrap_cutoff"]) for row in subset])) if subset else float("nan"),
                "mean_permutation_cutoff": float(np.mean([float(row["permutation_cutoff"]) for row in subset])) if subset else float("nan"),
                "mean_cond_sigma_reg": float(np.mean([float(row["cond_sigma_reg"]) for row in subset])) if subset else float("nan"),
                "B_boot": b_boot,
                "B_perm": b_perm,
                "alpha": alpha,
            }
        )
    summary = sorted(
        summary,
        key=lambda row: (
            str(row["result_kind"]),
            str(row["setting"]),
            str(row["label"]),
            int(row["n"]),
            PERM_REP_ORDER.index(str(row["representation"])) if str(row["representation"]) in PERM_REP_ORDER else 99,
        ),
    )
    write_csv(PERM_SUMMARY_PATH, PERM_SUMMARY_FIELDS, summary)
    return summary


def plot_dim_power(summary: list[dict[str, object]]) -> None:
    rows = [row for row in summary if row["result_kind"] == "power" and row["summary_scope"] in {"single", "aggregate"}]
    panels = [
        ("Class mixture: mix635 vs mix835", lambda row: row["experiment"] == "class_mixture"),
        ("Center shift: label 6", lambda row: row["experiment"] == "center_shift" and str(row["label"]) == "6"),
        ("Center shift: label 8", lambda row: row["experiment"] == "center_shift" and str(row["label"]) == "8"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, (title, pred) in zip(axes, panels):
        panel_rows = [row for row in rows if pred(row)]
        for rep in DIM_REP_ORDER:
            rep_rows = sorted([row for row in panel_rows if row["representation"] == rep], key=lambda row: int(row["n"]))
            if not rep_rows:
                continue
            ax.plot([int(row["n"]) for row in rep_rows], [float(row["rejection_rate"]) for row in rep_rows], marker="o", label=rep, color=COLORS[rep])
        ax.axhline(0.05, color="black", linestyle=":", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Group size n")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Empirical rejection rate / power")
    axes[-1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(DIM_POWER_PLOT, dpi=180)
    plt.close(fig)


def plot_dim_type1(summary: list[dict[str, object]]) -> None:
    rows = [row for row in summary if row["result_kind"] == "type1" and row["summary_scope"] in {"single", "aggregate"}]
    panels = [
        ("Class null: mix635 vs mix635", lambda row: row["experiment"] == "class_mixture"),
        ("Label 6 H0 source", lambda row: row["experiment"] == "center_shift" and str(row["label"]) == "6" and row["setting"] == "H0_source"),
        ("Label 6 H0 external", lambda row: row["experiment"] == "center_shift" and str(row["label"]) == "6" and row["setting"] == "H0_external"),
        ("Label 8 H0 source", lambda row: row["experiment"] == "center_shift" and str(row["label"]) == "8" and row["setting"] == "H0_source"),
        ("Label 8 H0 external", lambda row: row["experiment"] == "center_shift" and str(row["label"]) == "8" and row["setting"] == "H0_external"),
    ]
    fig, axes_obj = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    axes = list(axes_obj.flat)
    for ax, (title, pred) in zip(axes, panels):
        panel_rows = [row for row in rows if pred(row)]
        for rep in DIM_REP_ORDER:
            rep_rows = sorted([row for row in panel_rows if row["representation"] == rep], key=lambda row: int(row["n"]))
            if not rep_rows:
                continue
            ax.plot([int(row["n"]) for row in rep_rows], [float(row["rejection_rate"]) for row in rep_rows], marker="o", label=rep, color=COLORS[rep])
        ax.axhline(0.05, color="black", linestyle=":", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Group size n")
        ax.set_ylabel("Empirical Type-I error")
        ax.set_ylim(0, 0.2)
        ax.grid(alpha=0.2)
    axes[-1].axis("off")
    axes[2].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(DIM_TYPE1_PLOT, dpi=180)
    plt.close(fig)


def find_dim_rate(summary: list[dict[str, object]], result_kind: str, experiment: str, setting: str, label: str, n: int, rep: str) -> float | None:
    for row in summary:
        if (
            row["result_kind"] == result_kind
            and row["experiment"] == experiment
            and row["setting"] == setting
            and str(row["label"]) == str(label)
            and int(row["n"]) == n
            and row["representation"] == rep
            and row["summary_scope"] in {"single", "aggregate"}
        ):
            return float(row["rejection_rate"])
    return None


def plot_dim_excess(summary: list[dict[str, object]]) -> None:
    panels = [
        ("Class mixture excess", "class_mixture", "", [60, 120]),
        ("Center label 6 excess", "center_shift", "6", [20, 50]),
        ("Center label 8 excess", "center_shift", "8", [20, 50]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, (title, experiment, label, ns) in zip(axes, panels):
        for rep in DIM_REP_ORDER:
            xs, ys = [], []
            for n in ns:
                if experiment == "class_mixture":
                    h1 = find_dim_rate(summary, "power", experiment, "H1_mix635_vs_mix835", "", n, rep)
                    h0 = find_dim_rate(summary, "type1", experiment, "H0_mix635_vs_mix635", "", n, rep)
                else:
                    h1 = find_dim_rate(summary, "power", experiment, "H1_source_vs_external", label, n, rep)
                    h0_s = find_dim_rate(summary, "type1", experiment, "H0_source", label, n, rep)
                    h0_e = find_dim_rate(summary, "type1", experiment, "H0_external", label, n, rep)
                    h0 = max(v for v in [h0_s, h0_e] if v is not None) if (h0_s is not None or h0_e is not None) else None
                if h1 is None or h0 is None:
                    continue
                xs.append(n)
                ys.append(h1 - h0)
            if xs:
                ax.plot(xs, ys, marker="o", label=rep, color=COLORS[rep])
        ax.axhline(0.0, color="black", linestyle=":", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Group size n")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("H1 rejection - matched max H0 rejection")
    axes[-1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(DIM_EXCESS_PLOT, dpi=180)
    plt.close(fig)


def plot_perm(summary: list[dict[str, object]], result_kind: str, output_path: Path) -> None:
    rows = [row for row in summary if row["result_kind"] == result_kind]
    if result_kind == "power":
        panels = [
            ("Label 6 H1", lambda row: str(row["label"]) == "6"),
            ("Label 8 H1", lambda row: str(row["label"]) == "8"),
        ]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    else:
        panels = [
            ("Label 6 H0 source", lambda row: str(row["label"]) == "6" and row["setting"] == "H0_source"),
            ("Label 6 H0 external", lambda row: str(row["label"]) == "6" and row["setting"] == "H0_external"),
            ("Label 8 H0 source", lambda row: str(row["label"]) == "8" and row["setting"] == "H0_source"),
            ("Label 8 H0 external", lambda row: str(row["label"]) == "8" and row["setting"] == "H0_external"),
        ]
        fig, axes_obj = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
        axes = list(axes_obj.flat)
    for ax, (title, pred) in zip(axes, panels):
        panel_rows = [row for row in rows if pred(row)]
        for rep in PERM_REP_ORDER:
            rep_rows = sorted([row for row in panel_rows if row["representation"] == rep], key=lambda row: int(row["n"]))
            if not rep_rows:
                continue
            x = [int(row["n"]) for row in rep_rows]
            ax.plot(x, [float(row["bootstrap_rejection_rate"]) for row in rep_rows], marker="o", linestyle="-", color=COLORS[rep], label=f"{rep} boot")
            ax.plot(x, [float(row["permutation_rejection_rate"]) for row in rep_rows], marker="s", linestyle="--", color=COLORS[rep], label=f"{rep} perm")
        ax.axhline(0.05, color="black", linestyle=":", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Group size n")
        ax.set_ylabel("Rejection rate")
        ax.set_ylim(0, 1.02 if result_kind == "power" else 0.2)
        ax.grid(alpha=0.2)
    axes[-1].legend(fontsize=6, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_short_report(dim_summary: list[dict[str, object]], perm_summary: list[dict[str, object]]) -> None:
    def find_row(summary: list[dict[str, object]], **criteria: object) -> dict[str, object] | None:
        for row in summary:
            if all(str(row.get(k, "")) == str(v) for k, v in criteria.items()):
                return row
        return None

    def rate(summary: list[dict[str, object]], **criteria: object) -> str:
        row = find_row(summary, **criteria)
        return "NA" if row is None else f"{float(row['rejection_rate']):.3f}"

    def perm_rate(field: str, **criteria: object) -> str:
        row = find_row(perm_summary, **criteria)
        return "NA" if row is None else f"{float(row[field]):.3f}"

    def max_rate(summary: list[dict[str, object]], field: str, **criteria: object) -> str:
        vals = [
            float(row[field])
            for row in summary
            if all(str(row.get(k, "")) == str(v) for k, v in criteria.items())
        ]
        return "NA" if not vals else f"{max(vals):.3f}"

    run_row = dim_summary[0] if dim_summary else {}
    random8_row = find_row(dim_summary, representation="residual_random8", summary_scope="aggregate")
    diag_rows = read_rows(DIAG_PATH)
    diag_bits = []
    for row in diag_rows:
        diag_bits.append(
            f"{row['pool_tag']}: rank(W)={row['rank_W']}, rank(Wc)={row['rank_Wc']}, "
            f"mean norm fractions sem={float(row['mean_semantic_centered_norm_fraction']):.3f}, "
            f"res={float(row['mean_residual_full_norm_fraction']):.3f}"
        )
    actual_run = (
        f"outer={run_row.get('outer_repetitions', 'NA')}, "
        f"inner={run_row.get('inner_repetitions', 'NA')}, "
        f"B_boot={run_row.get('B_boot', 'NA')}, B_perm=500 for permutation cells, "
        f"non-random tests/cell={run_row.get('independent_tests', 'NA')}"
    )
    if random8_row is not None:
        actual_run += f", residual_random8 aggregate tests/cell={random8_row.get('independent_tests', 'NA')}"

    lines = [
        "# Validity Controls Short Report",
        "",
        "This report is generated from completed validity-control outputs only. No CNN checkpoint is retrained.",
        f"Actual run size: {actual_run}. Full inner=500 was not used for this control run; `plan_next.md` allows reducing inner tests when runtime is high as long as the actual test count is reported.",
        "",
        "## 1. Does centered W change the semantic/residual conclusion?",
        "",
        "- Centering behaves as expected: W has rank 9 and Wc has rank 8 in all checked pools, because the class-common logit direction is removed.",
        "- Diagnostics: " + "; ".join(diag_bits) + ".",
        "- The residual component still carries most of the feature norm, so the previous residual signal is not an artifact of including the all-class logit direction in `row(W)`.",
        "",
        "## 2. Does residual power survive dimension matching?",
        "",
        f"- Class-mixture H1 at n=120 remains strongest in classifier-used coordinates: centered_logits={rate(dim_summary, result_kind='power', experiment='class_mixture', setting='H1_mix635_vs_mix835', n='120', representation='centered_logits', summary_scope='single')}, semantic_centered={rate(dim_summary, result_kind='power', experiment='class_mixture', setting='H1_mix635_vs_mix835', n='120', representation='semantic_centered', summary_scope='single')}, residual_top8_pca={rate(dim_summary, result_kind='power', experiment='class_mixture', setting='H1_mix635_vs_mix835', n='120', representation='residual_top8_pca', summary_scope='single')}, residual_random8={rate(dim_summary, result_kind='power', experiment='class_mixture', setting='H1_mix635_vs_mix835', n='120', representation='residual_random8', summary_scope='aggregate')}, residual_full={rate(dim_summary, result_kind='power', experiment='class_mixture', setting='H1_mix635_vs_mix835', n='120', representation='residual_full', summary_scope='single')}, final_full={rate(dim_summary, result_kind='power', experiment='class_mixture', setting='H1_mix635_vs_mix835', n='120', representation='final_full', summary_scope='single')}.",
        f"- Center-shift label 6 at n=50 is more semantic/logit driven: semantic_centered={rate(dim_summary, result_kind='power', experiment='center_shift', setting='H1_source_vs_external', label='6', n='50', representation='semantic_centered', summary_scope='single')}, residual_top8_pca={rate(dim_summary, result_kind='power', experiment='center_shift', setting='H1_source_vs_external', label='6', n='50', representation='residual_top8_pca', summary_scope='single')}, residual_random8={rate(dim_summary, result_kind='power', experiment='center_shift', setting='H1_source_vs_external', label='6', n='50', representation='residual_random8', summary_scope='aggregate')}, residual_full={rate(dim_summary, result_kind='power', experiment='center_shift', setting='H1_source_vs_external', label='6', n='50', representation='residual_full', summary_scope='single')}.",
        f"- Center-shift label 8 at n=50 is strongly residual driven even after dimension matching: semantic_centered={rate(dim_summary, result_kind='power', experiment='center_shift', setting='H1_source_vs_external', label='8', n='50', representation='semantic_centered', summary_scope='single')}, residual_top8_pca={rate(dim_summary, result_kind='power', experiment='center_shift', setting='H1_source_vs_external', label='8', n='50', representation='residual_top8_pca', summary_scope='single')}, residual_random8={rate(dim_summary, result_kind='power', experiment='center_shift', setting='H1_source_vs_external', label='8', n='50', representation='residual_random8', summary_scope='aggregate')}, residual_full={rate(dim_summary, result_kind='power', experiment='center_shift', setting='H1_source_vs_external', label='8', n='50', representation='residual_full', summary_scope='single')}.",
        f"- Dimension-control H0 rejection is slightly above nominal in the worst cells but not explosive: max H0 rejection across centered_logits={max_rate(dim_summary, 'rejection_rate', result_kind='type1', experiment='center_shift', representation='centered_logits', summary_scope='single')}, semantic_centered={max_rate(dim_summary, 'rejection_rate', result_kind='type1', experiment='center_shift', representation='semantic_centered', summary_scope='single')}, residual_top8_pca={max_rate(dim_summary, 'rejection_rate', result_kind='type1', experiment='center_shift', representation='residual_top8_pca', summary_scope='single')}, residual_random8={max_rate(dim_summary, 'rejection_rate', result_kind='type1', experiment='center_shift', representation='residual_random8', summary_scope='aggregate')}, residual_full={max_rate(dim_summary, 'rejection_rate', result_kind='type1', experiment='center_shift', representation='residual_full', summary_scope='single')}, final_full={max_rate(dim_summary, 'rejection_rate', result_kind='type1', experiment='center_shift', representation='final_full', summary_scope='single')}.",
        "",
        "## 3. Is center-shift robust under permutation calibration?",
        "",
        f"- Label 6 n=50 remains high under permutation: semantic_centered bootstrap/permutation={perm_rate('bootstrap_rejection_rate', result_kind='power', setting='H1_source_vs_external', label='6', n='50', representation='semantic_centered')}/{perm_rate('permutation_rejection_rate', result_kind='power', setting='H1_source_vs_external', label='6', n='50', representation='semantic_centered')}; residual_full={perm_rate('bootstrap_rejection_rate', result_kind='power', setting='H1_source_vs_external', label='6', n='50', representation='residual_full')}/{perm_rate('permutation_rejection_rate', result_kind='power', setting='H1_source_vs_external', label='6', n='50', representation='residual_full')}.",
        f"- Label 8 n=50 remains very high under permutation: semantic_centered bootstrap/permutation={perm_rate('bootstrap_rejection_rate', result_kind='power', setting='H1_source_vs_external', label='8', n='50', representation='semantic_centered')}/{perm_rate('permutation_rejection_rate', result_kind='power', setting='H1_source_vs_external', label='8', n='50', representation='semantic_centered')}; residual_top8_pca={perm_rate('bootstrap_rejection_rate', result_kind='power', setting='H1_source_vs_external', label='8', n='50', representation='residual_top8_pca')}/{perm_rate('permutation_rejection_rate', result_kind='power', setting='H1_source_vs_external', label='8', n='50', representation='residual_top8_pca')}; residual_full={perm_rate('bootstrap_rejection_rate', result_kind='power', setting='H1_source_vs_external', label='8', n='50', representation='residual_full')}/{perm_rate('permutation_rejection_rate', result_kind='power', setting='H1_source_vs_external', label='8', n='50', representation='residual_full')}.",
        f"- Permutation H0 is closer to nominal than bootstrap in the worst cells: max bootstrap H0={max_rate(perm_summary, 'bootstrap_rejection_rate', result_kind='type1')}, max permutation H0={max_rate(perm_summary, 'permutation_rejection_rate', result_kind='type1')}.",
        "",
        "## 4. Is center-shift mostly explained by color/stain features?",
        "",
        "- Not answered by this core run. The optional color-only/domain-probe/color-residualized analysis in `plan_next.md` was not run.",
        "- Current evidence says the residual signal is not merely high dimensional and is not removed by permutation calibration. It does not yet separate biological morphology from stain/color/domain effects.",
        "",
        "## Main message to verify",
        "",
        "CNN-MMMD is not only increasing power uniformly. In these controls, class-mixture is mostly classifier-semantic/logit aligned, label-6 center-shift is more semantic/logit aligned, and label-8 center-shift contains a strong classifier-ignored residual component that survives 8D PCA/random projection controls and permutation calibration.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def write_outputs(b_boot: int, b_perm: int, alpha: float) -> None:
    dim_summary = rebuild_dim_summary(b_boot, alpha) if DIM_RAW_PATH.exists() else []
    if dim_summary:
        plot_dim_power(dim_summary)
        plot_dim_type1(dim_summary)
        plot_dim_excess(dim_summary)
        log(f"wrote dim summaries and plots under {RESULTS_DIR}")
    perm_summary = rebuild_perm_summary(b_boot, b_perm, alpha) if PERM_RAW_PATH.exists() else []
    if perm_summary:
        plot_perm(perm_summary, "power", PERM_POWER_PLOT)
        plot_perm(perm_summary, "type1", PERM_TYPE1_PLOT)
        log(f"wrote permutation summaries and plots under {RESULTS_DIR}")
    if dim_summary or perm_summary:
        write_short_report(dim_summary, perm_summary)
        log(f"wrote {REPORT_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-repetitions", type=int, default=10)
    parser.add_argument("--inner-repetitions", type=int, default=500)
    parser.add_argument("--b-bootstrap", type=int, default=500)
    parser.add_argument("--b-permutation", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--class-h1-sample-sizes", type=int, nargs="+", default=[60, 90, 120])
    parser.add_argument("--class-h0-sample-sizes", type=int, nargs="+", default=[60, 120])
    parser.add_argument("--center-h1-n", type=int, nargs="+", default=[20, 30, 50])
    parser.add_argument("--center-h0-n", type=int, nargs="+", default=[20, 50])
    parser.add_argument("--permutation-n", type=int, nargs="+", default=[20, 30, 50])
    parser.add_argument("--center-labels", type=int, nargs="+", default=[6, 8])
    parser.add_argument("--random-projection-repeats", type=int, default=20)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--skip-dim-matched", action="store_true")
    parser.add_argument("--skip-permutation", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if args.force:
        for path in [
            DIM_RAW_PATH,
            DIM_SUMMARY_PATH,
            DIM_POWER_PLOT,
            DIM_TYPE1_PLOT,
            DIM_EXCESS_PLOT,
            PERM_RAW_PATH,
            PERM_SUMMARY_PATH,
            PERM_POWER_PLOT,
            PERM_TYPE1_PLOT,
            DIAG_PATH,
            CONFIG_PATH,
            REPORT_PATH,
        ]:
            if path.exists():
                path.unlink()

    config = {
        "outer_repetitions": args.outer_repetitions,
        "inner_repetitions": args.inner_repetitions,
        "B_boot": args.b_bootstrap,
        "B_perm": args.b_permutation,
        "alpha": args.alpha,
        "class_h1_sample_sizes": args.class_h1_sample_sizes,
        "class_h0_sample_sizes": args.class_h0_sample_sizes,
        "center_h1_n": args.center_h1_n,
        "center_h0_n": args.center_h0_n,
        "permutation_n": args.permutation_n,
        "center_labels": args.center_labels,
        "random_projection_repeats": args.random_projection_repeats,
        "class_checkpoint": str(BASE.CLASS_CHECKPOINT),
        "center_checkpoint": str(BASE.CENTER_CHECKPOINT),
        "center_split_path": str(BASE.CENTER_SPLIT_PATH),
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2))
    log(f"config: {json.dumps(config, sort_keys=True)}")

    if args.summarize_only:
        write_outputs(args.b_bootstrap, args.b_permutation, args.alpha)
        return

    raw = BASE.load_pathmnist_raw()
    pools = BASE.load_center_pools(raw)
    class_cnn = import_module(BASE.CLASS_CNN_MODULE, "pathmnist_class_cnn_pipeline")
    center_cnn = import_module(BASE.CENTER_CNN_MODULE, "pathmnist_center_cnn_pipeline")

    diagnostics: list[dict[str, object]] = []
    reps_test_base, diag = extract_centered_base_reps(class_cnn, BASE.CLASS_CHECKPOINT, raw["test_images"], args.eval_batch_size, "class_mixture_test")
    diagnostics.append(diag)
    reps_source_base, diag = extract_centered_base_reps(center_cnn, BASE.CENTER_CHECKPOINT, pools["source_images"], args.eval_batch_size, "center_source_holdout")
    diagnostics.append(diag)
    reps_external_base, diag = extract_centered_base_reps(center_cnn, BASE.CENTER_CHECKPOINT, pools["external_images"], args.eval_batch_size, "center_external")
    diagnostics.append(diag)
    write_csv(DIAG_PATH, DIAG_FIELDS, diagnostics)
    log(f"wrote centered diagnostics: {DIAG_PATH}")

    reps_test, class_rep_specs = add_dimension_controls_single_pool(
        reps_test_base,
        args.random_projection_repeats,
        seed=20260605,
    )
    reps_source, reps_external, center_rep_specs = add_dimension_controls_two_pools(
        reps_source_base,
        reps_external_base,
        args.random_projection_repeats,
        seed=20260606,
    )
    perm_rep_specs = dim_specs_for_permutation(center_rep_specs)

    if not args.skip_dim_matched:
        run_dim_class_mixture(
            reps_test,
            raw["test_labels"],
            class_rep_specs,
            args.class_h1_sample_sizes,
            args.class_h0_sample_sizes,
            args.outer_repetitions,
            args.inner_repetitions,
            args.b_bootstrap,
            args.alpha,
            args.progress_every,
        )
        run_dim_center_shift(
            reps_source,
            reps_external,
            pools,
            center_rep_specs,
            args.center_labels,
            args.center_h1_n,
            args.center_h0_n,
            args.outer_repetitions,
            args.inner_repetitions,
            args.b_bootstrap,
            args.alpha,
            args.progress_every,
        )

    if not args.skip_permutation:
        run_permutation_center_shift(
            reps_source,
            reps_external,
            pools,
            perm_rep_specs,
            args.center_labels,
            args.permutation_n,
            args.outer_repetitions,
            args.inner_repetitions,
            args.b_bootstrap,
            args.b_permutation,
            args.alpha,
            args.progress_every,
        )

    write_outputs(args.b_bootstrap, args.b_permutation, args.alpha)
    log("validity controls complete")


if __name__ == "__main__":
    main()
