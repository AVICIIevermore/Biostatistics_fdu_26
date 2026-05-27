#!/usr/bin/env python3
"""Formula and timing audit for Phase 4 MMMD implementation."""
from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
POWER_PATH = EXPERIMENT_ROOT / "python" / "pathmnist_power.py"
RESULTS_DIR = EXPERIMENT_ROOT / "Results"


def import_power():
    spec = importlib.util.spec_from_file_location("pathmnist_power", POWER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def direct_pairwise_sq_dists(x, y):
    out = np.empty((x.shape[0], y.shape[0]), dtype=float)
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            diff = x[i] - y[j]
            out[i, j] = float(np.dot(diff, diff))
    return out


def direct_mmd_vec(kxx_list, kyy_list, kxy_list):
    n = kxx_list[0].shape[0]
    vals = []
    for kxx, kyy, kxy in zip(kxx_list, kyy_list, kxy_list):
        terms = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    terms.append(kxx[i, j] + kyy[i, j] - 2.0 * kxy[i, j])
        vals.append(n * float(np.mean(terms)))
    return np.array(vals)


def direct_sigma(kxx_list):
    n = kxx_list[0].shape[0]
    c = np.eye(n) - np.ones((n, n)) / n
    centered = [c @ kxx @ c for kxx in kxx_list]
    out = np.empty((len(centered), len(centered)), dtype=float)
    for i in range(len(centered)):
        for j in range(len(centered)):
            total = 0.0
            prod = centered[i] @ centered[j]
            for d in range(n):
                total += prod[d, d]
            out[i, j] = (8.0 / (n ** 2)) * total
    return out


def main():
    power = import_power()
    images, labels = power.load_test_data()
    config = json.loads((RESULTS_DIR / "power_config.json").read_text())
    n = 150
    rng = np.random.default_rng(int(config["seed"]) + 1000 * n + 1)
    x_idx, y_idx = power.sample_balanced_h1(labels, n, rng)
    raw = images.reshape(images.shape[0], -1).astype(np.float32) / 255.0
    x = raw[x_idx]
    y = raw[y_idx]

    t0 = time.perf_counter()
    d_fast = power.pairwise_sq_dists(x, y)
    t_pairwise_fast = time.perf_counter() - t0
    # Use a tiny slice for direct nested-loop equality check to keep audit cheap.
    d_direct = direct_pairwise_sq_dists(x[:8], y[:8])
    d_fast_small = power.pairwise_sq_dists(x[:8], y[:8])

    kxx, kyy, kxy = power.kernel_mats_for_features(x[:30], y[:30], "gaussian5")
    helper = power.run_mmmd_from_kernel_lists(kxx, kyy, kxy, b_boot=20, alpha=0.05, rng=np.random.default_rng(123))
    direct_mmd = direct_mmd_vec(kxx, kyy, kxy)
    helper_mmd_manual = []
    offdiag = ~np.eye(30, dtype=bool)
    for a, b, c in zip(kxx, kyy, kxy):
        helper_mmd_manual.append(30 * np.mean(a[offdiag] + b[offdiag] - 2.0 * c[offdiag]))
    helper_mmd_manual = np.array(helper_mmd_manual)
    sigma_direct = direct_sigma(kxx)
    cmat = np.eye(30) - np.ones((30, 30)) / 30
    centered = [cmat @ mat @ cmat for mat in kxx]
    sigma_fast = np.array([[(8.0 / (30 ** 2)) * np.trace(centered[i] @ centered[j]) for j in range(len(centered))] for i in range(len(centered))])

    # Full one-repetition timing at largest n and all four methods.
    cnn = power.load_cnn_module()
    t1 = time.perf_counter()
    emb_all = cnn.extract_embeddings(power.CHECKPOINT_PATH, images[np.concatenate([x_idx, y_idx])], batch_size=512, layers=["layer1_gap", "layer2_gap", "final_fc128"])
    t_embed = time.perf_counter() - t1
    emb_x = {name: value[:n] for name, value in emb_all.items()}
    emb_y = {name: value[n:] for name, value in emb_all.items()}
    t2 = time.perf_counter()
    _ = power.run_single_rep_methods(x, y, emb_x, emb_y, b_boot=500, alpha=0.05, seed=999)
    t_methods = time.perf_counter() - t2

    report = {
        "pairwise_fast_n150_sec": t_pairwise_fast,
        "embedding_extract_300_images_sec": t_embed,
        "four_methods_n150_B500_sec": t_methods,
        "pairwise_direct_vs_fast_max_abs_diff_small": float(np.max(np.abs(d_direct - d_fast_small))),
        "mmd_direct_vs_vectorized_max_abs_diff": float(np.max(np.abs(direct_mmd - helper_mmd_manual))),
        "sigma_direct_vs_trace_max_abs_diff": float(np.max(np.abs(sigma_direct - sigma_fast))),
        "largest_n": n,
        "B_boot": 500,
        "helper_stat_example": helper["stat"],
        "helper_cutoff_example": helper["cutoff"],
    }
    (RESULTS_DIR / "phase4_formula_timing_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
