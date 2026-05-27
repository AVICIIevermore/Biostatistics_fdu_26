#!/usr/bin/env python3
"""PathMNIST-28 matched-null Type-I check for MMMD."""
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
POWER_MODULE_PATH = DATASET_ROOT / "experiments" / "03_sample_size_power_mix635_vs_mix835" / "python" / "pathmnist_power.py"
RESULTS_DIR = EXPERIMENT_ROOT / "Results"
LOG_DIR = EXPERIMENT_ROOT / "logs"


def import_power_module():
    spec = importlib.util.spec_from_file_location("pathmnist_power", POWER_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

power = import_power_module()
METHODS = power.METHODS
RESULT_FIELDS = power.RESULT_FIELDS
SUMMARY_FIELDS = [
    "scenario", "sample_size", "method", "outer_repetitions", "inner_repetitions", "independent_tests",
    "type1_error", "mean_stat", "mean_cutoff", "mean_cond_sigma_reg",
]


def log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    with (LOG_DIR / "type1_run.log").open("a") as f:
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


def sample_balanced_h0(labels: np.ndarray, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if n % 3 != 0:
        raise ValueError("sample size must be divisible by 3")
    per_class = n // 3
    x_parts, y_parts = [], []
    for label in [6, 3, 5]:
        pool = np.where(labels == label)[0]
        needed = 2 * per_class
        if len(pool) < needed:
            raise ValueError(f"class {label} has {len(pool)} test images, need {needed}")
        chosen = rng.choice(pool, needed, replace=False)
        x_parts.append(chosen[:per_class])
        y_parts.append(chosen[per_class:])
    x_idx = np.concatenate(x_parts)
    y_idx = np.concatenate(y_parts)
    rng.shuffle(x_idx)
    rng.shuffle(y_idx)
    return x_idx, y_idx


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
                "scenario": "mix635_vs_mix635_null",
                "sample_size": sample_size,
                "method": method,
                "outer_repetitions": outer_count,
                "inner_repetitions": inner_count,
                "independent_tests": len(rows),
                "type1_error": float(np.mean([int(row["reject"]) for row in rows])),
                "mean_stat": float(np.mean([float(row["stat"]) for row in rows])),
                "mean_cutoff": float(np.mean([float(row["cutoff"]) for row in rows])),
                "mean_cond_sigma_reg": float(np.mean([float(row["cond_sigma_reg"]) for row in rows])),
            })
    write_csv(RESULTS_DIR / "type1_summary.csv", SUMMARY_FIELDS, summary_rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in METHODS:
        rows = [row for row in summary_rows if row["method"] == method]
        if rows:
            ax.plot([row["sample_size"] for row in rows], [row["type1_error"] for row in rows], marker="o", label=method)
    ax.axhline(0.05, color="black", linestyle="--", linewidth=1, label="alpha=0.05")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Group size n")
    ax.set_ylabel("Empirical Type-I error")
    ax.set_title("PathMNIST Type-I Check: mix635 null")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "type1_error_check.png", dpi=160)
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
    ax.set_title("PathMNIST Type-I Covariance Diagnostics")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "sigma_condition_number.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[60, 120])
    parser.add_argument("--outer-repetitions", type=int, default=10)
    parser.add_argument("--inner-repetitions", type=int, default=500)
    parser.add_argument("--b-bootstrap", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "type1_results.csv"
    diag_path = RESULTS_DIR / "sigma_diagnostics.csv"
    if args.force:
        for path in [results_path, diag_path, RESULTS_DIR / "type1_summary.csv", RESULTS_DIR / "type1_error_check.png", RESULTS_DIR / "sigma_condition_number.png"]:
            if path.exists():
                path.unlink()

    config = {
        "scenario": "mix635_vs_mix635_null",
        "sample_sizes": args.sample_sizes,
        "outer_repetitions": args.outer_repetitions,
        "inner_repetitions": args.inner_repetitions,
        "B_boot": args.b_bootstrap,
        "alpha": args.alpha,
        "seed": args.seed,
        "methods": METHODS,
        "checkpoint_path": str(power.CHECKPOINT_PATH),
    }
    (RESULTS_DIR / "type1_config.json").write_text(json.dumps(config, indent=2))
    log(f"type1 config: {json.dumps(config, sort_keys=True)}")

    images, labels = power.load_test_data()
    flat_pixels = images.reshape(images.shape[0], -1).astype(np.float32) / 255.0
    cnn = power.load_cnn_module()
    all_embeddings = power.extract_test_embeddings(cnn, images, args.eval_batch_size)
    done = completed_outer_keys(results_path, args.inner_repetitions)
    log(f"completed sample_size/outer_iter keys at start: {len(done)}")

    for sample_size in args.sample_sizes:
        for outer_iter in range(1, args.outer_repetitions + 1):
            key = (sample_size, outer_iter)
            if key in done:
                log(f"skip completed sample_size={sample_size} outer_iter={outer_iter}")
                continue
            outer_start = time.time()
            rows = []
            for inner_iter in range(1, args.inner_repetitions + 1):
                if inner_iter == 1 or inner_iter % 50 == 0 or inner_iter == args.inner_repetitions:
                    log(
                        f"progress sample_size={sample_size} outer_iter={outer_iter}/{args.outer_repetitions} "
                        f"inner_iter={inner_iter}/{args.inner_repetitions}"
                    )
                inner_start = time.time()
                rep_seed = args.seed + 1_000_000 * sample_size + 10_000 * outer_iter + inner_iter
                rng = np.random.default_rng(rep_seed)
                x_idx, y_idx = sample_balanced_h0(labels, sample_size, rng)
                raw_x = flat_pixels[x_idx]
                raw_y = flat_pixels[y_idx]
                emb_x = {name: value[x_idx] for name, value in all_embeddings.items()}
                emb_y = {name: value[y_idx] for name, value in all_embeddings.items()}
                method_outputs = power.run_single_rep_methods(raw_x, raw_y, emb_x, emb_y, args.b_bootstrap, args.alpha, rep_seed)

                runtime = time.time() - inner_start
                for method in METHODS:
                    out = method_outputs[method]
                    rows.append({
                        "scenario": "mix635_vs_mix635_null",
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
    log("Phase 5 Type-I run complete")


if __name__ == "__main__":
    main()
