#!/usr/bin/env python3
"""Audit Phase 4 PathMNIST power run against the agreed plan."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(__file__).resolve().parents[3]
POWER_PATH = EXPERIMENT_ROOT / "python" / "pathmnist_power.py"
RESULTS_DIR = EXPERIMENT_ROOT / "Results"


def import_power():
    spec = importlib.util.spec_from_file_location("pathmnist_power", POWER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    power = import_power()
    config = json.loads((RESULTS_DIR / "power_config.json").read_text())
    results = list(csv.DictReader((RESULTS_DIR / "sample_size_power_results.csv").open()))
    images, labels = power.load_test_data()

    expected_methods = [
        "raw_pixel_gaussian5",
        "cnn_final_fc128_gaussian5",
        "cnn_multilayer_single_gaussian",
        "cnn_multilayer_gaussian15",
    ]
    expected_kernel_counts = {
        "raw_pixel_gaussian5": 5,
        "cnn_final_fc128_gaussian5": 5,
        "cnn_multilayer_single_gaussian": 3,
        "cnn_multilayer_gaussian15": 15,
    }

    checks: list[dict[str, object]] = []
    def add_check(name: str, passed: bool, detail: object) -> None:
        checks.append({"check": name, "passed": int(bool(passed)), "detail": detail})

    add_check("scenario", config.get("scenario") == "mix635_vs_mix835", config.get("scenario"))
    add_check("sample_sizes", config.get("sample_sizes") == [30, 60, 90, 120, 150], config.get("sample_sizes"))
    add_check("outer_repetitions", config.get("outer_repetitions") == 10, config.get("outer_repetitions"))
    add_check("B_boot", config.get("B_boot") == 500, config.get("B_boot"))
    add_check("alpha", abs(float(config.get("alpha")) - 0.05) < 1e-12, config.get("alpha"))
    add_check("methods", config.get("methods") == expected_methods, config.get("methods"))
    add_check("row_count", len(results) == 5 * 10 * 4, len(results))

    row_keys = {(int(r["sample_size"]), int(r["outer_iter"]), r["method"]) for r in results}
    missing = []
    for n in [30, 60, 90, 120, 150]:
        for outer in range(1, 11):
            for method in expected_methods:
                if (n, outer, method) not in row_keys:
                    missing.append((n, outer, method))
    add_check("all_sample_outer_method_rows_present", not missing, missing[:5])

    bad_kernel_rows = [
        (r["sample_size"], r["outer_iter"], r["method"], r["kernel_count"])
        for r in results
        if int(r["kernel_count"]) != expected_kernel_counts[r["method"]]
    ]
    add_check("kernel_counts", not bad_kernel_rows, bad_kernel_rows[:5])

    nonfinite = []
    for r in results:
        for col in ["stat", "cutoff", "lambda", "cond_sigma_reg"]:
            if not np.isfinite(float(r[col])):
                nonfinite.append((r["sample_size"], r["outer_iter"], r["method"], col, r[col]))
    add_check("finite_numeric_outputs", not nonfinite, nonfinite[:5])

    audit_rows: list[dict[str, object]] = []
    overlap_failures = []
    balance_failures = []
    for n in [30, 60, 90, 120, 150]:
        per_class = n // 3
        for outer in range(1, 11):
            rng = np.random.default_rng(int(config["seed"]) + 1000 * n + outer)
            x_idx, y_idx = power.sample_balanced_h1(labels, n, rng)
            x_labels = labels[x_idx]
            y_labels = labels[y_idx]
            row = {
                "sample_size": n,
                "outer_iter": outer,
                "x_n": len(x_idx),
                "y_n": len(y_idx),
                "xy_overlap": len(set(x_idx.tolist()).intersection(y_idx.tolist())),
                "x_class_3": int(np.sum(x_labels == 3)),
                "x_class_5": int(np.sum(x_labels == 5)),
                "x_class_6": int(np.sum(x_labels == 6)),
                "x_class_8": int(np.sum(x_labels == 8)),
                "y_class_3": int(np.sum(y_labels == 3)),
                "y_class_5": int(np.sum(y_labels == 5)),
                "y_class_6": int(np.sum(y_labels == 6)),
                "y_class_8": int(np.sum(y_labels == 8)),
            }
            audit_rows.append(row)
            if row["xy_overlap"] != 0:
                overlap_failures.append((n, outer, row["xy_overlap"]))
            expected = {
                "x_class_3": per_class, "x_class_5": per_class, "x_class_6": per_class, "x_class_8": 0,
                "y_class_3": per_class, "y_class_5": per_class, "y_class_6": 0, "y_class_8": per_class,
            }
            for key, val in expected.items():
                if row[key] != val:
                    balance_failures.append((n, outer, key, row[key], val))
    write_csv(RESULTS_DIR / "sample_audit.csv", list(audit_rows[0].keys()), audit_rows)
    add_check("sampling_xy_disjoint", not overlap_failures, overlap_failures[:5])
    add_check("sampling_class_balanced", not balance_failures, balance_failures[:5])

    # Check embedding dimensions on one deterministic sample without recomputing full experiment.
    cnn = power.load_cnn_module()
    x_idx, y_idx = power.sample_balanced_h1(labels, 30, np.random.default_rng(int(config["seed"]) + 1000 * 30 + 1))
    emb = cnn.extract_embeddings(power.CHECKPOINT_PATH, images[np.concatenate([x_idx, y_idx])], batch_size=512, layers=["layer1_gap", "layer2_gap", "final_fc128"])
    emb_shapes = {k: tuple(v.shape) for k, v in emb.items()}
    add_check("embedding_shapes", emb_shapes == {"layer1_gap": (60, 32), "layer2_gap": (60, 64), "final_fc128": (60, 128)}, emb_shapes)

    # Timing scale sanity: estimate primitive matrix counts for n=150.
    primitive_counts = {
        "draws": 5 * 10,
        "method_results": 5 * 10 * 4,
        "max_n": 150,
        "B_boot": config["B_boot"],
        "largest_kernel_count": 15,
    }
    add_check("computation_scale_recorded", True, primitive_counts)

    write_csv(RESULTS_DIR / "phase4_audit_checks.csv", ["check", "passed", "detail"], checks)
    passed = all(bool(row["passed"]) for row in checks)
    print(json.dumps({"passed": passed, "checks": checks}, indent=2, default=str))


if __name__ == "__main__":
    main()
