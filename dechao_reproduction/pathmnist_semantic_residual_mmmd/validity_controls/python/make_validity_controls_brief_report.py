from __future__ import annotations

import csv
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


ROOT = Path("/home/dechao/kernel_two-sample")
CONTROL_ROOT = ROOT / "dechao_reproduction" / "pathmnist_semantic_residual_mmmd" / "validity_controls"
RESULTS = CONTROL_ROOT / "Results"
OUT = ROOT / "pathmnist_validity_controls_brief_report_20260605.pdf"

DIM_SUMMARY = RESULTS / "dim_matched_residual_mmmd_summary.csv"
PERM_SUMMARY = RESULTS / "permutation_calibration_center_shift_summary.csv"
DIAG_CSV = RESULTS / "semantic_residual_centered_diagnostics.csv"
CONFIG_JSON = RESULTS / "validity_controls_config.json"

PAGE = (11, 8.5)
REPS = [
    "centered_logits",
    "semantic_centered",
    "residual_top8_pca",
    "residual_random8",
    "residual_full",
    "final_full",
]
REP_LABELS = {
    "centered_logits": "centered logits",
    "semantic_centered": "semantic centered",
    "residual_top8_pca": "residual top-8 PCA",
    "residual_random8": "residual random-8",
    "residual_full": "residual full",
    "final_full": "final full",
}


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": "#333333",
    }
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def fmt(value: str | float | None, digits: int = 3) -> str:
    if value in (None, ""):
        return "NA"
    return f"{float(value):.{digits}f}"


def find(rows: list[dict[str, str]], **criteria: object) -> dict[str, str] | None:
    for row in rows:
        if all(str(row.get(k, "")) == str(v) for k, v in criteria.items()):
            return row
    return None


def rate(rows: list[dict[str, str]], field: str = "rejection_rate", **criteria: object) -> str:
    row = find(rows, **criteria)
    return "NA" if row is None else fmt(row[field])


def max_rate(rows: list[dict[str, str]], field: str, **criteria: object) -> str:
    vals = [
        float(row[field])
        for row in rows
        if all(str(row.get(k, "")) == str(v) for k, v in criteria.items())
    ]
    return "NA" if not vals else fmt(max(vals))


def make_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    header = "  ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    sep = "  ".join("-" * widths[i] for i in range(len(headers)))
    body = ["  ".join(row[i].ljust(widths[i]) for i in range(len(row))) for row in rows]
    return "\n".join([header, sep] + body)


def add_text_page(pdf: PdfPages, title: str, blocks: list[tuple[str, str | list[str]]]) -> None:
    fig = plt.figure(figsize=PAGE)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.055, 0.94, title, fontsize=21, fontweight="bold", va="top", color="#111111")
    y = 0.88
    for kind, content in blocks:
        if kind == "h":
            y -= 0.01
            ax.text(0.055, y, str(content), fontsize=13, fontweight="bold", va="top", color="#111111")
            y -= 0.045
        elif kind == "p":
            for line in textwrap.wrap(str(content), width=118):
                ax.text(0.06, y, line, fontsize=10.5, va="top", color="#202020")
                y -= 0.028
            y -= 0.012
        elif kind == "b":
            for item in content:
                for idx, line in enumerate(textwrap.wrap(item, width=112)):
                    prefix = "- " if idx == 0 else "  "
                    ax.text(0.075, y, prefix + line, fontsize=10.2, va="top", color="#202020")
                    y -= 0.027
            y -= 0.010
        elif kind == "code":
            ax.text(0.065, y, str(content), fontsize=8.8, va="top", family="DejaVu Sans Mono", color="#111111")
            y -= 0.023 * (str(content).count("\n") + 1) + 0.025
    ax.text(0.055, 0.035, f"Generated from {CONTROL_ROOT.relative_to(ROOT)}. No experiments rerun.", fontsize=8.2, color="#555555")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_image_page(pdf: PdfPages, title: str, image_path: Path, caption: str) -> None:
    fig = plt.figure(figsize=PAGE)
    fig.patch.set_facecolor("white")
    bg = fig.add_axes([0, 0, 1, 1])
    bg.axis("off")
    bg.text(0.055, 0.95, title, fontsize=20, fontweight="bold", va="top", color="#111111")
    bg.text(0.055, 0.895, caption, fontsize=9.5, va="top", color="#333333")
    ax = fig.add_axes([0.05, 0.07, 0.90, 0.78])
    ax.imshow(Image.open(image_path))
    ax.axis("off")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_tables(dim_rows: list[dict[str, str]], perm_rows: list[dict[str, str]], diag_rows: list[dict[str, str]]) -> dict[str, str]:
    class_rows = []
    for rep in REPS:
        class_rows.append(
            [
                REP_LABELS[rep],
                rate(dim_rows, result_kind="power", experiment="class_mixture", setting="H1_mix635_vs_mix835", n="120", representation=rep, summary_scope="aggregate" if rep == "residual_random8" else "single"),
            ]
        )

    center_rows = []
    for label in ["6", "8"]:
        for rep in ["semantic_centered", "residual_top8_pca", "residual_random8", "residual_full"]:
            center_rows.append(
                [
                    f"label {label}",
                    REP_LABELS[rep],
                    rate(dim_rows, result_kind="power", experiment="center_shift", setting="H1_source_vs_external", label=label, n="50", representation=rep, summary_scope="aggregate" if rep == "residual_random8" else "single"),
                ]
            )

    perm_rows_out = []
    for label in ["6", "8"]:
        for rep in ["semantic_centered", "residual_top8_pca", "residual_full"]:
            row = find(perm_rows, result_kind="power", setting="H1_source_vs_external", label=label, n="50", representation=rep)
            perm_rows_out.append(
                [
                    f"label {label}",
                    REP_LABELS[rep],
                    "NA" if row is None else fmt(row["bootstrap_rejection_rate"]),
                    "NA" if row is None else fmt(row["permutation_rejection_rate"]),
                ]
            )

    diag_out = []
    for row in diag_rows:
        diag_out.append(
            [
                row["pool_tag"],
                row["rank_W"],
                row["rank_Wc"],
                fmt(row["mean_semantic_centered_norm_fraction"]),
                fmt(row["mean_residual_full_norm_fraction"]),
            ]
        )

    h0_rows = []
    for rep in REPS:
        h0_rows.append(
            [
                REP_LABELS[rep],
                max_rate(dim_rows, "rejection_rate", result_kind="type1", experiment="center_shift", representation=rep, summary_scope="aggregate" if rep == "residual_random8" else "single"),
            ]
        )

    perm_h0 = make_table(
        ["calibration", "max H0"],
        [
            ["bootstrap", max_rate(perm_rows, "bootstrap_rejection_rate", result_kind="type1")],
            ["permutation", max_rate(perm_rows, "permutation_rejection_rate", result_kind="type1")],
        ],
    )

    return {
        "class": make_table(["class-mixture n=120", "power"], class_rows),
        "center": make_table(["center H1 n=50", "representation", "power"], center_rows),
        "perm": make_table(["center H1 n=50", "representation", "bootstrap", "permutation"], perm_rows_out),
        "diag": make_table(["pool", "rank W", "rank Wc", "sem frac", "res frac"], diag_out),
        "h0": make_table(["center H0 representation", "max rejection"], h0_rows),
        "perm_h0": perm_h0,
    }


def main() -> None:
    dim_rows = read_csv(DIM_SUMMARY)
    perm_rows = read_csv(PERM_SUMMARY)
    diag_rows = read_csv(DIAG_CSV)
    config = json.loads(CONFIG_JSON.read_text())
    tables = build_tables(dim_rows, perm_rows, diag_rows)

    run_size = (
        f"outer={config['outer_repetitions']}, inner={config['inner_repetitions']}, "
        f"B_boot={config['B_boot']}, B_perm={config['B_perm']}; "
        "200 tests/cell for non-random representations, 4000 tests/cell for residual_random8 aggregate."
    )

    with PdfPages(OUT) as pdf:
        meta = pdf.infodict()
        meta["Title"] = "PathMNIST Validity Controls Brief Report"
        meta["Author"] = "kernel_two-sample project"
        meta["Subject"] = "Brief report for semantic-residual validity controls"

        add_text_page(
            pdf,
            "PathMNIST Validity Controls: Brief Report",
            [
                ("p", "This PDF reports only the latest validity-control round for the semantic-vs-residual CNN-MMMD experiment. Existing CNN checkpoints were reused; no retraining was done."),
                ("h", "Run Size"),
                ("p", run_size),
                ("h", "Core Conclusion"),
                (
                    "b",
                    [
                        "Centered-W decomposition behaves as intended: rank(W)=9 and rank(Wc)=8 in all pools.",
                        "Class-mixture is mostly classifier-semantic/logit aligned, although residual features still carry signal.",
                        "Center-shift label 6 is more semantic/logit aligned.",
                        "Center-shift label 8 has a strong classifier-ignored residual component that survives 8D PCA, random 8D projections, and permutation calibration.",
                        "Color/stain dominance is not answered in this core run; the optional color covariate analysis was not run.",
                    ],
                ),
            ],
        )

        add_text_page(
            pdf,
            "Key Numeric Results",
            [
                ("h", "Centered-W diagnostics"),
                ("code", tables["diag"]),
                ("h", "Class-mixture H1: mix635_vs_mix835"),
                ("code", tables["class"]),
                ("h", "Center-shift H1 at n=50"),
                ("code", tables["center"]),
                ("h", "Permutation calibration at n=50"),
                ("code", tables["perm"]),
                ("h", "H0 checks"),
                ("code", tables["h0"] + "\n\n" + tables["perm_h0"]),
            ],
        )

        add_image_page(
            pdf,
            "Dimension-matched Power",
            RESULTS / "dim_matched_residual_power.png",
            "Compare semantic/logit coordinates with residual_full, residual_top8_pca, and residual_random8.",
        )
        add_image_page(
            pdf,
            "Dimension-matched Type-I Error",
            RESULTS / "dim_matched_residual_type1.png",
            "Matched-null rejection rates. Worst cells are mildly above nominal alpha=0.05, but not explosive.",
        )
        add_image_page(
            pdf,
            "Permutation vs Bootstrap",
            RESULTS / "permutation_vs_bootstrap_power.png",
            "Center-shift H1 remains strong under permutation calibration, especially label 8 residual features.",
        )

        add_text_page(
            pdf,
            "Files Used",
            [
                ("p", "The report was generated from completed summary outputs only."),
                (
                    "code",
                    "\n".join(
                        [
                            str(DIM_SUMMARY.relative_to(ROOT)),
                            str(PERM_SUMMARY.relative_to(ROOT)),
                            str(DIAG_CSV.relative_to(ROOT)),
                            str((RESULTS / "dim_matched_residual_power.png").relative_to(ROOT)),
                            str((RESULTS / "dim_matched_residual_type1.png").relative_to(ROOT)),
                            str((RESULTS / "permutation_vs_bootstrap_power.png").relative_to(ROOT)),
                        ]
                    ),
                ),
            ],
        )

    print(OUT)
    print(f"size_mb={OUT.stat().st_size / 1024 / 1024:.2f}")


if __name__ == "__main__":
    main()
