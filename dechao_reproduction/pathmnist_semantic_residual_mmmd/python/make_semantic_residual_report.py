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
EXP = ROOT / "dechao_reproduction" / "pathmnist_semantic_residual_mmmd"
RES = EXP / "Results"
OUT = ROOT / "semantic_residual_mmmd_report_20260605.pdf"

POWER_CSV = RES / "semantic_residual_mmmd_summary.csv"
TYPE1_CSV = RES / "semantic_residual_type1_summary.csv"
DIAG_JSON = RES / "projection_diagnostics.json"

REPS = ["final_full", "logits", "final_semantic", "final_residual"]
REP_LABEL = {
    "final_full": "final_full h",
    "logits": "logits Wh+b",
    "final_semantic": "semantic proj",
    "final_residual": "residual",
}
PAGE = (11, 8.5)


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


POWER_ROWS = read_csv(POWER_CSV)
TYPE1_ROWS = read_csv(TYPE1_CSV)
DIAG = json.loads(DIAG_JSON.read_text())


def fmt(x: float | str | None, digits: int = 3) -> str:
    if x in (None, ""):
        return ""
    return f"{float(x):.{digits}f}"


def power_lookup(
    experiment: str,
    scenario: str,
    n: int,
    rep: str,
    preprocess: str = "none",
) -> float | None:
    for row in POWER_ROWS:
        if (
            row["experiment"] == experiment
            and row["scenario"] == scenario
            and row["n"] == str(n)
            and row["representation"] == rep
            and row["preprocess"] == preprocess
        ):
            return float(row["power"])
    return None


def type1_max(rep: str, preprocess: str = "none", experiment: str | None = None) -> float | None:
    values = []
    for row in TYPE1_ROWS:
        if row["representation"] != rep or row["preprocess"] != preprocess:
            continue
        if experiment is not None and row["experiment"] != experiment:
            continue
        values.append(float(row["type1_error"]))
    return max(values) if values else None


def type1_max_setting(
    rep: str,
    preprocess: str = "none",
    experiment: str | None = None,
    setting: str | None = None,
) -> float | None:
    values = []
    for row in TYPE1_ROWS:
        if row["representation"] != rep or row["preprocess"] != preprocess:
            continue
        if experiment is not None and row["experiment"] != experiment:
            continue
        if setting is not None and row["setting"] != setting:
            continue
        values.append(float(row["type1_error"]))
    return max(values) if values else None


def make_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))
    line = "  ".join(str(h).ljust(widths[idx]) for idx, h in enumerate(headers))
    sep = "  ".join("-" * widths[idx] for idx in range(len(headers)))
    body = ["  ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(row)) for row in rows]
    return "\n".join([line, sep] + body)


def add_text_page(
    pdf: PdfPages,
    title: str,
    blocks: list[tuple[str, str | list[str]]],
    footer: str | None = None,
) -> None:
    fig = plt.figure(figsize=PAGE)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.055, 0.94, title, fontsize=21, fontweight="bold", va="top", color="#111111")
    y = 0.88
    for kind, content in blocks:
        if kind == "h":
            y -= 0.012
            ax.text(0.055, y, str(content), fontsize=13, fontweight="bold", va="top", color="#111111")
            y -= 0.045
        elif kind == "p":
            for line in textwrap.wrap(str(content), width=118):
                ax.text(0.06, y, line, fontsize=10.5, va="top", color="#202020")
                y -= 0.027
            y -= 0.015
        elif kind == "b":
            for item in content:
                for idx, line in enumerate(textwrap.wrap(item, width=112)):
                    prefix = "- " if idx == 0 else "  "
                    ax.text(0.075, y, prefix + line, fontsize=10.2, va="top", color="#202020")
                    y -= 0.026
            y -= 0.012
        elif kind == "code":
            ax.text(
                0.065,
                y,
                str(content),
                fontsize=9.2,
                va="top",
                family="DejaVu Sans Mono",
                color="#111111",
            )
            y -= 0.023 * (str(content).count("\n") + 1) + 0.025
        if y < 0.08:
            if footer:
                ax.text(0.055, 0.035, footer, fontsize=8.2, color="#555555")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            fig = plt.figure(figsize=PAGE)
            fig.patch.set_facecolor("white")
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis("off")
            ax.text(
                0.055,
                0.94,
                f"{title} (continued)",
                fontsize=18,
                fontweight="bold",
                va="top",
                color="#111111",
            )
            y = 0.88
    if footer:
        ax.text(0.055, 0.035, footer, fontsize=8.2, color="#555555")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_single_image_page(pdf: PdfPages, title: str, image_path: Path, caption: str | None = None) -> None:
    fig = plt.figure(figsize=PAGE)
    fig.patch.set_facecolor("white")
    ax_title = fig.add_axes([0, 0, 1, 1])
    ax_title.axis("off")
    ax_title.text(0.055, 0.95, title, fontsize=20, fontweight="bold", va="top", color="#111111")
    if caption:
        ax_title.text(0.055, 0.895, caption, fontsize=9.7, va="top", color="#333333")
    ax = fig.add_axes([0.05, 0.07, 0.90, 0.78])
    ax.imshow(Image.open(image_path))
    ax.axis("off")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_two_image_page(
    pdf: PdfPages,
    title: str,
    image1: Path,
    image2: Path,
    cap1: str,
    cap2: str,
    mode: str = "vertical",
) -> None:
    fig = plt.figure(figsize=PAGE)
    fig.patch.set_facecolor("white")
    ax_title = fig.add_axes([0, 0, 1, 1])
    ax_title.axis("off")
    ax_title.text(0.055, 0.95, title, fontsize=20, fontweight="bold", va="top", color="#111111")
    if mode == "horizontal":
        axes = [fig.add_axes([0.055, 0.12, 0.42, 0.74]), fig.add_axes([0.525, 0.12, 0.42, 0.74])]
        captions = [(0.055, 0.08, cap1), (0.525, 0.08, cap2)]
    else:
        axes = [fig.add_axes([0.055, 0.50, 0.89, 0.34]), fig.add_axes([0.055, 0.10, 0.89, 0.34])]
        captions = [(0.055, 0.465, cap1), (0.055, 0.065, cap2)]
    for ax, image_path in zip(axes, [image1, image2], strict=True):
        ax.imshow(Image.open(image_path))
        ax.axis("off")
    for x, y, caption in captions:
        ax_title.text(x, y, caption, fontsize=9.2, va="top", color="#333333")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_tables() -> dict[str, str]:
    class_rows = []
    for rep in REPS:
        class_rows.append(
            [
                REP_LABEL[rep],
                fmt(power_lookup("class_mixture", "mix635_vs_mix835", 60, rep)),
                fmt(power_lookup("class_mixture", "mix635_vs_mix835", 90, rep)),
                fmt(power_lookup("class_mixture", "mix635_vs_mix835", 120, rep)),
            ]
        )

    center6_rows = []
    center8_rows = []
    for rep in REPS:
        center6_rows.append(
            [
                REP_LABEL[rep],
                fmt(power_lookup("center_shift", "label6_source_vs_external", 20, rep)),
                fmt(power_lookup("center_shift", "label6_source_vs_external", 30, rep)),
                fmt(power_lookup("center_shift", "label6_source_vs_external", 50, rep)),
            ]
        )
        center8_rows.append(
            [
                REP_LABEL[rep],
                fmt(power_lookup("center_shift", "label8_source_vs_external", 20, rep)),
                fmt(power_lookup("center_shift", "label8_source_vs_external", 30, rep)),
                fmt(power_lookup("center_shift", "label8_source_vs_external", 50, rep)),
            ]
        )

    std_rows = []
    for label, scenario in [("6", "label6_source_vs_external"), ("8", "label8_source_vs_external")]:
        for rep in ["final_semantic", "final_residual"]:
            std_rows.append(
                [
                    f"label {label}",
                    REP_LABEL[rep],
                    fmt(power_lookup("center_shift", scenario, 20, rep, "channel_standardized")),
                    fmt(power_lookup("center_shift", scenario, 30, rep, "channel_standardized")),
                    fmt(power_lookup("center_shift", scenario, 50, rep, "channel_standardized")),
                ]
            )

    type_rows = []
    for rep in REPS:
        type_rows.append(
            [
                REP_LABEL[rep],
                fmt(type1_max(rep, "none", "class_mixture")),
                fmt(type1_max(rep, "none", "center_shift")),
                fmt(type1_max(rep, "none")),
            ]
        )

    std_type_rows = []
    for rep in ["final_semantic", "final_residual"]:
        std_type_rows.append(
            [
                REP_LABEL[rep],
                fmt(type1_max(rep, "channel_standardized", "center_shift")),
                fmt(type1_max_setting(rep, "channel_standardized", "center_shift", "H0_source")),
                fmt(type1_max_setting(rep, "channel_standardized", "center_shift", "H0_external")),
            ]
        )

    diag_rows = []
    for item in DIAG:
        diag_rows.append(
            [
                item["tag"] + (" + std" if item["preprocess"] == "channel_standardized" else ""),
                f"{item['row_space_rank']}/128",
                fmt(item["mean_norm_final_full"]),
                fmt(item["mean_semantic_norm_fraction"]),
                fmt(item["mean_residual_norm_fraction"]),
            ]
        )

    artifact_rows = []
    for name in [
        "semantic_residual_mmmd_summary.csv",
        "semantic_residual_type1_summary.csv",
        "semantic_vs_residual_power_plot.png",
        "semantic_vs_residual_type1_plot.png",
        "witness_examples_class_mixture_semantic.png",
        "witness_examples_class_mixture_residual.png",
        "witness_examples_center_shift_semantic.png",
        "witness_examples_center_shift_residual.png",
        "short_interpretation.md",
    ]:
        path = RES / name
        artifact_rows.append([name, f"{path.stat().st_size / 1024:.1f} KB"])

    raw_rows = [
        ["semantic_residual_power_results.csv", f"{(RES / 'semantic_residual_power_results.csv').stat().st_size / 1024 / 1024:.1f} MB"],
        ["semantic_residual_type1_results.csv", f"{(RES / 'semantic_residual_type1_results.csv').stat().st_size / 1024 / 1024:.1f} MB"],
    ]

    return {
        "class": make_table(["Representation", "n=60", "n=90", "n=120"], class_rows),
        "center6": make_table(["Representation", "n=20", "n=30", "n=50"], center6_rows),
        "center8": make_table(["Representation", "n=20", "n=30", "n=50"], center8_rows),
        "std": make_table(["Label", "Representation", "n=20", "n=30", "n=50"], std_rows),
        "type1": make_table(["Representation", "class H0 max", "center H0 max", "all H0 max"], type_rows),
        "std_type1": make_table(["Representation", "center std max", "H0-source max", "H0-external max"], std_type_rows),
        "diag": make_table(["Pool/preprocess", "rank", "mean ||h||", "semantic frac", "residual frac"], diag_rows),
        "artifacts": make_table(["File", "Size"], artifact_rows),
        "raw": make_table(["Raw file", "Size"], raw_rows),
    }


def main() -> None:
    tables = build_tables()
    footer = (
        f"Generated from {EXP.relative_to(ROOT)} on 2026-06-05. "
        "No experiments were rerun for this report."
    )

    with PdfPages(OUT) as pdf:
        meta = pdf.infodict()
        meta["Title"] = "Semantic vs Residual CNN-MMMD Decomposition: PathMNIST Report"
        meta["Author"] = "kernel_two-sample project"
        meta["Subject"] = "PathMNIST semantic-residual MMMD experiment summary"
        meta["Keywords"] = "MMMD, PathMNIST, CNN, semantic residual, kernel two-sample test"

        add_text_page(
            pdf,
            "Semantic vs Residual CNN-MMMD Decomposition",
            [
                ("p", "PathMNIST report for discussing the semantic-residual biostatistics extension."),
                ("h", "Core Question"),
                (
                    "p",
                    "What does the CNN-MMMD statistic detect when final_fc128 is decomposed into the "
                    "classifier-used semantic row space and the classifier-ignored residual subspace?",
                ),
                ("h", "One-line Result"),
                (
                    "p",
                    "Class-mixture differences are strong in all representations, while center-shift differences "
                    "can remain strong in residual space, especially for label 8. This supports the interpretation "
                    "that CNN-MMMD is not only reading out class logits; it can also detect classifier-ignored "
                    "domain, staining, morphology, or acquisition variation.",
                ),
                ("h", "Highest-signal Findings"),
                (
                    "b",
                    [
                        "The classifier row space has rank 9 inside a 128-dimensional final_fc128 embedding.",
                        "Class mixture mix635_vs_mix835 at n=120: logits 0.980, final_full 0.963, final_semantic 0.958, final_residual 0.955.",
                        "Center shift label 6 at n=50: logits/semantic are strongest, but residual remains high: logits 0.858, semantic 0.825, residual 0.750, full 0.763.",
                        "Center shift label 8 at n=50: final_residual 0.958 and final_full 0.957 exceed logits 0.820 and semantic 0.840.",
                        "Per-image channel standardization sharply reduces center-shift power, suggesting that color/stain/channel differences are a major component of the center signal.",
                        "Matched-null Type-I checks are mildly liberal, with maxima near 0.10 rather than nominal alpha=0.05.",
                    ],
                ),
            ],
            footer,
        )

        add_text_page(
            pdf,
            "Experiment Design",
            [
                ("h", "Feature Decomposition"),
                ("p", "For each image, the existing CNN checkpoint produces final_fc128 h and classifier logits:"),
                ("code", "logits = W h + b\nh_sem = projection of h onto row(W), computed by SVD\nh_res = h - h_sem"),
                (
                    "p",
                    "MMMD Gaussian5 was evaluated on final_full=h, logits, final_semantic=h_sem, and final_residual=h_res.",
                ),
                ("h", "Tasks"),
                (
                    "b",
                    [
                        "Class mixture H1: mix635_vs_mix835 at n = 60, 90, 120.",
                        "Class mixture matched null: mix635_vs_mix635 at n = 60, 120.",
                        "Center shift H1: label 6 and label 8 source_holdout vs external at n = 20, 30, 50.",
                        "Center H0 checks: H0-source and H0-external for labels 6 and 8.",
                        "Witness examples: top positive/negative MMD witness image grids for semantic and residual spaces.",
                        "Optional intervention: per-image channel standardization for center-shift semantic and residual spaces.",
                    ],
                ),
                ("h", "Loop Configuration"),
                (
                    "code",
                    "outer Monte Carlo repetitions = 10\n"
                    "inner independent two-sample tests per outer repetition = 500\n"
                    "total independent tests per cell = 5,000\n"
                    "multiplier bootstrap B_boot = 500\n"
                    "alpha = 0.05",
                ),
                ("h", "Reuse Policy"),
                (
                    "p",
                    "No CNN was retrained. final_full rows reuse completed cnn_final_fc128_gaussian5 formal summaries. "
                    "logits, final_semantic, and final_residual rows were computed for this decomposition experiment.",
                ),
            ],
            footer,
        )

        add_text_page(
            pdf,
            "Projection Diagnostics",
            [
                (
                    "p",
                    "The classifier weight W has shape 9 x 128. SVD gives a rank-9 classifier row space, "
                    "leaving most final_fc128 dimensions in the residual subspace.",
                ),
                ("code", tables["diag"]),
                (
                    "p",
                    "Interpretation: the residual vector has a larger norm fraction because it spans the 119-dimensional "
                    "orthogonal complement. A strong residual MMMD result means the two-sample signal is not exhausted "
                    "by the classifier row space.",
                ),
            ],
            footer,
        )

        add_single_image_page(
            pdf,
            "Power Curves",
            RES / "semantic_vs_residual_power_plot.png",
            "Solid lines use original images/features. Dashed + channel std lines are the optional center-shift intervention.",
        )

        add_text_page(
            pdf,
            "Power Tables",
            [
                ("h", "Class Mixture: mix635_vs_mix835"),
                ("code", tables["class"]),
                ("h", "Center Shift: label 6 source_holdout vs external"),
                ("code", tables["center6"]),
                ("h", "Center Shift: label 8 source_holdout vs external"),
                ("code", tables["center8"]),
                (
                    "p",
                    "Readout: label 6 has a stronger semantic/logit component, while label 8 is dominated by "
                    "residual/full features. This split is the main evidence that CNN-MMMD is detecting different "
                    "biological or technical variation depending on the label.",
                ),
            ],
            footer,
        )

        add_single_image_page(
            pdf,
            "Type-I Calibration Checks",
            RES / "semantic_vs_residual_type1_plot.png",
            "Matched null checks should be read against nominal alpha=0.05. Several cells are closer to 0.08-0.10.",
        )

        add_text_page(
            pdf,
            "Type-I Summary",
            [
                (
                    "p",
                    "The table reports the maximum empirical Type-I error observed across matched-null cells for each representation.",
                ),
                ("code", tables["type1"]),
                (
                    "p",
                    "These maxima are above nominal 0.05. Because each summary cell contains 5,000 independent tests, "
                    "the binomial standard error around p=0.05 is about 0.003. Values around 0.10 indicate liberal "
                    "calibration, not just Monte Carlo noise.",
                ),
                ("h", "Channel-standardized Center H0"),
                ("code", tables["std_type1"]),
                (
                    "p",
                    "For future claims, report H1 power together with the matched H0-source/H0-external calibration for "
                    "the same n and representation.",
                ),
            ],
            footer,
        )

        add_text_page(
            pdf,
            "Channel Standardization Intervention",
            [
                (
                    "p",
                    "The optional intervention standardizes each image by channel before feature extraction, then reruns "
                    "center-shift MMMD only for final_semantic and final_residual. This tests whether center signals are "
                    "driven by per-image channel/stain/color information.",
                ),
                ("code", tables["std"]),
                (
                    "p",
                    "At n=50, label 6 drops from semantic/residual 0.825/0.750 to 0.338/0.372 after channel standardization. "
                    "Label 8 drops from 0.840/0.958 to 0.564/0.599. This is strong evidence that color/stain/channel "
                    "structure accounts for a large part of the center-shift detectability.",
                ),
            ],
            footer,
        )

        add_two_image_page(
            pdf,
            "Witness Examples: Class Mixture",
            RES / "witness_examples_class_mixture_semantic.png",
            RES / "witness_examples_class_mixture_residual.png",
            "Semantic-space witness grid: top positive and negative examples for the class-mixture contrast.",
            "Residual-space witness grid: top positive and negative examples for the same class-mixture contrast.",
            mode="vertical",
        )

        add_two_image_page(
            pdf,
            "Witness Examples: Center Shift",
            RES / "witness_examples_center_shift_semantic.png",
            RES / "witness_examples_center_shift_residual.png",
            "Semantic-space witness grid for center-shift contrast.",
            "Residual-space witness grid for center-shift contrast.",
            mode="horizontal",
        )

        add_text_page(
            pdf,
            "Interpretation for Next Discussion",
            [
                ("h", "Biostatistical Meaning"),
                (
                    "b",
                    [
                        "Class-mixture MMMD is mostly a class-distribution signal, but not exclusively a logit signal: the residual subspace also carries strong class-separable variation.",
                        "Center-shift MMMD is partly biological and partly technical/domain-sensitive. Label 8 is the clearest residual-dominated case; label 6 is more classifier-semantic.",
                        "The residual subspace is not noise. It can encode morphology, staining, scanner/site, or other acquisition/domain structure ignored by the supervised classifier objective.",
                        "Channel standardization is a useful negative-control style intervention: if power collapses, much of the detected shift is channel/stain-like rather than purely shape/semantic.",
                    ],
                ),
                ("h", "Recommended Follow-up Questions"),
                (
                    "b",
                    [
                        "How should we phrase the statistical estimand: class-mixture difference, center/domain shift, or classifier-ignored residual distribution shift?",
                        "Should Type-I liberalness be handled by more bootstrap draws, a stricter threshold, permutation calibration, or reporting power alongside matched null inflation?",
                        "For a biostatistics final project, should the next extension prioritize calibration robustness or biological interpretability of witness examples?",
                        "Would a site-adjustment/control experiment, such as stain normalization or covariate-adjusted kernel testing, make the center-shift claim stronger?",
                    ],
                ),
            ],
            footer,
        )

        add_text_page(
            pdf,
            "Artifacts and Reproducibility",
            [
                ("p", "All deliverables are in the semantic/residual experiment folder. The concise files below are the ones to share first."),
                ("code", tables["artifacts"]),
                (
                    "p",
                    "Raw per-test CSVs are much larger and usually do not need to be uploaded unless summaries need to be recomputed.",
                ),
                ("code", tables["raw"]),
                ("h", "Important Paths"),
                ("code", f"experiment root: {EXP}\nresults folder:  {RES}\nreport PDF:      {OUT}"),
                (
                    "p",
                    "The report was generated only from completed outputs. No CNN training, embedding extraction, or MMMD experiment was rerun while making this PDF.",
                ),
            ],
            footer,
        )

    print(OUT)
    print(f"size_mb={OUT.stat().st_size / 1024 / 1024:.2f}")


if __name__ == "__main__":
    main()
