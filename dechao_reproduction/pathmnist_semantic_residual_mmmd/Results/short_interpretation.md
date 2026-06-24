# Semantic vs Residual MMMD Interpretation

## Setup

- `final_full` reuses the completed `cnn_final_fc128_gaussian5` formal PathMNIST MMMD summaries.
- `logits`, `final_semantic`, and `final_residual` are newly evaluated with Gaussian5 MMMD.
- The projection uses SVD of the classifier weight `W`; no CNN checkpoint is retrained.
- Formal loops: outer repetitions = 10, inner independent tests = 500, multiplier bootstrap B = 500.

## Projection Diagnostics

- `class_mixture_test` / `none`: row-space rank 9 of 128, mean norm fractions semantic=0.349, residual=0.934.
- `center_source_holdout` / `none`: row-space rank 9 of 128, mean norm fractions semantic=0.320, residual=0.944.
- `center_external` / `none`: row-space rank 9 of 128, mean norm fractions semantic=0.308, residual=0.947.

## Key Readout

Class mixture power at n=120:

- `final_full`: 0.9628
- `logits`: 0.9798
- `final_semantic`: 0.9582
- `final_residual`: 0.9546

Center-shift power at n=50, label 6:

- `final_full`: 0.7634
- `logits`: 0.8578
- `final_semantic`: 0.8250
- `final_residual`: 0.7504

Center-shift power at n=50, label 8:

- `final_full`: 0.9574
- `logits`: 0.8200
- `final_semantic`: 0.8400
- `final_residual`: 0.9580

Maximum Type-I error observed across the matched null checks, by representation:

- `final_full`: 0.0986
- `logits`: 0.1002
- `final_semantic`: 0.0990
- `final_residual`: 0.1010

## Interpretation

- If `final_semantic` tracks class-mixture power and `final_residual` is weaker, the class-mixture MMMD signal is mostly classifier-used class semantics.
- If `final_residual` remains strong for center-shift while `logits` or `final_semantic` are weaker, the center-shift signal includes classifier-ignored morphology, staining, scanner, or domain residual variation.
- `logits` is a 9-dimensional compressed semantic representation. Similar behavior between `logits` and `final_semantic` means the classifier row space is largely enough for the detected signal; divergence means extra geometry inside the row-space projection matters.
- The center-shift H0-source/H0-external rows should be read as calibration checks. High H1 power is only compelling when Type-I error for the matching representation and n is near alpha=0.05.

## Optional Channel Standardization

Included in the summary CSVs and plotted as dashed `+ channel std` lines for center-shift semantic/residual rows.

Center-shift power at n=50 after per-image channel standardization:

Label 6:

- `final_semantic`: 0.3376
- `final_residual`: 0.3722

Label 8:

- `final_semantic`: 0.5636
- `final_residual`: 0.5992

## Outputs

- Power summary: `Results/semantic_residual_mmmd_summary.csv`
- Type-I summary: `Results/semantic_residual_type1_summary.csv`
- Power plot: `Results/semantic_vs_residual_power_plot.png`
- Type-I plot: `Results/semantic_vs_residual_type1_plot.png`
- Witness grids: `witness_examples_*_{semantic,residual}.png`
