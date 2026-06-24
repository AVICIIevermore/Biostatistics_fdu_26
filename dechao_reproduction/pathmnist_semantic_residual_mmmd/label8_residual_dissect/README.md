# Label-8 Residual Signal Dissection

This follow-up experiment explains why PathMNIST label 8 center-shift is residual-dominated.

Scope:

- no new dataset
- no CNN retraining
- no ordinary layer-wise comparison
- reuse existing PathMNIST center-shift split and CNN checkpoint
- use centered-W semantic/residual representations

Main outputs are written to `Results/`:

- `cross_label_domain_axis_transfer.csv`
- `cross_label_domain_axis_transfer.png`
- `color_domain_probe_auc.csv`
- `color_only_mmmd_summary.csv`
- `color_adjusted_embedding_mmmd_summary.csv`
- `color_adjusted_center_shift_plot.png`
- `residual_intrinsic_dimension_summary.csv`
- `residual_intrinsic_dimension_curve.png`
- `label8_residual_dissect_short_report.md`
