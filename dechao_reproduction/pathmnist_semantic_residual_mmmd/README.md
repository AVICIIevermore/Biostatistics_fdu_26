# PathMNIST Semantic/Residual MMMD Decomposition

This experiment decomposes the frozen PathMNIST CNN `final_fc128` representation into:

- `final_full`: the original 128-dimensional penultimate embedding.
- `logits`: `W h + b` from the trained classifier head.
- `final_semantic`: projection of `h` onto the row space of classifier weight `W`.
- `final_residual`: residual component orthogonal to the classifier row space.

The projection uses SVD of the classifier weight for numerical stability. Existing PathMNIST CNN checkpoints are reused; no retraining is performed.

Primary outputs are written to `Results/`:

- `semantic_residual_mmmd_summary.csv`
- `semantic_residual_type1_summary.csv`
- `semantic_vs_residual_power_plot.png`
- `semantic_vs_residual_type1_plot.png`
- `witness_examples_class_mixture_semantic.png`
- `witness_examples_class_mixture_residual.png`
- `witness_examples_center_shift_semantic.png`
- `witness_examples_center_shift_residual.png`
- `short_interpretation.md`

`final_full` rows are reused from completed formal PathMNIST MMMD summaries where the configuration matches exactly.
