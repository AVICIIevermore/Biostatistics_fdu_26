# PathMNIST Center-Shift MMMD

This experiment tests within-class source-to-external distribution shift on PathMNIST-28.

Data responsibilities:

- `cnn_train_pool`: stratified 80% of official train split, used only for CNN training.
- `source_holdout_pool`: stratified 20% of official train split, used only for source-domain MMMD samples.
- `val`: official validation split, used only for checkpoint selection.
- `external_pool`: official test split, used only for source-vs-external MMMD and external-domain null checks.

Formal MMMD scale, once smoke tests pass:

```text
outer_repetitions = 10
inner_repetitions = 500
B_boot = 500
```

The CNN checkpoint in `shared_model/` is specific to this center-shift design and should not be replaced by the prior full-train PathMNIST checkpoint.
