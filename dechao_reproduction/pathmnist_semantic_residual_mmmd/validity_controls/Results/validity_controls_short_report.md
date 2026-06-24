# Validity Controls Short Report

This report is generated from completed validity-control outputs only. No CNN checkpoint is retrained.
Actual run size: outer=10, inner=20, B_boot=500, B_perm=500 for permutation cells, non-random tests/cell=200, residual_random8 aggregate tests/cell=4000. Full inner=500 was not used for this control run; `plan_next.md` allows reducing inner tests when runtime is high as long as the actual test count is reported.

## 1. Does centered W change the semantic/residual conclusion?

- Centering behaves as expected: W has rank 9 and Wc has rank 8 in all checked pools, because the class-common logit direction is removed.
- Diagnostics: class_mixture_test: rank(W)=9, rank(Wc)=8, mean norm fractions sem=0.214, res=0.972; center_source_holdout: rank(W)=9, rank(Wc)=8, mean norm fractions sem=0.215, res=0.970; center_external: rank(W)=9, rank(Wc)=8, mean norm fractions sem=0.177, res=0.979.
- The residual component still carries most of the feature norm, so the previous residual signal is not an artifact of including the all-class logit direction in `row(W)`.

## 2. Does residual power survive dimension matching?

- Class-mixture H1 at n=120 remains strongest in classifier-used coordinates: centered_logits=0.990, semantic_centered=0.970, residual_top8_pca=0.915, residual_random8=0.836, residual_full=0.940, final_full=0.960.
- Center-shift label 6 at n=50 is more semantic/logit driven: semantic_centered=0.920, residual_top8_pca=0.735, residual_random8=0.661, residual_full=0.815.
- Center-shift label 8 at n=50 is strongly residual driven even after dimension matching: semantic_centered=0.705, residual_top8_pca=0.965, residual_random8=0.889, residual_full=0.975.
- Dimension-control H0 rejection is slightly above nominal in the worst cells but not explosive: max H0 rejection across centered_logits=0.100, semantic_centered=0.105, residual_top8_pca=0.100, residual_random8=0.103, residual_full=0.105, final_full=0.110.

## 3. Is center-shift robust under permutation calibration?

- Label 6 n=50 remains high under permutation: semantic_centered bootstrap/permutation=0.920/0.960; residual_full=0.815/0.850.
- Label 8 n=50 remains very high under permutation: semantic_centered bootstrap/permutation=0.715/0.825; residual_top8_pca=0.960/0.970; residual_full=0.970/0.975.
- Permutation H0 is closer to nominal than bootstrap in the worst cells: max bootstrap H0=0.130, max permutation H0=0.100.

## 4. Is center-shift mostly explained by color/stain features?

- Not answered by this core run. The optional color-only/domain-probe/color-residualized analysis in `plan_next.md` was not run.
- Current evidence says the residual signal is not merely high dimensional and is not removed by permutation calibration. It does not yet separate biological morphology from stain/color/domain effects.

## Main message to verify

CNN-MMMD is not only increasing power uniformly. In these controls, class-mixture is mostly classifier-semantic/logit aligned, label-6 center-shift is more semantic/logit aligned, and label-8 center-shift contains a strong classifier-ignored residual component that survives 8D PCA/random projection controls and permutation calibration.
