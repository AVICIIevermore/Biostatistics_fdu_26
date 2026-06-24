# Label-8 Residual Signal Dissection: Short Report

This report uses existing PathMNIST center-shift CNN checkpoints and centered-W residual features. No CNN was retrained.

## 1. Is label-8 residual shift shared with label 6, or label-specific?

- residual_full logistic AUC: train 8 -> test 8 = 0.866, train 8 -> test 6 = 0.677, train 6 -> test 8 = 0.595.
- Interpretation: 更像 label-specific residual/domain interaction.

## 2. How much is explained by color/stain features?

- color-only logistic AUC: label 6 = 0.821, label 8 = 0.773.
- color-only MMMD power at n=50 for label 8 = 0.960.
- after color adjustment, residual_full MMMD power at n=50: label 6 = 0.300, label 8 = 0.500.
- If color-only is high and adjusted residual drops, the residual shift is substantially stain/site/channel driven. If adjusted residual remains high, non-color texture/morphology/acquisition structure remains.

## 3. Is residual shift low-dimensional or distributed?

- label 8 residual PCA power at n=50: k=1 = 0.595, k=8 = 0.970, k=32 = 0.965.
- max intrinsic-dimension H0 rejection = 0.115.
- If k=1/2 is already high, the shift is close to a low-dimensional domain axis. If power rises gradually through k=16/32, it is distributed across residual texture/morphology directions.

## 4. Safest interpretation for final presentation

- The label-8 center-shift residual signal is robust, but it should be presented as a residual domain-shift signal unless color-adjusted residual power and transfer results clearly support a stronger morphology-domain interaction claim.
- Avoid claiming a purely biological morphology difference unless color/stain adjustment still leaves high residual power and cross-label transfer is weak.
