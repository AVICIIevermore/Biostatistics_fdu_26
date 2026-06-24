# Label-8 Residual Signal by Color: Short Report

This run reused the existing PathMNIST center-shift CNN checkpoint and cached centered residual features. No model was retrained and no new dataset was added.

## Domain-axis color R2

- residual_top8_pca: color15 R2 = 0.728, domain AUC before/after score color residualization = 0.778 / 0.622.
- residual_full logistic axis: color15 R2 = 0.534, domain AUC before/after score color residualization = 0.869 / 0.737.
- top correlated color features for residual_top8_pca: q10_G:0.8092;q10_B:0.8027;q10_R:0.7467;std_R:0.6599;std_B:0.5897.
- top correlated color features for residual_full: q10_G:0.6814;q10_B:0.6786;q10_R:0.6135;std_R:0.5423;std_B:0.4908.

## Color-matched residual MMMD

- matching quality: matched pairs = 846, mean propensity logit gap = 0.009, matched color-only AUC = 0.632.
- label 8 color-matched H1 power at n=50: residual_top8_pca = 0.455, residual_full = 0.455.
- max H0 rejection rate across color-matched residual checks = 0.110.

## Safe conclusion

- Color/stain explains a substantial part of the label-8 residual domain signal: the residual_top8_pca domain score has color R2 0.728 and its AUC drops from 0.778 to 0.622 after score-level color residualization.
- Some residual structure remains after stricter color-propensity matching: at n=50, both residual_top8_pca and residual_full reject at 0.455 / 0.455, while the largest matched H0 rejection is 0.110.
- Because matched color-only AUC is still above 0.5, this is best described as color-reduced evidence rather than a perfectly color-randomized comparison.
- If color R2 is high and the domain AUC drops strongly after score residualization, a large part of the label-8 residual signal should be described as color/stain-associated residual domain shift.
- If color-matched residual MMMD remains clearly above H0, then some color-adjusted residual structure remains, but it should be presented conservatively as residual texture/morphology/acquisition structure rather than a pure biological morphology claim.
