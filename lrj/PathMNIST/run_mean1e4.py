"""run_mean1e4.py - re-run the PathMNIST experiment with ridge rule
   lambda = 1e-4 * mean(diag(Sigma_hat))
   (vs the default 1e-5 * min(diag(Sigma_hat))).

   Everything else (sampling, n_iter, n_boot, seed, both H1/H0 scenarios) is
   identical to the default run, so the only thing that changes between this
   output and mmmd_pathmnist_results_min1e5.csv is the ridge convention."""

import numpy as np
import mmmd_pathmnist as mp

mp.RIDGE_FACTOR = 1e-4
mp.RIDGE_AGG = np.mean

df = mp.run_experiment(
    n_seq=(60, 120), n_iter=500, n_boot=200,
    per_iter_csv="mmmd_pathmnist_per_iter_mean1e4.csv",
)
df.to_csv("mmmd_pathmnist_results_mean1e4.csv", index=False)
print("\n========= summary table (ridge = 1e-4 * mean(diag)) =========")
print(df.to_string(index=False))
print("\nSaved: mmmd_pathmnist_results_mean1e4.csv, mmmd_pathmnist_per_iter_mean1e4.csv")
