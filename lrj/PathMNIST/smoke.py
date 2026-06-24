"""Smoke test: verify fast_cov_gaussian_t matches est_cov_naive, and run 5 MC
iterations with each method at n=60 to catch bugs before the full run."""

import numpy as np
import time
from mmmd_pathmnist import (
    load_pathmnist, class_pools, sample_mix_635_vs_835,
    est_cov_naive, fast_cov_gaussian_t, t_med_pooled,
    gexp5_one, new_one,
)

print("=== sanity check: fast_cov_gaussian_t vs est_cov_naive ===")
rng = np.random.default_rng(1)
X_toy = rng.normal(size=(30, 2))
t_vec = 2 ** np.linspace(-2, 2, 5)
S_naive = est_cov_naive(X_toy, t_vec)
S_fast = fast_cov_gaussian_t(X_toy, t_vec)
print(f"max |naive - fast| = {np.max(np.abs(S_naive - S_fast)):.3e}")
assert np.max(np.abs(S_naive - S_fast)) < 1e-8, "FAST COV MISMATCH"
print("PASS")

print("\n=== smoke run: 5 MC iters, n=60, both methods ===")
all_X, all_y = load_pathmnist()
pools = class_pools(all_y)
master = np.random.default_rng(0)
for name, runner in [("GEXP-5", gexp5_one), ("NEW-MMMD", new_one)]:
    t0 = time.time()
    rej_list, q_list = [], []
    for it in range(5):
        ss = master.spawn(2)
        X, Y = sample_mix_635_vs_835(60, np.random.default_rng(ss[0]), all_X, pools)
        rej, q, cr, crg, lam = runner(X, Y, n_boot=200, rng=np.random.default_rng(ss[1]))
        rej_list.append(rej); q_list.append(q)
        print(f"  iter {it}: rej={rej} q={q} cond_raw={cr:.2e} cond_reg={crg:.2e} lam={lam:.2e}")
    print(f"  {name}: 5-iter power={np.mean(rej_list):.2f}  elapsed={time.time()-t0:.1f}s\n")
