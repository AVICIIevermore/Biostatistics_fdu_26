"""Type-I error sanity check: X and Y both drawn from class 3 (same distribution).
Rejection rate should be approximately alpha = 0.05."""

import numpy as np
import time
from mmmd_pathmnist import load_pathmnist, class_pools, gexp5_one, new_one

all_X, all_y = load_pathmnist()
pools = class_pools(all_y)

def sample_null(n, rng):
    """Both X and Y from class 3, disjoint."""
    pool = rng.choice(pools[3], 2 * n, replace=False)
    return all_X[pool[:n]], all_X[pool[n:]]

master = np.random.default_rng(42)
n_iter = 100
for n in (60, 120):
    for name, runner in [("GEXP-5", gexp5_one), ("NEW-MMMD", new_one)]:
        rej = []
        t0 = time.time()
        for it in range(n_iter):
            ss = master.spawn(2)
            X, Y = sample_null(n, np.random.default_rng(ss[0]))
            r, *_ = runner(X, Y, n_boot=200, rng=np.random.default_rng(ss[1]))
            rej.append(r)
        print(f"[n={n} {name}] type-I error = {np.mean(rej):.3f}  (target ~0.05)  "
              f"time={time.time()-t0:.1f}s")
