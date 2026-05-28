"""
mmmd_pathmnist.py

PathMNIST mix635_vs_mix835 two-sample test, comparing:
  1. GEXP-5 MMMD (paper baseline, 5 Gaussian kernels, naive est.cov)
  2. NEW-MMMD  (13 Gaussian kernels arithmetic t-grid, multiplicative-closure
                 fast cov + pivoted Cholesky kernel pruning)

Data setup:
  X group = class 6 + class 3 + class 5, class-balanced
  Y group = class 8 + class 3 + class 5, class-balanced
  The shared classes (3, 5) are drawn from disjoint pools for X and Y.
  Feature: raw pixels in [0,1] (28*28*3 = 2352-dim flattened).

Sample sizes: n in {60, 120} (per-group). Per-class share is n/3.

Outputs per (method, n):
  power, median q, median cond(Sigma), median cond(Sigma + lambda I),
  median lambda, runtime.
"""

import numpy as np
import pandas as pd
import time
from scipy.spatial.distance import pdist, squareform, cdist


# ============================================================================
# 1. Data loading and sampling
# ============================================================================
def load_pathmnist(path="pathmnist.npz"):
    d = np.load(path)
    X = np.concatenate([d["train_images"], d["val_images"], d["test_images"]], axis=0)
    y = np.concatenate([d["train_labels"], d["val_labels"], d["test_labels"]],
                       axis=0).ravel().astype(int)
    X = X.reshape(X.shape[0], -1).astype(np.float64) / 255.0
    return X, y


def class_pools(y, classes=(3, 5, 6, 8)):
    return {c: np.where(y == c)[0] for c in classes}


def sample_mix_635_vs_835(n, rng, all_X, pools):
    """H1 scenario. X = (6, 3, 5) class-balanced; Y = (8, 3, 5) class-balanced.
    Shared classes (3, 5) drawn from disjoint pools so X/Y are independent draws."""
    assert n % 3 == 0
    k = n // 3
    idx_6 = rng.choice(pools[6], k, replace=False)
    idx_8 = rng.choice(pools[8], k, replace=False)
    pool_3 = rng.choice(pools[3], 2 * k, replace=False)
    pool_5 = rng.choice(pools[5], 2 * k, replace=False)
    iX = np.concatenate([idx_6, pool_3[:k], pool_5[:k]])
    iY = np.concatenate([idx_8, pool_3[k:], pool_5[k:]])
    return all_X[iX], all_X[iY]


def sample_mix_635_vs_635_null(n, rng, all_X, pools):
    """H0 scenario. X = (6, 3, 5) class-balanced; Y = (6, 3, 5) class-balanced.
    Same composition as the alternative, but ALL three classes drawn from disjoint
    pools between X and Y so the test sees two independent draws from the same
    structured mixture distribution. Used for size calibration on the same shape
    of background as the alternative."""
    assert n % 3 == 0
    k = n // 3
    pool_6 = rng.choice(pools[6], 2 * k, replace=False)
    pool_3 = rng.choice(pools[3], 2 * k, replace=False)
    pool_5 = rng.choice(pools[5], 2 * k, replace=False)
    iX = np.concatenate([pool_6[:k], pool_3[:k], pool_5[:k]])
    iY = np.concatenate([pool_6[k:], pool_3[k:], pool_5[k:]])
    return all_X[iX], all_X[iY]


# ============================================================================
# 2. MMD U-statistic and bandwidth (matching R Functions.R conventions)
# ============================================================================
def t_med_pooled(X, Y):
    """t_med = 1 / median squared pairwise distance over pooled (X, Y)."""
    Z = np.vstack([X, Y])
    return 1.0 / np.median(pdist(Z) ** 2)


def sqdist_mat(X, Y=None):
    if Y is None:
        return squareform(pdist(X)) ** 2
    return cdist(X, Y, "sqeuclidean")


def mmd_offdiag(K_X, K_Y, K_XY):
    """MMD U-statistic in the R convention: mean over (i != j) of
       K_X[i,j] + K_Y[i,j] - 2*K_XY[i,j]."""
    m = K_X.shape[0]
    a = (K_X.sum() - np.trace(K_X)) / (m * (m - 1))
    b = (K_Y.sum() - np.trace(K_Y)) / (m * (m - 1))
    c = (K_XY.sum() - np.trace(K_XY)) / (m * (m - 1))
    return a + b - 2 * c


# ============================================================================
# 3. Naive O(R^2 m^3) covariance for GEXP-5
# ============================================================================
def est_cov_naive(X, t_vec, D2_XX=None):
    """Returns Sigma (unregularized), R x R."""
    m = X.shape[0]
    if D2_XX is None:
        D2_XX = sqdist_mat(X)
    C = np.eye(m) - np.ones((m, m)) / m
    Kc_list = [C @ np.exp(-t * D2_XX) @ C for t in t_vec]
    R = len(t_vec)
    S = np.zeros((R, R))
    for i in range(R):
        for j in range(R):
            S[i, j] = (8.0 / m ** 2) * np.trace(Kc_list[i] @ Kc_list[j])
    return S


# ============================================================================
# 4. Fast covariance for the new method (multiplicative closure)
# ============================================================================
def fast_cov_gaussian_t(X, t_vec, D2_XX=None):
    """Returns Sigma (unregularized), R x R, in O(R m^2 + |unique sums| * m^2)."""
    m = X.shape[0]
    R = len(t_vec)
    if D2_XX is None:
        D2_XX = sqdist_mat(X)

    s_mat = np.zeros((m, R))
    S_vec = np.zeros(R)
    for a in range(R):
        Ka = np.exp(-t_vec[a] * D2_XX)
        s_mat[:, a] = Ka.sum(axis=1)
        S_vec[a] = Ka.sum()

    t_sum = np.add.outer(t_vec, t_vec)
    t_sum_round = np.round(t_sum, 10)
    uniq, inv_idx = np.unique(t_sum_round, return_inverse=True)
    H_unique = np.array([np.exp(-u * D2_XX).sum() for u in uniq])
    H_mat = H_unique[inv_idx].reshape(R, R)

    G_mat = s_mat.T @ s_mat
    inner = H_mat - (2.0 / m) * G_mat + (1.0 / m ** 2) * np.outer(S_vec, S_vec)
    Sigma = (8.0 / m ** 2) * inner
    Sigma = 0.5 * (Sigma + Sigma.T)
    return Sigma


# ============================================================================
# 5. Pivoted Cholesky kernel-subset selection
# ============================================================================
def pivoted_chol_select(Sigma, eps=1e-4, q_max=None):
    R = Sigma.shape[0]
    if q_max is None:
        q_max = R
    diag_S = np.diag(Sigma).copy()
    resid = diag_S.copy()
    L = np.zeros((R, q_max))
    selected = []
    threshold = eps * diag_S.max()
    for t in range(q_max):
        mask = np.ones(R, dtype=bool)
        mask[selected] = False
        if not mask.any():
            break
        cand_resid = np.where(mask, resid, -np.inf)
        pivot = int(np.argmax(cand_resid))
        if resid[pivot] <= threshold:
            break
        selected.append(pivot)
        if t == 0:
            L[:, t] = Sigma[:, pivot] / np.sqrt(resid[pivot])
        else:
            corr = L[:, :t] @ L[pivot, :t]
            L[:, t] = (Sigma[:, pivot] - corr) / np.sqrt(resid[pivot])
        resid = np.maximum(resid - L[:, t] ** 2, 0.0)
    return np.array(selected, dtype=int), len(selected)


# ============================================================================
# 6. Multiplier (Gaussian) bootstrap cutoff
# ============================================================================
def boot_cutoff(X, t_vec_sel, Sigma_inv, n_boot, rng, D2_XX=None):
    """Bootstrap on X-only centered Gram matrices, matching multi.H0.cutoff."""
    m = X.shape[0]
    if D2_XX is None:
        D2_XX = sqdist_mat(X)
    C = np.eye(m) - np.ones((m, m)) / m
    Kc_list = [(1.0 / m) * (C @ np.exp(-t * D2_XX) @ C) for t in t_vec_sel]
    U = rng.normal(0.0, np.sqrt(2.0), size=(n_boot, m))  # N(0, 2 I)
    T_mat = np.zeros((n_boot, len(t_vec_sel)))
    for a, Kc in enumerate(Kc_list):
        Ku = U @ Kc
        T_mat[:, a] = np.einsum("ij,ij->i", U, Ku) - 2.0 * np.trace(Kc)
    ts = np.einsum("ij,jk,ik->i", T_mat, Sigma_inv, T_mat)
    return np.quantile(ts, 0.95)


# ============================================================================
# 7. Per-MC-iteration test wrappers
#
# Ridge convention: lambda = RIDGE_FACTOR * min(diag(Sigma_hat)), matching the
# R project's est.cov / fast.cov.gaussian.t default (1e-5 * min). If aligning
# with a teammate baseline that uses "1e-4 * mean(diag)" instead, switch
# RIDGE_FACTOR and RIDGE_AGG below (kept here as named constants so the change
# is local and explicit).
# ============================================================================
RIDGE_FACTOR = 1e-5
RIDGE_AGG = np.min  # change to np.mean and RIDGE_FACTOR=1e-4 to switch convention


def ridge_lambda(Sigma_raw):
    return float(RIDGE_FACTOR * RIDGE_AGG(np.diag(Sigma_raw)))


def safe_cond(M):
    try:
        return float(np.linalg.cond(M))
    except Exception:
        return np.inf


def gexp5_one(X, Y, n_boot, rng):
    """Returns (rej, q, cond_raw, cond_reg, lam)."""
    m = X.shape[0]
    t_med = t_med_pooled(X, Y)
    t_vec = np.array([(2.0 ** l) * t_med for l in (-2, -1, 0, 1, 2)])
    D2_XX = sqdist_mat(X)

    Sigma_raw = est_cov_naive(X, t_vec, D2_XX=D2_XX)
    lam = ridge_lambda(Sigma_raw)
    Sigma_reg = Sigma_raw + lam * np.eye(len(t_vec))
    cond_raw = safe_cond(Sigma_raw)
    cond_reg = safe_cond(Sigma_reg)

    D2_YY = sqdist_mat(Y)
    D2_XY = sqdist_mat(X, Y)
    mmd_vec = np.empty(len(t_vec))
    for a, t in enumerate(t_vec):
        K_X = np.exp(-t * D2_XX)
        K_Y = np.exp(-t * D2_YY)
        K_XY = np.exp(-t * D2_XY)
        mmd_vec[a] = mmd_offdiag(K_X, K_Y, K_XY)

    Sigma_inv = np.linalg.inv(Sigma_reg)
    Tn = m * mmd_vec
    T_stat = Tn @ Sigma_inv @ Tn
    thr = boot_cutoff(X, t_vec, Sigma_inv, n_boot, rng, D2_XX=D2_XX)
    return int(T_stat > thr), len(t_vec), cond_raw, cond_reg, lam


def new_one(X, Y, n_boot, rng, R=13, eps=1e-4):
    """Returns (rej, q, cond_raw, cond_reg, lam) - reported on the q x q
    selected sub-matrix (the one actually inverted in the test)."""
    m = X.shape[0]
    t_med = t_med_pooled(X, Y)
    t_vec = np.linspace(0.25 * t_med, 4.0 * t_med, R)
    D2_XX = sqdist_mat(X)

    Sigma_full_raw = fast_cov_gaussian_t(X, t_vec, D2_XX=D2_XX)
    lam_full = ridge_lambda(Sigma_full_raw)
    Sigma_full_reg = Sigma_full_raw + lam_full * np.eye(R)

    sel, q = pivoted_chol_select(Sigma_full_reg, eps=eps)
    if q == 0:
        return 0, 0, np.inf, np.inf, np.nan
    Sigma_SS_raw = Sigma_full_raw[np.ix_(sel, sel)]
    lam_sub = ridge_lambda(Sigma_SS_raw)
    Sigma_SS_reg = Sigma_SS_raw + lam_sub * np.eye(q)
    cond_raw = safe_cond(Sigma_SS_raw)
    cond_reg = safe_cond(Sigma_SS_reg)

    t_sel = t_vec[sel]
    D2_YY = sqdist_mat(Y)
    D2_XY = sqdist_mat(X, Y)
    mmd_vec = np.empty(q)
    for a, t in enumerate(t_sel):
        K_X = np.exp(-t * D2_XX)
        K_Y = np.exp(-t * D2_YY)
        K_XY = np.exp(-t * D2_XY)
        mmd_vec[a] = mmd_offdiag(K_X, K_Y, K_XY)

    Sigma_inv = np.linalg.inv(Sigma_SS_reg)
    Tn = m * mmd_vec
    T_stat = Tn @ Sigma_inv @ Tn
    thr = boot_cutoff(X, t_sel, Sigma_inv, n_boot, rng, D2_XX=D2_XX)
    return int(T_stat > thr), q, cond_raw, cond_reg, lam_sub


# ============================================================================
# 8. Driver
# ============================================================================
def run_experiment(n_seq=(60, 120), n_iter=100, n_boot=200, seed=20260528,
                   data_path="pathmnist.npz", per_iter_csv=None):
    all_X, all_y = load_pathmnist(data_path)
    pools = class_pools(all_y)
    print(f"Loaded PathMNIST: pixels in [0,1], dim = {all_X.shape[1]}")
    for c in (3, 5, 6, 8):
        print(f"  class {c}: {len(pools[c])} samples")
    print(f"Ridge convention: lambda = {RIDGE_FACTOR:g} * "
          f"{RIDGE_AGG.__name__}(diag(Sigma_hat))")

    scenarios = [
        ("mix635_vs_mix835",      "H1", sample_mix_635_vs_835),
        ("mix635_vs_mix635_null", "H0", sample_mix_635_vs_635_null),
    ]

    master_rng = np.random.default_rng(seed)
    rows = []
    per_iter_rows = []
    for scen_name, hyp, sampler in scenarios:
        print(f"\n--- scenario: {scen_name} ({hyp}) ---")
        for n in n_seq:
            for method_name, runner in [("GEXP-5", gexp5_one), ("NEW-MMMD", new_one)]:
                rej_list, q_list = [], []
                cond_raw_list, cond_reg_list, lam_list = [], [], []
                t0 = time.time()
                for it in range(n_iter):
                    ss = master_rng.spawn(2)
                    draw_rng = np.random.default_rng(ss[0])
                    boot_rng = np.random.default_rng(ss[1])
                    X, Y = sampler(n, draw_rng, all_X, pools)
                    rej, q, cr, crg, lam = runner(X, Y, n_boot, boot_rng)
                    rej_list.append(rej); q_list.append(q)
                    cond_raw_list.append(cr); cond_reg_list.append(crg); lam_list.append(lam)
                    per_iter_rows.append(dict(scenario=scen_name, hypothesis=hyp,
                                              n=n, method=method_name, iter=it,
                                              rej=rej, q=q, cond_raw=cr,
                                              cond_reg=crg, lam=lam))
                elapsed = time.time() - t0
                row = dict(
                    scenario=scen_name, hypothesis=hyp, method=method_name, n=n,
                    power=float(np.mean(rej_list)),
                    median_q=float(np.median(q_list)),
                    median_cond_Sigma=float(np.median(cond_raw_list)),
                    median_cond_Sigma_plus_lamI=float(np.median(cond_reg_list)),
                    median_lambda=float(np.median(lam_list)),
                    runtime_sec=elapsed,
                )
                rows.append(row)
                label = "power" if hyp == "H1" else "type-I"
                print(f"  [n={n} {method_name}] {label}={row['power']:.3f}  "
                      f"q_med={row['median_q']:.1f}  "
                      f"cond(S)={row['median_cond_Sigma']:.2e}  "
                      f"cond(S+lI)={row['median_cond_Sigma_plus_lamI']:.2e}  "
                      f"lam_med={row['median_lambda']:.2e}  "
                      f"time={elapsed:.1f}s")
    if per_iter_csv is not None:
        pd.DataFrame(per_iter_rows).to_csv(per_iter_csv, index=False)
    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = run_experiment(n_seq=(60, 120), n_iter=500, n_boot=200,
                        per_iter_csv="mmmd_pathmnist_per_iter.csv")
    df.to_csv("mmmd_pathmnist_results.csv", index=False)
    print("\n========= summary table =========")
    print(df.to_string(index=False))
    print(f"\nRidge rule used: lambda = {RIDGE_FACTOR:g} * "
          f"{RIDGE_AGG.__name__}(diag(Sigma_hat))")
    print("Saved: mmmd_pathmnist_results.csv, mmmd_pathmnist_per_iter.csv")
