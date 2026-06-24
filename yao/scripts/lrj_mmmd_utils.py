#!/usr/bin/env python3
"""Generic GEXP-5 and NEW-MMMD utilities adapted from lrj's PathMNIST work."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform


RIDGE_FACTOR = 1e-5
RIDGE_AGG = np.min


def sqdist_mat(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    if Y is None:
        return squareform(pdist(X)) ** 2
    return cdist(X, Y, "sqeuclidean")


def t_med_pooled(X: np.ndarray, Y: np.ndarray) -> float:
    Z = np.vstack([X, Y])
    d = pdist(Z) ** 2
    med = float(np.median(d))
    if not np.isfinite(med) or med <= 0:
        raise ValueError("Median squared pairwise distance is not positive.")
    return 1.0 / med


def mmd_offdiag(K_X: np.ndarray, K_Y: np.ndarray, K_XY: np.ndarray) -> float:
    m = K_X.shape[0]
    a = (K_X.sum() - np.trace(K_X)) / (m * (m - 1))
    b = (K_Y.sum() - np.trace(K_Y)) / (m * (m - 1))
    c = K_XY.mean()
    return float(a + b - 2 * c)


def est_cov_naive(X: np.ndarray, t_vec: np.ndarray, D2_XX: np.ndarray | None = None) -> np.ndarray:
    m = X.shape[0]
    if D2_XX is None:
        D2_XX = sqdist_mat(X)
    C = np.eye(m) - np.ones((m, m)) / m
    Kc_list = [C @ np.exp(-t * D2_XX) @ C for t in t_vec]
    R = len(t_vec)
    S = np.zeros((R, R), dtype=np.float64)
    for i in range(R):
        for j in range(R):
            S[i, j] = (8.0 / m**2) * np.trace(Kc_list[i] @ Kc_list[j])
    return S


def fast_cov_gaussian_t(X: np.ndarray, t_vec: np.ndarray, D2_XX: np.ndarray | None = None) -> np.ndarray:
    m = X.shape[0]
    R = len(t_vec)
    if D2_XX is None:
        D2_XX = sqdist_mat(X)

    s_mat = np.zeros((m, R), dtype=np.float64)
    S_vec = np.zeros(R, dtype=np.float64)
    for a in range(R):
        Ka = np.exp(-t_vec[a] * D2_XX)
        s_mat[:, a] = Ka.sum(axis=1)
        S_vec[a] = Ka.sum()

    t_sum = np.add.outer(t_vec, t_vec)
    t_sum_round = np.round(t_sum, 10)
    uniq, inv_idx = np.unique(t_sum_round, return_inverse=True)
    H_unique = np.array([np.exp(-u * D2_XX).sum() for u in uniq], dtype=np.float64)
    H_mat = H_unique[inv_idx].reshape(R, R)

    G_mat = s_mat.T @ s_mat
    inner = H_mat - (2.0 / m) * G_mat + (1.0 / m**2) * np.outer(S_vec, S_vec)
    Sigma = (8.0 / m**2) * inner
    return 0.5 * (Sigma + Sigma.T)


def pivoted_chol_select(Sigma: np.ndarray, eps: float = 1e-4, q_max: int | None = None) -> tuple[np.ndarray, int]:
    R = Sigma.shape[0]
    if q_max is None:
        q_max = R
    diag_S = np.diag(Sigma).copy()
    resid = diag_S.copy()
    L = np.zeros((R, q_max), dtype=np.float64)
    selected: list[int] = []
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
    return np.asarray(selected, dtype=int), len(selected)


def boot_cutoff(X: np.ndarray, t_vec_sel: np.ndarray, Sigma_inv: np.ndarray, n_boot: int, rng: np.random.Generator, D2_XX: np.ndarray | None = None) -> float:
    m = X.shape[0]
    if D2_XX is None:
        D2_XX = sqdist_mat(X)
    C = np.eye(m) - np.ones((m, m)) / m
    Kc_list = [(1.0 / m) * (C @ np.exp(-t * D2_XX) @ C) for t in t_vec_sel]
    U = rng.normal(0.0, np.sqrt(2.0), size=(n_boot, m))
    T_mat = np.zeros((n_boot, len(t_vec_sel)), dtype=np.float64)
    for a, Kc in enumerate(Kc_list):
        Ku = U @ Kc
        T_mat[:, a] = np.einsum("ij,ij->i", U, Ku) - 2.0 * np.trace(Kc)
    ts = np.einsum("ij,jk,ik->i", T_mat, Sigma_inv, T_mat)
    return float(np.quantile(ts, 0.95))


def ridge_lambda(Sigma_raw: np.ndarray) -> float:
    return float(RIDGE_FACTOR * RIDGE_AGG(np.diag(Sigma_raw)))


def safe_cond(M: np.ndarray) -> float:
    try:
        return float(np.linalg.cond(M))
    except Exception:
        return np.inf


def gexp5_one(X: np.ndarray, Y: np.ndarray, n_boot: int, rng: np.random.Generator) -> dict[str, float | int]:
    m = X.shape[0]
    t_med = t_med_pooled(X, Y)
    t_vec = np.array([(2.0**l) * t_med for l in (-2, -1, 0, 1, 2)], dtype=np.float64)
    D2_XX = sqdist_mat(X)

    Sigma_raw = est_cov_naive(X, t_vec, D2_XX=D2_XX)
    lam = ridge_lambda(Sigma_raw)
    Sigma_reg = Sigma_raw + lam * np.eye(len(t_vec))
    cond_raw = safe_cond(Sigma_raw)
    cond_reg = safe_cond(Sigma_reg)

    D2_YY = sqdist_mat(Y)
    D2_XY = sqdist_mat(X, Y)
    mmd_vec = np.empty(len(t_vec), dtype=np.float64)
    for a, t in enumerate(t_vec):
        K_X = np.exp(-t * D2_XX)
        K_Y = np.exp(-t * D2_YY)
        K_XY = np.exp(-t * D2_XY)
        mmd_vec[a] = mmd_offdiag(K_X, K_Y, K_XY)

    Sigma_inv = np.linalg.inv(Sigma_reg)
    Tn = m * mmd_vec
    T_stat = float(Tn @ Sigma_inv @ Tn)
    thr = boot_cutoff(X, t_vec, Sigma_inv, n_boot, rng, D2_XX=D2_XX)
    return {
        "reject": int(T_stat > thr),
        "q": int(len(t_vec)),
        "cond_raw": cond_raw,
        "cond_reg": cond_reg,
        "ridge_lambda": lam,
        "stat": T_stat,
        "cutoff": thr,
    }


def new_one(X: np.ndarray, Y: np.ndarray, n_boot: int, rng: np.random.Generator, R: int = 13, eps: float = 1e-4) -> dict[str, float | int]:
    m = X.shape[0]
    t_med = t_med_pooled(X, Y)
    t_vec = np.linspace(0.25 * t_med, 4.0 * t_med, R)
    D2_XX = sqdist_mat(X)

    Sigma_full_raw = fast_cov_gaussian_t(X, t_vec, D2_XX=D2_XX)
    lam_full = ridge_lambda(Sigma_full_raw)
    Sigma_full_reg = Sigma_full_raw + lam_full * np.eye(R)

    sel, q = pivoted_chol_select(Sigma_full_reg, eps=eps)
    if q == 0:
        return {
            "reject": 0,
            "q": 0,
            "cond_raw": np.inf,
            "cond_reg": np.inf,
            "ridge_lambda": np.nan,
            "stat": np.nan,
            "cutoff": np.nan,
        }
    Sigma_SS_raw = Sigma_full_raw[np.ix_(sel, sel)]
    lam_sub = ridge_lambda(Sigma_SS_raw)
    Sigma_SS_reg = Sigma_SS_raw + lam_sub * np.eye(q)
    cond_raw = safe_cond(Sigma_SS_raw)
    cond_reg = safe_cond(Sigma_SS_reg)

    t_sel = t_vec[sel]
    D2_YY = sqdist_mat(Y)
    D2_XY = sqdist_mat(X, Y)
    mmd_vec = np.empty(q, dtype=np.float64)
    for a, t in enumerate(t_sel):
        K_X = np.exp(-t * D2_XX)
        K_Y = np.exp(-t * D2_YY)
        K_XY = np.exp(-t * D2_XY)
        mmd_vec[a] = mmd_offdiag(K_X, K_Y, K_XY)

    Sigma_inv = np.linalg.inv(Sigma_SS_reg)
    Tn = m * mmd_vec
    T_stat = float(Tn @ Sigma_inv @ Tn)
    thr = boot_cutoff(X, t_sel, Sigma_inv, n_boot, rng, D2_XX=D2_XX)
    return {
        "reject": int(T_stat > thr),
        "q": int(q),
        "cond_raw": cond_raw,
        "cond_reg": cond_reg,
        "ridge_lambda": lam_sub,
        "stat": T_stat,
        "cutoff": thr,
    }

