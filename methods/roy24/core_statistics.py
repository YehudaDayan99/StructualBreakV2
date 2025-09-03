"""
Core statistical functions for ADIA structural breakpoint detection.

This module contains fundamental statistical functions including:
- Kernel smoothing (Epanechnikov kernel)
- Grid generation for nonparametric estimation
- Robust standardization
- Hypothesis testing (t-tests, Mann-Whitney U)
- Energy distance calculations
- Autocorrelation functions
"""

import numpy as np
from math import erfc
from typing import Tuple
from .config import GRID_MIN_POINTS, GRID_MAX_POINTS, EPSILON


def epanechnikov_kernel(u: np.ndarray) -> np.ndarray:
    w = np.maximum(0.0, 1.0 - u*u)
    return 0.75 * w


def make_grid(X: np.ndarray, h: float, min_pts: int = None, max_pts: int = None) -> np.ndarray:
    if min_pts is None:
        min_pts = GRID_MIN_POINTS
    if max_pts is None:
        max_pts = GRID_MAX_POINTS
    x_min = float(np.min(X))
    x_max = float(np.max(X))
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        return np.linspace(-1.0, 1.0, min_pts)
    if x_max <= x_min + EPSILON:
        eps = max(2*h, 1e-6)
        return np.linspace(x_min - eps, x_min + eps, min_pts)
    step = max(2*h, 1e-6)
    grid_pts = int(np.ceil((x_max - x_min) / step)) + 1
    grid_pts = int(np.clip(grid_pts, min_pts, max_pts))
    return np.linspace(x_min, x_max, grid_pts)


def nw_mean(x_grid: np.ndarray, X: np.ndarray, Y: np.ndarray, h: float) -> Tuple[np.ndarray, np.ndarray]:
    U = (X[:, None] - x_grid[None, :]) / h
    W = epanechnikov_kernel(U)
    denom = W.sum(axis=0) + EPSILON
    mu = (W * Y[:, None]).sum(axis=0) / denom
    fhat = denom / (len(X) * h)
    return mu, fhat


def jackknife_mu(x_grid: np.ndarray, X: np.ndarray, Y: np.ndarray, h: float) -> Tuple[np.ndarray, np.ndarray]:
    mu_h, fhat = nw_mean(x_grid, X, Y, h)
    mu_h2, _ = nw_mean(x_grid, X, Y, h/np.sqrt(2))
    mu_star = 2*mu_h - mu_h2
    return mu_star, fhat


def nw_variance(x_grid: np.ndarray, X: np.ndarray, Y: np.ndarray, mu_grid: np.ndarray, h: float) -> np.ndarray:
    U = (X[:, None] - x_grid[None, :]) / h
    W = epanechnikov_kernel(U)
    resid2 = (Y[:, None] - mu_grid[None, :])**2
    denom = W.sum(axis=0) + EPSILON
    sig2 = (W * resid2).sum(axis=0) / denom
    return sig2


def robust_standardize(y: np.ndarray) -> np.ndarray:
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    scale = (mad * 1.4826) if mad > 0 else (np.std(y) + EPSILON)
    return (y - med) / (scale + EPSILON)


def mean_variance_test(y0: np.ndarray, y1: np.ndarray) -> Tuple[float, float]:
    n0, n1 = len(y0), len(y1)
    m0, m1 = float(np.mean(y0)), float(np.mean(y1))
    v0, v1 = float(np.var(y0, ddof=1)), float(np.var(y1, ddof=1))
    se = np.sqrt(v0/n0 + v1/n1)
    z_mean = 0.0 if se == 0 else abs(m0 - m1) / se
    p_mean = erfc(z_mean / np.sqrt(2))
    se_var = np.sqrt(2*(v0**2)/(max(n0-1,1)) + 2*(v1**2)/(max(n1-1,1)))
    z_var = 0.0 if se_var == 0 else abs(v0 - v1) / se_var
    p_var = erfc(z_var / np.sqrt(2))
    return p_mean, p_var


def ranks_with_ties(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x)+1)
    i = 0
    while i < len(x):
        j = i
        while j+1 < len(x) and x[order[j+1]] == x[order[i]]:
            j += 1
        if j > i:
            avg = 0.5*(ranks[order[i]] + ranks[order[j]])
            ranks[order[i:j+1]] = avg
        i = j+1
    return ranks


def tie_counts(ranks: np.ndarray) -> np.ndarray:
    r = np.round(ranks, 12)
    _, counts = np.unique(r, return_counts=True)
    return counts[counts > 1]


def mann_whitney_p(y0: np.ndarray, y1: np.ndarray) -> float:
    x = np.concatenate([y0, y1])
    n0, n1 = len(y0), len(y1)
    ranks = ranks_with_ties(x)
    R0 = np.sum(ranks[:n0])
    U0 = R0 - n0*(n0+1)/2
    muU = n0*n1/2
    tie_counts_array = tie_counts(ranks)
    n = n0 + n1
    tie_term = np.sum(t*(t*t - 1) for t in tie_counts_array)
    varU = n0*n1*(n0+n1+1 - tie_term/(n*(n-1))) / 12.0
    z = 0.0 if varU <= 0 else abs(U0 - muU) / np.sqrt(varU)
    return float(erfc(z / np.sqrt(2)))


def energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    def avg_abs_diff(a, b):
        return np.mean(np.abs(a[:, None] - b[None, :]))
    xy = avg_abs_diff(x, y)
    xx = avg_abs_diff(x, x)
    yy = avg_abs_diff(y, y)
    return 2*xy - xx - yy


def permutation_pvalue(stat_obs: float, x: np.ndarray, y: np.ndarray, stat_fn, B: int = 80, seed: int = 123) -> float:
    rng = np.random.default_rng(seed)
    z = np.concatenate([x, y])
    n = len(x)
    count = 0
    for b in range(B):
        rng.shuffle(z)
        xb = z[:n]
        yb = z[n:]
        if stat_fn(xb, yb) >= stat_obs - EPSILON:
            count += 1
    return (1 + count) / (B + 1)


def autocorrelation(series: np.ndarray, lag: int) -> float:
    if lag <= 0 or lag >= len(series):
        return 0.0
    s = series - np.mean(series)
    num = np.dot(s[lag:], s[:-lag])
    den = np.dot(s, s) + EPSILON
    return float(num/den)
