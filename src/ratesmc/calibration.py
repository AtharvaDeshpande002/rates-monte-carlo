"""Calibrate short-rate models to history.

Two estimators for Vasicek and why they agree:
  * The exact OU transition is r(t+dt) = theta(1-b) + b r(t) + e, e ~ N(0, v),
    with b = exp(-kappa dt) and v = sigma^2 (1 - b^2) / (2 kappa).
  * That is an AR(1) regression, so ordinary least squares gives the same point
    estimates as conditional maximum likelihood. We still write out the likelihood
    because it gives standard errors (from the Hessian) and generalises to CIR,
    whose transition density (non-central chi-square) is not Gaussian.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats


def ols_ar1(x: np.ndarray, dt: float) -> dict:
    """OLS on r(t+1) = c + b r(t) + e, mapped back to (kappa, theta, sigma)."""
    y, X = x[1:], x[:-1]
    b, c = np.polyfit(X, y, 1)
    resid = y - (c + b * X)
    s2 = resid.var(ddof=2)
    if b >= 1.0:  # no mean reversion detected in this sample (unit root)
        return dict(kappa=np.nan, theta=np.nan, sigma=np.sqrt(s2 / dt), b=b)
    kappa = -np.log(b) / dt
    theta = c / (1.0 - b)
    sigma = np.sqrt(s2 * 2 * kappa / (1.0 - b * b))
    return dict(kappa=kappa, theta=theta, sigma=sigma, b=b)


def _vasicek_nll(p, x, dt):
    kappa, theta, sigma = p
    if kappa <= 0 or sigma <= 0:
        return 1e12
    b = np.exp(-kappa * dt)
    v = sigma ** 2 * (1 - b * b) / (2 * kappa)
    m = theta + (x[:-1] - theta) * b
    return -stats.norm.logpdf(x[1:], m, np.sqrt(v)).sum()


def _num_hessian(f, p, rel=1e-4):
    """Central-difference Hessian (used for standard errors)."""
    p = np.asarray(p, float)
    n = len(p)
    h = np.maximum(np.abs(p) * rel, 1e-8)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            pp = p.copy(); pp[i] += h[i]; pp[j] += h[j]; fpp = f(pp)
            pm = p.copy(); pm[i] += h[i]; pm[j] -= h[j]; fpm = f(pm)
            mp = p.copy(); mp[i] -= h[i]; mp[j] += h[j]; fmp = f(mp)
            mm = p.copy(); mm[i] -= h[i]; mm[j] -= h[j]; fmm = f(mm)
            H[i, j] = (fpp - fpm - fmp + fmm) / (4 * h[i] * h[j])
    return H


def mle_vasicek(x: np.ndarray, dt: float) -> dict:
    """Exact-likelihood MLE for Vasicek, with standard errors from the inverse Hessian."""
    start = ols_ar1(x, dt)
    p0 = [start["kappa"] if np.isfinite(start["kappa"]) else 0.1,
          start["theta"] if np.isfinite(start["theta"]) else float(np.mean(x)),
          start["sigma"]]
    res = optimize.minimize(_vasicek_nll, p0, args=(x, dt), method="Nelder-Mead",
                            options=dict(xatol=1e-10, fatol=1e-8, maxiter=20000))
    p = res.x
    H = _num_hessian(lambda q: _vasicek_nll(q, x, dt), p)
    try:
        se = np.sqrt(np.diag(np.linalg.inv(H)))
    except np.linalg.LinAlgError:
        se = np.full(3, np.nan)
    return dict(kappa=p[0], theta=p[1], sigma=p[2], se_kappa=se[0], se_theta=se[1],
                se_sigma=se[2], loglik=-res.fun, n=len(x) - 1)


def _cir_nll(p, x, dt):
    kappa, theta, sigma = p
    if kappa <= 0 or theta <= 0 or sigma <= 0:
        return 1e12
    c = 2 * kappa / (sigma ** 2 * (1 - np.exp(-kappa * dt)))
    df = 4 * kappa * theta / sigma ** 2
    nc = 2 * c * x[:-1] * np.exp(-kappa * dt)
    # 2 c r(t+dt) ~ non-central chi-square(df, nc); change of variables adds log(2c)
    ll = stats.ncx2.logpdf(2 * c * x[1:], df, nc) + np.log(2 * c)
    if not np.all(np.isfinite(ll)):
        return 1e12
    return -ll.sum()


def mle_cir(x: np.ndarray, dt: float, start: dict) -> dict:
    """Exact MLE for CIR using the non-central chi-square transition density."""
    x = np.maximum(x, 1e-4)  # CIR needs positive rates; floor 1bp (zero-rate era)
    p0 = [max(start["kappa"], 0.05), max(start["theta"], 0.005),
          start["sigma"] / np.sqrt(max(np.mean(x), 1e-3))]
    res = optimize.minimize(_cir_nll, p0, args=(x, dt), method="Nelder-Mead",
                            options=dict(xatol=1e-10, fatol=1e-8, maxiter=20000))
    p = res.x
    return dict(kappa=p[0], theta=p[1], sigma=p[2], loglik=-res.fun, n=len(x) - 1)


def rolling_vasicek(s: pd.Series, window: int, step: int, dt: float) -> pd.DataFrame:
    """Rolling 2-year Vasicek estimates with standard errors (shows regime changes)."""
    rows = []
    x = s.to_numpy()
    for end in range(window, len(x) + 1, step):
        w = x[end - window:end]
        est = mle_vasicek(w, dt)
        rows.append(dict(date=s.index[end - 1], **est))
    return pd.DataFrame(rows).set_index("date")
