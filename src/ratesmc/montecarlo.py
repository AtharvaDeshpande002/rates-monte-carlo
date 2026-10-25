"""Monte Carlo engine: path simulation, convergence and variance reduction.

Key facts the code demonstrates
  * Monte Carlo standard error falls like 1/sqrt(N): 100x more paths -> 10x more accurate.
  * Discretisation error is a separate, systematic error: it does not shrink with N,
    only with a smaller time step (Euler: weak error O(dt); Milstein: strong error O(dt)).
  * Variance reduction buys accuracy without more paths:
      - antithetic variates: pair every shock z with -z (negatively correlated estimates)
      - control variate: subtract a correlated quantity whose mean is known exactly
"""
from __future__ import annotations

import numpy as np

from .models import (cir_step_euler, cir_step_milstein, ou_step_euler, ou_step_exact,
                     vasicek_bond, vasicek_integral_mean)


def simulate_ou(r0, kappa, theta, sigma, T, n_steps, n_paths, rng, scheme="exact",
                antithetic=False, store_every=None):
    """Simulate OU short-rate paths.

    Returns dict with
      t       times of stored snapshots
      r       array (n_store, n_paths) of rates at those times
      integral  trapezoid integral of r over [0, T] for each path (for discounting)
    """
    dt = T / n_steps
    step = ou_step_exact if scheme == "exact" else ou_step_euler
    half = n_paths // 2 if antithetic else n_paths
    r = np.full(n_paths, r0, dtype=float)
    integral = np.zeros(n_paths)
    store_every = store_every or n_steps
    snaps, times = [r.copy()], [0.0]
    for k in range(1, n_steps + 1):
        z = rng.standard_normal(half)
        if antithetic:
            z = np.concatenate([z, -z])
        r_new = step(r, kappa, theta, sigma, dt, z)
        integral += 0.5 * (r + r_new) * dt
        r = r_new
        if k % store_every == 0:
            snaps.append(r.copy()); times.append(k * dt)
    return dict(t=np.array(times), r=np.array(snaps), integral=integral)


def bond_estimators(r0, kappa, theta, sigma, T, n_paths, n_steps, rng):
    """Estimate P(0,T) three ways and return (estimate, standard error) for each.

    plain       mean of exp(-integral r)
    antithetic  same with paired +z/-z shocks (SE from the pair averages)
    control     control variate Y = integral r, whose mean is known in closed form
    """
    out = {}
    s = simulate_ou(r0, kappa, theta, sigma, T, n_steps, n_paths, rng)
    X = np.exp(-s["integral"])
    out["plain"] = (X.mean(), X.std(ddof=1) / np.sqrt(n_paths))

    s = simulate_ou(r0, kappa, theta, sigma, T, n_steps, n_paths, rng, antithetic=True)
    Xa = np.exp(-s["integral"])
    h = n_paths // 2
    pairs = 0.5 * (Xa[:h] + Xa[h:])
    out["antithetic"] = (pairs.mean(), pairs.std(ddof=1) / np.sqrt(h))

    s = simulate_ou(r0, kappa, theta, sigma, T, n_steps, n_paths, rng)
    X = np.exp(-s["integral"]); Y = s["integral"]
    EY = vasicek_integral_mean(r0, kappa, theta, T)
    c = -np.cov(X, Y)[0, 1] / Y.var(ddof=1)            # optimal coefficient
    Z = X + c * (Y - EY)
    out["control"] = (Z.mean(), Z.std(ddof=1) / np.sqrt(n_paths))
    return out


def convergence_table(r0, kappa, theta, sigma, T, n_grid, n_steps, seed):
    """Error and standard error of the bond-price estimators against the closed form."""
    exact = float(vasicek_bond(r0, kappa, theta, sigma, T))
    rows = []
    for n in n_grid:
        rng = np.random.default_rng(seed + n)
        est = bond_estimators(r0, kappa, theta, sigma, T, n, n_steps, rng)
        for name, (m, se) in est.items():
            rows.append(dict(n=n, method=name, estimate=m, se=se, error=m - exact,
                             z=(m - exact) / se))
    return exact, rows


def ou_euler_weak_bias(r0, kappa, theta, sigma, T, steps_per_year_grid):
    """Exact (noise-free) bias of the Euler scheme in E[r_T] and Var[r_T].

    Euler maps r -> (1 - kappa dt) r + kappa theta dt + sigma sqrt(dt) z, so after n steps
    the mean is theta + (r0 - theta)(1 - kappa dt)^n and the variance is a geometric sum.
    Comparing with the exact OU moments isolates discretisation error with zero MC noise.
    """
    rows = []
    for m in steps_per_year_grid:
        dt = 1.0 / m
        n = int(round(T * m))
        g = 1 - kappa * dt
        mean_e = theta + (r0 - theta) * g ** n
        var_e = sigma ** 2 * dt * (1 - g ** (2 * n)) / (1 - g ** 2)
        mean_x = theta + (r0 - theta) * np.exp(-kappa * T)
        var_x = sigma ** 2 * (1 - np.exp(-2 * kappa * T)) / (2 * kappa)
        rows.append(dict(steps_per_year=m, dt=dt, mean_bias_bp=(mean_e - mean_x) * 1e4,
                         sd_bias_bp=(np.sqrt(var_e) - np.sqrt(var_x)) * 1e4))
    return rows


def cir_strong_error(r0, kappa, theta, sigma, T, n_paths, levels, rng):
    """Strong error E|r_T(coarse) - r_T(reference)| for Euler and Milstein on CIR.

    All schemes use the SAME Brownian path (fine increments summed into coarse ones),
    so the difference is pure discretisation error. Reference = Milstein on the finest grid.
    Expected slopes on a log-log plot: Euler ~ 0.5, Milstein ~ 1.
    """
    n_fine = 2 ** max(levels) * 4
    dW = rng.standard_normal((n_fine, n_paths)) * np.sqrt(T / n_fine)

    def run(stepper, n):
        dt = T / n
        agg = dW.reshape(n, n_fine // n, n_paths).sum(axis=1)
        r = np.full(n_paths, r0)
        for k in range(n):
            r = stepper(r, kappa, theta, sigma, dt, agg[k] / np.sqrt(dt))
        return r

    ref = run(cir_step_milstein, n_fine)
    rows = []
    for lv in levels:
        n = 2 ** lv
        rows.append(dict(n_steps=n, dt=T / n,
                         euler=np.mean(np.abs(run(cir_step_euler, n) - ref)),
                         milstein=np.mean(np.abs(run(cir_step_milstein, n) - ref))))
    return rows


def simulate_hw(hw, T, n_steps, n_paths, rng, store_times=(), antithetic=False):
    """Hull-White paths via the exact OU recursion for x(t); r = x + alpha(t).
    Returns rates at T, the discount integral to T, and rates at requested times."""
    dt = T / n_steps
    a, s = hw.a, hw.sigma
    x = np.zeros(n_paths)
    t = 0.0
    r_prev = np.full(n_paths, float(hw.alpha(0.0)))
    integral = np.zeros(n_paths)
    store = {}
    store_steps = {int(round(st / dt)): st for st in store_times}
    for k in range(1, n_steps + 1):
        z = rng.standard_normal(n_paths // 2 if antithetic else n_paths)
        if antithetic:
            z = np.concatenate([z, -z])
        x = ou_step_exact(x, a, 0.0, s, dt, z)
        t = k * dt
        r = x + hw.alpha(t)
        integral += 0.5 * (r_prev + r) * dt
        r_prev = r
        if k in store_steps:
            store[store_steps[k]] = r.copy()
    return dict(r=r_prev, integral=integral, store=store)
