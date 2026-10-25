"""Correctness checks: closed forms vs simulation, estimator identities, curve fit."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ratesmc.calibration import mle_vasicek, ols_ar1  # noqa: E402
from ratesmc.data import clean_spikes  # noqa: E402
from ratesmc.fomc import implied_moves  # noqa: E402
from ratesmc.models import (HullWhite, ou_mean, ou_var, vasicek_bond,  # noqa: E402
                            vasicek_integral_mean)
from ratesmc.montecarlo import (bond_estimators, ou_euler_weak_bias,  # noqa: E402
                                simulate_hw, simulate_ou)
from ratesmc.risk import var_es  # noqa: E402

K, TH, SG, R0 = 0.5, 0.04, 0.01, 0.03


def test_exact_ou_moments_match_simulation():
    rng = np.random.default_rng(0)
    s = simulate_ou(R0, K, TH, SG, 2.0, 24, 200_000, rng)
    rT = s["r"][-1]
    assert abs(rT.mean() - ou_mean(R0, K, TH, 2.0)) < 4 * rT.std() / np.sqrt(rT.size)
    assert abs(rT.var() / ou_var(K, SG, 2.0) - 1) < 0.02


def test_bond_price_estimators_agree_with_closed_form():
    rng = np.random.default_rng(1)
    exact = float(vasicek_bond(R0, K, TH, SG, 2.0))
    for m, (est, se) in bond_estimators(R0, K, TH, SG, 2.0, 20_000, 104, rng).items():
        assert abs(est - exact) < 5 * se + 2e-6, m   # 2e-6: trapezoid discretisation


def test_control_variate_mean_is_exact():
    rng = np.random.default_rng(2)
    s = simulate_ou(R0, K, TH, SG, 1.0, 252, 100_000, rng)
    m = vasicek_integral_mean(R0, K, TH, 1.0)
    assert abs(s["integral"].mean() - m) < 4 * s["integral"].std() / np.sqrt(1e5)


def test_ols_equals_mle_for_vasicek():
    rng = np.random.default_rng(3)
    s = simulate_ou(R0, 2.0, TH, SG, 8.0, 2016, 1, rng, store_every=1)
    x = s["r"][:, 0]
    o, m = ols_ar1(x, 8.0 / 2016), mle_vasicek(x, 8.0 / 2016)
    assert abs(o["kappa"] - m["kappa"]) < 1e-3
    # sigma differs only by the degrees-of-freedom correction (OLS divides by n-2, MLE by n)
    n = len(x) - 1
    assert o["sigma"] / m["sigma"] == pytest.approx(np.sqrt(n / (n - 2)), rel=1e-4)


def test_euler_bias_is_first_order():
    rows = ou_euler_weak_bias(R0, 2.0, TH, SG, 2.0, [12, 24, 48, 96])
    b = np.abs([r["mean_bias_bp"] for r in rows])
    ratios = b[:-1] / b[1:]
    assert np.all((ratios > 1.8) & (ratios < 2.2))   # halving dt halves the bias


def _curve():
    t = np.array([0, 0.25, 0.5, 1, 2])
    z = np.array([0.039, 0.040, 0.041, 0.043, 0.045])
    from scipy.interpolate import PchipInterpolator
    g = np.linspace(0, 2, 241)
    return g, PchipInterpolator(t, z)(g)


def test_hull_white_reprices_curve_by_simulation():
    g, z = _curve()
    hw = HullWhite(g, z, 0.15, 0.01)
    h = simulate_hw(hw, 1.0, 252, 100_000, np.random.default_rng(4), antithetic=True)
    P_mc = np.exp(-h["integral"]).mean()
    assert abs(P_mc / hw.P0(1.0) - 1) < 2e-4


def test_caplet_monte_carlo_matches_closed_form():
    g, z = _curve()
    hw = HullWhite(g, z, 0.15, 0.01)
    T, tau, K_ = 1.0, 0.25, 0.045
    h = simulate_hw(hw, T, 252, 200_000, np.random.default_rng(5))
    P = hw.bond(T, T + tau, h["r"])
    pay = np.exp(-h["integral"]) * P * tau * np.maximum((1 / P - 1) / tau - K_, 0)
    se = pay.std() / np.sqrt(pay.size)
    assert abs(pay.mean() - hw.caplet(T, tau, K_)) < 4 * se


def test_convexity_positive_and_increasing():
    g, z = _curve()
    hw = HullWhite(g, z, 0.15, 0.01)
    ca = [hw.convexity_adjustment(t, t + 0.25) for t in [0.25, 0.5, 1.0, 1.5]]
    assert all(c > 0 for c in ca) and np.all(np.diff(ca) > 0)


def test_fomc_probabilities_reproduce_expected_change():
    g, z = _curve()
    hw = HullWhite(g, z, 0.15, 0.01)
    mv, _ = implied_moves(hw, "2026-10-22", ["2026-12-09", "2027-01-27", "2027-03-17"], 1.0)
    implied = mv.lower_bp + mv.p_upper * (mv.upper_bp - mv.lower_bp)
    assert np.allclose(implied, mv.exp_change_bp)
    assert ((mv.p_upper >= 0) & (mv.p_upper <= 1)).all()


def test_spike_filter_removes_reversing_spike_only():
    import pandas as pd
    s = pd.Series([0.02, 0.02, 0.05, 0.02, 0.02, 0.0275, 0.0275],
                  index=pd.date_range("2024-01-01", periods=7))
    c, m = clean_spikes(s, 50)
    assert m.sum() == 1 and m.iloc[2]            # one-day spike removed
    assert c.iloc[-1] == pytest.approx(0.0275)    # a step (FOMC-like) move is kept


def test_es_exceeds_var():
    pnl = np.random.default_rng(6).standard_t(4, 100_000)
    r = var_es(pnl)
    assert r["ES99"] > r["VaR99"] > r["VaR95"] > 0
