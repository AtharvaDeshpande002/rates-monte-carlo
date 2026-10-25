"""Load the public data used in the project.

Everything here is public: FRED series SOFR, EFFR and the constant-maturity
Treasury yields. Run scripts/download_data.py to fetch fresh copies.
Rates are converted from percent to decimals (4.25% -> 0.0425).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import ROOT


def load_series(cfg: dict, name: str) -> pd.Series:
    """Daily FRED series as a pandas Series of decimals, indexed by date."""
    fname = cfg["data"][f"{name.lower()}_file"]
    df = pd.read_csv(ROOT / cfg["data"]["raw_dir"] / fname, parse_dates=["observation_date"])
    s = pd.to_numeric(df[name.upper()], errors="coerce").dropna() / 100.0
    s.index = df.loc[s.index, "observation_date"]
    s.name = name.upper()
    return s.sort_index()


def clean_spikes(s: pd.Series, winsor_bp: float) -> tuple[pd.Series, pd.Series]:
    """Remove one-day spikes that reverse the next day (repo stress, month-ends).

    SOFR jumped ~280bp on 17-Sep-2019 and reversed within days; month-end prints
    also spike. These are funding-market events, not changes in rate expectations,
    and they would dominate a Gaussian volatility estimate. A day is flagged when
    it moves more than `winsor_bp` AND the next day moves back by more than half of it.
    FOMC moves (which do not reverse) are kept.
    Returns the cleaned series and a boolean mask of the flagged days.
    """
    x = s.copy()
    d = x.diff()
    nxt = d.shift(-1)
    thr = winsor_bp / 1e4
    spike = (d.abs() > thr) & (np.sign(nxt) == -np.sign(d)) & (nxt.abs() > 0.5 * d.abs())
    x[spike] = np.nan
    x = x.interpolate(limit_direction="both")
    return x, spike


def load_curve(cfg: dict) -> pd.DataFrame:
    """As-of Treasury constant-maturity yields (decimals) by tenor in years."""
    df = pd.read_csv(ROOT / cfg["data"]["raw_dir"] / cfg["data"]["curve_file"])
    df["yield"] = df["yield_pct"] / 100.0
    return df[["tenor_years", "series", "yield"]]


def policy_curve(cfg: dict, r0: float) -> tuple[np.ndarray, np.ndarray]:
    """A smooth zero-rate curve for the expected path of the overnight rate.

    Treasury bills trade at a spread to SOFR (supply, tax and collateral effects),
    so the Treasury curve is shifted in parallel so its short end equals today's
    SOFR. What remains is the SHAPE of the curve, which is what tells us how much
    hiking or cutting the market prices. This is an approximation; with SR3
    settlement prices one would build the curve directly from futures.
    Returns (tenors, zero rates) on a fine grid, continuously compounded.
    """
    c = load_curve(cfg)
    t = np.r_[0.0, c["tenor_years"].to_numpy()]
    y = c["yield"].to_numpy()
    y = y - y[0] + r0                     # parallel shift: 1M bill -> SOFR
    y = np.r_[r0, y]
    grid = np.linspace(0, t[-1], 241)
    # monotone-preserving piecewise cubic so forwards stay well behaved
    from scipy.interpolate import PchipInterpolator
    z = PchipInterpolator(t, y)(grid)
    return grid, z
