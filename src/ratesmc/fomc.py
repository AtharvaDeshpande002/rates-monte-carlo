"""FOMC step model: the policy rate moves only on meeting days, in 25bp steps.

Real short rates do not diffuse smoothly like an OU process. They sit still and jump at
FOMC meetings. This module reads the market's expected path from today's curve and
turns it into meeting-by-meeting move probabilities, the way a STIR desk quotes
"x% chance of a 25bp hike in December".

Method
  1. Expected policy rate between meetings = average instantaneous forward rate
     between those meeting dates (from the curve).
  2. Expected change at meeting m = E[rate after m] - E[rate after m-1].
  3. Split that expected change between the two neighbouring 25bp outcomes, e.g.
     +10bp expected -> 40% chance of +25bp, 60% chance of no change.
This "two-outcome" split is the minimum-variance distribution consistent with the
curve, so it shows the market's central path but understates tail risk. The report
contrasts it with the history-calibrated Vasicek model, which has realistic dispersion
but ignores the market's view.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def year_frac(as_of: str, dates) -> np.ndarray:
    d0 = pd.Timestamp(as_of)
    return np.array([(pd.Timestamp(d) - d0).days / 365.25 for d in dates])


def implied_moves(hw, as_of: str, meetings: list[str], horizon: float, step_bp: float = 25):
    """Expected rate after each meeting and the two-outcome move probabilities."""
    tm = year_frac(as_of, meetings)
    tm = tm[(tm > 0) & (tm < horizon)]
    edges = np.r_[tm, horizon]
    after = []
    for a, b in zip(edges[:-1], edges[1:]):
        grid = np.linspace(a, b, 50)
        after.append(float(np.mean(hw.f0(grid))))
    before0 = float(np.mean(hw.f0(np.linspace(0, tm[0], 50)))) if tm[0] > 0 else float(hw.f0(0))
    exp_rates = np.array(after)
    exp_change = np.diff(np.r_[before0, exp_rates]) * 1e4  # bp
    step = step_bp
    lower = np.floor(exp_change / step) * step
    p_upper = (exp_change - lower) / step
    rows = []
    for i, t in enumerate(tm):
        rows.append(dict(meeting=meetings[i] if i < len(meetings) else None, t=t,
                         exp_change_bp=exp_change[i], lower_bp=lower[i],
                         upper_bp=lower[i] + step, p_upper=p_upper[i],
                         exp_rate_after=exp_rates[i]))
    return pd.DataFrame(rows), before0


def simulate_steps(moves: pd.DataFrame, r_start: float, horizon: float, n_steps: int,
                   n_paths: int, noise_bp: float, rng, store_every: int):
    """Simulate policy-rate paths: independent two-outcome moves at each meeting,
    plus small daily noise (SOFR prints around the policy rate)."""
    dt = horizon / n_steps
    meet_steps = np.round(moves["t"].to_numpy() / dt).astype(int)
    policy = np.full(n_paths, r_start)
    times, snaps = [0.0], [policy.copy()]
    mi = 0
    for k in range(1, n_steps + 1):
        while mi < len(meet_steps) and meet_steps[mi] == k:
            row = moves.iloc[mi]
            up = rng.random(n_paths) < row["p_upper"]
            policy = policy + np.where(up, row["upper_bp"], row["lower_bp"]) / 1e4
            mi += 1
        if k % store_every == 0:
            times.append(k * dt)
            snaps.append(policy + rng.standard_normal(n_paths) * noise_bp / 1e4)
    return dict(t=np.array(times), r=np.array(snaps))
