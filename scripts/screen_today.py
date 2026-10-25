"""Desk screen for the latest date in the data.

    python scripts/screen_today.py

Prints
  1. Market-implied move probabilities for the next FOMC meetings
  2. Scenario probabilities: history-calibrated Vasicek vs market-consistent step model
  3. Convexity adjustment for the next 12 quarterly rate-futures contracts
  4. 1-month VaR / Expected Shortfall for long 1 SR3 contract
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ratesmc import ROOT, load_config  # noqa: E402
from ratesmc.calibration import mle_vasicek  # noqa: E402
from ratesmc.data import clean_spikes, load_series, policy_curve  # noqa: E402
from ratesmc.fomc import implied_moves  # noqa: E402
from ratesmc.models import HullWhite  # noqa: E402


def main():
    cfg = load_config()
    dt = cfg["calibration"]["dt_years"]
    sofr = load_series(cfg, "SOFR")
    x, _ = clean_spikes(sofr, cfg["calibration"]["winsor_bp"])
    r0 = float(sofr.iloc[-1])
    v = mle_vasicek(x.to_numpy(), dt)
    tg, zg = policy_curve(cfg, r0)
    hw = HullWhite(tg, zg, cfg["pricing"]["hw_a"], v["sigma"])
    moves, _ = implied_moves(hw, cfg["data"]["as_of"], cfg["fomc"]["meetings"], 2.0)

    pd.set_option("display.width", 120)
    print(f"\nAs of {cfg['data']['as_of']}   SOFR {r0*100:.2f}%   "
          f"Vasicek: kappa {v['kappa']:.2f} (se {v['se_kappa']:.2f}), theta {v['theta']*100:.2f}%, "
          f"sigma {v['sigma']*1e4:.0f}bp/sqrt(yr)\n")

    print("1. Market-implied FOMC moves (two-outcome split of the curve's expected change)")
    m = moves.head(6).copy()
    m["move"] = m.apply(lambda r: f"{r.p_upper*100:4.0f}% {r.upper_bp:+.0f}bp / "
                                  f"{(1-r.p_upper)*100:4.0f}% {r.lower_bp:+.0f}bp", axis=1)
    m["exp_rate_after_%"] = (m["exp_rate_after"] * 100).round(3)
    print(m[["meeting", "exp_change_bp", "move", "exp_rate_after_%"]].round(1).to_string(index=False))

    tab = ROOT / "reports" / "tables" / "scenario_probabilities.csv"
    if tab.exists():
        print("\n2. Scenario probabilities (from the last run_all.py)")
        print(pd.read_csv(tab).round(3).to_string(index=False))

    print("\n3. Convexity adjustment, futures minus forward (bp), quarterly contracts")
    rows = [(f"{t:.2f}y", hw.convexity_adjustment(t, t + 0.25) * 1e4) for t in np.arange(0.25, 3.01, 0.25)]
    print(pd.DataFrame(rows, columns=["start", "convexity_bp"]).round(2).to_string(index=False))

    tab = ROOT / "reports" / "tables" / "var_es.csv"
    if tab.exists():
        print("\n4. One-month VaR / ES, long 1 SR3 (USD)")
        print(pd.read_csv(tab).round(0).to_string(index=False))


if __name__ == "__main__":
    main()
