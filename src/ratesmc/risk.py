"""Value-at-Risk and Expected Shortfall for a SOFR futures position.

Position: long 1 SR3 (3-month SOFR) futures contract whose reference quarter starts
~6 months out. Price = 100 - rate, so the position gains $25 for every 1bp fall in
the futures rate.

Under Vasicek the futures rate (ignoring convexity) is the average expected short rate
over the reference quarter, and it depends linearly on today's short rate:
    F(t) = theta + (r(t) - theta) * L,  L = average of exp(-kappa (s - t)) over the quarter.
So a 1-month change in F is L x (change in r). We simulate r over one month and map it.

VaR(q)  = loss not exceeded with probability q (a quantile of the loss distribution).
ES(q)   = average loss in the worst (1 - q) of cases. ES looks into the tail, VaR does not.
"""
from __future__ import annotations

import numpy as np


def loading(kappa: float, t: float, start: float, length: float = 0.25) -> float:
    s = np.linspace(start, start + length, 200)
    return float(np.mean(np.exp(-kappa * (s - t))))


def var_es(pnl: np.ndarray, levels=(0.95, 0.99)) -> dict:
    loss = -np.asarray(pnl)
    out = {}
    for q in levels:
        v = np.quantile(loss, q)
        out[f"VaR{int(q*100)}"] = float(v)
        out[f"ES{int(q*100)}"] = float(loss[loss >= v].mean())
    return out
