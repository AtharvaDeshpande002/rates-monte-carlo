"""ratesmc: Monte Carlo simulation of short-term US interest-rate paths.

Modules
-------
data         load FRED data and the as-of Treasury curve
models       Vasicek (Ornstein-Uhlenbeck), CIR and Hull-White short-rate models
calibration  maximum likelihood and OLS calibration, rolling estimates
montecarlo   path simulation, convergence and variance reduction
pricing      bond prices, caplet pricing, futures convexity adjustment
fomc         FOMC step-jump model with market-implied move probabilities
risk         Value-at-Risk and Expected Shortfall
plots        matplotlib figures and Plotly specs for the HTML report
"""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path | None = None) -> dict:
    """Read config.yaml from the repo root (all parameters live there)."""
    path = path or ROOT / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)
