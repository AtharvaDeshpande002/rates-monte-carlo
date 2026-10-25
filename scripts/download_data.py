"""Download the public input data from FRED into data/raw/.

    python scripts/download_data.py            # latest data
    python scripts/download_data.py 2026-10-22 # curve on a specific date

Series: SOFR, EFFR (overnight rates) and the Treasury constant-maturity yields
DGS1MO, DGS3MO, DGS6MO, DGS1, DGS2 for the curve on the as-of date.
"""
from __future__ import annotations

import io
import sys
import urllib.request
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
CURVE = [("DGS1MO", 1 / 12), ("DGS3MO", 0.25), ("DGS6MO", 0.5), ("DGS1", 1.0), ("DGS2", 2.0)]


def fetch(series: str) -> pd.DataFrame:
    with urllib.request.urlopen(URL.format(series), timeout=60) as r:
        df = pd.read_csv(io.BytesIO(r.read()))
    df.columns = ["observation_date", series]
    return df


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for s, start in [("SOFR", "2018-04-01"), ("EFFR", "2015-01-01")]:
        df = fetch(s)
        df = df[df.observation_date >= start]
        df.to_csv(RAW / f"{s}.csv", index=False)
        print(s, len(df), "rows")
    frames = [fetch(s).set_index("observation_date") for s, _ in CURVE]
    curve = pd.concat(frames, axis=1).apply(pd.to_numeric, errors="coerce").dropna()
    as_of = sys.argv[1] if len(sys.argv) > 1 else curve.index[-1]
    row = curve.loc[as_of]
    out = pd.DataFrame(dict(tenor_years=[t for _, t in CURVE], series=[s for s, _ in CURVE],
                            yield_pct=[row[s] for s, _ in CURVE]))
    out.to_csv(RAW / f"treasury_curve_{as_of}.csv", index=False)
    print("curve as of", as_of, "-> update data.as_of and data.curve_file in config.yaml")


if __name__ == "__main__":
    main()
