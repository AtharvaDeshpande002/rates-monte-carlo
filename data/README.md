# Data

All inputs are public and downloaded by `python scripts/download_data.py` into `data/raw/` (not committed).

| File | Source | Columns | Units | Frequency | Range used |
|---|---|---|---|---|---|
| `SOFR.csv` | FRED `SOFR` (NY Fed) | `observation_date`, `SOFR` | percent | business daily | 2018-04-03 to 2026-10-22 |
| `EFFR.csv` | FRED `EFFR` (NY Fed) | `observation_date`, `EFFR` | percent | business daily | 2015-01-02 to 2026-10-22 |
| `treasury_curve_<date>.csv` | FRED `DGS1MO`, `DGS3MO`, `DGS6MO`, `DGS1`, `DGS2` | `tenor_years`, `series`, `yield_pct` | percent (CMT, bond-equivalent) | one date | 2026-10-22 |

Set `data.as_of` and `data.curve_file` in `config.yaml` to match the curve file.
The curve used in the published results (22-Oct-2026): 1M 4.11%, 3M 4.22%, 6M 4.29%, 1Y 4.41%, 2Y 4.65%.
