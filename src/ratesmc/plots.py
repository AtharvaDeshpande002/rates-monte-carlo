"""Charts: static PNGs (matplotlib) for the README and Plotly specs for the HTML report.

Every chart title states the finding, axes carry units, and each figure ends with a
source note. The same data feeds both the PNG and the interactive version.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK = "#1d2433"
MUTED = "#6b7280"
BLUE = "#2563eb"
ORANGE = "#ea580c"
GREEN = "#059669"
GREY = "#9ca3af"
SOURCE = "Source: FRED (SOFR, EFFR, Treasury CMT yields); author's calculations."

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#cbd5e1", "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": "#eef2f7", "grid.linewidth": 0.8,
    "legend.frameon": False, "font.family": "DejaVu Sans", "text.usetex": False,
})


def save(fig, path, subtitle=None):
    """Footnote (method note) on one line and the source on the line below it."""
    note = (subtitle + "\n" if subtitle else "") + SOURCE
    fig.text(0.01, 0.005, note, fontsize=7.5, color=MUTED, ha="left", va="bottom", linespacing=1.4)
    fig.tight_layout(rect=(0, 0.06 if subtitle else 0.035, 1, 1))
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


class Report:
    """Collects Plotly figure specs and writes one self-contained interactive HTML page."""

    def __init__(self, title: str, intro: str):
        self.title, self.intro, self.sections = title, intro, []

    def add(self, heading: str, text: str, data: list, layout: dict):
        base = dict(template="plotly_white", margin=dict(l=60, r=20, t=40, b=50),
                    font=dict(family="Inter, Arial, sans-serif", size=12),
                    legend=dict(orientation="h", y=-0.2))
        base.update(layout)
        self.sections.append(dict(h=heading, p=text, data=data, layout=base))

    def add_table(self, heading: str, text: str, df):
        self.sections.append(dict(h=heading, p=text, table=df.to_html(index=False,
                                  float_format=lambda x: f"{x:,.4g}", border=0)))

    def write(self, path):
        parts = []
        for i, s in enumerate(self.sections):
            if "table" in s:
                parts.append(f'<section><h2>{s["h"]}</h2><p>{s["p"]}</p><div class="tbl">{s["table"]}</div></section>')
            else:
                parts.append(
                    f'<section><h2>{s["h"]}</h2><p>{s["p"]}</p><div id="f{i}" class="fig"></div>'
                    f'<script>Plotly.newPlot("f{i}", {json.dumps(s["data"])}, '
                    f'{json.dumps(s["layout"])}, {{responsive:true, displaylogo:false}});</script></section>')
        html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{self.title}</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>
body{{font-family:Inter,Arial,sans-serif;color:#1d2433;background:#fff;margin:0;padding:24px max(16px,4vw);line-height:1.5}}
main{{max-width:1050px;margin:auto}} h1{{font-size:28px;margin:0 0 6px}} h2{{font-size:19px;margin:34px 0 6px}}
p{{color:#475467;max-width:820px}} .fig{{width:100%;height:440px}} .tbl{{overflow-x:auto}}
table{{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}}
th,td{{padding:6px 10px;border-bottom:1px solid #eaecf0;text-align:right}} th{{background:#f8fafc}}
footer{{margin-top:40px;color:#98a2b3;font-size:12px}}
</style></head><body><main><h1>{self.title}</h1><p>{self.intro}</p>
{''.join(parts)}
<footer>{SOURCE} Code: github.com/AtharvaDeshpande002/rates-monte-carlo</footer></main></body></html>"""
        with open(path, "w") as f:
            f.write(html)


def tolist(a):
    return [None if (isinstance(v, float) and not np.isfinite(v)) else v
            for v in np.asarray(a, dtype=float).round(6).tolist()]
