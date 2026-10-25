"""Reproduce every number, table and chart in the project.

    python scripts/run_all.py

Steps
  1. Load SOFR / EFFR, remove one-day funding spikes
  2. Calibrate Vasicek (OLS and exact MLE) and CIR (exact MLE); rolling estimates
  3. Monte Carlo: convergence, variance reduction, discretisation error
  4. Today's curve -> Hull-White model and FOMC move probabilities
  5. Fan charts and scenario probabilities: history-based model vs market-implied
  6. Caplet pricing (Monte Carlo vs closed form) and futures convexity adjustment
  7. 1-month VaR / Expected Shortfall for a SOFR futures position
Outputs go to reports/ (figures, tables, results.json, report.html).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ratesmc import ROOT, load_config  # noqa: E402
from ratesmc.calibration import mle_cir, mle_vasicek, ols_ar1, rolling_vasicek  # noqa: E402
from ratesmc.data import clean_spikes, load_series, policy_curve  # noqa: E402
from ratesmc.fomc import implied_moves, simulate_steps, year_frac  # noqa: E402
from ratesmc.models import HullWhite, feller_ok, ou_half_life, ou_step_exact  # noqa: E402
from ratesmc.montecarlo import (cir_strong_error, convergence_table,  # noqa: E402
                                ou_euler_weak_bias, simulate_hw, simulate_ou)
from ratesmc.plots import (BLUE, GREEN, GREY, INK, MUTED, ORANGE, Report,  # noqa: E402
                           plt, save, tolist)
from ratesmc.risk import loading, var_es  # noqa: E402

FIG = ROOT / "reports" / "figures"
TAB = ROOT / "reports" / "tables"


def main():
    cfg = load_config()
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    dt = cfg["calibration"]["dt_years"]
    mc = cfg["monte_carlo"]
    seed = mc["seed"]
    res: dict = {}
    rep = Report("Monte Carlo Simulation of US Short-Rate Paths",
                 "Short-rate models (Vasicek / Ornstein-Uhlenbeck, CIR, Hull-White) calibrated to "
                 "SOFR history and to today's curve; Monte Carlo convergence and variance reduction; "
                 "FOMC scenario probabilities, option pricing, futures convexity and VaR. "
                 f"Data as of {cfg['data']['as_of']}.")

    # ------------------------------------------------------------------ 1. data
    sofr = load_series(cfg, "SOFR")
    effr = load_series(cfg, "EFFR")
    sofr_c, spikes = clean_spikes(sofr, cfg["calibration"]["winsor_bp"])
    effr_c, _ = clean_spikes(effr, cfg["calibration"]["winsor_bp"])
    r0 = float(sofr.iloc[-1])
    res["data"] = dict(sofr_obs=len(sofr), sofr_start=str(sofr.index[0].date()),
                       sofr_end=str(sofr.index[-1].date()), r0=r0,
                       spikes_removed=[str(d.date()) for d in sofr.index[spikes]])

    # ------------------------------------------------------------------ 2. calibration
    x = sofr_c.to_numpy()
    ols = ols_ar1(x, dt)
    vas = mle_vasicek(x, dt)
    vas_raw = mle_vasicek(sofr.to_numpy(), dt)
    cir = mle_cir(x, dt, vas)
    k, th, sg = vas["kappa"], vas["theta"], vas["sigma"]
    calib = pd.DataFrame([
        dict(model="Vasicek (OLS on AR(1))", kappa=ols["kappa"], theta=ols["theta"], sigma=ols["sigma"]),
        dict(model="Vasicek (exact MLE)", kappa=k, theta=th, sigma=sg,
             se_kappa=vas["se_kappa"], se_theta=vas["se_theta"], se_sigma=vas["se_sigma"],
             loglik=vas["loglik"]),
        dict(model="Vasicek MLE, spikes kept", kappa=vas_raw["kappa"], theta=vas_raw["theta"],
             sigma=vas_raw["sigma"], loglik=vas_raw["loglik"]),
        dict(model="CIR (exact MLE)", kappa=cir["kappa"], theta=cir["theta"], sigma=cir["sigma"],
             loglik=cir["loglik"]),
    ])
    calib.to_csv(TAB / "calibration.csv", index=False)
    res["vasicek"] = dict(kappa=k, theta=th, sigma=sg, se_kappa=vas["se_kappa"],
                          se_theta=vas["se_theta"], se_sigma=vas["se_sigma"],
                          half_life_years=ou_half_life(k),
                          kappa_ci95=[k - 1.96 * vas["se_kappa"], k + 1.96 * vas["se_kappa"]],
                          sigma_raw=vas_raw["sigma"], ols_kappa=ols["kappa"])
    res["cir"] = dict(**cir, feller=bool(feller_ok(cir["kappa"], cir["theta"], cir["sigma"])),
                      lr_vs_vasicek=2 * (cir["loglik"] - vas["loglik"]))
    show = calib.copy()
    for c in ["theta", "sigma", "se_theta", "se_sigma"]:
        show[c] = show[c] * 100
    show = show.rename(columns={"theta": "theta (%)", "sigma": "sigma (%/sqrt yr)",
                                "se_theta": "se theta (%)", "se_sigma": "se sigma (%)"})
    rep.add_table("1. Calibration to SOFR history (2018-2026)",
                  f"OLS on the AR(1) form and exact maximum likelihood give identical Vasicek "
                  f"estimates, as theory says. Volatility is pinned down precisely "
                  f"({sg*1e4:.0f} bp per sqrt-year), but mean reversion is not: kappa = {k:.2f} "
                  f"with a standard error of {vas['se_kappa']:.2f}, so the 95% interval includes zero. "
                  f"Daily rates move in FOMC steps, so history alone says little about the pull "
                  f"back to a long-run level. Keeping the two repo-spike days inflates sigma to "
                  f"{vas_raw['sigma']*1e4:.0f} bp. CIR fits better (log-likelihood "
                  f"+{cir['loglik']-vas['loglik']:.0f}) because volatility scales with the rate level.",
                  show.round(4))

    # rolling estimates on EFFR (longer history)
    roll = rolling_vasicek(effr_c, cfg["calibration"]["rolling_window_days"],
                           cfg["calibration"]["rolling_step_days"], dt)
    roll.to_csv(TAB / "rolling_vasicek_effr.csv")
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(roll.index, roll["sigma"] * 1e4, color=BLUE)
    ax[0].fill_between(roll.index, (roll["sigma"] - 1.96 * roll["se_sigma"]) * 1e4,
                       (roll["sigma"] + 1.96 * roll["se_sigma"]) * 1e4, color=BLUE, alpha=0.15)
    ax[0].set_ylabel("sigma (bp / sqrt yr)")
    ax[0].set_title("Rate volatility is regime-dependent: it tripled in the 2022-23 hiking cycle")
    kk = roll["kappa"].clip(upper=3)
    ax[1].plot(roll.index, kk, color=ORANGE)
    ax[1].fill_between(roll.index, (roll["kappa"] - 1.96 * roll["se_kappa"]).clip(-1, 3),
                       (roll["kappa"] + 1.96 * roll["se_kappa"]).clip(-1, 3), color=ORANGE, alpha=0.15)
    ax[1].axhline(0, color=GREY, lw=0.8)
    ax[1].set_ylabel("kappa (1/yr)")
    ax[1].set_title("Mean-reversion speed is poorly identified: its 95% band almost always spans zero")
    for a_ in ax:
        for d0, d1, lab in [("2022-03-01", "2023-07-31", "hikes"), ("2024-09-01", "2025-12-31", "cuts")]:
            a_.axvspan(pd.Timestamp(d0), pd.Timestamp(d1), color="#f1f5f9", zorder=0)
    save(fig, FIG / "02_rolling_parameters.png", "Rolling 2-year exact-MLE Vasicek fits on EFFR, 95% bands from the Hessian. Shaded: hiking and cutting cycles.")
    rep.add("2. Rolling calibration (EFFR, 2-year windows)",
            "Volatility and mean reversion change with the policy regime, with 95% confidence bands. "
            "A single historical calibration is not a stable description of rates.",
            [dict(x=[str(d.date()) for d in roll.index], y=tolist(roll["sigma"] * 1e4),
                  name="sigma (bp/sqrt yr)", line=dict(color=BLUE)),
             dict(x=[str(d.date()) for d in roll.index], y=tolist(roll["kappa"].clip(upper=3)),
                  name="kappa (1/yr)", yaxis="y2", line=dict(color=ORANGE))],
            dict(yaxis=dict(title="sigma (bp / sqrt yr)"),
                 yaxis2=dict(title="kappa (1/yr)", overlaying="y", side="right")))

    # empirical vs stationary distribution
    stat_sd = sg / np.sqrt(2 * k)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.hist(x * 100, bins=60, density=True, color=GREY, alpha=0.7, label="SOFR, daily 2018-2026")
    g = np.linspace(-4, 10, 400)
    from scipy.stats import norm
    ax.plot(g, norm.pdf(g, th * 100, stat_sd * 100), color=BLUE, lw=2,
            label=f"Vasicek stationary N({th*100:.1f}%, {stat_sd*100:.1f}%^2)")
    ax.axvline(0, color=ORANGE, lw=1, ls="--")
    ax.set_xlabel("rate (%)"); ax.set_ylabel("density")
    ax.set_title(f"The Gaussian stationary law puts {norm.cdf(0, th, stat_sd)*100:.0f}% mass below zero; real rates cluster at policy levels")
    ax.legend()
    save(fig, FIG / "03_stationary_distribution.png", "Stationary density from the Fokker-Planck equation of the OU process: N(theta, sigma^2 / 2kappa).")
    res["stationary"] = dict(sd=stat_sd, p_negative=float(norm.cdf(0, th, stat_sd)))

    # ------------------------------------------------------------------ 3. MC accuracy
    T = 2.0
    exact, conv = convergence_table(r0, k, th, sg, T, mc["n_grid"], 104, seed)
    conv = pd.DataFrame(conv)
    conv.to_csv(TAB / "convergence.csv", index=False)
    last = conv[conv.n == max(mc["n_grid"])].set_index("method")
    vr = (last.loc["plain", "se"] / last.loc[["antithetic", "control"], "se"]) ** 2
    res["convergence"] = dict(exact_bond_2y=exact,
                              se_plain_100k=last.loc["plain", "se"],
                              se_antithetic_100k=last.loc["antithetic", "se"],
                              se_control_100k=last.loc["control", "se"],
                              variance_reduction_antithetic=float(vr["antithetic"]),
                              variance_reduction_control=float(vr["control"]))
    slope = np.polyfit(np.log(conv[conv.method == "plain"].n), np.log(conv[conv.method == "plain"].se), 1)[0]
    res["convergence"]["plain_slope"] = float(slope)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for m, col in [("plain", INK), ("antithetic", BLUE), ("control", GREEN)]:
        d = conv[conv.method == m]
        ax.loglog(d.n, d.se * 1e4, "o-", color=col, label=f"{m} estimator")
    nn = np.array(mc["n_grid"], float)
    d = conv[conv.method == "plain"]
    ax.loglog(nn, d.se.iloc[0] * 1e4 * np.sqrt(nn[0] / nn), "--", color=GREY, label="1/sqrt(N) reference")
    ax.set_xlabel("number of paths N"); ax.set_ylabel("standard error of P(0,2y) (bp of price)")
    ax.set_title(f"Error falls as 1/sqrt(N) (fitted slope {slope:.2f}); variance reduction cuts it ~{np.sqrt(vr['control']):.0f}x")
    ax.legend()
    save(fig, FIG / "04_mc_convergence.png", "Two-year zero-coupon bond under calibrated Vasicek; closed-form price used as the benchmark.")
    rep.add("3. Monte Carlo convergence and variance reduction",
            f"Standard error of the simulated 2-year bond price against the closed form "
            f"({exact:.6f}). Plain Monte Carlo follows the 1/sqrt(N) law (slope {slope:.2f}). "
            f"Antithetic variates reduce variance {vr['antithetic']:,.0f}x and the control variate "
            f"(the integral of r, whose mean is known) {vr['control']:,.0f}x at the same N.",
            [dict(x=tolist(conv[conv.method == m].n), y=tolist(conv[conv.method == m].se * 1e4),
                  name=m, mode="lines+markers") for m in ["plain", "antithetic", "control"]],
            dict(xaxis=dict(type="log", title="paths N"),
                 yaxis=dict(type="log", title="standard error (bp of price)")))

    weak = pd.DataFrame(ou_euler_weak_bias(r0, 2.0, 0.03, 0.01, 2.0, mc["dt_grid_per_year"]))
    strong = pd.DataFrame(cir_strong_error(0.03, 1.0, 0.04, 0.1, 1.0, 4000, [2, 3, 4, 5, 6, 7],
                                           np.random.default_rng(seed)))
    weak.to_csv(TAB / "euler_weak_bias.csv", index=False)
    strong.to_csv(TAB / "cir_strong_error.csv", index=False)
    se_ = np.polyfit(np.log(strong.dt), np.log(strong.euler), 1)[0]
    sm_ = np.polyfit(np.log(strong.dt), np.log(strong.milstein), 1)[0]
    sw_ = np.polyfit(np.log(weak.dt), np.log(weak.mean_bias_bp.abs()), 1)[0]
    res["discretisation"] = dict(euler_weak_order=float(sw_), cir_euler_strong_order=float(se_),
                                 cir_milstein_strong_order=float(sm_))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].loglog(weak.dt, weak.mean_bias_bp.abs(), "o-", color=INK, label="|bias| in E[r_T]")
    ax[0].loglog(weak.dt, weak.sd_bias_bp.abs(), "o-", color=BLUE, label="|bias| in sd(r_T)")
    ax[0].set_xlabel("time step dt (years)"); ax[0].set_ylabel("bias (bp)")
    ax[0].set_title(f"Euler weak error is first order (slope {sw_:.2f})")
    ax[0].legend()
    ax[1].loglog(strong.dt, strong.euler * 1e4, "o-", color=ORANGE, label=f"Euler (slope {se_:.2f})")
    ax[1].loglog(strong.dt, strong.milstein * 1e4, "o-", color=GREEN, label=f"Milstein (slope {sm_:.2f})")
    ax[1].set_xlabel("time step dt (years)"); ax[1].set_ylabel("E|r_T - r_T(ref)| (bp)")
    ax[1].set_title("CIR: Milstein converges faster than Euler")
    ax[1].legend()
    save(fig, FIG / "05_discretisation_error.png", "Left: exact moments of the Euler OU recursion vs the exact OU law (fast-reverting test case, kappa=2). Right: CIR, same Brownian paths, reference = fine-grid Milstein.")
    rep.add("4. Discretisation error",
            f"Euler's bias in the OU mean shrinks in proportion to the time step (order {sw_:.2f}). "
            f"For CIR, the strong error of Milstein falls with order {sm_:.2f} against {se_:.2f} for Euler.",
            [dict(x=tolist(strong.dt), y=tolist(strong.euler * 1e4), name="CIR Euler", mode="lines+markers"),
             dict(x=tolist(strong.dt), y=tolist(strong.milstein * 1e4), name="CIR Milstein", mode="lines+markers"),
             dict(x=tolist(weak.dt), y=tolist(weak.mean_bias_bp.abs()), name="OU Euler mean bias", mode="lines+markers")],
            dict(xaxis=dict(type="log", title="dt (years)"), yaxis=dict(type="log", title="error (bp)")))

    # ------------------------------------------------------------------ 4. today's curve
    tg, zg = policy_curve(cfg, r0)
    hw = HullWhite(tg, zg, cfg["pricing"]["hw_a"], sg)
    moves, _ = implied_moves(hw, cfg["data"]["as_of"], cfg["fomc"]["meetings"], T,
                             cfg["fomc"]["step_bp"])
    moves.to_csv(TAB / "fomc_implied_moves.csv", index=False)
    res["fomc"] = moves.head(4).round(4).to_dict(orient="records")
    reprice = [(t_, float(hw.P0(t_)), float(np.exp(-np.interp(t_, tg, zg) * t_))) for t_ in [0.25, 0.5, 1, 2]]
    res["hw_reprice_error_bp"] = max(abs(np.log(a / b)) / t * 1e4 for t, a, b in reprice)

    # ------------------------------------------------------------------ 5. fans and probabilities
    rng = np.random.default_rng(seed)
    n_steps = int(T * mc["steps_per_year"])
    store = 5
    vs = simulate_ou(r0, k, th, sg, T, n_steps, mc["n_paths_fan"], rng, store_every=store)
    fs = simulate_steps(moves, r0, T, n_steps, mc["n_paths_fan"], cfg["fomc"]["noise_bp"],
                        np.random.default_rng(seed + 1), store)
    tt = vs["t"]
    fwd = hw.f0(tt)
    exp_path = np.array([np.mean(hw.f0(np.linspace(max(t_ - 0.02, 0), t_, 5))) for t_ in tt])
    q = [5, 25, 50, 75, 95]
    pv = np.percentile(vs["r"], q, axis=1) * 100
    pf = np.percentile(fs["r"], q, axis=1) * 100
    asof = pd.Timestamp(cfg["data"]["as_of"])
    dates = [asof + pd.Timedelta(days=float(t_) * 365.25) for t_ in tt]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i in range(150):
        ax.plot(dates, vs["r"][:, i] * 100, color=BLUE, alpha=0.04, lw=0.6)
    ax.fill_between(dates, pv[0], pv[4], color=BLUE, alpha=0.10, label="Vasicek (history) 5-95%")
    ax.fill_between(dates, pv[1], pv[3], color=BLUE, alpha=0.18, label="Vasicek 25-75%")
    ax.fill_between(dates, pf[0], pf[4], color=ORANGE, alpha=0.25, label="FOMC-step model 5-95%")
    ax.plot(dates, fwd * 100, color=INK, lw=2, label="market forward path (curve)")
    ax.plot(dates, pv[2], color=BLUE, lw=1.5, ls="--", label="Vasicek median")
    ax.set_ylabel("short rate (%)")
    ax.set_title(f"History implies far wider outcomes; the market path prices ~{(fwd[-1]-r0)*1e4:.0f}bp of hikes over 2 years")
    ax.legend(ncol=3, fontsize=8, loc="upper left")
    save(fig, FIG / "01_fan_chart.png", f"{mc['n_paths_fan']:,} simulated paths per model, start SOFR {r0*100:.2f}% on {cfg['data']['as_of']}. Market path: Treasury curve shifted to SOFR (see README).")
    rep.add("5. Fan chart: simulated short-rate paths",
            "Blue: Vasicek calibrated to history (5-95% and 25-75% bands). Orange: FOMC step model whose "
            "meeting probabilities reproduce the market's forward path. Black: the forward path itself.",
            [dict(x=[str(d.date()) for d in dates], y=tolist(pv[4]), line=dict(width=0), showlegend=False, hoverinfo="skip"),
             dict(x=[str(d.date()) for d in dates], y=tolist(pv[0]), fill="tonexty", fillcolor="rgba(37,99,235,0.12)", line=dict(width=0), name="Vasicek 5-95%"),
             dict(x=[str(d.date()) for d in dates], y=tolist(pf[4]), line=dict(width=0), showlegend=False, hoverinfo="skip"),
             dict(x=[str(d.date()) for d in dates], y=tolist(pf[0]), fill="tonexty", fillcolor="rgba(234,88,12,0.25)", line=dict(width=0), name="FOMC-step 5-95%"),
             dict(x=[str(d.date()) for d in dates], y=tolist(fwd * 100), name="market forward path", line=dict(color=INK, width=2)),
             dict(x=[str(d.date()) for d in dates], y=tolist(pv[2]), name="Vasicek median", line=dict(color=BLUE, dash="dash"))],
            dict(yaxis=dict(title="short rate (%)")))

    # scenario probabilities at three horizons
    horizons = [("Dec-2026", "2026-12-20"), ("Jun-2027", "2027-06-30"), ("Dec-2027", "2027-12-31")]
    rows = []
    for lab, d in horizons:
        i = int(np.argmin(np.abs(tt - year_frac(cfg["data"]["as_of"], [d])[0])))
        for nm, sim in [("Vasicek (history)", vs["r"][i]), ("FOMC step (market)", fs["r"][i])]:
            ch = (sim - r0) * 1e4
            rows.append(dict(horizon=lab, model=nm, p_hike_25plus=np.mean(ch >= 12.5),
                             p_hike_50plus=np.mean(ch >= 37.5), p_cut_25plus=np.mean(ch <= -12.5),
                             mean_change_bp=ch.mean(), sd_bp=ch.std()))
    probs = pd.DataFrame(rows)
    probs.to_csv(TAB / "scenario_probabilities.csv", index=False)
    res["scenarios"] = probs.round(3).to_dict(orient="records")
    fig, ax = plt.subplots(figsize=(10, 4.3))
    w = 0.2
    xs = np.arange(len(horizons))
    for j, (col, nm, c) in enumerate([("p_hike_50plus", "Vasicek (history)", BLUE),
                                      ("p_hike_50plus", "FOMC step (market)", ORANGE),
                                      ("p_cut_25plus", "Vasicek (history)", "#93c5fd"),
                                      ("p_cut_25plus", "FOMC step (market)", "#fdba74")]):
        v = probs[probs.model == nm][col].to_numpy() * 100
        lab = ("P(>= +50bp) " if col == "p_hike_50plus" else "P(<= -25bp) ") + nm
        ax.bar(xs + (j - 1.5) * w, v, w, color=c, label=lab)
    ax.set_xticks(xs, [h for h, _ in horizons]); ax.set_ylabel("probability (%)")
    ax.set_title("History gives ~35-40% odds of a cut; the market-consistent path implies steady hikes")
    ax.legend(fontsize=8, ncol=2)
    save(fig, FIG / "06_scenario_probabilities.png", "Change in the overnight rate from today's SOFR. The step model is the minimum-variance distribution consistent with the curve.")
    rep.add_table("6. Scenario probabilities: history-based model vs market-implied steps",
                  "Probability that the overnight rate is at least 25bp / 50bp higher, or 25bp lower, "
                  "than today at each horizon.", probs.round(3))

    # ------------------------------------------------------------------ 6. pricing
    pc = cfg["pricing"]
    Te, tau, K = pc["caplet_expiry_years"], pc["caplet_tenor_years"], pc["caplet_strike_pct"] / 100
    cap_cf = hw.caplet(Te, tau, K)
    est = []
    for n in [10_000, 100_000]:
        h = simulate_hw(hw, Te, int(Te * 252), n, np.random.default_rng(seed + n))
        P_TS = hw.bond(Te, Te + tau, h["r"])
        L = (1 / P_TS - 1) / tau
        pay = np.exp(-h["integral"]) * P_TS * tau * np.maximum(L - K, 0)
        est.append(dict(n=n, price=pay.mean(), se=pay.std(ddof=1) / np.sqrt(n)))
    res["caplet"] = dict(closed_form=cap_cf, mc=est, strike=K, expiry=Te, tenor=tau,
                         price_bp_running=cap_cf / tau * 1e4)

    ca_T = np.arange(0.25, pc["convexity_max_years"] + 1e-9, 0.25)
    ca = np.array([hw.convexity_adjustment(t_, t_ + 0.25) for t_ in ca_T]) * 1e4
    checks = []
    for t1 in [0.5, 1.0, 2.0]:
        h = simulate_hw(hw, t1, int(t1 * 252), 100_000, np.random.default_rng(seed + int(t1 * 100)),
                        antithetic=True)
        Rcc = -np.log(hw.bond(t1, t1 + 0.25, h["r"])) / 0.25
        Rp = 0.5 * (Rcc[:50_000] + Rcc[50_000:])           # antithetic pair averages
        fwd_cc = (np.log(hw.P0(t1)) - np.log(hw.P0(t1 + 0.25))) / 0.25
        checks.append(dict(T1=t1, mc_bp=(Rp.mean() - fwd_cc) * 1e4,
                           se_bp=Rp.std(ddof=1) / np.sqrt(len(Rp)) * 1e4,
                           formula_bp=hw.convexity_adjustment(t1, t1 + 0.25) * 1e4))
    pd.DataFrame(dict(start_years=ca_T, convexity_bp=ca)).to_csv(TAB / "convexity.csv", index=False)
    res["convexity"] = dict(at_1y=float(ca[ca_T == 1.0][0]), at_2y=float(ca[ca_T == 2.0][0]),
                            at_5y=float(ca[-1]), mc_checks=checks)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(ca_T, ca, width=0.18, color=BLUE)
    for c_ in checks:
        ax.errorbar(c_["T1"], c_["mc_bp"], yerr=1.96 * c_["se_bp"], fmt="o", color=ORANGE, ms=5)
    ax.plot([], [], "o", color=ORANGE, label="Monte Carlo check (95% CI)")
    ax.set_xlabel("contract start (years)"); ax.set_ylabel("futures - forward rate (bp)")
    ax.set_title(f"Convexity grows with maturity: ~{res['convexity']['at_2y']:.1f}bp at 2y, {res['convexity']['at_5y']:.1f}bp at 5y")
    ax.legend()
    save(fig, FIG / "07_convexity_adjustment.png", f"Hull-White, a={pc['hw_a']}, sigma={sg*1e4:.0f}bp. Applies to 3-month rate futures; SOFR futures average the rate in arrears (small difference).")
    rep.add("7. Futures convexity adjustment",
            "Futures rates sit above forward rates because futures are margined daily. Hull-White formula "
            "(bars) and Monte Carlo checks (orange, 95% CI).",
            [dict(type="bar", x=tolist(ca_T), y=tolist(ca), name="formula"),
             dict(x=[c_["T1"] for c_ in checks], y=[round(c_["mc_bp"], 4) for c_ in checks],
                  error_y=dict(type="data", array=[round(1.96 * c_["se_bp"], 4) for c_ in checks]),
                  mode="markers", name="Monte Carlo")],
            dict(xaxis=dict(title="contract start (years)"), yaxis=dict(title="bp")))

    # ------------------------------------------------------------------ 7. risk
    rk = cfg["risk"]
    h_days = rk["horizon_days"]
    L_ = loading(k, h_days / 252, rk["sr3_start_years"])
    rr = np.full(200_000, r0)
    g_ = np.random.default_rng(seed + 7)
    for _ in range(h_days):
        rr = ou_step_exact(rr, k, th, sg, dt, g_.standard_normal(rr.size))
    dF_mc = (rr - r0) * L_ * 1e4                   # bp change in futures rate
    pnl_mc = -dF_mc * rk["dollars_per_bp"]
    chg = (sofr_c.shift(-h_days) - sofr_c).dropna().to_numpy() * 1e4
    pnl_hist = -chg * L_ * rk["dollars_per_bp"]
    risk_tbl = pd.DataFrame([dict(method="Monte Carlo (Vasicek)", **var_es(pnl_mc, rk["levels"])),
                             dict(method="Historical (2018-2026)", **var_es(pnl_hist, rk["levels"]))])
    risk_tbl.to_csv(TAB / "var_es.csv", index=False)
    res["risk"] = risk_tbl.round(1).to_dict(orient="records")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bins = np.linspace(-2500, 2500, 101)
    ax.hist(pnl_hist, bins=bins, density=True, color=GREY, alpha=0.6, label="historical 21-day changes")
    ax.hist(pnl_mc, bins=bins, density=True, histtype="step", color=BLUE, lw=2, label="Monte Carlo (Vasicek)")
    for _, rw in risk_tbl.iterrows():
        col = BLUE if rw["method"].startswith("Monte") else INK
        ax.axvline(-rw["ES99"], color=col, ls="--", lw=1)
    ax.set_xlabel("1-month P&L, long 1 SR3 contract (USD)"); ax.set_ylabel("density")
    h99 = risk_tbl.set_index("method")
    ax.set_title(f"99% Expected Shortfall: USD {h99.iloc[0]['ES99']:,.0f} (Gaussian model) vs USD {h99.iloc[1]['ES99']:,.0f} (history)")
    ax.legend()
    save(fig, FIG / "08_var_es.png", "Dashed lines: 99% Expected Shortfall. Contract referencing the quarter starting ~6 months out; $25 per bp.")
    rep.add_table("8. One-month VaR and Expected Shortfall (long 1 SR3 contract, USD)",
                  "Parametric Monte Carlo under Vasicek vs historical simulation of 21-day SOFR changes.",
                  risk_tbl.round(1))

    rep.write(ROOT / "reports" / "report.html")
    with open(ROOT / "reports" / "results.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
