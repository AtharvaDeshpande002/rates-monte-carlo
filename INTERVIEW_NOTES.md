# Interview Notes — Monte Carlo Simulation of US Short-Rate Paths

A plain-English guide to every idea, number and design choice in this project.

## The project in 10 sentences

1. I modelled the overnight US interest rate as a random process that is pulled back towards a long-run level (Vasicek / Ornstein–Uhlenbeck), plus two variants: CIR, which keeps rates positive, and Hull–White, which fits today's curve exactly.
2. I calibrated them to 8 years of daily SOFR by maximum likelihood and showed OLS gives the same answer.
3. Volatility came out precise (95 bp per √year), but the speed of mean reversion was statistically indistinguishable from zero.
4. I built a Monte Carlo engine and validated it: error falls like 1/√N, and every estimate matches a closed-form price within its confidence interval.
5. Two variance-reduction tricks (antithetic variates and a control variate) cut the variance of the bond-price estimate about 5,000–10,000 times.
6. I measured discretisation error separately: Euler's bias shrinks in proportion to the time step, and Milstein converges faster than Euler for CIR.
7. From today's curve I read the market's expected Fed path (~100 bp of hikes over two years) and turned it into meeting-by-meeting probabilities.
8. The history-based model and the market disagree: history gives ~35–40% odds of a cut, the curve implies steady hikes.
9. I priced a caplet and the futures convexity adjustment in Hull–White, checking both by simulation.
10. A Gaussian model understates the tail risk of a SOFR futures position: historical Expected Shortfall is 22% higher at 95%.

## Each method in plain words

**Ornstein–Uhlenbeck (OU) / Vasicek.** dr = κ(θ − r)dt + σdW. Think of a ball on a spring being kicked randomly. κ is the spring strength, θ where the spring rests, σ the kick size. *Example:* κ = 0.5 per year means the gap to θ halves every ln2/0.5 = 1.4 years.

**Why OU and not a random walk?** A random walk wanders without limit; interest rates are anchored by policy and inflation targets. OU builds in that anchor. (Honest caveat: on our data the anchor is weak, κ ≈ 0.11 with a large standard error.)

**Half-life.** ln2/κ: how long until the expected gap to θ halves. κ = 0.11 gives 6.2 years, which is really a sign of "no reliable mean reversion" in this sample.

**Langevin and Fokker–Planck (the physics link).** The OU equation is the Langevin equation of a Brownian particle in a harmonic potential. The Fokker–Planck equation describes how its probability density evolves; its steady state is Gaussian with variance σ²/2κ, the Boltzmann distribution of statistical physics.

**Exact simulation vs Euler.** For OU the next value given today's is exactly Gaussian with known mean and variance, so we can sample it with no error. Euler replaces the true dynamics with a straight-line step: r + κ(θ − r)Δt + σ√Δt·z, which is wrong by O(Δt).

**Maximum likelihood (MLE).** Choose parameters that make the observed data most probable: maximise Σ log p(r_{t+1} | r_t). Standard errors come from the curvature of the log-likelihood at the maximum (inverse Hessian, the Fisher information).

**MLE vs OLS.** The exact OU transition is r_{t+1} = c + b·r_t + Gaussian noise, an AR(1). OLS on that regression gives the same b and c as MLE. Then κ = −ln(b)/Δt, θ = c/(1 − b), σ² = s²·2κ/(1 − b²). σ differs only by the n vs n−2 degrees-of-freedom correction (a test checks this).

**CIR.** dr = κ(θ − r)dt + σ√r dW. Volatility shrinks as rates approach zero, so rates stay positive if 2κθ ≥ σ² (the Feller condition). Its exact transition is a scaled non-central chi-square, which is why we need MLE rather than OLS.

**Hull–White.** Like Vasicek but θ(t) varies with time, chosen so the model reproduces today's discount curve exactly. Used for pricing because it is consistent with market prices, unlike a history-calibrated model.

**Monte Carlo error 1/√N.** The estimate is an average of N independent draws; the standard deviation of an average is s/√N. To halve the error you need 4× the paths.

**Antithetic variates.** Use each shock z and its mirror −z. The two outcomes are negatively correlated, so their average varies less. It works extremely well here because the bond payoff is nearly linear in the shocks.

**Control variate.** Find a quantity Y correlated with the payoff X whose mean you know exactly (here Y = ∫r dt, mean θT + (r₀ − θ)(1 − e^(−κT))/κ). Estimate E[X + c(Y − E[Y])] with c = −Cov(X,Y)/Var(Y). The correction removes the part of the noise explained by Y.

**Weak vs strong error.** Weak error: how wrong averages (prices) are. Strong error: how far individual simulated paths are from the true path. Euler: weak order 1, strong order ½. Milstein adds ¼σ²Δt(z² − 1) and reaches strong order 1.

**Market-implied FOMC probabilities.** Expected rate after a meeting = average forward rate until the next meeting. If the expected change is +10 bp and moves come in 25 bp steps, the simplest consistent split is 40% × 25 bp + 60% × 0.

**Caplet.** A call on a future 3-month rate. In Hull–White it equals a put on a zero-coupon bond, which has a closed form. Monte Carlo must match it within its confidence interval.

**Convexity adjustment.** Futures are margined daily; forwards settle once. If rates rise, the short futures holder receives cash immediately and reinvests it at higher rates. That advantage makes futures rates sit above forward rates, more so for longer maturities and higher volatility.

**VaR and Expected Shortfall.** VaR(95%): the loss exceeded only 5% of the time. ES(95%): the average loss in those worst 5% of cases. ES looks into the tail; VaR stops at its edge.

## Every number and where it comes from

| Number | Source |
|---|---|
| 2,137 SOFR fixings, 3.90% on 22-Oct-2026 | `data/raw/SOFR.csv` (FRED) |
| κ = 0.11, se 0.17, 95% CI [−0.22, 0.45] | `mle_vasicek` on cleaned SOFR; `reports/tables/calibration.csv` |
| σ = 95 bp/√yr (166 bp with spikes) | same; spikes removed by `clean_spikes` |
| CIR log-likelihood +44 vs Vasicek | `mle_cir` |
| slope −0.50; variance reduction ~4,900× / ~9,700× | `convergence_table`, `reports/tables/convergence.csv` |
| Euler weak order 0.95; CIR Milstein 1.07 vs Euler 0.65 | `ou_euler_weak_bias`, `cir_strong_error` |
| ~100 bp of hikes priced; 76% for Dec-2026 | `implied_moves`, `reports/tables/fomc_implied_moves.csv` |
| 35–41% odds of a cut (history) | `reports/tables/scenario_probabilities.csv` |
| caplet 8.61 bp (closed form), 8.68 ± 0.08 bp (MC) | `HullWhite.caplet`, `simulate_hw` |
| convexity 0.5 / 1.5 / 5.8 bp at 1 / 2 / 5 years | `HullWhite.convexity_adjustment` |
| ES95 $1,351 (model) vs $1,654 (history) | `reports/tables/var_es.csv` |

## What didn't work, and why

- **History-based mean reversion.** κ is not statistically different from zero, so θ (4.88%) is not a usable forecast. Rates move in policy steps with long flat stretches; an OU process is a poor description of that at a daily frequency.
- **Raw SOFR volatility** is dominated by two repo-stress days in September 2019. Without cleaning, σ jumps from 95 to 166 bp.
- **The convexity Monte Carlo check is "too good"**: in Hull–White the continuously compounded rate is linear in r(T), so antithetic pairs cancel all noise. It validates the formula and the θ(t) fit, but it is not a hard test of the simulator. The caplet check is the harder test.

## 15 likely interview questions

1. **Why OU and not geometric Brownian motion?** Rates are mean-reverting and can be near or below zero; GBM is for prices that grow and stay positive.
2. **Your κ is insignificant. Is the model useless?** Not for volatility or short-horizon risk, where σ dominates. It is useless for long-run forecasts. That is why pricing uses Hull–White fitted to the curve, not history.
3. **Why do OLS and MLE agree?** The exact OU transition is a Gaussian AR(1); for Gaussian errors conditional MLE equals least squares.
4. **How do you get standard errors?** From the inverse Hessian of the negative log-likelihood at the optimum (Fisher information).
5. **Why remove the September 2019 spike?** It was a one-day funding squeeze that reversed, not a change in rate expectations; it would dominate a Gaussian volatility estimate. FOMC moves, which do not reverse, are kept.
6. **Why does Monte Carlo error fall like 1/√N?** Central limit theorem: the sample mean's standard deviation is s/√N.
7. **Why do antithetic variates work so well here?** The discount factor is almost linear in the Gaussian shocks, so z and −z errors nearly cancel.
8. **What makes a good control variate?** High correlation with the payoff and an exactly known mean.
9. **Euler vs exact scheme: when does it matter?** When κΔt is not small (fast mean reversion or large steps); Euler's bias is O(Δt).
10. **What is the Feller condition?** 2κθ ≥ σ²: the CIR process cannot reach zero.
11. **What is the sign of the convexity adjustment and why?** Positive (futures rate above forward), because of daily margining and the correlation between rates and the reinvestment rate.
12. **How did you get FOMC probabilities?** From average forward rates between meetings, split between the two neighbouring 25 bp outcomes. That is the minimum-variance split consistent with the curve.
13. **Why do the history-based and market probabilities differ so much?** History has no view on direction and wide dispersion; the curve has a strong directional view but, in a two-outcome model, little dispersion. The truth is in between; the market's past forecast errors (~69 bp at 12 months in my Fed-pricing project) show the curve is not a certainty.
14. **Limits of one-factor models?** All maturities move in lockstep; you cannot model steepeners or flatteners. Use two factors (G2++) for curve trades.
15. **How would you improve the risk numbers?** Add jumps at FOMC dates, use a fat-tailed shock distribution, or use filtered historical simulation.

## Walk me through the code

- `data.py`: loads FRED CSVs, removes one-day reversing spikes, builds the SOFR-anchored curve.
- `models.py`: OU exact/Euler steps, closed-form moments and bond price, CIR Euler/Milstein, the `HullWhite` class (curve fit, bond prices, caplet, convexity).
- `calibration.py`: OLS AR(1), Vasicek MLE with Hessian standard errors, CIR MLE, rolling fits.
- `montecarlo.py`: path simulation, the three bond estimators, convergence table, Euler bias, CIR strong error, Hull–White simulation.
- `fomc.py`: meeting probabilities from the curve and the step-path simulator.
- `risk.py`: futures-rate loading, VaR and ES.
- `scripts/run_all.py`: runs everything and writes figures, tables, `results.json` and `report.html`.
- `tests/`: 11 checks against closed forms and statistical identities.
