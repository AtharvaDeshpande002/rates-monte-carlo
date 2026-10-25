# Resume bullets (numbers from reports/results.json)

- Calibrated Vasicek (Ornstein–Uhlenbeck), CIR and Hull–White short-rate models by maximum likelihood on 8 years of SOFR, with Hessian standard errors; showed volatility is precise (95 bp/√yr) but mean reversion is statistically insignificant
- Built a Monte Carlo engine (exact, Euler, Milstein schemes) validated against closed forms: 1/√N convergence and ~9,700× variance reduction with control variates
- Compared history-based vs market-implied FOMC scenario probabilities; priced caplets and futures convexity adjustments; estimated VaR and Expected Shortfall

One-line version:
- Monte Carlo short-rate models (OU/Vasicek, CIR, Hull–White) calibrated by MLE; variance reduction, convergence analysis, option pricing and VaR/ES
