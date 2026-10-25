"""Short-rate models: Vasicek (Ornstein-Uhlenbeck), CIR and Hull-White.

Notation (all rates are decimals, time in years):
    Vasicek / OU   dr = kappa (theta - r) dt + sigma dW
    CIR            dr = kappa (theta - r) dt + sigma sqrt(r) dW
    Hull-White     dr = (theta(t) - a r) dt + sigma dW   (theta(t) fitted to today's curve)

Physics link: the OU process is the Langevin equation for a Brownian particle held by
a spring (a harmonic potential). kappa is the spring strength (how fast the rate is pulled
back), theta the spring's resting point, sigma the size of the random kicks. The
Fokker-Planck equation for its density has the Gaussian stationary solution
N(theta, sigma^2 / (2 kappa)).
"""
from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------------- Vasicek


def ou_step_exact(r: np.ndarray, kappa: float, theta: float, sigma: float, dt: float,
                  z: np.ndarray) -> np.ndarray:
    """One exact OU step: the transition density is Gaussian, so no discretisation error."""
    e = np.exp(-kappa * dt)
    sd = sigma * np.sqrt((1.0 - e * e) / (2.0 * kappa))
    return theta + (r - theta) * e + sd * z


def ou_step_euler(r: np.ndarray, kappa: float, theta: float, sigma: float, dt: float,
                  z: np.ndarray) -> np.ndarray:
    """One Euler-Maruyama step: first-order approximation, bias of order dt."""
    return r + kappa * (theta - r) * dt + sigma * np.sqrt(dt) * z


def ou_mean(r0: float, kappa: float, theta: float, t: np.ndarray | float):
    """E[r(t) | r(0) = r0] for the OU process."""
    return theta + (r0 - theta) * np.exp(-kappa * np.asarray(t))


def ou_var(kappa: float, sigma: float, t: np.ndarray | float):
    """Var[r(t) | r(0)] for the OU process; tends to sigma^2/(2 kappa) (stationary)."""
    return sigma ** 2 * (1.0 - np.exp(-2.0 * kappa * np.asarray(t))) / (2.0 * kappa)


def ou_half_life(kappa: float) -> float:
    """Time for the expected gap to theta to halve: ln 2 / kappa."""
    return np.log(2.0) / kappa


def vasicek_bond(r0: float, kappa: float, theta: float, sigma: float, T):
    """Closed-form Vasicek zero-coupon bond price P(0, T) (parameters under Q)."""
    T = np.asarray(T, dtype=float)
    B = (1.0 - np.exp(-kappa * T)) / kappa
    lnA = (theta - sigma ** 2 / (2 * kappa ** 2)) * (B - T) - sigma ** 2 * B ** 2 / (4 * kappa)
    return np.exp(lnA - B * r0)


def vasicek_integral_mean(r0: float, kappa: float, theta: float, T: float) -> float:
    """E[ integral_0^T r(s) ds ]: used as a control variate (its mean is known exactly)."""
    return theta * T + (r0 - theta) * (1.0 - np.exp(-kappa * T)) / kappa

# ----------------------------------------------------------------------------- CIR


def cir_step_euler(r, kappa, theta, sigma, dt, z):
    """Full-truncation Euler for CIR: use max(r, 0) inside drift and diffusion."""
    rp = np.maximum(r, 0.0)
    return r + kappa * (theta - rp) * dt + sigma * np.sqrt(rp * dt) * z


def cir_step_milstein(r, kappa, theta, sigma, dt, z):
    """Milstein scheme for CIR: adds the 0.25 sigma^2 dt (z^2 - 1) correction term
    (strong order 1 instead of 1/2 for Euler)."""
    rp = np.maximum(r, 0.0)
    out = (r + kappa * (theta - rp) * dt + sigma * np.sqrt(rp * dt) * z
           + 0.25 * sigma ** 2 * dt * (z ** 2 - 1.0))
    return np.maximum(out, 0.0)


def feller_ok(kappa: float, theta: float, sigma: float) -> bool:
    """Feller condition 2 kappa theta >= sigma^2: the CIR rate never touches zero."""
    return 2 * kappa * theta >= sigma ** 2

# ----------------------------------------------------------------------------- Hull-White


class HullWhite:
    """One-factor Hull-White model fitted exactly to today's zero curve.

    Write r(t) = x(t) + alpha(t) with dx = -a x dt + sigma dW, x(0) = 0 and
    alpha(t) = f(0,t) + sigma^2/(2a^2) (1 - e^{-a t})^2, where f(0,t) is today's
    instantaneous forward rate. Then the model reprices every zero-coupon bond on the
    input curve by construction.
    """

    def __init__(self, tenors: np.ndarray, zeros: np.ndarray, a: float, sigma: float):
        from scipy.interpolate import CubicSpline
        self.a, self.sigma = a, sigma
        self._lnP = CubicSpline(tenors, -zeros * tenors, extrapolate=True)
        self.tmax = tenors[-1]

    def P0(self, T):
        """Today's discount factor P(0, T) from the curve."""
        return np.exp(self._lnP(np.asarray(T, dtype=float)))

    def f0(self, t):
        """Instantaneous forward rate f(0, t) = -d ln P(0,t) / dt."""
        return -self._lnP(np.asarray(t, dtype=float), 1)

    def B(self, t, T):
        return (1.0 - np.exp(-self.a * (np.asarray(T) - np.asarray(t)))) / self.a

    def alpha(self, t):
        a, s = self.a, self.sigma
        t = np.asarray(t, dtype=float)
        return self.f0(t) + s ** 2 / (2 * a ** 2) * (1.0 - np.exp(-a * t)) ** 2

    def bond(self, t: float, T: float, r_t: np.ndarray) -> np.ndarray:
        """P(t, T) given the short rate r(t) on each path (analytic in Hull-White)."""
        a, s = self.a, self.sigma
        B = self.B(t, T)
        lnA = (np.log(self.P0(T) / self.P0(t)) + B * self.f0(t)
               - s ** 2 / (4 * a) * (1.0 - np.exp(-2 * a * t)) * B ** 2)
        return np.exp(lnA - B * r_t)

    def zbp(self, T: float, S: float, K: float) -> float:
        """Price today of a European put, expiry T, on a zero-coupon bond maturing S."""
        from scipy.stats import norm
        a, s = self.a, self.sigma
        sp = s / a * (1 - np.exp(-a * (S - T))) * np.sqrt((1 - np.exp(-2 * a * T)) / (2 * a))
        h = np.log(self.P0(S) / (self.P0(T) * K)) / sp + sp / 2
        return float(K * self.P0(T) * norm.cdf(-h + sp) - self.P0(S) * norm.cdf(-h))

    def caplet(self, T: float, tau: float, K: float) -> float:
        """Caplet on the simple rate over [T, T+tau], strike K, notional 1, paid at T+tau.
        Equals (1 + K tau) puts on a zero-coupon bond with strike 1/(1 + K tau)."""
        return (1 + K * tau) * self.zbp(T, T + tau, 1.0 / (1 + K * tau))

    def convexity_adjustment(self, T1: float, T2: float) -> float:
        """Futures rate minus forward rate (continuously compounded) for a futures
        contract on the rate over [T1, T2] (Hull's formula for Hull-White).
        Positive: daily margining makes a short futures position worth more than a
        forward, so futures rates sit above forward rates."""
        a, s = self.a, self.sigma
        tau = T2 - T1
        B12 = self.B(T1, T2)
        B01 = self.B(0.0, T1)
        return float(B12 / tau * (B12 * (1 - np.exp(-2 * a * T1)) + 2 * a * B01 ** 2)
                     * s ** 2 / (4 * a))
