"""Black-Scholes closed-form pricing and Greeks for European options.

Assumptions: constant vol and rate, continuous dividend yield q,
European exercise, no transaction costs.

Greek scaling follows market convention:
    theta : per calendar day (annualised theta / 365)
    vega  : per 1-point (1%) move in implied vol
    rho   : per 1-point (1%) move in the risk-free rate
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from scipy.stats import norm


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


@dataclass(frozen=True)
class BSGreeks:
    delta: float   # dV/dS
    gamma: float   # d2V/dS2
    vega: float    # dV/dsigma, per 1% move
    theta: float   # dV/dt, per calendar day (negative = decay)
    rho: float     # dV/dr, per 1% rate move
    vanna: float   # d2V/dS dsigma
    volga: float   # d2V/dsigma2


@dataclass(frozen=True)
class BSResult:
    price: float
    greeks: BSGreeks
    d1: float
    d2: float
    intrinsic: float
    time_value: float


def _d1_d2(
    S: float, K: float, T: float, r: float, sigma: float, q: float
) -> tuple[float, float]:
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if S <= 0 or K <= 0:
        raise ValueError(f"S and K must be positive, got S={S} K={K}")
    vol_sqrt_t = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def black_scholes(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType | str = OptionType.CALL,
    q: float = 0.0,
) -> BSResult:
    """Price a European option and return the price plus all Greeks.

    Parameters
    ----------
    S           : spot price
    K           : strike
    T           : time to expiry in years
    r           : continuously compounded risk-free rate (0.05 = 5%)
    sigma       : implied volatility (0.20 = 20%)
    option_type : "call" or "put"
    q           : continuous dividend yield
    """
    opt = OptionType(option_type)
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)

    n_d1 = norm.cdf(d1)
    n_d2 = norm.cdf(d2)
    n_neg_d1 = norm.cdf(-d1)
    n_neg_d2 = norm.cdf(-d2)
    pdf_d1 = norm.pdf(d1)

    disc = math.exp(-r * T)
    div_disc = math.exp(-q * T)
    sqrt_t = math.sqrt(T)

    if opt is OptionType.CALL:
        price = S * div_disc * n_d1 - K * disc * n_d2
    else:
        price = K * disc * n_neg_d2 - S * div_disc * n_neg_d1

    # Identical for calls and puts by put-call parity
    gamma = (div_disc * pdf_d1) / (S * sigma * sqrt_t)
    vega_raw = S * div_disc * pdf_d1 * sqrt_t
    vega = vega_raw / 100.0
    vanna = -(div_disc * pdf_d1 * d2) / sigma
    volga = vega_raw * d1 * d2 / sigma

    if opt is OptionType.CALL:
        delta = div_disc * n_d1
        theta_annual = (
            -(S * div_disc * pdf_d1 * sigma) / (2.0 * sqrt_t)
            - r * K * disc * n_d2
            + q * S * div_disc * n_d1
        )
        rho_raw = K * T * disc * n_d2
    else:
        delta = div_disc * (n_d1 - 1.0)
        theta_annual = (
            -(S * div_disc * pdf_d1 * sigma) / (2.0 * sqrt_t)
            + r * K * disc * n_neg_d2
            - q * S * div_disc * n_neg_d1
        )
        rho_raw = -K * T * disc * n_neg_d2

    intrinsic = max(S - K, 0.0) if opt is OptionType.CALL else max(K - S, 0.0)

    return BSResult(
        price=price,
        greeks=BSGreeks(
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta_annual / 365.0,
            rho=rho_raw / 100.0,
            vanna=vanna,
            volga=volga,
        ),
        d1=d1,
        d2=d2,
        intrinsic=intrinsic,
        time_value=price - intrinsic,
    )


def put_call_parity_check(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    tol: float = 1e-10,
) -> bool:
    """Return True if C - P == S*exp(-qT) - K*exp(-rT) within tol."""
    call = black_scholes(S, K, T, r, sigma, OptionType.CALL, q).price
    put = black_scholes(S, K, T, r, sigma, OptionType.PUT, q).price
    return abs((call - put) - (S * math.exp(-q * T) - K * math.exp(-r * T))) < tol
