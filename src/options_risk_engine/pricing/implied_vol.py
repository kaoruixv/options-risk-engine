"""Implied volatility solver for European options.

The solver inverts the Black-Scholes price using a robust Brent root finder.
This module is the bridge between single-option pricing and volatility-surface
construction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.optimize import root_scalar

from options_risk_engine.config import pricing as pricing_cfg
from options_risk_engine.pricing.black_scholes import OptionType, black_scholes


@dataclass(frozen=True)
class ImpliedVolResult:
    implied_vol: float
    model_price: float
    market_price: float
    absolute_error: float
    iterations: int
    converged: bool


def _no_arbitrage_bounds(
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType,
    q: float,
) -> tuple[float, float]:
    spot_pv = S * math.exp(-q * T)
    strike_pv = K * math.exp(-r * T)

    if option_type is OptionType.CALL:
        return max(spot_pv - strike_pv, 0.0), spot_pv

    return max(strike_pv - spot_pv, 0.0), strike_pv


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType | str = OptionType.CALL,
    q: float = 0.0,
    lower_vol: float = 1e-6,
    upper_vol: float = 5.0,
    tol: float | None = None,
    max_iter: int | None = None,
) -> ImpliedVolResult:
    """Return the Black-Scholes implied volatility for a market option price.

    Parameters
    ----------
    market_price:
        Observed option price.
    S:
        Spot price.
    K:
        Strike price.
    T:
        Time to expiry in years.
    r:
        Continuously compounded risk-free rate.
    option_type:
        "call" or "put".
    q:
        Continuous dividend yield.
    lower_vol:
        Lower volatility bracket.
    upper_vol:
        Upper volatility bracket.
    """
    opt = OptionType(option_type)

    if market_price <= 0:
        raise ValueError("market_price must be positive")
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive")
    if T <= 0:
        raise ValueError("T must be positive")
    if lower_vol <= 0 or upper_vol <= lower_vol:
        raise ValueError("volatility bracket must satisfy 0 < lower_vol < upper_vol")

    lower_bound, upper_bound = _no_arbitrage_bounds(S, K, T, r, opt, q)
    tolerance = tol if tol is not None else pricing_cfg.iv_tol
    iterations = max_iter if max_iter is not None else pricing_cfg.iv_max_iter

    eps = 1e-10
    if market_price < lower_bound - eps or market_price > upper_bound + eps:
        raise ValueError(
            "market_price violates no-arbitrage bounds: "
            f"{lower_bound:.8f} <= price <= {upper_bound:.8f}"
        )

    def objective(vol: float) -> float:
        return black_scholes(S, K, T, r, vol, opt, q).price - market_price

    low_value = objective(lower_vol)
    high_value = objective(upper_vol)

    if abs(low_value) < tolerance:
        iv = lower_vol
        model_price = black_scholes(S, K, T, r, iv, opt, q).price
        return ImpliedVolResult(
            implied_vol=iv,
            model_price=model_price,
            market_price=market_price,
            absolute_error=abs(model_price - market_price),
            iterations=0,
            converged=True,
        )

    if abs(high_value) < tolerance:
        iv = upper_vol
        model_price = black_scholes(S, K, T, r, iv, opt, q).price
        return ImpliedVolResult(
            implied_vol=iv,
            model_price=model_price,
            market_price=market_price,
            absolute_error=abs(model_price - market_price),
            iterations=0,
            converged=True,
        )

    if low_value * high_value > 0:
        raise ValueError(
            "could not bracket implied volatility; "
            "increase upper_vol or check market_price"
        )

    root = root_scalar(
        objective,
        bracket=(lower_vol, upper_vol),
        method="brentq",
        xtol=tolerance,
        maxiter=iterations,
    )

    iv = float(root.root)
    model_price = black_scholes(S, K, T, r, iv, opt, q).price

    return ImpliedVolResult(
        implied_vol=iv,
        model_price=model_price,
        market_price=market_price,
        absolute_error=abs(model_price - market_price),
        iterations=int(root.iterations),
        converged=bool(root.converged),
    )
