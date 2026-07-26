"""Discrete delta-hedging backtest for European options.

The simulation models the PnL of a short option position hedged with
Black-Scholes delta. At inception, the strategy sells one option, buys
delta shares, and finances the hedge through a cash account. The hedge is
rebalanced at discrete time steps until expiry.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from options_risk_engine.config import backtest as backtest_cfg
from options_risk_engine.config import pricing as pricing_cfg
from options_risk_engine.pricing.black_scholes import OptionType, black_scholes


@dataclass(frozen=True)
class HedgingSummary:
    mean_pnl: float
    std_pnl: float
    min_pnl: float
    max_pnl: float
    p05_pnl: float
    p50_pnl: float
    p95_pnl: float
    mean_transaction_cost: float


@dataclass(frozen=True)
class HedgingBacktestResult:
    summary: HedgingSummary
    pnl: pd.DataFrame
    n_paths: int
    n_steps: int
    option_type: str


def _simulate_gbm_paths(
    S0: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
    n_steps: int,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    dt = T / n_steps
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((n_paths, n_steps))

    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = S0

    drift = (r - q - 0.5 * sigma**2) * dt
    diffusion = sigma * math.sqrt(dt)

    for step in range(n_steps):
        paths[:, step + 1] = paths[:, step] * np.exp(drift + diffusion * shocks[:, step])

    return paths


def _vectorized_delta(
    spot: np.ndarray,
    K: float,
    tau: float,
    r: float,
    sigma: float,
    option_type: OptionType,
    q: float,
) -> np.ndarray:
    if tau <= 0:
        if option_type is OptionType.CALL:
            return (spot > K).astype(float)
        return -(spot < K).astype(float)

    d1 = (np.log(spot / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * math.sqrt(tau))

    if option_type is OptionType.CALL:
        return np.exp(-q * tau) * norm.cdf(d1)

    return np.exp(-q * tau) * (norm.cdf(d1) - 1.0)


def simulate_delta_hedge(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType | str = OptionType.CALL,
    q: float = 0.0,
    n_steps: int = 50,
    n_paths: int | None = None,
    seed: int | None = None,
    transaction_cost_bps: float | None = None,
) -> HedgingBacktestResult:
    """Simulate discrete delta hedging PnL for a short European option.

    Positive PnL means the hedge outperformed the option payoff after
    financing and transaction costs. With zero transaction costs and many
    rebalancing steps, the average PnL should be close to zero.
    """
    if S0 <= 0 or K <= 0:
        raise ValueError("S0 and K must be positive")
    if T <= 0:
        raise ValueError("T must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if n_steps < 2:
        raise ValueError("n_steps must be at least 2")

    paths_count = n_paths if n_paths is not None else pricing_cfg.mc_paths
    if paths_count < 2:
        raise ValueError("n_paths must be at least 2")

    rng_seed = seed if seed is not None else pricing_cfg.mc_seed
    cost_bps = (
        transaction_cost_bps
        if transaction_cost_bps is not None
        else backtest_cfg.transaction_cost_bps
    )
    if cost_bps < 0:
        raise ValueError("transaction_cost_bps must be non-negative")

    opt = OptionType(option_type)
    paths = _simulate_gbm_paths(S0, T, r, sigma, q, n_steps, paths_count, rng_seed)
    dt = T / n_steps

    initial = black_scholes(S0, K, T, r, sigma, opt, q)
    shares = np.full(paths_count, initial.greeks.delta)
    cash = np.full(paths_count, initial.price - initial.greeks.delta * S0)
    total_cost = np.zeros(paths_count)

    for step in range(1, n_steps):
        cash *= math.exp(r * dt)

        tau = T - step * dt
        spot = paths[:, step]
        new_delta = _vectorized_delta(spot, K, tau, r, sigma, opt, q)

        trade = new_delta - shares
        trade_value = trade * spot
        trade_cost = np.abs(trade_value) * cost_bps / 10_000.0

        cash -= trade_value + trade_cost
        total_cost += trade_cost
        shares = new_delta

    cash *= math.exp(r * dt)
    terminal_spot = paths[:, -1]

    if opt is OptionType.CALL:
        payoff = np.maximum(terminal_spot - K, 0.0)
    else:
        payoff = np.maximum(K - terminal_spot, 0.0)

    pnl = cash + shares * terminal_spot - payoff

    pnl_frame = pd.DataFrame(
        {
            "path": np.arange(paths_count),
            "terminal_spot": terminal_spot,
            "pnl": pnl,
            "transaction_cost": total_cost,
        }
    )

    summary = HedgingSummary(
        mean_pnl=float(np.mean(pnl)),
        std_pnl=float(np.std(pnl, ddof=1)),
        min_pnl=float(np.min(pnl)),
        max_pnl=float(np.max(pnl)),
        p05_pnl=float(np.quantile(pnl, 0.05)),
        p50_pnl=float(np.quantile(pnl, 0.50)),
        p95_pnl=float(np.quantile(pnl, 0.95)),
        mean_transaction_cost=float(np.mean(total_cost)),
    )

    return HedgingBacktestResult(
        summary=summary,
        pnl=pnl_frame,
        n_paths=paths_count,
        n_steps=n_steps,
        option_type=opt.value,
    )
