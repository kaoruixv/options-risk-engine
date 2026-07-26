"""Monte Carlo European option pricer with antithetic variates.

The simulation uses the closed-form terminal distribution of geometric
Brownian motion, so it is suitable for vanilla European calls and puts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from options_risk_engine.config import pricing as pricing_cfg
from options_risk_engine.pricing.black_scholes import OptionType


@dataclass(frozen=True)
class MCResult:
    price: float
    std_error: float
    n_paths: int
    ci_95_lower: float
    ci_95_upper: float


def monte_carlo_european(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType | str = OptionType.CALL,
    n_paths: int | None = None,
    seed: int | None = None,
    q: float = 0.0,
) -> MCResult:
    """Price a European option using Monte Carlo with antithetic variates."""
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive")
    if T <= 0:
        raise ValueError("T must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    opt = OptionType(option_type)
    requested_paths = n_paths if n_paths is not None else pricing_cfg.mc_paths
    if requested_paths < 2:
        raise ValueError("n_paths must be at least 2")

    half = requested_paths // 2
    rng = np.random.default_rng(seed if seed is not None else pricing_cfg.mc_seed)

    z = rng.standard_normal(half)
    z_all = np.concatenate([z, -z])

    drift = (r - q - 0.5 * sigma**2) * T
    diffusion = sigma * math.sqrt(T)
    terminal_spot = S * np.exp(drift + diffusion * z_all)

    if opt is OptionType.CALL:
        payoffs = np.maximum(terminal_spot - K, 0.0)
    else:
        payoffs = np.maximum(K - terminal_spot, 0.0)

    discount = math.exp(-r * T)
    paired_payoffs = 0.5 * discount * (payoffs[:half] + payoffs[half:])

    price = float(paired_payoffs.mean())
    std_error = float(paired_payoffs.std(ddof=1) / math.sqrt(half))

    return MCResult(
        price=price,
        std_error=std_error,
        n_paths=2 * half,
        ci_95_lower=price - 1.96 * std_error,
        ci_95_upper=price + 1.96 * std_error,
    )
