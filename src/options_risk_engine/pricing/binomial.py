"""Cox-Ross-Rubinstein binomial tree pricer for American and European options.

Greeks are computed by bump-and-reprice rather than read off the tree nodes,
which keeps the estimate consistent between American and European modes.

Expected behaviour, verified in tests:
  - European binomial converges to Black-Scholes as n_steps grows
  - American put >= European put (early-exercise premium is non-negative)
  - American call == European call when q = 0 (never optimal to exercise early)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from options_risk_engine.pricing.black_scholes import OptionType


@dataclass(frozen=True)
class BinomialResult:
    price: float
    delta: float
    gamma: float
    n_steps: int
    american: bool


def _tree_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    opt: OptionType,
    n_steps: int,
    q: float,
    american: bool,
) -> float:
    """Backward induction through the tree. Returns the option value at t=0."""
    dt = T / n_steps
    disc = math.exp(-r * dt)
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p_up = (math.exp((r - q) * dt) - d) / (u - d)
    p_down = 1.0 - p_up

    if not 0.0 < p_up < 1.0:
        raise ValueError(
            f"Risk-neutral probability {p_up:.4f} outside (0,1); "
            "increase n_steps or check inputs"
        )

    def payoff(spot: np.ndarray) -> np.ndarray:
        if opt is OptionType.CALL:
            return np.maximum(spot - K, 0.0)
        return np.maximum(K - spot, 0.0)

    j = np.arange(n_steps + 1, dtype=np.float64)
    values = payoff(S * u ** (n_steps - j) * d**j)

    for step in range(n_steps - 1, -1, -1):
        continuation = disc * (p_up * values[:-1] + p_down * values[1:])
        if american:
            j_step = np.arange(step + 1, dtype=np.float64)
            spot_step = S * u ** (step - j_step) * d**j_step
            values = np.maximum(continuation, payoff(spot_step))
        else:
            values = continuation

    return float(values[0])


def binomial_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType | str = OptionType.PUT,
    n_steps: int = 200,
    q: float = 0.0,
    american: bool = True,
) -> BinomialResult:
    """Price an option via the CRR binomial tree.

    Parameters
    ----------
    american : bool
        True checks early exercise at every node. False gives European
        exercise, which converges to the Black-Scholes price.
    """
    opt = OptionType(option_type)
    price = _tree_price(S, K, T, r, sigma, opt, n_steps, q, american)

    h = S * 0.01
    up = _tree_price(S + h, K, T, r, sigma, opt, n_steps, q, american)
    down = _tree_price(S - h, K, T, r, sigma, opt, n_steps, q, american)

    return BinomialResult(
        price=price,
        delta=(up - down) / (2.0 * h),
        gamma=(up - 2.0 * price + down) / h**2,
        n_steps=n_steps,
        american=american,
    )
