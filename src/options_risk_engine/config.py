"""Centralised configuration.

All credentials and tunable constants live here. Never hardcode an API key
or a magic number in module code — import from this file instead.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)


@dataclass(frozen=True)
class PricingConfig:
    """Default numerical parameters for the pricers."""

    mc_paths: int = 100_000
    mc_seed: int = 42
    binomial_steps: int = 200
    iv_max_iter: int = 100
    iv_tol: float = 1e-8


@dataclass(frozen=True)
class MarketDataConfig:
    """Parameters for pulling option chain data."""

    proxy: str | None = field(default_factory=lambda: os.getenv("MARKET_DATA_PROXY"))
    request_timeout: int = 10


@dataclass(frozen=True)
class BacktestConfig:
    """Parameters for the delta-hedge backtest."""

    transaction_cost_bps: float = 5.0
    bid_ask_half_spread: float = 0.001


pricing = PricingConfig()
market_data = MarketDataConfig()
backtest = BacktestConfig()
