"""End-to-end volatility-surface pipeline.

This module connects market-data ingestion with implied-volatility surface
construction. It is intentionally small, testable, and dependency-injected
so tests do not need live network calls.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date

import pandas as pd

from options_risk_engine.market_data.yahoo import fetch_yahoo_option_chain
from options_risk_engine.vol_surface.surface import build_vol_surface, surface_matrix


@dataclass(frozen=True)
class SurfacePipelineResult:
    option_chain: pd.DataFrame
    points: pd.DataFrame
    rejected: pd.DataFrame
    matrix: pd.DataFrame
    input_rows: int
    accepted_rows: int
    rejected_rows: int


def build_surface_from_chain(
    option_chain: pd.DataFrame,
    spot: float,
    rate: float,
    dividend_yield: float = 0.0,
    option_type: str = "call",
    min_price: float = 0.01,
    max_spread_pct: float = 0.50,
    min_time_to_expiry: float = 1.0 / 365.0,
) -> SurfacePipelineResult:
    """Build a volatility surface from an already-normalized option chain."""
    build_result = build_vol_surface(
        option_chain=option_chain,
        spot=spot,
        rate=rate,
        dividend_yield=dividend_yield,
        min_price=min_price,
        max_spread_pct=max_spread_pct,
        min_time_to_expiry=min_time_to_expiry,
    )

    matrix = surface_matrix(build_result.points, option_type=option_type)

    return SurfacePipelineResult(
        option_chain=option_chain,
        points=build_result.points,
        rejected=build_result.rejected,
        matrix=matrix,
        input_rows=build_result.input_rows,
        accepted_rows=build_result.accepted_rows,
        rejected_rows=build_result.rejected_rows,
    )


def build_surface_from_yahoo(
    symbol: str,
    spot: float,
    rate: float,
    valuation_date: date | str | pd.Timestamp | None = None,
    expiries: Iterable[str] | None = None,
    dividend_yield: float = 0.0,
    option_type: str = "call",
    fetcher: Callable[..., pd.DataFrame] = fetch_yahoo_option_chain,
    min_price: float = 0.01,
    max_spread_pct: float = 0.50,
    min_time_to_expiry: float = 1.0 / 365.0,
) -> SurfacePipelineResult:
    """Fetch Yahoo option-chain rows and build an implied-volatility surface."""
    option_chain = fetcher(
        symbol=symbol,
        valuation_date=valuation_date,
        expiries=expiries,
    )

    return build_surface_from_chain(
        option_chain=option_chain,
        spot=spot,
        rate=rate,
        dividend_yield=dividend_yield,
        option_type=option_type,
        min_price=min_price,
        max_spread_pct=max_spread_pct,
        min_time_to_expiry=min_time_to_expiry,
    )
