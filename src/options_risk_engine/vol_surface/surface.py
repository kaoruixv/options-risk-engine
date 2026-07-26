"""Implied-volatility surface construction.

This module converts option-chain rows into clean implied-volatility points.
It deliberately separates three steps:

1. validate quotes
2. invert each quote into Black-Scholes implied volatility
3. reshape the point cloud into an expiry-by-strike matrix
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from options_risk_engine.pricing.implied_vol import implied_volatility


REQUIRED_COLUMNS = {
    "option_type",
    "strike",
    "time_to_expiry",
    "bid",
    "ask",
}


@dataclass(frozen=True)
class VolSurfaceBuildResult:
    points: pd.DataFrame
    rejected: pd.DataFrame
    input_rows: int
    accepted_rows: int
    rejected_rows: int


def _validate_required_columns(option_chain: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(option_chain.columns)
    if missing:
        raise ValueError(f"option_chain is missing required columns: {sorted(missing)}")


def _mid_price(bid: float, ask: float) -> float:
    return 0.5 * (bid + ask)


def build_vol_surface(
    option_chain: pd.DataFrame,
    spot: float,
    rate: float,
    dividend_yield: float = 0.0,
    min_price: float = 0.01,
    max_spread_pct: float = 0.50,
    min_time_to_expiry: float = 1.0 / 365.0,
) -> VolSurfaceBuildResult:
    """Build implied-volatility points from option-chain rows.

    Expected input columns
    ----------------------
    option_type:
        "call" or "put".
    strike:
        Option strike price.
    time_to_expiry:
        Time to expiry in years.
    bid:
        Quoted bid price.
    ask:
        Quoted ask price.

    Output
    ------
    points:
        Clean rows with computed implied volatility.
    rejected:
        Rejected rows with a rejection reason.
    """
    if spot <= 0:
        raise ValueError("spot must be positive")
    if min_price <= 0:
        raise ValueError("min_price must be positive")
    if max_spread_pct <= 0:
        raise ValueError("max_spread_pct must be positive")

    _validate_required_columns(option_chain)

    accepted: list[dict] = []
    rejected: list[dict] = []

    for row_index, row in option_chain.reset_index(drop=False).iterrows():
        original_index = row["index"]

        try:
            option_type = str(row["option_type"]).lower()
            strike = float(row["strike"])
            time_to_expiry = float(row["time_to_expiry"])
            bid = float(row["bid"])
            ask = float(row["ask"])

            if option_type not in {"call", "put"}:
                raise ValueError("option_type must be call or put")
            if strike <= 0:
                raise ValueError("strike must be positive")
            if time_to_expiry < min_time_to_expiry:
                raise ValueError("time_to_expiry is too small")
            if bid < 0 or ask < 0:
                raise ValueError("bid and ask must be non-negative")
            if ask < bid:
                raise ValueError("ask must be greater than or equal to bid")

            market_price = _mid_price(bid, ask)
            if market_price < min_price:
                raise ValueError("mid price is below min_price")

            spread_pct = (ask - bid) / market_price
            if spread_pct > max_spread_pct:
                raise ValueError("bid-ask spread is too wide")

            iv = implied_volatility(
                market_price=market_price,
                S=spot,
                K=strike,
                T=time_to_expiry,
                r=rate,
                option_type=option_type,
                q=dividend_yield,
            )

            accepted.append(
                {
                    "source_index": original_index,
                    "option_type": option_type,
                    "strike": strike,
                    "time_to_expiry": time_to_expiry,
                    "bid": bid,
                    "ask": ask,
                    "market_price": market_price,
                    "spread_pct": spread_pct,
                    "moneyness": strike / spot,
                    "implied_vol": iv.implied_vol,
                    "model_price": iv.model_price,
                    "absolute_error": iv.absolute_error,
                    "converged": iv.converged,
                }
            )

        except Exception as exc:
            rejected.append(
                {
                    "source_index": original_index,
                    "row_number": row_index,
                    "reason": str(exc),
                }
            )

    points = pd.DataFrame(accepted)
    rejected_df = pd.DataFrame(rejected)

    return VolSurfaceBuildResult(
        points=points,
        rejected=rejected_df,
        input_rows=len(option_chain),
        accepted_rows=len(points),
        rejected_rows=len(rejected_df),
    )


def surface_matrix(
    points: pd.DataFrame,
    option_type: str = "call",
    value_col: str = "implied_vol",
) -> pd.DataFrame:
    """Return an expiry-by-strike matrix from clean volatility points."""
    if points.empty:
        return pd.DataFrame()

    required = {"option_type", "time_to_expiry", "strike", value_col}
    missing = required.difference(points.columns)
    if missing:
        raise ValueError(f"points is missing required columns: {sorted(missing)}")

    filtered = points[points["option_type"] == option_type.lower()]
    if filtered.empty:
        return pd.DataFrame()

    return filtered.pivot_table(
        index="time_to_expiry",
        columns="strike",
        values=value_col,
        aggfunc="mean",
    ).sort_index().sort_index(axis=1)
