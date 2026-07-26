"""Portfolio-level option risk aggregation.

This module turns single-contract pricing results into portfolio-level
value and Greeks. It is intentionally model-agnostic at the portfolio layer:
each position is priced with Black-Scholes, then values are scaled by
quantity and contract multiplier.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from options_risk_engine.pricing.black_scholes import OptionType, black_scholes


@dataclass(frozen=True)
class OptionPosition:
    symbol: str
    option_type: OptionType | str
    quantity: float
    S: float
    K: float
    T: float
    r: float
    sigma: float
    q: float = 0.0
    contract_multiplier: float = 100.0


@dataclass(frozen=True)
class GreekSummary:
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    vanna: float
    volga: float


@dataclass(frozen=True)
class PortfolioRiskResult:
    total_value: float
    greeks: GreekSummary
    positions: pd.DataFrame


def _position_row(position: OptionPosition) -> dict:
    result = black_scholes(
        S=position.S,
        K=position.K,
        T=position.T,
        r=position.r,
        sigma=position.sigma,
        option_type=position.option_type,
        q=position.q,
    )

    scale = position.quantity * position.contract_multiplier
    greeks = result.greeks

    return {
        "symbol": position.symbol,
        "option_type": OptionType(position.option_type).value,
        "quantity": position.quantity,
        "contract_multiplier": position.contract_multiplier,
        "spot": position.S,
        "strike": position.K,
        "time_to_expiry": position.T,
        "rate": position.r,
        "sigma": position.sigma,
        "unit_price": result.price,
        "market_value": result.price * scale,
        "delta": greeks.delta * scale,
        "gamma": greeks.gamma * scale,
        "vega": greeks.vega * scale,
        "theta": greeks.theta * scale,
        "rho": greeks.rho * scale,
        "vanna": greeks.vanna * scale,
        "volga": greeks.volga * scale,
    }


def aggregate_portfolio_risk(positions: list[OptionPosition]) -> PortfolioRiskResult:
    """Aggregate value and Greeks across a list of option positions."""
    if not positions:
        empty = pd.DataFrame()
        return PortfolioRiskResult(
            total_value=0.0,
            greeks=GreekSummary(
                delta=0.0,
                gamma=0.0,
                vega=0.0,
                theta=0.0,
                rho=0.0,
                vanna=0.0,
                volga=0.0,
            ),
            positions=empty,
        )

    rows = [_position_row(position) for position in positions]
    frame = pd.DataFrame(rows)

    return PortfolioRiskResult(
        total_value=float(frame["market_value"].sum()),
        greeks=GreekSummary(
            delta=float(frame["delta"].sum()),
            gamma=float(frame["gamma"].sum()),
            vega=float(frame["vega"].sum()),
            theta=float(frame["theta"].sum()),
            rho=float(frame["rho"].sum()),
            vanna=float(frame["vanna"].sum()),
            volga=float(frame["volga"].sum()),
        ),
        positions=frame,
    )


def scenario_pnl(
    positions: list[OptionPosition],
    spot_pct_change: float = 0.0,
    vol_abs_change: float = 0.0,
    rate_abs_change: float = 0.0,
    time_passed_years: float = 0.0,
) -> float:
    """Return revaluation PnL under a simple market scenario.

    Parameters
    ----------
    spot_pct_change:
        Relative spot move. Example: -0.05 means spot falls 5%.
    vol_abs_change:
        Absolute volatility move. Example: 0.02 means vol rises from 20%
        to 22%.
    rate_abs_change:
        Absolute rate move. Example: 0.01 means rates rise by 1 percentage point.
    time_passed_years:
        Calendar time passed in years. The remaining option maturity is floored
        at one day to keep Black-Scholes well-defined.
    """
    base = aggregate_portfolio_risk(positions).total_value

    shocked_positions = [
        OptionPosition(
            symbol=position.symbol,
            option_type=position.option_type,
            quantity=position.quantity,
            S=position.S * (1.0 + spot_pct_change),
            K=position.K,
            T=max(position.T - time_passed_years, 1.0 / 365.0),
            r=position.r + rate_abs_change,
            sigma=max(position.sigma + vol_abs_change, 1e-6),
            q=position.q,
            contract_multiplier=position.contract_multiplier,
        )
        for position in positions
    ]

    shocked = aggregate_portfolio_risk(shocked_positions).total_value
    return float(shocked - base)
