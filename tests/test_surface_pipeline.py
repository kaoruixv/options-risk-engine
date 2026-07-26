"""End-to-end volatility-surface pipeline tests."""
import pandas as pd

from options_risk_engine.pricing.black_scholes import black_scholes
from options_risk_engine.vol_surface.pipeline import (
    build_surface_from_chain,
    build_surface_from_yahoo,
)


def _fake_chain(spot=100.0, rate=0.05, sigma=0.20) -> pd.DataFrame:
    rows = []
    for time_to_expiry in [0.5, 1.0]:
        for strike in [90.0, 100.0, 110.0]:
            price = black_scholes(
                S=spot,
                K=strike,
                T=time_to_expiry,
                r=rate,
                sigma=sigma,
                option_type="call",
            ).price
            rows.append(
                {
                    "symbol": "FAKE",
                    "expiry": "2030-01-17",
                    "option_type": "call",
                    "strike": strike,
                    "time_to_expiry": time_to_expiry,
                    "bid": price - 0.01,
                    "ask": price + 0.01,
                }
            )
    return pd.DataFrame(rows)


def _fake_fetcher(symbol, valuation_date=None, expiries=None):
    return _fake_chain()


def test_build_surface_from_chain_returns_points_and_matrix():
    result = build_surface_from_chain(
        option_chain=_fake_chain(),
        spot=100.0,
        rate=0.05,
        option_type="call",
    )

    assert result.accepted_rows == 6
    assert result.rejected_rows == 0
    assert result.matrix.shape == (2, 3)


def test_build_surface_from_yahoo_uses_injected_fetcher():
    result = build_surface_from_yahoo(
        symbol="FAKE",
        spot=100.0,
        rate=0.05,
        valuation_date="2029-01-17",
        expiries=["2030-01-17"],
        option_type="call",
        fetcher=_fake_fetcher,
    )

    assert result.input_rows == 6
    assert result.accepted_rows == 6
    assert result.points["implied_vol"].sub(0.20).abs().max() < 1e-8
