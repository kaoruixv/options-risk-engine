"""Volatility-surface construction tests."""
import pandas as pd
import pytest

from options_risk_engine.pricing.black_scholes import black_scholes
from options_risk_engine.vol_surface.surface import build_vol_surface, surface_matrix


def _synthetic_chain(spot=100.0, rate=0.05, sigma=0.20) -> pd.DataFrame:
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
                    "option_type": "call",
                    "strike": strike,
                    "time_to_expiry": time_to_expiry,
                    "bid": price - 0.01,
                    "ask": price + 0.01,
                }
            )
    return pd.DataFrame(rows)


def test_build_vol_surface_recovers_synthetic_iv():
    chain = _synthetic_chain()
    result = build_vol_surface(chain, spot=100.0, rate=0.05)

    assert result.input_rows == 6
    assert result.accepted_rows == 6
    assert result.rejected_rows == 0
    assert result.points["implied_vol"].sub(0.20).abs().max() < 1e-8


def test_surface_matrix_has_expiry_by_strike_shape():
    chain = _synthetic_chain()
    result = build_vol_surface(chain, spot=100.0, rate=0.05)
    matrix = surface_matrix(result.points, option_type="call")

    assert list(matrix.index) == [0.5, 1.0]
    assert list(matrix.columns) == [90.0, 100.0, 110.0]
    assert matrix.shape == (2, 3)


def test_wide_spread_quote_is_rejected():
    chain = pd.DataFrame(
        [
            {
                "option_type": "call",
                "strike": 100.0,
                "time_to_expiry": 1.0,
                "bid": 1.0,
                "ask": 10.0,
            }
        ]
    )

    result = build_vol_surface(chain, spot=100.0, rate=0.05, max_spread_pct=0.20)

    assert result.accepted_rows == 0
    assert result.rejected_rows == 1
    assert "spread" in result.rejected.iloc[0]["reason"]


def test_missing_required_columns_raise():
    with pytest.raises(ValueError, match="missing required columns"):
        build_vol_surface(pd.DataFrame({"strike": [100.0]}), spot=100.0, rate=0.05)
