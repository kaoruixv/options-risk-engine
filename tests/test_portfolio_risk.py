"""Portfolio risk aggregation tests."""
import pytest

from options_risk_engine.pricing.black_scholes import black_scholes
from options_risk_engine.risk.portfolio import (
    OptionPosition,
    aggregate_portfolio_risk,
    scenario_pnl,
)


def test_single_position_value_matches_black_scholes():
    position = OptionPosition(
        symbol="AAPL",
        option_type="call",
        quantity=2,
        S=100.0,
        K=100.0,
        T=1.0,
        r=0.05,
        sigma=0.20,
    )

    result = aggregate_portfolio_risk([position])
    expected_unit = black_scholes(100.0, 100.0, 1.0, 0.05, 0.20, "call").price

    assert result.total_value == pytest.approx(expected_unit * 2 * 100.0)
    assert len(result.positions) == 1


def test_synthetic_forward_delta_from_call_minus_put():
    positions = [
        OptionPosition("TEST", "call", 1, 100.0, 100.0, 1.0, 0.05, 0.20),
        OptionPosition("TEST", "put", -1, 100.0, 100.0, 1.0, 0.05, 0.20),
    ]

    result = aggregate_portfolio_risk(positions)

    assert result.greeks.delta == pytest.approx(100.0, abs=1e-8)
    assert abs(result.greeks.gamma) < 1e-8
    assert abs(result.greeks.vega) < 1e-8


def test_empty_portfolio_returns_zeroes():
    result = aggregate_portfolio_risk([])

    assert result.total_value == 0.0
    assert result.greeks.delta == 0.0
    assert result.positions.empty


def test_scenario_pnl_responds_to_spot_move_for_long_call():
    positions = [
        OptionPosition("TEST", "call", 1, 100.0, 100.0, 1.0, 0.05, 0.20),
    ]

    up_pnl = scenario_pnl(positions, spot_pct_change=0.05)
    down_pnl = scenario_pnl(positions, spot_pct_change=-0.05)

    assert up_pnl > 0
    assert down_pnl < 0


def test_scenario_pnl_responds_to_vol_move_for_long_option():
    positions = [
        OptionPosition("TEST", "call", 1, 100.0, 100.0, 1.0, 0.05, 0.20),
    ]

    vol_up_pnl = scenario_pnl(positions, vol_abs_change=0.05)

    assert vol_up_pnl > 0
