"""Discrete delta-hedging backtest tests."""
import pytest

from options_risk_engine.hedging_backtest.backtest import simulate_delta_hedge


def test_delta_hedging_result_has_expected_shape():
    result = simulate_delta_hedge(
        S0=100.0,
        K=100.0,
        T=1.0,
        r=0.02,
        sigma=0.20,
        option_type="call",
        n_steps=50,
        n_paths=1000,
        seed=123,
        transaction_cost_bps=0.0,
    )

    assert result.n_paths == 1000
    assert result.n_steps == 50
    assert len(result.pnl) == 1000
    assert {"terminal_spot", "pnl", "transaction_cost"}.issubset(result.pnl.columns)


def test_zero_cost_average_pnl_is_close_to_zero():
    result = simulate_delta_hedge(
        S0=100.0,
        K=100.0,
        T=1.0,
        r=0.02,
        sigma=0.20,
        option_type="call",
        n_steps=50,
        n_paths=5000,
        seed=123,
        transaction_cost_bps=0.0,
    )

    assert abs(result.summary.mean_pnl) < 0.10


def test_transaction_cost_reduces_average_pnl():
    no_cost = simulate_delta_hedge(
        S0=100.0,
        K=100.0,
        T=1.0,
        r=0.02,
        sigma=0.20,
        option_type="call",
        n_steps=50,
        n_paths=3000,
        seed=123,
        transaction_cost_bps=0.0,
    )

    with_cost = simulate_delta_hedge(
        S0=100.0,
        K=100.0,
        T=1.0,
        r=0.02,
        sigma=0.20,
        option_type="call",
        n_steps=50,
        n_paths=3000,
        seed=123,
        transaction_cost_bps=5.0,
    )

    assert with_cost.summary.mean_transaction_cost > 0
    assert with_cost.summary.mean_pnl < no_cost.summary.mean_pnl


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"S0": 0.0}, "S0 and K must be positive"),
        ({"T": 0.0}, "T must be positive"),
        ({"sigma": 0.0}, "sigma must be positive"),
        ({"n_steps": 1}, "n_steps must be at least 2"),
        ({"n_paths": 1}, "n_paths must be at least 2"),
        ({"transaction_cost_bps": -1.0}, "transaction_cost_bps must be non-negative"),
    ],
)
def test_invalid_inputs_raise(kwargs, match):
    params = {
        "S0": 100.0,
        "K": 100.0,
        "T": 1.0,
        "r": 0.02,
        "sigma": 0.20,
        "option_type": "call",
        "n_steps": 50,
        "n_paths": 100,
        "seed": 123,
        "transaction_cost_bps": 0.0,
    }
    params.update(kwargs)

    with pytest.raises(ValueError, match=match):
        simulate_delta_hedge(**params)
