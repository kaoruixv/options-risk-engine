"""Visualization utility tests."""
import pandas as pd
import pytest

from options_risk_engine.hedging_backtest import simulate_delta_hedge
from options_risk_engine.risk import OptionPosition, aggregate_portfolio_risk
from options_risk_engine.visualization import (
    plot_hedging_pnl_histogram,
    plot_portfolio_greeks,
    plot_vol_surface_heatmap,
)


def test_plot_vol_surface_heatmap_writes_file(tmp_path):
    matrix = pd.DataFrame(
        [[0.20, 0.21], [0.22, 0.23]],
        index=[0.5, 1.0],
        columns=[95.0, 105.0],
    )

    path = plot_vol_surface_heatmap(matrix, tmp_path / "surface.png")

    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_hedging_pnl_histogram_writes_file(tmp_path):
    result = simulate_delta_hedge(
        S0=100.0,
        K=100.0,
        T=1.0,
        r=0.02,
        sigma=0.20,
        option_type="call",
        n_steps=20,
        n_paths=500,
        seed=123,
        transaction_cost_bps=0.0,
    )

    path = plot_hedging_pnl_histogram(result.pnl, tmp_path / "pnl.png")

    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_portfolio_greeks_writes_file(tmp_path):
    risk = aggregate_portfolio_risk(
        [
            OptionPosition("TEST", "call", 1, 100.0, 100.0, 1.0, 0.05, 0.20),
        ]
    )

    path = plot_portfolio_greeks(risk.greeks, tmp_path / "greeks.png")

    assert path.exists()
    assert path.stat().st_size > 0


def test_empty_surface_raises(tmp_path):
    with pytest.raises(ValueError, match="matrix must not be empty"):
        plot_vol_surface_heatmap(pd.DataFrame(), tmp_path / "bad.png")
