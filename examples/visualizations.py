"""Create example visualization outputs.

Run:
    python examples/visualizations.py

Outputs:
    results/demo_vol_surface_heatmap.png
    results/demo_hedging_pnl_histogram.png
    results/demo_portfolio_greeks.png
"""
from __future__ import annotations

import pandas as pd

from options_risk_engine.hedging_backtest import simulate_delta_hedge
from options_risk_engine.pricing.black_scholes import black_scholes
from options_risk_engine.risk import OptionPosition, aggregate_portfolio_risk
from options_risk_engine.visualization import (
    plot_hedging_pnl_histogram,
    plot_portfolio_greeks,
    plot_vol_surface_heatmap,
)
from options_risk_engine.vol_surface import build_surface_from_chain, surface_matrix


def make_demo_chain() -> pd.DataFrame:
    rows = []
    for time_to_expiry in [0.25, 0.5, 1.0]:
        for strike in [80.0, 90.0, 100.0, 110.0, 120.0]:
            smile = 0.20 + 0.0015 * abs(strike - 100.0)
            price = black_scholes(
                S=100.0,
                K=strike,
                T=time_to_expiry,
                r=0.04,
                sigma=smile,
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


def main() -> None:
    chain = make_demo_chain()
    surface = build_surface_from_chain(chain, spot=100.0, rate=0.04)
    matrix = surface_matrix(surface.points, option_type="call")

    hedge = simulate_delta_hedge(
        S0=100.0,
        K=100.0,
        T=1.0,
        r=0.02,
        sigma=0.20,
        option_type="call",
        n_steps=50,
        n_paths=3000,
        seed=42,
        transaction_cost_bps=0.0,
    )

    risk = aggregate_portfolio_risk(
        [
            OptionPosition("DEMO", "call", 1, 100.0, 100.0, 1.0, 0.05, 0.20),
            OptionPosition("DEMO", "put", -1, 100.0, 100.0, 1.0, 0.05, 0.20),
        ]
    )

    plot_vol_surface_heatmap(matrix, "results/demo_vol_surface_heatmap.png")
    plot_hedging_pnl_histogram(hedge.pnl, "results/demo_hedging_pnl_histogram.png")
    plot_portfolio_greeks(risk.greeks, "results/demo_portfolio_greeks.png")

    print("Wrote results/demo_vol_surface_heatmap.png")
    print("Wrote results/demo_hedging_pnl_histogram.png")
    print("Wrote results/demo_portfolio_greeks.png")


if __name__ == "__main__":
    main()
