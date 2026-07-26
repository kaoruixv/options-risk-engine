"""Quickstart example for options-risk-engine.

Run:
    python examples/quickstart.py
"""
from __future__ import annotations

from options_risk_engine.hedging_backtest import simulate_delta_hedge
from options_risk_engine.pricing import (
    binomial_price,
    black_scholes,
    implied_volatility,
    monte_carlo_european,
)
from options_risk_engine.risk import OptionPosition, aggregate_portfolio_risk


def main() -> None:
    params = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20)

    bs = black_scholes(**params, option_type="call")
    tree = binomial_price(**params, option_type="call", american=False, n_steps=1000)
    mc = monte_carlo_european(**params, option_type="call", n_paths=100_000, seed=42)

    iv = implied_volatility(
        market_price=bs.price,
        S=params["S"],
        K=params["K"],
        T=params["T"],
        r=params["r"],
        option_type="call",
    )

    portfolio = aggregate_portfolio_risk(
        [
            OptionPosition("DEMO", "call", 1, 100.0, 100.0, 1.0, 0.05, 0.20),
            OptionPosition("DEMO", "put", -1, 100.0, 100.0, 1.0, 0.05, 0.20),
        ]
    )

    hedge = simulate_delta_hedge(
        S0=100.0,
        K=100.0,
        T=1.0,
        r=0.02,
        sigma=0.20,
        option_type="call",
        n_steps=50,
        n_paths=2000,
        seed=42,
        transaction_cost_bps=0.0,
    )

    print("Black-Scholes call price:", round(bs.price, 4))
    print("CRR binomial call price :", round(tree.price, 4))
    print("Monte Carlo call price  :", round(mc.price, 4))
    print("Recovered implied vol   :", round(iv.implied_vol, 4))
    print("Portfolio value         :", round(portfolio.total_value, 4))
    print("Portfolio delta         :", round(portfolio.greeks.delta, 4))
    print("Mean hedge PnL          :", round(hedge.summary.mean_pnl, 4))


if __name__ == "__main__":
    main()
