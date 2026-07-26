"""Implied-volatility solver tests."""
import pytest

from options_risk_engine.pricing.black_scholes import black_scholes
from options_risk_engine.pricing.implied_vol import implied_volatility


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_implied_vol_recovers_known_sigma(atm_call, option_type):
    true_sigma = atm_call["sigma"]
    market_price = black_scholes(**atm_call, option_type=option_type).price

    result = implied_volatility(
        market_price=market_price,
        S=atm_call["S"],
        K=atm_call["K"],
        T=atm_call["T"],
        r=atm_call["r"],
        option_type=option_type,
    )

    assert result.converged
    assert abs(result.implied_vol - true_sigma) < 1e-8
    assert result.absolute_error < 1e-8


def test_higher_market_price_implies_higher_vol(atm_call):
    low_params = {**atm_call, "sigma": 0.15}
    high_params = {**atm_call, "sigma": 0.35}
    low_price = black_scholes(**low_params).price
    high_price = black_scholes(**high_params).price

    low_iv = implied_volatility(
        market_price=low_price,
        S=atm_call["S"],
        K=atm_call["K"],
        T=atm_call["T"],
        r=atm_call["r"],
        option_type="call",
    ).implied_vol

    high_iv = implied_volatility(
        market_price=high_price,
        S=atm_call["S"],
        K=atm_call["K"],
        T=atm_call["T"],
        r=atm_call["r"],
        option_type="call",
    ).implied_vol

    assert high_iv > low_iv


def test_no_arbitrage_violation_raises(atm_call):
    with pytest.raises(ValueError, match="no-arbitrage bounds"):
        implied_volatility(
            market_price=200.0,
            S=atm_call["S"],
            K=atm_call["K"],
            T=atm_call["T"],
            r=atm_call["r"],
            option_type="call",
        )


def test_invalid_market_price_raises(atm_call):
    with pytest.raises(ValueError, match="market_price must be positive"):
        implied_volatility(
            market_price=0.0,
            S=atm_call["S"],
            K=atm_call["K"],
            T=atm_call["T"],
            r=atm_call["r"],
            option_type="call",
        )
