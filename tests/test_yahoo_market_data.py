"""Yahoo Finance adapter tests.

These tests do not call the network. They use a fake ticker object that has
the same minimal interface as yfinance.Ticker.
"""
import pandas as pd
import pytest

from options_risk_engine.market_data.yahoo import fetch_yahoo_option_chain


class FakeOptionChain:
    def __init__(self) -> None:
        self.calls = pd.DataFrame(
            [
                {
                    "contractSymbol": "FAKE300117C00100000",
                    "strike": 100.0,
                    "lastPrice": 10.5,
                    "bid": 10.4,
                    "ask": 10.6,
                    "volume": 123,
                    "openInterest": 456,
                    "impliedVolatility": 0.20,
                }
            ]
        )
        self.puts = pd.DataFrame(
            [
                {
                    "contractSymbol": "FAKE300117P00100000",
                    "strike": 100.0,
                    "lastPrice": 5.5,
                    "bid": 5.4,
                    "ask": 5.6,
                    "volume": 321,
                    "openInterest": 654,
                    "impliedVolatility": 0.22,
                }
            ]
        )


class FakeTicker:
    options = ["2030-01-17", "2030-06-21"]

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol

    def option_chain(self, expiry: str) -> FakeOptionChain:
        assert expiry in self.options
        return FakeOptionChain()


def test_fetch_yahoo_option_chain_normalizes_rows():
    df = fetch_yahoo_option_chain(
        "fake",
        valuation_date="2029-01-17",
        expiries=["2030-01-17"],
        ticker_factory=FakeTicker,
    )

    assert len(df) == 2
    assert set(df["option_type"]) == {"call", "put"}
    assert set(df["symbol"]) == {"FAKE"}
    assert set(df["strike"]) == {100.0}
    assert df["time_to_expiry"].iloc[0] == pytest.approx(1.0)
    assert {"bid", "ask", "yahoo_implied_vol"}.issubset(df.columns)


def test_fetch_yahoo_option_chain_uses_all_expiries_when_not_specified():
    df = fetch_yahoo_option_chain(
        "FAKE",
        valuation_date="2029-01-17",
        ticker_factory=FakeTicker,
    )

    assert len(df) == 4
    assert set(df["expiry"]) == {"2030-01-17", "2030-06-21"}


def test_empty_symbol_raises():
    with pytest.raises(ValueError, match="symbol must be a non-empty string"):
        fetch_yahoo_option_chain("   ", ticker_factory=FakeTicker)
