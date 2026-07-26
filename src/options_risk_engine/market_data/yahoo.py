"""Yahoo Finance option-chain adapter.

This module converts raw yfinance option-chain data into the normalized
schema expected by the volatility-surface builder:

    option_type, strike, time_to_expiry, bid, ask

The function is intentionally thin. Quote filtering and implied-volatility
inversion are handled by the vol_surface module.
"""
from __future__ import annotations

from collections.abc import Iterable
from datetime import date

import pandas as pd
import yfinance as yf


NORMALIZED_COLUMNS = [
    "symbol",
    "expiry",
    "option_type",
    "contract_symbol",
    "strike",
    "time_to_expiry",
    "bid",
    "ask",
    "last_price",
    "volume",
    "open_interest",
    "yahoo_implied_vol",
]


def _to_valuation_date(value: date | str | pd.Timestamp | None) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp.today().normalize()
    return pd.Timestamp(value).normalize()


def _years_to_expiry(expiry: str, valuation_date: pd.Timestamp) -> float:
    expiry_ts = pd.Timestamp(expiry).normalize()
    days = max((expiry_ts - valuation_date).days, 0)
    return days / 365.0


def _normalize_side(
    frame: pd.DataFrame,
    symbol: str,
    expiry: str,
    option_type: str,
    valuation_date: pd.Timestamp,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)

    out = pd.DataFrame(
        {
            "symbol": symbol,
            "expiry": expiry,
            "option_type": option_type,
            "contract_symbol": frame.get("contractSymbol"),
            "strike": frame.get("strike"),
            "time_to_expiry": _years_to_expiry(expiry, valuation_date),
            "bid": frame.get("bid"),
            "ask": frame.get("ask"),
            "last_price": frame.get("lastPrice"),
            "volume": frame.get("volume"),
            "open_interest": frame.get("openInterest"),
            "yahoo_implied_vol": frame.get("impliedVolatility"),
        }
    )

    return out[NORMALIZED_COLUMNS]


def fetch_yahoo_option_chain(
    symbol: str,
    valuation_date: date | str | pd.Timestamp | None = None,
    expiries: Iterable[str] | None = None,
    ticker_factory=yf.Ticker,
) -> pd.DataFrame:
    """Fetch and normalize option-chain data from Yahoo Finance.

    Parameters
    ----------
    symbol:
        Ticker symbol, for example "AAPL".
    valuation_date:
        Date used to compute time_to_expiry. Defaults to today.
    expiries:
        Optional subset of Yahoo expiry strings. If omitted, all available
        expiries returned by yfinance are fetched.
    ticker_factory:
        Dependency-injection hook for tests. Defaults to yfinance.Ticker.
    """
    if not symbol or not symbol.strip():
        raise ValueError("symbol must be a non-empty string")

    clean_symbol = symbol.upper().strip()
    val_date = _to_valuation_date(valuation_date)
    ticker = ticker_factory(clean_symbol)

    selected_expiries = list(expiries) if expiries is not None else list(ticker.options)
    if not selected_expiries:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)

    frames: list[pd.DataFrame] = []

    for expiry in selected_expiries:
        chain = ticker.option_chain(expiry)
        frames.append(_normalize_side(chain.calls, clean_symbol, expiry, "call", val_date))
        frames.append(_normalize_side(chain.puts, clean_symbol, expiry, "put", val_date))

    if not frames:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)

    return pd.concat(frames, ignore_index=True)
