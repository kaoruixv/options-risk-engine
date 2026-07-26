"""Build an implied-volatility surface from Yahoo option-chain data.

Example:
    python scripts/build_surface.py --symbol AAPL --spot 200 --rate 0.04

The script writes three CSV files to results/:
    SYMBOL_iv_points.csv
    SYMBOL_rejected_quotes.csv
    SYMBOL_surface_matrix.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

from options_risk_engine.vol_surface.pipeline import build_surface_from_yahoo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an implied-volatility surface.")
    parser.add_argument("--symbol", required=True, help="Ticker symbol, for example AAPL")
    parser.add_argument("--spot", required=True, type=float, help="Current spot price")
    parser.add_argument("--rate", required=True, type=float, help="Risk-free rate, for example 0.04")
    parser.add_argument("--dividend-yield", default=0.0, type=float, help="Continuous dividend yield")
    parser.add_argument("--valuation-date", default=None, help="Valuation date, for example 2026-07-26")
    parser.add_argument("--option-type", default="call", choices=["call", "put"])
    parser.add_argument(
        "--expiries",
        default=None,
        help="Comma-separated Yahoo expiry dates, for example 2026-08-21,2026-09-18",
    )
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--max-spread-pct", default=0.50, type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expiries = None
    if args.expiries:
        expiries = [item.strip() for item in args.expiries.split(",") if item.strip()]

    result = build_surface_from_yahoo(
        symbol=args.symbol,
        spot=args.spot,
        rate=args.rate,
        valuation_date=args.valuation_date,
        expiries=expiries,
        dividend_yield=args.dividend_yield,
        option_type=args.option_type,
        max_spread_pct=args.max_spread_pct,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    symbol = args.symbol.upper()
    result.points.to_csv(output_dir / f"{symbol}_iv_points.csv", index=False)
    result.rejected.to_csv(output_dir / f"{symbol}_rejected_quotes.csv", index=False)
    result.matrix.to_csv(output_dir / f"{symbol}_surface_matrix.csv")

    print(f"input rows    : {result.input_rows}")
    print(f"accepted rows : {result.accepted_rows}")
    print(f"rejected rows : {result.rejected_rows}")
    print(f"matrix shape  : {result.matrix.shape}")
    print(f"output dir    : {output_dir.resolve()}")


if __name__ == "__main__":
    main()
