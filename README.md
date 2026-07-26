# options-risk-engine

Derivatives pricing, implied-vol surface construction, delta-hedging
simulation, and portfolio risk measurement.

Status: in development. Pricing module (Black-Scholes, CRR binomial)
complete with test coverage. Vol surface, hedging backtest, and risk
modules to follow.

## Quickstart

    uv venv .venv --python 3.12
    source .venv/bin/activate
    uv pip install -e ".[dev]"
    pytest

## Other projects

- https://github.com/kaoruixv/qt-engine
- https://github.com/kaoruixv/meta-equity-research
