"""Visualization utilities for volatility surfaces, hedging PnL, and Greeks."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from options_risk_engine.risk.portfolio import GreekSummary


def _prepare_output_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_vol_surface_heatmap(
    matrix: pd.DataFrame,
    output_path: str | Path,
    title: str = "Implied Volatility Surface",
) -> Path:
    """Save an expiry-by-strike implied-volatility heatmap."""
    if matrix.empty:
        raise ValueError("matrix must not be empty")

    path = _prepare_output_path(output_path)

    fig, ax = plt.subplots(figsize=(9, 5))
    image = ax.imshow(matrix.values, aspect="auto", origin="lower")

    ax.set_title(title)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Time to expiry")

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels([f"{value:g}" for value in matrix.columns], rotation=45, ha="right")

    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels([f"{value:.3f}" for value in matrix.index])

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Implied volatility")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

    return path


def plot_hedging_pnl_histogram(
    pnl: pd.DataFrame | pd.Series,
    output_path: str | Path,
    bins: int = 40,
    title: str = "Delta-Hedging PnL Distribution",
) -> Path:
    """Save a histogram of delta-hedging PnL."""
    if isinstance(pnl, pd.DataFrame):
        if "pnl" not in pnl.columns:
            raise ValueError("pnl DataFrame must contain a 'pnl' column")
        values = pnl["pnl"]
    else:
        values = pnl

    if values.empty:
        raise ValueError("pnl must not be empty")
    if bins <= 0:
        raise ValueError("bins must be positive")

    path = _prepare_output_path(output_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=bins)
    ax.axvline(float(values.mean()), linestyle="--", linewidth=1.5)

    ax.set_title(title)
    ax.set_xlabel("PnL")
    ax.set_ylabel("Frequency")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

    return path


def plot_portfolio_greeks(
    greeks: GreekSummary | Mapping[str, float],
    output_path: str | Path,
    title: str = "Portfolio Greeks",
) -> Path:
    """Save a bar chart of portfolio Greeks."""
    if isinstance(greeks, GreekSummary):
        data = {
            "delta": greeks.delta,
            "gamma": greeks.gamma,
            "vega": greeks.vega,
            "theta": greeks.theta,
            "rho": greeks.rho,
            "vanna": greeks.vanna,
            "volga": greeks.volga,
        }
    else:
        data = dict(greeks)

    if not data:
        raise ValueError("greeks must not be empty")

    path = _prepare_output_path(output_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(list(data.keys()), list(data.values()))

    ax.set_title(title)
    ax.set_xlabel("Greek")
    ax.set_ylabel("Exposure")
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

    return path
