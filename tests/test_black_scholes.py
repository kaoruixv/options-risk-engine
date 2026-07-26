"""Black-Scholes tests: parity, boundaries, Greeks vs finite differences."""
import math

import pytest

from options_risk_engine.pricing.black_scholes import (
    black_scholes,
    put_call_parity_check,
)

EPS = 1e-5  # bump size for central finite differences


def _price(S, K, T, r, sigma, opt="call", q=0.0) -> float:
    return black_scholes(S, K, T, r, sigma, opt, q).price


@pytest.mark.parametrize(
    "S,K,T,r,sigma,q",
    [
        (100, 100, 1.00, 0.05, 0.20, 0.00),
        (100, 110, 0.50, 0.03, 0.30, 0.02),
        (80, 100, 2.00, 0.01, 0.40, 0.00),
        (120, 100, 0.25, 0.06, 0.15, 0.01),
    ],
)
def test_put_call_parity(S, K, T, r, sigma, q):
    assert put_call_parity_check(S, K, T, r, sigma, q, tol=1e-10)


def test_deep_itm_call_approaches_forward():
    price = _price(200, 100, 1.0, 0.05, 0.20, "call")
    assert abs(price - (200 - 100 * math.exp(-0.05))) < 0.10


def test_deep_otm_call_near_zero():
    assert _price(50, 200, 1.0, 0.05, 0.20, "call") < 0.001


def test_deep_itm_put_approaches_pv_of_strike():
    price = _price(50, 200, 1.0, 0.05, 0.20, "put")
    assert abs(price - (200 * math.exp(-0.05) - 50)) < 0.10


def test_delta_matches_finite_difference(atm_call):
    p = atm_call
    fd = (
        _price(p["S"] + EPS, p["K"], p["T"], p["r"], p["sigma"])
        - _price(p["S"] - EPS, p["K"], p["T"], p["r"], p["sigma"])
    ) / (2 * EPS)
    assert abs(black_scholes(**p).greeks.delta - fd) < 1e-6


def test_gamma_matches_finite_difference(atm_call):
    p = atm_call
    fd = (
        _price(p["S"] + EPS, p["K"], p["T"], p["r"], p["sigma"])
        - 2 * _price(**p)
        + _price(p["S"] - EPS, p["K"], p["T"], p["r"], p["sigma"])
    ) / EPS**2
    assert abs(black_scholes(**p).greeks.gamma - fd) < 1e-4


def test_vega_matches_finite_difference(atm_call):
    p = atm_call
    fd = (
        _price(p["S"], p["K"], p["T"], p["r"], p["sigma"] + EPS)
        - _price(p["S"], p["K"], p["T"], p["r"], p["sigma"] - EPS)
    ) / (2 * EPS) / 100.0
    assert abs(black_scholes(**p).greeks.vega - fd) < 1e-6


def test_theta_matches_finite_difference(atm_call):
    p = atm_call
    dv_dt = (
        _price(p["S"], p["K"], p["T"] + EPS, p["r"], p["sigma"])
        - _price(p["S"], p["K"], p["T"] - EPS, p["r"], p["sigma"])
    ) / (2 * EPS)
    fd_theta = -dv_dt / 365.0
    assert abs(black_scholes(**p).greeks.theta - fd_theta) < 1e-5


def test_rho_matches_finite_difference(atm_call):
    p = atm_call
    fd = (
        _price(p["S"], p["K"], p["T"], p["r"] + EPS, p["sigma"])
        - _price(p["S"], p["K"], p["T"], p["r"] - EPS, p["sigma"])
    ) / (2 * EPS) / 100.0
    assert abs(black_scholes(**p).greeks.rho - fd) < 1e-6


def test_put_greek_signs(atm_call):
    g = black_scholes(**atm_call, option_type="put").greeks
    assert g.delta < 0
    assert g.gamma > 0
    assert g.vega > 0
    assert g.theta < 0
    assert g.rho < 0


@pytest.mark.parametrize(
    "S,K,T,r,sigma,match",
    [
        (100, 100, 0.0, 0.05, 0.20, "T must be positive"),
        (100, 100, 1.0, 0.05, 0.00, "sigma must be positive"),
        (-10, 100, 1.0, 0.05, 0.20, "S and K must be positive"),
    ],
)
def test_invalid_inputs_raise(S, K, T, r, sigma, match):
    with pytest.raises(ValueError, match=match):
        black_scholes(S, K, T, r, sigma)
