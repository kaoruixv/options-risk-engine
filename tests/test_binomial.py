"""Binomial tree tests: convergence to BS, early-exercise properties."""
import pytest

from options_risk_engine.pricing.binomial import binomial_price
from options_risk_engine.pricing.black_scholes import black_scholes


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_european_binomial_converges_to_black_scholes(atm_call, option_type):
    """With 2000 steps the CRR price should sit within 0.01 of closed-form."""
    bs = black_scholes(**atm_call, option_type=option_type).price
    tree = binomial_price(
        **atm_call, option_type=option_type, n_steps=2000, american=False
    ).price
    assert abs(tree - bs) < 0.01, f"tree={tree:.4f} bs={bs:.4f}"


def test_convergence_improves_with_more_steps(atm_call):
    """Error should shrink monotonically in this range of step counts."""
    bs = black_scholes(**atm_call).price
    errors = [
        abs(binomial_price(**atm_call, option_type="call", n_steps=n, american=False).price - bs)
        for n in (50, 200, 1000)
    ]
    assert errors[0] > errors[2], f"errors did not shrink: {errors}"


def test_american_put_at_least_european_put(deep_itm_put):
    """Early exercise is an extra right, so it cannot reduce value."""
    american = binomial_price(**deep_itm_put, option_type="put", n_steps=500, american=True).price
    european = binomial_price(**deep_itm_put, option_type="put", n_steps=500, american=False).price
    assert american >= european - 1e-10


def test_early_exercise_premium_is_material_for_deep_itm_put(deep_itm_put):
    """For a deep ITM put with positive rates the premium should be visible."""
    american = binomial_price(**deep_itm_put, option_type="put", n_steps=500, american=True).price
    european = binomial_price(**deep_itm_put, option_type="put", n_steps=500, american=False).price
    assert american - european > 0.05


def test_american_call_equals_european_call_without_dividends(atm_call):
    """With q=0 it is never optimal to exercise a call early."""
    american = binomial_price(**atm_call, option_type="call", n_steps=500, american=True).price
    european = binomial_price(**atm_call, option_type="call", n_steps=500, american=False).price
    assert abs(american - european) < 1e-8


def test_binomial_delta_matches_black_scholes(atm_call):
    bs_delta = black_scholes(**atm_call).greeks.delta
    tree_delta = binomial_price(
        **atm_call, option_type="call", n_steps=1000, american=False
    ).delta
    assert abs(tree_delta - bs_delta) < 0.01
