"""Monte Carlo tests: convergence, reproducibility, and input validation."""
import pytest

from options_risk_engine.pricing.black_scholes import black_scholes
from options_risk_engine.pricing.monte_carlo import monte_carlo_european


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_monte_carlo_converges_to_black_scholes(atm_call, option_type):
    bs = black_scholes(**atm_call, option_type=option_type).price
    mc = monte_carlo_european(
        **atm_call,
        option_type=option_type,
        n_paths=200_000,
        seed=123,
    )
    assert abs(mc.price - bs) < max(0.04, 4.0 * mc.std_error)


def test_monte_carlo_is_reproducible_with_seed(atm_call):
    a = monte_carlo_european(**atm_call, option_type="call", n_paths=50_000, seed=7)
    b = monte_carlo_european(**atm_call, option_type="call", n_paths=50_000, seed=7)
    assert a.price == b.price
    assert a.std_error == b.std_error


def test_confidence_interval_is_well_formed(atm_call):
    mc = monte_carlo_european(**atm_call, option_type="call", n_paths=50_000, seed=11)
    assert mc.ci_95_lower < mc.price < mc.ci_95_upper
    assert mc.std_error > 0
    assert mc.n_paths == 50_000


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"S": 0.0}, "S and K must be positive"),
        ({"T": 0.0}, "T must be positive"),
        ({"sigma": 0.0}, "sigma must be positive"),
        ({"n_paths": 1}, "n_paths must be at least 2"),
    ],
)
def test_invalid_inputs_raise(atm_call, kwargs, match):
    params = {**atm_call, **kwargs}
    with pytest.raises(ValueError, match=match):
        monte_carlo_european(**params)
