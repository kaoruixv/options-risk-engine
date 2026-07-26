"""Shared fixtures for the test suite."""
import pytest


@pytest.fixture
def atm_call() -> dict:
    """At-the-money call: S=K=100, T=1y, r=5%, sigma=20%."""
    return dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20)


@pytest.fixture
def deep_itm_put() -> dict:
    """Deep in-the-money put, where early exercise carries real value."""
    return dict(S=70.0, K=100.0, T=1.0, r=0.05, sigma=0.20)
