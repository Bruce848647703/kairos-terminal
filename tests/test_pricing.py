"""定价测试：闭式解一致性、收敛性、隐含波动率、对冲误差。"""
from __future__ import annotations

import math

import pytest

from kairos_terminal.pricing import (
    binomial_crr,
    black_scholes,
    compare_models,
    delta_hedge_experiment,
    implied_vol,
    monte_carlo,
)
from kairos_terminal.processes import create


def test_black_scholes_known_atm_value():
    res = black_scholes(100, 100, 1.0, 0.0, 0.2)
    # S=K=100, r=0, σ=20%, T=1 → 7.965567...
    assert res["price"] == pytest.approx(7.965567, rel=1e-5)
    assert res["delta"] == pytest.approx(0.539828, rel=1e-4)
    assert res["gamma"] == pytest.approx(0.019847, rel=1e-4)
    assert res["vega"] == pytest.approx(39.694, rel=1e-3)


def test_put_call_parity():
    S, K, T, r, q, vol = 100.0, 105.0, 0.75, 0.03, 0.01, 0.25
    call = black_scholes(S, K, T, r, vol, "call", q)["price"]
    put = black_scholes(S, K, T, r, vol, "put", q)["price"]
    assert call - put == pytest.approx(S * math.exp(-q * T) - K * math.exp(-r * T), rel=1e-10)


def test_expired_option_is_intrinsic():
    assert black_scholes(120, 100, 0.0, 0.02, 0.2, "call")["price"] == pytest.approx(20.0)
    assert black_scholes(80, 100, 0.0, 0.02, 0.2, "put")["price"] == pytest.approx(20.0)
    assert black_scholes(90, 100, 0.0, 0.02, 0.2, "call")["price"] == 0.0


def test_zero_vol_limit():
    res = black_scholes(100, 100, 1.0, 0.05, 0.0, "call")
    assert res["price"] == pytest.approx(max(100 * math.exp(0.05) - 100, 0) * math.exp(-0.05),
                                         rel=1e-12)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        black_scholes(-1, 100, 1, 0.02, 0.2)
    with pytest.raises(ValueError):
        black_scholes(100, 100, -1, 0.02, 0.2)
    with pytest.raises(ValueError):
        black_scholes(100, 100, 1, 0.02, -0.2)
    with pytest.raises(ValueError):
        black_scholes(100, 100, 1, 0.02, 0.2, "straddle")


def test_binomial_converges_to_black_scholes():
    bs = black_scholes(100, 100, 1.0, 0.03, 0.25)["price"]
    tree = binomial_crr(100, 100, 1.0, 0.03, 0.25, "call", steps=800)["price"]
    assert tree == pytest.approx(bs, abs=0.01)


def test_american_put_not_cheaper_than_european():
    euro = binomial_crr(100, 90, 1.0, 0.05, 0.25, "put", steps=300)["price"]
    amer = binomial_crr(100, 90, 1.0, 0.05, 0.25, "put", steps=300, american=True)["price"]
    assert amer >= euro - 1e-12


def test_monte_carlo_matches_black_scholes():
    process = create("gbm", mu=0.03, sigma=0.2)
    bs = black_scholes(100, 100, 1.0, 0.03, 0.2)["price"]
    mc = monte_carlo(process, 100, 100, 1.0, 0.03, "call",
                     n_paths=40_000, n_steps=252, seed=11)
    assert mc["price"] == pytest.approx(bs, abs=5 * mc["stderr"] + 0.01)
    assert mc["ci_low"] < mc["price"] < mc["ci_high"]


def test_mc_asian_is_cheaper_than_vanilla_for_call():
    process = create("gbm", mu=0.03, sigma=0.25)
    common = dict(n_paths=10_000, n_steps=64, seed=3, rate=0.03)
    vanilla = monte_carlo(process, 100, 100, 1.0, payoff="vanilla", **common)["price"]
    asian = monte_carlo(process, 100, 100, 1.0, payoff="asian", **common)["price"]
    assert asian <= vanilla


def test_barrier_up_and_out_is_cheaper():
    process = create("gbm", mu=0.03, sigma=0.25)
    common = dict(n_paths=10_000, n_steps=64, seed=3)
    vanilla = monte_carlo(process, 100, 100, 1.0, 0.03, payoff="vanilla", **common)["price"]
    knockout = monte_carlo(process, 100, 100, 1.0, 0.03, payoff="barrier",
                           barrier=130, barrier_kind="up-out", **common)["price"]
    assert knockout < vanilla
    with pytest.raises(ValueError, match="barrier"):
        monte_carlo(process, 100, 100, 1.0, 0.03, payoff="barrier", **common)


def test_digital_call_price():
    process = create("gbm", mu=0.03, sigma=0.2)
    res = monte_carlo(process, 100, 100, 1.0, 0.03, payoff="digital",
                      n_paths=40_000, n_steps=252, seed=7)
    d2 = black_scholes(100, 100, 1.0, 0.03, 0.2)["d2"]
    expected = math.exp(-0.03) * 0.5 * (1 + math.erf(d2 / math.sqrt(2)))
    assert res["price"] == pytest.approx(expected, abs=0.02)


def test_implied_vol_roundtrip():
    price = black_scholes(100, 120, 0.5, 0.02, 0.33, "put")["price"]
    vol = implied_vol(price, 100, 120, 0.5, 0.02, "put")
    assert vol == pytest.approx(0.33, abs=1e-6)


def test_implied_vol_rejects_arbitrage_violation():
    with pytest.raises(ValueError, match="无套利区间"):
        implied_vol(1000.0, 100, 100, 1.0, 0.02)


def test_delta_hedge_pnl_is_centered_when_model_matches():
    process = create("gbm", mu=0.02, sigma=0.2)
    res = delta_hedge_experiment(process, 100, 100, 1.0, 0.02, 0.2,
                                 n_paths=3000, n_steps=52, seed=5)
    assert abs(res["mean_pnl"]) < 0.05
    assert res["std_pnl"] > 0
    assert 0.0 <= res["prob_loss"] <= 1.0
    assert len(res["pnl_percentiles"]) == 7


def test_delta_hedge_under_misspecification_has_dispersion():
    process = create("heston", mu=0.02, v0=0.09, theta=0.09, kappa=2.0, xi=0.8, rho=-0.8)
    res = delta_hedge_experiment(process, 100, 100, 1.0, 0.02, 0.20,
                                 n_paths=800, n_steps=26, seed=2)
    assert res["std_pnl"] > 0
    assert res["hist"]["counts"]
    assert len(res["sample_pnl"]) == 200


def test_mc_expired_option_returns_intrinsic():
    process = create("gbm")
    res = monte_carlo(process, 110, 100, 0.0, 0.02)
    assert res["price"] == pytest.approx(10.0)
    assert res["stderr"] == 0.0


def test_compare_models_contract():
    process = create("heston")
    res = compare_models(process, 100, 100, 1.0, 0.02, 0.2,
                         n_paths=5000, n_steps=128, seed=1)
    assert set(res) >= {"black_scholes", "binomial", "monte_carlo", "mc_stderr", "greeks"}
    assert abs(res["black_scholes"] - res["binomial"]) < 0.05