"""价格过程测试：形状、确定性、参数校验与统计性质。"""
from __future__ import annotations

import numpy as np
import pytest

from kairos_terminal.processes import (
    BootstrapProcess,
    CustomProcess,
    GBMProcess,
    HestonProcess,
    LogOUProcess,
    VarianceGammaProcess,
    create,
    describe_all,
    keys,
)


def test_registry_lists_ui_processes():
    all_keys = keys()
    for key in ("gbm", "trend", "heston", "merton", "bates", "ou", "cir", "regime", "vg"):
        assert key in all_keys
    ui = keys(ui_only=True)
    assert "bootstrap" not in ui and "custom" not in ui
    metas = describe_all(ui_only=True)
    assert all("params" in m and "defaults" in m for m in metas)


def test_unknown_process_raises():
    with pytest.raises(KeyError):
        create("does-not-exist")


@pytest.mark.parametrize("key", ["gbm", "trend", "heston", "merton", "bates",
                                 "ou", "cir", "regime", "vg"])
def test_all_processes_shape_and_positivity(key):
    process = create(key)
    paths = process.simulate(40, s0=123.0, seed=5, n_paths=3)
    assert paths.shape == (3, 41)
    assert np.all(paths > 0)
    assert np.allclose(paths[:, 0], 123.0)
    assert np.all(np.isfinite(paths))


@pytest.mark.parametrize("key", ["gbm", "heston", "merton", "ou", "cir", "regime", "vg"])
def test_determinism(key):
    a = create(key).simulate(20, seed=11, n_paths=2)
    b = create(key).simulate(20, seed=11, n_paths=2)
    c = create(key).simulate(20, seed=12, n_paths=2)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_unknown_and_out_of_range_params():
    with pytest.raises(ValueError, match="不认识参数"):
        GBMProcess(mystery=1)
    with pytest.raises(ValueError, match="不能大于"):
        GBMProcess(sigma=99)
    with pytest.raises(ValueError, match="无法解析"):
        GBMProcess(sigma="abc")


def test_gbm_log_return_moments():
    mu, sigma, dt = 0.08, 0.25, 1 / 252
    paths = GBMProcess(mu=mu, sigma=sigma).simulate(252, dt=dt, seed=3, n_paths=400)
    log_ret = np.diff(np.log(paths), axis=1)
    expected_mean = (mu - 0.5 * sigma ** 2) * dt
    assert abs(log_ret.mean() - expected_mean) < 5e-4
    assert abs(log_ret.std() - sigma * np.sqrt(dt)) < 5e-4


def test_gbm_zero_vol_is_deterministic():
    paths = GBMProcess(mu=0.1, sigma=0.0).simulate(10, seed=1, n_paths=2)
    assert np.allclose(paths[0], paths[1])


def test_heston_feller_flag_and_positive_variance():
    bad = HestonProcess(xi=4.0, kappa=1.0, theta=0.01)
    assert bad.feller_ok is False
    good = HestonProcess()
    assert good.feller_ok is True
    with pytest.raises(ValueError):
        HestonProcess(rho=1.5)


def test_jumps_create_fat_tails():
    dt = 1 / 252
    gbm = GBMProcess(mu=0.05, sigma=0.2).simulate(252, dt=dt, seed=2, n_paths=600)
    jump = create("merton", mu=0.05, sigma=0.2, lam=6.0, jump_sigma=0.12) \
        .simulate(252, dt=dt, seed=2, n_paths=600)
    def kurtosis(p):
        r = np.diff(np.log(p), axis=1).ravel()
        r = (r - r.mean()) / r.std()
        return float((r ** 4).mean())
    assert kurtosis(jump) > kurtosis(gbm) * 1.2


def test_ou_mean_reverts():
    process = LogOUProcess(kappa=10.0, long_run=100.0, sigma=0.05)
    paths = process.simulate(252, s0=150.0, seed=4, n_paths=300)
    assert abs(paths[:, -1].mean() - 100.0) < 3.0
    assert abs(paths[:, -1].std()) < 10.0


def test_cir_stays_non_negative():
    paths = create("cir", kappa=0.5, theta=2.0, sigma=5.0).simulate(200, s0=1.0, seed=9, n_paths=50)
    assert np.all(paths >= 0)


def test_vg_rejects_explosive_params():
    with pytest.raises(ValueError, match="参数组合非法"):
        VarianceGammaProcess(nu=1.0, theta=1.0, sigma=1.0)


def test_risk_neutral_replaces_drift():
    gbm = GBMProcess(mu=0.5)
    rn = gbm.risk_neutral(0.02)
    assert rn["mu"] == pytest.approx(0.02)
    assert gbm["mu"] == pytest.approx(0.5)          # 原对象不变
    ou = LogOUProcess()
    assert ou.risk_neutral(0.02) is ou              # 无 mu 参数 -> 原样返回


def test_bootstrap_from_prices():
    prices = [100, 101, 99, 102, 98, 103, 97]
    process = BootstrapProcess.from_prices(prices, block=2)
    a = process.simulate(30, seed=1, n_paths=2)
    b = process.simulate(30, seed=1, n_paths=2)
    assert a.shape == (2, 31)
    assert np.array_equal(a, b)
    assert np.all(a > 0)


def test_bootstrap_requires_enough_data():
    with pytest.raises(ValueError):
        BootstrapProcess([0.01])
    with pytest.raises(ValueError):
        BootstrapProcess.from_prices([100.0, 101.0])


def test_custom_process_contract():
    def fn(*, n_steps, dt, s0, seed, n_paths):
        return np.full((n_paths, n_steps + 1), s0)

    process = CustomProcess(fn)
    assert process.simulate(5, s0=10, n_paths=2).shape == (2, 6)

    def bad(*, n_steps, dt, s0, seed, n_paths):
        return np.zeros((n_paths, n_steps))

    with pytest.raises(ValueError, match="期望"):
        CustomProcess(bad).simulate(5, n_paths=1)


def test_invalid_simulation_inputs():
    with pytest.raises(ValueError):
        GBMProcess().simulate(0)
    with pytest.raises(ValueError):
        GBMProcess().simulate(5, s0=0)
    with pytest.raises(ValueError):
        GBMProcess().simulate(5, dt=0)

# --------------------------------------------------------------------------- #
# 进阶过程：SABR / rough Bergomi / GARCH / 双因子
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", ["sabr", "rough_bergomi", "garch", "two_factor"])
def test_advanced_processes_shape_and_determinism(key):
    process = create(key)
    a = process.simulate(24, s0=50.0, seed=9, n_paths=3)
    b = process.simulate(24, s0=50.0, seed=9, n_paths=3)
    c = process.simulate(24, s0=50.0, seed=10, n_paths=3)
    assert a.shape == (3, 25)
    assert np.all(a > 0) and np.all(np.isfinite(a))
    assert np.allclose(a[:, 0], 50.0)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_advanced_process_validation():
    with pytest.raises(ValueError, match="beta"):
        create("sabr", beta=1.5)
    with pytest.raises(ValueError, match="H"):
        create("rough_bergomi", H=0.6)
    with pytest.raises(ValueError, match="alpha \\+ beta"):
        create("garch", alpha=0.5, beta=0.5)
    with pytest.raises(ValueError, match="rho1"):
        create("two_factor", rho1=1.0)


def test_garch_volatility_clustering():
    paths = create("garch", omega=0.0002, alpha=0.12, beta=0.85) \
        .simulate(252, seed=4, n_paths=300)
    rets = np.diff(np.log(paths), axis=1).ravel()
    centered = (rets - rets.mean()) / rets.std()
    assert float((centered ** 4).mean() - 3.0) > 0.3      # 肥尾/聚集


def test_rough_bergomi_kernel_is_lower_triangular():
    kernel = create("rough_bergomi")._kernel(6, 1 / 252, 0.1)
    assert kernel.shape == (6, 6)
    assert np.all(np.triu(kernel, k=1) == 0.0)
    assert kernel[1, 0] > 0 and kernel[0, 1] == 0.0
