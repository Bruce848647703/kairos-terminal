"""均值回复类过程：对数 OU、CIR 方根过程。"""
from __future__ import annotations

import numpy as np

from .base import DEFAULT_DT, ParamSpec, PriceProcess, require_positive
from .registry import register


@register
class LogOUProcess(PriceProcess):
    key = "ou"
    name = "对数均值回复 (OU)"
    description = (
        "对数价格向长期均衡值回归：d(lnS) = κ(lnS̄ - lnS)dt + σdW。κ 越大回归越快。"
        "适合利率、价差、商品与「被锚定」的品种；用作股价时长趋势会被压制，"
        "正好可以用来测试趋势策略在无趋势世界里的真实表现。"
    )
    tags = ("均值回复", "价差")
    params = (
        ParamSpec("kappa", "回归速度 κ", 2.0, "float", 0.01, 20.0, 0.1),
        ParamSpec("long_run", "长期均衡价格", 100.0, "float", 0.01, 100000.0, 1.0),
        ParamSpec("sigma", "年化波动率", 0.20, "float", 0.0, 5.0, 0.01, unit="/年"),
    )

    def validate(self) -> None:
        require_positive("kappa", self["kappa"])
        require_positive("long_run", self["long_run"])
        if float(self["sigma"]) < 0:
            raise ValueError("参数 sigma 不能为负")

    def simulate(self, n_steps, *, dt=DEFAULT_DT, s0=100.0, seed=None, n_paths=1):
        n_steps, n_paths = self._prepare(n_steps, n_paths, s0, dt)
        rng = self._rng(seed)
        kappa = float(self["kappa"])
        anchor = np.log(float(self["long_run"]))
        sigma = float(self["sigma"])
        sqrt_dt = np.sqrt(dt)

        log_s = np.empty((n_paths, n_steps + 1))
        log_s[:, 0] = np.log(s0)
        z = rng.standard_normal((n_paths, n_steps))
        for t in range(1, n_steps + 1):
            prev = log_s[:, t - 1]
            log_s[:, t] = prev + kappa * (anchor - prev) * dt + sigma * sqrt_dt * z[:, t - 1]
        return np.exp(log_s)


@register
class CIRProcess(PriceProcess):
    key = "cir"
    name = "CIR 方根过程"
    description = (
        "dX = κ(θ-X)dt + σ√X dW：方差随水平值放大、永远非负，是 Heston 方差过程的原型，"
        "也常用来建模短期利率。用来观察「波动率随价格水平变化」对策略与定价的影响。"
    )
    tags = ("利率", "非负", "方根")
    params = (
        ParamSpec("kappa", "回归速度 κ", 1.5, "float", 0.01, 20.0, 0.1),
        ParamSpec("theta", "长期均值 θ", 100.0, "float", 0.01, 100000.0, 1.0),
        ParamSpec("sigma", "波动系数 σ", 3.0, "float", 0.0, 100.0, 0.1,
                  help="瞬时波动 = σ√X；X=100、σ=3 时约 30% 年化"),
    )

    def validate(self) -> None:
        require_positive("kappa", self["kappa"])
        require_positive("theta", self["theta"])
        if float(self["sigma"]) < 0:
            raise ValueError("参数 sigma 不能为负")

    def simulate(self, n_steps, *, dt=DEFAULT_DT, s0=100.0, seed=None, n_paths=1):
        n_steps, n_paths = self._prepare(n_steps, n_paths, s0, dt)
        rng = self._rng(seed)
        kappa, theta = float(self["kappa"]), float(self["theta"])
        sigma = float(self["sigma"])

        x = np.empty((n_paths, n_steps + 1))
        x[:, 0] = s0
        z = rng.standard_normal((n_paths, n_steps))
        for t in range(1, n_steps + 1):
            prev = np.maximum(x[:, t - 1], 0.0)
            x[:, t] = prev + kappa * (theta - prev) * dt + sigma * np.sqrt(prev * dt) * z[:, t - 1]
        np.maximum(x, 0.0, out=x)
        x[:, 0] = s0
        return x