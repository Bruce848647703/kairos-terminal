"""连续扩散类过程：几何布朗运动、确定性趋势、Heston 随机波动率。"""
from __future__ import annotations

import numpy as np

from .base import DEFAULT_DT, ParamSpec, PriceProcess, require_positive
from .registry import register


@register
class GBMProcess(PriceProcess):
    key = "gbm"
    name = "几何布朗运动"
    description = (
        "最经典的股价模型：dS = μS dt + σS dW。对数收益独立同分布、价格恒正，"
        "是 Black-Scholes 的底层假设，适合作为基准，但它**没有波动率聚集、没有肥尾**。"
    )
    tags = ("扩散", "基准")
    has_drift_param = True
    params = (
        ParamSpec("mu", "年化漂移率", 0.08, "float", -2.0, 2.0, 0.01,
                  unit="/年", help="风险中性定价时会被自动替换为无风险利率"),
        ParamSpec("sigma", "年化波动率", 0.25, "float", 0.0, 5.0, 0.01,
                  unit="/年", help="0 表示无波动的确定性路径"),
    )

    def validate(self) -> None:
        if float(self["sigma"]) < 0:
            raise ValueError("参数 sigma 不能为负")

    def simulate(self, n_steps, *, dt=DEFAULT_DT, s0=100.0, seed=None, n_paths=1):
        n_steps, n_paths = self._prepare(n_steps, n_paths, s0, dt)
        rng = self._rng(seed)
        mu, sigma = float(self["mu"]), float(self["sigma"])
        z = rng.standard_normal((n_paths, n_steps))
        increments = (mu - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * z
        return self._from_log_increments(s0, increments, n_paths, n_steps)


@register
class DeterministicTrendProcess(GBMProcess):
    key = "trend"
    name = "确定性趋势"
    description = (
        "零波动（或给定固定波动）的指数趋势：S_t = S_0·exp(μt)。用来做对照组——"
        "把「价格随机性」抽掉之后，策略与对冲的表现就只剩下成本项。"
    )
    tags = ("退化", "对照")
    params = (
        ParamSpec("mu", "年化漂移率", 0.08, "float", -2.0, 2.0, 0.01, unit="/年"),
        ParamSpec("sigma", "年化波动率", 0.0, "float", 0.0, 5.0, 0.01, unit="/年"),
    )


@register
class HestonProcess(PriceProcess):
    key = "heston"
    name = "Heston 随机波动率"
    description = (
        "让方差本身服从 CIR 过程：dv = κ(θ-v)dt + ξ√v dW₂，corr(dW₁,dW₂)=ρ。"
        "能复现「波动率聚集」与「杠杆效应」（ρ<0 时下跌伴随波动放大），"
        "因此同一套参数下 Black-Scholes 会系统性错价——这是本项目定价实验的主场景。"
    )
    tags = ("随机波动率", "肥尾", "杠杆效应")
    has_drift_param = True
    params = (
        ParamSpec("mu", "年化漂移率", 0.06, "float", -2.0, 2.0, 0.01, unit="/年"),
        ParamSpec("v0", "初始方差", 0.04, "float", 0.0001, 4.0, 0.005,
                  help="0.04 对应初始波动率 20%"),
        ParamSpec("kappa", "方差均值回复速度 κ", 2.0, "float", 0.01, 20.0, 0.1),
        ParamSpec("theta", "长期方差 θ", 0.04, "float", 0.0001, 4.0, 0.005,
                  help="长期波动率 = √θ"),
        ParamSpec("xi", "方差波动率 ξ (vol-of-vol)", 0.3, "float", 0.01, 5.0, 0.05),
        ParamSpec("rho", "价格-方差相关系数 ρ", -0.7, "float", -0.999, 0.999, 0.01,
                  help="负值产生杠杆效应（跌得越多、波动越大）"),
    )

    def validate(self) -> None:
        require_positive("v0", self["v0"])
        require_positive("theta", self["theta"])
        require_positive("kappa", self["kappa"])
        require_positive("xi", self["xi"])
        if abs(float(self["rho"])) >= 1.0:
            raise ValueError("参数 rho 必须在 (-1, 1) 内")
        # Feller 条件：2κθ > ξ² 时方差过程不会触零（不满足仅提示，不报错）
        self.feller_ok = 2.0 * float(self["kappa"]) * float(self["theta"]) > float(self["xi"]) ** 2

    def simulate(self, n_steps, *, dt=DEFAULT_DT, s0=100.0, seed=None, n_paths=1):
        n_steps, n_paths = self._prepare(n_steps, n_paths, s0, dt)
        rng = self._rng(seed)
        mu = float(self["mu"])
        v0, kappa = float(self["v0"]), float(self["kappa"])
        theta, xi, rho = float(self["theta"]), float(self["xi"]), float(self["rho"])
        sqrt_dt = np.sqrt(dt)

        log_s = np.zeros((n_paths, n_steps + 1))
        var = np.full((n_paths, n_steps + 1), v0)
        for t in range(1, n_steps + 1):
            v_prev = np.maximum(var[:, t - 1], 0.0)
            z1 = rng.standard_normal(n_paths)
            z2 = rho * z1 + np.sqrt(max(1.0 - rho * rho, 0.0)) * rng.standard_normal(n_paths)
            log_s[:, t] = (log_s[:, t - 1]
                           + (mu - 0.5 * v_prev) * dt
                           + np.sqrt(v_prev * dt) * z1)
            var[:, t] = (v_prev
                         + kappa * (theta - v_prev) * dt
                         + xi * np.sqrt(v_prev * dt) * z2)
        paths = s0 * np.exp(log_s)
        paths[:, 0] = s0
        return paths