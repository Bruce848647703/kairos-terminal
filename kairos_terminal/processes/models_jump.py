"""跳跃类过程：Merton 跳跃扩散、Bates（随机波动 + 跳跃）。"""
from __future__ import annotations

import numpy as np

from .base import DEFAULT_DT, ParamSpec, PriceProcess, require_positive
from .models_diffusion import HestonProcess
from .registry import register


@register
class MertonJumpProcess(PriceProcess):
    key = "merton"
    name = "Merton 跳跃扩散"
    description = (
        "在 GBM 上叠加复合泊松跳跃：dS/S = (μ-λk)dt + σdW + (e^J-1)dN。"
        "跳跃幅度 J~N(μ_J, σ_J²) 让单日出现大幅跳空，能生成真实市场里的肥尾，"
        "但方向对称、没有波动率聚集。"
    )
    tags = ("跳跃", "肥尾")
    has_drift_param = True
    params = (
        ParamSpec("mu", "年化漂移率", 0.08, "float", -2.0, 2.0, 0.01, unit="/年"),
        ParamSpec("sigma", "扩散波动率", 0.18, "float", 0.0, 5.0, 0.01, unit="/年"),
        ParamSpec("lam", "跳跃强度 λ (次/年)", 0.5, "float", 0.0, 50.0, 0.1),
        ParamSpec("jump_mu", "跳跃对数幅度均值 μ_J", -0.02, "float", -2.0, 2.0, 0.005),
        ParamSpec("jump_sigma", "跳跃对数幅度标准差 σ_J", 0.10, "float", 0.0, 2.0, 0.005),
    )

    def validate(self) -> None:
        if float(self["lam"]) < 0:
            raise ValueError("参数 lam 不能为负")
        if float(self["jump_sigma"]) < 0:
            raise ValueError("参数 jump_sigma 不能为负")

    def simulate(self, n_steps, *, dt=DEFAULT_DT, s0=100.0, seed=None, n_paths=1):
        n_steps, n_paths = self._prepare(n_steps, n_paths, s0, dt)
        rng = self._rng(seed)
        mu, sigma = float(self["mu"]), float(self["sigma"])
        lam = float(self["lam"])
        jmu, jsig = float(self["jump_mu"]), float(self["jump_sigma"])

        k = np.exp(jmu + 0.5 * jsig * jsig) - 1.0          # 跳跃补偿项
        z = rng.standard_normal((n_paths, n_steps))
        jumps = rng.poisson(lam * dt, size=(n_paths, n_steps)).astype("float64")
        jump_z = rng.standard_normal((n_paths, n_steps))
        jump_part = jumps * jmu + np.sqrt(jumps) * jsig * jump_z

        increments = ((mu - 0.5 * sigma * sigma - lam * k) * dt
                      + sigma * np.sqrt(dt) * z
                      + jump_part)
        return self._from_log_increments(s0, increments, n_paths, n_steps)


@register
class BatesProcess(PriceProcess):
    key = "bates"
    name = "Bates（随机波动 + 跳跃）"
    description = (
        "Heston 随机波动叠加 Merton 跳跃：既能复现波动率聚集与杠杆效应，又能复现"
        "财报/黑天鹅式跳空。学术界常用的「最接近真实期权微笑」的简约模型之一，"
        "也是检验 Black-Scholes 定价误差最苛刻的场景。"
    )
    tags = ("随机波动率", "跳跃", "肥尾", "杠杆效应")
    has_drift_param = True
    params = HestonProcess.params + (
        ParamSpec("lam", "跳跃强度 λ (次/年)", 0.3, "float", 0.0, 50.0, 0.1),
        ParamSpec("jump_mu", "跳跃对数幅度均值 μ_J", -0.03, "float", -2.0, 2.0, 0.005),
        ParamSpec("jump_sigma", "跳跃对数幅度标准差 σ_J", 0.10, "float", 0.0, 2.0, 0.005),
    )

    def validate(self) -> None:
        HestonProcess.validate(self)
        if float(self["lam"]) < 0:
            raise ValueError("参数 lam 不能为负")
        if float(self["jump_sigma"]) < 0:
            raise ValueError("参数 jump_sigma 不能为负")

    def simulate(self, n_steps, *, dt=DEFAULT_DT, s0=100.0, seed=None, n_paths=1):
        n_steps, n_paths = self._prepare(n_steps, n_paths, s0, dt)
        rng = self._rng(seed)
        mu = float(self["mu"])
        v0, kappa = float(self["v0"]), float(self["kappa"])
        theta, xi, rho = float(self["theta"]), float(self["xi"]), float(self["rho"])
        lam = float(self["lam"])
        jmu, jsig = float(self["jump_mu"]), float(self["jump_sigma"])
        k = np.exp(jmu + 0.5 * jsig * jsig) - 1.0

        log_s = np.zeros((n_paths, n_steps + 1))
        var = np.full((n_paths, n_steps + 1), v0)
        for t in range(1, n_steps + 1):
            v_prev = np.maximum(var[:, t - 1], 0.0)
            z1 = rng.standard_normal(n_paths)
            z2 = rho * z1 + np.sqrt(max(1.0 - rho * rho, 0.0)) * rng.standard_normal(n_paths)
            counts = rng.poisson(lam * dt, size=n_paths).astype("float64")
            jump = counts * jmu + np.sqrt(counts) * jsig * rng.standard_normal(n_paths)

            log_s[:, t] = (log_s[:, t - 1]
                           + (mu - 0.5 * v_prev - lam * k) * dt
                           + np.sqrt(v_prev * dt) * z1
                           + jump)
            var[:, t] = (v_prev
                         + kappa * (theta - v_prev) * dt
                         + xi * np.sqrt(v_prev * dt) * z2)
        paths = s0 * np.exp(log_s)
        paths[:, 0] = s0
        return paths