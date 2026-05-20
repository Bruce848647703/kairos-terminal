"""区制转换（Markov switching）与方差 Gamma 过程。"""
from __future__ import annotations

import numpy as np

from .base import DEFAULT_DT, ParamSpec, PriceProcess, ensure_probability, require_positive
from .registry import register


@register
class RegimeSwitchingProcess(PriceProcess):
    key = "regime"
    name = "区制转换 (Markov)"
    description = (
        "市场在「平静」与「危机」两个状态间按马尔可夫链切换，两态的漂移与波动显著不同。"
        "它产生的收益序列是**非平稳**的：长期统计量稳定，但分段看完全不像同一个分布——"
        "用来检验策略/风险模型在结构性突变下会不会失灵。"
    )
    tags = ("非平稳", "结构性突变", "波动率聚集")
    params = (
        ParamSpec("mu_calm", "平静期漂移", 0.10, "float", -2.0, 2.0, 0.01, unit="/年"),
        ParamSpec("sigma_calm", "平静期波动", 0.15, "float", 0.0, 5.0, 0.01, unit="/年"),
        ParamSpec("mu_stress", "危机期漂移", -0.35, "float", -2.0, 2.0, 0.01, unit="/年"),
        ParamSpec("sigma_stress", "危机期波动", 0.45, "float", 0.0, 5.0, 0.01, unit="/年"),
        ParamSpec("p_enter", "平静→危机 日概率", 0.02, "float", 0.0, 1.0, 0.005),
        ParamSpec("p_exit", "危机→平静 日概率", 0.10, "float", 0.0, 1.0, 0.005),
        ParamSpec("start_stress", "从危机态开始", False, "bool"),
    )

    def validate(self) -> None:
        ensure_probability("p_enter", self["p_enter"])
        ensure_probability("p_exit", self["p_exit"])

    def simulate(self, n_steps, *, dt=DEFAULT_DT, s0=100.0, seed=None, n_paths=1):
        n_steps, n_paths = self._prepare(n_steps, n_paths, s0, dt)
        rng = self._rng(seed)
        mu_c, sg_c = float(self["mu_calm"]), float(self["sigma_calm"])
        mu_s, sg_s = float(self["mu_stress"]), float(self["sigma_stress"])
        p_enter, p_exit = float(self["p_enter"]), float(self["p_exit"])

        state = np.full(n_paths, 1 if self["start_stress"] else 0, dtype="int8")
        log_s = np.zeros((n_paths, n_steps + 1))
        sqrt_dt = np.sqrt(dt)
        for t in range(1, n_steps + 1):
            u = rng.random(n_paths)
            to_stress = (state == 0) & (u < p_enter)
            to_calm = (state == 1) & (u < p_exit)
            state = np.where(to_stress, 1, np.where(to_calm, 0, state)).astype("int8")

            mu = np.where(state == 1, mu_s, mu_c)
            sigma = np.where(state == 1, sg_s, sg_c)
            z = rng.standard_normal(n_paths)
            log_s[:, t] = log_s[:, t - 1] + (mu - 0.5 * sigma * sigma) * dt + sigma * sqrt_dt * z
        return s0 * np.exp(log_s)


@register
class VarianceGammaProcess(PriceProcess):
    key = "vg"
    name = "方差 Gamma (VG)"
    description = (
        "用 Gamma 时间变换构造的纯跳跃 Lévy 过程：X(t)=θG(t)+σW(G(t))，"
        "G(t)~Gamma(t/ν, ν)。ν 控制尾部厚度与偏度，能同时刻画「尖峰肥尾」和随机到达的"
        "信息流，是期权微笑建模里比 GBM 更轻量的选择。"
    )
    tags = ("Lévy", "肥尾", "尖峰")
    has_drift_param = True
    params = (
        ParamSpec("mu", "年化漂移率", 0.06, "float", -2.0, 2.0, 0.01, unit="/年"),
        ParamSpec("theta", "偏度参数 θ", -0.10, "float", -2.0, 2.0, 0.01,
                  help="负值使分布左偏（下跌更极端）"),
        ParamSpec("sigma", "扩散尺度 σ", 0.20, "float", 0.001, 5.0, 0.01),
        ParamSpec("nu", "Gamma 时间方差 ν", 0.25, "float", 0.001, 2.0, 0.01,
                  help="越大越肥尾；ν→0 退化为 GBM"),
    )

    def validate(self) -> None:
        require_positive("sigma", self["sigma"])
        require_positive("nu", self["nu"])
        # 需要 1 - ν(θ + σ²/2) > 0 才能定义指数矩（否则价格过程爆炸）
        bound = 1.0 - float(self["nu"]) * (float(self["theta"]) + 0.5 * float(self["sigma"]) ** 2)
        if bound <= 0:
            raise ValueError(
                "参数组合非法：需满足 1 - ν(θ + σ²/2) > 0，当前为 "
                f"{bound:.4f}（请减小 nu 或 theta/sigma）")

    def simulate(self, n_steps, *, dt=DEFAULT_DT, s0=100.0, seed=None, n_paths=1):
        n_steps, n_paths = self._prepare(n_steps, n_paths, s0, dt)
        rng = self._rng(seed)
        mu = float(self["mu"])
        theta, sigma, nu = float(self["theta"]), float(self["sigma"]), float(self["nu"])

        # 指数鞅补偿项：E[exp(dX)] = exp(omega·dt)
        omega = -np.log(1.0 - nu * (theta + 0.5 * sigma * sigma)) / nu

        dg = rng.gamma(shape=dt / nu, scale=nu, size=(n_paths, n_steps))
        z = rng.standard_normal((n_paths, n_steps))
        increments = ((mu - omega) * dt
                      + theta * dg
                      + sigma * np.sqrt(dg) * z)
        return self._from_log_increments(s0, increments, n_paths, n_steps)