"""进阶过程：SABR、粗糙 Bergomi、GARCH(1,1)、双因子随机波动率。"""
from __future__ import annotations

import numpy as np

from .base import DEFAULT_DT, ParamSpec, PriceProcess, require_positive
from .registry import register


@register
class SABRProcess(PriceProcess):
    key = "sabr"
    name = "SABR 随机波动率"
    description = (
        "SABR 的动态版本：dF = α·F^β·dW₁，dα = ν·α·dW₂，corr(dW₁,dW₂)=ρ。"
        "β 控制「价格越高波动越小」的弹性（β=1 退化对数正态，β=0 近似正态），"
        "ν 是 vol-of-vol、ρ 产生偏斜。常用来理解期权微笑的形状来源。"
    )
    tags = ("随机波动率", "微笑", "弹性")
    has_drift_param = True
    params = (
        ParamSpec("mu", "年化漂移率", 0.06, "float", -2.0, 2.0, 0.01, unit="/年"),
        ParamSpec("alpha", "初始波动参数 α", 0.25, "float", 0.001, 5.0, 0.01,
                  help="β=1 时即为初始波动率"),
        ParamSpec("beta", "弹性 β", 0.8, "float", 0.0, 1.0, 0.05,
                  help="1=对数正态，0=正态（常说的 SABR β）"),
        ParamSpec("nu", "波动率的波动 ν", 0.4, "float", 0.0, 5.0, 0.05),
        ParamSpec("rho", "价格-波动相关 ρ", -0.5, "float", -0.999, 0.999, 0.01),
    )

    def validate(self) -> None:
        require_positive("alpha", self["alpha"])
        if not 0.0 <= float(self["beta"]) <= 1.0:
            raise ValueError("参数 beta 必须在 [0, 1] 内")
        if float(self["nu"]) < 0:
            raise ValueError("参数 nu 不能为负")
        if abs(float(self["rho"])) >= 1.0:
            raise ValueError("参数 rho 必须在 (-1, 1) 内")

    def simulate(self, n_steps, *, dt=DEFAULT_DT, s0=100.0, seed=None, n_paths=1):
        n_steps, n_paths = self._prepare(n_steps, n_paths, s0, dt)
        rng = self._rng(seed)
        mu = float(self["mu"])
        beta, nu, rho = float(self["beta"]), float(self["nu"]), float(self["rho"])
        sqrt_dt = np.sqrt(dt)

        fwd = np.full(n_paths, float(s0))
        alpha = np.full(n_paths, float(self["alpha"]))
        log_path = np.zeros((n_paths, n_steps + 1))
        for t in range(1, n_steps + 1):
            z1 = rng.standard_normal(n_paths)
            z2 = rho * z1 + np.sqrt(max(1.0 - rho * rho, 0.0)) * rng.standard_normal(n_paths)
            local = alpha * np.power(np.maximum(fwd, 1e-12), beta - 1.0)
            log_path[:, t] = (log_path[:, t - 1]
                              + (mu - 0.5 * local * local) * dt
                              + local * sqrt_dt * z1)
            alpha = alpha * np.exp(-0.5 * nu * nu * dt + nu * sqrt_dt * z2)
            fwd = s0 * np.exp(log_path[:, t])
        return s0 * np.exp(log_path)


@register
class RoughBergomiProcess(PriceProcess):
    key = "rough_bergomi"
    name = "粗糙 Bergomi (rough vol)"
    description = (
        "H<0.5 的分数布朗运动驱动的方差：Y_t=∫(t-s)^{H-1/2}dW_s，v_t=ξ·exp(ηY_t-½η²t^{2H})。"
        "粗糙波动率能同时解释「波动率路径比价格更粗糙」与陡峭的短端微笑，"
        "是 2018 年后期权建模的主流方向。这里用离散 Volterra 卷积实现。"
    )
    tags = ("粗糙波动率", "分数布朗", "微笑")
    has_drift_param = True
    params = (
        ParamSpec("mu", "年化漂移率", 0.05, "float", -2.0, 2.0, 0.01, unit="/年"),
        ParamSpec("H", "Hurst 指数 H", 0.1, "float", 0.02, 0.49, 0.01,
                  help="越小路径越粗糙；0.5 退化为普通布朗"),
        ParamSpec("eta", "vol-of-vol η", 1.2, "float", 0.01, 5.0, 0.05),
        ParamSpec("rho", "价格-波动相关 ρ", -0.75, "float", -0.999, 0.999, 0.01),
        ParamSpec("xi", "初始方差 ξ", 0.04, "float", 0.0001, 4.0, 0.005,
                  help="0.04 对应初始波动率 20%"),
    )

    def validate(self) -> None:
        if not 0.0 < float(self["H"]) < 0.5:
            raise ValueError("参数 H 必须在 (0, 0.5) 内（H=0.5 请用 gbm）")
        require_positive("eta", self["eta"])
        require_positive("xi", self["xi"])
        if abs(float(self["rho"])) >= 1.0:
            raise ValueError("参数 rho 必须在 (-1, 1) 内")

    @staticmethod
    def _kernel(n_steps: int, dt: float, hurst: float) -> np.ndarray:
        """下三角 Volterra 卷积核：K[t, s] = √(2H)·((t-s)dt)^{H-1/2}·√dt。"""
        t = np.arange(1, n_steps + 1)[:, None].astype("float64")
        s = np.arange(n_steps)[None, :].astype("float64")
        lag = t - s
        kernel = np.zeros((n_steps, n_steps))
        mask = lag > 0
        kernel[mask] = (np.sqrt(2.0 * hurst)
                        * np.power(lag[mask] * dt, hurst - 0.5)
                        * np.sqrt(dt))
        return kernel

    def simulate(self, n_steps, *, dt=DEFAULT_DT, s0=100.0, seed=None, n_paths=1):
        n_steps, n_paths = self._prepare(n_steps, n_paths, s0, dt)
        rng = self._rng(seed)
        mu = float(self["mu"])
        hurst, eta = float(self["H"]), float(self["eta"])
        rho, xi = float(self["rho"]), float(self["xi"])

        z_vol = rng.standard_normal((n_paths, n_steps))
        z_ind = rng.standard_normal((n_paths, n_steps))
        z_price = rho * z_vol + np.sqrt(max(1.0 - rho * rho, 0.0)) * z_ind

        # Y_t = √(2H)∫(t-s)^{H-1/2}dW_s，一次性卷积（BLAS 加速）
        y = z_vol @ self._kernel(n_steps, dt, hurst).T            # (P, T)
        grid = np.arange(1, n_steps + 1) * dt
        variance = xi * np.exp(eta * y - 0.5 * eta * eta * np.power(grid, 2.0 * hurst))
        np.maximum(variance, 1e-12, out=variance)

        log_ret = (mu - 0.5 * variance) * dt + np.sqrt(variance * dt) * z_price
        return self._from_log_increments(s0, log_ret, n_paths, n_steps)


@register
class GarchProcess(PriceProcess):
    key = "garch"
    name = "GARCH(1,1)"
    description = (
        "离散时间的波动率聚集模型：v_{t+1}=ω+α·r_t²/dt+β·v_t。α 越大对冲击反应越快，"
        "β 越大波动率记忆越长，α+β 接近 1 时出现长记忆。与连续时间 SV 不同，"
        "它把「今天的波动决定明天」写成显式递推，是业界最常用的风险模型基线。"
    )
    tags = ("波动率聚集", "离散时间", "风险模型")
    has_drift_param = True
    params = (
        ParamSpec("mu", "年化漂移率", 0.06, "float", -2.0, 2.0, 0.01, unit="/年"),
        ParamSpec("omega", "常数项 ω", 0.0008, "float", 1e-8, 0.5, 0.0001,
                  help="长期方差 = ω/(1-α-β)"),
        ParamSpec("alpha", "冲击系数 α", 0.08, "float", 0.0, 0.99, 0.01),
        ParamSpec("beta", "记忆系数 β", 0.90, "float", 0.0, 0.999, 0.01),
    )

    def validate(self) -> None:
        require_positive("omega", self["omega"])
        alpha, beta = float(self["alpha"]), float(self["beta"])
        if alpha < 0 or beta < 0:
            raise ValueError("alpha / beta 不能为负")
        if alpha + beta >= 1.0:
            raise ValueError("需满足 alpha + beta < 1（否则方差非平稳）")

    def simulate(self, n_steps, *, dt=DEFAULT_DT, s0=100.0, seed=None, n_paths=1):
        n_steps, n_paths = self._prepare(n_steps, n_paths, s0, dt)
        rng = self._rng(seed)
        mu = float(self["mu"])
        omega, alpha, beta = float(self["omega"]), float(self["alpha"]), float(self["beta"])
        sqrt_dt = np.sqrt(dt)

        variance = np.full(n_paths, omega / (1.0 - alpha - beta))
        log_path = np.zeros((n_paths, n_steps + 1))
        for t in range(1, n_steps + 1):
            shock = np.sqrt(variance * dt) * rng.standard_normal(n_paths)
            log_path[:, t] = log_path[:, t - 1] + (mu - 0.5 * variance) * dt + shock
            variance = omega + alpha * (shock * shock / dt) + beta * variance
            variance = np.maximum(variance, 1e-12)
        return s0 * np.exp(log_path)


@register
class TwoFactorVolProcess(PriceProcess):
    key = "two_factor"
    name = "双因子随机波动率"
    description = (
        "v = v₁ + v₂，两个独立的 CIR 方差因子（一快一慢）分别驱动，价格与两者相关。"
        "单因子 Heston 只能拟合一个期限的 smile，双因子能同时拟合短期与长期波动结构，"
        "是利率与长周期期权的标准配置。"
    )
    tags = ("随机波动率", "双因子", "期限结构")
    has_drift_param = True
    params = (
        ParamSpec("mu", "年化漂移率", 0.06, "float", -2.0, 2.0, 0.01, unit="/年"),
        ParamSpec("v1_0", "快因子初始方差", 0.02, "float", 0.0, 4.0, 0.005),
        ParamSpec("kappa1", "快因子回归 κ₁", 8.0, "float", 0.01, 50.0, 0.1),
        ParamSpec("theta1", "快因子长期 θ₁", 0.02, "float", 0.0, 4.0, 0.005),
        ParamSpec("xi1", "快因子 vol-of-vol ξ₁", 0.6, "float", 0.0, 5.0, 0.05),
        ParamSpec("v2_0", "慢因子初始方差", 0.02, "float", 0.0, 4.0, 0.005),
        ParamSpec("kappa2", "慢因子回归 κ₂", 0.8, "float", 0.01, 50.0, 0.05),
        ParamSpec("theta2", "慢因子长期 θ₂", 0.03, "float", 0.0, 4.0, 0.005),
        ParamSpec("xi2", "慢因子 vol-of-vol ξ₂", 0.25, "float", 0.0, 5.0, 0.05),
        ParamSpec("rho1", "价格-快因子相关 ρ₁", -0.6, "float", -0.999, 0.999, 0.01),
        ParamSpec("rho2", "价格-慢因子相关 ρ₂", -0.3, "float", -0.999, 0.999, 0.01),
    )

    def validate(self) -> None:
        for name in ("kappa1", "kappa2"):
            require_positive(name, self[name])
        for name in ("v1_0", "theta1", "v2_0", "theta2"):
            if float(self[name]) < 0:
                raise ValueError(f"参数 {name} 不能为负")
        for name in ("xi1", "xi2"):
            if float(self[name]) < 0:
                raise ValueError(f"参数 {name} 不能为负")
        for name in ("rho1", "rho2"):
            if abs(float(self[name])) >= 1.0:
                raise ValueError(f"参数 {name} 必须在 (-1, 1) 内")

    def simulate(self, n_steps, *, dt=DEFAULT_DT, s0=100.0, seed=None, n_paths=1):
        n_steps, n_paths = self._prepare(n_steps, n_paths, s0, dt)
        rng = self._rng(seed)
        mu = float(self["mu"])
        v1 = np.full(n_paths, float(self["v1_0"]))
        v2 = np.full(n_paths, float(self["v2_0"]))
        k1, t1, x1 = float(self["kappa1"]), float(self["theta1"]), float(self["xi1"])
        k2, t2, x2 = float(self["kappa2"]), float(self["theta2"]), float(self["xi2"])
        rho1, rho2 = float(self["rho1"]), float(self["rho2"])
        sqrt_dt = np.sqrt(dt)

        log_path = np.zeros((n_paths, n_steps + 1))
        for t in range(1, n_steps + 1):
            z1 = rng.standard_normal(n_paths)
            e1 = rng.standard_normal(n_paths)
            e2 = rng.standard_normal(n_paths)
            total = np.maximum(v1 + v2, 0.0)
            log_path[:, t] = (log_path[:, t - 1]
                              + (mu - 0.5 * total) * dt
                              + np.sqrt(total * dt) * z1)
            zp1 = rho1 * z1 + np.sqrt(max(1.0 - rho1 ** 2, 0.0)) * e1
            zp2 = rho2 * z1 + np.sqrt(max(1.0 - rho2 ** 2, 0.0)) * e2
            v1 = np.maximum(v1 + k1 * (t1 - np.maximum(v1, 0.0)) * dt
                            + x1 * np.sqrt(np.maximum(v1, 0.0) * dt) * zp1, 0.0)
            v2 = np.maximum(v2 + k2 * (t2 - np.maximum(v2, 0.0)) * dt
                            + x2 * np.sqrt(np.maximum(v2, 0.0) * dt) * zp2, 0.0)
        return s0 * np.exp(log_path)