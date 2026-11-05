"""定价与对冲实验：Black-Scholes / CRR 二叉树 / 任意过程的蒙特卡洛。

这个模块是项目的「金融数学」半边：终端负责**生成**价格路径，定价模块负责
**评价**这些路径值多少钱。两者共用同一个 :class:`~kairos_terminal.processes.base.PriceProcess`，
因此可以直接回答这类问题：

- 用 Heston 生成真实路径，却用 Black-Scholes 定价与 Delta 对冲，PnL 分布长什么样？
- 同一个期权，BS、二叉树、蒙特卡洛三个价格差多少？收敛速度如何？
- 障碍/亚式/数字期权在肥尾过程下的价格比 BS 高多少？

所有实现只依赖 ``numpy`` 与标准库 ``math``（正态 CDF 用 ``math.erf``），
缺失 ``scipy`` 也能完整运行。
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

from .processes.base import PriceProcess

_SQRT_2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)

KINDS = ("call", "put")


# --------------------------------------------------------------------------- #
# 正态分布工具
# --------------------------------------------------------------------------- #
def norm_cdf(x: float) -> float:
    """标准正态分布函数 Φ(x)（用 ``math.erf``，不依赖 scipy）。"""
    return 0.5 * (1.0 + math.erf(float(x) / _SQRT_2))


def norm_pdf(x: float) -> float:
    """标准正态密度函数 φ(x)。"""
    return math.exp(-0.5 * float(x) ** 2) / _SQRT_2PI


def _check_kind(kind: str) -> str:
    kind = str(kind).lower()
    if kind not in KINDS:
        raise ValueError(f"kind 必须是 {KINDS}，收到 {kind!r}")
    return kind


def _check_inputs(spot: float, strike: float, maturity: float, vol: float) -> None:
    if not spot > 0:
        raise ValueError(f"spot 必须为正，收到 {spot!r}")
    if not strike > 0:
        raise ValueError(f"strike 必须为正，收到 {strike!r}")
    if maturity < 0:
        raise ValueError(f"maturity 不能为负，收到 {maturity!r}")
    if vol < 0:
        raise ValueError(f"vol 不能为负，收到 {vol!r}")


# --------------------------------------------------------------------------- #
# Black-Scholes
# --------------------------------------------------------------------------- #
def black_scholes(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
    kind: str = "call",
    dividend: float = 0.0,
) -> Dict[str, Any]:
    """Black-Scholes-Merton 闭式解，返回价格与五个希腊字母。

    ``vega`` 是 ∂V/∂σ（波动率变动 1.00 的价格变化，实际常用 vega/100 表示 1 个
    波动率点）；``theta`` 是 ∂V/∂t（每年），``rho`` 是 ∂V/∂r（利率变动 1.00）。
    """
    kind = _check_kind(kind)
    _check_inputs(spot, strike, maturity, vol)
    spot, strike = float(spot), float(strike)
    rate, dividend, vol = float(rate), float(dividend), float(vol)

    sign = 1.0 if kind == "call" else -1.0
    intrinsic = max(sign * (spot - strike), 0.0)
    if maturity == 0.0:
        return {
            "price": intrinsic, "delta": float(sign if intrinsic > 0 else 0.0),
            "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0,
            "d1": 0.0, "d2": 0.0, "kind": kind, "intrinsic": intrinsic,
        }
    if vol == 0.0:                                   # 退化：确定性的贴现内在价值
        fwd = spot * math.exp((rate - dividend) * maturity)
        price = math.exp(-rate * maturity) * max(sign * (fwd - strike), 0.0)
        delta = (math.exp(-dividend * maturity)
                 * (1.0 if (sign > 0 and fwd > strike) else 0.0)
                 if sign > 0 else
                 -math.exp(-dividend * maturity) * (1.0 if fwd < strike else 0.0))
        return {"price": price, "delta": delta, "gamma": 0.0, "vega": 0.0,
                "theta": 0.0, "rho": 0.0, "d1": 0.0, "d2": 0.0, "kind": kind,
                "intrinsic": intrinsic}

    sqrt_t = math.sqrt(maturity)
    d1 = (math.log(spot / strike) + (rate - dividend + 0.5 * vol * vol) * maturity) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    df_q = math.exp(-dividend * maturity)
    df_r = math.exp(-rate * maturity)

    if kind == "call":
        price = spot * df_q * norm_cdf(d1) - strike * df_r * norm_cdf(d2)
        delta = df_q * norm_cdf(d1)
        theta = (-spot * df_q * norm_pdf(d1) * vol / (2.0 * sqrt_t)
                 - rate * strike * df_r * norm_cdf(d2)
                 + dividend * spot * df_q * norm_cdf(d1))
        rho = strike * maturity * df_r * norm_cdf(d2)
    else:
        price = strike * df_r * norm_cdf(-d2) - spot * df_q * norm_cdf(-d1)
        delta = -df_q * norm_cdf(-d1)
        theta = (-spot * df_q * norm_pdf(d1) * vol / (2.0 * sqrt_t)
                 + rate * strike * df_r * norm_cdf(-d2)
                 - dividend * spot * df_q * norm_cdf(-d1))
        rho = -strike * maturity * df_r * norm_cdf(-d2)

    gamma = df_q * norm_pdf(d1) / (spot * vol * sqrt_t)
    vega = spot * df_q * norm_pdf(d1) * sqrt_t

    return {
        "price": price, "delta": delta, "gamma": gamma, "vega": vega,
        "theta": theta, "rho": rho, "d1": d1, "d2": d2,
        "kind": kind, "intrinsic": intrinsic,
    }


def implied_vol(
    price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    kind: str = "call",
    dividend: float = 0.0,
    *,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> float:
    """由期权价格反解隐含波动率（二分法，稳健、无需初值）。"""
    kind = _check_kind(kind)
    _check_inputs(spot, strike, maturity, 0.0)
    if maturity <= 0:
        raise ValueError("implied_vol 需要 maturity > 0")
    price = float(price)
    sign = 1.0 if kind == "call" else -1.0
    lower = math.exp(-rate * maturity) * max(sign * (spot * math.exp((rate - dividend) * maturity) - strike), 0.0)
    upper = spot * math.exp(-dividend * maturity) if kind == "call" else strike * math.exp(-rate * maturity)
    if not (lower - 1e-10 <= price <= upper + 1e-10):
        raise ValueError(
            f"价格 {price:.6f} 超出无套利区间 [{lower:.6f}, {upper:.6f}]，无法反解隐含波动率")

    lo, hi = 1e-6, 5.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        value = black_scholes(spot, strike, maturity, rate, mid, kind, dividend)["price"]
        if abs(value - price) < tol:
            return mid
        if value < price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
# 二叉树
# --------------------------------------------------------------------------- #
def binomial_crr(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
    kind: str = "call",
    steps: int = 200,
    *,
    american: bool = False,
    dividend: float = 0.0,
) -> Dict[str, Any]:
    """Cox-Ross-Rubinstein 二叉树（支持美式提前行权）。"""
    kind = _check_kind(kind)
    _check_inputs(spot, strike, maturity, vol)
    steps = int(steps)
    if steps < 1:
        raise ValueError("steps 必须 >= 1")
    if maturity == 0:
        intrinsic = max(spot - strike, 0.0) if kind == "call" else max(strike - spot, 0.0)
        return {"price": intrinsic, "steps": steps, "american": bool(american)}

    dt = maturity / steps
    u = math.exp(vol * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-rate * dt)
    p = (math.exp((rate - dividend) * dt) - d) / (u - d)
    if not (0.0 < p < 1.0):
        raise ValueError("参数导致风险中性概率越界（请检查 rate / vol / steps）")

    sign = 1.0 if kind == "call" else -1.0
    j = np.arange(steps + 1)
    terminal = spot * (u ** (steps - j)) * (d ** j)
    values = np.maximum(sign * (terminal - strike), 0.0)

    for i in range(steps - 1, -1, -1):
        values = disc * (p * values[:-1] + (1.0 - p) * values[1:])
        if american:
            j = np.arange(i + 1)
            underlying = spot * (u ** (i - j)) * (d ** j)
            intrinsic = np.maximum(sign * (underlying - strike), 0.0)
            values = np.maximum(values, intrinsic)

    return {"price": float(values[0]), "steps": steps, "american": bool(american)}


# --------------------------------------------------------------------------- #
# 蒙特卡洛
# --------------------------------------------------------------------------- #
PAYOFF_KINDS = ("vanilla", "asian", "digital", "barrier")
BARRIER_KINDS = ("up-out", "down-out", "up-in", "down-in")


def monte_carlo(
    process: PriceProcess,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    kind: str = "call",
    *,
    dividend: float = 0.0,
    n_paths: int = 20000,
    n_steps: int = 252,
    seed: Optional[int] = None,
    risk_neutral: bool = True,
    payoff: str = "vanilla",
    barrier: Optional[float] = None,
    barrier_kind: str = "up-out",
) -> Dict[str, Any]:
    """用**同一个价格过程**做蒙特卡洛定价。

    ``risk_neutral=True`` 时会把过程的漂移替换为 ``rate - dividend``（存在 ``mu``
    参数的过程才会被替换，例如 GBM/Heston/Merton/VG）；均值回复过程保持原样，
    用于观察「模型本身不满足风险中性」时的定价偏差。
    """
    kind = _check_kind(kind)
    _check_inputs(spot, strike, maturity, 0.0)
    payoff = str(payoff).lower()
    if payoff not in PAYOFF_KINDS:
        raise ValueError(f"payoff 必须是 {PAYOFF_KINDS}，收到 {payoff!r}")
    if n_paths < 100:
        raise ValueError("n_paths 至少为 100 以保证统计误差可控")
    if n_steps < 1:
        raise ValueError("n_steps 必须 >= 1")
    if maturity == 0:                                # 已到期：直接付内在价值
        intrinsic = (max(spot - strike, 0.0) if kind == "call"
                     else max(strike - spot, 0.0))
        return {"price": intrinsic, "stderr": 0.0, "ci_low": intrinsic,
                "ci_high": intrinsic, "n_paths": 0, "n_steps": 0,
                "kind": kind, "payoff": payoff, "discount": 1.0,
                "terminal_mean": spot, "terminal_percentiles": [spot] * 5}

    model = process.risk_neutral(rate, dividend) if risk_neutral else process
    dt = maturity / n_steps
    paths = model.simulate(n_steps, dt=dt, s0=spot, seed=seed, n_paths=int(n_paths))
    terminal = paths[:, -1]
    if kind == "call":
        vanilla = np.maximum(terminal - strike, 0.0)
    else:
        vanilla = np.maximum(strike - terminal, 0.0)

    extra: Dict[str, Any] = {}
    if payoff == "vanilla":
        payoff_values = vanilla
    elif payoff == "asian":
        average = paths[:, 1:].mean(axis=1)
        payoff_values = (np.maximum(average - strike, 0.0) if kind == "call"
                         else np.maximum(strike - average, 0.0))
        extra["monitoring"] = "算术平均"
    elif payoff == "digital":
        payoff_values = (terminal > strike).astype("float64") if kind == "call" \
            else (terminal < strike).astype("float64")
    else:                                            # barrier
        if barrier is None or not float(barrier) > 0:
            raise ValueError("障碍期权需要给出正的 barrier")
        if barrier_kind not in BARRIER_KINDS:
            raise ValueError(f"barrier_kind 必须是 {BARRIER_KINDS}")
        level = float(barrier)
        if barrier_kind.startswith("up"):
            hit = paths.max(axis=1) >= level
        else:
            hit = paths.min(axis=1) <= level
        alive = ~hit if barrier_kind.endswith("out") else hit
        payoff_values = np.where(alive, vanilla, 0.0)
        extra["barrier"] = level
        extra["barrier_kind"] = barrier_kind

    discount = math.exp(-rate * maturity)
    samples = discount * payoff_values
    price = float(samples.mean())
    stderr = float(samples.std(ddof=1) / math.sqrt(samples.size))
    return {
        "price": price,
        "stderr": stderr,
        "ci_low": price - 1.96 * stderr,
        "ci_high": price + 1.96 * stderr,
        "n_paths": int(n_paths),
        "n_steps": int(n_steps),
        "kind": kind,
        "payoff": payoff,
        "discount": discount,
        "terminal_mean": float(terminal.mean()),
        "terminal_percentiles": np.percentile(terminal, [5, 25, 50, 75, 95]).tolist(),
        **extra,
    }


# --------------------------------------------------------------------------- #
# 动态对冲误差实验
# --------------------------------------------------------------------------- #
def delta_hedge_experiment(
    process: PriceProcess,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    hedge_vol: float,
    kind: str = "call",
    *,
    dividend: float = 0.0,
    n_paths: int = 2000,
    n_steps: int = 52,
    seed: Optional[int] = None,
    risk_neutral: bool = False,
) -> Dict[str, Any]:
    """卖出 1 份期权并用 BS Delta 逐步对冲，返回终端 PnL 分布。

    如果「真实过程」就是波动率为 ``hedge_vol`` 的 GBM，那么 PnL 均值应接近 0，
    离散度只来自离散对冲；如果真实过程是 Heston/跳跃，均值与方差都会显著偏离 ——
    这个偏差就是**模型错配的代价**。
    """
    kind = _check_kind(kind)
    _check_inputs(spot, strike, maturity, hedge_vol)
    if maturity <= 0 or n_steps < 1:
        raise ValueError("需要 maturity > 0 且 n_steps >= 1")

    model = process.risk_neutral(rate, dividend) if risk_neutral else process
    dt = maturity / n_steps
    paths = model.simulate(n_steps, dt=dt, s0=spot, seed=seed, n_paths=int(n_paths))

    premium = black_scholes(spot, strike, maturity, rate, hedge_vol, kind, dividend)["price"]
    is_call = kind == "call"
    n_paths = int(n_paths)

    # 逐路径模拟「卖出 1 份期权 + 用 BS Delta 自融资对冲」的现金账户
    cash = np.full(n_paths, premium)                     # 收到权利金
    shares = np.zeros(n_paths)
    grow = math.exp(rate * dt)

    for i in range(n_steps):
        tau = maturity - i * dt
        s_i = paths[:, i]
        if tau <= 0:
            nd1 = (s_i > strike).astype("float64") if is_call else (s_i < strike).astype("float64")
        else:
            d1 = (np.log(s_i / strike) + (rate - dividend + 0.5 * hedge_vol ** 2) * tau) \
                / (hedge_vol * math.sqrt(tau))
            nd1 = np.array([norm_cdf(x) for x in d1])
        delta = math.exp(-dividend * tau) * (nd1 if is_call else -(1.0 - nd1))

        cash -= (delta - shares) * s_i                   # 调仓：买入/卖出股票
        shares = delta
        if dividend:
            cash += dividend * shares * s_i * dt         # 持有股票收到的股息
        cash *= grow                                     # 现金账户计息

    terminal = paths[:, -1]
    payoff = np.maximum(terminal - strike, 0.0) if is_call else np.maximum(strike - terminal, 0.0)
    # 初始现金 = 权利金，买入对冲组合；期末「对冲组合价值 - 应付赔付」即为套保盈亏
    pnl = cash + shares * terminal - payoff

    hist, edges = np.histogram(pnl, bins=24)
    return {
        "premium": premium,
        "hedge_vol": float(hedge_vol),
        "mean_pnl": float(pnl.mean()),
        "std_pnl": float(pnl.std(ddof=1)),
        "pnl_percentiles": np.percentile(pnl, [1, 5, 25, 50, 75, 95, 99]).tolist(),
        "prob_loss": float((pnl < 0).mean()),
        "n_paths": int(n_paths),
        "n_steps": int(n_steps),
        "hist": {"counts": hist.tolist(), "edges": edges.tolist()},
        "sample_pnl": pnl[:200].tolist(),
    }


def compare_models(
    process: PriceProcess,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
    kind: str = "call",
    *,
    dividend: float = 0.0,
    n_paths: int = 20000,
    n_steps: int = 252,
    steps_binomial: int = 300,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """同一份期权在 BS / 二叉树 / 蒙特卡洛三种方法下的价格对照。"""
    bs = black_scholes(spot, strike, maturity, rate, vol, kind, dividend)
    tree = binomial_crr(spot, strike, maturity, rate, vol, kind, steps_binomial,
                        dividend=dividend)
    mc = monte_carlo(process, spot, strike, maturity, rate, kind,
                     dividend=dividend, n_paths=n_paths, n_steps=n_steps, seed=seed)
    return {
        "black_scholes": bs["price"],
        "binomial": tree["price"],
        "monte_carlo": mc["price"],
        "mc_stderr": mc["stderr"],
        "mc_ci": [mc["ci_low"], mc["ci_high"]],
        "model_process": process.key,
        "greeks": {k: bs[k] for k in ("delta", "gamma", "vega", "theta", "rho")},
    }


__all__ = [
    "BARRIER_KINDS",
    "KINDS",
    "PAYOFF_KINDS",
    "binomial_crr",
    "black_scholes",
    "compare_models",
    "delta_hedge_experiment",
    "implied_vol",
    "monte_carlo",
    "norm_cdf",
    "norm_pdf",
]