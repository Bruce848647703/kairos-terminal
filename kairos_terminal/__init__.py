"""Kairos Terminal —— 随机过程驱动的交易终端（QMT-like）。

一句话概括：**把「价格从哪来」变成可配置项**。

- 每个品种挂一个可插拔的 :class:`~kairos_terminal.processes.base.PriceProcess`
  （GBM / Heston / 跳跃 / 区制转换 / 方差 Gamma / 历史重采样 / 自定义）；
- 终端提供 K 线、下单、持仓、盈亏、策略挂载 —— 与真实交易终端一致的操作体验；
- 定价实验室复用**同一套过程**做 BS / 二叉树 / 蒙特卡洛定价与 Delta 对冲实验，
  于是「模型错配的代价」可以被直接量化。

快速开始::

    pip install -e .
    python -m kairos_terminal            # 打开浏览器终端（默认 127.0.0.1:8848）
    python -m kairos_terminal demo       # 纯命令行跑一段
"""
from __future__ import annotations

from .broker import Broker, Fill, Order, Position
from .instruments import Instrument
from .market import Bar, MarketEngine, build_bars
from .pricing import (
    binomial_crr,
    black_scholes,
    compare_models,
    delta_hedge_experiment,
    implied_vol,
    monte_carlo,
)
from .session import Session, default_instruments, default_session
from .strategy import (
    OrderIntent,
    Strategy,
    StrategyContext,
    create_strategy,
    describe_strategies,
)

__version__ = "0.5.0"

__all__ = [
    "Bar",
    "Broker",
    "Fill",
    "Instrument",
    "MarketEngine",
    "Order",
    "OrderIntent",
    "Position",
    "Session",
    "Strategy",
    "StrategyContext",
    "__version__",
    "binomial_crr",
    "black_scholes",
    "build_bars",
    "compare_models",
    "create_strategy",
    "default_instruments",
    "default_session",
    "delta_hedge_experiment",
    "describe_strategies",
    "implied_vol",
    "monte_carlo",
]