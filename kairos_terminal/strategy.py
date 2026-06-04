"""策略基类与内置示例策略。

终端里策略与手工交易走**完全相同**的下单通道：策略在每根 bar 收盘后产出
:class:`OrderIntent`，引擎统一在下一根 bar 撮合，因此不存在任何未来函数。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .pricing import black_scholes


@dataclass
class OrderIntent:
    """策略想下的单。"""

    symbol: str
    side: str                       # buy / sell
    qty: int
    order_type: str = "market"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None


class StrategyContext:
    """策略能看到的只读视图（严格限制在「当前及历史 bar」）。"""

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    @property
    def index(self) -> int:
        return self._engine.index

    @property
    def symbols(self) -> List[str]:
        return list(self._engine.instruments)

    @property
    def prices(self) -> Dict[str, float]:
        return self._engine.prices()

    @property
    def cash(self) -> float:
        return self._engine.broker.cash

    @property
    def equity(self) -> float:
        return self._engine.broker.equity()

    @property
    def dt(self) -> float:
        return self._engine.dt

    @property
    def instruments(self) -> Dict[str, Any]:
        return dict(self._engine.instruments)

    def instrument(self, symbol: str) -> Any:
        inst = self._engine.instruments.get(str(symbol).upper())
        if inst is None:
            raise ValueError(f"未知品种 {symbol!r}")
        return inst

    def history(self, symbol: str, n: Optional[int] = None) -> List[Any]:
        bars = self._engine.bars(symbol)
        return bars if n is None else bars[-int(n):]

    def closes(self, symbol: str, n: int) -> np.ndarray:
        bars = self.history(symbol, n)
        return np.asarray([b.close for b in bars], dtype="float64")

    def position(self, symbol: str) -> float:
        pos = self._engine.broker.positions.get(symbol.upper())
        return pos.qty if pos else 0.0


class Strategy(ABC):
    """策略基类。"""

    name: str = "策略"
    description: str = ""
    default_params: Dict[str, Any] = {}

    def __init__(self, **params: Any) -> None:
        self.params = {**self.default_params, **params}
        self.reset()

    def reset(self) -> None:
        """会话重置时清理内部状态。"""

    @abstractmethod
    def on_bar(self, ctx: StrategyContext) -> List[OrderIntent]:
        """每根 bar 收盘后调用，返回要提交的委托（可返回空列表）。"""


class SmaCrossStrategy(Strategy):
    """双均线交叉：快线上穿慢线做多，下穿平仓。"""

    key = "sma_cross"
    name = "双均线交叉"
    description = "快慢均线金叉买入、死叉卖出，最经典的趋势跟随基线。"
    default_params = {"fast": 5, "slow": 20, "qty": 100, "symbol": ""}

    def on_bar(self, ctx: StrategyContext) -> List[OrderIntent]:
        symbol = self.params["symbol"] or ctx.symbols[0]
        fast, slow = int(self.params["fast"]), int(self.params["slow"])
        closes = ctx.closes(symbol, slow + 1)
        if closes.size < slow + 1:
            return []
        f_now, s_now = closes[-fast:].mean(), closes[-slow:].mean()
        f_prev, s_prev = closes[-fast - 1:-1].mean(), closes[-slow - 1:-1].mean()
        held = ctx.position(symbol)
        qty = int(self.params["qty"])
        if f_prev <= s_prev and f_now > s_now and held <= 0:
            return [OrderIntent(symbol, "buy", qty)]
        if f_prev >= s_prev and f_now < s_now and held > 0:
            return [OrderIntent(symbol, "sell", int(held))]
        return []


class GridStrategy(Strategy):
    """等比网格：每下跌一格买入、每上涨一格卖出。"""

    key = "grid"
    name = "网格交易"
    description = "在设定区间内均分网格，跌破网格买入、涨破网格卖出，赚取震荡收益。"
    default_params = {"levels": 6, "qty": 100, "symbol": ""}

    def reset(self) -> None:
        self._anchor: Optional[float] = None
        self._level = 0

    def on_bar(self, ctx: StrategyContext) -> List[OrderIntent]:
        symbol = self.params["symbol"] or ctx.symbols[0]
        price = ctx.prices[symbol]
        levels = max(int(self.params["levels"]), 2)
        if self._anchor is None:
            self._anchor = price
            self._level = 0
            return []

        edge = np.log(price / self._anchor) * levels
        level = int(np.floor(edge))
        qty = int(self.params["qty"])
        if level < self._level:                      # 价格下移 -> 买
            self._level = level
            return [OrderIntent(symbol, "buy", qty)]
        if level > self._level:                      # 价格上移 -> 卖
            self._level = level
            held = ctx.position(symbol)
            if held > 0:
                return [OrderIntent(symbol, "sell", int(min(held, qty)))]
        return []


class MomentumStrategy(Strategy):
    """时序动量：过去 N 根收益超过阈值做多，跌破负阈值离场/做空。"""

    key = "momentum"
    name = "时序动量"
    description = "过去 lookback 根 bar 的累计收益超过阈值即顺势开仓，反向则平仓。"
    default_params = {"lookback": 20, "threshold": 0.02, "qty": 100,
                      "symbol": "", "allow_short": False}

    def on_bar(self, ctx: StrategyContext) -> List[OrderIntent]:
        symbol = self.params["symbol"] or ctx.symbols[0]
        lookback = int(self.params["lookback"])
        closes = ctx.closes(symbol, lookback + 1)
        if closes.size < lookback + 1 or closes[0] <= 0:
            return []
        ret = closes[-1] / closes[0] - 1.0
        thr = float(self.params["threshold"])
        held = ctx.position(symbol)
        qty = int(self.params["qty"])
        if ret > thr and held <= 0:
            return [OrderIntent(symbol, "buy", qty)]
        if ret < -thr:
            if held > 0:
                return [OrderIntent(symbol, "sell", int(held))]
            if self.params.get("allow_short") and held == 0:
                return [OrderIntent(symbol, "sell", qty)]
        return []


class DeltaHedgeStrategy(Strategy):
    """Delta 中性对冲：持有期权头寸，用标的做动态对冲。

    期权价格用做市商的隐含波动率 ``iv``（合约条款里）定价，而对冲比例来自
    BS Delta。如果标的真实过程是 Heston/跳跃，对冲误差会真实地体现在权益曲线上——
    这正是「模型错配」在终端里的可视化。
    """

    key = "delta_hedge"
    name = "Delta 中性对冲"
    description = ("用标的多空头寸把期权敞口的 Delta 对冲回中性，按带宽再平衡；"
                   "可选自动卖空 1 份期权建立初始头寸。")
    default_params = {
        "option": "",          # 留空 = 第一个期权品种
        "qty": 1,              # 自动开仓的期权数量
        "band": 0.10,          # 相对目标仓位的再平衡带宽
        "open_short": False,   # 是否自动卖空期权建立头寸
    }

    def reset(self) -> None:
        self._opened = False

    def on_bar(self, ctx: StrategyContext) -> List[OrderIntent]:
        option_symbol = self.params.get("option") or next(
            (s for s, i in ctx.instruments.items() if i.is_option), "")
        if not option_symbol:
            return []
        inst = ctx.instrument(option_symbol)
        spec = inst.option
        if spec is None:
            return []

        intents: List[OrderIntent] = []
        qty = int(self.params.get("qty", 1))
        if self.params.get("open_short") and not self._opened \
                and ctx.position(option_symbol) == 0:
            intents.append(OrderIntent(option_symbol, "sell", qty))
            self._opened = True

        under = ctx.instrument(spec.underlying)
        position = ctx.position(option_symbol)
        if ctx.index >= spec.expiry_index or position == 0:
            target = 0.0
        else:
            tau = max((spec.expiry_index - ctx.index) * ctx.dt, 1e-8)
            spot = ctx.prices[spec.underlying]
            greeks = black_scholes(spot, spec.strike, tau, spec.rate,
                                   spec.iv, spec.kind, spec.dividend)
            target = -position * greeks["delta"] * (inst.multiplier / under.multiplier)

        held = ctx.position(spec.underlying)
        diff = target - held
        lot = under.lot_size
        if abs(diff) < max(lot, abs(target) * float(self.params.get("band", 0.10))):
            return intents

        qty = int(abs(diff) // lot) * lot
        if qty <= 0:
            return intents
        # 资金约束：不要把账户一次性打爆
        spot = ctx.prices[spec.underlying]
        affordable = int(ctx.cash / max(spot * under.multiplier, 1e-9) // lot) * lot
        if diff > 0:
            qty = min(qty, affordable)
        if qty <= 0:
            return intents
        intents.append(OrderIntent(spec.underlying, "buy" if diff > 0 else "sell", qty))
        return intents


STRATEGIES: Dict[str, type] = {
    "sma_cross": SmaCrossStrategy,
    "grid": GridStrategy,
    "momentum": MomentumStrategy,
    "delta_hedge": DeltaHedgeStrategy,
}


def create_strategy(key: str, **params: Any) -> Strategy:
    key = str(key)
    if key not in STRATEGIES:
        raise KeyError(f"未知策略 {key!r}，可用：{list(STRATEGIES)}")
    return STRATEGIES[key](**params)


def describe_strategies() -> List[Dict[str, Any]]:
    return [
        {"key": key, "name": cls.name, "description": cls.description,
         "defaults": dict(cls.default_params)}
        for key, cls in STRATEGIES.items()
    ]


__all__ = [
    "STRATEGIES",
    "DeltaHedgeStrategy",
    "GridStrategy",
    "MomentumStrategy",
    "OrderIntent",
    "SmaCrossStrategy",
    "Strategy",
    "StrategyContext",
    "create_strategy",
    "describe_strategies",
]