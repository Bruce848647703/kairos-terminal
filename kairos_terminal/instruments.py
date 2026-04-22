"""交易品种定义：把「价格来源 + 交易规则」绑成一个可交易标的。

支持两类品种：

- **普通品种**（股票/ETF/期货/加密）：价格由可插拔的
  :class:`~kairos_terminal.processes.base.PriceProcess` 生成；
- **期权**：价格由标的行情 + Black-Scholes 推导（每根 bar 重估），
  因此可以做真实的 Delta 对冲实验。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .processes.base import PriceProcess

#: 支持的资产类别
ASSET_CLASSES = ("stock", "etf", "futures", "crypto", "option")

#: 资产类别 -> 中文名
ASSET_CLASS_LABELS = {
    "stock": "股票",
    "etf": "ETF",
    "futures": "期货",
    "crypto": "加密货币",
    "option": "期权",
}


def round_to_tick(price: float, tick: float) -> float:
    """把价格对齐到最小变动价位。"""
    if tick <= 0:
        return float(price)
    return round(round(float(price) / tick) * tick, 10)


@dataclass
class OptionSpec:
    """一张欧式期权的合约条款（现金交割）。"""

    underlying: str
    strike: float
    kind: str = "call"                 # call / put
    expiry_index: int = 252            # 第几根 bar 到期
    iv: float = 0.20                   # 定价用隐含波动率（做市商报价口径）
    rate: float = 0.02                 # 定价用无风险利率
    dividend: float = 0.0
    smile: bool = False               # True 时用 SSVI 微笑（需 kairos-vol）

    def __post_init__(self) -> None:
        self.underlying = str(self.underlying).strip().upper()
        if not self.underlying:
            raise ValueError("期权必须指定标的 underlying")
        self.kind = str(self.kind).strip().lower()
        if self.kind not in ("call", "put"):
            raise ValueError(f"期权 kind 必须是 call / put，收到 {self.kind!r}")
        if not float(self.strike) > 0:
            raise ValueError("行权价 strike 必须为正")
        if int(self.expiry_index) < 1:
            raise ValueError("expiry_index 必须 >= 1")
        if not 0.0 < float(self.iv) < 5.0:
            raise ValueError("隐含波动率 iv 必须在 (0, 5) 内")
        self.strike = float(self.strike)
        self.expiry_index = int(self.expiry_index)
        self.iv = float(self.iv)
        self.rate = float(self.rate)
        self.dividend = float(self.dividend)

    @property
    def kind_label(self) -> str:
        return "看涨" if self.kind == "call" else "看跌"

    def to_dict(self, *, tau: Optional[float] = None) -> Dict[str, Any]:
        out = {
            "underlying": self.underlying,
            "strike": self.strike,
            "kind": self.kind,
            "kind_label": self.kind_label,
            "expiry_index": self.expiry_index,
            "iv": self.iv,
            "rate": self.rate,
            "dividend": self.dividend,
            "smile": self.smile,
        }
        if tau is not None:
            out["tau"] = tau
        return out


@dataclass
class Instrument:
    """一个可交易品种。

    普通品种的价格走势完全由 ``process`` 决定；期权则由 ``option`` 指定的标的
    与条款推导。``asset_class`` 只影响交易规则（保证金、乘数、最小手数、印花税），
    与价格来源解耦，因此同一只股票可以随时换成 Heston 重新模拟。
    """

    symbol: str
    name: str
    process: Optional[PriceProcess] = None
    s0: float = 100.0
    asset_class: str = "stock"
    multiplier: float = 1.0
    lot_size: int = 1
    tick_size: float = 0.01
    margin_rate: float = 1.0
    allow_short: bool = False
    fee_rate: float = 0.0003
    min_fee: float = 0.0
    stamp_tax: float = 0.0
    precision: int = 2
    option: Optional[OptionSpec] = None

    def __post_init__(self) -> None:
        self.symbol = str(self.symbol).strip().upper()
        if not self.symbol:
            raise ValueError("symbol 不能为空")
        if self.asset_class not in ASSET_CLASSES:
            raise ValueError(f"asset_class 必须是 {ASSET_CLASSES}，收到 {self.asset_class!r}")

        if self.asset_class == "option":
            if self.option is None:
                raise ValueError("期权品种必须提供 option=OptionSpec(...)")
            self.option.underlying = self.option.underlying.upper()
            if self.option.underlying == self.symbol:
                raise ValueError("期权标的不能是它自己")
        else:
            if not isinstance(self.process, PriceProcess):
                raise TypeError("普通品种的 process 必须是 PriceProcess 实例")
            if self.option is not None:
                raise ValueError("只有 asset_class='option' 的品种才能设置 option")

        if not float(self.s0) > 0:
            raise ValueError("s0 必须为正")
        if int(self.lot_size) < 1:
            raise ValueError("lot_size 必须 >= 1")
        if not (0.0 < float(self.margin_rate) <= 1.0):
            raise ValueError("margin_rate 必须在 (0, 1] 内")
        self.s0 = float(self.s0)
        self.multiplier = float(self.multiplier)
        self.tick_size = float(self.tick_size)
        self.margin_rate = float(self.margin_rate)
        self.fee_rate = float(self.fee_rate)
        self.min_fee = float(self.min_fee)
        self.stamp_tax = float(self.stamp_tax)
        self.lot_size = int(self.lot_size)

    # ------------------------------------------------------------------ 展示
    @property
    def asset_class_label(self) -> str:
        return ASSET_CLASS_LABELS.get(self.asset_class, self.asset_class)

    @property
    def is_option(self) -> bool:
        return self.asset_class == "option"

    def to_dict(self, *, tau: Optional[float] = None,
                greeks: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "symbol": self.symbol,
            "name": self.name,
            "asset_class": self.asset_class,
            "asset_class_label": self.asset_class_label,
            "process": self.process.key if self.process else None,
            "process_name": self.process.name if self.process else "期权定价",
            "process_params": self.process.params_dict() if self.process else {},
            "s0": self.s0,
            "multiplier": self.multiplier,
            "lot_size": self.lot_size,
            "tick_size": self.tick_size,
            "margin_rate": self.margin_rate,
            "allow_short": self.allow_short,
            "fee_rate": self.fee_rate,
            "stamp_tax": self.stamp_tax,
            "precision": self.precision,
            "option": None,
        }
        if self.option is not None:
            out["option"] = self.option.to_dict(tau=tau)
            if greeks:
                out["option"]["greeks"] = greeks
        return out


__all__ = [
    "ASSET_CLASSES",
    "ASSET_CLASS_LABELS",
    "Instrument",
    "OptionSpec",
    "round_to_tick",
]