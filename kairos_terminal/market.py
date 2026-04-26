"""模拟行情引擎：路径 -> K 线 -> 逐 bar 撮合 -> 盯市。

撮合规则（**无未来函数**）
--------------------------
用户在「第 k 根 bar 收盘后」下的单（``created_index = k``）只会在**下一根 bar**
（索引 k）的 OHLC 里撮合，因此策略永远拿不到当根 bar 的收盘价去成交：

- 市价单：下一根 bar 开盘价成交（按 ``slippage_bps`` 加上不利滑点）；
- 限价买单：下一根 bar 最低价 ≤ 限价才成交，成交价 ``min(限价, 开盘价)``；
- 限价卖单：下一根 bar 最高价 ≥ 限价才成交，成交价 ``max(限价, 开盘价)``；
- 止损买单：最高价 ≥ 触发价后成交，价格 ``max(触发价, 开盘价)``；
- 止损卖单：最低价 ≤ 触发价后成交，价格 ``min(触发价, 开盘价)``。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .broker import Broker, Order
from .instruments import Instrument, round_to_tick
from .pricing import black_scholes
from .processes.base import DEFAULT_DT
from .volsurface import SmileSurface


@dataclass
class Bar:
    """一根 OHLCV K 线。"""

    index: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def change_pct(self) -> float:
        if self.open == 0:
            return 0.0
        return (self.close / self.open - 1.0) * 100.0

    def to_dict(self) -> Dict[str, float]:
        return {"t": self.index, "o": self.open, "h": self.high,
                "l": self.low, "c": self.close, "v": round(self.volume, 2)}


def build_bars(
    path: Sequence[float],
    *,
    tick_size: float = 0.01,
    seed: Optional[int] = None,
    wick: float = 0.6,
    base_volume: float = 1_000_000.0,
) -> List[Bar]:
    """把价格路径粗化成 OHLCV：开盘=上一期收盘，收盘=本期价格，影线按波动幅度缩放。"""
    p = np.asarray(path, dtype="float64")
    if p.size < 2:
        raise ValueError("价格路径至少需要 2 个点")
    if not np.all(np.isfinite(p)) or np.any(p <= 0):
        raise ValueError("价格路径存在非有限值或非正值")
    rng = np.random.default_rng(seed)
    n = p.size - 1
    log_ret = np.abs(np.diff(np.log(p)))
    ref = float(np.median(log_ret)) or 0.01

    bars: List[Bar] = []
    for i in range(n):
        o, c = float(p[i]), float(p[i + 1])
        amp = max(abs(c - o), tick_size)
        u1, u2 = rng.random(2)
        high = max(o, c) + amp * wick * u1
        low = min(o, c) - amp * wick * u2
        low = max(low, tick_size * 0.5)
        volume = base_volume * (0.4 + 2.5 * min(log_ret[i] / ref, 5.0)) \
            * float(np.exp(0.3 * rng.standard_normal()))

        o = round_to_tick(o, tick_size)
        c = round_to_tick(c, tick_size)
        high = round_to_tick(max(high, o, c), tick_size)
        low = round_to_tick(min(low, o, c), tick_size)
        low = max(low, tick_size)
        bars.append(Bar(index=i, open=o, high=max(high, o, c),
                        low=min(low, o, c), close=c, volume=max(volume, 1.0)))
    return bars


class MarketEngine:
    """多品种模拟行情 + 撮合 + 账户。"""

    def __init__(
        self,
        instruments: Sequence[Instrument],
        *,
        initial_cash: float = 1_000_000.0,
        n_steps: int = 500,
        seed: int = 42,
        dt: float = DEFAULT_DT,
        slippage_bps: float = 2.0,
        matching: str = "bar",
        book_levels: int = 12,
    ) -> None:
        if not instruments:
            raise ValueError("至少需要一个交易品种")
        self.instruments: Dict[str, Instrument] = {i.symbol: i for i in instruments}
        if len(self.instruments) != len(instruments):
            raise ValueError("品种代码不能重复")
        self.initial_cash = float(initial_cash)
        self.n_steps = int(n_steps)
        self.seed = int(seed)
        self.dt = float(dt)
        self.slippage_bps = float(slippage_bps)
        self.matching = str(matching).lower()
        if self.matching not in ("bar", "book"):
            raise ValueError("matching 必须是 bar / book")
        self.book_levels = int(book_levels)
        if self.book_levels < 1:
            raise ValueError("book_levels 必须 >= 1")
        self.strategy = None
        self.reset(seed=self.seed, n_steps=self.n_steps)

    # ------------------------------------------------------------------ 生命周期
    def reset(self, *, seed: Optional[int] = None, n_steps: Optional[int] = None) -> None:
        """重新生成全部路径并清空账户。"""
        if seed is not None:
            self.seed = int(seed)
        if n_steps is not None:
            if int(n_steps) < 10:
                raise ValueError("n_steps 至少为 10")
            self.n_steps = int(n_steps)

        rng = np.random.default_rng(self.seed)
        self._smiles = {
            symbol: SmileSurface(inst.option.iv)
            for symbol, inst in self.instruments.items()
            if inst.option is not None and inst.option.smile}
        self._paths: Dict[str, np.ndarray] = {}
        self._bars: Dict[str, List[Bar]] = {}

        # 1) 基础品种：各自用自己的价格过程生成路径
        for symbol, inst in self.instruments.items():
            if inst.is_option:
                continue
            path_seed = int(rng.integers(0, 2 ** 32 - 1))
            path = inst.process.simulate(
                self.n_steps, dt=self.dt, s0=inst.s0, seed=path_seed, n_paths=1)[0]
            self._paths[symbol] = np.asarray(path, dtype="float64")
            self._bars[symbol] = build_bars(
                self._paths[symbol], tick_size=inst.tick_size,
                seed=path_seed + 1, base_volume=1_000_000.0 * max(inst.multiplier, 1.0))

        # 2) 期权：由标的路径 + BS 逐 bar 重估（必须先有标的路径）
        for symbol, inst in self.instruments.items():
            if not inst.is_option:
                continue
            spec = inst.option
            if spec.underlying not in self._paths:
                raise ValueError(f"期权 {symbol} 的标的 {spec.underlying} 不存在")
            path = self._option_path(inst)
            inst.s0 = float(max(path[0], inst.tick_size))   # 首日理论价作为展示基准
            self._paths[symbol] = path
            self._bars[symbol] = build_bars(
                np.maximum(path, inst.tick_size), tick_size=inst.tick_size,
                seed=int(rng.integers(0, 2 ** 32 - 1)), base_volume=100_000.0)

        self.index = 0
        self.broker = Broker(self.instruments, self.initial_cash)
        self.logs: List[Dict[str, Any]] = []
        self._log("info", f"会话重置：seed={self.seed}，{self.n_steps} 根 bar，"
                          f"{len(self.instruments)} 个品种")

    @property
    def finished(self) -> bool:
        return self.index >= self.n_steps

    # ------------------------------------------------------------------ 推进
    def step(self, n: int = 1) -> int:
        """推进 ``n`` 根 bar，返回实际推进的根数。"""
        moved = 0
        for _ in range(max(int(n), 0)):
            if self.finished:
                break
            bar_index = self.index
            self.index += 1

            prices = self.prices()
            self._match(bar_index)
            self.broker.mark(prices)                 # 成交后按收盘价重新盯市
            self._settle_expired()
            self.broker.record_equity()

            if self.strategy is not None:
                self._run_strategy()

            moved += 1
        return moved

    def prices(self) -> Dict[str, float]:
        return {s: float(self._paths[s][self.index]) for s in self.instruments}

    def bar(self, symbol: str, index: Optional[int] = None) -> Optional[Bar]:
        symbol = symbol.upper()
        if symbol not in self._bars:
            raise ValueError(f"未知品种 {symbol!r}")
        idx = self.index if index is None else int(index)
        if not (0 <= idx < min(self.index, self.n_steps)):
            return None
        return self._bars[symbol][idx]

    def bars(self, symbol: str, start: int = 0, end: Optional[int] = None) -> List[Bar]:
        symbol = symbol.upper()
        if symbol not in self._bars:
            raise ValueError(f"未知品种 {symbol!r}")
        end = min(self.index if end is None else int(end), self.index)
        return self._bars[symbol][max(int(start), 0):end]

    def full_bars(self, symbol: str) -> List[Bar]:
        """全部 bar（含尚未发生的，**仅供研究/竞赛场景**，终端不会下发给前端）。"""
        return list(self._bars[symbol.upper()])

    # ------------------------------------------------------------------ 下单
    def submit(
        self,
        symbol: str,
        side: str,
        qty: int,
        *,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Order:
        order = self.broker.submit(
            symbol, side, qty, order_type=order_type,
            limit_price=limit_price, stop_price=stop_price, index=self.index)
        self._log("order", f"#{order.id} {order.symbol} "
                           f"{'买入' if side == 'buy' else '卖出'} {qty} "
                           f"{ {'market': '市价', 'limit': '限价', 'stop': '止损'}[order_type] }")
        return order

    def cancel(self, order_id: int) -> Order:
        order = self.broker.cancel(order_id)
        self._log("order", f"撤单 #{order.id} {order.symbol}")
        return order

    # ------------------------------------------------------------------ 撮合
    def _option_path(self, inst: Instrument) -> np.ndarray:
        """由标的路径 + 合约条款推导期权的逐 bar 理论价（到期后为内在价值）。"""
        spec = inst.option
        underlying = self._paths[spec.underlying]
        sign = 1.0 if spec.kind == "call" else -1.0
        expiry = min(spec.expiry_index, self.n_steps)
        out = np.empty(underlying.size, dtype="float64")
        for t in range(underlying.size):
            spot = float(underlying[t])
            tau = (expiry - t) * self.dt
            if tau <= 0.0:
                out[t] = max(sign * (spot - spec.strike), 0.0)
            else:
                iv = self._option_iv(inst.symbol, inst, spot, tau)
                out[t] = black_scholes(
                    spot, spec.strike, tau, spec.rate, iv,
                    spec.kind, spec.dividend)["price"]
        return np.maximum(out, 0.0)

    def _option_iv(self, symbol: str, inst: Instrument, spot: float,
                   maturity: float) -> float:
        """期权定价用的隐含波动率：有微笑时按 k 取值，否则用合约常数。"""
        surface = self._smiles.get(symbol)
        if surface is None:
            return float(inst.option.iv)
        return surface.iv_at_strike(spot, inst.option.strike, maturity)

    def _settle_expired(self) -> None:
        """到期 bar 收盘后按内在价值现金交割。"""
        for symbol, inst in self.instruments.items():
            spec = inst.option
            if spec is None or spec.expiry_index != self.index:
                continue
            price = float(self._paths[symbol][self.index])
            fill = self.broker.settle(symbol, price, self.index)
            if fill is not None:
                self._log("fill", f"行权结算 {symbol} {fill.qty}@{price:.4f} "
                                  f"已实现盈亏 {fill.realized_pnl:+.2f}")

    def _match(self, bar_index: int) -> None:
        for order in self.broker.pending_orders():
            if order.created_index > bar_index:          # 当根 bar 之后才下的单
                continue
            bar = self._bars[order.symbol][bar_index]
            price = self._fill_price(order, bar)
            if price is None:
                continue
            fill = self.broker.execute(order, price, self.index)
            if fill.qty > 0:
                self._log("fill", f"成交 #{order.id} {order.symbol} "
                                  f"{order.qty}@{fill.price:.2f} 费用 {fill.fee:.2f}"
                                  + (f" 已实现盈亏 {fill.realized_pnl:+.2f}"
                                     if abs(fill.realized_pnl) > 1e-9 else ""))

    def _fill_price(self, order: Order, bar: Bar) -> Optional[float]:
        slip = self.slippage_bps / 10_000.0
        buy = order.side == "buy"

        if order.type == "market":
            if self.matching == "book":
                return self._book_fill_price(order, bar)
            return bar.open * (1.0 + slip) if buy else bar.open * (1.0 - slip)

        if order.type == "limit":
            limit = float(order.limit_price)
            if buy and bar.low <= limit:
                return min(limit, bar.open)
            if (not buy) and bar.high >= limit:
                return max(limit, bar.open)
            return None

        stop = float(order.stop_price)                   # stop
        if buy and bar.high >= stop:
            return max(stop, bar.open) * (1.0 + slip)
        if (not buy) and bar.low <= stop:
            return min(stop, bar.open) * (1.0 - slip)
        return None

    def _book_fill_price(self, order: Order, bar: Bar) -> float:
        """在合成订单簿上逐档吃单，返回加权成交均价。

        每档深度取 ``max(bar.volume·0.1%, qty/(levels·0.6))``（大成交量品种更深，
        同时保证订单能在簿内成交完）；档间距取 ``max(tick, bar振幅/(2·levels))``
        —— 只用 1 个 tick 会严重低估冲击，因为真实盘口是有跳档的。
        """
        inst = self.instruments[order.symbol]
        tick = max(inst.tick_size, 1e-9)
        levels = self.book_levels
        bar_range = max(float(bar.high) - float(bar.low), tick)
        step = max(tick, bar_range / (2.0 * levels))
        level_size = max(bar.volume * 0.001, order.qty / (levels * 0.6), 1.0)
        sign = 1.0 if order.side == "buy" else -1.0
        remaining = float(order.qty)
        notional = 0.0
        for i in range(1, levels + 1):
            price = bar.open + sign * (i - 0.5) * step
            take = min(level_size, remaining)
            notional += take * price
            remaining -= take
            if remaining <= 1e-9:
                break
        if remaining > 1e-9:                      # 极端情况：按最后一档补足
            notional += remaining * (bar.open + sign * levels * step)
        return notional / float(order.qty)

    # ------------------------------------------------------------------ 策略
    def attach_strategy(self, strategy: Any) -> None:
        self.strategy = strategy
        if strategy is not None:
            strategy.reset()
            self._log("strategy", f"已挂载策略：{getattr(strategy, 'name', type(strategy).__name__)}")
        else:
            self._log("strategy", "已卸载策略")

    def _run_strategy(self) -> None:
        from .strategy import StrategyContext

        ctx = StrategyContext(self)
        try:
            intents = self.strategy.on_bar(ctx) or []
        except Exception as exc:                     # 策略异常不应打断行情推进
            self._log("error", f"策略执行异常：{exc}")
            return
        for intent in intents:
            try:
                self.submit(
                    intent.symbol, intent.side, intent.qty,
                    order_type=intent.order_type,
                    limit_price=intent.limit_price, stop_price=intent.stop_price)
            except ValueError as exc:
                self._log("error", f"策略下单失败：{exc}")

    # ------------------------------------------------------------------ 日志
    def _log(self, kind: str, message: str) -> None:
        self.logs.append({"index": self.index, "kind": kind, "message": message})
        if len(self.logs) > 500:
            del self.logs[: len(self.logs) - 500]

    def log_dicts(self, limit: int = 80) -> List[Dict[str, Any]]:
        return list(reversed(self.logs[-limit:]))

    # ------------------------------------------------------------------ 快照
    def snapshot(self, *, active_symbol: Optional[str] = None,
                 bars: int = 200, order_limit: int = 30) -> Dict[str, Any]:
        """终端状态快照（可直接 JSON 序列化）。"""
        prices = self.prices()
        account = self.broker.snapshot()

        instruments = []
        for symbol, inst in self.instruments.items():
            price = prices[symbol]
            prev = float(self._paths[symbol][self.index - 1]) if self.index > 0 else inst.s0
            pos = self.broker.positions.get(symbol)
            tau = 0.0
            greeks = None
            if inst.option is not None:
                spec = inst.option
                tau = max(0.0, (spec.expiry_index - self.index) * self.dt)
                if tau > 0:
                    spot = prices.get(spec.underlying, inst.s0)
                    effective_iv = self._option_iv(symbol, inst, spot, tau)
                    g = black_scholes(
                        spot, spec.strike, tau,
                        spec.rate, effective_iv, spec.kind, spec.dividend)
                    greeks = {k: g[k] for k in ("delta", "gamma", "vega", "theta", "rho")}
            instruments.append({
                **inst.to_dict(tau=tau if inst.option else None, greeks=greeks),
                "price": price,
                "prev_close": prev,
                "change_pct": (price / prev - 1.0) * 100.0 if prev else 0.0,
                "position_qty": pos.qty if pos else 0.0,
                "position_pnl": pos.unrealized_pnl(inst.multiplier) if pos else 0.0,
            })

        active = (active_symbol or next(iter(self.instruments))).upper()
        if active not in self.instruments:
            active = next(iter(self.instruments))
        window = self.bars(active, max(0, self.index - max(int(bars), 10)))

        return {
            "seed": self.seed,
            "index": self.index,
            "n_steps": self.n_steps,
            "finished": self.finished,
            "active_symbol": active,
            "instruments": instruments,
            "account": account,
            "bars": [b.to_dict() for b in window],
            "bar_window_start": max(0, self.index - max(int(bars), 10)),
            "orders": self.broker.order_dicts(order_limit),
            "fills": self.broker.fill_dicts(order_limit),
            "logs": self.log_dicts(),
            "strategy": {
                "key": getattr(self.strategy, "key", None),
                "name": getattr(self.strategy, "name", None),
                "params": getattr(self.strategy, "params", {}),
            } if self.strategy is not None else None,
            "slippage_bps": self.slippage_bps,
            "matching": self.matching,
        }


__all__ = ["Bar", "MarketEngine", "build_bars"]