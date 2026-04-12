"""账户与撮合记账：委托、成交、持仓、保证金与盈亏。

记账口径（对股票与期货统一）
---------------------------
每个持仓占用一笔**保证金** ``margin = |qty| · entry_price · multiplier · margin_rate``：

- 股票 ``margin_rate = 1``：等于买入成本，等价于「现金换股票」；
- 期货 ``margin_rate < 1``：只冻结保证金，杠杆部分靠浮动盈亏体现。

因此恒等式对两种资产都成立::

    权益 = 现金 + Σ持仓保证金 + Σ浮动盈亏

平仓时按**开仓价**释放保证金，同时把 ``(平仓价 - 开仓价) · qty · multiplier``
计入现金，于是「买入→卖出」与「卖空→买回」都能自动得到正确的已实现盈亏。
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .instruments import Instrument, round_to_tick

ORDER_TYPES = ("market", "limit", "stop")
ORDER_STATUSES = ("pending", "filled", "cancelled", "rejected")


@dataclass
class Order:
    """一笔委托。"""

    id: int
    symbol: str
    side: str                       # buy / sell
    qty: int
    type: str = "market"            # market / limit / stop
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: str = "pending"         # pending / filled / cancelled / rejected
    filled_qty: int = 0
    avg_price: float = 0.0
    created_index: int = 0          # 下单时已完成的 bar 数
    filled_index: Optional[int] = None
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side,
            "side_label": "买入" if self.side == "buy" else "卖出",
            "qty": self.qty,
            "type": self.type,
            "type_label": {"market": "市价", "limit": "限价", "stop": "止损"}[self.type],
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "status": self.status,
            "status_label": {
                "pending": "未成交", "filled": "已成交",
                "cancelled": "已撤单", "rejected": "已拒绝",
            }[self.status],
            "filled_qty": self.filled_qty,
            "avg_price": self.avg_price,
            "created_index": self.created_index,
            "filled_index": self.filled_index,
            "note": self.note,
        }


@dataclass
class Fill:
    """一笔成交回报。"""

    order_id: int
    symbol: str
    side: str
    qty: int
    price: float
    fee: float
    index: int
    realized_pnl: float = 0.0

    @property
    def notional(self) -> float:
        return self.qty * self.price

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.qty,
            "price": self.price,
            "fee": self.fee,
            "index": self.index,
            "realized_pnl": self.realized_pnl,
        }


@dataclass
class Position:
    """一个标的的持仓（支持多空）。"""

    symbol: str
    qty: float = 0.0
    entry_price: float = 0.0
    margin: float = 0.0
    realized_pnl: float = 0.0
    last_price: float = 0.0

    @property
    def is_flat(self) -> bool:
        return abs(self.qty) < 1e-12

    def unrealized_pnl(self, multiplier: float = 1.0) -> float:
        return (self.last_price - self.entry_price) * self.qty * multiplier

    def to_dict(self, instrument: Instrument) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "qty": self.qty,
            "entry_price": self.entry_price,
            "last_price": self.last_price,
            "margin": self.margin,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl(instrument.multiplier),
            "market_value": self.qty * self.last_price * instrument.multiplier,
            "direction": "多头" if self.qty > 0 else ("空头" if self.qty < 0 else "空仓"),
        }


class Broker:
    """账户 + 委托簿 + 成交簿。"""

    def __init__(self, instruments: Dict[str, Instrument], initial_cash: float = 1_000_000.0):
        if not instruments:
            raise ValueError("Broker 至少需要一个品种")
        self.instruments = instruments
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[int, Order] = {}
        self.fills: List[Fill] = []
        self.equity_curve: List[float] = [float(initial_cash)]
        self.realized_pnl: float = 0.0
        self.rejected_count = 0
        self._next_id = 1
        self._last_prices: Dict[str, float] = {s: i.s0 for s, i in instruments.items()}

    # ------------------------------------------------------------------ 委托
    def submit(
        self,
        symbol: str,
        side: str,
        qty: int,
        *,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        index: int = 0,
    ) -> Order:
        """提交委托（此时不成交，等下一根 bar 撮合）。"""
        symbol = str(symbol).strip().upper()
        if symbol not in self.instruments:
            raise ValueError(f"未知品种 {symbol!r}")
        inst = self.instruments[symbol]
        side = str(side).strip().lower()
        if side not in ("buy", "sell"):
            raise ValueError(f"side 必须是 buy / sell，收到 {side!r}")
        if inst.option is not None and int(index) >= inst.option.expiry_index:
            raise ValueError(
                f"{symbol} 已于第 {inst.option.expiry_index} 根 bar 到期，不能再交易")
        if order_type not in ORDER_TYPES:
            raise ValueError(f"type 必须是 {ORDER_TYPES}，收到 {order_type!r}")
        qty = int(qty)
        if qty <= 0:
            raise ValueError("qty 必须为正整数")
        if qty % inst.lot_size != 0:
            raise ValueError(f"{symbol} 的最小交易单位是 {inst.lot_size}，收到 {qty}")

        if order_type == "limit":
            if limit_price is None or not float(limit_price) > 0:
                raise ValueError("限价单必须给出正的 limit_price")
            limit_price = round_to_tick(float(limit_price), inst.tick_size)
        elif order_type == "stop":
            if stop_price is None or not float(stop_price) > 0:
                raise ValueError("止损单必须给出正的 stop_price")
            stop_price = round_to_tick(float(stop_price), inst.tick_size)
        else:
            limit_price = stop_price = None

        position = self.positions.get(symbol)
        held = position.qty if position else 0.0
        if side == "sell" and not inst.allow_short:
            if held < qty - 1e-9:
                raise ValueError(
                    f"{symbol} 不支持做空（持仓 {held:g}，卖出 {qty}）；"
                    "请先买入或改用 allow_short 的品种")

        order = Order(
            id=self._next_id, symbol=symbol, side=side, qty=qty,
            type=order_type, limit_price=limit_price, stop_price=stop_price,
            created_index=int(index),
        )
        self._next_id += 1
        self.orders[order.id] = order
        return order

    def cancel(self, order_id: int) -> Order:
        order = self.orders.get(int(order_id))
        if order is None:
            raise ValueError(f"委托 {order_id} 不存在")
        if order.status != "pending":
            raise ValueError(f"委托 {order_id} 当前状态为 {order.status}，不能撤单")
        order.status = "cancelled"
        order.note = "用户撤单"
        return order

    def pending_orders(self, symbol: Optional[str] = None) -> List[Order]:
        out = [o for o in self.orders.values() if o.status == "pending"]
        if symbol is not None:
            out = [o for o in out if o.symbol == symbol]
        return sorted(out, key=lambda o: o.id)

    # ------------------------------------------------------------------ 成交
    def execute(self, order: Order, price: float, index: int) -> Fill:
        """按给定价格全额成交一笔委托（由撮合引擎调用）。"""
        inst = self.instruments[order.symbol]
        price = round_to_tick(float(price), inst.tick_size)
        signed = order.qty if order.side == "buy" else -order.qty
        notional = order.qty * price * inst.multiplier
        fee = max(inst.min_fee, notional * inst.fee_rate)
        if order.side == "sell":
            fee += notional * inst.stamp_tax

        position = self.positions.get(order.symbol) or Position(
            symbol=order.symbol, last_price=price)
        backup = copy.copy(position)                     # 资金不足时回滚
        old_margin = position.margin
        realized = self._apply_position(position, signed, price, inst)
        delta_margin = position.margin - old_margin
        required = delta_margin - realized + fee          # 平仓时 realized 可抵扣

        if required > self.cash + 1e-9:                 # 资金不足 -> 拒单
            if order.symbol in self.positions:
                self.positions[order.symbol] = backup
            order.status = "rejected"
            order.note = f"资金不足：需要 {required:,.2f}，可用 {self.cash:,.2f}"
            self.rejected_count += 1
            return Fill(order_id=order.id, symbol=order.symbol, side=order.side,
                        qty=0, price=price, fee=0.0, index=index)

        self.cash -= required
        self.realized_pnl += realized
        position.last_price = price
        if position.is_flat and abs(position.margin) < 1e-9:
            self.positions.pop(order.symbol, None)       # 平仓后清理空持仓
        else:
            self.positions[order.symbol] = position

        order.status = "filled"
        order.filled_qty = order.qty
        order.avg_price = price
        order.filled_index = index

        fill = Fill(order_id=order.id, symbol=order.symbol, side=order.side,
                    qty=order.qty, price=price, fee=fee, index=index,
                    realized_pnl=realized)
        self.fills.append(fill)
        self._last_prices[order.symbol] = price
        return fill

    @staticmethod
    def _apply_position(position: Position, signed_qty: float, price: float,
                        inst: Instrument) -> float:
        """更新持仓并返回本次已实现盈亏。"""
        old_qty = position.qty
        new_qty = old_qty + signed_qty
        realized = 0.0

        same_direction = old_qty == 0 or (old_qty > 0) == (signed_qty > 0)
        if same_direction:
            add = abs(signed_qty)
            total = abs(old_qty) + add
            position.entry_price = (
                (abs(old_qty) * position.entry_price + add * price) / total
                if old_qty != 0 else price)
            position.margin += add * price * inst.multiplier * inst.margin_rate
        else:
            closing = min(abs(signed_qty), abs(old_qty))
            direction = 1.0 if old_qty > 0 else -1.0
            realized = (price - position.entry_price) * closing * direction * inst.multiplier
            position.realized_pnl += realized
            position.margin -= closing * position.entry_price * inst.multiplier * inst.margin_rate

            if abs(signed_qty) > abs(old_qty):           # 反手开仓
                excess = abs(signed_qty) - abs(old_qty)
                position.entry_price = price
                position.margin += excess * price * inst.multiplier * inst.margin_rate
            elif abs(new_qty) < 1e-12:
                position.entry_price = 0.0
                position.margin = 0.0

        position.qty = 0.0 if abs(new_qty) < 1e-12 else new_qty
        position.last_price = price
        return realized

    # ------------------------------------------------------------------ 估值
    def settle(self, symbol: str, price: float, index: int) -> Optional[Fill]:
        """到期现金交割：按给定价格强制平掉全部持仓（不计手续费）。"""
        symbol = str(symbol).upper()
        position = self.positions.get(symbol)
        if position is None or abs(position.qty) < 1e-12:
            return None
        inst = self.instruments[symbol]
        price = round_to_tick(float(price), inst.tick_size)
        qty = position.qty
        old_margin = position.margin
        realized = self._apply_position(position, -qty, price, inst)
        delta_margin = position.margin - old_margin
        self.cash += realized - delta_margin
        self.realized_pnl += realized
        self.positions.pop(symbol, None)
        fill = Fill(order_id=0, symbol=symbol,
                    side="sell" if qty > 0 else "buy", qty=int(abs(qty)),
                    price=price, fee=0.0, index=index, realized_pnl=realized)
        self.fills.append(fill)
        self._last_prices[symbol] = price
        return fill

    def mark(self, prices: Dict[str, float]) -> None:
        """按最新价逐日盯市。"""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].last_price = float(price)
            self._last_prices[symbol] = float(price)

    def equity(self) -> float:
        total = self.cash
        for symbol, pos in self.positions.items():
            total += pos.margin + pos.unrealized_pnl(self.instruments[symbol].multiplier)
        return total

    def record_equity(self) -> float:
        value = self.equity()
        self.equity_curve.append(value)
        return value

    def snapshot(self) -> Dict[str, Any]:
        equity = self.equity()
        realized = self.realized_pnl
        unrealized = sum(p.unrealized_pnl(self.instruments[s].multiplier)
                         for s, p in self.positions.items())
        return {
            "initial_cash": self.initial_cash,
            "cash": self.cash,
            "margin_used": sum(p.margin for p in self.positions.values()),
            "equity": equity,
            "realized_pnl": realized,
            "unrealized_pnl": unrealized,
            "total_pnl": equity - self.initial_cash,
            "return_pct": (equity / self.initial_cash - 1.0) * 100.0,
            "positions": [p.to_dict(self.instruments[s]) for s, p in self.positions.items()],
            "equity_curve": list(self.equity_curve),
            "rejected_count": self.rejected_count,
        }

    def order_dicts(self, limit: int = 50) -> List[Dict[str, Any]]:
        orders = sorted(self.orders.values(), key=lambda o: o.id, reverse=True)[:limit]
        return [o.to_dict() for o in orders]

    def fill_dicts(self, limit: int = 50) -> List[Dict[str, Any]]:
        return [f.to_dict() for f in self.fills[-limit:][::-1]]


__all__ = ["ORDER_STATUSES", "ORDER_TYPES", "Broker", "Fill", "Order", "Position"]