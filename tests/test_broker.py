"""账户记账测试：现金、费用、保证金、已实现盈亏、拒单回滚。"""
from __future__ import annotations

import pytest

from kairos_terminal.broker import Broker
from kairos_terminal.instruments import Instrument
from kairos_terminal.processes import create


def spot_broker(cash: float = 100_000.0) -> tuple:
    inst = Instrument(symbol="X", name="测试股", process=create("trend"), s0=100.0,
                      fee_rate=0.001, min_fee=5.0, stamp_tax=0.001)
    return Broker({"X": inst}, cash), inst


def futures_instrument() -> Instrument:
    return Instrument(symbol="F", name="测试期货", process=create("trend"), s0=100.0,
                      asset_class="futures", multiplier=10.0, margin_rate=0.1,
                      allow_short=True, fee_rate=0.0, tick_size=0.01)


def test_buy_then_sell_accounting():
    broker, _ = spot_broker()
    order = broker.submit("X", "buy", 100)
    fill = broker.execute(order, 100.0, 1)
    assert fill.qty == 100
    assert fill.fee == pytest.approx(10.0)                 # 10000 * 0.1%
    assert broker.cash == pytest.approx(100_000 - 10_000 - 10)

    pos = broker.positions["X"]
    assert pos.qty == 100 and pos.entry_price == 100.0
    assert pos.margin == pytest.approx(10_000.0)

    sell = broker.submit("X", "sell", 100)
    fill2 = broker.execute(sell, 110.0, 2)
    fee2 = 11.0 + 11_000 * 0.001                            # 佣金 + 印花税
    assert fill2.fee == pytest.approx(fee2)
    assert fill2.realized_pnl == pytest.approx(1_000.0)
    assert broker.cash == pytest.approx(100_000 - 10_010 + 11_000 - fee2)
    assert "X" not in broker.positions
    assert broker.equity() == pytest.approx(broker.cash)
    assert broker.realized_pnl == pytest.approx(1_000.0)


def test_min_fee_applies():
    broker, _ = spot_broker()
    order = broker.submit("X", "buy", 1, order_type="limit", limit_price=100)
    fill = broker.execute(order, 100.0, 1)
    assert fill.fee == pytest.approx(5.0)


def test_equity_identity_includes_margin_and_unrealized():
    broker, _ = spot_broker()
    broker.execute(broker.submit("X", "buy", 100), 100.0, 1)
    broker.mark({"X": 123.0})
    snap = broker.snapshot()
    expected = snap["cash"] + snap["margin_used"] + snap["unrealized_pnl"]
    assert snap["equity"] == pytest.approx(expected)
    assert snap["unrealized_pnl"] == pytest.approx(2_300.0)


def test_futures_margin_and_short_roundtrip():
    inst = futures_instrument()
    broker = Broker({"F": inst}, 10_000.0)

    opened = broker.submit("F", "sell", 2)                  # 做空 2 手
    broker.execute(opened, 100.0, 1)
    pos = broker.positions["F"]
    assert pos.qty == -2
    assert pos.margin == pytest.approx(2 * 100 * 10 * 0.1)  # 200
    assert broker.cash == pytest.approx(10_000 - 200)

    closed = broker.submit("F", "buy", 2)
    fill = broker.execute(closed, 90.0, 2)
    assert fill.realized_pnl == pytest.approx((90 - 100) * (-2) * 10)   # +200
    assert broker.cash == pytest.approx(10_000 - 200 + 200 + 200)
    assert "F" not in broker.positions


def test_flip_position_from_long_to_short():
    inst = futures_instrument()
    broker = Broker({"F": inst}, 100_000.0)
    broker.execute(broker.submit("F", "buy", 1), 100.0, 1)
    fill = broker.execute(broker.submit("F", "sell", 3), 110.0, 2)
    pos = broker.positions["F"]
    assert pos.qty == -2
    assert pos.entry_price == pytest.approx(110.0)
    assert fill.realized_pnl == pytest.approx(100.0)          # 平掉 1 手多头
    assert pos.margin == pytest.approx(2 * 110 * 10 * 0.1)    # 220


def test_insufficient_funds_rejects_and_rolls_back():
    broker, _ = spot_broker(cash=100.0)
    order = broker.submit("X", "buy", 100)                  # 需要 1 万，资金不足
    fill = broker.execute(order, 100.0, 1)
    assert fill.qty == 0
    assert order.status == "rejected"
    assert "资金不足" in order.note
    assert broker.cash == pytest.approx(100.0)
    assert "X" not in broker.positions
    assert broker.rejected_count == 1
    assert broker.realized_pnl == 0.0


def test_sell_without_position_is_blocked():
    broker, _ = spot_broker()
    with pytest.raises(ValueError, match="不支持做空"):
        broker.submit("X", "sell", 10)


def test_lot_size_and_bad_inputs():
    inst = Instrument(symbol="L", name="整手", process=create("trend"), lot_size=100)
    broker = Broker({"L": inst}, 1_000_000.0)
    with pytest.raises(ValueError, match="最小交易单位"):
        broker.submit("L", "buy", 50)
    with pytest.raises(ValueError, match="qty 必须为正"):
        broker.submit("L", "buy", 0)
    with pytest.raises(ValueError, match="side 必须是"):
        broker.submit("L", "hold", 100)
    with pytest.raises(ValueError, match="限价单"):
        broker.submit("L", "buy", 100, order_type="limit")
    with pytest.raises(ValueError, match="止损单"):
        broker.submit("L", "buy", 100, order_type="stop")
    with pytest.raises(ValueError, match="未知品种"):
        broker.submit("NOPE", "buy", 100)


def test_cancel_rules():
    broker, _ = spot_broker()
    order = broker.submit("X", "buy", 10)
    assert broker.pending_orders("X")[0].id == order.id
    broker.cancel(order.id)
    assert order.status == "cancelled"
    assert broker.pending_orders() == []
    with pytest.raises(ValueError, match="不能撤单"):
        broker.cancel(order.id)
    with pytest.raises(ValueError, match="不存在"):
        broker.cancel(9999)