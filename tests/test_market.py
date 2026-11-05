"""行情引擎测试：K 线构造、无未来函数撮合、各类委托的成交价。"""
from __future__ import annotations

import pytest

from kairos_terminal.instruments import Instrument
from kairos_terminal.market import MarketEngine, build_bars
from kairos_terminal.processes import create


def make_engine(symbol: str = "X", **kwargs) -> MarketEngine:
    inst = Instrument(symbol=symbol, name="测试", process=create("gbm", mu=0.05, sigma=0.25),
                      s0=100.0, fee_rate=0.0, tick_size=0.01)
    return MarketEngine([inst], n_steps=80, seed=5, slippage_bps=0.0, **kwargs)


def test_build_bars_invariants():
    path = create("gbm").simulate_path(200, seed=3)
    bars = build_bars(path, tick_size=0.01, seed=7)
    assert len(bars) == 200
    for bar in bars:
        assert bar.high >= max(bar.open, bar.close) - 1e-9
        assert bar.low <= min(bar.open, bar.close) + 1e-9
        assert bar.low > 0
        assert bar.volume > 0
        assert bar.index >= 0


def test_build_bars_deterministic():
    path = create("gbm").simulate_path(30, seed=1)
    a = [b.to_dict() for b in build_bars(path, seed=9)]
    b = [b.to_dict() for b in build_bars(path, seed=9)]
    c = [b.to_dict() for b in build_bars(path, seed=10)]
    assert a == b
    assert a != c


def test_reset_is_deterministic_and_bars_hidden_before_time():
    e1, e2 = make_engine(), make_engine()
    assert e1.prices() == pytest.approx(e2.prices())
    assert e1.bars("X") == []                       # 还没推进，不泄露任何 bar
    e1.step(3)
    assert len(e1.bars("X")) == 3
    assert len(e1.full_bars("X")) == 80             # 研究接口能看到全部


def test_no_lookahead_order_waits_one_bar():
    engine = make_engine()
    engine.step(5)
    order = engine.submit("X", "buy", 10)
    assert order.status == "pending"
    assert engine.broker.fills == []
    engine.step(1)                                   # 第 5 根 bar（0-based）成交
    assert order.status == "filled"
    fill = engine.broker.fills[0]
    assert fill.index == 6
    assert fill.price == pytest.approx(engine.full_bars("X")[5].open)


def test_market_order_uses_next_open_with_slippage():
    inst = Instrument(symbol="X", name="t", process=create("gbm"), s0=100.0, fee_rate=0.0)
    engine = MarketEngine([inst], n_steps=30, seed=2, slippage_bps=10.0)
    engine.step(4)
    engine.submit("X", "buy", 1)
    engine.step(1)
    expected = engine.full_bars("X")[4].open * (1 + 10.0 / 10_000)
    assert engine.broker.fills[0].price == pytest.approx(expected, abs=0.02)


def test_limit_and_stop_fill_prices():
    inst = Instrument(symbol="X", name="t", process=create("gbm"), s0=100.0, fee_rate=0.0)
    engine = MarketEngine([inst], n_steps=30, seed=4, slippage_bps=0.0)
    engine.step(6)
    bar = engine.full_bars("X")[6]

    limit = engine.submit("X", "buy", 1, order_type="limit", limit_price=bar.high)
    engine.step(1)
    assert limit.status == "filled"
    assert engine.broker.fills[0].price == pytest.approx(min(bar.high, bar.open))

    engine.step(4)
    bar2 = engine.full_bars("X")[11]
    stop = engine.submit("X", "sell", 1, order_type="stop", stop_price=bar2.low)
    engine.step(1)
    assert stop.status == "filled"
    assert engine.broker.fills[1].price == pytest.approx(min(bar2.low, bar2.open))


def test_untriggered_limit_stays_pending_then_cancels():
    engine = make_engine()
    engine.step(5)
    price = engine.prices()["X"]
    order = engine.submit("X", "buy", 10, order_type="limit", limit_price=price * 0.1)
    engine.step(10)
    assert order.status == "pending"
    engine.cancel(order.id)
    assert order.status == "cancelled"


def test_step_clamps_at_end():
    engine = make_engine()
    assert engine.step(1000) == 80
    assert engine.finished
    assert engine.step(5) == 0
    snap = engine.snapshot()
    assert snap["finished"] is True


def test_snapshot_contract():
    engine = make_engine()
    engine.step(10)
    snap = engine.snapshot(bars=5)
    for key in ("seed", "index", "n_steps", "instruments", "account", "bars",
                "orders", "fills", "logs", "active_symbol"):
        assert key in snap
    assert len(snap["bars"]) <= 10
    assert snap["index"] == 10
    assert snap["account"]["equity"] == pytest.approx(engine.broker.equity())


def test_duplicate_symbols_rejected():
    inst = Instrument(symbol="X", name="t", process=create("gbm"))
    with pytest.raises(ValueError, match="品种代码不能重复"):
        MarketEngine([inst, inst])


def test_strategy_exception_does_not_break_stepping():
    from kairos_terminal.strategy import Strategy

    class Broken(Strategy):
        name = "broken"

        def on_bar(self, ctx):
            raise RuntimeError("boom")

    engine = make_engine()
    engine.attach_strategy(Broken())
    assert engine.step(3) == 3
    assert engine.index == 3
    assert any("策略执行异常" in log["message"] for log in engine.logs)


def test_strategy_runs_and_trades():
    from kairos_terminal.strategy import create_strategy

    engine = MarketEngine([
        Instrument(symbol="X", name="t", process=create("gbm", mu=0.1, sigma=0.6),
                   s0=100.0, fee_rate=0.0),
    ], n_steps=120, seed=1)
    engine.attach_strategy(create_strategy("sma_cross", fast=2, slow=4, qty=10))
    engine.step(120)
    assert len(engine.broker.orders) > 0

# --------------------------------------------------------------------------- #
# 订单簿撮合模式
# --------------------------------------------------------------------------- #
def make_book_engine(matching: str = "book", n_steps: int = 60) -> MarketEngine:
    inst = Instrument(symbol="X", name="测试", process=create("gbm", mu=0.05, sigma=0.25),
                      s0=100.0, fee_rate=0.0, tick_size=0.01)
    return MarketEngine([inst], n_steps=n_steps, seed=5, matching=matching)


def test_book_matching_impact_increases_with_size():
    prices = []
    for qty in (1000, 3000, 6000):
        engine = make_book_engine()
        engine.step(5)
        engine.submit("X", "buy", qty)
        engine.step(1)
        prices.append(engine.broker.fills[-1].price)
    assert prices[0] < prices[1] < prices[2]


def test_book_matching_is_worse_than_bar_for_buys():
    bar = make_book_engine("bar")
    book = make_book_engine("book")
    for engine in (bar, book):
        engine.step(5)
        engine.submit("X", "buy", 8000)
        engine.step(1)
    assert book.broker.fills[-1].price > bar.broker.fills[-1].price


def test_book_matching_sell_impact_is_downwards():
    small, large = make_book_engine(), make_book_engine()
    for engine, qty in ((small, 100), (large, 5000)):
        engine.step(5)
        engine.submit("X", "buy", 6000)          # 先建仓
        engine.step(1)
        engine.submit("X", "sell", qty)
        engine.step(1)
    assert large.broker.fills[-1].price < small.broker.fills[-1].price


def test_book_matching_small_order_near_open():
    engine = make_book_engine()
    engine.step(5)
    engine.submit("X", "buy", 100)
    engine.step(1)
    bar = engine.full_bars("X")[5]
    assert engine.broker.fills[-1].price - bar.open < 0.05 * bar.open


def test_invalid_matching_mode():
    inst = Instrument(symbol="X", name="t", process=create("gbm"))
    with pytest.raises(ValueError, match="matching"):
        MarketEngine([inst], n_steps=10, matching="nope")
    with pytest.raises(ValueError, match="book_levels"):
        MarketEngine([inst], n_steps=10, matching="book", book_levels=0)
