"""会话层测试：品种管理、策略挂载、定价分发。"""
from __future__ import annotations

import pytest

from kairos_terminal.session import Session


def test_default_instruments_cover_multiple_processes():
    session = Session(seed=1, n_steps=60)
    keys = {i.process.key for i in session.instruments.values() if i.process is not None}
    assert len(session.instruments) >= 6
    assert {"gbm", "heston", "regime", "merton", "vg"} <= keys
    assert any(i.is_option for i in session.instruments.values())


def test_snapshot_exposes_meta_and_performance():
    session = Session(seed=1, n_steps=60)
    session.step(10)
    snap = session.snapshot()
    assert snap["index"] == 10
    assert "performance" in snap and "sharpe" in snap["performance"]
    assert snap["processes"] and snap["strategies"]


def test_set_process_rebuilds_session():
    session = Session(seed=3, n_steps=60)
    session.step(5)
    out = session.set_process("510300", "heston", {"v0": 0.09, "theta": 0.09})
    assert out["process"] == "heston"
    assert session.engine.index == 0
    assert session.instruments["510300"].process["v0"] == pytest.approx(0.09)
    assert session.instruments["510300"].s0 == pytest.approx(3.80)


def test_set_process_unknown_symbol_or_process():
    session = Session(seed=1, n_steps=60)
    with pytest.raises(ValueError, match="未知品种"):
        session.set_process("NOPE", "gbm")
    with pytest.raises(KeyError):
        session.set_process("510300", "not-a-process")


def test_add_and_remove_instrument():
    session = Session(seed=1, n_steps=60)
    meta = session.add_instrument("000001", "平安银行", "gbm",
                                  params={"sigma": 0.3}, s0=12.5,
                                  asset_class="stock")
    assert meta["symbol"] == "000001"
    assert session.engine.index == 0
    with pytest.raises(ValueError, match="已存在"):
        session.add_instrument("000001", "重复", "gbm")
    session.remove_instrument("000001")
    assert "000001" not in session.instruments
    with pytest.raises(ValueError, match="未知品种"):
        session.remove_instrument("000001")


def test_cannot_remove_last_instrument():
    session = Session(seed=1, n_steps=60)
    for symbol in list(session.instruments)[1:]:
        session.remove_instrument(symbol)
    with pytest.raises(ValueError, match="至少保留一个"):
        session.remove_instrument(list(session.instruments)[0])


def test_strategy_lifecycle():
    session = Session(seed=1, n_steps=60)
    out = session.set_strategy("sma_cross", {"fast": 3, "slow": 6, "qty": 10})
    assert out["strategy"] == "sma_cross"
    assert session.engine.strategy is not None
    session.step(30)
    session.set_strategy(None)
    assert session.engine.strategy is None
    with pytest.raises(KeyError):
        session.set_strategy("no-such-strategy")


def test_price_dispatch_all_methods():
    session = Session(seed=1, n_steps=60)
    base = {"symbol": "510300", "strike": 4.0, "maturity": 0.5, "vol": 0.2, "rate": 0.02}

    bs = session.price({**base, "method": "bs"})
    assert bs["price"] > 0 and "delta" in bs

    tree = session.price({**base, "method": "binomial", "steps": 100})
    assert tree["price"] > 0

    mc = session.price({**base, "method": "mc", "n_paths": 2000, "n_steps": 64, "seed": 1})
    assert mc["price"] > 0 and mc["stderr"] > 0

    iv = session.price({**base, "method": "implied_vol", "price": bs["price"]})
    assert iv["implied_vol"] == pytest.approx(0.2, abs=1e-4)

    hedge = session.price({**base, "method": "hedge", "n_paths": 300, "n_steps": 26, "seed": 1})
    assert "mean_pnl" in hedge

    cmp = session.price({**base, "method": "compare", "n_paths": 2000, "n_steps": 64, "seed": 1})
    assert "black_scholes" in cmp and "monte_carlo" in cmp


def test_price_rejects_unknown_method():
    session = Session(seed=1, n_steps=60)
    with pytest.raises(ValueError, match="method 必须是"):
        session.price({"method": "nope"})


def test_reset_changes_seed_and_clears_account():
    session = Session(seed=1, n_steps=60)
    session.step(20)
    session.submit("510300", "buy", 100)
    session.step(2)
    session.reset(seed=99, n_steps=80)
    assert session.engine.index == 0
    assert session.engine.seed == 99
    assert session.engine.n_steps == 80
    assert session.engine.broker.fills == []