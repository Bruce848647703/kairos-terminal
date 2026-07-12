"""期权品种测试：BS 推导价格、现金交割、Greeks、Delta 对冲策略。"""
from __future__ import annotations

import pytest

from kairos_terminal.instruments import Instrument, OptionSpec
from kairos_terminal.market import MarketEngine
from kairos_terminal.pricing import black_scholes
from kairos_terminal.processes import create
from kairos_terminal.session import Session
from kairos_terminal.strategy import create_strategy


def make_engine(n_steps: int = 80, expiry: int = 20, kind: str = "call",
                iv: float = 0.2) -> MarketEngine:
    underlying = Instrument(symbol="U", name="标的", process=create("gbm", mu=0.05, sigma=0.2),
                            s0=100.0, fee_rate=0.0, tick_size=0.01)
    option = Instrument(
        symbol="U-C100" if kind == "call" else "U-P100", name="期权",
        asset_class="option",
        option=OptionSpec(underlying="U", strike=100.0, kind=kind,
                          expiry_index=expiry, iv=iv, rate=0.02),
        multiplier=100.0, tick_size=0.0001, precision=4, fee_rate=0.0, allow_short=True)
    return MarketEngine([underlying, option], n_steps=n_steps, seed=3)


def test_option_price_matches_black_scholes_at_start():
    engine = make_engine(expiry=20)
    expected = black_scholes(100.0, 100.0, 20 * engine.dt, 0.02, 0.2)["price"]
    assert engine.prices()["U-C100"] == pytest.approx(expected, rel=1e-9)


def test_option_price_tracks_underlying():
    engine = make_engine(expiry=30)
    engine.step(10)
    spot = engine.prices()["U"]
    expected = black_scholes(spot, 100.0, 20 * engine.dt, 0.02, 0.2)["price"]
    assert engine.prices()["U-C100"] == pytest.approx(expected, rel=1e-9)


def test_put_option_delta_is_negative_and_bounded():
    engine = make_engine(kind="put")
    snap = engine.snapshot()
    opt = next(i for i in snap["instruments"] if i["symbol"] == "U-P100")
    assert -1.0 < opt["option"]["greeks"]["delta"] < 0.0
    assert opt["option"]["tau"] > 0


def test_buy_option_cash_flow_and_position():
    engine = make_engine()
    engine.step(2)
    cash0 = engine.broker.cash
    engine.submit("U-C100", "buy", 1)
    engine.step(1)

    fill = engine.broker.fills[-1]
    assert fill.qty == 1
    assert engine.broker.cash == pytest.approx(cash0 - fill.price * 100.0)
    pos = engine.broker.positions["U-C100"]
    assert pos.qty == 1
    assert pos.entry_price == pytest.approx(fill.price)
    equity = engine.broker.equity()
    expected = engine.broker.cash + pos.margin + (engine.prices()["U-C100"] - fill.price) * 100.0
    assert equity == pytest.approx(expected)


def test_option_settles_at_expiry():
    engine = make_engine(n_steps=60, expiry=15)
    engine.step(2)
    engine.submit("U-C100", "buy", 1)
    engine.step(13)                                  # index == expiry
    assert "U-C100" not in engine.broker.positions

    settle = engine.broker.fills[-1]
    intrinsic = max(engine.prices()["U"] - 100.0, 0.0)
    assert settle.price == pytest.approx(intrinsic, abs=1e-3)
    assert settle.fee == 0.0
    assert any("行权结算" in log["message"] for log in engine.logs)


def test_cannot_trade_after_expiry():
    engine = make_engine(n_steps=40, expiry=10)
    engine.step(10)
    with pytest.raises(ValueError, match="到期"):
        engine.submit("U-C100", "buy", 1)


def test_option_requires_spec_and_valid_underlying():
    with pytest.raises(ValueError, match="OptionSpec"):
        Instrument(symbol="X", name="x", asset_class="option")
    with pytest.raises(ValueError, match="行权价"):
        OptionSpec(underlying="U", strike=0.0)
    orphan = Instrument(symbol="X", name="x", asset_class="option",
                        option=OptionSpec(underlying="NOPE", strike=100),
                        multiplier=100, tick_size=0.01)
    with pytest.raises(ValueError, match="标的"):
        MarketEngine([orphan], n_steps=10)


def test_probability_of_default_instruments_includes_option():
    session = Session(seed=1, n_steps=60)
    options = [i for i in session.instruments.values() if i.is_option]
    assert options and options[0].option.underlying in session.instruments
    with pytest.raises(ValueError, match="期权"):
        session.set_process(options[0].symbol, "gbm")


def test_delta_hedge_strategy_neutralizes_exposure():
    engine = make_engine(n_steps=80, expiry=40)
    engine.step(2)
    engine.submit("U-C100", "sell", 1)
    engine.step(1)
    engine.attach_strategy(create_strategy("delta_hedge",
                                           option="U-C100", band=0.05))
    engine.step(8)

    position = engine.broker.positions.get("U")
    assert position is not None and position.qty > 0

    spot = engine.prices()["U"]
    tau = (40 - engine.index) * engine.dt
    delta = black_scholes(spot, 100.0, tau, 0.02, 0.2)["delta"]
    net = position.qty - 1 * delta * 100.0          # 空 1 份期权 (mult=100)
    assert abs(net) < 40


def test_session_add_option_instrument():
    session = Session(seed=1, n_steps=60)
    meta = session.add_instrument(
        "510300P4", "300ETF沽4.00", asset_class="option",
        option={"underlying": "510300", "strike": 4.0, "kind": "put",
                "expiry_index": 40, "iv": 0.25, "multiplier": 10000})
    assert meta["asset_class"] == "option"
    assert meta["option"]["kind"] == "put"
    assert session.instruments["510300P4"].multiplier == 10000
    with pytest.raises(ValueError, match="已存在"):
        session.add_instrument("510300P4", "重复", asset_class="option",
                               option={"underlying": "510300", "strike": 4.0})

# --------------------------------------------------------------------------- #
# 微笑曲面（可选依赖 kairos-vol）
# --------------------------------------------------------------------------- #
def test_smile_surface_skew_direction():
    from kairos_terminal.volsurface import AVAILABLE, SmileSurface

    surface = SmileSurface(0.20)
    low, atm, high = (surface.iv(-0.2, 1.0), surface.iv(0.0, 1.0), surface.iv(0.2, 1.0))
    assert atm == pytest.approx(0.20, abs=1e-6)
    if AVAILABLE:
        assert low > atm > high                     # 负偏斜：低行权价 IV 更高
    else:
        assert low == pytest.approx(atm) and high == pytest.approx(atm)


def test_option_with_smile_is_priced_off_skewed_vol():
    from kairos_terminal.volsurface import AVAILABLE

    flat = Instrument(
        symbol="U-P", name="平", asset_class="option",
        option=OptionSpec(underlying="U", strike=80.0, kind="put",
                          expiry_index=20, iv=0.20),
        multiplier=100.0, tick_size=0.0001, precision=4, fee_rate=0.0)
    smile = Instrument(
        symbol="U-P", name="斜", asset_class="option",
        option=OptionSpec(underlying="U", strike=80.0, kind="put",
                          expiry_index=20, iv=0.20, smile=True),
        multiplier=100.0, tick_size=0.0001, precision=4, fee_rate=0.0)
    underlying = Instrument(symbol="U", name="标的", process=create("trend", sigma=0.0),
                            s0=100.0, fee_rate=0.0)

    engines = []
    for option in (flat, smile):
        engines.append(MarketEngine([underlying, option], n_steps=40, seed=1))
    if AVAILABLE:
        assert engines[1].prices()["U-P"] > engines[0].prices()["U-P"]
    else:
        assert engines[1].prices()["U-P"] == pytest.approx(engines[0].prices()["U-P"])


def test_add_option_chain_creates_strikes_and_expiries():
    session = Session(seed=1, n_steps=200)
    before = len(session.instruments)
    result = session.add_option_chain("510300", expiries=(60, 120), n_strikes=3)

    assert result["count"] == 2 * 3 * 2
    assert len(session.instruments) == before + result["count"]
    assert session.engine.index == 0                     # 只重建一次

    options = [session.instruments[s] for s in result["created"]]
    assert all(i.is_option and i.option.underlying == "510300" for i in options)
    assert {i.option.expiry_index for i in options} == {60, 120}
    assert {i.option.kind for i in options} == {"call", "put"}
    assert len({i.symbol for i in options}) == result["count"]


def test_option_chain_validation():
    session = Session(seed=1, n_steps=120)
    with pytest.raises(ValueError, match="标的不存在"):
        session.add_option_chain("NOPE")
    with pytest.raises(ValueError, match="奇数"):
        session.add_option_chain("510300", n_strikes=4)
    with pytest.raises(ValueError, match="width"):
        session.add_option_chain("510300", width=0.9)
    with pytest.raises(ValueError, match="非期权"):
        session.add_option_chain("510300C4")
