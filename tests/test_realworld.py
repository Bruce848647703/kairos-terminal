"""真实数据 -> 终端世界 的校准与加载测试。"""
from __future__ import annotations

import json

import numpy as np
import pytest

from kairos_terminal import realworld
from kairos_terminal.session import Session


def write_csv(path, n=300, sigma_daily=0.015, volume=200_000.0, seed=4):
    rng = np.random.default_rng(seed)
    prices = 50.0 * np.exp(np.cumsum(rng.normal(0.0002, sigma_daily, size=n - 1)))
    prices = np.concatenate([[50.0], prices])
    lines = ["date,close,volume"]
    for i, (p, v) in enumerate(zip(prices, np.full(n, volume))):
        lines.append(f"2021-01-{i % 28 + 1:02d},{p:.6f},{v:.0f}")
    path.write_text("\n".join(lines), encoding="utf-8")
    return prices


def test_load_series_and_calibrate(tmp_path):
    path = tmp_path / "x.csv"
    prices = write_csv(path)
    loaded, volumes = realworld.load_series(str(path))
    assert loaded.size == prices.size and volumes.size == prices.size

    stats = realworld.calibrate(loaded, volumes)
    assert stats["n_obs"] == prices.size
    assert stats["last_price"] == pytest.approx(prices[-1])
    assert 0.1 < stats["sigma"] < 0.4
    assert stats["adv"] == pytest.approx(200_000)


def test_load_series_rejects_bad_input(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("date,close\n2021-01-01,10\n", encoding="utf-8")
    with pytest.raises(ValueError, match="不足"):
        realworld.load_series(str(path))
    with pytest.raises(ValueError):
        realworld.calibrate([1.0] * 10, [1.0] * 10)


def test_world_json_roundtrip(tmp_path):
    path = tmp_path / "world.json"
    path.write_text(json.dumps({
        "AAA": {"sigma": 0.25, "last_price": 100.0, "adv": 1e6},
        "BBB": {"symbol": "BBB", "sigma": 0.4, "last_price": 20.0},
    }), encoding="utf-8")
    entries = realworld.load_world(str(path))
    assert len(entries) == 2
    conf = realworld.build_instrument_conf(entries[0])
    assert conf["symbol"] in ("AAA", "BBB")
    assert conf["sigma"] > 0 and conf["s0"] > 0

    list_path = tmp_path / "list.json"
    list_path.write_text(json.dumps([{"symbol": "CCC", "sigma": 0.3}]), encoding="utf-8")
    assert realworld.load_world(str(list_path))[0]["symbol"] == "CCC"


def test_add_real_instrument_builds_calibrated_world(tmp_path):
    path = tmp_path / "real.csv"
    prices = write_csv(path)
    session = Session(seed=1, n_steps=200)
    info = session.add_real_instrument("REAL", str(path), name="真实标的", chain=False)

    inst = session.instruments["REAL"]
    assert inst.s0 == pytest.approx(prices[-1])
    assert inst.process["sigma"] == pytest.approx(info["sigma"])
    assert inst.process["mu"] == pytest.approx(info["mu"])
    assert abs(inst.process["sigma"] - 0.015 * np.sqrt(252)) < 0.05

    chained = session.add_real_instrument("REAL2", str(path), chain=True)
    assert chained["symbol"] == "REAL2"
    assert sum(1 for s in session.instruments if s.startswith("REAL2")) == 13


def test_world_file_can_build_a_session(tmp_path):
    path = tmp_path / "w.json"
    path.write_text(json.dumps({
        "X": {"sigma": 0.22, "last_price": 88.0, "adv": 5e5},
    }), encoding="utf-8")
    entries = [realworld.build_instrument_conf(e) for e in realworld.load_world(str(path))]
    assert entries[0]["symbol"] == "X" and entries[0]["s0"] == 88.0
