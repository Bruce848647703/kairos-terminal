"""HTTP 接口测试：启动真实服务进程内实例，走完整 JSON 往返。"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from kairos_terminal.session import Session
from kairos_terminal.web import TerminalServer


@pytest.fixture(scope="module")
def live():
    session = Session(seed=2, n_steps=120)
    server = TerminalServer(session, port=0)
    url, port = server.start()
    assert port > 0
    yield url, session
    server.stop()


def _get(url: str, path: str):
    with urllib.request.urlopen(url + path, timeout=15) as resp:
        return resp.status, resp.headers.get("Content-Type", ""), resp.read()


def get_json(url: str, path: str):
    status, _, body = _get(url, path)
    return status, json.loads(body)


def post_json(url: str, path: str, payload: dict):
    req = urllib.request.Request(
        url + path, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.status, json.loads(resp.read())


def post_error(url: str, path: str, payload: dict):
    req = urllib.request.Request(
        url + path, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        urllib.request.urlopen(req, timeout=30)
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())
    raise AssertionError("请求本应失败")


def test_health_and_index(live):
    url, _ = live
    status, body = get_json(url, "/api/health")
    assert status == 200 and body["data"]["status"] == "alive"

    status, ctype, body = _get(url, "/")
    assert status == 200 and "text/html" in ctype
    assert b"KAIROS" in body


def test_static_assets(live):
    url, _ = live
    status, ctype, body = _get(url, "/static/app.js")
    assert status == 200 and "javascript" in ctype and b"drawChart" in body
    status, ctype, _ = _get(url, "/static/style.css")
    assert status == 200 and "text/css" in ctype


def test_state_and_meta(live):
    url, _ = live
    status, body = get_json(url, "/api/state")
    assert status == 200 and body["ok"]
    data = body["data"]
    assert data["account"]["equity"] > 0
    assert len(data["instruments"]) >= 5
    assert data["processes"]

    status, body = get_json(url, "/api/meta")
    assert status == 200
    assert any(p["key"] == "heston" for p in body["data"]["processes"])


def test_step_and_order_flow(live):
    url, session = live
    before = session.engine.index
    status, body = post_json(url, "/api/step", {"n": 3})
    assert status == 200 and body["data"]["moved"] == 3
    assert session.engine.index == before + 3

    status, body = post_json(url, "/api/order", {
        "symbol": "510300", "side": "buy", "qty": 100, "type": "market"})
    assert status == 200 and body["data"]["status"] == "pending"

    post_json(url, "/api/step", {"n": 1})
    status, body = get_json(url, "/api/state")
    symbols = [p["symbol"] for p in body["data"]["account"]["positions"]]
    assert "510300" in symbols


def test_bad_order_returns_400(live):
    url, _ = live
    status, body = post_error(url, "/api/order", {
        "symbol": "NOPE", "side": "buy", "qty": 1})
    assert status == 400 and "未知品种" in body["error"]


def test_unknown_route_404(live):
    url, _ = live
    try:
        _get(url, "/api/nothing")
    except urllib.error.HTTPError as exc:
        assert exc.code == 404
    else:
        raise AssertionError("应返回 404")


def test_price_endpoint(live):
    url, _ = live
    status, body = post_json(url, "/api/price", {
        "method": "bs", "spot": 100, "strike": 100, "maturity": 1.0,
        "rate": 0.02, "vol": 0.2, "kind": "call"})
    assert status == 200 and body["data"]["price"] > 0

    status, body = post_json(url, "/api/price", {
        "method": "mc", "symbol": "510300", "strike": 4.0, "maturity": 0.5,
        "n_paths": 1000, "n_steps": 32, "seed": 1})
    assert status == 200 and body["data"]["stderr"] > 0


def test_process_and_strategy_endpoints(live):
    url, session = live
    status, body = post_json(url, "/api/process", {
        "symbol": "510300", "process": "merton",
        "params": {"lam": 2.0}})
    assert status == 200
    assert session.instruments["510300"].process.key == "merton"
    assert session.engine.index == 0

    status, body = post_json(url, "/api/strategy", {"key": "grid", "params": {}})
    assert status == 200 and body["data"]["strategy"] == "grid"

    status, body = post_json(url, "/api/strategy", {"key": None})
    assert status == 200 and body["data"]["strategy"] is None


def test_instrument_endpoints(live):
    url, session = live
    status, body = post_json(url, "/api/instrument", {
        "action": "add", "symbol": "ETH/USDT", "name": "以太坊",
        "process": "gbm", "s0": 3000, "asset_class": "crypto"})
    assert status == 200 and "ETH/USDT" in session.instruments

    status, body = post_json(url, "/api/instrument", {
        "action": "remove", "symbol": "ETH/USDT"})
    assert status == 200 and "ETH/USDT" not in session.instruments


def test_datasets_endpoint(live):
    url, _ = live
    status, body = get_json(url, "/api/datasets")
    assert status == 200
    data = body["data"]
    assert isinstance(data["items"], list)
    assert isinstance(data["roots"], list)
    for item in data["items"][:3]:
        assert {"name", "rows", "start", "end", "last_close"} <= set(item)


def test_chain_endpoint(live):
    url, session = live
    status, body = post_json(url, "/api/chain", {
        "underlying": "510300", "expiries": [60], "n_strikes": 3})
    assert status == 200
    data = body["data"]
    assert data["count"] == 6
    assert all(s in session.instruments for s in data["created"])


def test_reset_endpoint(live):
    url, session = live
    status, body = post_json(url, "/api/reset", {"seed": 77, "n_steps": 90})
    assert status == 200
    assert session.engine.seed == 77 and session.engine.n_steps == 90
    assert session.engine.index == 0