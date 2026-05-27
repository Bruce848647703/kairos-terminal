"""用真实日线校准「终端世界」：σ、漂移、ADV 与最新价。

只读本地 CSV（``date,close,volume``，列名大小写不敏感），不联网、零依赖。
输出可以直接喂给 :class:`~kairos_terminal.instruments.Instrument` 的 GBM 过程，
也可以由 ``kairos-lab`` 写成 ``world.json`` 供终端加载。
"""
from __future__ import annotations

import csv
import math
from typing import Any, Dict, List, Tuple

import numpy as np


def load_series(path: str, *, price_column: str = "close",
                volume_column: str = "volume") -> Tuple[np.ndarray, np.ndarray]:
    """读取收盘价与成交量（跳过无效行）。"""
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path} 没有数据")
    lowered = {str(k).strip().lower(): k for k in rows[0].keys()}
    p_key = lowered.get(price_column.lower())
    if p_key is None:
        raise ValueError(f"{path} 缺少价格列 {price_column!r}")
    v_key = lowered.get(volume_column.lower())

    prices, volumes = [], []
    for row in rows:
        try:
            price = float(row[p_key])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(price) or price <= 0:
            continue
        volume = 0.0
        if v_key is not None:
            try:
                volume = max(float(row[v_key] or 0.0), 0.0)
            except (TypeError, ValueError):
                volume = 0.0
        prices.append(price)
        volumes.append(volume)
    if len(prices) < 30:
        raise ValueError(f"{path} 有效数据不足（{len(prices)} 行，至少 30 行）")
    return np.asarray(prices, dtype="float64"), np.asarray(volumes, dtype="float64")


def calibrate(prices, volumes, *, periods_per_year: float = 252.0) -> Dict[str, Any]:
    """估年化波动、年化漂移、ADV 与最新价。"""
    p = np.asarray(prices, dtype="float64").ravel()
    v = np.asarray(volumes, dtype="float64").ravel()
    if p.size != v.size or p.size < 30:
        raise ValueError("价格/成交量长度必须一致且至少 30 行")
    if np.any(p <= 0):
        raise ValueError("价格必须为正")

    returns = np.diff(np.log(p))
    sigma = float(returns.std(ddof=1) * math.sqrt(periods_per_year))
    mu = float(returns.mean() * periods_per_year)
    volume = v[1:]
    adv = float(np.median(volume)) if np.any(volume > 0) else 0.0
    return {
        "n_obs": int(p.size),
        "last_price": float(p[-1]),
        "sigma": sigma,
        "mu": mu,
        "adv": adv,
        "dollar_adv": adv * float(np.median(p)),
    }


def build_instrument_conf(entry: Dict[str, Any]) -> Dict[str, Any]:
    """把 ``world.json`` 的一项转成 ``Session.add_real_instrument`` 的参数。"""
    if "symbol" not in entry:
        raise ValueError("world 条目缺少 symbol")
    return {
        "symbol": str(entry["symbol"]).upper(),
        "name": str(entry.get("name") or entry["symbol"]),
        "sigma": float(entry.get("sigma", 0.25)),
        "mu": float(entry.get("mu", 0.05)),
        "s0": float(entry.get("last_price", entry.get("s0", 100.0))),
        "adv": float(entry.get("adv", 0.0)),
        "asset_class": str(entry.get("asset_class", "stock")),
        "tick_size": float(entry.get("tick_size", 0.01)),
        "precision": int(entry.get("precision", 2)),
    }


def load_world(path: str) -> List[Dict[str, Any]]:
    """读取 ``world.json``：``{symbol: {...}}`` 或 ``[{...}, ...]`` 两种形态都支持。"""
    import json

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        entries = []
        for key, value in data.items():
            item = dict(value or {})
            item.setdefault("symbol", key)
            entries.append(item)
        return entries
    if isinstance(data, list):
        return [dict(item) for item in data]
    raise ValueError("world.json 必须是对象或数组")


__all__ = ["build_instrument_conf", "calibrate", "load_series", "load_world"]
