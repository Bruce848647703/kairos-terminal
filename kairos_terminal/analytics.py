"""绩效统计（纯 numpy，单位为「年」）。"""
from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np

TRADING_DAYS = 252


def equity_returns(equity: Sequence[float]) -> np.ndarray:
    """净值曲线 -> 简单收益率序列。"""
    arr = np.asarray(equity, dtype="float64")
    if arr.size < 2:
        return np.empty(0)
    prev = arr[:-1]
    prev = np.where(prev == 0, np.nan, prev)
    return np.diff(arr) / prev


def total_return(equity: Sequence[float]) -> float:
    arr = np.asarray(equity, dtype="float64")
    if arr.size < 2 or arr[0] == 0:
        return 0.0
    return float(arr[-1] / arr[0] - 1.0)


def max_drawdown(equity: Sequence[float]) -> float:
    """最大回撤（负数，例如 -0.18 表示 -18%）。"""
    arr = np.asarray(equity, dtype="float64")
    if arr.size < 2:
        return 0.0
    peak = np.maximum.accumulate(arr)
    dd = np.where(peak > 0, arr / peak - 1.0, 0.0)
    return float(dd.min())


def annualized_vol(returns: Sequence[float], periods_per_year: int = TRADING_DAYS) -> float:
    r = np.asarray(returns, dtype="float64")
    r = r[np.isfinite(r)]
    if r.size < 2:
        return 0.0
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe(returns: Sequence[float], risk_free: float = 0.0,
           periods_per_year: int = TRADING_DAYS) -> float:
    r = np.asarray(returns, dtype="float64")
    r = r[np.isfinite(r)]
    if r.size < 2:
        return 0.0
    excess = r - risk_free / periods_per_year
    sd = excess.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def sortino(returns: Sequence[float], periods_per_year: int = TRADING_DAYS) -> float:
    r = np.asarray(returns, dtype="float64")
    r = r[np.isfinite(r)]
    if r.size < 2:
        return 0.0
    downside = r[r < 0]
    if downside.size < 2:
        return 0.0
    dd = downside.std(ddof=1)
    if dd == 0:
        return 0.0
    return float(r.mean() / dd * np.sqrt(periods_per_year))


def performance_summary(equity: Sequence[float],
                        periods_per_year: int = TRADING_DAYS) -> Dict[str, Any]:
    """终端顶部展示用的一组指标。"""
    arr = np.asarray(equity, dtype="float64")
    rets = equity_returns(arr)
    return {
        "total_return": total_return(arr),
        "max_drawdown": max_drawdown(arr),
        "volatility": annualized_vol(rets, periods_per_year),
        "sharpe": sharpe(rets, periods_per_year=periods_per_year),
        "sortino": sortino(rets, periods_per_year=periods_per_year),
        "bars": max(int(arr.size) - 1, 0),
    }


__all__ = [
    "TRADING_DAYS",
    "annualized_vol",
    "equity_returns",
    "max_drawdown",
    "performance_summary",
    "sharpe",
    "sortino",
    "total_return",
]