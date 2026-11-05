"""数据驱动过程：历史收益块重采样、自定义回调。

这两个过程**不出现在 Web 端的过程下拉框里**（``ui_selectable = False``），
因为它们需要外部数据 / 代码：

- :class:`BootstrapProcess`：从任意历史收益率序列做循环块重采样，保留真实数据的
  自相关与波动率聚集结构。可以接 ``kairos-data`` 里的后复权 CSV。
- :class:`CustomProcess`：直接传一个回调，实现任何你想要的路径生成器。
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from .base import DEFAULT_DT, ParamSpec, PriceProcess
from .registry import register

PathFn = Callable[..., np.ndarray]


@register
class BootstrapProcess(PriceProcess):
    key = "bootstrap"
    name = "历史块重采样"
    description = (
        "把一段真实历史对数收益率按「块」循环重采样，拼成同样长度的新路径。"
        "保留原始数据的肥尾、波动率聚集与短期自相关，但不假设任何参数模型——"
        "是检验模型假设的天然对照组。"
    )
    tags = ("数据驱动", "重采样")
    ui_selectable = False
    params = (
        ParamSpec("block", "块长度（交易日）", 20, "int", 1, 250, 1,
                  help="1 = 独立同分布重采样；越大越保留长记忆结构"),
    )

    def __init__(self, returns: Sequence[float], **params: Any) -> None:
        arr = np.asarray(list(returns), dtype="float64").ravel()
        if arr.size < 2:
            raise ValueError("BootstrapProcess 需要至少 2 个历史收益率")
        if not np.all(np.isfinite(arr)):
            raise ValueError("历史收益率中存在 NaN / inf，请先清洗")
        self._returns = arr
        super().__init__(**params)

    @classmethod
    def from_prices(cls, prices: Sequence[float], **params: Any) -> "BootstrapProcess":
        """由价格序列构造（内部取对数收益率）。"""
        p = np.asarray(list(prices), dtype="float64").ravel()
        if p.size < 3 or np.any(p <= 0):
            raise ValueError("价格序列至少需要 3 个正值")
        return cls(np.diff(np.log(p)), **params)

    def simulate(self, n_steps, *, dt=DEFAULT_DT, s0=100.0, seed=None, n_paths=1):
        n_steps, n_paths = self._prepare(n_steps, n_paths, s0, dt)
        rng = self._rng(seed)
        n = self._returns.size
        block = int(self["block"])
        increments = np.empty((n_paths, n_steps))
        for p in range(n_paths):
            idx = np.empty(n_steps, dtype="int64")
            filled = 0
            while filled < n_steps:
                start = int(rng.integers(0, n))
                take = min(block, n_steps - filled)
                idx[filled:filled + take] = (start + np.arange(take)) % n
                filled += take
            increments[p] = self._returns[idx]
        return self._from_log_increments(s0, increments, n_paths, n_steps)


@register
class CustomProcess(PriceProcess):
    key = "custom"
    name = "自定义过程"
    description = (
        "传入任意回调实现自己的价格过程。回调签名为 "
        "``fn(n_steps, dt, s0, seed, n_paths) -> ndarray``，"
        "返回 (n_paths, n_steps+1) 或 (n_steps+1,) 的价格矩阵。"
    )
    tags = ("自定义", "扩展")
    ui_selectable = False
    params = ()

    def __init__(self, fn: PathFn) -> None:
        if not callable(fn):
            raise TypeError("CustomProcess 需要一个可调用对象 fn")
        self._fn = fn
        super().__init__()

    def simulate(self, n_steps, *, dt=DEFAULT_DT, s0=100.0, seed=None, n_paths=1):
        n_steps, n_paths = self._prepare(n_steps, n_paths, s0, dt)
        out = self._fn(n_steps=n_steps, dt=dt, s0=s0, seed=seed, n_paths=n_paths)
        arr = np.asarray(out, dtype="float64")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape != (n_paths, n_steps + 1):
            raise ValueError(
                f"自定义过程返回形状 {arr.shape}，期望 {(n_paths, n_steps + 1)}")
        if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
            raise ValueError("自定义过程返回了非有限值或非正价格")
        arr[:, 0] = s0
        return arr