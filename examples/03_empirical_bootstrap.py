"""示例 3：把真实历史 CSV 变成「可交易的价格过程」（历史块重采样）。

用法：
    python examples/03_empirical_bootstrap.py                 # 用内置合成历史
    python examples/03_empirical_bootstrap.py path/to.csv     # 用你的真实行情

CSV 约定：含 ``date`` 列与 ``close`` 列（也可以直接给单列收盘价）。
kairos-data 仓库里已提交的后复权行情可以直接拿来用。
"""
from __future__ import annotations

import csv
import sys

import numpy as np

from kairos_terminal.processes import BootstrapProcess, create


def load_closes(path: str) -> np.ndarray:
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    if reader.fieldnames is None:
        raise SystemExit("CSV 为空")
    column = "close" if "close" in reader.fieldnames else reader.fieldnames[-1]
    closes = [float(r[column]) for r in rows if r.get(column)]
    return np.asarray(closes, dtype="float64")


def main() -> None:
    if len(sys.argv) > 1:
        prices = load_closes(sys.argv[1])
        print(f"读取 {sys.argv[1]}：{prices.size} 个收盘价")
    else:
        # 没有外部数据时，用 regime 过程造一段"真实历史"作为演示
        prices = create("regime").simulate_path(1200, s0=100.0, seed=3)
        print(f"未提供 CSV，使用内置 regime 合成历史：{prices.size} 个收盘价")

    process = BootstrapProcess.from_prices(prices, block=10)
    paths = process.simulate(252, s0=float(prices[-1]), seed=1, n_paths=4000)
    terminal = paths[:, -1]
    rets = np.diff(np.log(paths), axis=1)
    print(f"重采样路径：期末均值 {terminal.mean():.2f}，"
          f"5%~95% 区间 [{np.percentile(terminal, 5):.2f}, {np.percentile(terminal, 95):.2f}]，"
          f"年化波动 {rets.std() * np.sqrt(252):.1%}")
    print("用途：把它挂到 Session.add_instrument(..., process=...) 上，")
    print("      就能在终端里交易一条「保留真实波动率聚集」的路径。")


if __name__ == "__main__":
    main()