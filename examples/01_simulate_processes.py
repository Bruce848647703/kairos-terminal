"""示例 1：同一个「品种」换不同价格过程，终端看到的世界完全不同。

运行：python examples/01_simulate_processes.py
"""
from __future__ import annotations

import numpy as np

from kairos_terminal.processes import create

PROCESSES = ["gbm", "trend", "heston", "merton", "bates", "ou", "cir", "regime", "vg"]


def main() -> None:
    print(f"{'过程':<10}{'期末均值':>12}{'5%分位':>12}{'95%分位':>12}"
          f"{'年化波动':>12}{'超额峰度':>12}")
    print("-" * 70)
    for key in PROCESSES:
        process = create(key)
        paths = process.simulate(252, s0=100.0, seed=7, n_paths=4000)
        terminal = paths[:, -1]
        log_ret = np.diff(np.log(paths), axis=1).ravel()
        vol = log_ret.std() * np.sqrt(252)
        centered = (log_ret - log_ret.mean()) / log_ret.std()
        kurt = float((centered ** 4).mean() - 3.0)
        q05, q95 = np.percentile(terminal, [5, 95])
        print(f"{process.name:<10}{terminal.mean():>12.2f}{q05:>12.2f}{q95:>12.2f}"
              f"{vol:>11.1%}{kurt:>12.2f}")
    print("\n提示：GBM 的峰度≈0；跳跃/方差Gamma 显著为正（肥尾）；"
          "OU 会向 long_run 回归；regime 的期末分布是两态混合。")


if __name__ == "__main__":
    main()