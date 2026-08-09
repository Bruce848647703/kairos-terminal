"""示例 4：在终端里挂策略，跑一段并打印成交与绩效。

运行：python examples/04_run_strategy.py
"""
from __future__ import annotations

from kairos_terminal.analytics import performance_summary
from kairos_terminal.session import Session


def main() -> None:
    session = Session(seed=2024, n_steps=250)
    session.set_strategy("sma_cross", {"symbol": "510300", "fast": 5, "slow": 20,
                                       "qty": 1000})
    session.step(250)

    snap = session.snapshot(symbol="510300")
    perf = performance_summary(snap["account"]["equity_curve"])
    print(f"bars={snap['index']}  成交={len(snap['fills'])}  委托={len(snap['orders'])}")
    print(f"权益 {snap['account']['equity']:,.2f}  "
          f"收益率 {snap['account']['return_pct']:+.2f}%  "
          f"最大回撤 {perf['max_drawdown'] * 100:.2f}%  "
          f"夏普 {perf['sharpe']:.2f}")
    for fill in snap["fills"][:10]:
        print(f"  bar {fill['index']:>3}  {fill['symbol']:<8}{fill['side']:<5}"
              f"{fill['qty']:>6}@{fill['price']:.2f}  费用 {fill['fee']:.2f}")


if __name__ == "__main__":
    main()