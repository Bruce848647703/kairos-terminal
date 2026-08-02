"""示例 2：模型错配的代价 —— 用 Heston 生成真实世界，却用 BS 定价与 Delta 对冲。

运行：python examples/02_price_under_misspecification.py
"""
from __future__ import annotations

from kairos_terminal.pricing import (
    black_scholes,
    compare_models,
    delta_hedge_experiment,
)
from kairos_terminal.processes import create


def main() -> None:
    spot, strike, maturity, rate = 100.0, 100.0, 1.0, 0.02
    hedge_vol = 0.20

    print("=" * 66)
    print("1) 同一份期权：BS / 二叉树 / 蒙特卡洛（真实过程 = Heston）")
    print("=" * 66)
    heston = create("heston", mu=0.02, v0=0.04, theta=0.09, kappa=1.5,
                    xi=0.6, rho=-0.8)
    for key in ("gbm", "heston", "merton", "vg"):
        process = create(key, mu=0.02) if key != "heston" else heston
        res = compare_models(process, spot, strike, maturity, rate, hedge_vol,
                             n_paths=20000, n_steps=252, seed=11)
        print(f"  {key:<8} BS={res['black_scholes']:.4f}  "
              f"树={res['binomial']:.4f}  MC={res['monte_carlo']:.4f} "
              f"(±{1.96 * res['mc_stderr']:.4f})")

    print()
    print("=" * 66)
    print("2) 卖出一份期权，用 σ=20% 的 BS Delta 对冲，真实波动结构不同会怎样")
    print("=" * 66)
    scenarios = {
        "GBM σ=20%（模型正确）": create("gbm", mu=rate, sigma=0.20),
        "Heston 随机波动": create("heston", mu=rate, v0=0.04, theta=0.09,
                                  kappa=1.5, xi=0.6, rho=-0.8),
        "Merton 跳跃": create("merton", mu=rate, sigma=0.18, lam=1.0,
                              jump_mu=-0.05, jump_sigma=0.15),
        "VG 肥尾": create("vg", mu=rate, theta=-0.1, sigma=0.2, nu=0.3),
    }
    print(f"  {'场景':<22}{'均值PnL':>10}{'标准差':>10}{'5%分位':>10}{'亏损概率':>10}")
    for name, process in scenarios.items():
        res = delta_hedge_experiment(process, spot, strike, maturity, rate,
                                     hedge_vol, n_paths=4000, n_steps=52, seed=5)
        p05 = res["pnl_percentiles"][1]
        print(f"  {name:<22}{res['mean_pnl']:>10.3f}{res['std_pnl']:>10.3f}"
              f"{p05:>10.3f}{res['prob_loss']:>10.1%}")

    print()
    print(f"参考：BS 理论权利金 = {black_scholes(spot, strike, maturity, rate, hedge_vol)['price']:.4f}")
    print("结论：模型正确时对冲 PnL 均值≈0，只剩离散对冲噪声；")
    print("      真实过程含随机波动/跳跃时，均值与尾部会系统性偏移 —— 这就是模型风险。")


if __name__ == "__main__":
    main()