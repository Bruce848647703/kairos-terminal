"""可选的波动率微笑：用 ``kairos-vol`` 的 SSVI 给期权定价。

终端本身保持零依赖：只有安装了 ``kairos-vol``（``pip install -e ".[surface]"``）
时才会启用微笑；否则退化为合约里的常数隐含波动率。
"""
from __future__ import annotations

import math

try:                                    # 可选依赖
    import kairos_vol as _kv
    AVAILABLE = True
except Exception:                       # pragma: no cover - 取决于环境
    _kv = None
    AVAILABLE = False

DEFAULT_SKEW = {"rho": -0.6, "eta": 0.8, "gamma": 0.3}


class SmileSurface:
    """以 ATM 波动率为锚的 SSVI 微笑。

    对负偏斜的股票市场：低行权价（k<0）隐含波动率更高。
    """

    def __init__(self, atm_vol: float, *, rho: float = -0.6, eta: float = 0.8,
                 gamma: float = 0.3) -> None:
        atm_vol = float(atm_vol)
        if not atm_vol > 0:
            raise ValueError("atm_vol 必须为正")
        self.atm_vol = atm_vol
        self.params = {"rho": rho, "eta": eta, "gamma": gamma}
        self.model = _kv.SSVISurface(rho, eta, gamma) if AVAILABLE else None

    def iv(self, k: float, maturity: float) -> float:
        """对数货币度 ``k``、剩余期限 ``maturity`` 处的隐含波动率。"""
        t = max(float(maturity), 1e-8)
        if self.model is None:
            return self.atm_vol
        theta = self.atm_vol * self.atm_vol * t
        w = float(self.model.w(float(k), theta))
        return math.sqrt(max(w, 1e-12) / t)

    def iv_at_strike(self, spot: float, strike: float, maturity: float) -> float:
        if spot <= 0 or strike <= 0:
            raise ValueError("spot / strike 必须为正")
        return self.iv(math.log(float(strike) / float(spot)), maturity)


__all__ = ["AVAILABLE", "DEFAULT_SKEW", "SmileSurface"]
