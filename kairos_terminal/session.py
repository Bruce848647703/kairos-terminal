"""终端会话：把「品种配置 + 行情引擎 + 账户 + 策略 + 定价」组装成一个可操作对象。

Web 端与 CLI 都只跟 :class:`Session` 打交道，这样界面层完全不接触引擎内部状态，
也保证了「手工交易」与「策略交易」走同一条执行路径。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from . import datasets
from . import pricing
from . import realworld
from .analytics import performance_summary
from .instruments import Instrument, OptionSpec
from .market import MarketEngine
from .processes import create as create_process
from .processes import describe_all, keys as process_keys
from .strategy import create_strategy, describe_strategies


def default_instruments() -> List[Instrument]:
    """终端启动时的默认自选列表：覆盖 6 类过程、5 类资产（含一张期权）。"""
    return [
        Instrument(
            symbol="510300", name="沪深300ETF", asset_class="etf",
            process=create_process("gbm", mu=0.06, sigma=0.22), s0=3.80,
            tick_size=0.001, precision=3, fee_rate=0.0003),
        Instrument(
            symbol="510300C4", name="300ETF购4.00",
            asset_class="option",
            option=OptionSpec(underlying="510300", strike=4.00, kind="call",
                              expiry_index=200, iv=0.22),
            s0=0.10, multiplier=10000.0, tick_size=0.0001, precision=4,
            fee_rate=0.0003, allow_short=True),
        Instrument(
            symbol="600519", name="贵州茅台", asset_class="stock",
            process=create_process("heston", mu=0.10, v0=0.05, theta=0.05,
                                   kappa=2.5, xi=0.35, rho=-0.75),
            s0=1600.0, tick_size=0.01, fee_rate=0.0003, stamp_tax=0.0005),
        Instrument(
            symbol="IF2412", name="沪深300股指期货", asset_class="futures",
            process=create_process("regime", mu_calm=0.08, sigma_calm=0.14,
                                   mu_stress=-0.30, sigma_stress=0.40),
            s0=3800.0, multiplier=300.0, margin_rate=0.12, allow_short=True,
            tick_size=0.2, precision=1, fee_rate=0.000023),
        Instrument(
            symbol="BTC/USDT", name="比特币", asset_class="crypto",
            process=create_process("merton", mu=0.20, sigma=0.55, lam=1.2,
                                   jump_mu=-0.04, jump_sigma=0.12),
            s0=65000.0, tick_size=0.1, precision=2, fee_rate=0.0004),
        Instrument(
            symbol="AU2412", name="沪金主力", asset_class="futures",
            process=create_process("vg", mu=0.07, theta=-0.08, sigma=0.16, nu=0.30),
            s0=560.0, multiplier=1000.0, margin_rate=0.10, allow_short=True,
            tick_size=0.02, precision=2, fee_rate=0.000023),
    ]


class Session:
    """一个交易终端会话。"""

    def __init__(
        self,
        instruments: Optional[Sequence[Instrument]] = None,
        *,
        initial_cash: float = 1_000_000.0,
        n_steps: int = 500,
        seed: int = 42,
        slippage_bps: float = 2.0,
        matching: str = "bar",
    ) -> None:
        self._instruments: Dict[str, Instrument] = {
            i.symbol: i for i in (instruments or default_instruments())}
        self.engine = MarketEngine(
            list(self._instruments.values()), initial_cash=initial_cash,
            n_steps=n_steps, seed=seed, slippage_bps=slippage_bps,
            matching=matching)

    # ------------------------------------------------------------------ 查询
    @property
    def instruments(self) -> Dict[str, Instrument]:
        return dict(self._instruments)

    def snapshot(self, *, symbol: Optional[str] = None, bars: int = 200) -> Dict[str, Any]:
        snap = self.engine.snapshot(active_symbol=symbol, bars=bars)
        snap["performance"] = performance_summary(snap["account"]["equity_curve"])
        snap["processes"] = describe_all(ui_only=True)
        snap["strategies"] = describe_strategies()
        return snap

    def bars(self, symbol: str, start: int = 0) -> Dict[str, Any]:
        symbol = str(symbol).upper()
        data = self.engine.bars(symbol, start=int(start))
        return {"symbol": symbol, "start": int(start),
                "bars": [b.to_dict() for b in data]}

    # ------------------------------------------------------------------ 操作
    def step(self, n: int = 1) -> Dict[str, Any]:
        moved = self.engine.step(n)
        return {"moved": moved, "index": self.engine.index}

    def reset(self, *, seed: Optional[int] = None, n_steps: Optional[int] = None) -> Dict[str, Any]:
        self.engine.reset(seed=seed, n_steps=n_steps)
        return {"seed": self.engine.seed, "n_steps": self.engine.n_steps}

    def submit(self, symbol: str, side: str, qty: int, *,
               order_type: str = "market",
               limit_price: Optional[float] = None,
               stop_price: Optional[float] = None) -> Dict[str, Any]:
        order = self.engine.submit(symbol, side, qty, order_type=order_type,
                                   limit_price=limit_price, stop_price=stop_price)
        return order.to_dict()

    def cancel(self, order_id: int) -> Dict[str, Any]:
        return self.engine.cancel(order_id).to_dict()

    # ------------------------------------------------------------------ 配置
    def set_process(self, symbol: str, process_key: str,
                    params: Optional[Dict[str, Any]] = None,
                    s0: Optional[float] = None) -> Dict[str, Any]:
        """替换某品种的价格过程（会重新生成全部路径，账户清零）。"""
        symbol = str(symbol).upper()
        if symbol not in self._instruments:
            raise ValueError(f"未知品种 {symbol!r}")
        old = self._instruments[symbol]
        if old.is_option:
            raise ValueError(
                f"{symbol} 是期权，价格由标的推导；请改标的的过程，或新增合约")
        process = create_process(process_key, **(params or {}))
        self._instruments[symbol] = Instrument(
            symbol=old.symbol, name=old.name, process=process,
            s0=float(s0) if s0 is not None else old.s0,
            asset_class=old.asset_class, multiplier=old.multiplier,
            lot_size=old.lot_size, tick_size=old.tick_size,
            margin_rate=old.margin_rate, allow_short=old.allow_short,
            fee_rate=old.fee_rate, min_fee=old.min_fee,
            stamp_tax=old.stamp_tax, precision=old.precision)
        self._rebuild()
        return {"symbol": symbol, "process": process.key,
                "params": process.params_dict(), "reset": True}

    def add_instrument(self, symbol: str, name: str, process_key: str = "gbm",
                       *, params: Optional[Dict[str, Any]] = None,
                       s0: Optional[float] = None, asset_class: str = "stock",
                       multiplier: float = 1.0, margin_rate: float = 1.0,
                       allow_short: bool = False, tick_size: float = 0.01,
                       precision: int = 2,
                       data_path: Optional[str] = None, block: int = 20,
                       option: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """新增品种。

        - 普通品种：`process_key` + `params`，或 `data_path`（本地 CSV 重采样）；
        - 期权：`asset_class="option"` + `option={underlying, strike, kind, ...}`。
        """
        symbol = str(symbol).strip().upper()
        if not symbol:
            raise ValueError("symbol 不能为空")
        if symbol in self._instruments:
            raise ValueError(f"品种 {symbol} 已存在")

        spec = None
        process = None
        if asset_class == "option":
            spec = self._make_option_spec(option)
            if spec.underlying not in self._instruments:
                raise ValueError(f"期权标的 {spec.underlying} 不存在")
            if self._instruments[spec.underlying].is_option:
                raise ValueError("期权的标的不能是另一个期权")
            multiplier = float(option.get("multiplier") or multiplier)
            allow_short = bool(option.get("allow_short", True))   # 默认允许卖出开仓
            initial_price = 0.1                       # 期权价格由标的推导，占位
        elif data_path:
            process = datasets.make_process(data_path, block=int(block))
            initial_price = float(s0) if s0 is not None else float(
                datasets.load_prices(data_path)[-1])
        else:
            process = create_process(process_key, **(params or {}))
            initial_price = float(s0) if s0 is not None else 100.0

        inst = Instrument(
            symbol=symbol, name=str(name or symbol), asset_class=asset_class,
            process=process, option=spec, s0=initial_price,
            multiplier=float(multiplier), margin_rate=float(margin_rate),
            allow_short=bool(allow_short), tick_size=float(tick_size),
            precision=int(precision))
        self._instruments[symbol] = inst
        self._rebuild()
        return inst.to_dict()

    @staticmethod
    def _make_option_spec(option: Optional[Dict[str, Any]]) -> OptionSpec:
        if not option:
            raise ValueError("新增期权需要 option 参数（underlying/strike/kind/expiry_index/iv）")
        return OptionSpec(
            underlying=option.get("underlying", ""),
            strike=option.get("strike", 0.0),
            kind=option.get("kind", "call"),
            expiry_index=option.get("expiry_index", 60),
            iv=option.get("iv", 0.2),
            rate=option.get("rate", 0.02),
            dividend=option.get("dividend", 0.0))

    def add_real_instrument(self, symbol: str, data_path: str, *,
                            name: Optional[str] = None,
                            asset_class: str = "stock",
                            tick_size: float = 0.01,
                            precision: int = 2,
                            chain: bool = False) -> Dict[str, Any]:
        """用真实 CSV 校准 σ/漂移/ADV，建一个「该标的的仿真世界」。"""
        prices, volumes = realworld.load_series(data_path)
        stats = realworld.calibrate(prices, volumes)
        symbol = str(symbol).upper()
        if symbol in self._instruments:
            raise ValueError(f"品种 {symbol} 已存在")
        process = create_process("gbm", mu=stats["mu"], sigma=max(stats["sigma"], 1e-4))
        self._instruments[symbol] = Instrument(
            symbol=symbol, name=str(name or symbol), process=process,
            s0=stats["last_price"], asset_class=asset_class,
            tick_size=float(tick_size), precision=int(precision))
        self._rebuild()
        if chain:
            self.add_option_chain(symbol)
        return {**stats, "symbol": symbol, "data_path": data_path}

    def add_option_chain(
        self,
        underlying: str,
        *,
        expiries: Sequence[int] = (60, 120),
        n_strikes: int = 3,
        width: float = 0.03,
        iv: Optional[float] = None,
        multiplier: float = 10000.0,
        smile: bool = True,
    ) -> Dict[str, Any]:
        """按当前标的价格生成一组欧式期权（平价 ± n_strikes//2 档 × 看涨/看跌）。

        只做一次 ``_rebuild``，避免逐个添加时反复重建整条路径。
        """
        underlying = str(underlying).upper()
        if underlying not in self._instruments:
            raise ValueError(f"标的不存在：{underlying}")
        base = self._instruments[underlying]
        if base.is_option:
            raise ValueError("请选择非期权标的来生成期权链")
        if n_strikes < 1 or n_strikes % 2 == 0:
            raise ValueError("n_strikes 必须是正奇数（保证有平值档）")
        if not 0 < width < 0.5:
            raise ValueError("width 必须在 (0, 0.5) 内")
        if not expiries:
            raise ValueError("expiries 不能为空")

        spot = float(self.engine.prices().get(underlying, base.s0))
        anchor_iv = float(iv) if iv is not None else float(base.process.get("sigma", 0.2) or 0.2)
        anchor_iv = min(max(anchor_iv, 0.02), 2.0)
        half = n_strikes // 2
        created = []

        for expiry in expiries:
            expiry = int(expiry)
            if expiry < 1:
                raise ValueError("expiries 必须为正整数")
            for offset in range(-half, half + 1):
                strike = round(spot * (1.0 + offset * width), 2)
                if strike <= 0:
                    continue
                for kind, tag in (("call", "C"), ("put", "P")):
                    symbol = f"{underlying}{tag}{strike:g}_{expiry}"
                    if symbol in self._instruments:
                        continue
                    spec = OptionSpec(
                        underlying=underlying, strike=strike, kind=kind,
                        expiry_index=expiry, iv=anchor_iv, smile=smile)
                    self._instruments[symbol] = Instrument(
                        symbol=symbol,
                        name=f"{base.name}{'购' if kind == 'call' else '沽'}{strike:g}·{expiry}",
                        asset_class="option", option=spec, s0=0.1,
                        multiplier=float(multiplier), tick_size=0.0001,
                        precision=4, fee_rate=base.fee_rate, allow_short=True)
                    created.append(symbol)

        if not created:
            raise ValueError("没有新增合约（可能已存在同名合约）")
        self._rebuild()
        return {"underlying": underlying, "created": created, "count": len(created),
                "expiries": list(expiries), "n_strikes": int(n_strikes),
                "anchor_iv": anchor_iv}

    def remove_instrument(self, symbol: str) -> Dict[str, Any]:
        symbol = str(symbol).upper()
        if symbol not in self._instruments:
            raise ValueError(f"未知品种 {symbol!r}")
        if len(self._instruments) == 1:
            raise ValueError("至少保留一个品种")
        self._instruments.pop(symbol)
        self._rebuild()
        return {"removed": symbol}

    def set_strategy(self, key: Optional[str],
                     params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not key:
            self.engine.attach_strategy(None)
            return {"strategy": None}
        strategy = create_strategy(key, **(params or {}))
        self.engine.attach_strategy(strategy)
        return {"strategy": key, "params": strategy.params}

    def _rebuild(self) -> None:
        """品种集合变化后按原 seed / 步数重建引擎。"""
        old = self.engine
        self.engine = MarketEngine(
            list(self._instruments.values()),
            initial_cash=old.initial_cash, n_steps=old.n_steps,
            seed=old.seed, slippage_bps=old.slippage_bps,
            matching=old.matching, book_levels=old.book_levels)
        if old.strategy is not None:
            self.engine.attach_strategy(old.strategy)

    # ------------------------------------------------------------------ 定价
    def _process_for(self, symbol: str):
        """取定价要用的价格过程（期权取其标的的过程）。"""
        inst = self._instruments[symbol]
        if inst.is_option:
            inst = self._instruments[inst.option.underlying]
        return inst.process

    def _spot_for(self, symbol: str) -> float:
        inst = self._instruments[symbol]
        if inst.is_option:
            return self.engine.prices().get(inst.option.underlying, inst.s0)
        return self.engine.prices().get(symbol, inst.s0)

    def price(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """定价实验室的统一入口，``method`` 决定走哪条分支。"""
        payload = dict(payload or {})
        method = str(payload.get("method", "bs")).lower()
        kind = str(payload.get("kind", "call"))
        spot = payload.get("spot")
        symbol = str(payload.get("symbol") or next(iter(self._instruments))).upper()
        if symbol not in self._instruments:
            symbol = next(iter(self._instruments))
        if spot is None:
            spot = self._spot_for(symbol)
        spot = float(spot)
        strike = float(payload.get("strike", spot))
        maturity = float(payload.get("maturity", 1.0))
        rate = float(payload.get("rate", 0.02))
        dividend = float(payload.get("dividend", 0.0))
        vol = float(payload.get("vol", 0.2))

        if method == "bs":
            return pricing.black_scholes(spot, strike, maturity, rate, vol, kind, dividend)

        if method == "binomial":
            return pricing.binomial_crr(
                spot, strike, maturity, rate, vol, kind,
                int(payload.get("steps", 200)),
                american=bool(payload.get("american", False)), dividend=dividend)

        if method == "implied_vol":
            iv = pricing.implied_vol(
                float(payload.get("price", 0.0)), spot, strike, maturity, rate,
                kind, dividend)
            return {"implied_vol": iv, "vol": iv}

        if method == "mc":
            instrument = self._instruments.get(symbol)
            if instrument is None:
                raise ValueError(f"未知品种 {symbol!r}")
            return pricing.monte_carlo(
                self._process_for(symbol), spot, strike, maturity, rate, kind,
                dividend=dividend,
                n_paths=int(payload.get("n_paths", 20000)),
                n_steps=int(payload.get("n_steps", 252)),
                seed=payload.get("seed"),
                risk_neutral=bool(payload.get("risk_neutral", True)),
                payoff=str(payload.get("payoff", "vanilla")),
                barrier=payload.get("barrier"),
                barrier_kind=str(payload.get("barrier_kind", "up-out")))

        if method == "hedge":
            if symbol not in self._instruments:
                raise ValueError(f"未知品种 {symbol!r}")
            return pricing.delta_hedge_experiment(
                self._process_for(symbol), spot, strike, maturity, rate,
                float(payload.get("hedge_vol", vol)), kind,
                dividend=dividend,
                n_paths=int(payload.get("n_paths", 2000)),
                n_steps=int(payload.get("n_steps", 52)),
                seed=payload.get("seed"))

        if method == "compare":
            if symbol not in self._instruments:
                raise ValueError(f"未知品种 {symbol!r}")
            return pricing.compare_models(
                self._process_for(symbol), spot, strike, maturity, rate, vol, kind,
                dividend=dividend,
                n_paths=int(payload.get("n_paths", 20000)),
                n_steps=int(payload.get("n_steps", 252)),
                steps_binomial=int(payload.get("steps_binomial", 300)),
                seed=payload.get("seed"))

        raise ValueError(
            "method 必须是 bs / binomial / mc / implied_vol / hedge / compare，"
            f"收到 {method!r}")

    # ------------------------------------------------------------------ 元信息
    def meta(self) -> Dict[str, Any]:
        return {
            "processes": describe_all(ui_only=True),
            "strategies": describe_strategies(),
            "instruments": [i.to_dict() for i in self._instruments.values()],
            "process_keys": process_keys(),
            "seed": self.engine.seed,
            "n_steps": self.engine.n_steps,
            "slippage_bps": self.engine.slippage_bps,
            "matching": self.engine.matching,
        }


def default_session(**kwargs: Any) -> Session:
    return Session(**kwargs)


__all__ = ["Session", "default_instruments", "default_session"]