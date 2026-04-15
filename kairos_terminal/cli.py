"""命令行入口：``kairos-terminal`` / ``python -m kairos_terminal``。"""
from __future__ import annotations

import argparse
import sys
from typing import Any, List, Optional

from .analytics import performance_summary
from .processes import create as create_process
from .processes import describe_all, keys as process_keys

#: CLI 里可选的过程（排除需要外部数据/代码的 bootstrap / custom）
UI_PROCESS_KEYS = tuple(process_keys(ui_only=True))
from .pricing import compare_models
from .session import Session


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kairos-terminal",
        description="随机过程驱动的交易终端：价格过程可插拔，用于随机过程与金融数学定价研究。")
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="启动 Web 终端（默认）")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8848)
    serve.add_argument("--no-browser", action="store_true")
    serve.add_argument("--seed", type=int, default=42)
    serve.add_argument("--steps", type=int, default=500)
    serve.add_argument("--matching", default="bar", choices=("bar", "book"))

    world = sub.add_parser("world", help="用真实 CSV 校准并启动终端世界")
    world.add_argument("--data", default="", help="OHLCV CSV 路径")
    world.add_argument("--world-file", default="", help="kairos-lab 产出的 world.json")
    world.add_argument("--host", default="127.0.0.1")
    world.add_argument("--port", type=int, default=8848)
    world.add_argument("--no-browser", action="store_true")
    world.add_argument("--steps", type=int, default=500)
    world.add_argument("--seed", type=int, default=42)
    world.add_argument("--matching", default="book", choices=("bar", "book"))

    demo = sub.add_parser("demo", help="命令行跑一段并打印绩效")
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--steps", type=int, default=250)
    demo.add_argument("--strategy", default="", help="sma_cross / grid / momentum")
    demo.add_argument("--symbol", default="510300")
    demo.add_argument("--matching", default="bar", choices=("bar", "book"))

    price = sub.add_parser("price", help="同一期权在 BS / 二叉树 / 蒙特卡洛下的价格")
    price.add_argument("--process", default="heston", choices=UI_PROCESS_KEYS)
    price.add_argument("--spot", type=float, default=100.0)
    price.add_argument("--strike", type=float, default=100.0)
    price.add_argument("--maturity", type=float, default=1.0)
    price.add_argument("--rate", type=float, default=0.02)
    price.add_argument("--vol", type=float, default=0.2)
    price.add_argument("--kind", default="call", choices=("call", "put"))
    price.add_argument("--paths", type=int, default=20000)
    price.add_argument("--steps", type=int, default=252)

    sim = sub.add_parser("simulate", help="导出某过程的模拟路径 CSV")
    sim.add_argument("--process", default="gbm", choices=UI_PROCESS_KEYS)
    sim.add_argument("--data", default="", help="用本地数据集做历史块重采样（名称或路径）")
    sim.add_argument("--block", type=int, default=20, help="重采样块长度")
    sim.add_argument("--steps", type=int, default=252)
    sim.add_argument("--paths", type=int, default=1)
    sim.add_argument("--s0", type=float, default=0.0, help="0 表示用数据集最后收盘价")
    sim.add_argument("--seed", type=int, default=1)
    sim.add_argument("--out", default="-", help="输出文件，- 表示标准输出")

    data = sub.add_parser("data", help="列出可用的本地行情数据集")
    data.add_argument("--root", default="", help="额外指定数据目录")
    data.add_argument("--query", default="", help="按名称过滤")
    data.add_argument("--limit", type=int, default=30)

    sub.add_parser("processes", help="列出所有可用价格过程")
    return parser


def _cmd_serve(args: argparse.Namespace) -> int:
    from .web import TerminalServer

    session = Session(seed=args.seed, n_steps=args.steps, matching=args.matching)
    server = TerminalServer(session, host=args.host, port=args.port)
    server.serve_forever(open_browser=not args.no_browser)
    return 0


def _cmd_world(args: argparse.Namespace) -> int:
    from . import realworld
    from .instruments import Instrument
    from .processes import create as _create
    from .session import Session
    from .web import TerminalServer

    if not args.data and not args.world_file:
        raise SystemExit("world 需要 --data 或 --world-file")

    if args.world_file:
        entries = realworld.load_world(args.world_file)
        if not entries:
            raise SystemExit(f"{args.world_file} 里没有可用标的")
        instruments = []
        for entry in entries:
            conf = realworld.build_instrument_conf(entry)
            instruments.append(Instrument(
                symbol=conf["symbol"], name=conf["name"], asset_class=conf["asset_class"],
                process=_create("gbm", mu=conf["mu"], sigma=max(conf["sigma"], 1e-4)),
                s0=conf["s0"], tick_size=conf["tick_size"], precision=conf["precision"]))
    else:
        prices, volumes = realworld.load_series(args.data)
        stats = realworld.calibrate(prices, volumes)
        entries = [{"symbol": "ASSET", "name": "真实标的", **stats}]
        instruments = [Instrument(
            symbol="ASSET", name="真实标的",
            process=_create("gbm", mu=stats["mu"], sigma=max(stats["sigma"], 1e-4)),
            s0=stats["last_price"])]

    session = Session(instruments, seed=args.seed, n_steps=args.steps,
                      matching=args.matching)
    for entry in entries[:3]:
        session.add_option_chain(str(entry["symbol"]).upper(),
                                 expiries=(60, 120), n_strikes=3)

    server = TerminalServer(session, host=args.host, port=args.port)
    server.serve_forever(open_browser=not args.no_browser)
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    session = Session(seed=args.seed, n_steps=args.steps, matching=args.matching)
    if args.strategy:
        params = {"symbol": args.symbol}
        if args.strategy == "delta_hedge":
            params = {"open_short": True}          # 演示：先卖空 1 份期权再对冲
        session.set_strategy(args.strategy, params)
    session.step(args.steps)
    snap = session.snapshot(symbol=args.symbol)
    acct = snap["account"]
    perf = snap["performance"]

    print(f"Kairos Terminal demo · seed={args.seed} · {snap['index']} bars")
    print("-" * 56)
    for inst in snap["instruments"]:
        print(f"  {inst['symbol']:<10} {inst['name']:<12} "
              f"{inst['price']:>12,.2f}  {inst['change_pct']:>+7.2f}%  {inst['process_name']}")
    print("-" * 56)
    print(f"  权益      {acct['equity']:>14,.2f}")
    print(f"  现金      {acct['cash']:>14,.2f}")
    print(f"  已实现    {acct['realized_pnl']:>14,.2f}")
    print(f"  浮动盈亏  {acct['unrealized_pnl']:>14,.2f}")
    print(f"  收益率    {acct['return_pct']:>13.2f}%")
    print(f"  最大回撤  {perf['max_drawdown'] * 100:>13.2f}%")
    print(f"  夏普      {perf['sharpe']:>14.2f}")
    print(f"  成交笔数  {len(session.engine.broker.fills):>14d}")
    return 0


def _cmd_price(args: argparse.Namespace) -> int:
    process = create_process(args.process)
    result = compare_models(
        process, args.spot, args.strike, args.maturity, args.rate, args.vol,
        args.kind, n_paths=args.paths, n_steps=args.steps, seed=7)
    print(f"标的 {args.spot:.2f} · 行权 {args.strike:.2f} · 到期 {args.maturity:.2f}y "
          f"· r={args.rate:.2%} · σ={args.vol:.2%} · {args.kind}")
    print(f"模拟过程: {args.process}")
    print("-" * 46)
    print(f"  Black-Scholes {result['black_scholes']:>12.6f}")
    print(f"  二叉树        {result['binomial']:>12.6f}")
    print(f"  蒙特卡洛      {result['monte_carlo']:>12.6f}  "
          f"(±{1.96 * result['mc_stderr']:.6f}, 95%)")
    print("-" * 46)
    greeks = result["greeks"]
    print("  Delta {:.4f}  Gamma {:.4f}  Vega {:.4f}  Theta {:.4f}  Rho {:.4f}".format(
        greeks["delta"], greeks["gamma"], greeks["vega"], greeks["theta"], greeks["rho"]))
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    from pathlib import Path

    from . import datasets

    if args.data:
        process = datasets.make_process(args.data, block=int(args.block))
        s0 = args.s0 if args.s0 > 0 else float(datasets.load_prices(args.data)[-1])
    else:
        process = create_process(args.process)
        s0 = args.s0 if args.s0 > 0 else 100.0
    paths = process.simulate(args.steps, s0=s0, seed=args.seed, n_paths=args.paths)
    header = "step," + ",".join(f"path_{i}" for i in range(paths.shape[0]))
    lines = [header]
    for t in range(paths.shape[1]):
        lines.append(str(t) + "," + ",".join(f"{v:.6f}" for v in paths[:, t]))
    text = "\n".join(lines) + "\n"
    if args.out == "-":
        sys.stdout.write(text)
    else:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"已写入 {args.out}（{paths.shape[0]} 条路径 × {paths.shape[1]} 个点）")
    return 0


def _cmd_data(args: argparse.Namespace) -> int:
    from pathlib import Path

    from . import datasets

    roots = datasets.default_roots()
    if args.root:
        roots = [Path(args.root).expanduser()] + roots
    items = datasets.list_datasets(roots=roots, ttl=0)
    if args.query:
        q = args.query.lower()
        items = [d for d in items if q in d.name.lower()]
    print("搜索路径：" + (" , ".join(str(r) for r in roots) or "（无）"))
    print(f"{'数据集':<32}{'行数':>8}{'区间':>26}{'年化波动':>10}{'累计收益':>10}")
    print("-" * 90)
    for d in items[: args.limit]:
        span = f"{d.start} → {d.end}" if d.start else "—"
        print(f"{d.name:<32}{d.rows:>8}{span:>26}{d.annual_vol:>9.1%}{d.total_return:>10.1%}")
    if len(items) > args.limit:
        print(f"… 共 {len(items)} 个，仅显示前 {args.limit} 个")
    return 0


def _cmd_processes(_: argparse.Namespace) -> int:
    for meta in describe_all():
        flag = "" if meta["ui_selectable"] else "  [需外部数据/代码]"
        print(f"{meta['key']:<10} {meta['name']}{flag}")
        print(f"           {meta['description'][:88]}")
        params = ", ".join(f"{p['name']}={p['default']}" for p in meta["params"])
        if params:
            print(f"           参数: {params}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    if not argv_list or argv_list[0].startswith("-"):   # 不带子命令时等价于 serve
        args = parser.parse_args(["serve"] + argv_list)
    else:
        args = parser.parse_args(argv_list)
    if args.command == "serve":
        return _cmd_serve(args)
    return {
        "demo": _cmd_demo,
        "world": _cmd_world,
        "price": _cmd_price,
        "simulate": _cmd_simulate,
        "data": _cmd_data,
        "processes": _cmd_processes,
    }[args.command](args)


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())