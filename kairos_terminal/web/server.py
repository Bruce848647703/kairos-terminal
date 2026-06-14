"""Web 终端服务端：标准库 ``http.server`` + JSON API，零第三方依赖。

前端是一个自包含的单页应用（``static/`` 下的 HTML/CSS/JS），图表用原生
``<canvas>`` 手绘，因此不需要 npm、不需要 CDN，断网也能跑。
"""
from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np

from ..session import Session, default_session

STATIC_DIR = Path(__file__).parent / "static"

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".json": "application/json; charset=utf-8",
}


def _json_default(value: Any) -> Any:
    """把 numpy 标量/数组转成可 JSON 序列化的 Python 对象。"""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"无法序列化 {type(value)!r}")


class TerminalHandler(BaseHTTPRequestHandler):
    """单页终端的所有路由。"""

    server_version = "KairosTerminal/0.1"
    protocol_version = "HTTP/1.1"

    # 由 TerminalServer 注入
    session: Session
    lock: threading.Lock
    quiet: bool = True

    # ------------------------------------------------------------------ 基础
    def log_message(self, fmt: str, *args: Any) -> None:      # noqa: D401
        if not self.quiet:
            super().log_message(fmt, *args)

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False,
                          default=_json_default).encode("utf-8")
        self._send(body, "application/json; charset=utf-8", status)

    def _error(self, message: str, status: int = 400) -> None:
        self._json({"ok": False, "error": message}, status)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > 4_000_000:
            raise ValueError("请求体过大")
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON 解析失败：{exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("请求体必须是 JSON 对象")
        return data

    # ------------------------------------------------------------------ 静态
    def _serve_static(self, path: str) -> None:
        name = path.lstrip("/")
        target = (STATIC_DIR / name).resolve()
        if not target.is_relative_to(STATIC_DIR.resolve()) or not target.is_file():
            self._error("文件不存在", 404)
            return
        self._send(target.read_bytes(),
                   CONTENT_TYPES.get(target.suffix.lower(), "application/octet-stream"))

    # ------------------------------------------------------------------ GET
    def do_GET(self) -> None:                                  # noqa: N802
        parsed = urlparse(self.path)
        route = parsed.path
        query = parse_qs(parsed.query)
        try:
            if route in ("/", "/index.html"):
                self._serve_static("index.html")
            elif route.startswith("/static/"):
                self._serve_static(route[len("/static/"):])
            elif route == "/api/meta":
                with self.lock:
                    self._json({"ok": True, "data": self.session.meta()})
            elif route == "/api/state":
                symbol = (query.get("symbol") or [None])[0]
                bars = int((query.get("bars") or ["200"])[0])
                with self.lock:
                    self._json({"ok": True, "data": self.session.snapshot(
                        symbol=symbol, bars=bars)})
            elif route == "/api/bars":
                symbol = (query.get("symbol") or [""])[0]
                start = int((query.get("start") or ["0"])[0])
                with self.lock:
                    self._json({"ok": True, "data": self.session.bars(symbol, start)})
            elif route == "/api/datasets":
                from .. import datasets

                q = (query.get("q") or [""])[0].lower()
                items = datasets.list_datasets()
                if q:
                    items = [d for d in items if q in d.name.lower()]
                self._json({"ok": True, "data": {
                    "roots": [str(r) for r in datasets.default_roots()],
                    "items": [d.to_dict() for d in items][:300],
                }})
            elif route == "/api/health":
                self._json({"ok": True, "data": {"status": "alive"}})
            else:
                self._error("未知路由", 404)
        except Exception as exc:                                # pragma: no cover
            self._error(str(exc), 400)

    # ------------------------------------------------------------------ POST
    def do_POST(self) -> None:                                 # noqa: N802
        route = urlparse(self.path).path
        try:
            payload = self._read_json()
            handler = {
                "/api/step": self._post_step,
                "/api/reset": self._post_reset,
                "/api/order": self._post_order,
                "/api/cancel": self._post_cancel,
                "/api/process": self._post_process,
                "/api/instrument": self._post_instrument,
                "/api/chain": self._post_chain,
                "/api/strategy": self._post_strategy,
                "/api/price": self._post_price,
            }.get(route)
            if handler is None:
                self._error("未知路由", 404)
                return
            with self.lock:
                data = handler(payload)
            self._json({"ok": True, "data": data})
        except (ValueError, KeyError) as exc:
            self._error(str(exc), 400)
        except Exception as exc:                                # pragma: no cover
            self._error(f"服务端错误：{exc}", 500)

    def _post_step(self, payload: Dict[str, Any]) -> Any:
        return self.session.step(int(payload.get("n", 1)))

    def _post_reset(self, payload: Dict[str, Any]) -> Any:
        return self.session.reset(
            seed=payload.get("seed"),
            n_steps=payload.get("n_steps"))

    def _post_order(self, payload: Dict[str, Any]) -> Any:
        return self.session.submit(
            payload.get("symbol", ""), payload.get("side", "buy"),
            int(payload.get("qty", 0)),
            order_type=str(payload.get("type", "market")),
            limit_price=payload.get("limit_price"),
            stop_price=payload.get("stop_price"))

    def _post_cancel(self, payload: Dict[str, Any]) -> Any:
        return self.session.cancel(int(payload.get("id", 0)))

    def _post_process(self, payload: Dict[str, Any]) -> Any:
        return self.session.set_process(
            payload.get("symbol", ""), payload.get("process", "gbm"),
            payload.get("params") or {}, payload.get("s0"))

    def _post_instrument(self, payload: Dict[str, Any]) -> Any:
        action = str(payload.get("action", "add"))
        if action == "remove":
            return self.session.remove_instrument(payload.get("symbol", ""))
        return self.session.add_instrument(
            payload.get("symbol", ""), payload.get("name", ""),
            payload.get("process", "gbm"),
            params=payload.get("params") or {},
            s0=(None if payload.get("s0") in (None, "") else float(payload.get("s0"))),
            asset_class=str(payload.get("asset_class", "stock")),
            multiplier=float(payload.get("multiplier", 1.0)),
            margin_rate=float(payload.get("margin_rate", 1.0)),
            allow_short=bool(payload.get("allow_short", False)),
            tick_size=float(payload.get("tick_size", 0.01)),
            precision=int(payload.get("precision", 2)),
            data_path=payload.get("data_path") or None,
            block=int(payload.get("block", 20)),
            option=payload.get("option") or None)

    def _post_chain(self, payload: Dict[str, Any]) -> Any:
        return self.session.add_option_chain(
            payload.get("underlying", ""),
            expiries=tuple(payload.get("expiries") or (60, 120)),
            n_strikes=int(payload.get("n_strikes", 3)),
            width=float(payload.get("width", 0.03)),
            iv=payload.get("iv"),
            multiplier=float(payload.get("multiplier", 10000.0)),
            smile=bool(payload.get("smile", True)))

    def _post_strategy(self, payload: Dict[str, Any]) -> Any:
        return self.session.set_strategy(
            payload.get("key"), payload.get("params") or {})

    def _post_price(self, payload: Dict[str, Any]) -> Any:
        return self.session.price(payload)


class TerminalServer:
    """把 :class:`~kairos_terminal.session.Session` 暴露成一个 HTTP 服务。"""

    def __init__(self, session: Optional[Session] = None,
                 host: str = "127.0.0.1", port: int = 8848) -> None:
        self.session = session or default_session()
        self.host = host
        self.port = int(port)
        self._lock = threading.Lock()
        self._httpd: Optional[ThreadingHTTPServer] = None

    def _make_server(self, port: int) -> ThreadingHTTPServer:
        handler = type("BoundTerminalHandler", (TerminalHandler,), {
            "session": self.session, "lock": self._lock, "quiet": True})
        httpd = ThreadingHTTPServer((self.host, port), handler)
        httpd.daemon_threads = True
        return httpd

    def start(self, *, port: Optional[int] = None, max_tries: int = 10) -> Tuple[str, int]:
        """启动服务（非阻塞），返回实际监听地址。端口被占用时自动顺延。"""
        base = int(port or self.port)
        last_error: Optional[Exception] = None
        for offset in range(max_tries):
            try:
                self._httpd = self._make_server(base + offset)
                self.port = int(self._httpd.server_address[1])
                break
            except OSError as exc:                       # 端口占用
                last_error = exc
        else:
            raise RuntimeError(f"端口 {base}~{base + max_tries - 1} 均被占用") from last_error

        thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        thread.start()
        return f"http://{self.host}:{self.port}", self.port

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    def serve_forever(self, *, open_browser: bool = True) -> None:
        """阻塞式运行（CLI 入口用）。"""
        url, _ = self.start()
        print(f"Kairos Terminal 已启动： {url}")
        print("按 Ctrl+C 退出")
        if open_browser:
            threading.Timer(0.6, lambda: webbrowser.open(url)).start()
        try:
            while self._httpd is not None:
                threading.Event().wait(0.5)
        except KeyboardInterrupt:
            print("\n正在关闭…")
        finally:
            self.stop()


__all__ = ["TerminalHandler", "TerminalServer", "STATIC_DIR"]