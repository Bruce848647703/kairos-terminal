"""Web 终端（标准库 HTTP 服务 + 单页前端）。"""
from __future__ import annotations

from .server import STATIC_DIR, TerminalHandler, TerminalServer

__all__ = ["STATIC_DIR", "TerminalHandler", "TerminalServer"]