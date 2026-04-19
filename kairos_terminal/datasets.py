"""本地行情数据集：扫描 CSV，给「历史块重采样」过程与终端提供真实价格。

数据集目录按以下顺序查找（可同时存在多个）：

1. 环境变量 ``KAIROS_TERMINAL_DATA``（多个目录用 ``:`` 分隔）
2. 当前目录下的 ``data/``
3. ``../kairos-data/data``（Kairos 系列的真实行情仓库）
4. ``~/Documents/Github/kairos-data/data``

CSV 约定：``date,open,high,low,close,volume``（大小写不敏感，``close`` 必需；
只有一列数值时按收盘价处理）。``kairos-data`` 里已提交的后复权行情可以直接用。

只读取本地文件，不联网。
"""
from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .processes import BootstrapProcess

PRICE_COLUMNS = ("close", "收盘", "close_price", "price", "adj_close")
DATE_COLUMNS = ("date", "日期", "datetime", "time")

#: 数据集元信息缓存：path -> (mtime, size, prices, meta)
_CACHE: Dict[str, Tuple[float, int, np.ndarray, Dict[str, Any]]] = {}
#: 目录扫描结果缓存：roots -> (时间戳, 列表)
_LIST_CACHE: Dict[str, Tuple[float, List["Dataset"]]] = {}


@dataclass(frozen=True)
class Dataset:
    """一个本地 CSV 数据集的摘要。"""

    name: str
    path: str
    rows: int
    start: str
    end: str
    last_close: float
    annual_vol: float
    total_return: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "path": self.path, "rows": self.rows,
            "start": self.start, "end": self.end, "last_close": self.last_close,
            "annual_vol": self.annual_vol, "total_return": self.total_return,
        }


def default_roots() -> List[Path]:
    """按优先级返回候选数据根目录（只保留真实存在的）。"""
    roots: List[Path] = []
    env = os.environ.get("KAIROS_TERMINAL_DATA", "")
    for part in env.split(os.pathsep):
        if part.strip():
            roots.append(Path(part).expanduser())
    here = Path.cwd()
    roots.extend([
        here / "data",
        here.parent / "kairos-data" / "data",
        Path.home() / "Documents" / "Github" / "kairos-data" / "data",
    ])
    out: List[Path] = []
    seen = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def _iter_csv(roots: Sequence[Path]) -> Iterable[Path]:
    for root in roots:
        for path in sorted(root.rglob("*.csv")):
            parts = path.relative_to(root).parts
            if any(p.startswith(".") for p in parts):
                continue
            yield path


def _read_prices(path: Path) -> Tuple[np.ndarray, List[str]]:
    """读取 CSV 中的收盘价序列（跳过无效行）。"""
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return np.empty(0), []

    header = [c.strip() for c in rows[0]]
    has_header = any(not _is_number(c) for c in header)
    body = rows[1:] if has_header else rows

    col = None
    if has_header:
        lowered = [h.lower() for h in header]
        for candidate in PRICE_COLUMNS:
            if candidate in lowered:
                col = lowered.index(candidate)
                break
        if col is None:
            col = len(header) - 1

    dates: List[str] = []
    prices: List[float] = []
    date_col = None
    if has_header:
        lowered = [h.lower() for h in header]
        for candidate in DATE_COLUMNS:
            if candidate in lowered:
                date_col = lowered.index(candidate)
                break

    for row in body:
        if not row:
            continue
        try:
            value = float(row[col]) if col is not None else float(row[-1])
        except (ValueError, IndexError):
            continue
        if not np.isfinite(value) or value <= 0:
            continue
        prices.append(value)
        if date_col is not None and date_col < len(row):
            dates.append(row[date_col])
    return np.asarray(prices, dtype="float64"), dates


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except (TypeError, ValueError):
        return False


def _summarize(name: str, path: Path, prices: np.ndarray, dates: List[str]) -> Dataset:
    log_ret = np.diff(np.log(prices)) if prices.size > 1 else np.empty(0)
    vol = float(log_ret.std(ddof=1) * np.sqrt(252)) if log_ret.size > 1 else 0.0
    total = float(prices[-1] / prices[0] - 1.0) if prices.size > 1 else 0.0
    return Dataset(
        name=name, path=str(path), rows=int(prices.size),
        start=dates[0] if dates else "", end=dates[-1] if dates else "",
        last_close=float(prices[-1]), annual_vol=vol, total_return=total)


def list_datasets(roots: Optional[Sequence[Path]] = None,
                  limit: int = 500, ttl: float = 10.0) -> List[Dataset]:
    """扫描所有数据根目录，返回按名称排序的数据集摘要（结果缓存 10 秒）。"""
    roots = list(roots) if roots is not None else default_roots()
    key = "|".join(str(r) for r in roots) + f"|{limit}"
    now = time.monotonic()
    cached = _LIST_CACHE.get(key)
    if cached is not None and now - cached[0] < ttl:
        return cached[1]

    out: List[Dataset] = []
    for path in _iter_csv(roots):
        if len(out) >= limit:
            break
        try:
            prices, dates = _read_prices(path)
        except OSError:
            continue
        if prices.size < 3:
            continue
        root = next((r for r in roots if str(path).startswith(str(r))), path.parent)
        name = path.relative_to(root).with_suffix("").as_posix()
        out.append(_summarize(name, path, prices, dates))
    _LIST_CACHE[key] = (now, out)
    return out


def resolve(name_or_path: str, roots: Optional[Sequence[Path]] = None) -> Path:
    """把数据集名称（``ashare/sh600519``）或路径解析成真实文件。"""
    candidate = Path(name_or_path).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    roots = list(roots) if roots is not None else default_roots()
    target = str(name_or_path).replace("\\", "/").lstrip("./")
    for path in _iter_csv(roots):
        root = next((r for r in roots if str(path).startswith(str(r))), path.parent)
        name = path.relative_to(root).with_suffix("").as_posix()
        if name == target or path.stem == target:
            return path.resolve()
    raise ValueError(
        f"找不到数据集 {name_or_path!r}；可用 KAIROS_TERMINAL_DATA 指定目录，"
        f"当前搜索路径：{[str(r) for r in roots] or '（无）'}")


def load_prices(name_or_path: str, roots: Optional[Sequence[Path]] = None) -> np.ndarray:
    """读取某个数据集的收盘价序列（带 mtime/size 缓存）。"""
    path = resolve(name_or_path, roots)
    stat = path.stat()
    key = str(path)
    cached = _CACHE.get(key)
    if cached and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
        return cached[2]
    prices, dates = _read_prices(path)
    if prices.size < 3:
        raise ValueError(f"{path} 有效价格点不足（{prices.size} 个）")
    _CACHE[key] = (stat.st_mtime, stat.st_size, prices, {"dates": dates})
    return prices


def load_dataset(name_or_path: str, roots: Optional[Sequence[Path]] = None) -> Dataset:
    path = resolve(name_or_path, roots)
    prices = load_prices(path)
    cached = _CACHE.get(str(path), (0, 0, None, {}))
    dates = cached[3].get("dates", []) if cached[3] else []
    target = str(name_or_path).replace("\\", "/").lstrip("./")
    return _summarize(target, path, prices, dates)


def make_process(name_or_path: str, *, block: int = 20,
                 roots: Optional[Sequence[Path]] = None) -> BootstrapProcess:
    """用本地数据集构造一个历史块重采样过程。"""
    return BootstrapProcess.from_prices(load_prices(name_or_path, roots), block=block)


def dataset_choices(roots: Optional[Sequence[Path]] = None) -> List[Dict[str, Any]]:
    """给 Web/CLI 用的轻量列表（不含完整路径细节）。"""
    return [d.to_dict() for d in list_datasets(roots)]


__all__ = [
    "Dataset",
    "dataset_choices",
    "default_roots",
    "list_datasets",
    "load_dataset",
    "load_prices",
    "make_process",
    "resolve",
]