"""价格过程注册表：按 ``key`` 查找 / 实例化 / 列出可用过程。"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Type

from .base import PriceProcess

_REGISTRY: Dict[str, Type[PriceProcess]] = {}


def register(cls: Type[PriceProcess]) -> Type[PriceProcess]:
    """类装饰器：把一个价格过程登记到全局注册表。"""
    key = getattr(cls, "key", "")
    if not key:
        raise ValueError(f"{cls.__name__} 缺少类属性 key")
    if not getattr(cls, "name", ""):
        raise ValueError(f"{cls.__name__} 缺少类属性 name")
    if key in _REGISTRY and _REGISTRY[key] is not cls:
        raise ValueError(f"价格过程 key 重复：{key!r}")
    _REGISTRY[key] = cls
    return cls


def keys(*, ui_only: bool = False) -> List[str]:
    """全部已注册过程的 key（按 name 排序，保证 UI 顺序稳定）。"""
    items = [c for c in _REGISTRY.values() if not ui_only or c.ui_selectable]
    return [c.key for c in sorted(items, key=lambda c: (not c.ui_selectable, c.name))]


def get(key: str) -> Type[PriceProcess]:
    """按 key 取类；不存在时给出可用列表。"""
    try:
        return _REGISTRY[str(key)]
    except KeyError:
        raise KeyError(f"未知价格过程 {key!r}，可用：{keys()}") from None


def create(key: str, **params: Any) -> PriceProcess:
    """按 key 实例化一个过程。"""
    return get(key)(**params)


def describe_all(*, ui_only: bool = False) -> List[Dict[str, Any]]:
    """所有过程的元数据（供 Web 端下拉框 + 参数表单）。"""
    out = [get(k).describe() for k in keys(ui_only=ui_only)]
    return out


def registered() -> Iterable[Type[PriceProcess]]:
    return tuple(_REGISTRY.values())


__all__ = ["create", "describe_all", "get", "keys", "register", "registered"]