"""价格过程抽象层。

设计目标
--------
本项目的核心命题是「**价格走势可插拔**」：同一个品种（股票 / ETF / 期货 / 加密）
在不同研究场景下应该可以换用不同的随机过程生成路径，而下游的撮合、账户、
策略、定价模块完全不需要改动。因此这里定义一个极薄的接口：

- :class:`ParamSpec`：一个参数的**自描述元数据**（类型 / 默认值 / 上下界 / 枚举 /
  单位 / 说明）。前端参数表单、CLI 帮助、参数校验全部由它自动生成，新增过程
  只需声明参数，不用写任何 UI 代码。
- :class:`PriceProcess`：所有过程的基类，唯一必须实现的方法是
  :meth:`PriceProcess.simulate`，返回形状 ``(n_paths, n_steps + 1)`` 的价格矩阵，
  第 0 列为初始价格 ``s0``。

约定
----
- 所有过程都是**纯函数式**的：给定 ``seed`` 结果完全确定，可复现。
- 时间单位统一为「年」，默认步长 ``dt = 1/252``（一个交易日）。
- 价格严格为正（均值回复过程在**对数价格**上做，CIR 过程在水平值上做但截断非负）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: 默认时间步长：一个交易日（按每年 252 个交易日折算）
DEFAULT_DT: float = 1.0 / 252.0

#: 参数类型 -> 中文名（用于自动生成表单 / 帮助）
KIND_LABELS: Dict[str, str] = {
    "float": "浮点数",
    "int": "整数",
    "bool": "开关",
    "choice": "枚举",
    "str": "文本",
}


@dataclass(frozen=True)
class ParamSpec:
    """一个过程参数的自描述元数据。"""

    name: str
    label: str
    default: Any
    kind: str = "float"
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    step: Optional[float] = None
    choices: Tuple[Any, ...] = ()
    unit: str = ""
    help: str = ""

    def coerce(self, value: Any) -> Any:
        """把外部输入（表单字符串 / JSON 数字）转成规范类型并做范围校验。"""
        if value is None:
            value = self.default
        try:
            if self.kind == "float":
                out: Any = float(value)
                if not np.isfinite(out):
                    raise ValueError(f"参数 {self.name} 必须是有限实数，收到 {value!r}")
            elif self.kind == "int":
                as_float = float(value)
                if not float(as_float).is_integer():
                    raise ValueError(f"参数 {self.name} 必须是整数，收到 {value!r}")
                out = int(as_float)
            elif self.kind == "bool":
                out = value if isinstance(value, bool) else str(value).strip().lower() in (
                    "1", "true", "yes", "on", "是")
            elif self.kind == "choice":
                out = value
                if out not in self.choices:
                    raise ValueError(
                        f"参数 {self.name} 只能取 {list(self.choices)}，收到 {value!r}")
            else:
                out = str(value)
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith("参数"):
                raise
            raise ValueError(f"参数 {self.name} 无法解析为 {self.kind}：{value!r}") from exc

        if self.kind in ("float", "int"):
            if self.minimum is not None and out < self.minimum:
                raise ValueError(
                    f"参数 {self.name} 不能小于 {self.minimum}（收到 {out}）")
            if self.maximum is not None and out > self.maximum:
                raise ValueError(
                    f"参数 {self.name} 不能大于 {self.maximum}（收到 {out}）")
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "default": self.default,
            "kind": self.kind,
            "kind_label": KIND_LABELS.get(self.kind, self.kind),
            "minimum": self.minimum,
            "maximum": self.maximum,
            "step": self.step,
            "choices": list(self.choices),
            "unit": self.unit,
            "help": self.help,
        }


class PriceProcess(ABC):
    """价格过程基类（子类通过类属性声明元数据）。"""

    key: ClassVar[str] = ""
    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    params: ClassVar[Tuple[ParamSpec, ...]] = ()
    tags: ClassVar[Tuple[str, ...]] = ()
    #: 是否出现在 Web 端「新增品种」的过程下拉框中（需要外部数据的设为 False）
    ui_selectable: ClassVar[bool] = True
    #: 是否为风险中性可切换过程（存在 ``mu`` 参数时自动可切换）
    has_drift_param: ClassVar[bool] = False

    def __init__(self, **overrides: Any) -> None:
        specs = {spec.name: spec for spec in self.params}
        unknown = sorted(set(overrides) - set(specs))
        if unknown:
            raise ValueError(
                f"{self.key or self.__class__.__name__} 不认识参数 {unknown}，"
                f"可用参数：{sorted(specs)}")
        self._values: Dict[str, Any] = {
            name: spec.coerce(overrides.get(name, spec.default))
            for name, spec in specs.items()
        }
        self.validate()

    # ---------------------------------------------------------------- 参数
    def __getitem__(self, name: str) -> Any:
        return self._values[name]

    def get(self, name: str, default: Any = None) -> Any:
        return self._values.get(name, default)

    def params_dict(self) -> Dict[str, Any]:
        """当前参数值（可直接 JSON 序列化）。"""
        return dict(self._values)

    def validate(self) -> None:
        """子类可覆盖：跨参数约束校验，失败时抛 :class:`ValueError`。"""

    # ---------------------------------------------------------------- 模拟
    @abstractmethod
    def simulate(
        self,
        n_steps: int,
        *,
        dt: float = DEFAULT_DT,
        s0: float = 100.0,
        seed: Optional[int] = None,
        n_paths: int = 1,
    ) -> np.ndarray:
        """生成价格路径矩阵，形状 ``(n_paths, n_steps + 1)``，第 0 列恒为 ``s0``。"""

    def simulate_path(
        self,
        n_steps: int,
        *,
        dt: float = DEFAULT_DT,
        s0: float = 100.0,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """单条路径的便捷入口，返回形状 ``(n_steps + 1,)``。"""
        return self.simulate(n_steps, dt=dt, s0=s0, seed=seed, n_paths=1)[0]

    def risk_neutral(self, rate: float, dividend: float = 0.0) -> "PriceProcess":
        """返回漂移被替换为 ``rate - dividend`` 的副本（用于无套利定价）。

        只有显式建模了 ``mu``（总漂移）的过程可以这样切换；均值回复类过程
        按其自身长期均衡演化（研究模式下保持原样，不强行改写）。
        """
        if "mu" not in self._values:
            return self
        return self.__class__(**{**self._values, "mu": float(rate) - float(dividend)})

    # ---------------------------------------------------------------- 工具
    @staticmethod
    def _prepare(n_steps: int, n_paths: int, s0: float, dt: float) -> Tuple[int, int]:
        n_steps, n_paths = int(n_steps), int(n_paths)
        if n_steps < 1:
            raise ValueError(f"n_steps 必须 >= 1，收到 {n_steps}")
        if n_paths < 1:
            raise ValueError(f"n_paths 必须 >= 1，收到 {n_paths}")
        if not np.isfinite(s0) or s0 <= 0:
            raise ValueError(f"s0 必须为正的有限数，收到 {s0!r}")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError(f"dt 必须为正的有限数，收到 {dt!r}")
        return n_steps, n_paths

    @staticmethod
    def _rng(seed: Optional[int]) -> np.random.Generator:
        return np.random.default_rng(seed)

    @staticmethod
    def _from_log_increments(
        s0: float, increments: np.ndarray, n_paths: int, n_steps: int
    ) -> np.ndarray:
        """对数收益率增量 ``(n_paths, n_steps)`` -> 价格 ``(n_paths, n_steps + 1)``。"""
        log_path = np.cumsum(increments, axis=1)
        paths = s0 * np.exp(log_path)
        head = np.full((n_paths, 1), float(s0), dtype="float64")
        return np.concatenate([head, paths], axis=1)

    # ---------------------------------------------------------------- 元数据
    @classmethod
    def describe(cls) -> Dict[str, Any]:
        """给前端/CLI 用的过程说明。"""
        return {
            "key": cls.key,
            "name": cls.name,
            "description": cls.description,
            "tags": list(cls.tags),
            "ui_selectable": cls.ui_selectable,
            "has_drift_param": "mu" in {spec.name for spec in cls.params},
            "params": [spec.to_dict() for spec in cls.params],
            "defaults": {spec.name: spec.default for spec in cls.params},
        }

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        args = ", ".join(f"{k}={v!r}" for k, v in self._values.items())
        return f"{self.__class__.__name__}({args})"


def require_positive(name: str, value: float) -> float:
    """常见的「必须为正」参数校验。"""
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"参数 {name} 必须为正的有限数，收到 {value!r}")
    return value


def ensure_probability(name: str, value: float) -> float:
    """常见的「必须在 [0, 1] 内」参数校验。"""
    value = float(value)
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"参数 {name} 必须在 [0, 1] 内，收到 {value!r}")
    return value


__all__ = [
    "DEFAULT_DT",
    "KIND_LABELS",
    "ParamSpec",
    "PriceProcess",
    "ensure_probability",
    "require_positive",
]