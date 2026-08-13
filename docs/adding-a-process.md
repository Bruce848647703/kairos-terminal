# 新增一个价格过程

整个过程只需要一个类。注册表、Web 参数表单、CLI 帮助、参数校验、前端下拉框
都会自动包含它 —— 不需要改任何界面代码。

## 1. 最小实现

在 `kairos_terminal/processes/models_*.py`（或你自己的模块）里写：

```python
import numpy as np
from .base import DEFAULT_DT, ParamSpec, PriceProcess
from .registry import register

@register
class StudentTProcess(PriceProcess):
    key = "student_t"                     # URL / API / 配置里用的唯一标识
    name = "学生 t 收益"
    description = "用 t 分布刻画肥尾，ν 越小尾部越厚。"
    tags = ("肥尾", "扩散")
    params = (
        ParamSpec("mu", "年化漂移率", 0.05, "float", -2.0, 2.0, 0.01, unit="/年"),
        ParamSpec("sigma", "年化波动率", 0.25, "float", 0.01, 5.0, 0.01, unit="/年"),
        ParamSpec("df", "自由度 ν", 4.0, "float", 2.1, 100.0, 0.1,
                  help="越小尾部越厚；ν>30 接近正态"),
    )

    def simulate(self, n_steps, *, dt=DEFAULT_DT, s0=100.0, seed=None, n_paths=1):
        n_steps, n_paths = self._prepare(n_steps, n_paths, s0, dt)
        rng = self._rng(seed)
        mu, sigma, df = float(self["mu"]), float(self["sigma"]), float(self["df"])
        # t 分布方差 = df/(df-2)，标准化后再缩放，保证年化波动等于 sigma
        z = rng.standard_t(df, size=(n_paths, n_steps)) / np.sqrt(df / (df - 2.0))
        increments = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
        return self._from_log_increments(s0, increments, n_paths, n_steps)
```

在 `kairos_terminal/processes/__init__.py` 里导入它（触发注册）：

```python
from .models_student_t import StudentTProcess
```

## 2. 必须遵守的约定

| 约定 | 说明 |
|---|---|
| 返回形状 | `(n_paths, n_steps + 1)`，第 0 列必须等于 `s0` |
| 正价格 | 价格必须 > 0（用对数收益率累乘；水平过程要截断非负） |
| 确定性 | 只用 `self._rng(seed)`，同一 seed 必须完全一致 |
| 参数 | 全部通过 `ParamSpec` 声明，不要在 `simulate` 里硬编码 |
| 范围 | 单位统一为「年」，默认 `dt = 1/252` |

`_from_log_increments(s0, increments, n_paths, n_steps)` 会把对数收益率增量
累乘成价格路径，是最常用的收尾方式。

## 3. 需要外部数据 / 代码时

过程需要额外的构造参数（历史序列、回调）时：

1. 设置 `ui_selectable = False`，避免出现在 Web 下拉框里；
2. 重写 `__init__`，先保存外部数据再调用 `super().__init__(**params)`。

参考 `BootstrapProcess`（历史块重采样）与 `CustomProcess`（任意回调）。

## 4. 自检清单

```bash
python -m kairos_terminal processes         # 能否列出、参数是否正确
python -m pytest tests/test_processes.py    # 形状/确定性/参数校验
python -m kairos_terminal serve             # Web 端下拉框是否出现
```

再补一个测试：形状、`paths[:, 0] == s0`、正数、同 seed 可复现、非法参数报错。