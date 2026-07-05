"""本地数据集测试：CSV 解析、扫描、按名解析、构造重采样过程。"""
from __future__ import annotations

import numpy as np
import pytest

from kairos_terminal import datasets


def write_csv(path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


@pytest.fixture()
def data_root(tmp_path):
    root = tmp_path / "data"
    (root / "ashare").mkdir(parents=True)
    write_csv(root / "ashare" / "sh600519.csv",
              "date,open,high,low,close,volume\n"
              "2024-01-02,100,101,99,100.5,1000\n"
              "2024-01-03,100.5,102,100,101.2,1200\n"
              "2024-01-04,101.2,103,101,102.8,900\n"
              "2024-01-05,102.8,104,102,103.9,1100\n")
    write_csv(root / "single.csv", "close\n10\n11\n12\n13\n")
    write_csv(root / "dirty.csv",
              "date,close\n2024-01-01,50\nbad,oops\n2024-01-02,NaN\n"
              "2024-01-03,52\n2024-01-04,-3\n2024-01-05,54\n")
    return root


def test_list_datasets(data_root):
    items = datasets.list_datasets(roots=[data_root], ttl=0)
    names = {d.name for d in items}
    assert "ashare/sh600519" in names
    assert "single" in names
    item = next(d for d in items if d.name == "ashare/sh600519")
    assert item.rows == 4
    assert item.start == "2024-01-02" and item.end == "2024-01-05"
    assert item.last_close == pytest.approx(103.9)
    assert item.total_return > 0


def test_load_prices_handles_single_column_and_dirty_rows(data_root):
    prices = datasets.load_prices("single", roots=[data_root])
    assert prices.tolist() == [10.0, 11.0, 12.0, 13.0]
    dirty = datasets.load_prices("dirty", roots=[data_root])
    assert dirty.tolist() == [50.0, 52.0, 54.0]        # NaN 与负价被丢弃


def test_resolve_by_name_and_path(data_root):
    by_name = datasets.resolve("ashare/sh600519", roots=[data_root])
    assert by_name.is_file()
    by_stem = datasets.resolve("single", roots=[data_root])
    assert by_stem.name == "single.csv"
    by_path = datasets.resolve(str(by_name))
    assert by_path == by_name
    with pytest.raises(ValueError, match="找不到数据集"):
        datasets.resolve("nope", roots=[data_root])


def test_make_process_is_deterministic(data_root):
    process = datasets.make_process("ashare/sh600519", block=2, roots=[data_root])
    a = process.simulate(20, seed=3, n_paths=2)
    b = process.simulate(20, seed=3, n_paths=2)
    assert a.shape == (2, 21)
    assert np.array_equal(a, b)
    assert process.key == "bootstrap"


def test_default_roots_respects_env(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIROS_TERMINAL_DATA", str(tmp_path))
    roots = datasets.default_roots()
    assert roots and roots[0] == tmp_path.resolve()


def test_dataset_choices_are_json_ready(data_root):
    items = datasets.dataset_choices(roots=[data_root])
    assert items and all(isinstance(d["rows"], int) for d in items)
    import json

    json.dumps(items)                              # 不应抛异常