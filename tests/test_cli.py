"""命令行入口测试：版本号、顶层帮助与默认子命令。"""
from __future__ import annotations

import pytest

from kairos_terminal.cli import main


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith("kairos-terminal ")


def test_cli_top_level_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "{serve,world,demo,price,simulate,data,processes}" in out


def test_cli_default_is_serve(monkeypatch):
    calls = []

    def fake_serve(args):
        calls.append(args)
        return 0

    monkeypatch.setattr("kairos_terminal.cli._cmd_serve", fake_serve)
    # 无子命令时默认 serve；顶层选项也要正确转发给 serve
    assert main([]) == 0
    assert main(["--port", "9999"]) == 0
    assert [c.port for c in calls] == [8848, 9999]