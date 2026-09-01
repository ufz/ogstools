import os
import re
import shutil
from pathlib import Path

import pytest

import ogstools as ot
from ogstools import _find_ogs
from ogstools.examples import mechanics_2D


@pytest.mark.tools
@pytest.mark.skipif(
    shutil.which("identifySubdomains") is None, reason="binaries missing."
)
@pytest.mark.parametrize("stderr", [None, False])
def test_hide_cli_stderr(capfd, stderr):
    os.environ["OGS_BIN_PATH"] = str(Path(shutil.which("ogs")).parent)
    ot.cli().identifySubdomains(stderr=stderr)
    captured = capfd.readouterr()
    assert ("PARSE ERROR" in captured.err) == (stderr is None)


@pytest.mark.tools
@pytest.mark.skipif(
    shutil.which("checkMesh") is None, reason="binaries missing."
)
@pytest.mark.parametrize("stdout", [None, False])
def test_hide_cli_stdout(capfd, stdout):
    os.environ["OGS_BIN_PATH"] = str(Path(shutil.which("ogs")).parent)
    ot.cli().checkMesh(mechanics_2D, stdout=stdout)
    captured = capfd.readouterr()
    assert ("info" in captured.out) == (stdout is None)


@pytest.mark.tools
def test_dashed_args(tmp_path, capfd):
    ot.cli().NodeReordering(i=mechanics_2D, o=tmp_path / "test.vtu", l="info")
    captured = capfd.readouterr()
    assert "PARSE ERROR" not in captured.err


@pytest.mark.tools
def test_ogs_version_resolves():
    assert re.match(r"^\d+\.\d+\.\d+", _find_ogs.ogs_version() or "")


def test_ogs_version_without_ogs_is_none(monkeypatch):
    """No ogs anywhere (incl. a stale OGS_BIN_PATH) -> None, never raises."""
    monkeypatch.setenv("OGS_BIN_PATH", "/does/not/exist")
    monkeypatch.setattr(_find_ogs, "has_ogs_wheel", lambda *a, **k: False)
    monkeypatch.setattr(_find_ogs.shutil, "which", lambda _name: None)
    assert _find_ogs.ogs_version() is None
