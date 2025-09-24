import os
import shutil
from pathlib import Path

import pytest

import ogstools as ot
from ogstools.examples import mechanics_2D


@pytest.mark.tools()
@pytest.mark.skipif(
    shutil.which("identifySubdomains") is None, reason="binaries missing."
)
@pytest.mark.parametrize("stderr", [None, False])
def test_hide_cli_stderr(capfd, stderr):
    os.environ["OGS_BIN_PATH"] = str(Path(shutil.which("ogs")).parent)
    ot.cli().identifySubdomains(stderr=stderr)
    captured = capfd.readouterr()
    assert ("PARSE ERROR" in captured.err) == (stderr is None)


@pytest.mark.tools()
@pytest.mark.skipif(
    shutil.which("checkMesh") is None, reason="binaries missing."
)
@pytest.mark.parametrize("stdout", [None, False])
def test_hide_cli_stdout(capfd, stdout):
    os.environ["OGS_BIN_PATH"] = str(Path(shutil.which("ogs")).parent)
    ot.cli().checkMesh(mechanics_2D, stdout=stdout)
    captured = capfd.readouterr()
    assert ("info" in captured.out) == (stdout is None)
