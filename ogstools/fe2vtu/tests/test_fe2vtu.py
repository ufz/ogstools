import subprocess

import pytest

pytest.importorskip("ifm")


def test_cli():
    subprocess.run(["fe2vtu", "--help"], check=True)
