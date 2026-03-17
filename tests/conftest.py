import shutil

import pytest
from hypothesis import Verbosity, settings

import ogstools as ot


@pytest.fixture
def require_ogs_containers() -> None:
    """Skip the test if apptainer is not available."""
    if shutil.which("apptainer") is None:
        pytest.skip("apptainer not found. Run: make pull_containers")


@pytest.fixture
def require_ogs_wheel() -> None:
    """Skip the test if the OGS Python wheel is not installed."""
    from ogstools._find_ogs import has_ogs_wheel

    if not has_ogs_wheel():
        pytest.skip("OGS wheel not installed. Run: pip install ogstools[ogs]")


settings.register_profile("ci", max_examples=250, deadline=1000)
settings.register_profile("default", max_examples=50, deadline=350)
settings.register_profile("debug", max_examples=10, verbosity=Verbosity.verbose)

settings.load_profile("default")


@pytest.fixture(scope="session", autouse=True)
def set_userpath(tmp_path_factory):
    ot.StorageBase.Userpath = tmp_path_factory.getbasetemp()


def pytest_make_parametrize_id(config, val, argname):  # noqa: ARG001
    if hasattr(val, "output_name"):
        return val.output_name
    if isinstance(val, dict):
        values = "_".join(f"{k}_{v}" for k, v in val.items())
        if argname == "kwargs":
            return values
        return f"{argname}_{values}"
    if callable(val):
        return "lambda"
    return None  # lets pytest handle the rest
