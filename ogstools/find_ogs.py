import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def _find_all_executables(cmd: str) -> list[str]:
    """Find all occurrences of an executable in PATH (cross-platform)."""
    if platform.system() == "Windows":
        command = ["where", cmd]
    else:
        command = ["which", "-a", cmd]

    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=True
        )
        return list(set(result.stdout.strip().split("\n")))
    except subprocess.CalledProcessError:
        return []


def _is_outside_python_env(path: str) -> bool:
    """Check if the executable is outside the active Python environment."""
    python_env = sys.prefix  # Base directory of the active Python environment
    return not path.startswith(python_env)


def _check_path_complete(expected_binaries: list[str]) -> bool:
    existing_files = {
        tool for tool in expected_binaries if _find_all_executables(tool)
    }
    missing_files = set(expected_binaries) - existing_files
    if missing_files:
        print(
            f"Warning: Missing ogs binary tools. The following tools are missing: {missing_files}."
        )
        return False
    return True


def _has_ogs_wheel() -> bool:
    import importlib.util

    return importlib.util.find_spec("ogs") is not None


def _has_ogs_in_path() -> bool:
    ogs_executables = _find_all_executables("ogs")
    return any(_is_outside_python_env(x) for x in ogs_executables)


def _has_defined_ogs_path() -> bool:
    maybe_ogs_path = os.getenv("OGS_PATH", None)
    return maybe_ogs_path is not None


def check_ogs(print_only: bool = True) -> Any:
    """
    Check if OGS is installed correctly.
    """

    ogs_path = _has_defined_ogs_path()
    ogs_wheel = _has_ogs_wheel()
    ogs_in_path = _has_ogs_in_path()

    if print_only:
        print("OGS Wheel installed:", ogs_wheel)
        print("OGS_PATH set:", ogs_path)
        print("OGS in PATH:", ogs_in_path)

    err_missing = "Warning: No OGS installation found."

    executables_on_path = _find_all_executables("ogs")
    if print_only:
        print("OGS Executables on path:", executables_on_path)

    err_ambiguous = f"Warning: Too many occurrences of 'ogs' found in PATH. Found: {executables_on_path}.\n"

    resolution_missing = "Please install OGS via `pip install ogs`. OR specify an environment variable OGS_PATH locating the directory of a ogs binary and ogs binary tools.\n"

    resolution_select = "To continue without the Warning: Please remove either the OGS wheel with `pip uninstall ogs` OR unset OGS_PATH OR clean you global PATH environment variable.\n"
    continue_with_ambiguous = "If you continue with Warning OGS is taken with the precedence: OGS Wheel (first) -> OGS_PATH -> PATH (last). \n"
    recommendation_with_PATH = "Recommended: Clean your PATH environment.\n"
    continue_with_missing = "If you continue without any OGS installation, some OGSTools functionality will not work.\n"
    error_mapping = {
        #    Wheel, OGS_PATH, OGS in PATH  5/8 cases are error cases
        (False, False, False): err_missing
        + resolution_missing
        + continue_with_missing,
        (True, True, False): err_ambiguous
        + resolution_select
        + continue_with_ambiguous,
        (True, False, True): err_ambiguous
        + resolution_select
        + recommendation_with_PATH
        + continue_with_ambiguous,
        (False, True, True): err_ambiguous
        + resolution_select
        + recommendation_with_PATH
        + continue_with_ambiguous,
        (True, True, True): err_ambiguous
        + resolution_select
        + recommendation_with_PATH
        + continue_with_ambiguous,
    }

    msg = error_mapping.get((ogs_wheel, ogs_path, ogs_in_path), None)
    if msg:
        print(msg)
        return False

    if ogs_wheel:
        from ogs._internal.wrap_cli_tools import CLI as CLI_WHEEL

        if print_only:
            print("Using OGS wheel.")
            return True

        return CLI_WHEEL()

    if ogs_in_path:
        parent = Path(_find_all_executables("ogs")[0]).parent
        #       expected_binaries = [parent/bin for bin in binaries_list]
        #       if check_path_complete(expected_binaries):
        from ogstools.cli.wrap_cli_tools import CLI_ON_PATH

        if print_only:
            print("Using define in OGS_PATH:", parent)
            return True

        return CLI_ON_PATH(parent)

    if ogs_path:
        maybe_ogs_path_str = os.getenv("OGS_PATH", None)
        maybe_ogs_path = (
            Path(maybe_ogs_path_str) if maybe_ogs_path_str else None
        )
        if maybe_ogs_path and not maybe_ogs_path.exists():
            msg = (
                f"OGS_PATH is invalid. It is set to {maybe_ogs_path_str}"
                + err_missing
            )
            raise ImportError(msg)

        #        expected_binaries = [ogs_path/bin for bin in binaries_list]
        #        if check_path_complete(expected_binaries):
        from ogstools.cli.wrap_cli_tools import CLI_ON_PATH

        assert maybe_ogs_path is not None
        if print_only:
            print("Using define in OGS_PATH:", maybe_ogs_path.parent)
            return True
        return CLI_ON_PATH(maybe_ogs_path)

    return False
