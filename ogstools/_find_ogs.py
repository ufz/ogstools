import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, cast


def find_all_executables(cmd: str) -> list[str]:
    """Find all occurrences of an executable in PATH (cross-platform)."""
    if platform.system() == "Windows":
        command = ["where.exe", cmd]
    else:
        command = ["which", "-a", cmd]

    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=True
        )
        return result.stdout.strip().split("\n")
    except subprocess.CalledProcessError:
        return []


def is_outside_python_env(path: str) -> bool:
    """Check if the executable is outside the active Python environment."""
    python_env = sys.prefix  # Base directory of the active Python environment
    return not path.startswith(python_env)


# Find all occurrences of 'ogs'


def check_path() -> None:
    ogs_executables = find_all_executables("ogs")
    if not ogs_executables and not os.getenv("OGS_BIN_PATH", None):
        print(
            "No OGS installation found. Please install ogs via `pip install ogs` OR specify an environment variable OGS_BIN_PATH locating the directory of a ogs binary and ogs binary tools."
        )
    elif len(ogs_executables) > 1:
        print("Occurrences of 'ogs' found in PATH: \n")
        for path in ogs_executables:
            print(f"- {path}")
    elif is_outside_python_env(ogs_executables[0]):
        print(f"✅ Custom OGS found on {ogs_executables[0]}.")


def has_ogs_wheel(verbose: bool = False) -> bool:
    import importlib.util

    if verbose:
        print("OGS wheel: ", importlib.util.find_spec("ogs"), ".\n")
    return importlib.util.find_spec("ogs") is not None


def has_ogs_in_path(verbose: bool = False) -> bool:
    ogs_executables = find_all_executables("ogs")
    all_outside_python_env = [
        x for x in ogs_executables if is_outside_python_env(x)
    ]
    if verbose:
        print("OGS in PATH: ", all_outside_python_env, ".\n")
    return any(is_outside_python_env(x) for x in ogs_executables)


def read_ogs_path(verbose: bool = False) -> Path | None:
    optional_ogs_path_str: str | None = os.getenv("OGS_BIN_PATH", None)
    if verbose:
        print("OGS_BIN_PATH: ", optional_ogs_path_str, ".\n")

    if optional_ogs_path_str is None:
        return None

    ogs_path: Path = Path(optional_ogs_path_str)

    if not ogs_path.exists():
        msg = f"OGS_BIN_PATH is invalid. It is set to {ogs_path!s}.\n"
        raise ImportError(msg)

    return ogs_path


def status(verbose: bool = False) -> bool:
    """
    Checks if OGS is installed correctly. It prints detailed error message if OGS is not installed correctly.
        :param verbose: If verbose is True it prints always the status of related environment variables. (OGS_BIN_PATH, PATH, virtual environment)

    :returns: True if OGS is installed correctly, False otherwise.
    """
    ogs_in_specified_path = read_ogs_path(verbose) is not None
    ogs_wheel = has_ogs_wheel(verbose)
    ogs_in_global_path = has_ogs_in_path(verbose)

    err_missing = "No OGS installation found. Please install ogs via `pip install ogs`. OR specify an environment variable OGS_BIN_PATH locating the directory of a ogs binary and ogs binary tools.\n"
    # {find_all_executables("ogs")}
    err_ambiguous = "Warning: Too many occurrences of 'ogs' found in PATH.\n"
    resolution_select = "Please remove either the ogs wheel with `pip uninstall ogs` OR unset OGS_BIN_PATH.\n"
    resolution_clean_path = "Please remove OGS from PATH.\n"

    error_mapping = {
        #    Wheel, OGS_BIN_PATH, OGS in PATH  5/8 cases are error cases
        (False, False, False): err_missing,
        (True, True, False): err_ambiguous + resolution_select,
        (True, False, True): err_ambiguous + resolution_clean_path,
        (False, True, True): err_ambiguous + resolution_clean_path,
        (True, True, True): err_ambiguous
        + resolution_select
        + resolution_clean_path,
    }

    msg = error_mapping.get(
        (ogs_wheel, ogs_in_specified_path, ogs_in_global_path), None
    )

    if msg and verbose:
        print(msg)
    return not msg


def cli() -> Any:
    """
    Allows access to ogs binary tools via python and performs checks to see if OGS is installed correctly.

    Example
    cli().vtkdiff("file1.vtu", "file2.vtu")

    :returns: A CLI object that supports ogs command line tools.
    """

    has_ogs_bin_path = read_ogs_path() is not None
    ogs_wheel = has_ogs_wheel()
    ogs_in_global_path = has_ogs_in_path()

    if not status():
        status(verbose=True)

    if has_ogs_bin_path:
        specified_path: Path = cast(Path, read_ogs_path())
        from ogstools._cli.wrap_cli_tools import CLI_ON_PATH

        return CLI_ON_PATH(specified_path)

    if ogs_wheel:
        from ogs._internal.wrap_cli_tools import CLI as CLI_WHEEL

        return CLI_WHEEL()

    if ogs_in_global_path:
        parent = Path(find_all_executables("ogs")[0]).parent
        from ogstools._cli.wrap_cli_tools import CLI_ON_PATH

        return CLI_ON_PATH(parent)

    return None
