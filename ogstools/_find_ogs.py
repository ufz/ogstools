import platform
import subprocess
import sys
from pathlib import Path


def find_all_executables(cmd: str) -> list[str]:
    """Find all occurrences of an executable in PATH (cross-platform)."""
    if platform.system() == "Windows":
        command = ["where", cmd]
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
    if not ogs_executables:
        print("ogs not found in PATH.")
    elif len(ogs_executables) > 1:
        print("Warning: Too many occurrences of 'ogs' found in PATH.")
        for path in ogs_executables:
            print(f"- {path}")
    elif is_outside_python_env(ogs_executables[0]):
        print(
            f"✅ Custom OGS found on {ogs_executables[0]}. For full functionality of OGSTools all ogs binary tools (e.g. vtkdiff) should be built and locate in {Path(ogs_executables[0]).parent}."
        )
