import os
from pathlib import Path

import pytest

from ogstools.definitions import ROOT_DIR

FOLDER_PATH = ROOT_DIR  # <-- Replace with your folder path
KEYWORD = "# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)"
CHECK_FIRST_N_LINES = 10  # bash shebang can be before the copyright


def get_py_files_missing_keyword(
    folder: Path, keyword: str, max_lines: int = 10
) -> list[str]:
    missing = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".py"):
                full_filename_path = folder / Path(root) / file
                try:
                    with full_filename_path.open("r", encoding="utf-8") as f:
                        lines = [f.readline().lower() for _ in range(max_lines)]
                        joined = "".join(lines).strip()
                        if joined == "":
                            # Skip empty files
                            continue
                        if not any(keyword.lower() in line for line in lines):
                            missing.append(file)
                except (UnicodeDecodeError, OSError) as e:
                    pytest.fail(f"Failed to read {file}: {e}")
    return missing


def test_all_py_files_contain_copyright() -> None:
    missing_files = get_py_files_missing_keyword(
        FOLDER_PATH, KEYWORD, CHECK_FIRST_N_LINES
    )
    assert not missing_files, "Missing copyright in files:\n" + "\n".join(
        missing_files
    )
