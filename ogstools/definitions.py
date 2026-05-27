# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

import getpass
import tempfile
import uuid
from pathlib import Path

ROOT_DIR = Path(__file__).parent.resolve()
EXAMPLES_DIR = ROOT_DIR / "examples"
MATERIALS_DIR = EXAMPLES_DIR / "materiallib"


def usr_tmp_dir(id: str = "") -> Path:
    """Create a user-specific temporary directory for a given identifier.

    :param id: Optional subdirectory name.

    :returns: The pathlib.Path to the created directory.
    """
    base_tmp_dir = Path(tempfile.gettempdir())
    usr_tmp_dir = base_tmp_dir / ("ogstools" + "_" + getpass.getuser())
    (usr_tmp_dir / id).mkdir(parents=True, exist_ok=True)
    return usr_tmp_dir / id


def temp_file(suffix: str, dir: str = "", prefix: str = "") -> Path:
    """A file path in the ogstools user tmp directory.

    :param suffix: file extension/suffix.
    :param dir: Subdirectory name under the user temp path.
    :param prefix: filename prefix.
    """
    name = f"{prefix}{uuid.uuid4().hex}{suffix}"
    return usr_tmp_dir(dir) / name


def temp_dir(prefix: str = "", dir: str = "", suffix: str = "") -> Path:
    """A folder path in the ogstools user tmp directory.

    :param prefix: directory prefix.
    :param dir: Subdirectory name under the user temp path.
    :param suffix: directory suffix.
    """

    name = f"{prefix}{uuid.uuid4().hex}{suffix}/"
    folder = usr_tmp_dir(dir) / name
    folder.mkdir(parents=True)
    return folder
