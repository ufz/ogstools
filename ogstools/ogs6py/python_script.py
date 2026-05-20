# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

from lxml import etree as ET

from ogstools.ogs6py.referenced_file import ReferencedFile


class PythonScript(ReferencedFile):
    """
    Class managing the python script file (.py) for an OGS project.

    Tracks both the XML reference to the script and the actual file,
    enabling proper save/copy operations.
    """

    __hash__ = None
    _NAME = "PythonScript"
    _EXT = "py"
    _XPATH = "python_script"

    def __init__(self, tree: ET.ElementTree) -> None:
        ReferencedFile.__init__(self, tree)
        self.populate_tree(self.root, "python_script", overwrite=True)

    def add_python_script(self, filename: str | Path) -> None:
        """
        Add/set a python script file.

        :param filename: The file path and name of the python script
        """
        filename = Path(filename)
        self._bind_to_path(filename)
        self.populate_tree(
            self.root,
            "python_script",
            text=str(filename.name),
            overwrite=True,
        )

    def set_pyscript(self, filename: str) -> None:
        """
        Set a filename for a python script.

        :param filename:
        """
        self.add_python_script(filename)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PythonScript):
            return NotImplemented
        if self.active_target is None or other.active_target is None:
            return self.active_target == other.active_target
        if not self.active_target.exists() or not other.active_target.exists():
            return False
        return (
            self.active_target.read_bytes() == other.active_target.read_bytes()
        )

    def __repr__(self) -> str:
        return f"PythonScript(filename={self.filename!r}, is_saved={self.is_saved})"

    def __str__(self) -> str:
        if not self.filename:
            return "PythonScript: (no script defined)"
        return f"PythonScript: {self.filename})"
