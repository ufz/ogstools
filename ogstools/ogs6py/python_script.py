# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

import copy
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

    def _save_impl(self, dry_run: bool = False) -> list[Path]:
        # OGS imports from the folder of the script, so the python files next
        # to it are saved (linked) with it, the same way as the script.
        files = super()._save_impl(dry_run)
        source = self.active_target
        if not files or source is None or not source.exists():
            return files
        target_dir = self.next_target.parent
        if source.parent.resolve() == target_dir.resolve():
            return files
        for module in sorted(source.parent.glob("*.py")):
            if module.name != source.name and module.is_file():
                module_file = copy.copy(self)
                module_file._active_target = module
                module_file.link(target_dir / module.name, dry_run)
                files.append(target_dir / module.name)
        return files

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
