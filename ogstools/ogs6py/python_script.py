# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

import shutil
from pathlib import Path

from lxml import etree as ET

from ogstools.core.storage import StorageBase
from ogstools.ogs6py import build_tree


class PythonScript(build_tree.BuildTree, StorageBase):
    """
    Class managing the python script file (.py) for an OGS project.

    Tracks both the XML reference to the script and the actual file,
    enabling proper save/copy operations.
    """

    def __init__(self, tree: ET.ElementTree) -> None:
        """
        Initialize a PythonScript object.

        :param tree: The Project's XML ElementTree (shared reference)
        """
        build_tree.BuildTree.__init__(self, tree)
        StorageBase.__init__(self, "PythonScript", "py")
        self.root = self.tree.getroot()
        self.populate_tree(self.root, "python_script", overwrite=True)

    @property
    def filename(self) -> str | None:
        """Get the python script filename from the XML tree."""
        script_elem = self.root.find("python_script")
        if script_elem is not None and script_elem.text:
            return script_elem.text.strip() or None
        return None

    def add_python_script(self, file_pathname: str | Path) -> None:
        """
        Add/set a python script file.

        :param file_pathname: The file path and name of the python script
        """
        file_pathname = Path(file_pathname)
        self._bind_to_path(file_pathname)
        self.populate_tree(
            self.root,
            "python_script",
            text=str(file_pathname.name),
            overwrite=True,
        )

    def set_pyscript(self, filename: str) -> None:
        """
        Set a filename for a python script.

        :param filename:
        """
        self.add_python_script(filename)

    def _propagate_target(self) -> None:
        """No children to propagate to."""

    def _save_impl(self, dry_run: bool = False) -> list[Path]:
        """
        Save the python script file to the target location.

        :param dry_run: If True, don't actually copy the file
        :returns: List of saved file paths
        """
        if not self.filename:
            return []

        target = self.next_target

        if dry_run:
            return [target]

        target.parent.mkdir(parents=True, exist_ok=True)

        if (
            self.active_target
            and self.active_target.exists()
            and self.active_target.resolve() != target.resolve()
        ):
            shutil.copy2(self.active_target, target)

        return [target]

    def save(
        self,
        target: Path | str | None = None,
        overwrite: bool | None = None,
        dry_run: bool = False,
        archive: bool = False,
        id: str | None = None,
    ) -> list[Path]:
        """
        Save the python script file.

        :param target:    Optional target path
        :param overwrite: If True, overwrite existing files
        :param dry_run:   If True, simulate without writing
        :param archive:   If True, materialize symlinks
        :param id:        Optional identifier. Mutually exclusive with target.
        :returns: List of saved file paths
        """
        if not self.filename:
            return []

        user_defined = self._pre_save(target, overwrite, dry_run, id=id)
        files = self._save_impl(dry_run)
        if files:
            self._post_save(user_defined, archive, dry_run)
        return files

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PythonScript):
            return NotImplemented
        # Compare file contents by full path
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
