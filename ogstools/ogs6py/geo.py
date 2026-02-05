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


class Geo(build_tree.BuildTree, StorageBase):
    """
    Class managing the geometry file (.gml) for an OGS project.

    Tracks both the XML reference to the geometry and the actual file,
    enabling proper save/copy operations.
    """

    def __init__(
        self,
        tree: ET.ElementTree,
        id: str | None = None,
    ) -> None:
        """
        Initialize a Geo object.

        :param tree:        The Project's XML ElementTree (shared reference)
        :param source_path: Path to the source .gml file (if known)
        :param id:          Optional unique identifier
        """
        build_tree.BuildTree.__init__(self, tree)
        StorageBase.__init__(self, "Geo", "gml", id=id)
        self.root = self.tree.getroot()
        # Ensure <geometry> element exists in tree
        self.populate_tree(self.root, "geometry", overwrite=True)

    @property
    def filename(self) -> str | None:
        """Get the geometry filename from the XML tree."""
        geo_elem = self.root.find("geometry")
        if geo_elem is not None and geo_elem.text:
            return geo_elem.text.strip() or None
        return None

    @property
    def has_geometry(self) -> bool:
        """Check if geometry is defined (either as file or inline in XML)."""
        # Check if geometry element has content (inline definition)
        return self.root.find("geometry") is not None

    def add_geometry(self, file_pathname: str | Path) -> None:
        """
        Add/set a geometry file.

        :param filename:    The file path and name of the gml file
        """
        file_pathname = Path(file_pathname)
        self._bind_to_path(file_pathname)
        self.populate_tree(
            self.root,
            "geometry",
            text=str(file_pathname.name),
            overwrite=True,
        )

    def _propagate_target(self) -> None:
        """No children to propagate to."""

    def _save_impl(self, dry_run: bool = False) -> list[Path]:
        """
        Save the geometry file to the target location.

        :param dry_run: If True, don't actually copy the file
        :returns: List of saved file paths
        """
        if not self.filename:
            return []

        target = self.next_target

        if dry_run:
            return [target]

        # Ensure parent directory exists
        target.parent.mkdir(parents=True, exist_ok=True)

        # Copy the file if source != target
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
        Save the geometry file.

        :param target:    Optional target path
        :param overwrite: If True, overwrite existing files
        :param dry_run:   If True, simulate without writing
        :param archive:   If True, materialize symlinks
        :param id:        Optional identifier. Mutually exclusive with target.
        :returns: List of saved file paths
        """
        if not self.has_geometry:
            return []

        user_defined = self._pre_save(target, overwrite, dry_run, id=id)
        files = self._save_impl(dry_run)
        if files:  # Only post_save if we actually saved something
            self._post_save(user_defined, archive, dry_run)
        return files

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Geo):
            return NotImplemented
        return self.filename == other.filename

    def __repr__(self) -> str:
        return f"Geo(filename={self.filename!r}, is_saved={self.is_saved})"

    def __str__(self) -> str:
        if not self.has_geometry:
            return "Geo: (no geometry defined)"
        return f"Geo: {self.filename})"
