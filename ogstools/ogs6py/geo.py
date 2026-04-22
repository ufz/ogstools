# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

from lxml import etree as ET

from ogstools.ogs6py.referenced_file import ReferencedFile


class Geo(ReferencedFile):
    """
    Class managing the geometry file (.gml) for an OGS project.

    Tracks both the XML reference to the geometry and the actual file,
    enabling proper save/copy operations.
    """

    __hash__ = None
    _NAME = "Geo"
    _EXT = "gml"
    _XPATH = "geometry"

    def __init__(self, tree: ET.ElementTree) -> None:
        ReferencedFile.__init__(self, tree)
        self.populate_tree(self.root, "geometry", overwrite=True)

    @property
    def has_geometry(self) -> bool:
        """Check if geometry is defined (either as file or inline in XML)."""
        return self.root.find("geometry") is not None

    def add_geometry(self, filename: str | Path) -> None:
        """
        Add/set a geometry file.

        :param filename: The file path and name of the gml file
        """
        filename = Path(filename)
        self._bind_to_path(filename)
        self.populate_tree(
            self.root,
            "geometry",
            text=str(filename.name),
            overwrite=True,
        )

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
