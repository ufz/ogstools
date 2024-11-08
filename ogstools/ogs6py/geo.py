"""
Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
            Distributed under a Modified BSD License.
              See accompanying file LICENSE or
              http://www.opengeosys.org/project/license

"""
from lxml import etree as ET

from ogstools.ogs6py import build_tree


class Geo(build_tree.BuildTree):
    """
    Class containing the geometry file.
    """

    def __init__(self, tree: ET.ElementTree) -> None:
        self.tree = tree
        self.root = self.tree.getroot()
        self.populate_tree(self.root, "geometry", overwrite=True)

    def add_geometry(self, filename: str) -> None:
        """
        Adds a geometry file.

        :param filename:
        """
        self.populate_tree(self.root, "geometry", text=filename, overwrite=True)
