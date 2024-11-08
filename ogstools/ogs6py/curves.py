"""
Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
            Distributed under a Modified BSD License.
              See accompanying file LICENSE or
              http://www.opengeosys.org/project/license

"""
from lxml import etree as ET

from ogstools.ogs6py import build_tree


class Curves(build_tree.BuildTree):
    """
    Class to create the curve section of the project file.
    """

    def __init__(self, tree: ET.ElementTree) -> None:
        self.tree = tree
        self.root = self.tree.getroot()
        self.curves = self.populate_tree(self.root, "curves", overwrite=True)

    def add_curve(self, name: str, coords: list, values: list) -> None:
        """
        Adds a new curve.

        :param name:
        :param coords:
        :param values:
        """
        if len(coords) != len(values):
            msg = """Number of time coordinate points differs \
                     from number of values"""
            raise ValueError(msg)
        curve = self.populate_tree(self.curves, "curve")
        self.populate_tree(curve, "name", name)
        coord_str = ""
        value_str = ""
        for i, coord in enumerate(coords):
            if i < (len(coords) - 1):
                coord_str = coord_str + str(coord) + " "
                value_str = value_str + str(values[i]) + " "
            if i == (len(coords) - 1):
                coord_str = coord_str + str(coord)
                value_str = value_str + str(values[i])
        self.populate_tree(curve, "coords", text=coord_str)
        self.populate_tree(curve, "values", text=value_str)
