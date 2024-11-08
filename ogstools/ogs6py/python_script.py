"""
Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
            Distributed under a Modified BSD License.
              See accompanying file LICENSE or
              http://www.opengeosys.org/project/license

"""
from lxml import etree as ET

from ogstools.ogs6py import build_tree


class PythonScript(build_tree.BuildTree):
    """
    Class managing python script in the project file.
    """

    def __init__(self, tree: ET.ElementTree) -> None:
        self.tree = tree
        self.root = self.tree.getroot()
        self.populate_tree(self.root, "python_script", overwrite=True)

    def set_pyscript(self, filename: str) -> None:
        """
        Set a filename for a python script.

        :param filename:
        """
        self.populate_tree(
            self.root, "python_script", text=filename, overwrite=True
        )
