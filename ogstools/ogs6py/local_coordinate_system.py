# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from typing import Any

from lxml import etree as ET

from ogstools.ogs6py import build_tree


class LocalCoordinateSystem(build_tree.BuildTree):
    """
    Class for defining a local coordinate system in the project file.
    """

    def __init__(self, tree: ET.ElementTree) -> None:
        self.tree = tree
        self.root = self.tree.getroot()
        self.lcs = self.populate_tree(
            self.root, "local_coordinate_system", overwrite=True
        )

    def add_basis_vec(self, **args: Any) -> None:
        """
        Adds basis vectors.

        Parameters
        ----------
        basis_vector_0 : `str`
                         name of the parameter containing the basis vector
        basis_vector_1 : `str`
                         name of the parameter containing the basis vector
        basis_vector_2 : `str`
                         name of the parameter containing the basis vector
        """
        if "basis_vector_0" not in args:
            msg = "No vector given."
            raise KeyError(msg)
        if args["basis_vector_0"] is None:
            self.populate_tree(
                self.lcs, "basis_vector_0", attr={"implicit": "true"}
            )
        else:
            self.populate_tree(
                self.lcs, "basis_vector_0", text=args["basis_vector_0"]
            )
        if "basis_vector_1" in args:
            if args["basis_vector_1"] is None:
                self.populate_tree(
                    self.lcs, "basis_vector_1", attr={"implicit": "true"}
                )
            else:
                self.populate_tree(
                    self.lcs, "basis_vector_1", text=args["basis_vector_1"]
                )
        if "basis_vector_2" in args:
            if args["basis_vector_2"] is None:
                self.populate_tree(
                    self.lcs, "basis_vector_2", attr={"implicit": "true"}
                )
            else:
                self.populate_tree(
                    self.lcs, "basis_vector_2", text=args["basis_vector_2"]
                )
