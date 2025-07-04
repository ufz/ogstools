# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from lxml import etree as ET

from ogstools.ogs6py import build_tree


class Mesh(build_tree.BuildTree):
    """
    Class for defining meshes in the project file.
    """

    def __init__(self, tree: ET.ElementTree) -> None:
        self.tree = tree
        self.root = self.tree.getroot()
        self.geometry = self.root.find("./geometry")
        self.meshes = self.root.find("./meshes")
        self.mesh = None
        if self.meshes is None:
            self.mesh = self.populate_tree(self.root, "mesh", overwrite=True)

    def add_mesh(
        self, filename: str, axially_symmetric: bool | str | None = None
    ) -> None:
        """
        Adds a mesh to the project file.

        :param filename:
        :param axially_symmetric:
        """
        attr_dict = {}
        if isinstance(axially_symmetric, bool):
            if axially_symmetric is True:
                attr_dict = {"axially_symmetric": "true"}
            else:
                attr_dict = {"axially_symmetric": "false"}
        elif isinstance(axially_symmetric, str):
            attr_dict = {"axially_symmetric": axially_symmetric}
        if self.mesh is not None:
            if (self.mesh.text == "") or (self.mesh.text is None):
                self.populate_tree(
                    self.root,
                    "mesh",
                    text=filename,
                    attr=attr_dict,
                    overwrite=True,
                )
            else:
                entry = self.mesh.text
                attrib = self.mesh.get("axially_symmetric")
                mesh0attr_dict = {}
                if isinstance(attrib, str):
                    mesh0attr_dict = {"axially_symmetric": attrib}
                self.mesh.tag = "meshes"
                self.meshes = self.populate_tree(
                    self.root, "meshes", text="", attr={}, overwrite=True
                )
                self.mesh = None
                if self.geometry is not None:
                    self.geometry.getparent().remove(self.geometry)
                    self.geometry = self.root.find("./geometry")
                self.populate_tree(
                    self.meshes, "mesh", text=entry, attr=mesh0attr_dict
                )
                self.populate_tree(
                    self.meshes, "mesh", text=filename, attr=attr_dict
                )
        elif self.meshes is not None:
            self.populate_tree(
                self.meshes, "mesh", text=filename, attr=attr_dict
            )
            self.geometry = self.root.find("./geometry")
            if self.geometry is not None:
                self.geometry.getparent().remove(self.geometry)
                self.geometry = self.root.find("./geometry")
        else:
            msg = "Mesh reference in the project file missing."
            raise RuntimeError(msg)
