# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

from ogstools import plot
from ogstools._internal import copy_method_signature

from . import data_processing, geo, ip_mesh


class Mesh(pv.UnstructuredGrid):
    """
    A wrapper around pyvista.UnstructuredGrid.

    Contains additional data and functions mainly for postprocessing.
    """

    filepath: Path | None = None

    # pylint: disable=C0116
    @copy_method_signature(data_processing.difference)
    def difference(self, *args: Any, **kwargs: Any) -> Any:
        return data_processing.difference(self, *args, **kwargs)

    @copy_method_signature(geo.depth)
    def depth(self, *args: Any, **kwargs: Any) -> Any:
        return geo.depth(self, *args, **kwargs)

    @copy_method_signature(geo.p_fluid)
    def p_fluid(self, *args: Any, **kwargs: Any) -> Any:
        return geo.p_fluid(self, *args, **kwargs)

    @copy_method_signature(plot.contourf)
    def plot_contourf(self, *args: Any, **kwargs: Any) -> Any:
        return plot.contourf(self, *args, **kwargs)

    @copy_method_signature(plot.quiver)
    def plot_quiver(self, *args: Any, **kwargs: Any) -> Any:
        return plot.quiver(self, *args, **kwargs)

    @copy_method_signature(plot.streamlines)
    def plot_streamlines(self, *args: Any, **kwargs: Any) -> Any:
        return plot.streamlines(self, *args, **kwargs)

    def to_ip_mesh(self) -> Mesh:
        return Mesh(ip_mesh.to_ip_mesh(self))

    def to_ip_point_cloud(self) -> Mesh:
        return Mesh(ip_mesh.to_ip_point_cloud(self))

    to_ip_mesh.__doc__ = ip_mesh.to_ip_mesh.__doc__
    to_ip_point_cloud.__doc__ = ip_mesh.to_ip_point_cloud.__doc__

    # pylint: enable=C0116

    def __init__(
        self,
        pv_mesh: pv.UnstructuredGrid | None = None,
        **kwargs: dict,
    ):
        """
        Initialize a Mesh object

        :param pv_mesh: Underlying pyvista mesh. If None, the constructor
                        assumes it is being used as a copy constructor, and
                        kwargs are passed to the superclass constructor.
        """
        if not pv_mesh:
            # for copy constructor
            # TODO: maybe better way?
            super().__init__(**kwargs)
        else:
            super().__init__(pv_mesh, **kwargs)

    @classmethod
    def read(cls, filepath: str | Path) -> Mesh:
        """
        Initialize a Mesh object

            :param filepath:            Path to the mesh file.

            :returns:                   A Mesh object
        """

        mesh = cls(pv.read(filepath))
        mesh.filepath = Path(filepath).with_suffix(".vtu")
        return mesh

    @staticmethod
    def max_dim(mesh: pv.DataSet) -> int:
        return int(np.max([cell.dimension for cell in mesh.cell]))

    def reindex_material_ids(self) -> None:
        unique_mat_ids = np.unique(self["MaterialIDs"])
        id_map = dict(
            zip(*np.unique(unique_mat_ids, return_inverse=True), strict=True)
        )
        self["MaterialIDs"] = np.int32(
            list(map(id_map.get, self["MaterialIDs"]))
        )
        return

    if find_spec("ifm") is not None:

        @classmethod
        def read_feflow(
            cls,
            feflow_file: Path | str,
        ) -> Mesh:
            """
            Initialize a Mesh object read from a FEFLOW file. This mesh stores all model specific
            information such as boundary conditions or material parameters.

                :param feflow_file:         Path to the feflow file.

                :returns:                   A Mesh object
            """
            import ifm_contrib as ifm

            from ogstools.feflowlib._feflowlib import convert_properties_mesh

            doc = ifm.loadDocument(str(feflow_file))
            return cls(convert_properties_mesh(doc))

    @classmethod
    def from_simulator(
        cls,
        simulation: Any,
        name: str,
        node_properties: list[str] | None = None,
        cell_properties: list[str] | None = None,
        field_properties: list[str] | None = None,
    ) -> Mesh:
        """
        Constructs a pyvista mesh from a running simulation. It always contains points (geometry) and cells (topology)
        and optionally the given node-based or cell-based properties
        Properties must be added afterwards

            :param simulator:       Initialized and not finalized simulator object
            :param name:            Name of the submesh (e.g. domain, left, ... )
            :param node_properties: Given properties will be added to the mesh
                                    None or [] -> no properties will be added

            :param cell_properties: Given properties will be added to the mesh
                                    None or [] -> no properties will be added

            :returns:               A Mesh (Pyvista Unstructured Grid) object

        """
        from ogs import OGSMesh
        from ogs.OGSMesh import MeshItemType
        from vtk.util import numpy_support

        from ogstools.meshlib.vtk_pyvista import construct_cells

        in_situ_mesh: OGSMesh = simulation.mesh(name)
        points_flat = in_situ_mesh.points()
        points = np.array(points_flat, dtype=float).reshape(-1, 3)

        cells_and_types = in_situ_mesh.cells()
        cells = construct_cells(cells_and_types[0], cells_and_types[1])

        pv_mesh = pv.UnstructuredGrid(cells, cells_and_types[1], points)

        properties_in_mesh = in_situ_mesh.data_array_names()

        node_properties_in_mesh = [
            prop
            for prop in properties_in_mesh
            if in_situ_mesh.mesh_item_type(prop) == MeshItemType.Node
        ]
        # 1 Edge, 2 Face not supported yet
        cell_properties_in_mesh = [
            prop
            for prop in properties_in_mesh
            if in_situ_mesh.mesh_item_type(prop) == MeshItemType.Cell
        ]
        field_properties_in_mesh = [
            prop
            for prop in properties_in_mesh
            if in_situ_mesh.mesh_item_type(prop)
            == MeshItemType.IntegrationPoint
        ]

        node_properties = node_properties or node_properties_in_mesh
        cell_properties = cell_properties or cell_properties_in_mesh
        field_properties = field_properties or field_properties_in_mesh

        for node_property_name in node_properties:
            data_type = "double"  # in_situ_mesh.
            arr = in_situ_mesh.data_array(node_property_name, data_type)
            vtk_arr = numpy_support.numpy_to_vtk(arr, deep=0)
            pv_mesh.point_data.set_array(vtk_arr, node_property_name)
            # print('shape of {node_property_name} is ')
            # print(in_situ_mesh.data_array("double", MeshItemType.Node, node_property_name, 1).shape)
        for cell_property_name in cell_properties:
            data_type = "int"
            pv_mesh.cell_data[cell_property_name] = in_situ_mesh.data_array(
                cell_property_name, data_type
            )

        for field_property_name in field_properties_in_mesh:
            data_type = "char"
            pv_mesh.field_data[field_property_name] = in_situ_mesh.data_array(
                field_property_name, data_type
            )

        return cls(pv_mesh)
