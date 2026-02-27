# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pyvista as pv


def from_simulator(
    simulation: Any,
    name: str,
    node_properties: Sequence[str] | None = None,
    cell_properties: Sequence[str] | None = None,
    field_properties: Sequence[str] | None = None,
) -> pv.UnstructuredGrid:
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

    from .vtk_pyvista import construct_cells

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
        if in_situ_mesh.mesh_item_type(prop) == MeshItemType.IntegrationPoint
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
        uint64 = cell_property_name in ["bulk_elem_ids", "bulk_element_ids"]
        data_type = "std::size_t" if uint64 else "int"
        pv_mesh.cell_data[cell_property_name] = in_situ_mesh.data_array(
            cell_property_name, data_type
        )

    for field_property_name in field_properties_in_mesh:
        data_type = "char"
        pv_mesh.field_data[field_property_name] = in_situ_mesh.data_array(
            field_property_name, data_type
        )

    return pv_mesh
