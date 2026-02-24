# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pyvista as pv


class RegionSet:
    """
    A class representing a set of regions composed of subsets, each identified by MaterialID.

    The RegionSet class represents a collection of regions, where each region is composed of
    subsets. Each subset within a region is uniquely identified by "MaterialID".
    """

    def __init__(self, input: Path | pv.UnstructuredGrid):
        if type(input) is Path:
            self.filename = input
            self.mesh = None
        else:
            self.filename = Path(tempfile.mkstemp(".vtu", "region_set")[1])
            self.mesh = input

    def box_boundaries(self) -> tuple[pv.UnstructuredGrid, ...]:
        """
        Retrieve the boundaries of the mesh in local coordinate system (u, v, w).

        This function extracts the boundaries of the mesh along the u, v, and w directions
        of the local coordinate system. The u-axis corresponds to the x-coordinate, the v-axis
        corresponds to the y-coordinate, and the w-axis corresponds to the z-coordinate.

        :returns:   A tuple (u_min, u_max, v_min, v_max, w_min, w_max)
                    representing the boundaries of the mesh in the local
                    coordinate system.

        notes:
            - If the original mesh was created from boundaries, this function returns the original boundaries.
            - The returned boundaries adhere to the definition of [Pyvista Box](https://docs.pyvista.org/version/stable/api/utilities/_autosummary/pyvista.Box.html).

        example:
            mesh = ...
            u_min, u_max, v_min, v_max, w_min, w_max = mesh.box_boundaries()
        """
        assert isinstance(self.mesh, pv.UnstructuredGrid)
        surface = self.mesh.extract_surface(algorithm="dataset_surface")
        u_max = to_boundary(surface, lambda normals: normals[:, 0] > 0.5)
        u_min = to_boundary(surface, lambda normals: normals[:, 0] < -0.5)
        v_max = to_boundary(surface, lambda normals: normals[:, 1] > 0.5)
        v_min = to_boundary(surface, lambda normals: normals[:, 1] < -0.5)
        w_max = to_boundary(surface, lambda normals: normals[:, 2] > 0.5)
        w_min = to_boundary(surface, lambda normals: normals[:, 2] < -0.5)

        return (u_min, u_max, v_min, v_max, w_min, w_max)


def to_boundary(
    surface_mesh: pv.PolyData,
    filter_condition: Callable[[np.ndarray], np.ndarray],
) -> pv.UnstructuredGrid:
    """
    Extract cells from a surface mesh that meet a filter condition for normals.

    This function takes a surface mesh represented by a `pv.PolyData` object and extracts
    cells that match a specified filter condition based on the normals of the mesh.

    :param surface_mesh:        The input surface mesh.
    :param filter_condition:    A callable filter condition that takes an array
                                of normals as input and returns an array
                                indicating whether the condition is met.

    :returns: A mesh containing only the cells that meet the filter condition.

    example:
        surface_mesh = ...
        specific_cells = to_boundary(surface_mesh, lambda normals: [n[2] > 0.5 for n in normals])
    """

    surface_mesh = surface_mesh.compute_normals(
        cell_normals=True, point_normals=True
    )

    ids = np.arange(surface_mesh.n_cells)[
        filter_condition(surface_mesh["Normals"])
    ]

    specific_cells = surface_mesh.extract_cells(ids)
    specific_cells.rename_array("vtkOriginalPointIds", "BULK_NODE_ID")
    specific_cells.rename_array("vtkOriginalCellIds", "BULK_ELEMENT_ID")
    specific_cells.cell_data.remove("Normals")
    return specific_cells
