# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

from ogstools import plot
from ogstools._internal import copy_method_signature
from ogstools.definitions import SPATIAL_UNITS_KEY
from ogstools.plot import lineplots

from . import data_processing, geo, ip_mesh
from .shape_meshing import shapefile_meshing


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

    @copy_method_signature(lineplots.linesample)
    def plot_linesample(self, *args: Any, **kwargs: Any) -> Any:
        return lineplots.linesample(self, *args, **kwargs)

    @copy_method_signature(lineplots.linesample_contourf)
    def plot_linesample_contourf(self, *args: Any, **kwargs: Any) -> Any:
        return lineplots.linesample_contourf(self, *args, **kwargs)

    @copy_method_signature(ip_mesh.to_ip_mesh)
    def to_ip_mesh(self, *args: Any, **kwargs: Any) -> Any:
        return Mesh(ip_mesh.to_ip_mesh(self, *args, **kwargs))

    @copy_method_signature(ip_mesh.to_ip_point_cloud)
    def to_ip_point_cloud(self, *args: Any, **kwargs: Any) -> Any:
        return Mesh(ip_mesh.to_ip_point_cloud(self, *args, **kwargs))

    # pylint: enable=C0116

    def __init__(
        self,
        pv_mesh: pv.UnstructuredGrid | None = None,
        spatial_unit: str = "m",
        spatial_output_unit: str = "m",
        **kwargs: dict,
    ):
        """
        Initialize a Mesh object

            :param pv_mesh:     Underlying pyvista mesh.
            :param data_length_unit:    Length unit of the mesh data.
            :param output_length_unit:  Length unit in plots.
        """
        if not pv_mesh:
            # for copy constructor
            # TODO: maybe better way?
            super().__init__(**kwargs)
        else:
            super().__init__(pv_mesh, **kwargs)
        self.field_data[SPATIAL_UNITS_KEY] = np.asarray(
            [ord(char) for char in f"{spatial_unit},{spatial_output_unit}"]
        )

    @classmethod
    def from_points_cells(cls, points: np.ndarray, cells: np.ndarray) -> "Mesh":
        """
        Create a PyVista UnstructuredGrid from points and cells. Pyvista requires point, cell and celltype
        array to set up a unstructured grid. This function creates the cell array and cell type array in the
        correct structure automatically. So, simply an array of points with coordinates and an array of cells
        with indices of points are needed. For more information see:
        https://docs.pyvista.org/version/stable/api/core/_autosummary/pyvista.unstructuredgrid

        :param points: An array of shape (n_points, 3) containing the coordinates of each point.
        :param cells: An array of lists, where each inner list represents a cell and contains the indices of its points.

        :return: A Mesh object
        """
        # Convert points to numpy array if it's not already
        points = np.asarray(points)
        cells = cells.astype(np.int32, copy=False)
        # Append the zeros column to the points, if they refer to 2D data.
        if points.shape[1] == 2:
            zeros_column = np.zeros((points.shape[0], 1), dtype=int)
            points = np.column_stack((points, zeros_column))
        # Prepare the cell array
        cell_array = np.concatenate([np.r_[len(cell), cell] for cell in cells])
        assert (
            len(np.unique([len(cell) for cell in cells])) == 1
        ), "All cells must be of the same type. Hence, have the same number of points. If not use pyvista.UnstructuredGrid directly."
        # choose the celltype:
        if cell_array[0] == 4:
            celltype = pv.CellType.TETRA
        elif cell_array[0] == 8:
            celltype = pv.CellType.HEXAHEDRON
        elif cell_array[0] == 6:
            celltype = pv.CellType.WEDGE
        elif cell_array[0] == 5:
            celltype = pv.CellType.PYRAMID
        elif cell_array[0] == 3:
            celltype = pv.CellType.TRIANGLE
        elif cell_array[0] == 2:
            celltype = pv.CellType.LINE
        else:
            celltype = pv.CellType.CONVEX_POINT_SET

        # Create the cell types array
        cell_types = np.full(len(cells), celltype)

        # Return the UnstructuredGrid
        return cls(
            pv.UnstructuredGrid(np.asarray(cell_array), cell_types, points)
        )

    @classmethod
    def read(
        cls,
        filepath: str | Path,
        spatial_unit: str = "m",
        spatial_output_unit: str = "m",
    ) -> "Mesh":
        """
        Initialize a Mesh object

            :param filepath:            Path to the mesh or shapefile file.
            :param data_length_unit:    Spatial data unit of the mesh.
            :param output_length_unit:  Spatial output unit of the mesh.

            :returns:                   A Mesh object
        """
        if Path(filepath).suffix == ".shp":
            points_cells = shapefile_meshing(filepath)
            mesh = cls.from_points_cells(points_cells[0], points_cells[1])
        else:
            mesh = cls(pv.read(filepath))
        mesh.filepath = Path(filepath).with_suffix(".vtu")
        mesh.field_data[SPATIAL_UNITS_KEY] = np.asarray(
            [ord(char) for char in f"{spatial_unit},{spatial_output_unit}"]
        )
        return mesh
