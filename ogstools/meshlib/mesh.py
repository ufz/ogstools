# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

import ogstools.meshlib as ml
from ogstools import plot
from ogstools._internal import copy_method_signature
from ogstools.definitions import SPATIAL_UNITS_KEY
from ogstools.plot import lineplots

from . import data_processing, geo, ip_mesh, shape_meshing


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

    def to_ip_mesh(self) -> "Mesh":
        return Mesh(ip_mesh.to_ip_mesh(self))

    def to_ip_point_cloud(self) -> "Mesh":
        return Mesh(ip_mesh.to_ip_point_cloud(self))

    to_ip_mesh.__doc__ = ip_mesh.to_ip_mesh.__doc__
    to_ip_point_cloud.__doc__ = ip_mesh.to_ip_point_cloud.__doc__

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
            mesh = cls(ml.read_shape(filepath))
        else:
            mesh = cls(pv.read(filepath))

        mesh.filepath = Path(filepath).with_suffix(".vtu")
        mesh.field_data[SPATIAL_UNITS_KEY] = np.asarray(
            [ord(char) for char in f"{spatial_unit},{spatial_output_unit}"]
        )
        return mesh

    @classmethod
    @copy_method_signature(shape_meshing.read_shape)
    def read_shape(
        cls,
        shapefile: str | Path,
        simplify: bool = False,
        mesh_generator: str = "triangle",
        cellsize: int | None = None,
    ) -> "Mesh":
        return cls(
            shape_meshing.read_shape(
                shapefile, simplify, mesh_generator, cellsize
            )
        )
