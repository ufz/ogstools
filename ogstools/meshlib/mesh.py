# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from pathlib import Path
from tempfile import mkdtemp
from typing import Any

import numpy as np
import ogs
import pyvista as pv

from ogstools import plot
from ogstools._internal import copy_method_signature
from ogstools.definitions import SPATIAL_UNITS_KEY
from ogstools.plot import lineplots

from . import data_processing, geo
from .ip_mesh import tessellate


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

            :param filepath:            Path to the vtu file.
            :param data_length_unit:    Spatial data unit of the mesh.
            :param output_length_unit:  Spatial output unit of the mesh.

            :returns:                   A Mesh object
        """
        mesh = cls(pv.XMLUnstructuredGridReader(str(filepath)).read())
        mesh.filepath = Path(filepath)
        mesh.field_data[SPATIAL_UNITS_KEY] = np.asarray(
            [ord(char) for char in f"{spatial_unit},{spatial_output_unit}"]
        )
        return mesh

    def to_ip_point_cloud(self) -> "Mesh":
        "Convert integration point data to a pyvista point cloud."
        if self.filepath is None:
            filepath = Path(mkdtemp()) / "mesh.vtu"
            self.save(filepath)
        else:
            filepath = self.filepath
        ip_mesh_path = filepath.parent / "ip_mesh.vtu"
        ogs.cli.ipDataToPointCloud(i=str(filepath), o=str(ip_mesh_path))
        return Mesh.read(filepath=ip_mesh_path)

    def to_ip_mesh(self) -> "Mesh":
        "Create a mesh with cells centered around integration points."
        meta = self.field_data["IntegrationPointMetaData"]
        meta_str = "".join([chr(val) for val in meta])
        integration_order = int(
            meta_str.split('"integration_order":')[1].split(",")[0]
        )
        ip_mesh = self.to_ip_point_cloud()

        cell_types = list({cell.type for cell in self.cell})
        new_meshes: list[pv.PolyData] = []
        for cell_type in cell_types:
            mesh = self.extract_cells_by_type(cell_type)
            new_meshes += [tessellate(mesh, cell_type, integration_order)]
        new_mesh = new_meshes[0]
        new_mesh.field_data[SPATIAL_UNITS_KEY] = self.field_data[
            SPATIAL_UNITS_KEY
        ]
        for mesh in new_meshes[1:]:
            new_mesh = new_mesh.merge(mesh)
        new_mesh = new_mesh.clean()

        ordering = new_mesh.find_containing_cell(ip_mesh.points)
        ip_data = {
            k: v[np.argsort(ordering)] for k, v in ip_mesh.point_data.items()
        }
        new_mesh.cell_data.update(ip_data)

        return Mesh(new_mesh)
