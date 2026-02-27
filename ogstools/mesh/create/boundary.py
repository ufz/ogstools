# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import ceil
from pathlib import Path

import numpy as np
import pyvista as pv

from ogstools import mesh

from .boundary_subset import Surface


class Boundary(ABC):
    """
    Abstract base class representing a boundary within a mesh.

    A boundary refers to the set of edges or faces that defines the delineation between
    the interior region and exterior regions of a mesh.
    In a 2D mesh, it is formed by a closed collection of line segments (1D).
    In a 3D mesh, it is formed by a closed collection of faces (2D).
    """

    @abstractmethod
    def dim(self) -> int:
        """
        Get the dimension of the boundary.

        :returns:   The dimension of the boundary. For example, the dimension
                    of a boundary of a cube (3D) is 2.
        """
        return 0


@dataclass(frozen=True)
class Layer(Boundary):
    top: Surface
    bottom: Surface
    material_id: int = 0
    num_subdivisions: int = 0
    """
    Class representing a geological layer with top and bottom surfaces.

    A geological layer is a distinct unit of rock or sediment that has unique properties
    and characteristics, associated by the material_id. It is often bounded by two surfaces:
    the top surface and the bottom surface. These surfaces delineate the spatial extent
    of the layer in the GIS system.
    """

    def __post_init__(self) -> None:
        if not self.material_id:
            object.__setattr__(self, "material_id", self.bottom._material_id)

    def create_raster(self, resolution: float) -> list[Path]:
        """
        Create raster representations for the layer.

        For each surface, including intermediate surfaces (num_of_subdivisions > 0),
        this method generates .asc files.

        :param resolution: The resolution for raster creation.

        :returns: A list of filenames to .asc raster files.
        """

        top_raster = self.top.create_raster_file(resolution)
        bottom_raster = self.bottom.create_raster_file(resolution)
        if self.num_subdivisions < 1:
            return [top_raster, bottom_raster]

        outfile = Path(tempfile.mkstemp(".asc")[1])

        from ogstools._find_ogs import cli

        ret = cli().createIntermediateRasters(  # type: ignore[union-attr]
            file1=top_raster,
            file2=bottom_raster,
            o=outfile,
            n=self.num_subdivisions,
        )
        if ret:
            raise ValueError
        new_names = [
            outfile.with_name(outfile.stem + str(i) + ".asc")
            for i in range(self.num_subdivisions)
        ]

        rasters = [top_raster]
        rasters.extend(new_names)
        rasters.append(bottom_raster)
        return rasters

    def dim(self) -> int:
        return 3

    def to_simplified_mesh(
        self, resolution: float, rank: int, bounds: list[float]
    ) -> pv.UnstructuredGrid:
        """Convert a geological layer to a simplified mesh.

        This function converts a geological `layer` into a simplified mesh using the specified
        `resolution`, `rank`, and bounding `bounds`.


        :param resolution: The desired spatial resolution of the mesh in units of the geological structure.
                           Be cautious with very high resolutions, as they may lead to distorted or incomplete meshes,
                           making them unsuitable for further analysis.
        :param rank: The rank of the mesh (2 for 2D, 3 for 3D). The mesh dimensionality must be consistent
                     with the provided `bounds`.
        :param bounds: A list of bounding values [min_x, max_x, min_y, max_y] for 2D mesh or
                       [min_x, max_x, min_y, max_y, min_z, max_z] for 3D mesh.
                       The `bounds` define the region of the geological structure that will be meshed.

        :returns: A simplified unstructured grid mesh representing the layer.

        raises:
            Exception: If the specified `rank` is not 2 or 3, indicating an invalid mesh dimensionality.

        example:
            layer = ...
            resolution = 1.5  # Example resolution in geological units
            rank = 2  # Mesh will be 2D
            bounds = [0, 10, 0, 10]  # Bounding box [min_x, max_x, min_y, max_y]
            mesh = layer.to_simplified_mesh(layer, resolution, rank, bounds)
        """

        axis0_range = np.arange(bounds[0], bounds[1], resolution)
        axis1_range = np.arange(bounds[2], bounds[3], resolution)

        heights = np.linspace(
            self.bottom.mesh.points[:, 2].mean(),
            self.top.mesh.points[:, 2].mean(),
            num=self.num_subdivisions + 2,
        )

        if rank == 2:
            AXIS1, AXIS2 = np.meshgrid(axis1_range, heights)
            AXIS0 = 0 * (AXIS1 + AXIS2)
            # resulting mesh is in xy - plane
            X, Y, Z = (AXIS1, AXIS2, AXIS0)
        elif rank == 3:
            X, Y, Z = np.meshgrid(axis0_range, axis1_range, heights)
        else:
            msg = "rank is {rank}, but must be 2 or 3."
            raise Exception(msg)

        grid = pv.StructuredGrid(X, Y, Z)
        material = self.material_id
        grid.cell_data["MaterialIDs"] = (
            np.ones(grid.n_cells) * material
        ).astype(np.int32)
        return grid


@dataclass
class LocationFrame:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def as_gml(self, filename: Path) -> None:
        """
        Generate GML representation of the location frame.

        :param filename: The filename to save the GML representation to.

        :returns: None
        """
        from ogstools._find_ogs import cli

        cli().generateGeometry(  # type: ignore[union-attr]
            geometry_name="SceneRectangle",
            x0=str(self.xmin),
            x1=str(self.xmax),
            y0=str(self.ymin),
            y1=str(self.ymax),
            z0="0",
            z1="0",
            nx="0",
            nx1="0",
            ny="0",
            ny1="0",
            nz="0",
            nz1="0",
            polyline_name="bounding_rectangle",
            o=str(filename),
        )


@dataclass
class Raster:
    """
    Class representing a raster representation of a location frame.

    This class provides methods to create and save a raster representation based on a
    specified location frame and resolution.
    """

    # 3D - 4 Points
    frame: LocationFrame
    resolution: float

    def as_vtu(self, outfilevtu: Path) -> Path:
        """
        Create and save a raster representation as a VTK unstructured grid.

        :param outfilevtu: The path to save the VTK unstructured grid.

        :returns: The path to the saved VTK unstructured grid representation.
        """
        centroid = (
            (self.frame.xmax + self.frame.xmin) / 2,
            (self.frame.ymax + self.frame.ymin) / 2,
            0,
        )
        i_size = self.frame.xmax - self.frame.xmin
        j_size = self.frame.ymax - self.frame.ymin

        abs_resolution = ceil(i_size / self.resolution)

        plane = pv.Plane(
            center=centroid,
            i_size=i_size,
            j_size=j_size,
            i_resolution=abs_resolution,
            j_resolution=abs_resolution,
        )

        pt = plane.triangulate()
        mesh.save(pt, outfilevtu)
        return outfilevtu
