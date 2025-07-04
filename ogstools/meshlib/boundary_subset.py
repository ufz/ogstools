# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

import tempfile
from pathlib import Path

import numpy as np
import pyvista as pv
from typeguard import typechecked


class Surface:
    """
    A surface is a sub group of a polygon mesh (2D). A surface is not closed and therefore does not represent a volume.
    (Geological) layers (stratigraphic units) can be defined by an upper and lower surface.
    By convention, properties (material_id and resolution ), actually associated to the stratigraphic unit layer,
    are given together with the lower boundary (class Surface) of a stratigraphic unit (class Layer).
    """

    @property
    def material_id(self) -> int:
        return self._material_id

    @typechecked
    def __init__(self, input: Path | pv.DataObject, material_id: int):
        """Initialize a surface mesh. Either from pyvista or from a file."""
        self._material_id = material_id

        if isinstance(input, Path):
            self.filename = input
            if self.filename.exists() is False:
                msg = f"{self.filename} does not exist."
                raise ValueError(msg)
            self.mesh = pv.get_reader(self.filename).read()
        elif isinstance(input, pv.DataObject):
            self.mesh = input
            self.filename = Path(tempfile.mkstemp(".vtu", "surface")[1])
            pv.save_meshio(self.filename, self.mesh, file_format="vtu")

        self.mesh.cell_data["MaterialIDs"] = (
            np.ones(self.mesh.n_cells) * self.material_id
        ).astype(np.int32)

    def __eq__(self, other: object) -> bool:
        return self.__dict__ == other.__dict__

    def create_raster_file(self, resolution: float) -> Path:
        """
        Creates a raster file specific to resolution. If outfile is not specified, the extension is replaced by .asc

        :returns the path and filename of the created file (.asc)
        """
        outfile = Path(tempfile.mkstemp(".asc", self.filename.stem)[1])

        from ogstools._find_ogs import cli

        ret = cli().Mesh2Raster(  # type: ignore[union-attr]
            i=str(self.filename), o=str(outfile), c=resolution
        )

        if ret:
            msg = (
                "Mesh2Raster -i ",
                str(self.filename),
                " -o ",
                str(outfile),
                " -c ",
                str(resolution),
            )
            raise ValueError(msg)
        return outfile


def Gaussian2D(
    bound2D: tuple,
    amplitude: float,
    spread: float,
    height_offset: float,
    n: int,
) -> pv.DataSet:
    """
    Generate a 2D Gaussian-like surface using the provided parameters.

    This method computes a 2D Gaussian-like surface by sampling the given bound and parameters.

    :param bound2D: Tuple of boundary coordinates (x_min, x_max, y_min, y_max).
    :param amplitude: Amplitude or peak value of the Gaussian curve.
    :param spread: Scaling factor that controls the spread or width of the Gaussian curve.
    :param height_offset: Constant offset added to elevate the entire surface.
    :param n: Number of points in each dimension for sampling.

    :returns:
        pyvista.PolyData: A PyVista PolyData object representing the generated surface.

    note:
        - The larger `amplitude`, the taller the peak of the surface.
        - The larger `spread`, the wider and flatter the surface.
        - `height_offset` shifts the entire surface vertically.

    example:
        Generating a 2D Gaussian-like surface:
        ```
        bound = (-1.0, 1.0, -1.0, 1.0)
        amplitude = 1.0
        spread = 0.5
        height_offset = 0.0
        n = 100
        surface = MyClass.Gaussian2D(bound, amplitude, spread, height_offset, n)
        ```
    """
    x = np.linspace(bound2D[0], bound2D[1], num=n)
    y = np.linspace(bound2D[2], bound2D[3], num=n)
    xx, yy = np.meshgrid(x, y)

    zz = (
        amplitude * np.exp(-0.5 * ((xx / spread) ** 2.0 + (yy / spread) ** 2.0))
        + height_offset
    )

    points = np.c_[xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)]
    points[0:5, :]
    cloud = pv.PolyData(points)
    return cloud.delaunay_2d()
