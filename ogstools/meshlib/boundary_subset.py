import tempfile
from pathlib import Path
from typing import Union

import numpy as np
import pyvista as pv
from ogs import cli


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

    def __init__(
        self,
        input: Union[Path, pv.DataSet],
        material_id: int,
    ):
        """Initialize a surface mesh. Either from pyvista or from a file."""
        self._material_id = material_id

        if isinstance(input, Path):
            self.filename = input
            if self.filename.exists() is False:
                print(self.filename, "does not exist.")
                raise ValueError
            self.mesh = pv.get_reader(self.filename).read()
        elif isinstance(input, pv.DataSet):
            self.mesh = input
            self.filename = Path(tempfile.mkstemp(".vtu", "surface")[1])
            pv.save_meshio(self.filename, self.mesh, file_format="vtu")
        else:
            raise ValueError

        self.mesh.cell_data["MaterialIDs"] = (
            np.ones(self.mesh.n_cells) * self.material_id
        ).astype(np.int32)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def create_raster_file(self, resolution: float) -> Path:
        """
        Creates a raster file specific to resolution. If outfile is not specified, the extension is replaced by .asc

        :returns the path and filename of the created file (.asc)
        """
        outfile = Path(tempfile.mkstemp(".asc", self.filename.stem)[1])

        ret = cli.Mesh2Raster(
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
):
    """
    Generate a 2D Gaussian-like surface using the provided parameters.

    This method computes a 2D Gaussian-like surface by sampling the given bound and parameters.

    Args:
        bound2D (tuple): Tuple of boundary coordinates (x_min, x_max, y_min, y_max).
        amplitude (float): Amplitude or peak value of the Gaussian curve.
        spread (float): Scaling factor that controls the spread or width of the Gaussian curve.
        height_offset (float): Constant offset added to elevate the entire surface.
        n (int): Number of points in each dimension for sampling.

    Returns:
        pyvista.PolyData: A PyVista PolyData object representing the generated surface.

    Note:
        - The larger `amplitude`, the taller the peak of the surface.
        - The larger `spread`, the wider and flatter the surface.
        - `height_offset` shifts the entire surface vertically.

    Example:
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
