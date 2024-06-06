import numpy as np
import pytest
import pyvista as pv

from ogstools.definitions import EXAMPLES_DIR
from ogstools.meshlib.boundary_subset import Surface

meshpath = EXAMPLES_DIR / "meshlib"


class TestSurface:
    """
    Collects all steps that are necessary to create a prism mesh
    """

    def testsurface_from_file_valid(self):
        """
        OK if file can be loaded - if not it raises an exception
        """
        filename = meshpath / "mesh1/surface_data/00_KB.vtu"
        s = Surface(filename, 0)  # checks
        assert s.filename == filename

    def testurface_from_pyvista(self):
        """
        test if file can be loaded - if not it raises an exception
        """
        x = np.arange(-10, 10, 0.25)
        y = np.arange(-10, 10, 0.25)
        z1 = 0
        X1, Y1, Z1 = np.meshgrid(x, y, z1)
        surface_mesh = pv.StructuredGrid(X1, Y1, Z1)

        s = Surface(surface_mesh, 0)
        assert s.mesh.GetNumberOfPoints() > 0

        pv.save_meshio(filename=s.filename, mesh=s.mesh)
        s2 = Surface(s.filename, material_id=2)
        assert s2.mesh.GetNumberOfPoints() > 0

    def testsurface_from_file_invalid(self):
        """
        OK if file can be loaded - if not it raises an exception
        """
        with pytest.raises(
            ValueError, match=r".*notexisting.vtu does not exist."
        ):
            Surface(
                meshpath / "mesh1/surface_data/notexisting.vtu",
                0,
            )

    def testsurface_to_raster(self):
        s1 = Surface(
            meshpath / "mesh1/surface_data/00_KB.vtu",
            0,
        )
        outfile = s1.create_raster_file(10)

        with outfile.open() as f:
            lines = f.readlines()
            assert "cellsize" in lines[4]
