import unittest

import numpy as np
import pyvista as pv

from ogstools.definitions import EXAMPLES_DIR
from ogstools.meshlib.boundary_subset import Surface

meshpath = EXAMPLES_DIR / "meshlib"


class SurfaceTest(unittest.TestCase):
    """
    Collects all steps that are necessary to create a prism mesh
    """

    def testsurfaceFromFileValid(self):
        """
        OK if file can be loaded - if not it raises an exception
        """
        filename = meshpath / "mesh1/surface_data/00_KB.vtu"
        s = Surface(filename, 0)  # checks
        self.assertEqual(s.filename, filename)

    def testurfaceFromPyvista(self):
        """
        test if file can be loaded - if not it raises an exception
        """
        x = np.arange(-10, 10, 0.25)
        y = np.arange(-10, 10, 0.25)
        z1 = 0
        X1, Y1, Z1 = np.meshgrid(x, y, z1)
        surface_mesh = pv.StructuredGrid(X1, Y1, Z1)

        s = Surface(surface_mesh, 0)
        self.assertGreater(s.mesh.GetNumberOfPoints(), 0)

        pv.save_meshio(filename=s.filename, mesh=s.mesh)
        s2 = Surface(s.filename, material_id=2)
        self.assertGreater(s2.mesh.GetNumberOfPoints(), 0)

    def testsurfaceFromFileInvalid(self):
        """
        OK if file can be loaded - if not it raises an exception
        """
        self.assertRaises(
            Exception,
            Surface.__init__,
            meshpath / "mesh1/surface_data/notexisting.vtu",
            0,
            0,
        )

    def testsurfaceToRaster(self):
        s1 = Surface(
            meshpath / "mesh1/surface_data/00_KB.vtu",
            0,
        )
        outfile = s1.create_raster_file(10)

        with outfile.open() as f:
            lines = f.readlines()
            self.assertTrue("cellsize" in lines[4])
