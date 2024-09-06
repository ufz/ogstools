import subprocess
from tempfile import mkdtemp

import numpy as np

from ogstools.examples import circle_shapefile, test_shapefile
from ogstools.meshlib import Mesh, read_shape


def test_cli():
    subprocess.run(["shp2msh", "--help"], check=True)
    subprocess.run(
        ["shp2msh", "-i", str(test_shapefile), "-o", mkdtemp(".vtu")],
        check=True,
    )


class TestShapeFileMeshing:
    def test_default(self):
        # simplify_false, mesh_generator_triangle
        pyvista_mesh = read_shape(test_shapefile)

        assert np.isclose(pyvista_mesh.n_points, 233465, rtol=0.05)
        assert np.isclose(pyvista_mesh.n_cells, 344431, rtol=0.05)

    # Same for simplified mesh.
    def test_simplify(self):
        mesh_simplify = read_shape(test_shapefile, simplify=True)
        mesh_orignal = read_shape(test_shapefile, simplify=False)

        assert mesh_simplify.n_points > 0
        assert mesh_simplify.n_cells > 0

        assert mesh_simplify.n_points < mesh_orignal.n_points
        assert mesh_simplify.n_cells < mesh_orignal.n_cells

        mesh_simplify_finer = read_shape(
            test_shapefile, simplify=True, cellsize=2000
        )
        mesh_simplify_coarser = read_shape(
            test_shapefile, simplify=True, cellsize=10000
        )

        assert mesh_simplify_coarser.n_cells < mesh_simplify.n_cells
        assert mesh_simplify.n_cells < mesh_simplify_finer.n_cells

        assert mesh_simplify_coarser.n_points < mesh_simplify.n_points
        assert mesh_simplify.n_points < mesh_simplify_finer.n_points

    def test_simplify_gmsh(self):
        mesh_simplify = read_shape(
            circle_shapefile, simplify=True, mesh_generator="gmsh"
        )
        mesh_original = read_shape(
            circle_shapefile, simplify=False, mesh_generator="gmsh"
        )

        assert mesh_simplify.n_points < mesh_original.n_points
        assert mesh_simplify.n_cells < mesh_original.n_cells

    def test_meshclass_reading(self):
        ogs_mesh = Mesh.read(circle_shapefile)
        ogs_mesh_special = Mesh.read_shape(circle_shapefile)
        pyvista_mesh = read_shape(circle_shapefile)
        assert pyvista_mesh.n_points == ogs_mesh.n_points
        assert pyvista_mesh.n_cells == ogs_mesh.n_cells
        assert pyvista_mesh.n_points == ogs_mesh_special.n_points
        assert pyvista_mesh.n_cells == ogs_mesh_special.n_cells
