import subprocess

import ogstools.meshlib as ml
from ogstools.examples import test_shapefile


def test_cli():
    subprocess.run(["shp2msh", "--help"], check=True)


class TestMeshing:
    def test_meshing(self):
        pyvista_mesh = ml.read_shape(
            test_shapefile
        )  # simplify_false, triangle_true
        assert pyvista_mesh.n_points == 233465
        assert pyvista_mesh.n_cells == 344431

    # Same for simplified mesh.
    def test_simple_meshing(self):
        pyvista_mesh = ml.read_shape(test_shapefile, simplify=True)

        assert pyvista_mesh.n_points == 6413
        assert pyvista_mesh.n_cells == 10291

    def test_gmsh_meshing(self):
        pyvista_mesh = ml.read_shape(
            test_shapefile, simplify=True, triangle=False
        )

        assert pyvista_mesh.n_points == 12618
        assert pyvista_mesh.n_cells == 22915

    def test_meshclass_reading(self):
        ogs_mesh = ml.Mesh.read(test_shapefile)
        pyvista_mesh = ml.read_shape(test_shapefile)
        assert pyvista_mesh.n_points == ogs_mesh.n_points
        assert pyvista_mesh.n_cells == ogs_mesh.n_cells
