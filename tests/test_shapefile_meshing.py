import subprocess

import ogstools.meshlib as ml
from ogstools.examples import test_shapefile


def test_cli():
    subprocess.run(["shp2msh", "--help"], check=True)


class TestShapeFileMeshing:
    def test_default(self):
        pyvista_mesh = ml.read_shape(
            test_shapefile
        )  # simplify_false, triangle_true
        assert pyvista_mesh.n_points == 233465
        assert pyvista_mesh.n_cells == 344431

    # Same for simplified mesh.
    def test_simplify(self):
        mesh_simplify = ml.read_shape(test_shapefile, simplify=True)
        mesh_orignal = ml.read_shape(test_shapefile, simplify=False)

        assert mesh_simplify.n_points > 0
        assert mesh_simplify.n_cells > 0

        assert mesh_simplify.n_points < mesh_orignal.n_points
        assert mesh_simplify.n_cells < mesh_orignal.n_cells

        mesh_simplify_finer = ml.read_shape(
            test_shapefile, simplify=True, cellsize=20000
        )
        mesh_simplify_coarser = ml.read_shape(
            test_shapefile, simplify=True, cellsize=10000
        )

        assert mesh_simplify_coarser.n_cells < mesh_simplify.n_cells
        assert mesh_simplify.n_cells < mesh_simplify_finer.n_cells

        assert mesh_simplify_coarser.n_points < mesh_simplify.n_points
        assert mesh_simplify.n_points < mesh_simplify_finer.n_points

    def test_simpify_gmsh(self):
        mesh_simplify = ml.read_shape(
            test_shapefile, simplify=True, triangle=False
        )

        mesh_complex = ml.read_shape(
            test_shapefile, simplify=False, triangle=False
        )

        assert mesh_simplify.n_points == 12618
        assert mesh_simplify.n_cells == 22915
        assert mesh_simplify.n_points < mesh_complex.n_points
        assert mesh_simplify.n_cells < mesh_complex.n_cells

    def test_meshclass_reading(self):
        ogs_mesh = ml.Mesh.read(test_shapefile)
        pyvista_mesh = ml.read_shape(test_shapefile)
        assert pyvista_mesh.n_points == ogs_mesh.n_points
        assert pyvista_mesh.n_cells == ogs_mesh.n_cells
