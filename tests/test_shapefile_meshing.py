import subprocess
from tempfile import mkdtemp

import ogstools.meshlib as ml
from ogstools.examples import circle_shapefile, test_shapefile


def test_cli():
    subprocess.run(["shp2msh", "--help"], check=True)
    subprocess.run(
        ["shp2msh", "-i", str(test_shapefile), "-o", mkdtemp(".vtu")],
        check=True,
    )


class TestShapeFileMeshing:
    def test_default(self):
        pyvista_mesh = ml.read_shape(
            test_shapefile
        )  # simplify_false, mesh_generator_triangle
        assert pyvista_mesh.n_points > 233000
        assert pyvista_mesh.n_points < 234000
        assert pyvista_mesh.n_cells > 344000
        assert pyvista_mesh.n_points < 345000

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
            circle_shapefile, simplify=True, mesh_generator="gmsh"
        )
        mesh_original = ml.read_shape(
            circle_shapefile, simplify=False, mesh_generator="gmsh"
        )

        assert mesh_simplify.n_points < mesh_original.n_points
        assert mesh_simplify.n_cells < mesh_original.n_cells

    def test_meshclass_reading(self):
        ogs_mesh = ml.Mesh.read(circle_shapefile)
        ogs_mesh_special = ml.Mesh.read_shape(circle_shapefile)
        pyvista_mesh = ml.read_shape(circle_shapefile)
        assert pyvista_mesh.n_points == ogs_mesh.n_points
        assert pyvista_mesh.n_cells == ogs_mesh.n_cells
        assert pyvista_mesh.n_points == ogs_mesh_special.n_points
        assert pyvista_mesh.n_cells == ogs_mesh_special.n_cells
