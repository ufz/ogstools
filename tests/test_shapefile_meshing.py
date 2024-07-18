import subprocess

import ogstools.meshlib as ml
from ogstools.examples import test_shapefile


def test_cli():
    subprocess.run(["shp2msh", "--help"], check=True)


class TestMeshing:
    def test_meshing(self):
        points, cells = ml.shapefile_meshing(test_shapefile)
        pyvista_mesh = ml.Mesh.from_points_cells(points=points, cells=cells)
        assert pyvista_mesh.n_points == len(points)
        assert pyvista_mesh.n_cells == len(cells)

    # Same for simplified mesh.
    def test_simple_meshing(self):
        points, cells = ml.shapefile_meshing(test_shapefile, simplify=True)
        pyvista_mesh = ml.Mesh.from_points_cells(points=points, cells=cells)
        assert pyvista_mesh.n_points == len(points)
        assert pyvista_mesh.n_cells == len(cells)

    def test_gmsh_meshing(self):
        points, cells = ml.shapefile_meshing(
            test_shapefile, simplify=True, triangle=False
        )
        pyvista_mesh = ml.Mesh.from_points_cells(points=points, cells=cells)

        assert pyvista_mesh.n_points == len(points)
        assert pyvista_mesh.n_cells == len(cells)

    def test_meshclass_reading(self):
        pyvista_mesh = ml.Mesh.read(test_shapefile)
        points, cells = ml.shapefile_meshing(test_shapefile)

        assert pyvista_mesh.n_points == len(points)
        assert pyvista_mesh.n_cells == len(cells)
