import subprocess

import ogstools.meshlib as ml
from ogstools.examples import test_shapefile
from ogstools.meshlib import (
    create_pyvista_mesh,
    geodataframe_meshing,
    prepare_shp_for_meshing,
)


def test_cli():
    subprocess.run(["shp2msh", "--help"], check=True)


class TestMeshing:
    def setup_method(self):
        self.geodataframe = prepare_shp_for_meshing(test_shapefile)

    def test_meshing(self):
        points_cells = geodataframe_meshing(self.geodataframe)
        pyvista_mesh = create_pyvista_mesh(
            points=points_cells[0], cells=points_cells[1]
        )
        assert pyvista_mesh.n_points == len(points_cells[0])
        assert pyvista_mesh.n_cells == len(points_cells[1])

    # Same for simplified mesh.
    def test_simple_meshing(self):
        points_cells = geodataframe_meshing(self.geodataframe, True)
        pyvista_mesh = create_pyvista_mesh(
            points=points_cells[0], cells=points_cells[1]
        )
        assert pyvista_mesh.n_points == len(points_cells[0])
        assert pyvista_mesh.n_cells == len(points_cells[1])

    def test_gmsh_meshing(self):
        points_cells = geodataframe_meshing(
            self.geodataframe, True, False, 100000
        )
        print(points_cells)
        pyvista_mesh = create_pyvista_mesh(
            points=points_cells[0], cells=points_cells[1]
        )
        assert pyvista_mesh.n_points == len(points_cells[0])
        assert pyvista_mesh.n_cells == len(points_cells[1])

    def test_meshclass_reading(self):
        pyvista_mesh = ml.Mesh.read(test_shapefile)
        points_cells = geodataframe_meshing(self.geodataframe)

        assert pyvista_mesh.n_points == len(points_cells[0])
        assert pyvista_mesh.n_cells == len(points_cells[1])
