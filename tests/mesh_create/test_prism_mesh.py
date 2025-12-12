import pytest

from ogstools.definitions import EXAMPLES_DIR
from ogstools.mesh import create

meshpath = EXAMPLES_DIR / "meshlib"


class TestPrismMesh:
    layerset = meshpath / "compose_geomodel/layersets.csv"
    surfacedata = meshpath / "mesh1/surface_data/"

    @pytest.mark.tools()  # createLayeredMeshFromRasters
    def test_mesh_fine_xy_coarse_z(self):
        mesh1_df = create.dataframe_from_csv(1, self.layerset, self.surfacedata)
        layer_set = create.LayerSet.from_pandas(mesh1_df)
        prism_mesh = layer_set.to_region_prism(resolution=200)

        ## Test
        mesh_fine = prism_mesh.mesh
        assert (
            mesh_fine.number_of_cells % 4 == 0
        ), "There are 4 sublayers (covering the complete area) number of cells must be multiple of 4"

        assert (
            mesh_fine.number_of_points % 5 == 0
        ), "There are 4 surface + 1 top surface. Number of points must be multiple of 5"

        prism_mesh_coarse = layer_set.to_region_prism(resolution=300)
        mesh_coarse = prism_mesh_coarse.mesh
        assert mesh_fine.number_of_cells > mesh_coarse.number_of_cells
        assert mesh_fine.number_of_points > mesh_coarse.number_of_points
