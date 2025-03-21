import pytest

from ogstools.definitions import EXAMPLES_DIR
from ogstools.meshlib._utils import dataframe_from_csv
from ogstools.meshlib.boundary_set import LayerSet
from ogstools.meshlib.region import to_region_prism

meshpath = EXAMPLES_DIR / "meshlib"


class TestPrismMesh:
    layerset = meshpath / "compose_geomodel/layersets.csv"
    materialset = meshpath / "compose_geomodel/materialset.csv"
    surfacedata = meshpath / "mesh1/surface_data/"

    @pytest.mark.tools()  # createLayeredMeshFromRasters
    def test_mesh_fine_xy_coarse_z(self):
        mesh1_df = dataframe_from_csv(
            1,
            self.layerset,
            self.materialset,
            self.surfacedata,
        )
        layer_set = LayerSet.from_pandas(mesh1_df)
        prism_mesh = to_region_prism(layer_set=layer_set, resolution=200)

        ## Test
        mesh_fine = prism_mesh.mesh
        assert (
            mesh_fine.number_of_cells % 4 == 0
        ), "There are 4 sublayers (covering the complete area) number of cells must be multiple of 4"

        assert (
            mesh_fine.number_of_points % 5 == 0
        ), "There are 4 surface + 1 top surface. Number of points must be multiple of 5"

        prism_mesh_coarse = to_region_prism(layer_set=layer_set, resolution=300)
        mesh_coarse = prism_mesh_coarse.mesh
        assert mesh_fine.number_of_cells > mesh_coarse.number_of_cells
        assert mesh_fine.number_of_points > mesh_coarse.number_of_points
