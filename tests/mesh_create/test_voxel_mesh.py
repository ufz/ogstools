import pytest

from ogstools.definitions import EXAMPLES_DIR
from ogstools.mesh import create

meshpath = EXAMPLES_DIR / "meshlib"


class TestVoxelMesh:
    layerset = meshpath / "compose_geomodel/layersets.csv"
    materialset = meshpath / "compose_geomodel/materialset.csv"
    surfacedata = meshpath / "mesh1/surface_data/"

    @pytest.mark.tools()  # Layers2Grid
    def test_create(self):
        mesh1_df = create.dataframe_from_csv(
            1,
            self.layerset,
            self.materialset,
            self.surfacedata,
        )
        layer_set = create.LayerSet.from_pandas(mesh1_df)
        mesh_fine = layer_set.to_region_voxel(resolution=[200, 200, 50]).mesh
        mesh_coarse = layer_set.to_region_voxel(resolution=[200, 200, 100]).mesh
        assert mesh_fine.number_of_cells > mesh_coarse.number_of_cells
        assert mesh_fine.number_of_points > mesh_coarse.number_of_points
