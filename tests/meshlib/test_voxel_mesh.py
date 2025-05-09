import pytest

from ogstools.definitions import EXAMPLES_DIR
from ogstools.meshlib._utils import dataframe_from_csv
from ogstools.meshlib.boundary_set import LayerSet
from ogstools.meshlib.region import to_region_voxel

meshpath = EXAMPLES_DIR / "meshlib"


class TestVoxelMesh:
    layerset = meshpath / "compose_geomodel/layersets.csv"
    materialset = meshpath / "compose_geomodel/materialset.csv"
    surfacedata = meshpath / "mesh1/surface_data/"

    @pytest.mark.tools()  # Layers2Grid
    def test_create(self):
        mesh1_df = dataframe_from_csv(
            1,
            self.layerset,
            self.materialset,
            self.surfacedata,
        )
        layer_set = LayerSet.from_pandas(mesh1_df)
        mesh_fine = to_region_voxel(
            layer_set=layer_set, resolution=[200, 200, 50]
        ).mesh
        mesh_coarse = to_region_voxel(
            layer_set=layer_set, resolution=[200, 200, 100]
        ).mesh
        assert mesh_fine.number_of_cells > mesh_coarse.number_of_cells
        assert mesh_fine.number_of_points > mesh_coarse.number_of_points
