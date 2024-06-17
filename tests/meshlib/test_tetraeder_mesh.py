from ogstools.definitions import EXAMPLES_DIR
from ogstools.meshlib._utils import dataframe_from_csv
from ogstools.meshlib.boundary_set import LayerSet
from ogstools.meshlib.region import to_region_tetraeder

meshpath = EXAMPLES_DIR / "meshlib"


class TestTetraeder:
    layerset = meshpath / "compose_geomodel/layersets.csv"
    materialset = meshpath / "compose_geomodel/materialset.csv"
    surfacedata = meshpath / "mesh1/surface_data/"

    def test_mesh_coarse_xyz(self):
        mesh1_df = dataframe_from_csv(
            1,
            self.layerset,
            self.materialset,
            self.surfacedata,
        )
        layer_set = LayerSet.from_pandas(mesh1_df)
        tetraeder = to_region_tetraeder(layer_set=layer_set, resolution=400)

        mesh = tetraeder.mesh
        assert len(mesh.cell_data["MaterialIDs"]) > 0
        assert mesh.number_of_cells > 10000
        assert mesh.number_of_cells < 30000
