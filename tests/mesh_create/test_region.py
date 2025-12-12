from ogstools.definitions import EXAMPLES_DIR
from ogstools.mesh import create

meshpath = EXAMPLES_DIR / "meshlib"


class TestRegion:
    layerset = meshpath / "compose_geomodel/layersets.csv"
    surfacedata = meshpath / "mesh1/surface_data/"

    def test_top_boundary(self):
        df_coarseZ = create.dataframe_from_csv(
            1, self.layerset, self.surfacedata
        )
        layerset_coarse = create.LayerSet.from_pandas(df_coarseZ)
        mesh_fineXY_coarseZ = layerset_coarse.to_region_simplified(300, 3)
        # 3D
        bounds = mesh_fineXY_coarseZ.box_boundaries()
        assert len(bounds) == 6
