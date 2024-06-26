from ogstools.definitions import EXAMPLES_DIR
from ogstools.meshlib._utils import dataframe_from_csv
from ogstools.meshlib.boundary_set import LayerSet
from ogstools.meshlib.region import to_region_simplified

meshpath = EXAMPLES_DIR / "meshlib"


class TestSimplifiedMesh:
    layerset = meshpath / "compose_geomodel/layersets.csv"
    materialset = meshpath / "compose_geomodel/materialset.csv"
    surfacedata = meshpath / "mesh1/surface_data/"

    def test_3_d_points_and_cells_of_coarse_to_fine_meshes(self):
        mesh_df_coarseZ = dataframe_from_csv(
            1,
            self.layerset,
            self.materialset,
            self.surfacedata,
        )
        layerset_coarse = LayerSet.from_pandas(mesh_df_coarseZ)
        mesh_fineXY_coarseZ = to_region_simplified(layerset_coarse, 300, 3).mesh
        assert mesh_fineXY_coarseZ.number_of_cells > 0
        assert mesh_fineXY_coarseZ.number_of_points > 0
        mesh_coarseXYZ = to_region_simplified(layerset_coarse, 400, 3).mesh
        assert (
            mesh_fineXY_coarseZ.number_of_cells > mesh_coarseXYZ.number_of_cells
        )
        assert (
            mesh_fineXY_coarseZ.number_of_points
            > mesh_coarseXYZ.number_of_points
        )
        mesh_df_fineZ = dataframe_from_csv(
            2,
            self.layerset,
            self.materialset,
            self.surfacedata,
        )
        layerset_fine = LayerSet.from_pandas(mesh_df_fineZ)
        mesh_coarseXY_fineZ = to_region_simplified(layerset_fine, 400, 3).mesh
        assert (
            mesh_coarseXY_fineZ.number_of_cells > mesh_coarseXYZ.number_of_cells
        )
        assert (
            mesh_coarseXY_fineZ.number_of_points
            > mesh_coarseXYZ.number_of_points
        )

    def test_2_d_points_and_cells_of_coarse_to_fine_meshes(self):
        mesh_df_coarseZ = dataframe_from_csv(
            1,
            self.layerset,
            self.materialset,
            self.surfacedata,
        )
        layerset_coarse = LayerSet.from_pandas(mesh_df_coarseZ)
        mesh_fineXY_coarseZ = to_region_simplified(layerset_coarse, 300, 2).mesh
        assert mesh_fineXY_coarseZ.number_of_cells > 0
        assert mesh_fineXY_coarseZ.number_of_points > 0
        mesh_coarseXYZ = to_region_simplified(layerset_coarse, 400, 2).mesh
        assert (
            mesh_fineXY_coarseZ.number_of_cells > mesh_coarseXYZ.number_of_cells
        )
        assert (
            mesh_fineXY_coarseZ.number_of_points
            > mesh_coarseXYZ.number_of_points
        )
        mesh_df_fineZ = dataframe_from_csv(
            2,
            self.layerset,
            self.materialset,
            self.surfacedata,
        )
        layerset_fine = LayerSet.from_pandas(mesh_df_fineZ)
        sm = to_region_simplified(layerset_fine, 400, 2)
        mesh_coarseXY_fineZ = sm.mesh
        assert (
            mesh_coarseXY_fineZ.number_of_cells > mesh_coarseXYZ.number_of_cells
        )
        assert (
            mesh_coarseXY_fineZ.number_of_points
            > mesh_coarseXYZ.number_of_points
        )
