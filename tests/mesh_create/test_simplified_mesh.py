from ogstools.definitions import EXAMPLES_DIR
from ogstools.mesh import create

meshpath = EXAMPLES_DIR / "meshlib"


class TestSimplifiedMesh:
    layerset = meshpath / "compose_geomodel/layersets.csv"
    surfacedata = meshpath / "mesh1/surface_data/"

    def test_3_d_points_and_cells_of_coarse_to_fine_meshes(self):
        df_coarseZ = create.dataframe_from_csv(
            1, self.layerset, self.surfacedata
        )
        layerset_coarse = create.LayerSet.from_pandas(df_coarseZ)
        mesh_fineXY_coarseZ = layerset_coarse.to_region_simplified(300, 3).mesh
        assert mesh_fineXY_coarseZ.number_of_cells > 0
        assert mesh_fineXY_coarseZ.number_of_points > 0
        mesh_coarseXYZ = layerset_coarse.to_region_simplified(400, 3).mesh
        assert (
            mesh_fineXY_coarseZ.number_of_cells > mesh_coarseXYZ.number_of_cells
        )
        assert (
            mesh_fineXY_coarseZ.number_of_points
            > mesh_coarseXYZ.number_of_points
        )
        df_fineZ = create.dataframe_from_csv(2, self.layerset, self.surfacedata)
        layerset_fine = create.LayerSet.from_pandas(df_fineZ)
        mesh_coarseXY_fineZ = layerset_fine.to_region_simplified(400, 3).mesh
        assert (
            mesh_coarseXY_fineZ.number_of_cells > mesh_coarseXYZ.number_of_cells
        )
        assert (
            mesh_coarseXY_fineZ.number_of_points
            > mesh_coarseXYZ.number_of_points
        )

    def test_2_d_points_and_cells_of_coarse_to_fine_meshes(self):
        df_coarseZ = create.dataframe_from_csv(
            1, self.layerset, self.surfacedata
        )
        layerset_coarse = create.LayerSet.from_pandas(df_coarseZ)
        mesh_fineXY_coarseZ = layerset_coarse.to_region_simplified(300, 2).mesh
        assert mesh_fineXY_coarseZ.number_of_cells > 0
        assert mesh_fineXY_coarseZ.number_of_points > 0
        mesh_coarseXYZ = layerset_coarse.to_region_simplified(400, 2).mesh
        assert (
            mesh_fineXY_coarseZ.number_of_cells > mesh_coarseXYZ.number_of_cells
        )
        assert (
            mesh_fineXY_coarseZ.number_of_points
            > mesh_coarseXYZ.number_of_points
        )
        df_fineZ = create.dataframe_from_csv(2, self.layerset, self.surfacedata)
        layerset_fine = create.LayerSet.from_pandas(df_fineZ)
        sm = layerset_fine.to_region_simplified(400, 2)
        mesh_coarseXY_fineZ = sm.mesh
        assert (
            mesh_coarseXY_fineZ.number_of_cells > mesh_coarseXYZ.number_of_cells
        )
        assert (
            mesh_coarseXY_fineZ.number_of_points
            > mesh_coarseXYZ.number_of_points
        )
