import unittest

from ogstools.meshlib._utils import dataframe_from_csv
from ogstools.meshlib.boundary_set import LayerSet
from ogstools.meshlib.region import to_region_voxel
from ogstools.meshlib.tests import MeshPath


class VoxelMeshTest(unittest.TestCase):
    layerset = MeshPath("data/compose_geomodel/layersets.csv")
    materialset = MeshPath("data/compose_geomodel/materialset.csv")
    surfacedata = MeshPath("data/mesh1/surface_data/")

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
        self.assertGreater(
            mesh_fine.number_of_cells, mesh_coarse.number_of_cells
        )
        self.assertGreater(
            mesh_fine.number_of_points, mesh_coarse.number_of_points
        )
