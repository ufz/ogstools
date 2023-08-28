import unittest

from ogstools.meshlib.boundary_set import LayerSet
from ogstools.meshlib.region import to_region_tetraeder
from ogstools.meshlib.tests import MeshPath, dataframe_from_csv


class TetraederTest(unittest.TestCase):
    layerset = MeshPath("data/compose_geomodel/layersets.csv")
    materialset = MeshPath("data/compose_geomodel/materialset.csv")
    surfacedata = MeshPath("data/mesh1/surface_data/")

    def test_Mesh_coarseXYZ(self):
        mesh1_df = dataframe_from_csv(
            1,
            self.layerset,
            self.materialset,
            self.surfacedata,
        )
        layer_set = LayerSet.from_pandas(mesh1_df)
        tetraeder = to_region_tetraeder(layer_set=layer_set, resolution=400)

        mesh = tetraeder.mesh
        self.assertGreater(len(mesh.cell_data["MaterialIDs"]), 0)
        self.assertGreater(mesh.number_of_cells, 10000)
        self.assertLess(mesh.number_of_cells, 30000)
