import unittest

from ogstools.meshlib._utils import dataframe_from_csv
from ogstools.meshlib.boundary import Layer
from ogstools.meshlib.boundary_set import LayerSet
from ogstools.meshlib.boundary_subset import Surface
from ogstools.meshlib.tests import MeshPath


class LayerSetTest(unittest.TestCase):
    layerset = MeshPath("data/compose_geomodel/layersets.csv")
    materialset = MeshPath("data/compose_geomodel/materialset.csv")
    surfacedata = MeshPath("data/mesh1/surface_data/")

    def test_create_with_3_intermediate(self):
        mesh3_df = dataframe_from_csv(
            3,
            self.layerset,
            self.materialset,
            self.surfacedata,
        )
        layer_set1 = LayerSet.from_pandas(mesh3_df)
        self.assertEqual(
            len(layer_set1.layers), 4, "Expected 4 Layers in Layerset"
        )
        raster_set_1_300 = layer_set1.create_rasters(resolution=300)
        self.assertEqual(len(raster_set_1_300), 15)
        self.assertIn(
            "02_krl",
            str(raster_set_1_300[5]),
            "Index of base layer 02_krl is 2 (preceding base layers) + 1 intermediate layer in layer 1 and + 2 intermediate layer in layer 2. The name is {}".format(
                raster_set_1_300[5]
            ),
        )

    def test_create_with_no_intermediate(self):
        resolution = 300
        surface1 = Surface(
            MeshPath("data/mesh1/surface_data/00_KB.vtu"),
            0,
        )
        surface2 = Surface(
            MeshPath("data/mesh1/surface_data/01_q.vtu"),
            1,
        )
        surface3 = Surface(
            MeshPath("data/mesh1/surface_data/02_krl.vtu"),
            1,
        )

        layer1 = Layer(
            top=surface1,
            bottom=surface2,
            material_id=0,
            num_subdivisions=0,
        )

        layer2 = Layer(
            top=surface2,
            bottom=surface3,
            material_id=1,
            num_subdivisions=0,
        )

        layer_set = LayerSet(layers=[layer1, layer2])
        raster_set = layer_set.create_rasters(resolution=resolution)
        self.assertEqual(
            len(raster_set),
            2 + 1,
            "With no intermediate layers, expected num of layer to be equal to pairs of top/bottom layer + 1 for topmost layer",
        )

    def test_create_with_just1_intermediate(self):
        layerset2_df = dataframe_from_csv(
            2,
            self.layerset,
            self.materialset,
            self.surfacedata,
        )
        layer_set1 = LayerSet.from_pandas(layerset2_df)
        raster_set1 = layer_set1.create_rasters(300)
        self.assertEqual(len(raster_set1), 9)
