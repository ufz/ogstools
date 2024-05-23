import unittest

from ogstools.definitions import EXAMPLES_DIR
from ogstools.meshlib._utils import dataframe_from_csv
from ogstools.meshlib.boundary_set import LayerSet
from ogstools.meshlib.region import (
    to_region_simplified,
)

meshpath = EXAMPLES_DIR / "meshlib"


class RegionTest(unittest.TestCase):
    layerset = meshpath / "compose_geomodel/layersets.csv"
    materialset = meshpath / "compose_geomodel/materialset.csv"
    surfacedata = meshpath / "mesh1/surface_data/"

    def test_top_boundary(self):
        mesh_df_coarseZ = dataframe_from_csv(
            1,
            self.layerset,
            self.materialset,
            self.surfacedata,
        )
        layerset_coarse = LayerSet.from_pandas(mesh_df_coarseZ)
        mesh_fineXY_coarseZ = to_region_simplified(layerset_coarse, 300, 3)
        # 3D
        bounds = mesh_fineXY_coarseZ.box_boundaries()
        assert len(bounds) == 6
