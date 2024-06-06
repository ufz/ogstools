import pytest

from ogstools.definitions import EXAMPLES_DIR
from ogstools.meshlib._utils import dataframe_from_csv

meshpath = EXAMPLES_DIR / "meshlib"


class TestGeoModelComposeExample:
    """
    Just checks if example "How to generate a geo model is working
    no LayerMesh specific functionality is tested
    """

    def test_compose1(self):
        layerset3_df = dataframe_from_csv(
            3,
            meshpath / "compose_geomodel/layersets.csv",
            meshpath / "compose_geomodel/materialset.csv",
            meshpath / "mesh1/surface_data/",
        )
        assert len(layerset3_df) == 5
        # self.assertEqual(df["layer_id"][3], 3)

    def test_compose_invalid(self):
        pytest.raises(
            Exception,
            dataframe_from_csv,
            20,
            meshpath / "compose_geomodel/layersets.csv",
            meshpath / "compose_geomodel/materialset.csv",
            meshpath / "mesh1/surface_data/",
            match="no model defined with",
        )
