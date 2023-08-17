import unittest
from pathlib import Path

from ogstools.meshlib._utils import dataframe_from_csv


def MeshPath(filenamepath: str):
    """
    Never use MeshPath in your projects.
    It is supposed to be used in tests, only.
    """
    current_dir = Path(__file__).parent
    return current_dir / Path(filenamepath)


class GeoModelComposeExampleTest(unittest.TestCase):
    """
    Just checks if example "How to generate a geo model is working
    no LayerMesh specific functionality is tested
    """

    def test_compose1(self):
        layerset3_df = dataframe_from_csv(
            3,
            MeshPath("data/compose_geomodel/layersets.csv"),
            MeshPath("data/compose_geomodel/materialset.csv"),
            MeshPath("data/mesh1/surface_data/"),
        )
        self.assertEqual(len(layerset3_df), 5)
        # self.assertEqual(df["layer_id"][3], 3)

    def test_compose_invalid(self):
        self.assertRaises(
            Exception,
            dataframe_from_csv,
            20,
            MeshPath("data/compose_geomodel/layersets.csv"),
            MeshPath("data/compose_geomodel/materialset.csv"),
            MeshPath("data/mesh1/surface_data/"),
        )
