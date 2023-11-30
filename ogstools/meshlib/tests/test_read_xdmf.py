import unittest

# check if xdmf file exists
from pathlib import Path

from ogstools.meshlib.examples import xdmf_file
from ogstools.meshlib.mesh_series import MeshSeries


class MeshSeriesReadTest(unittest.TestCase):
    def test_read_xdmf(self):
        xdmf_file_path = Path(xdmf_file)
        self.assertTrue(xdmf_file_path.exists())
        ms = MeshSeries(xdmf_file)
        mesh_1 = ms.read(1)
        self.assertEqual(mesh_1.number_of_cells, 171)
