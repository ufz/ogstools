import subprocess
import unittest
from pathlib import Path

import pytest
import pyvista as pv

pytest.importorskip("ifm")
import ifm_contrib as ifm  # noqa: E402

from ogstools.feflowlib.feflowlib import points_and_cells  # noqa: E402


def test_cli():
    subprocess.run(["feflow2ogs", "--help"], check=True)


current_dir = Path(__file__).parent


class TestConverter(unittest.TestCase):
    def test_converter(self):
        doc = ifm.loadDocument(
            str(Path(current_dir / "data/feflowlib/2layers_model.fem"))
        )
        points, cells, celltypes = points_and_cells(doc)
        assert len(points) == 75
        assert len(celltypes) == 32
        assert celltypes[0] == pv.CellType.HEXAHEDRON
