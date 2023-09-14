import subprocess
import unittest
from pathlib import Path

import pytest
import pyvista as pv

pytest.importorskip("ifm")
import ifm_contrib as ifm  # noqa: E402

from ogstools.fe2vtu.feflowlib import get_pts_cells  # noqa: E402


def test_cli():
    subprocess.run(["fe2vtu", "--help"], check=True)


current_dir = Path(__file__).parent


class TestConverter(unittest.TestCase):
    def test_converter(self):
        doc = ifm.loadDocument(
            str(Path(current_dir / "data/fe2vtu/2layers_model.fem"))
        )
        points, cells, celltypes = get_pts_cells(doc)
        assert len(points) == 75
        assert len(celltypes) == 32
        assert celltypes[0] == pv.CellType.HEXAHEDRON


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)
