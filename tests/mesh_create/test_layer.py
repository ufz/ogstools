import re
import shutil
import tempfile
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

from ogstools.definitions import EXAMPLES_DIR
from ogstools.mesh import create

meshpath = EXAMPLES_DIR / "meshlib"


@pytest.mark.tools  # Mesh2Raster
class TestLayer:
    def test_intermediate_1(self):
        """Create a layer from vtu input"""
        surf1 = create.Surface(meshpath / "mesh1/surface_data/00_KB.vtu", 0)
        surf2 = create.Surface(meshpath / "mesh1/surface_data/01_q.vtu", 1)
        layer1 = create.Layer(surf1, surf2, material_id=0, num_subdivisions=1)
        all_subrasters = layer1.create_raster(300)
        assert len(all_subrasters) == 3, "top + bottom + 1 intermediate"
        assert all(Path(subraster).exists() for subraster in all_subrasters)

    @pytest.mark.xfail(
        shutil.which("tetgen") is None, reason="Tetgen not installed"
    )
    def test_plane_surfaces(self):
        """Create a layer from pyvista input"""

        heights = np.linspace(-1000, -1900, 3 + 1)  # +1 top surface
        gaussians = [
            create.Gaussian2D((9200, 18800, 9000, 21000), 0, 200, height, 20)
            for height in heights
        ]
        planes = [
            create.Surface(g, material_id=id) for id, g in enumerate(gaussians)
        ]
        base_layers = [
            create.Layer(top=top, bottom=bottom, num_subdivisions=2)
            for top, bottom in pairwise(planes)
        ]
        layerset = create.LayerSet(layers=base_layers)
        tm = layerset.to_region_tetrahedron(200)

        # Prism mesh does not work when a surface is plane with height 0, limitation/bug of
        # CLI tool createIntermediateRasters
        sm = layerset.to_region_simplified(200, 3)
        pm = layerset.to_region_prism(200)

        # Voxel mesh has some limitation that resolution must fit, here height (900) is multiple of resolution (100)
        vm = layerset.to_region_voxel([200, 200, 100])

        assert vm.mesh.number_of_cells > sm.mesh.number_of_cells
        assert pm.mesh.number_of_cells > vm.mesh.number_of_cells
        assert tm.mesh.number_of_cells > pm.mesh.number_of_cells


class TestRaster:
    @pytest.mark.tools  # generateGeometry
    def testgmlwrite(self):
        """
        Checks if GML is created and its content (some parts)
        """
        x = create.LocationFrame(0, 20, -1.4, 4000.4)
        outfile = Path(tempfile.mkstemp(".gml")[1])
        x.as_gml(outfile)
        ## start of test
        pattern = r'y="(-?\d+\.\d+)"'
        with outfile.open() as f:
            lines = f.readlines()
            match = re.search(pattern=pattern, string=lines[4])
            if match:
                y_value = float(match.group(1))
                assert x.ymin == pytest.approx(y_value)
            else:
                pytest.fail("Pattern not found in file.")

    def testrasterfilewrite(self):
        """
        Checks part of the content of the created VTU file
        """
        locFrame = create.LocationFrame(0.0, 20.0, -1.4, 12.4)
        raster = create.Raster(locFrame, 25)
        outfile = Path(tempfile.mkstemp(".vtu", "Raster")[1])
        vtu = raster.as_vtu(outfile)
        # start of test
        # expect a new file to written and now able to read as vtu
        mesh = pv.read(vtu)
        cells = mesh.number_of_cells
        assert cells == 2, "4 lines + 4 triangles"
        # self.assertAlmostEqual((0, 20.0, -1.4, 12.4, 0, 0), mesh.bounds,0.1)
