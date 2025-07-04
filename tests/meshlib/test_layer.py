import re
import shutil
import tempfile
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

from ogstools.definitions import EXAMPLES_DIR
from ogstools.meshlib.boundary import Layer, LocationFrame, Raster
from ogstools.meshlib.boundary_set import LayerSet
from ogstools.meshlib.boundary_subset import Gaussian2D, Surface
from ogstools.meshlib.region import (
    to_region_prism,
    to_region_simplified,
    to_region_tetraeder,
    to_region_voxel,
)

meshpath = EXAMPLES_DIR / "meshlib"


@pytest.mark.tools()  # Mesh2Raster
class TestLayer:
    def test_intermediate_1(self):
        """
        Create a layer from vtu input
        """
        layer1 = Layer(
            top=Surface(
                meshpath / "mesh1/surface_data/00_KB.vtu",
                0,
            ),
            bottom=Surface(
                meshpath / "mesh1/surface_data/01_q.vtu",
                1,
            ),
            material_id=0,
            num_subdivisions=1,
        )
        all_subrasters = layer1.create_raster(300)
        assert (
            len(all_subrasters) == 3
        ), "Expected 1 top + 1 bottom + 1 intermediate = 3 surfaces to be created."
        assert all(Path(subraster).exists() for subraster in all_subrasters)

    @pytest.mark.xfail(
        shutil.which("tetgen") is None, reason="Tetgen not installed"
    )
    def test_plane_surfaces(self):
        """
        Create a layer from pyvista input
        """

        heights = np.linspace(-1000, -1900, 3 + 1)  # +1 top surface
        planes = [
            Surface(
                Gaussian2D((9200, 18800, 9000, 21000), 0, 200, height, 20),
                material_id=id,
            )
            for id, height in enumerate(heights)
        ]
        layers = pairwise(planes)
        base_layers = [
            Layer(
                top=top,
                bottom=bottom,
                num_subdivisions=2,
            )
            for top, bottom in layers
        ]
        ls1 = LayerSet(layers=base_layers)
        tm = to_region_tetraeder(ls1, 200)

        # Prism mesh does not work when a surface is plane with height 0, limitation/bug of
        # CLI tool createIntermediateRasters
        sm = to_region_simplified(ls1, 200, 3)
        pm = to_region_prism(ls1, 200)

        # Voxel mesh has some limitation that resolution must fit, here height (900) is multiple of resolution (100)
        vm = to_region_voxel(ls1, [200, 200, 100])

        assert vm.mesh.number_of_cells > sm.mesh.number_of_cells
        assert pm.mesh.number_of_cells > vm.mesh.number_of_cells

        assert tm.mesh.number_of_cells > pm.mesh.number_of_cells


class TestRaster:
    @pytest.mark.tools()  # generateGeometry
    def testgmlwrite(self):
        """
        Checks if GML is created and its content (some parts)
        """
        x = LocationFrame(0, 20, -1.4, 4000.4)
        outfile = Path(tempfile.mkstemp(".gml")[1])
        x.as_gml(outfile)
        ## start of test
        pattern = r'y="(-?\d+\.\d+)"'
        with outfile.open() as f:
            lines = f.readlines()
            match = re.search(pattern=pattern, string=lines[4])
            if match:
                y_value = float(match.group(1))
                assert x.ymin == pytest.approx(y_value), 0.0001
            else:
                pytest.fail("Pattern not found in file.")

    def testrasterfilewrite(self):
        """
        Checks part of the content of the created VTU file
        """
        locFrame = LocationFrame(0.0, 20.0, -1.4, 12.4)
        raster = Raster(locFrame, 25)
        outfile = Path(tempfile.mkstemp(".vtu", "Raster")[1])
        vtu = raster.as_vtu(outfile)
        # start of test
        # expect a new file to written and now able to read as vtu
        mesh = pv.read(vtu)
        cells = mesh.number_of_cells
        assert cells == 2, "4 lines + 4 triangles"
        # self.assertAlmostEqual((0, 20.0, -1.4, 12.4, 0, 0), mesh.bounds,0.1)
