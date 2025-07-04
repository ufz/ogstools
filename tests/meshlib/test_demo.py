import shutil
from collections import Counter

import pytest

from ogstools.definitions import EXAMPLES_DIR
from ogstools.meshlib.boundary import Layer
from ogstools.meshlib.boundary_set import LayerSet
from ogstools.meshlib.boundary_subset import Gaussian2D, Surface
from ogstools.meshlib.region import (
    to_region_prism,
    to_region_simplified,
    to_region_tetraeder,
    to_region_voxel,
)

meshpath = EXAMPLES_DIR / "meshlib"


@pytest.mark.tools()  # multiple tools
@pytest.mark.xfail(
    shutil.which("tetgen") is None, reason="Tetgen not installed"
)
class TestDemo:
    def test_allcompare(self):
        # To define a mesh with 3 layers from example input, create 4 surfaces (3 bottom surface + 1 top surface)
        surface1 = Surface(
            meshpath / "mesh1/surface_data/00_KB.vtu",
            material_id=0,
        )
        surface2 = Surface(
            meshpath / "mesh1/surface_data/01_q.vtu",
            material_id=5,
        )
        surface3 = Surface(
            meshpath / "mesh1/surface_data/02_krl.vtu",
            material_id=2,
        )
        surface4 = Surface(
            meshpath / "mesh1/surface_data/03_S3.vtu",
            material_id=3,
        )
        layer1 = Layer(top=surface1, bottom=surface2, num_subdivisions=2)

        layer2 = Layer(top=surface2, bottom=surface3, num_subdivisions=1)

        layer3 = Layer(top=surface3, bottom=surface4, num_subdivisions=0)

        layer_set = LayerSet(layers=[layer1, layer2, layer3])
        number_of_layers = len(layer_set.layers)

        sm = to_region_simplified(layer_set, 200, 3)
        assert (
            len(Counter(sm.mesh.cell_data["MaterialIDs"]).keys())
            == number_of_layers
        )

        tm = to_region_tetraeder(layer_set, 200)
        assert (
            len(Counter(tm.mesh.cell_data["MaterialIDs"]).keys())
            == number_of_layers
        )

        pm = to_region_prism(layer_set, 200)
        assert (
            len(Counter(pm.mesh.cell_data["MaterialIDs"]).keys())
            == number_of_layers
        )

        vm = to_region_voxel(layer_set, [200, 200, 50])
        assert (
            len(Counter(vm.mesh.cell_data["MaterialIDs"]).keys())
            == number_of_layers
        )

    @pytest.mark.tools()  # createLayeredMeshFromRasters
    def test_gettingstarted(self):
        # Define a simple surface
        bounds = (-200, 200, -200, 200)
        surface_test1 = Surface(
            Gaussian2D(
                bound2D=bounds, amplitude=100, spread=100, height_offset=0, n=20
            ),
            material_id=0,
        )
        surface_test2 = Surface(
            Gaussian2D(bounds, 100, 100, -100, 20), material_id=1
        )

        ls = LayerSet([Layer(surface_test1, surface_test2, material_id=1)])
        pm = to_region_prism(ls, 40)
        assert pm.mesh.number_of_cells > 0
