from collections.abc import Callable
from itertools import pairwise

import pytest

from ogstools.definitions import EXAMPLES_DIR
from ogstools.mesh import create
from ogstools.mesh.create import LayerSet


def test_compose1(make_layerset: Callable[[int], LayerSet]):
    layerset = make_layerset(3)
    assert len(layerset.layers) == 4
    mat_ids = [layer.material_id for layer in layerset.layers]
    assert mat_ids == [1, 5, 12, 2]


@pytest.mark.xfail(Exception, reason="no model defined with this layerset_id")
def test_compose_invalid(make_layerset: Callable[[int], LayerSet]):
    make_layerset(20)


@pytest.mark.tools  # createIntermediateRasters
def test_create_with_1_intermediate(make_layerset: Callable[[int], LayerSet]):
    assert len(make_layerset(2).create_rasters(300)) == 9


@pytest.mark.tools  # createIntermediateRasters
def test_create_with_3_intermediate(make_layerset: Callable[[int], LayerSet]):
    layer_set = make_layerset(3)
    raster_set = layer_set.create_rasters(resolution=300)
    assert len(layer_set.layers) == 4, "Expected 4 Layers in Layerset"
    assert len(raster_set) == 15
    assert "02_krl" in str(
        raster_set[5]
    ), f"Index of base layer 02_krl is 2 (preceding base layers) + 1 intermediate layer in layer 1 and + 2 intermediate layer in layer 2. The name is {raster_set[5]}"


@pytest.mark.tools  # createIntermediateRasters
def test_create_with_no_intermediate():
    resolution = 300
    names = ["00_KB", "01_q", "02_krl"]
    surfacedata = EXAMPLES_DIR / "meshlib/mesh1/surface_data/"
    surfaces = [create.Surface(surfacedata / f"{s}.vtu", 1) for s in names]
    layers = [create.Layer(sf1, sf2) for sf1, sf2 in pairwise(surfaces)]
    raster_set = LayerSet(layers).create_rasters(resolution=resolution)
    assert (
        len(raster_set) == 2 + 1
    ), "With no intermediate layers, expected num of layer to be equal to pairs of top/bottom layer + 1 for topmost layer"
