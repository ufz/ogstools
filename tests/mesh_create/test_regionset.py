import shutil
from collections import Counter
from collections.abc import Callable
from itertools import pairwise

import pytest

from ogstools.definitions import EXAMPLES_DIR
from ogstools.mesh import create, validate
from ogstools.mesh.create import LayerSet


@pytest.mark.tools()  # NodeReordering
@pytest.mark.parametrize("dim", [2, 3])
def test_to_region_simplified(
    dim: int, make_layerset: Callable[[int], LayerSet]
):
    layerset_coarse = make_layerset(1)
    layerset_fine = make_layerset(2)

    coarseXYZ = layerset_coarse.to_region_simplified(400, dim).mesh
    coarseXY_fineZ = layerset_fine.to_region_simplified(400, dim).mesh
    fineXY_coarseZ = layerset_coarse.to_region_simplified(300, dim).mesh

    assert fineXY_coarseZ.number_of_cells > 0
    assert fineXY_coarseZ.number_of_points > 0
    assert fineXY_coarseZ.number_of_cells > coarseXYZ.number_of_cells
    assert fineXY_coarseZ.number_of_points > coarseXYZ.number_of_points
    assert coarseXY_fineZ.number_of_cells > coarseXYZ.number_of_cells
    assert coarseXY_fineZ.number_of_points > coarseXYZ.number_of_points


@pytest.mark.tools()  # NodeReordering
def test_box_boundaries(make_layerset: Callable[[int], LayerSet]):
    mesh_fineXY_coarseZ = make_layerset(1).to_region_simplified(300, 3)
    bounds = mesh_fineXY_coarseZ.box_boundaries()
    assert len(bounds) == 6


@pytest.mark.tools  # Layers2Grid
def test_to_region_voxel(make_layerset: Callable[[int], LayerSet]):
    layerset = make_layerset(1)
    mesh_fine = layerset.to_region_voxel(resolution=[200, 200, 50]).mesh
    mesh_coarse = layerset.to_region_voxel(resolution=[200, 200, 100]).mesh
    assert mesh_fine.number_of_cells > mesh_coarse.number_of_cells
    assert mesh_fine.number_of_points > mesh_coarse.number_of_points


@pytest.mark.tools  # createLayeredMeshFromRasters
def test_to_region_prism(make_layerset: Callable[[int], LayerSet]):
    layerset = make_layerset(1)
    mesh_fine = layerset.to_region_prism(resolution=200).mesh
    mesh_coarse = layerset.to_region_prism(resolution=300).mesh

    assert mesh_fine.number_of_cells > mesh_coarse.number_of_cells
    assert mesh_fine.number_of_points > mesh_coarse.number_of_points
    assert (
        mesh_fine.number_of_cells % 4 == 0
    ), "4 layers require a multiple of 4 cells"
    assert (
        mesh_fine.number_of_points % 5 == 0
    ), "5 surfaces require a multiple of 5 points"


@pytest.mark.tools  # createTetgenSmeshFromRasters
@pytest.mark.xfail(
    shutil.which("tetgen") is None, reason="Tetgen not installed"
)
def test_to_region_tetrahedron(make_layerset: Callable[[int], LayerSet]):
    mesh = make_layerset(1).to_region_tetrahedron(resolution=400).mesh

    assert len(mesh.cell_data["MaterialIDs"]) > 0
    assert mesh.number_of_cells > 10000
    assert mesh.number_of_cells < 30000


@pytest.fixture
def layerset_example_surfs() -> create.LayerSet:
    names = ["00_KB", "01_q", "02_krl", "03_S3"]
    mat_ids = [0, 5, 2, 3]
    surfacedata = EXAMPLES_DIR / "meshlib/mesh1/surface_data/"
    surfaces = [
        create.Surface(surfacedata / f"{s}.vtu", mat_id)
        for s, mat_id in zip(names, mat_ids, strict=True)
    ]
    layers = [create.Layer(sf1, sf2) for sf1, sf2 in pairwise(surfaces)]
    return create.LayerSet(layers)


@pytest.mark.tools  # multiple tools
@pytest.mark.xfail(
    shutil.which("tetgen") is None, reason="Tetgen not installed"
)
@pytest.mark.parametrize(
    "discretization",
    [
        lambda ls: ls.to_region_simplified(200, 2),
        lambda ls: ls.to_region_simplified(400, 3),
        lambda ls: ls.to_region_tetrahedron(500),
        lambda ls: ls.to_region_prism(400),
        lambda ls: ls.to_region_voxel([300, 300, 50]),
    ],
)
@pytest.mark.tools()  # checkMesh
def test_allcompare(discretization, layerset_example_surfs: create.LayerSet):
    number_of_layers = len(layerset_example_surfs.layers)
    mesh = discretization(layerset_example_surfs).mesh
    assert (
        len(Counter(mesh.cell_data["MaterialIDs"]).keys()) == number_of_layers
    )
    assert validate(mesh, strict=True)


@pytest.mark.tools  # createLayeredMeshFromRasters
def test_to_region_prism_docu():
    bounds = (-200, 210, -200, 210)
    args = {"bound2D": bounds, "amplitude": 100, "spread": 100, "n": 40}
    gaussians = [
        create.Gaussian2D(**args, height_offset=h) for h in [0, -100, -200]
    ]
    surfaces = [create.Surface(g, mat_id) for mat_id, g in enumerate(gaussians)]
    layers = [create.Layer(sf1, sf2) for sf1, sf2 in pairwise(surfaces)]
    ls = create.LayerSet(layers)
    mesh = ls.to_region_prism(40).mesh
    assert mesh.number_of_cells > 0
