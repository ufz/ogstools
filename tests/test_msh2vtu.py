"""
Tests (pytest) for msh2vtu and meshes_from_gmsh
"""

import os
import runpy
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from tempfile import mkdtemp

import gmsh
import meshio
import numpy as np
import pytest
from hypothesis import HealthCheck, Verbosity, example, given, settings
from hypothesis import strategies as st

import ogstools as ot
from ogstools.examples import msh_geolayers_2d, msh_geoterrain_3d
from ogstools.meshlib.gmsh_converter import meshes_from_gmsh
from ogstools.meshlib.gmsh_meshing import cuboid, rect
from ogstools.msh2vtu._cli import cli


def test_multiple_groups_per_element_2(tmp_path: Path) -> None:
    gmsh.initialize()
    gmsh.model.add("multiple_groups_per_element_2")

    block = gmsh.model.occ.addRectangle(0, 0, 0, 25, 50)
    tunnel = gmsh.model.occ.addRectangle(0, 0, 0, 3, 20)
    gmsh.model.occ.cut([(2, block)], [(2, tunnel)], removeTool=False)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(1, [8, 9], name="left")
    gmsh.model.addPhysicalGroup(1, [8], name="left_1")
    gmsh.model.addPhysicalGroup(1, [9], name="left_2")
    gmsh.model.addPhysicalGroup(1, [5], name="bottom_1")
    gmsh.model.addPhysicalGroup(1, [12], name="bottom_2")
    gmsh.model.addPhysicalGroup(1, [5, 12], name="bottom")

    distance = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance, "CurvesList", [6, 7])
    threshold = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold, "IField", distance)
    gmsh.model.mesh.field.setNumber(threshold, "LcMin", 3 / 12)
    gmsh.model.mesh.field.setNumber(threshold, "LcMax", 50 / 4)
    gmsh.model.mesh.field.setNumber(threshold, "DistMin", 3 / 6)
    gmsh.model.mesh.field.setNumber(threshold, "DistMax", 50)
    gmsh.model.mesh.field.add("Min", 99)
    gmsh.model.mesh.field.setNumbers(99, "FieldsList", [threshold])
    gmsh.model.mesh.field.setAsBackgroundMesh(99)
    gmsh.model.occ.synchronize()
    gmsh.model.occ.removeAllDuplicates()

    gmsh.model.mesh.generate(2)
    msh_file = Path(tmp_path, "multiple_groups_per_element_2.msh")
    gmsh.write(str(msh_file))
    # gmsh.fltk.run()
    gmsh.finalize()

    meshes = meshes_from_gmsh(msh_file)
    ncells_per_group = {
        "bottom": [19, 6, 13],
        "left": [43, 27, 16],
    }
    for group, n_cells in ncells_per_group.items():
        for index, suffix in enumerate(["", "_1", "_2"]):
            if group + suffix in meshes:
                assert n_cells[index] == meshes[group + suffix].n_cells
    assert meshes["bottom_1"].bounds[0] == 0
    assert meshes["bottom_1"].bounds[1] == 3
    assert meshes["bottom_2"].bounds[0] == 3
    assert meshes["bottom_2"].bounds[1] == 25
    assert meshes["left_1"].bounds[2] == 0
    assert meshes["left_1"].bounds[3] == 20
    assert meshes["left_2"].bounds[2] == 20
    assert meshes["left_2"].bounds[3] == 50


def test_multiple_groups_per_element(tmp_path: Path):
    """Test correct conversion, if element are assigned to multiple groups."""
    gmsh.initialize()
    # gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("multiple_groups_per_element")

    gmsh.model.geo.addPoint(-5, -5, 0, 1)
    gmsh.model.geo.addPoint(-2, -5, 0, 1)
    gmsh.model.geo.addPoint(2, -5, 0, 1)
    gmsh.model.geo.addPoint(5, -5, 0, 1)
    gmsh.model.geo.addPoint(5, 5, 0, 1)
    gmsh.model.geo.addPoint(-5, 5, 0, 1)

    for i in range(1, 7):
        gmsh.model.geo.addLine(i, i % 6 + 1, i)

    gmsh.model.geo.addCurveLoop([1, 2, 3, 4, 5, 6], 1)
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.geo.mesh.setTransfiniteCurve(2, 20)
    gmsh.model.geo.addPhysicalGroup(dim=1, tags=[2], name="bottom_center")
    gmsh.model.geo.addPhysicalGroup(dim=1, tags=[1, 2, 3], name="bottom")
    gmsh.model.geo.addPhysicalGroup(dim=1, tags=[4], name="right")
    gmsh.model.geo.addPhysicalGroup(dim=1, tags=[5], name="top")
    gmsh.model.geo.addPhysicalGroup(dim=1, tags=[6], name="left")
    gmsh.model.geo.addPhysicalGroup(
        dim=1, tags=[1, 2, 3, 4, 5, 6], name="boundaries"
    )

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    msh_file = Path(tmp_path, "multiple_groups_per_element.msh")
    gmsh.write(str(msh_file))
    gmsh.finalize()

    meshes = meshes_from_gmsh(msh_file)
    assert len(meshes) == 7

    def num_cells(boundary_name: str) -> int:
        return meshes[f"{boundary_name}"].number_of_cells

    names = ["left", "right", "top", "bottom"]
    assert num_cells("boundaries") == sum([num_cells(name) for name in names])
    bot = meshes["bottom"]
    bot_center = meshes["bottom_center"]
    assert bot.number_of_cells == 25
    assert bot_center.number_of_cells == 19
    assert np.all(np.isin(bot_center["bulk_node_ids"], bot["bulk_node_ids"]))
    assert np.all(
        np.isin(bot_center["bulk_element_ids"], bot["bulk_element_ids"])
    )


valid_edge_length = st.floats(
    allow_nan=False, allow_infinity=False, min_value=1e-5, max_value=1e9
)  # max_value: e.g. pore to ocean scale if interpreted as m


valid_edge_number = st.integers(min_value=1, max_value=5)
# low max value for low and consistent runtime, actual max 10.000 (100e6 cells)


@dataclass
class RectInput:
    edge_length: float = 2.0
    n_edge_cells: int = 1
    n_layers: int = 1
    structured: bool = False
    order: int = 1
    version: float | None = None
    mixed_elements: bool = False


rect_strategy = st.builds(
    RectInput,
    edge_length=st.one_of(
        valid_edge_length, st.tuples(valid_edge_length, valid_edge_length)
    ),
    n_edge_cells=st.one_of(
        valid_edge_number, st.tuples(valid_edge_number, valid_edge_number)
    ),
    n_layers=st.integers(min_value=1, max_value=5),
    structured=st.booleans(),
    order=st.sampled_from([1, 2]),
    version=st.one_of(st.none(), st.sampled_from([2.2])),
    mixed_elements=st.booleans(),
)


def is_typical_edge_length(val):
    # size of cell is determined by the smaller component, number of cell increases too much with the larger component
    if isinstance(val, float):
        return True
    if isinstance(val, tuple) and len(val) == 2:
        a, b = val
        return 0.05 <= b / a <= 50
    return False


# below the minimum
@example(rect_p=RectInput(edge_length=9e-6)).xfail(raises=ValueError)
# above the maximum
@example(rect_p=RectInput(edge_length=2e9)).xfail(raises=ValueError)
# below the minimum
@example(rect_p=RectInput(n_edge_cells=0)).xfail(raises=ValueError)
# below the minimum
@example(rect_p=RectInput(n_layers=0)).xfail(raises=ValueError)
@given(
    rect_p=rect_strategy.filter(lambda r: is_typical_edge_length(r.edge_length))
)
@settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    verbosity=Verbosity.normal,
)
def test_rect(tmp_path: Path, rect_p):
    """Property-based test for the function 'rect'. It uses meshes_from_gmsh."""
    msh_file = (
        tmp_path
        / f"rect_{rect_p.edge_length}_{rect_p.n_edge_cells}_{rect_p.n_layers}_{rect_p.structured}_{rect_p.order}_{rect_p.version}_{rect_p.mixed_elements}.msh"
    )

    print(msh_file)

    rect(
        lengths=rect_p.edge_length,
        n_edge_cells=rect_p.n_edge_cells,
        n_layers=rect_p.n_layers,
        structured_grid=rect_p.structured,
        order=rect_p.order,
        mixed_elements=rect_p.mixed_elements,
        out_name=msh_file,
        msh_version=rect_p.version,
    )

    meshes = ot.Meshes.from_gmsh(msh_file, log=False)
    n_meshes = len(meshes)
    msg = f"Expecting {4 + rect_p.n_layers} meshes, got {n_meshes=}."
    assert n_meshes == 4 + rect_p.n_layers, msg


class TestPhysGroups:
    tmp_path = Path(mkdtemp())

    # By default, the gmsh physical group tags translate directly to MaterialIDs
    # With reindex=True, we want to map these tags to incrementing integers
    # starting at 0

    @pytest.mark.parametrize(
        ("reindex", "layer_ids", "mat_ids"), [
        (False, [0], [0]),          (False, [999], [999]),
        (False, [0, 2], [0, 2]),    (False, [4, 8], [4, 8]),
        (True, [0], [0]),           (True, [999], [0]),
        (True, [0, 2], [0, 1]),     (True, [4, 8], [0, 1])
      ]  # fmt:skip
    )
    def test_phys_groups(self, reindex: bool, layer_ids: list, mat_ids: list):
        """Test different setups of physical groups."""
        msh_file = self.tmp_path / "rect.msh"
        rect(
            n_layers=len(layer_ids),
            out_name=msh_file,
            layer_ids=layer_ids,
            mixed_elements=True,
        )
        meshes = meshes_from_gmsh(msh_file, reindex=reindex, log=False)
        mesh = meshes["domain"]
        assert np.all(np.unique(mesh["MaterialIDs"]) == mat_ids)


def test_cuboid(tmp_path: Path):
    """Test different setups of a cuboid mesh."""
    msh_file = Path(tmp_path, "cuboid.msh")
    permutations = product(
        [1.0, 2.0], [1, 2], [1, 2], [True, False],
        [1, 2], [True, False], [None, 2.2],
    )  # fmt: skip
    for (edge_length,  n_edge_cells, n_layers, structured,
        order, mixed_elements, version) in permutations:  # fmt: skip
        # this combination doesn't work (yet?)
        if order == 2 and mixed_elements:
            continue
        cuboid(
            lengths=edge_length, n_edge_cells=n_edge_cells,
            n_layers=n_layers, structured_grid=structured,
            order=order, mixed_elements=mixed_elements,
            out_name=msh_file, msh_version=version,
        )  # fmt: skip
        meshes = meshes_from_gmsh(msh_file, log=False)
        assert len(meshes) == {1: 7, 2: 9}[n_layers]


def run_cli(cmd: str) -> int:
    "Execute the given command in CLI."
    with pytest.MonkeyPatch.context() as context:
        context.setattr(sys, "argv", cmd.split(" "))
        return cli()


@pytest.mark.parametrize(
    ("script", "num_meshes", "version"),
    [
        ("cube_mixed.py", 1, None),
        ("quarter_rectangle_with_hole.py", 11, 2.2),
        ("quarter_rectangle_with_hole.py", 11, 4.1),
        ("line.py", 3, None),
    ],
)
def test_gmsh(tmp_path: Path, script: str, num_meshes: int, version: float):
    os.chdir(tmp_path)
    if version is not None:
        gmsh.initialize()
        gmsh.option.setNumber("Mesh.MshFileVersion", version)
    runpy.run_module(f"ogstools.examples.gmsh.{Path(script).stem}")
    prefix = str(Path(script).stem)
    msh_file = Path(tmp_path, prefix + ".msh")
    assert len(meshes_from_gmsh(msh_file, log=False)) == num_meshes
    assert run_cli(f"msh2vtu {msh_file} -o {tmp_path} -p {prefix}") == 0

    for vtu_file in tmp_path.glob("*.vtu"):
        try:
            meshio.read(vtu_file)
        except Exception:
            msg = "Generated vtu-files are erroneous."
            raise ValueError(msg) from None


def test_subdomains_2D():
    "Test explicitly the correct number of cells and coordinates in subdomains"
    meshes = meshes_from_gmsh(msh_geolayers_2d, log=False)
    bounds = meshes["domain"].bounds
    boundaries = {
        "Bottom": (120, 1, bounds[2]),
        "Left": (8, 0, bounds[0]),
        "Right": (8, 0, bounds[1]),
        "Top": (120, 1, bounds[3]),
    }
    for name, (ref_num_cells, coord, coord_value) in boundaries.items():
        mesh = meshes[f"{name}"]
        assert ref_num_cells == mesh.number_of_cells
        assert np.all(mesh.points[:, coord] == coord_value)

    subdomains = {
        "SedimentLayer1": (69, [0.0, -623.7103154035065, 0.0]),
        "SedimentLayer2": (146, [0.0, -1248.505178045855, 0.0]),
        "SedimentLayer3": (230, [0.0, -1873.982884659469, 0.0]),
    }
    for name, (ref_num_cells, ref_center) in subdomains.items():
        mesh = meshes[f"{name}"]
        assert ref_num_cells == mesh.number_of_cells
        np.testing.assert_allclose(ref_center, mesh.center)


def test_subdomains_3D():
    "Test explicitly the correct number of cells and coordinates in subdomains"
    meshes = meshes_from_gmsh(msh_geoterrain_3d, log=False)

    boundaries = {
        "BottomSouthLine": (20, [0.5, 0.0, -0.5]),
        "BottomSurface": (944, [0.5, 0.5, -0.5]),
        "WestSurface": (520, [0.0, 0.5, -0.22519078490665562]),
        "EastSurface": (497, [1.0, 0.5, -0.22502145354703557]),
        "TopSouthLine": (22, [0.5, 0.0, 0.0001775511298476931]),
        "TopSurface": (1176, [0.5, 0.5, -1.3476350030128259e-05]),
    }
    for name, (ref_num_cells, ref_center) in boundaries.items():
        mesh = meshes[f"{name}"]
        assert ref_num_cells == mesh.number_of_cells
        np.testing.assert_allclose(ref_center, mesh.center)
