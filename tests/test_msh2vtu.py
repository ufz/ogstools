"""
Tests (pytest) for msh2vtu and meshes_from_gmsh
"""

import os
import runpy
import sys
from itertools import product
from pathlib import Path
from tempfile import mkdtemp
from unittest.mock import patch

import gmsh
import meshio
import numpy as np
from parameterized import parameterized

from ogstools import meshes_from_gmsh
from ogstools.examples import msh_geolayers_2d, msh_geoterrain_3d
from ogstools.meshlib.gmsh_meshing import cuboid, rect
from ogstools.msh2vtu._cli import cli


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
        return meshes[f"physical_group_{boundary_name}"].number_of_cells

    names = ["left", "right", "top", "bottom"]
    assert num_cells("boundaries") == sum([num_cells(name) for name in names])
    bot = meshes["physical_group_bottom"]
    bot_center = meshes["physical_group_bottom_center"]
    assert bot.number_of_cells == 25
    assert bot_center.number_of_cells == 19
    assert np.all(np.isin(bot_center["bulk_node_ids"], bot["bulk_node_ids"]))
    assert np.all(np.isin(bot_center["bulk_elem_ids"], bot["bulk_elem_ids"]))


def test_rect(tmp_path: Path):
    """Create different setups of a rectangular mesh."""
    msh_file = Path(tmp_path, "rect.msh")
    permutations = product(
        [1.0, 2.0], [1, 2], [1, 2], [True, False],
        [1, 2], [None, 2.2], [True, False],
    )  # fmt: skip
    for (edge_length, n_edge_cells, n_layers, structured,
        order, version, mixed_elements) in permutations:  # fmt: skip
        rect(
            lengths=edge_length, n_edge_cells=n_edge_cells,
            n_layers=n_layers, structured_grid=structured,
            order=order, mixed_elements=mixed_elements,
            out_name=msh_file, msh_version=version,
        )  # fmt: skip
        assert len(meshes_from_gmsh(msh_file)) == 4 + n_layers


class TestPhysGroups:
    tmp_path = Path(mkdtemp())

    # By default, the gmsh physical group tags translate directly to MaterialIDs
    # With reindex=True, we want to map these tags to incrementing integers
    # starting at 0
    PHYS_GROUPS_TEST_ARGS = (
        (False, [0], [0]),          (False, [999], [999]),
        (False, [0, 2], [0, 2]),    (False, [4, 8], [4, 8]),
        (True, [0], [0]),           (True, [999], [0]),
        (True, [0, 2], [0, 1]),     (True, [4, 8], [0, 1])
    )  # fmt:skip

    @parameterized.expand(PHYS_GROUPS_TEST_ARGS)
    def test_phys_groups(self, reindex: bool, layer_ids: list, mat_ids: list):
        """Test different setups of physical groups."""
        msh_file = Path(self.tmp_path, "rect.msh")
        rect(
            n_layers=len(layer_ids),
            out_name=msh_file,
            layer_ids=layer_ids,
            mixed_elements=True,
        )
        meshes = meshes_from_gmsh(msh_file, reindex=reindex)
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
        meshes = meshes_from_gmsh(msh_file)
        assert len(meshes) == {1: 7, 2: 9}[n_layers]


def run_cli(cmd: str) -> int:
    "Execute the given command in CLI."
    with patch.object(sys, "argv", cmd.split(" ")):
        return cli()


def test_gmsh(tmp_path: Path):
    os.chdir(tmp_path)
    for script, num_meshes, version in [
        ("cube_mixed.py", 1, None),
        ("quarter_rectangle_with_hole.py", 11, 2.2),
        ("quarter_rectangle_with_hole.py", 11, 4.1),
        ("line.py", 4, None),
    ]:
        if version is not None:
            gmsh.initialize()
            gmsh.option.setNumber("Mesh.MshFileVersion", version)
        runpy.run_module(f"ogstools.examples.gmsh.{Path(script).stem}")
        prefix = str(Path(script).stem)
        msh_file = Path(tmp_path, prefix + ".msh")
        assert len(meshes_from_gmsh(msh_file)) == num_meshes
        assert run_cli(f"msh2vtu {msh_file} -o {tmp_path} -p {prefix}") == 0

    for vtu_file in tmp_path.glob("*.vtu"):
        try:
            meshio.read(vtu_file)
        except Exception:
            msg = "Generated vtu-files are erroneous."
            raise ValueError(msg) from None


def test_subdomains_2D():
    "Test explicitly the correct number of cells and coordinates in subdomains"
    meshes = meshes_from_gmsh(msh_geolayers_2d)
    bounds = meshes["domain"].bounds
    boundaries = {
        "Bottom": (120, 1, bounds[2]),
        "Left": (8, 0, bounds[0]),
        "Right": (8, 0, bounds[1]),
        "Top": (120, 1, bounds[3]),
    }
    for name, (ref_num_cells, coord, coord_value) in boundaries.items():
        mesh = meshes[f"physical_group_{name}"]
        assert ref_num_cells == mesh.number_of_cells
        assert np.all(mesh.points[:, coord] == coord_value)

    subdomains = {
        "SedimentLayer1": (69, [0.0, -623.7103154035065, 0.0]),
        "SedimentLayer2": (146, [0.0, -1248.505178045855, 0.0]),
        "SedimentLayer3": (230, [0.0, -1873.982884659469, 0.0]),
    }
    for name, (ref_num_cells, ref_center) in subdomains.items():
        mesh = meshes[f"physical_group_{name}"]
        assert ref_num_cells == mesh.number_of_cells
        np.testing.assert_allclose(ref_center, mesh.center)


def test_subdomains_3D():
    "Test explicitly the correct number of cells and coordinates in subdomains"
    meshes = meshes_from_gmsh(msh_geoterrain_3d)

    boundaries = {
        "BottomSouthLine": (20, [0.5, 0.0, -0.5]),
        "BottomSurface": (944, [0.5, 0.5, -0.5]),
        "WestSurface": (520, [0.0, 0.5, -0.22519078490665562]),
        "EastSurface": (497, [1.0, 0.5, -0.22502145354703557]),
        "TopSouthLine": (22, [0.5, 0.0, 0.0001775511298476931]),
        "TopSurface": (1176, [0.5, 0.5, -1.3476350030128259e-05]),
    }
    for name, (ref_num_cells, ref_center) in boundaries.items():
        mesh = meshes[f"physical_group_{name}"]
        assert ref_num_cells == mesh.number_of_cells
        np.testing.assert_allclose(ref_center, mesh.center)
