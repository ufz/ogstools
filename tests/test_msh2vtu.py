"""
Tests (pytest) for msh2vtu
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
import pyvista as pv
from parameterized import parameterized

from ogstools.meshlib import gmsh_meshing
from ogstools.msh2vtu import msh2vtu
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
    gmsh.model.geo.addPhysicalGroup(dim=2, tags=[1], name="domain")
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
    model_name = "multiple_groups_per_element"
    msh_file = Path(tmp_path, model_name + ".msh")
    gmsh.write(str(msh_file))
    gmsh.finalize()

    assert msh2vtu(msh_file, tmp_path, output_prefix=model_name) == 0

    prefix = f"{model_name}_physical_group"

    def number_of_elements(boundary_name: str) -> int:
        file = f"{prefix}_{boundary_name}.vtu"
        return pv.read(str(Path(tmp_path, file))).number_of_cells

    assert number_of_elements("boundaries") == sum(
        [
            number_of_elements(name)
            for name in ["left", "right", "top", "bottom"]
        ]
    )
    bottom = pv.read(str(Path(tmp_path, f"{prefix}_bottom.vtu")))
    bottom_center = pv.read(str(Path(tmp_path, f"{prefix}_bottom_center.vtu")))
    assert np.all(
        np.isin(bottom_center["bulk_node_ids"], bottom["bulk_node_ids"])
    )
    assert np.all(
        np.isin(bottom_center["bulk_elem_ids"], bottom["bulk_elem_ids"])
    )


def test_rect(tmp_path: Path):
    """Create rectangular gmsh meshes and convert with msh2vtu."""
    msh_file = Path(tmp_path, "rect.msh")
    permutations = product(
        [1.0, 2.0],
        [1, 2],
        [1, 2],
        [True, False],
        [1, 2],
        [None, 2.2],
        [True, False],
    )
    for (
        edge_length,
        n_edge_cells,
        n_layers,
        structured,
        order,
        version,
        mixed_elements,
    ) in permutations:
        gmsh_meshing.rect(
            lengths=edge_length,
            n_edge_cells=n_edge_cells,
            n_layers=n_layers,
            structured_grid=structured,
            order=order,
            mixed_elements=mixed_elements,
            out_name=msh_file,
            msh_version=version,
        )
        assert msh2vtu(msh_file, tmp_path, output_prefix="rect") == 0


class TestPhysGroups:
    tmp_path = Path(mkdtemp())

    # By default, the gmsh physical group tags translate directly to MaterialIDs
    # With reindex=True, we want to map these tags to incrementing integers
    # starting at 0
    PHYS_GROUPS_TEST_ARGS = (
        (False, [0], [0]),          (False, [999], [999]),
        (False, [0, 2], [0, 2]),    (False, [4, 8], [4, 8]),
        (True, [0], [0]),           (True, [999], [0]),
        (True, [0, 2], [0, 1]),     (True, [4, 8], [0, 1])  # fmt:skip
    )

    @parameterized.expand(PHYS_GROUPS_TEST_ARGS)
    def test_phys_groups(self, reindex: bool, layer_ids: list, mat_ids: list):
        """Create rectangular gmsh meshes and convert with msh2vtu."""
        msh_file = Path(self.tmp_path, "rect.msh")
        # one physical group with tag 0 (default, layer_ids could be omitted)
        gmsh_meshing.rect(
            n_layers=len(layer_ids),
            out_name=msh_file,
            layer_ids=layer_ids,
            mixed_elements=True,
        )
        assert msh2vtu(msh_file, self.tmp_path, "rect", reindex=reindex) == 0
        mesh = pv.read(str(Path(self.tmp_path, "rect_domain.vtu")))
        assert np.all(np.unique(mesh["MaterialIDs"]) == mat_ids)


def test_cuboid(tmp_path: Path):
    """Create rectangular gmsh meshes and convert with msh2vtu."""
    msh_file = Path(tmp_path, "cuboid.msh")
    permutations = product(
        [1.0, 2.0],
        [1, 2],
        [1, 2],
        [True, False],
        [1, 2],
        [True, False],
        [None, 2.2],
    )
    for (
        edge_length,
        n_edge_cells,
        n_layers,
        structured,
        order,
        mixed_elements,
        version,
    ) in permutations:
        # this combination doesn't work with msh2vtu (yet?)
        if order == 2 and mixed_elements:
            continue
        gmsh_meshing.cuboid(
            lengths=edge_length,
            n_edge_cells=n_edge_cells,
            n_layers=n_layers,
            structured_grid=structured,
            order=order,
            mixed_elements=mixed_elements,
            out_name=msh_file,
            msh_version=version,
        )
        assert msh2vtu(msh_file, tmp_path, output_prefix="cuboid") == 0


def run_cli(cmd: str) -> int:
    "Execute the given command in CLI."
    with patch.object(sys, "argv", cmd.split(" ")):
        return cli()


def test_gmsh(tmp_path: Path):
    os.chdir(tmp_path)
    for script in [
        "cube_mixed.py",
        # no script for square_with_circular_hole.msh
        "quarter_rectangle_with_hole.py",
        "line.py",
    ]:
        runpy.run_module(f"ogstools.examples.gmsh.{Path(script).stem}")
        prefix = str(Path(script).stem)
        msh_file = Path(tmp_path, prefix + ".msh")
        assert msh2vtu(msh_file, tmp_path, output_prefix=prefix) == 0
        mesh = pv.read(prefix + "_domain.vtu")
        assert run_cli(f"msh2vtu {msh_file} -o {tmp_path} -p {prefix}") == 0
        mesh_cli = pv.read(prefix + "_domain.vtu")
        assert mesh == mesh_cli

    for vtu_file in tmp_path.glob("*.vtu"):
        try:
            meshio.read(vtu_file)
        except Exception:
            msg = "Generated vtu-files are erroneous."
            raise ValueError(msg) from None
