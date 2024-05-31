"""
Tests (pytest) for msh2vtu
"""

import os
import runpy
import sys
from itertools import product
from pathlib import Path
from unittest.mock import patch

import gmsh
import meshio
import numpy as np
import pyvista as pv

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

    def number_of_elements(boundary_name: str) -> int:
        file = f"{model_name}_physical_group_{boundary_name}.vtu"
        return pv.read(str(Path(tmp_path, file))).number_of_cells

    assert number_of_elements("boundaries") == sum(
        [
            number_of_elements(name)
            for name in ["left", "right", "top", "bottom"]
        ]
    )
    bottom = pv.read(
        str(Path(tmp_path, f"{model_name}_physical_group_bottom.vtu"))
    )
    bottom_center = pv.read(
        str(Path(tmp_path, f"{model_name}_physical_group_bottom_center.vtu"))
    )
    assert np.all(
        np.in1d(bottom_center["bulk_node_ids"], bottom["bulk_node_ids"])
    )
    assert np.all(
        np.in1d(bottom_center["bulk_elem_ids"], bottom["bulk_elem_ids"])
    )


def test_rect(tmp_path: Path):
    """Create rectangular gmsh meshes and convert with msh2vtu."""
    msh_file = Path(tmp_path, "rect.msh")
    permutations = product(
        [1.0, 2.0], [1, 2], [True, False], [1, 2], [None, 2.2]
    )
    for edge_length, n_edge_cells, structured, order, version in permutations:
        gmsh_meshing.rect(
            lengths=edge_length,
            n_edge_cells=n_edge_cells,
            structured_grid=structured,
            order=order,
            out_name=msh_file,
            msh_version=version,
        )
        assert msh2vtu(msh_file, tmp_path, output_prefix="rect") == 0


def test_cuboid(tmp_path: Path):
    """Create rectangular gmsh meshes and convert with msh2vtu."""
    msh_file = Path(tmp_path, "cuboid.msh")
    permutations = product(
        [1.0, 2.0], [1, 2], [True, False], [1, 2], [None, 2.2]
    )
    for edge_length, n_edge_cells, structured, order, version in permutations:
        gmsh_meshing.cuboid(
            lengths=edge_length,
            n_edge_cells=n_edge_cells,
            structured_grid=structured,
            order=order,
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
