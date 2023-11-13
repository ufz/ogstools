"""
Tests (pytest) for msh2vtu
"""
import os
import runpy
import subprocess
import sys
from itertools import product
from pathlib import Path
from unittest.mock import patch

import meshio

from ogstools import meshlib, msh2vtu
from ogstools.msh2vtu._cli import cli


def test_cli():
    subprocess.run(["msh2vtu", "--help"], check=True)


def test_rect(tmp_path: Path):
    """Create rectangular gmsh meshes andconvert with msh2vtu."""
    msh_file = Path(tmp_path, "rect.msh")
    permutations = product([1.0, 2.0], [1, 2], [True, False], [1, 2])
    for edge_length, n_edge_cells, structured, order in permutations:
        meshlib.gmsh_meshing.rect(
            lengths=edge_length,
            n_edge_cells=n_edge_cells,
            structured_grid=structured,
            order=order,
            out_name=msh_file,
        )
        assert msh2vtu.msh2vtu(msh_file, tmp_path, output_prefix="rect") == 0


def test_cuboid(tmp_path: Path):
    """Create rectangular gmsh meshes andconvert with msh2vtu."""
    msh_file = Path(tmp_path, "cuboid.msh")
    permutations = product([1.0, 2.0], [1, 2], [True, False], [1, 2])
    for edge_length, n_edge_cells, structured, order in permutations:
        meshlib.gmsh_meshing.cuboid(
            lengths=edge_length,
            n_edge_cells=n_edge_cells,
            structured_grid=structured,
            order=order,
            out_name=msh_file,
        )
        assert msh2vtu.msh2vtu(msh_file, tmp_path, output_prefix="cuboid") == 0


def test_gmsh(tmp_path: Path):
    os.chdir(tmp_path)
    for script in [
        "cube_mixed.py",
        # no script for square_with_circular_hole.msh
        "quarter_rectangle_with_hole.py",
        "line.py",
    ]:
        runpy.run_module(f"ogstools.msh2vtu.examples.gmsh.{Path(script).stem}")
        prefix = str(Path(script).stem)
        msh_file = Path(tmp_path, prefix + ".msh")
        assert msh2vtu.msh2vtu(msh_file, tmp_path, output_prefix=prefix) == 0
    testargs = ["msh2vtu", str(msh_file), "-o", str(tmp_path), "-p", prefix]
    with patch.object(sys, "argv", testargs):
        cli()

    for vtu_file in tmp_path.glob("*.vtu"):
        try:
            meshio.read(vtu_file)
        except Exception:
            msg = "Generated vtu-files are erroneous."
            raise ValueError(msg) from None
