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

import ogstools.msh2vtu as msh2vtu
import ogstools.msh2vtu.examples.gmsh as examples
from ogstools.msh2vtu._cli import cli


def test_cli():
    subprocess.run(["msh2vtu", "--help"], check=True)


def test_unit_square(tmp_path: Path):
    msh_file = Path(tmp_path, "unit_square.msh")
    permutations = product([0.5, 0.2], [True, False], [1, 2])
    for size, structured, order in permutations:
        examples.unit_square(
            out_name=msh_file,
            structured_grid=structured,
            element_size=size,
            order=order,
        )
        assert msh2vtu.run(msh_file, path=tmp_path, prefix="unit_square") == 0


def test_unit_cube(tmp_path: Path):
    msh_file = Path(tmp_path, "unit_cube.msh")
    permutations = product([1.0, 0.5], [True, False], [1, 2])
    for size, structured, order in permutations:
        examples.unit_cube(
            out_name=msh_file,
            structured_grid=structured,
            element_size=size,
            order=order,
        )
        assert msh2vtu.run(msh_file, path=tmp_path, prefix="unit_cube") == 0


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
        error_code = msh2vtu.run(
            filename=msh_file, path=tmp_path, prefix=prefix
        )
        assert error_code == 0
    testargs = ["msh2vtu", str(msh_file), "-o", str(tmp_path), "-p", prefix]
    with patch.object(sys, "argv", testargs):
        cli()

    for vtu_file in tmp_path.glob("*.vtu"):
        try:
            meshio.read(vtu_file)
        except Exception:
            msg = "Generated vtu-files are erroneous."
            raise ValueError(msg) from None
