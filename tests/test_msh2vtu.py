"""
Tests (pytest) for msh2vtu
"""
import argparse
import glob
import os
import runpy
import subprocess
from pathlib import Path

import meshio

import ogstools.msh2vtu as msh2vtu


def test_cli():
    subprocess.run(["msh2vtu", "--help"], check=True)


def test_howto_gmsh(tmp_path):
    working_dir = Path(tmp_path)
    os.chdir(working_dir)

    for script in [
        "cube_tet.py",
        "cube_mixed.py",
        "square_tri.py",
        "square_quad.py",
        # no script for square_with_circular_hole.msh
        "quarter_rectangle_with_hole.py",
        "cube_hex.py",
        "line.py",
    ]:
        runpy.run_module(
            f"ogstools.msh2vtu.examples.howto_gmsh.{Path(script).stem}"
        )
        msh_file = f"{Path(script).stem}.msh"
        args = argparse.Namespace(
            filename=msh_file,
            output=str(Path(msh_file).stem),
            dim=0,
            delz=False,
            swapxy=False,
            rdcd=True,
            ogs=True,
            ascii=False,
        )
        assert msh2vtu.run(args) == 0

    glob_vtu_files = glob.glob("*.vtu")
    for vtu_file in glob_vtu_files:
        ErrorCode = 0
        try:
            meshio.read(vtu_file)
        except Exception:
            ErrorCode = 1
        assert ErrorCode == 0, "Generated vtu-files are erroneous."
