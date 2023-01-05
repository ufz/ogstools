import argparse

from ogstools import __version__
from ogstools.msh2vtu import run


def cli():
    """command line use"""

    # parsing command line arguments
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(
        description=(
            "Prepares a Gmsh-mesh for use in OGS by extracting domain-,"
            " boundary- and physical group-submeshes, and saves them in"
            " vtu-format. Note that all mesh entities should belong to physical"
            " groups."
        ),
    )
    parser.add_argument("filename", help="Gmsh mesh file (*.msh) as input data")
    parser.add_argument(
        "-g",
        "--ogs",
        action="store_true",
        help=(
            'rename "gmsh:physical" to "MaterialIDs" for domains and change '
            "type of corresponding cell data to INT32"
        ),
    )
    parser.add_argument(
        "-r",
        "--rdcd",
        action="store_true",
        help=(
            "renumber domain cell data, physical IDs (cell data) of domains "
            "get numbered beginning with zero"
        ),
    )
    parser.add_argument(
        "-a",
        "--ascii",
        action="store_true",
        help="save output files (*.vtu) in ascii format",
    )
    parser.add_argument(
        "-d",
        "--dim",
        type=int,
        default=0,
        help=(
            "spatial dimension (1, 2 or 3), trying automatic detection, "
            "if not given"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default="",
        help=(
            "basename of output files; if not given, then it defaults to"
            " basename of inputfile"
        ),
    )
    parser.add_argument(
        "-z",
        "--delz",
        action="store_true",
        help=(
            "deleting z-coordinate, for 2D-meshes with z=0, note that"
            " vtu-format requires 3D points"
        ),
    )
    parser.add_argument(
        "-s",
        "--swapxy",
        action="store_true",
        help="swap x and y coordinate",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"msh2vtu (part of ogstools {__version__}, Dominik Kern)",
    )

    args = parser.parse_args()

    ErrorCode = run(args)
    if ErrorCode == 0:
        print("msh2vtu successfully finished")
    else:
        print("msh2vtu stopped with errors")
