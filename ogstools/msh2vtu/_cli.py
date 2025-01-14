# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

import argparse
import logging
from pathlib import Path

import pyvista as pv

from ogstools import __version__
from ogstools.meshlib.gmsh_converter import meshes_from_gmsh

logging.basicConfig()  # Important, initializes root logger


def argparser() -> argparse.ArgumentParser:
    # parsing command line arguments
    def get_help(arg: str) -> str:
        assert meshes_from_gmsh.__doc__ is not None
        return (
            meshes_from_gmsh.__doc__.split(arg + ":")[1]
            .split(":param")[0]
            .strip()
        )

    parser = argparse.ArgumentParser(
        description="""
            Convert a gmsh mesh (.msh) to unstructured grid files (.vtu).

            Prepares a Gmsh-mesh for use in OGS by extracting domain-, boundary-
            and physical group-submeshes, and saves them in vtu-format. Note,
            that all mesh entities should belong to physical groups.
        """
    )
    add_arg = parser.add_argument
    add_arg("filename", help=get_help("filename"))
    add_arg(
        "-o", "--output_path", default="",
        help="Path of output files, defaults to current working dir",
    )  # fmt: skip
    add_arg(
        "-p", "--prefix", default="",
        help="Filename prefix, defaults to basename of inputfile",
    )  # fmt: skip
    add_arg("-d", "--dim", type=int, nargs="*", default=0, help=get_help("dim"))
    add_arg("-z", "--delz", action="store_true", help="Set z-coordinate to 0.")
    add_arg("-s", "--swapxy", action="store_true", help="Swap x and y")
    add_arg("-r", "--reindex", action="store_true", help=get_help("reindex"))
    add_arg("-a", "--ascii", action="store_true", help="Save in ascii format.")
    add_arg("-l", "--log", action="store_true", help=get_help("log"))
    version = f"msh2vtu (part of ogstools {__version__})"
    add_arg("-v", "--version", action="version", version=version)

    return parser


def cli() -> int:
    """command line use"""
    args = argparser().parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO if args.log else logging.ERROR)

    meshes = meshes_from_gmsh(
        filename=args.filename, dim=args.dim, reindex=args.reindex, log=args.log
    )

    output_basename = args.filename.stem if args.prefix == "" else args.prefix

    for name, mesh in meshes.items():
        if args.delz:
            mesh.points[:, 2] = 0.0
        if args.swapxy:
            mesh.points[:, [0, 1]] = mesh.points[:, [1, 0]]
        filename = f"{output_basename}_{name}.vtu"
        logger.info("Writing mesh %s.", filename)
        mesh_path = Path(args.output_path, filename)
        # DO NOT use the default mesh.save method if you want to use the
        # meshes for simulation with OGS. This uses the VTK definitions for
        # the order of nodes in cells. Using save_meshio uses a different
        # definition which from meshio, which is the same as in OGS.
        pv.save_meshio(mesh_path, mesh, binary=not args.ascii)
    logger.info("Finished.")

    return 0
