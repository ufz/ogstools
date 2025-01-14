# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

import logging
from pathlib import Path

import pyvista as pv

from ogstools.meshlib.gmsh_converter import meshes_from_gmsh

logging.basicConfig()  # Important, initializes root logger


def msh2vtu(
    filename: Path,
    output_path: Path = Path(),
    output_prefix: str = "",
    dim: int | list[int] = 0,
    delz: bool = False,
    reindex: bool = False,
    swapxy: bool = False,
    ascii: bool = False,
    log: bool = True,
) -> list[Path]:
    """
    Convert a gmsh mesh (.msh) to unstructured grid files (.vtu).

    Prepares a Gmsh-mesh for use in OGS by extracting domain-,
    boundary- and physical group-submeshes, and saves them in
    vtu-format. Note that all mesh entities should belong to
    physical groups.

    :param filename:    Gmsh mesh file (.msh) as input data
    :param output_path: Path of output files, defaults to current working dir
    :param output_prefix: Output files prefix, defaults to basename of inputfile
    :param dim: Spatial dimension (1, 2 or 3), trying automatic detection,
                if not given. If multiple dimensions are provided, all elements
                of these dimensions are embedded in the resulting domain mesh.
    :param delz:    Set z-coordinate to 0.
    :param reindex: Renumber physical group / region / Material IDs to be
                    renumbered beginning with zero.
    :param swapxy:  Swap x and y coordinate
    :param ascii:   Save output files (.vtu) in ascii format.
    :param log:     If False, silence log messages

    :returns: Filepaths of the written meshes.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO if log else logging.ERROR)

    meshes = meshes_from_gmsh(filename, output_prefix, dim, reindex, log)

    mesh_paths = []
    for name, mesh in meshes.items():
        # meshio.write(Path(output_path, name + ".vtu"), mesh, binary=not ascii)
        if delz:
            mesh.points[:, 2] = 0.0
        if swapxy:
            mesh.points[:, [0, 1]] = mesh.points[:, [1, 0]]

        # ATTENTION:
        # DO NOT use the default mesh.save method if you want to use the meshes
        # for simulation with OGS. This uses the VTK definitions for the order
        # of nodes in cells. Using save_meshio uses a different definition which
        # from meshio, which is the same as in OGS.
        logger.info("Writing mesh %s.", name)
        mesh_path = Path(output_path, name + ".vtu")
        pv.save_meshio(mesh_path, mesh, binary=not ascii)
        mesh_paths += [Path(output_path, name + ".vtu")]
    logger.info("Finished.")

    return mesh_paths
