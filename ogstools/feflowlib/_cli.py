"""
Created on Tue Mar 14 2023

@author: heinzej
"""

import logging as log
from argparse import ArgumentParser
from pathlib import Path

import ifm_contrib as ifm

from ogstools.feflowlib import (
    convert_geometry_mesh,
    extract_cell_boundary_conditions,
    combine_material_properties,
    deactivate_cells,
    helpFormat,
    combine_material_properties,
    setup_prj_file,
    update_geometry,
    write_point_boundary_conditions,
)

parser = ArgumentParser(
    description="This tool converts FEFLOW binary files to VTK format.",
    formatter_class=helpFormat,
)

parser.add_argument("-i", "--input", help="The path to the input FEFLOW file.")
parser.add_argument("-o", "--output", help="The path to the output VTK file.")
parser.add_argument(
    "case",
    choices=[
        "geo_surface",
        "geometry",
        "properties",
        "properties_surface",
        "prepare_OGS",
    ],
    default="prepare_OGS",
    type=str,
    help="Different cases can be chosen for the conversion: \n"
    '1. "geometry" to convert only the geometries of the mesh.\n'
    '2. "properties" to convert all the mesh properties to nodes and cells.\n'
    '3. "surface" to convert only the surface of the mesh.\n'
    '4. "properties_surface" to convert the surface with properties.\n'
    '5. "prepare_OGS" to prepare a OGS-project file as much as possible.\n',
    nargs="?",
    const=1,
)

parser.add_argument(
    "BC",
    choices=["BC", "no_BC"],
    default="BC",
    type=str,
    help="This argument specifies whether the boundary conditions\n"
    "is extracted and a corresponding xml file is written. It\n"
    "should only be used if the input data consists of 3D.\n"
    "The boundary condition need to be extracted, when a OGS simulation wants to be setup.",
    nargs="?",
    const=1,
)


# log configuration
logger = log.getLogger(__name__)


def cli():
    # log feflow version
    logger.info(
        "The converter is working with FEFLOW %s (build %s).",
        ifm.getKernelVersion() / 1000,
        ifm.getKernelRevision(),
    )
    msg = {
        "geo_surface": "surface",
        "geometry": "geometry",
        "properties_surface": "surface with properties",
        "properties": "mesh with its properties",
        "prepare_OGS": "mesh with its properties and boundary condition",
    }

    args = parser.parse_args()

    if not Path(args.input).exists():
        print("The input file does not exist.")
        return 1

    doc = ifm.loadDocument(args.input)

    mesh = convert_geometry_mesh(doc)

    if "properties" in args.case or "prepare_OGS" in args.case:
        update_geometry(mesh, doc)
    mesh = mesh.extract_surface() if "surface" in args.case else mesh
    if (
        "properties" not in args.case and "prepare_OGS" not in args.case
    ) or args.BC != "BC":
        mesh.save(args.output)
        logger.info(
        "The conversion of the %s was successful.",
        msg[args.case],
        )
        return 0
    # create separate meshes for the boundary condition
    write_point_boundary_conditions(Path(args.output).parent, mesh)
    topsurface = extract_cell_boundary_conditions(Path(args.output), mesh)
    topsurface[1].save(topsurface[0])

    log.info(
        "Boundary conditions have been written to separate mesh vtu-files."
    )
    if "prepare_OGS" in args.case:
        # deactivate cells, all the cells that are inactive in FEFLOW, will be assigned to a
        # the same MaterialID multiplied by -1.
        deactivate_cells(mesh)
        log.info(
            "Inactive cells in FEFLOW are assigned to a MaterialID multiplied by -1."
        )
        # create a prj-file, which is not complete. Manual extensions are needed.
        setup_prj_file(
            Path(args.output),
            mesh,
            combine_material_properties(
                mesh, ["P_CONDX", "P_CONDY", "P_CONDZ"]
            ),
            process="steady state diffusion",
        )
        log.info(
            "A prj file has been created but needs to be completed in order to run an OGS simulation"
        )
    mesh.save(args.output)
    logger.info(
        "The conversion of the %s was successful.",
        msg[args.case],
    )
    return 0
