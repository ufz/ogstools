"""
Created on Tue Mar 14 2023

@author: heinzej
"""

import logging as log
from argparse import ArgumentParser
from pathlib import Path

import ifm_contrib as ifm
from ogs6py import ogs

from ogstools.feflowlib import (
    combine_material_properties,
    convert_geometry_mesh,
    deactivate_cells,
    extract_cell_boundary_conditions,
    helpFormat,
    liquid_flow,
    setup_prj_file,
    steady_state_diffusion,
    update_geometry,
    write_mesh_of_combined_properties,
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
        "OGS_steady_state_diffusion",
        "OGS_liquid_flow",
    ],
    default="OGS_steady_state_diffusion",
    type=str,
    help="Different cases can be chosen for the conversion: \n"
    '1. "geometry" to convert only the geometries of the mesh.\n'
    '2. "properties" to convert all the mesh properties to nodes and cells.\n'
    '3. "surface" to convert only the surface of the mesh.\n'
    '4. "properties_surface" to convert the surface with properties.\n'
    '5. "OGS_steady_state_diffusion" to prepare a OGS-project according to a steady state diffusion process.\n'
    '6. "OGS_liquid_flow" to prepare a OGS-project according to a liquid flow process.\n',
    nargs="?",
    const=1,
)

parser.add_argument(
    "BC",
    choices=["BC", "no_BC"],
    default="BC",
    type=str,
    help="This argument specifies whether the boundary conditions is written. It\n"
    "should only be used if the input data is 3D.\n"
    "The boundary condition need to be extracted, when a OGS simulation wants to be setup.",
    nargs="?",
    const=1,
)


# log configuration
logger = log.getLogger(__name__)


def feflow_converter(input: str, output: str, case: str, BC: str):
    """
    This function summarizes main functionality of the feflowlib. It show examplary how a
    workflow could look like to achieve the conversion of FEFLOW data to a vtu-file.

    :param input: input path to FEFLOW data
    :type input: str
    :param output: output path of vtu-file
    :type output: str
    :param case: option for conversion process
    :type case: str
    :param BC: option if boundary condition shall be extracted or not
    :type BC: str
    """
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
        "OGS_steady_state_diffusion": "mesh with its properties and boundary condition(s)",
        "OGS_liquid_flow": "mesh with its properties and boundary condition(s)",
    }

    args = parser.parse_args()

    if not Path(args.input).exists():
        print("The input file does not exist.")
        return 1

    doc = ifm.loadDocument(input)

    mesh = convert_geometry_mesh(doc)

    if "properties" in case or "OGS" in case:
        update_geometry(mesh, doc)
    mesh = mesh.extract_surface() if "surface" in case else mesh
    if ("properties" not in case and "OGS" not in case) or BC != "BC":
        mesh.save(output)
        logger.info(
            "The conversion of the %s was successful.",
            msg[case],
        )
        return 0
    # create separate meshes for the boundary condition
    write_point_boundary_conditions(Path(output).parent, mesh)
    path_topsurface, topsurface = extract_cell_boundary_conditions(
        Path(output), mesh
    )
    topsurface.save(path_topsurface)

    log.info(
        "Boundary conditions have been written to separate mesh vtu-files."
    )
    if "OGS" in case:
        # deactivate cells, all the cells that are inactive in FEFLOW, will be assigned to a
        # the same MaterialID multiplied by -1.
        if deactivate_cells(mesh):
            log.info(
                "There are inactive cells in FEFLOW, which are assigned to a MaterialID multiplied by -1 in the converted bulk mesh."
            )
        # create a prj-file, which is not complete. Manual extensions are needed.
        property_list = ["P_CONDX", "P_CONDY", "P_CONDZ"]
        material_properties = combine_material_properties(mesh, property_list)
        for material_id, property_value in material_properties.items():
            if any(prop == "inhomogeneous" for prop in property_value):
                write_mesh_of_combined_properties(
                    mesh,
                    property_list,
                    "KF",
                    material_id,
                    Path(output),
                )
        if "steady_state_diffusion" in case:
            template_model = steady_state_diffusion(
                str(Path(output).name),
                ogs.OGS(PROJECT_FILE=str(Path(output).with_suffix(".prj"))),
            )
            process = "steady state diffusion"
        elif "liquid_flow" in case:
            template_model = liquid_flow(
                str(Path(output).name),
                ogs.OGS(PROJECT_FILE=str(Path(output).with_suffix(".prj"))),
            )
            process = "liquid flow"
        else:
            error_msg = "Either you select 'OGS_steady_state_diffusion' to prepare a OGS project file for a steady state diffusion process or 'OGS_liquid_flow' for a liquid flow process."
            raise ValueError(error_msg)

        ogs_model = setup_prj_file(
            Path(output),
            mesh,
            material_properties,
            process,
            template_model,
        )

        ogs_model.write_input()
        log.info("A prj file has been created for the %s process.", process)
    mesh.save(output)
    logger.info(
        "The conversion of the %s was successful.",
        msg[case],
    )
    return 0


def cli():
    args = parser.parse_args()
    feflow_converter(args.input, args.output, args.case, args.BC)
