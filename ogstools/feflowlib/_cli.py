"""
Created on Tue Mar 14 2023

@author: heinzej
"""

import logging as log
from argparse import ArgumentParser
from pathlib import Path

import ifm_contrib as ifm

from ogstools.feflowlib import (
    helpFormat,
    read_geometry,
    update_geometry,
    write_cell_boundary_conditions,
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
    choices=["geo_surface", "geometry", "properties", "properties_surface"],
    default="properties",
    type=str,
    help="Different cases can be chosen for the conversion: \n"
    '1. "geometry" to convert only the geometries of the mesh.\n'
    '2. "properties" to convert all the mesh properties to nodes and cells.\n'
    '3. "surface" to convert only the surface of the mesh.\n'
    '4. "properties_surface" to convert the surface with properties.\n',
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
    "should only be used if the input data consists of 3D.\n",
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

    args = parser.parse_args()

    if not Path(args.input).exists():
        print("The input files does not exist.")
        return

    doc = ifm.loadDocument(args.input)

    mesh = read_geometry(doc)

    if "properties" in args.case:
        update_geometry(mesh, doc)
    mesh = mesh.extract_surface() if "surface" in args.case else mesh
    msg = {
        "geo_surface": "surface",
        "geometry": "geometry",
        "properties_surface": "surface with properties",
        "properties": "properties",
    }
    mesh.save(args.output)
    # save meshio changes node order -> not compatible with OGS
    # in the future meshio is desired for saving ! -> pv.save_meshio(args.output, mesh)
    logger.info(
        "The conversion of the %s was successful.",
        msg[args.case],
    )
    if "properties" not in args.case or args.BC != "BC":
        return

    write_point_boundary_conditions(Path(args.output), mesh)
    write_cell_boundary_conditions(Path(args.output), mesh)
