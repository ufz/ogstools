"""
Created on Tue Mar 14 2023

@author: heinzej
"""

import logging as log
from argparse import ArgumentParser, RawTextHelpFormatter
from sys import stdout

import ifm_contrib as ifm
import numpy as np
import pyvista as pv

from ogstools.fe2vtu import (
    get_geo_mesh,
    get_specific_surface,
    update_geo_mesh,
    write_xml,
)

parser = ArgumentParser(
    description="This tool converts FEFLOW binary files to VTK format.",
    formatter_class=RawTextHelpFormatter,
)

parser.add_argument("-i", "--input", help="The path to the input FEFLOW file.")
parser.add_argument("-o", "--output", help="The path to the output VTK file.")
parser.add_argument(
    "case",
    choices=["geo_surface", "geometry", "properties", "properties_surface"],
    default="properties",
    type=str,
    help="Different cases can be chosen for the conversion: \n"
    '- "geometry" to convert only the geometries of the mesh.\n'
    '- "properties" to convert all the mesh properties to nodes and cells.\n'
    '- "surface" to convert only the surface of the mesh.\n'
    '- "properties_surface" to convert the surface with properties.\n'
    ' If none is given, "properties" is taken by default.',
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
    "should only be used if the input data consists of 3D.",
    nargs="?",
    const=1,
)


# log configuration
log.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
    level=log.DEBUG,
    stream=stdout,
    datefmt="%d/%m/%Y %H:%M:%S",
)


def cli():
    args = parser.parse_args()

    doc = ifm.loadDocument(args.input)

    mesh = get_geo_mesh(doc)

    if args.case == "geometry" or args.case == "geo_surface":
        if args.case == "geo_surface":
            # surface
            surf = mesh.extract_surface()
            pv.save_meshio(args.output, surf, file_format="vtu")
            # log feflow version
            log.info(
                "The surface of the input mesh has been successfully converted."
            )
        else:
            pv.save_meshio(args.output, mesh, file_format="vtu")
            log.info(
                "The geometry of the input mesh has been successfully converted."
            )
    elif args.case == "properties" or args.case == "properties_surface":
        update_geo_mesh(mesh, doc)
        if args.case == "properties_surface":
            surf = mesh.extract_surface()
            pv.save_meshio(args.output, surf, file_format="vtu")
            log.info(
                "The surface with properties of the input mesh have been successfully converted."
            )
        else:
            pv.save_meshio(args.output, mesh, file_format="vtu")
            log.info("The input mesh has been successfully converted.")
        if args.BC == "BC":
            BC_mesh = mesh.copy()
            for cd in [
                cell_data
                for cell_data in BC_mesh.cell_data
                if cell_data not in ["P_SOUF", "P_IOFLOW"]
            ]:
                BC_mesh.cell_data.remove(cd)
            # Only cell data are needed
            BC_mesh.point_data.clear()
            # get the topsurface since there are the cells of interest
            topsurf = get_specific_surface(
                BC_mesh.extract_surface(), lambda normals: normals[:, 2] > 0
            )
            topsurf.save("topsurface_" + args.output)
            # create the xml-file
            write_xml(
                "topsurface_" + args.output,
                "Neumann",
                topsurf.cell_data,
                "MeshElement",
            )

            # remove all the point data that are not of interest
            for point_data in mesh.point_data:
                if not all(["_BC_" in point_data]):
                    mesh.point_data.remove(point_data)

            # Only selected point data is needed -> clear all cell data
            mesh.cell_data.clear()

            # remove all points with point data that are of "nan"-value
            for point_data in mesh.point_data:
                filtered_points = mesh.extract_points(
                    [not np.isnan(x) for x in mesh[point_data]],
                    include_cells=False,
                )
                # Only "BULK_NODE_ID" can be read by ogs
                filtered_points.rename_array(
                    "vtkOriginalPointIds", "BULK_NODE_ID"
                )
                filtered_points.save(point_data + ".vtu")

            # create the xml-file
            write_xml("", "Dirichlet", filtered_points.point_data, "MeshNode")
