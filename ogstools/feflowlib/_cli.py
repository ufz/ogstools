# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

"""
Created on Tue Mar 14 2023

@author: heinzej
"""

import logging as log
from argparse import ArgumentParser
from pathlib import Path

import ifm_contrib as ifm

from ogstools.feflowlib import FeflowModel

from ._tools import helpFormat

parser = ArgumentParser(
    description="This tool converts FEFLOW binary files to VTK format.",
    formatter_class=helpFormat,
)

parser.add_argument("-i", "--input", help="The path to the input FEFLOW file.")
parser.add_argument(
    "-o", "--output", help="The path to the output VTK file.", default=None
)
parser.add_argument(
    "case",
    choices=["bulk_mesh", "OGS"],
    default="bulk_mesh",
    type=str,
    help="Different cases can be chosen for the conversion: \n"
    '1. "bulk_mesh" to convert only the bulk mesh and the boundary meshes.\n'
    '2. "OGS" to prepare an OGS-model either for a steady state diffusion, liquid flow, hydro thermal or component transport process.',
    nargs="?",
    const=1,
)

parser.add_argument(
    "BC",
    choices=["BC", "no_BC"],
    default="BC",
    type=str,
    help="This argument specifies whether the boundary conditions are written. \n"
    "The boundary condition need to be extracted, when an OGS simulation wants to be setup.",
    nargs="?",
    const=1,
)


# log configuration
logger = log.getLogger(__name__)
logger.setLevel(log.INFO)


def feflow_converter(input: str, output: str, case: str, BC: str) -> int:
    """
    This function summarizes main functionality of the feflowlib. It show examplary how a
    workflow could look like to achieve the conversion of FEFLOW data to a vtu-file.

    :param input: input path to FEFLOW data
    :param output: output path of vtu-file
    :param case: option for conversion process
    :param BC: option if boundary condition shall be extracted or not
    :returns: error code if function failed (1) or was successful (0)
    """
    # log feflow version
    logger.info(
        "The converter is working with FEFLOW %s (build %s).",
        ifm.getKernelVersion() / 1000,
        ifm.getKernelRevision(),
    )
    # create file handler which logs even debug messages
    fh = log.FileHandler(output.replace("vtu", "log"))
    fh.setLevel(log.DEBUG)
    logger.addHandler(fh)

    if not Path(input).exists():
        print("The input file does not exist.")
        return 1
    if BC == "no_BC" and case == "OGS":
        logger.error(
            "An OGS model can only be complete if the existing boundary conditions have been converted."
        )
        return 1
    feflow_model = FeflowModel(Path(input), Path(output))
    # Create separate meshes for the boundary condition.
    if BC == "no_BC":
        feflow_model.mesh.save(feflow_model.mesh_path)
        logger.info(
            "The conversion of the bulk mesh was successful.",
        )
        return 0
    for path, boundary_mesh in feflow_model.subdomains.items():
        boundary_mesh.save(path)

    logger.info(
        "Boundary conditions have been written to separate mesh vtu-files."
    )

    if "OGS" in case:
        feflow_model.project.write_input()
        logger.info(
            """
            A prj file has been created for the %s process.\n
            Depending on the purpose and process to be modeled, manual adjustments are needed!
            """,
            feflow_model.process,
        )
    feflow_model.mesh.save(feflow_model.mesh_path)
    logger.info(
        """The conversion of the bulk mesh, boundary conditions was successful.\n
        """,
    )
    return 0


def cli() -> int:
    args = parser.parse_args()
    args.output = args.input if args.output is None else args.output
    print(args.output)
    feflow_converter(args.input, args.output, args.case, args.BC)
    return 0
