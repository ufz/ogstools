# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

import sys
from typing import Any

binaries_list = [
    "addDataToRaster",
    "AddElementQuality",
    "AddFaultToVoxelGrid",
    "AddLayer",
    "appendLinesAlongPolyline",
    "AssignRasterDataToMesh",
    "checkMesh",
    "ComputeNodeAreasFromSurfaceMesh",
    "computeSurfaceNodeIDsInPolygonalRegion",
    "constructMeshesFromGeometry",
    "convertGEO",
    "convertToLinearMesh",
    "convertVtkDataArrayToVtkDataArray",
    "CreateBoundaryConditionsAlongPolylines",
    "createIntermediateRasters",
    "createLayeredMeshFromRasters",
    "createMeshElemPropertiesFromASCRaster",
    "createNeumannBc",
    "createQuadraticMesh",
    "createRaster",
    "createTetgenSmeshFromRasters",
    "editMaterialID",
    "ExtractBoundary",
    "ExtractMaterials",
    "ExtractSurface",
    "generateGeometry",
    "generateMatPropsFromMatID",
    "generateStructuredMesh",
    "geometryToGmshGeo",
    "GMSH2OGS",
    "GocadSGridReader",
    "GocadTSurfaceReader",
    "identifySubdomains",
    "IntegrateBoreholesIntoMesh",
    "ipDataToPointCloud",
    "Layers2Grid",
    "MapGeometryToMeshSurface",
    "Mesh2Raster",
    "MoveGeometry",
    "MoveMesh",
    "moveMeshNodes",
    "mpmetis",
    "NodeReordering",
    "ogs",
    "OGS2VTK",
    "partmesh",
    "PVD2XDMF",
    "queryMesh",
    "Raster2Mesh",
    "RemoveGhostData",
    "removeMeshElements",
    "ReorderMesh",
    "ResetPropertiesInPolygonalRegion",
    "reviseMesh",
    "scaleProperty",
    "swapNodeCoordinateAxes",
    "TecPlotTools",
    "TIN2VTK",
    "VTK2OGS",
    "VTK2TIN",
    "vtkdiff",
    "Vtu2Grid",
]


# Not used when OGS_USE_PATH is true!
def ogs() -> None:
    raise SystemExit(ogs_with_args(sys.argv))


def ogs_with_args(argv: Any) -> int:
    import ogs.simulator as sim

    return_code = sim.initialize(argv)

    # map mangled TCLAP status to usual exit status
    if return_code == 3:  # EXIT_ARGPARSE_FAILURE
        sim.finalize()
        return 1  # EXIT_FAILURE
    if return_code == 2:  # EXIT_ARGPARSE_EXIT_OK
        sim.finalize()
        return 0  # EXIT_SUCCESS

    if return_code != 0:
        sim.finalize()
        return return_code

    return_code = sim.executeSimulation()
    sim.finalize()
    return return_code


# if "PEP517_BUILD_BACKEND" not in os.environ:
#     # Here, we assume that this script is installed, e.g., in a virtual environment
#     # alongside a "bin" directory.
#     OGS_BIN_DIR = Path(__file__).parent.parent.parent / "bin"  # installed wheel
#     if not OGS_BIN_DIR.exists():
#         OGS_BIN_DIR = OGS_BIN_DIR.parent  # build directory
#
#     if platform.system() == "Windows":
#         os.add_dll_directory(OGS_BIN_DIR)
#
#     def _program(name, args):
#         exe = OGS_BIN_DIR / name
#         if OGS_USE_PATH:
#             exe = name
#             print(f"OGS_USE_PATH is true: {name} from $PATH is used!")
#         return subprocess.run([exe] + args).returncode
#
#     FUNC_TEMPLATE = (
#         """def {0}(): raise SystemExit(_program("{0}", sys.argv[1:]))"""
#     )
#     for f in binaries_list:
#         if f == "ogs" and not OGS_USE_PATH:
#             continue  # provided by separate function
#         # When OGS_USE_PATH is true then ogs()-function above is not used!
#         exec(FUNC_TEMPLATE.format(f))
#
