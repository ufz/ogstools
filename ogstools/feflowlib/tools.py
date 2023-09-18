import argparse
import xml.etree.ElementTree as ET

import numpy as np
import pyvista as pv


class helpFormat(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter
):
    """
    A helper class for passing the correct format for the CLI arguments.
    """


def get_specific_surface(surface_mesh: pv.PolyData, filter_condition):
    """
    Return only cells that match the filter condition for the normals of the
    input-surface mesh. A standard use case could be to extract the cells that
    have a normal in a particular direction, e.g. upward in the z-direction. The
    filter condition would then be: `lambda normals: normals[:, 2] > 0`.

    :param surface_mesh: The surface mesh.
    :type surface_mesh: pyvista.PolyData
    :param filter_condition: A condition to set up the filter for the normals.
    :type filter_condition: Callable [[list], [list]]
    :return: specific_cells
    :rtype: pyvista.UnstructuredGird
    """
    # Compute the normals of the surface mesh
    surface_mesh = surface_mesh.compute_normals(
        cell_normals=True, point_normals=False
    )
    # Get list of cell IDs that meet condition
    ids = np.arange(surface_mesh.n_cells)[
        filter_condition(surface_mesh["Normals"])
    ]
    # Rename cell arrays to satisfy ogs-convention
    surface_mesh.rename_array("vtkOriginalPointIds", "bulk_node_ids")
    surface_mesh.rename_array("vtkOriginalCellIds", "bulk_element_ids")
    surface_mesh.cell_data.remove("Normals")
    # Extract cells that meet the filter condition
    specific_cells = surface_mesh.extract_cells(ids)
    specific_cells.cell_data.remove("vtkOriginalCellIds")
    specific_cells.point_data.remove("vtkOriginalPointIds")
    return specific_cells


def write_xml(mesh_name: str, data: pv.DataSetAttributes, mesh_type: str):
    """
    Writes three xml-files, one for parameters, one for boundary conditions and one for meshes (geometry).

    :param mesh_name: name of the mesh
    :type mesh_name: str
    :param data: point or cell data
    :type data: pyvista.DataSetAttributes
    :param mesh_type: type of the mesh (MeshNode or MeshElement)
    :type mesh_type: str
    """

    BC_type_dict = {
        "_BC_": "Dirichlet",
        "2ND": "Neumann",
        "3RD": "Robin",
        "4TH": "NodalSourceTerm",
        "P_IOFLOW": "Neumann?",
        "P_SOUF": "Neumann?",
    }
    mesh_name = mesh_name.replace(".vtu", "")
    xml_meshes = ET.Element("meshes")
    ET.SubElement(xml_meshes, "mesh").text = mesh_name
    xml_bc = ET.Element("boundary_conditions")
    xml_parameter = ET.Element("parameters")
    for parameter_name in data:
        if mesh_name == "":
            mesh_name = parameter_name
        if parameter_name not in [
            "bulk_node_ids",
            "bulk_element_ids",
            "vtkOriginalPointIds",
            "orig_indices",
        ]:
            ET.SubElement(xml_meshes, "mesh").text = parameter_name

            bc = ET.SubElement(xml_bc, "boundary_condition")
            ET.SubElement(bc, "mesh").text = parameter_name
            ET.SubElement(bc, "type").text = next(
                val
                for key, val in BC_type_dict.items()
                if key in parameter_name
            )
            ET.SubElement(bc, "parameter").text = parameter_name

            parameter = ET.SubElement(xml_parameter, "parameter")
            ET.SubElement(parameter, "name").text = parameter_name
            ET.SubElement(parameter, "type").text = mesh_type
            ET.SubElement(parameter, "field_name").text = parameter_name
            ET.SubElement(parameter, "mesh").text = parameter_name

    ET.indent(xml_meshes, space="\t", level=0)
    ET.ElementTree(xml_meshes).write("mesh_" + mesh_name + ".xml")
    ET.indent(xml_bc, space="\t", level=0)
    ET.ElementTree(xml_bc).write("BC_" + mesh_name + ".xml")
    ET.indent(xml_parameter, space="\t", level=0)
    ET.ElementTree(xml_parameter).write("parameter_" + mesh_name + ".xml")


def write_point_boundary_conditions(mesh_name: str, mesh: pv.UnstructuredGrid):
    """
    Writes the point boundary condition of the mesh. It works by iterating all point data and looking for
    data arrays that include the string "_BC". Depending on what follows, it defines the boundary condition type.
    This function also writes then the corresponding xml-files using the function "write_xml"

    :param mesh_name: name of the mesh
    :type mesh_name: str
    :param mesh: mesh
    :type mesh: pyvista.UnstructuredGrid
    """

    # assign an array with integer of the indices of the original mesh
    mesh["orig_indices"] = np.arange(mesh.n_points, dtype=np.uint64)
    mesh.cell_data["bulk_element_ids"] = np.arange(
        mesh.n_cells, dtype=np.uint64
    )
    # extract mesh since boundary condition are on the surface ?! (not safe!)
    mesh = mesh.extract_surface()
    # remove all the point data that are not of interest
    for point_data in mesh.point_data:
        if not all(["_BC" in point_data]) and point_data != "orig_indices":
            mesh.point_data.remove(point_data)
    # remove all points with point data that are of "nan"-value
    for point_data in mesh.point_data:
        if point_data != "orig_indices":
            filtered_points = mesh.extract_points(
                [not np.isnan(x) for x in mesh[point_data]],
                adjacent_cells=False,
                include_cells=True,
            )
            # Only selected point data is needed -> clear all cell data instead of the bulk_element_ids
            for cell_data in filtered_points.cell_data:
                if cell_data != "bulk_element_ids":
                    filtered_points.cell_data.remove(cell_data)
            # remove data of BC that are of no value for this part of the mesh
            for pt_data in mesh.point_data:
                if pt_data != point_data and pt_data != "orig_indices":
                    filtered_points.point_data.remove(pt_data)

            # Only "bulk_node_ids" can be read by ogs
            filtered_points.rename_array("orig_indices", "bulk_node_ids")
            filtered_points.save(point_data + ".vtu")
            # pv.save_meshio(
            #    point_data + ".vtu", filtered_points, file_format="vtu"
            # )
    # create the xml-file
    write_xml(mesh_name, mesh.point_data, "MeshNode")


def write_cell_boundary_conditions(mesh_name: str, mesh: pv.UnstructuredGrid):
    """
    Writes the cell boundary condition of the mesh. It works by iterating all cell data and looking for
    data arrays that include the strings "P_SOUF" or "P_IOFLOW".
    This function also writes then the corresponding xml-files using the function "write_xml".
    +++WARNING+++: This function still in a experimental state since it is not clear how exactly this function will
    be used in the future.

    :param mesh_name: name of the mesh
    :type mesh_name: str
    :param mesh: mesh
    :type mesh: pyvista.UnstructuredGrid
    """
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
    # TODO: Allow a generic definition of the normal vector for the filter condition.
    topsurf = get_specific_surface(
        BC_mesh.extract_surface(), lambda normals: normals[:, 2] > 0
    )
    topsurf.save("topsurface_" + mesh_name)
    # create the xml-file
    write_xml(
        "topsurface_" + mesh_name,
        topsurf.cell_data,
        "MeshElement",
    )
