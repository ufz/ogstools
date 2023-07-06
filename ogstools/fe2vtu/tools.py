import argparse
import xml.etree.ElementTree as ET

import numpy as np
import pyvista as pv


class helpFormat(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    """
    A helper class for passing the correct format for the CLI arguments.
    """
    pass


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
    # Extract those cells
    surface_mesh.rename_array("vtkOriginalPointIds", "BULK_NODE_ID")
    surface_mesh.rename_array("vtkOriginalCellIds", "BULK_ELEMENT_ID")
    surface_mesh.cell_data.remove("Normals")
    specific_cells = surface_mesh.extract_cells(ids)
    specific_cells.cell_data.remove("vtkOriginalCellIds")
    specific_cells.point_data.remove("vtkOriginalPointIds")
    return specific_cells


def write_xml(
    mesh_name: str, bc_type: str, data: pv.DataSetAttributes, mesh_type: str
):
    """
    Writes two xml-files, one for parameters and one for boundary conditions.

    :param mesh_name: name of the mesh
    :type mesh_name: str
    :param bc_type: type of the boundary condition (Neumann or Dirichlet)
    :type bc_type: str
    :param data: point or cell data
    :type data: pyvista.DataSetAttributes
    :param mesh_type: type of the mesh (MeshNode or MeshElement)
    :type mesh_type: str
    """

    mesh_name = mesh_name.replace(".vtu", "")
    xml_bc = ET.Element("boundary_conditions")
    xml_parameter = ET.Element("parameters")
    for parameter_name in data:
        if mesh_name == "":
            mesh_name = parameter_name
        if (
            parameter_name != "BULK_NODE_ID"
            and parameter_name != "BULK_ELEMENT_ID"
        ):
            bc = ET.SubElement(xml_bc, "boundary_condtion")
            ET.SubElement(bc, "mesh").text = mesh_name
            ET.SubElement(bc, "type").text = bc_type
            ET.SubElement(bc, "parameter").text = parameter_name

            parameter = ET.SubElement(xml_parameter, "parameter")
            ET.SubElement(parameter, "name").text = parameter_name
            ET.SubElement(parameter, "type").text = mesh_type
            ET.SubElement(parameter, "field_name").text = parameter_name
            ET.SubElement(parameter, "mesh").text = mesh_name

    ET.indent(xml_bc, space="\t", level=0)
    ET.ElementTree(xml_bc).write("bc_" + mesh_name + ".xml")
    ET.indent(xml_parameter, space="\t", level=0)
    ET.ElementTree(xml_parameter).write("parameter_" + mesh_name + ".xml")
