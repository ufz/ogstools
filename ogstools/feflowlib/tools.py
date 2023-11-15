import argparse
import logging as log
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyvista as pv

# log configuration
logger = log.getLogger(__name__)


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
    # Get list of cell IDs that meet the filter condition
    ids = np.arange(surface_mesh.n_cells)[
        filter_condition(surface_mesh["Normals"])
    ]
    surface_mesh.cell_data.remove("Normals")
    # Extract cells that meet the filter condition
    return surface_mesh.extract_cells(ids)


def write_xml(out_mesh_path: Path, data: pv.DataSetAttributes, mesh_type: str):
    """
    Writes three xml-files, one for parameters, one for boundary conditions and one for meshes (geometry).

    :param out_mesh_path: name of the mesh
    :type out_mesh_path: Path
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
        "P_IOFLOW": "Neumann",
        "P_SOUF": "volumentric source term",
    }
    xml_meshes = ET.Element("meshes")
    ET.SubElement(xml_meshes, "mesh").text = out_mesh_path.stem
    xml_bc = ET.Element("boundary_conditions")
    xml_parameter = ET.Element("parameters")
    for parameter_name in data:
        if out_mesh_path.stem == "":
            out_mesh_path = parameter_name
        if parameter_name not in [
            "bulk_node_ids",
            "bulk_element_ids",
            "vtkOriginalPointIds",
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
    ET.ElementTree(xml_meshes).write(
        out_mesh_path.with_name("mesh_" + out_mesh_path.stem + ".xml")
    )
    ET.indent(xml_bc, space="\t", level=0)
    ET.ElementTree(xml_bc).write(
        out_mesh_path.with_name("BC_" + out_mesh_path.stem + ".xml")
    )
    ET.indent(xml_parameter, space="\t", level=0)
    ET.ElementTree(xml_parameter).write(
        out_mesh_path.with_name("parameter_" + out_mesh_path.stem + ".xml")
    )


def assign_bulk_ids(mesh: pv.UnstructuredGrid):
    """
    Add fields bulk_node_ids and bulk_element_ids to the given bulk mesh.

    :param mesh_name: name of the mesh
    :type mesh_name: str
    """
    # The format must be unsigned integer, as it is required by OGS
    mesh["bulk_node_ids"] = np.arange(mesh.n_points, dtype=np.uint64)
    mesh.cell_data["bulk_element_ids"] = np.arange(
        mesh.n_cells, dtype=np.uint64
    )


def extract_point_boundary_conditions(
    out_mesh_path: Path, mesh: pv.UnstructuredGrid
):
    """
    Returns the point boundary condition of the mesh. It works by iterating all point data and looking for
    data arrays that include the string "_BC". Depending on what follows, it defines the boundary condition type.
    This function also writes then the corresponding xml-files using the function "write_xml"

    :param out_mesh_path: path of the output mesh
    :type out_mesh_path: Path
    :param mesh: mesh
    :type mesh: pyvista.UnstructuredGrid
    """
    dict_of_point_boundary_conditions = {}
    assign_bulk_ids(mesh)
    # extract mesh since boundary condition are on the surface ?! (not safe!)
    mesh = mesh.extract_surface()
    # remove all the point data that are not of interest
    for point_data in mesh.point_data:
        if not all(["_BC" in point_data]) and point_data != "bulk_node_ids":
            mesh.point_data.remove(point_data)
    # remove all points with point data that are of "nan"-value
    for point_data in mesh.point_data:
        if point_data != "bulk_node_ids":
            dirichlet_bool = "_BC_" not in point_data
            filtered_points = mesh.extract_points(
                [not np.isnan(x) for x in mesh[point_data]],
                adjacent_cells=False,
                include_cells=dirichlet_bool,
            )
            # Only selected point data is needed -> clear all cell data instead of the bulk_element_ids
            for cell_data in filtered_points.cell_data:
                if cell_data != "bulk_element_ids":
                    filtered_points.cell_data.remove(cell_data)
            for pt_data in filtered_points.point_data:
                if pt_data != point_data and pt_data != "bulk_node_ids":
                    filtered_points.point_data.remove(pt_data)
            dict_of_point_boundary_conditions[
                str(out_mesh_path / point_data) + ".vtu"
            ] = filtered_points
    # create the xml-file
    write_xml(out_mesh_path, mesh.point_data, "MeshNode")
    return dict_of_point_boundary_conditions


def write_point_boundary_conditions(
    out_mesh_path: Path, mesh: pv.UnstructuredGrid
):
    """
    Writes the point boundary conditions that are returned from 'extract_point_boundary_conditions()'
    """
    point_boundary_conditions_dict = extract_point_boundary_conditions(
        out_mesh_path, mesh
    )
    for path, boundary_condition in point_boundary_conditions_dict.items():
        boundary_condition.save(path)


def extract_cell_boundary_conditions(
    bulk_mesh_path: Path, mesh: pv.UnstructuredGrid
):
    """
    Returns the cell boundary condition of the mesh. It works by iterating all cell data and looking for
    data arrays that include the strings "P_SOUF" or "P_IOFLOW".
    This function also writes then the corresponding xml-files using the function "write_xml".
    +++WARNING+++: This function still in a experimental state since it is not clear how exactly this function will
    be used in the future.
    TODO: Allow a generic definition of the normal vector for the filter condition.

    :param bulk_mesh_path: name of the mesh
    :type bulk_mesh_path: Path
    :param mesh: mesh
    :type mesh: pyvista.UnstructuredGrid
    """
    assign_bulk_ids(mesh)
    if mesh.volume != 0:
        # get the topsurface since there are the cells of interest
        topsurf = get_specific_surface(
            mesh.extract_surface(), lambda normals: normals[:, 2] > 0
        )
    else:
        topsurf = mesh
    # Only selected cell data is needed -> clear all point data instead of the bulk_node_ids
    for cd in [
        cell_data
        for cell_data in topsurf.cell_data
        if cell_data not in ["P_SOUF", "P_IOFLOW", "bulk_element_ids"]
    ]:
        topsurf.cell_data.remove(cd)
    for pt_data in topsurf.point_data:
        if pt_data != "bulk_node_ids":
            topsurf.point_data.remove(pt_data)
    # create the xml-file
    write_xml(
        bulk_mesh_path.with_stem("topsurface_" + bulk_mesh_path.stem),
        topsurf.cell_data,
        "MeshElement",
    )
    return (
        bulk_mesh_path.with_stem("topsurface_" + bulk_mesh_path.stem),
        topsurf,
    )


def include_xml_snippet_in_prj_file(
    in_prj_file: str, out_prj_file: str, xml_snippet: str
):
    """
    Includes an xml snippet in a project-file. It only works if there is already a subelement in
    the project file that has the same tag/name as the root element of the xml-snippet to be included.

    :param in_prj_file: path of the input project-file
    :type in_prj_file: str
    :param out_prj_file: path of the output project-file
    :type out_prj_file: str
    :param xml_snippet: path of the xml-snippet
    :type xml_snippet: str
    """
    tree = ET.parse(in_prj_file)
    root = tree.getroot()
    # parse the XML file to be included
    include_tree = ET.parse(xml_snippet)
    include_root = include_tree.getroot()
    subelement = root.find(include_root.tag)

    # add the include root as a child of the subelement
    subelement.extend(include_root)  # type: ignore[union-attr]

    # write the modified tree to a file
    tree.write(out_prj_file)


def write_material_properties_to_xml(material_properties: dict):
    """
    Writes the material properties of a model that has data arrays according to FEFLOW syntax, as an xml snippet.
    This xml snippet can be included to OGS prj-file to set up a simulation.

    :param material_properties: properties referring to materials
    :type material_properties: dict
    """
    root = ET.Element("media")
    for material_id in material_properties:
        medium = ET.SubElement(root, "medium", {"id": str(material_id)})
        properties = ET.SubElement(medium, "properties")
        diffusion = ET.SubElement(properties, "property")
        reference_temperature = ET.SubElement(properties, "property")
        ET.SubElement(diffusion, "name").text = "diffusion"
        ET.SubElement(diffusion, "type").text = "Constant"
        ET.SubElement(diffusion, "value").text = str(
            material_properties[material_id]
        )
        ET.SubElement(
            reference_temperature, "name"
        ).text = "reference_temperature"
        ET.SubElement(reference_temperature, "type").text = "Constant"
        ET.SubElement(reference_temperature, "value").text = "293.15"

    ET.indent(root, space="\t", level=0)
    ET.ElementTree(root).write("material_properties.xml")


def get_material_properties(mesh: pv.UnstructuredGrid, property: str):
    """
    Get the material properties of the mesh converted from FEFLOW. There are several methods available
    to access the material properties. Either they are accessible with the FEFLOW API(ifm) or with brute-force methods,
    which check each element, like this function.

    :param mesh: mesh
    :type mesh: pyvista.UnstructuredGrid
    :param property: property
    :type property: str
    """
    material_ids = mesh.cell_data["MaterialIDs"]
    material_properties = {}
    # At the moment only properties named 'P_CONDX', 'P_CONDY', 'P_CONDZ' can be used.
    assert property in ["P_CONDX", "P_CONDY", "P_CONDZ"]
    for material_id in np.unique(material_ids):
        indices = np.where(material_ids == material_id)
        property_of_material = mesh.cell_data[property][indices]
        all_properties_equal = np.all(
            property_of_material == property_of_material[0]
        )
        if all_properties_equal:
            # Here it is divided by 86400 because in FEFLOW the unit is in m/d and not m/s
            # WARNING: This is not a generic method at the moment. A dictionary with all the
            # FEFLOW units is needed to know the conversion to SI-units as they are used in OGS
            material_properties[material_id] = [property_of_material[0] / 86400]
        else:
            material_properties[material_id] = ["inhomogeneous"]
            logger.info(
                "The property %s in material %s is inhomogeneously distributed.",
                property,
                material_id,
            )

    return material_properties


def combine_material_properties(
    mesh: pv.UnstructuredGrid, properties_list: list
):
    """
    Combine multiple material properties. The combined properties are returned
    as list of values in a dictionary.

    :param mesh: mesh
    :type mesh: pyvista.UnstructuredGrid
    :param properties_list: list of properties to be combined
    :type properties_list: list
    """
    # Use a default dict because it allows to extend the values in the list.
    # Also it initializes the value if there is an empty list.
    material_properties: defaultdict[str, list[float]] = defaultdict(list)

    for property in properties_list:
        for material_id, property_value in get_material_properties(
            mesh, property
        ).items():
            material_properties[material_id].extend(property_value)

    return material_properties


def write_mesh_of_combined_properties(
    mesh: pv.UnstructuredGrid,
    property_list: list,
    new_property: str,
    material_id: int,
    saving_path: Path,
):
    """
    Writes a separate mesh-file with a specific material that has inhomogeneous property values
    within the material group. It can also be used to write multiple properties
    into a "new property" data array. For example, write a data array for a tensor defined by
    data arrays representing values of different spatial directions. Nevertheless it can still be
    be used to write the inhomogeneous values of a single property into a separate mesh-file.

    :param mesh: mesh
    :type mesh: pyvista.UnstructuredGrid
    :param property_list: list of properties
    :type property_list: list
    :param new_property: name of the combined properties
    :type new_property: str
    :param material: material with inhomogeneous properties
    :type material: int
    :param saving_path: path to save the mesh
    :type saving_path: Path
    """
    mask = mesh.cell_data["MaterialIDs"] == material_id
    material_mesh = mesh.extract_cells(mask)
    zipped = list(zip(*[material_mesh[prop] for prop in property_list]))
    material_mesh[new_property] = zipped
    filename = str(saving_path.with_name(str(material_id) + ".vtu"))
    material_mesh.save(filename)
    return filename


def deactivate_cells(mesh: pv.UnstructuredGrid):
    """
    Multiplies the MaterialID of all cells that are inactive in FEFLOW by -1.
    Therefore, the input mesh is modified.
    :param mesh: mesh
    :type mesh: pyvista.UnstructuredGrid
    """
    inactive_cells = np.where(mesh.cell_data["P_INACTIVE_ELE"] == 0)
    mesh.cell_data["MaterialIDs"][inactive_cells] *= -1
