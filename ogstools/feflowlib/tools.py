import argparse
import logging as log
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyvista as pv
from ogs6py import ogs

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
    Returns the point boundary conditions of the mesh. It works by iterating all point data and looking for
    data arrays that include the string "_BC". Depending on what follows, it defines the boundary condition type.

    :param out_mesh_path: path of the output mesh
    :type out_mesh_path: Path
    :param mesh: mesh
    :type mesh: pyvista.UnstructuredGrid
    :return: dict_of_point_boundary_conditions
    :rtype: dict
    """
    dict_of_point_boundary_conditions = {}
    assign_bulk_ids(mesh)
    # extract mesh since boundary condition are on the surface ?! (not safe!)
    surf_mesh = mesh.extract_surface()
    # remove all the point data that are not of interest
    for point_data in surf_mesh.point_data:
        if not all(["_BC" in point_data]) and point_data != "bulk_node_ids":
            surf_mesh.point_data.remove(point_data)
    # remove all points with point data that are of "nan"-value
    for point_data in surf_mesh.point_data:
        if point_data != "bulk_node_ids":
            dirichlet_bool = "_BC_" not in point_data
            if "_4TH" in point_data:
                filtered_points = mesh.extract_points(
                    [not np.isnan(x) for x in mesh[point_data]],
                    adjacent_cells=False,
                    include_cells=False,
                )
            else:
                filtered_points = surf_mesh.extract_points(
                    [not np.isnan(x) for x in surf_mesh[point_data]],
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
            # In OGS Neumann and Robin boundary condition have a different sign than in FEFLOW!
            # Also in FEFOW the Neumann BC for flow is in m/d and ogs works with SI-units (m/s)
            dict_of_point_boundary_conditions[
                str(out_mesh_path / point_data) + ".vtu"
            ] = filtered_points
    return dict_of_point_boundary_conditions


def write_point_boundary_conditions(
    out_mesh_path: Path, mesh: pv.UnstructuredGrid
):
    """
    Writes the point boundary conditions that are returned from 'extract_point_boundary_conditions()'

    :param out_mesh_path: path for writing
    :type out_mesh_path: Path
    :param mesh: mesh
    :type mesh: pyvista.UnstructuredGrid
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
    Returns the cell boundary conditions of the mesh. It works by iterating all cell data and looking for
    data arrays that include the strings "P_SOUF" or "P_IOFLOW".
    +++WARNING+++: This function still in a experimental state since it is not clear how exactly this function will
    be used in the future.
    TODO: Allow a generic definition of the normal vector for the filter condition.

    :param bulk_mesh_path: name of the mesh
    :type bulk_mesh_path: Path
    :param mesh: mesh
    :type mesh: pyvista.UnstructuredGrid
    :return: path with name of mesh, topsurface mesh with cell boundary conditions
    :rtype: tuple
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
    # correct unit for P_IOFLOW, in FEFLOW m/d in ogs m/s
    topsurf.cell_data["P_IOFLOW"] = topsurf.cell_data["P_IOFLOW"]
    return (
        bulk_mesh_path.with_stem("topsurface_" + bulk_mesh_path.stem),
        topsurf,
    )


def get_material_properties(mesh: pv.UnstructuredGrid, property: str):
    """
    Get the material properties of the mesh converted from FEFLOW. There are several methods available
    to access the material properties. Either they are accessible with the FEFLOW API(ifm) or with brute-force methods,
    which check each element, like this function.

    :param mesh: mesh
    :type mesh: pyvista.UnstructuredGrid
    :param property: property
    :type property: str
    :return: material_properties
    :rtype: dict
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
            material_properties[material_id] = [property_of_material[0]]
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
    :return: material_properties
    :rtype: collections.defaultdict
    """
    # Use a default dict because it allows to extend the values in the list.
    # Also it initializes the value if there is an empty list.
    material_properties: defaultdict[str, list[float]] = defaultdict(list)

    for property in properties_list:
        assert property in ["P_CONDX", "P_CONDY", "P_CONDZ"]
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
    :return: filename
    :rtype: str
    """
    mask = mesh.cell_data["MaterialIDs"] == material_id
    material_mesh = mesh.extract_cells(mask)
    for prop in property_list:
        assert prop in ["P_CONDX", "P_CONDY", "P_CONDZ"]
    zipped = list(zip(*[material_mesh[prop] for prop in property_list]))
    material_mesh[new_property] = zipped
    # correct the unit
    material_mesh[new_property] = material_mesh[new_property]
    filename = str(saving_path.with_name(str(material_id) + ".vtu"))
    material_mesh.point_data.remove("vtkOriginalPointIds")
    for pt_data in material_mesh.point_data:
        if pt_data != "bulk_node_ids":
            material_mesh.point_data.remove(pt_data)
    for cell_data in material_mesh.cell_data:
        if cell_data not in ["bulk_element_ids", new_property]:
            material_mesh.cell_data.remove(cell_data)
    material_mesh.save(filename)
    return filename


def materials_in_steady_state_diffusion(
    material_properties: dict,
    model,
):
    """
    Create the section for material properties for steady state diffusion processes in the prj-file.

    :param bulk_mesh_path: path of bulk mesh
    :type bulk_mesh_path: Path
    :param mesh: mesh
    :type mesh: pyvista.UnstructuredGrid
    :param material_properties: material properties
    :type material_properties: dict
    :param model: model to setup prj-file
    :type model: ogs6py.OGS
    :return: model
    :rtype: ogs6py.OGS
    """
    for material_id, property_value in material_properties.items():
        if any(prop == "inhomogeneous" for prop in property_value):
            model.media.add_property(
                medium_id=material_id,
                name="diffusion",
                type="Parameter",
                parameter_name="diffusion_" + str(material_id),
            )
            model.mesh.add_mesh(filename=str(material_id) + ".vtu")
            model.parameters.add_parameter(
                name="diffusion_" + str(material_id),
                type="MeshElement",
                field_name="KF",
                mesh=str(material_id),
            )
        else:
            model.media.add_property(
                medium_id=material_id,
                name="diffusion",
                type="Constant",
                value=" ".join(str(element) for element in property_value),
            )
        model.media.add_property(
            medium_id=material_id,
            name="reference_temperature",
            type="Constant",
            value=293.15,
        )
    return model


def materials_in_liquid_flow(
    material_properties: dict,
    model,
):
    """
    Create the section for material properties in liquid flow processes in the prj-file.

    :param bulk_mesh_path: path of bulk mesh
    :type bulk_mesh_path: Path
    :param mesh: mesh
    :type mesh: pyvista.UnstructuredGrid
    :param material_properties: material properties
    :type material_properties: dict
    :param model: model to setup prj-file
    :type model: ogs6py.OGS
    :return: model
    :rtype: ogs6py.OGS
    """
    for material_id, property_value in material_properties.items():
        if any(prop == "inhomogeneous" for prop in property_value):
            model.media.add_property(
                medium_id=material_id,
                name="permeability",
                type="Parameter",
                parameter_name="permeability_" + str(material_id),
            )
            model.mesh.add_mesh(filename=str(material_id) + ".vtu")
            model.parameters.add_parameter(
                name="permeability_" + str(material_id),
                type="MeshElement",
                field_name="KF",
                mesh=str(material_id),
            )
        else:
            model.media.add_property(
                medium_id=material_id,
                name="permeability",
                type="Constant",
                value=" ".join(str(element) for element in property_value),
            )
        model.media.add_property(
            medium_id=material_id,
            name="reference_temperature",
            type="Constant",
            value=293.15,
        )
        model.media.add_property(
            medium_id=material_id,
            phase_type="AqueousLiquid",
            name="viscosity",
            type="Constant",
            value=1,
        )
        model.media.add_property(
            medium_id=material_id,
            phase_type="AqueousLiquid",
            name="density",
            type="Constant",
            value=1,
        )
        model.media.add_property(
            medium_id=material_id,
            name="storage",
            type="Constant",
            value=0,
        )
        model.media.add_property(
            medium_id=material_id,
            name="porosity",
            type="Constant",
            value=1,
        )
    return model


def setup_prj_file(
    bulk_mesh_path: Path,
    mesh: pv.UnstructuredGrid,
    material_properties: dict,
    process: str,
    model=None,
):
    """
    Sets up a prj-file for ogs simulations using ogs6py.

    :param bulk_mesh_path: path of bulk mesh
    :type bulk_mesh_path: Path
    :param mesh: mesh
    :type mesh: pyvista.UnstructuredGrid
    :param material_properties: material properties
    :type material_properties: dict
    :param process: the process to be prepared
    :type process: str
    :param model: model to setup prj-file
    :type model: ogs6py.OGS
    :return: model
    :rtype: ogs6py.OGS
    """

    prjfile = bulk_mesh_path.with_suffix(".prj")
    if model is None:
        model = ogs.OGS(PROJECT_FILE=prjfile)

    BC_type_dict = {
        "_BC_": "Dirichlet",
        "2ND": "Neumann",
        "3RD": "Robin",
        "4TH": "NodalSourceTerm",
        "P_IOFLOW": "Neumann",
        "P_SOUF": "Volumetric",
    }

    model.mesh.add_mesh(filename=bulk_mesh_path.name)
    model.mesh.add_mesh(filename="topsurface_" + bulk_mesh_path.name)
    model.processes.add_process_variable(
        process_variable="process_variable", process_variable_name="HEAD_OGS"
    )
    model.processvars.set_ic(
        process_variable_name="HEAD_OGS",
        components=1,
        order=1,
        initial_condition="p0",
    )
    model.parameters.add_parameter(name="p0", type="Constant", value=0)
    for point_data in mesh.point_data:
        if point_data[0:4] == "P_BC":
            # Every point boundary condition refers to a separate mesh
            model.mesh.add_mesh(filename=point_data + ".vtu")
            if "3RD" in point_data:
                model.parameters.add_parameter(
                    name="u_0",
                    type="MeshNode",
                    field_name=point_data,
                    mesh=point_data,
                )
                model.parameters.add_parameter(
                    name="alpha",
                    type="Constant",
                    value=np.unique(mesh.cell_data["P_TRAF_IN"])[1],
                )
                model.processvars.add_bc(
                    process_variable_name="HEAD_OGS",
                    type="Robin",
                    alpha="alpha",
                    u_0="u_0",
                    mesh=point_data,
                )
            elif "4TH" in point_data:
                model.parameters.add_parameter(
                    name=point_data,
                    type="MeshNode",
                    field_name=point_data,
                    mesh=point_data,
                )
                model.processvars.add_st(
                    process_variable_name="HEAD_OGS",
                    type="Nodal",
                    mesh=point_data,
                    parameter=point_data,
                )
            else:
                # Add boundary conditions
                model.processvars.add_bc(
                    process_variable_name="HEAD_OGS",
                    type=next(
                        val
                        for key, val in BC_type_dict.items()
                        if key in point_data
                    ),
                    parameter=point_data,
                    mesh=point_data,
                )
                # Every point boundary condition refers to a parameter with the same name
                model.parameters.add_parameter(
                    name=point_data,
                    type="MeshNode",
                    field_name=point_data,
                    mesh=point_data,
                )

    for cell_data in mesh.cell_data:
        if cell_data in ["P_IOFLOW", "P_SOUF"]:
            if cell_data in ["P_IOFLOW"]:
                # Add boundary conditions
                model.processvars.add_bc(
                    process_variable_name="HEAD_OGS",
                    type=next(
                        val
                        for key, val in BC_type_dict.items()
                        if key in cell_data
                    ),
                    parameter=cell_data,
                    mesh="topsurface_" + bulk_mesh_path.stem,
                )
            elif cell_data in ["P_SOUF"]:
                model.processvars.add_st(
                    process_variable_name="HEAD_OGS",
                    type=next(
                        val
                        for key, val in BC_type_dict.items()
                        if key in cell_data
                    ),
                    parameter=cell_data,
                    mesh="topsurface_" + bulk_mesh_path.stem,
                )
            # Every point boundary condition refers to a parameter with the same name
            model.parameters.add_parameter(
                name=cell_data,
                type="MeshElement",
                field_name=cell_data,
                mesh="topsurface_" + bulk_mesh_path.stem,
            )

    # include material properties in the prj-file
    if process == "steady state diffusion":
        materials_in_steady_state_diffusion(material_properties, model)
    elif process == "liquid flow":
        materials_in_liquid_flow(material_properties, model)
    else:
        msg = "Only 'steady state diffusion' and 'liquid flow' processes are supported."
        raise ValueError(msg)

    # add deactivated subdomains if existing
    if 0 in mesh.cell_data["P_INACTIVE_ELE"]:
        tags = ["material_ids"]
        material_ids = mesh.cell_data["MaterialIDs"]
        deactivated_materials = set(material_ids[material_ids < 0])
        values = [
            " ".join(str(material) for material in deactivated_materials),
        ]
        xpath = "./process_variables/process_variable"
        model.add_element(parent_xpath=xpath, tag="deactivated_subdomains")
        model.add_block(
            blocktag="deactivated_subdomain",
            parent_xpath=xpath + "/deactivated_subdomains",
            taglist=tags,
            textlist=values,
        )
        model.add_block(
            blocktag="time_interval",
            parent_xpath=xpath
            + "/deactivated_subdomains/deactivated_subdomain",
            taglist=["start", "end"],
            textlist=["0", "1"],
        )

    return model


def deactivate_cells(mesh: pv.UnstructuredGrid):
    """
    Multiplies the MaterialID of all cells that are inactive in FEFLOW by -1.
    Therefore, the input mesh is modified.
    :param mesh: mesh
    :type mesh: pyvista.UnstructuredGrid
    :return: 0 for no cells have been deactivated and 1 for cells have been deactivated
    :rytpe: int
    """
    inactive_cells = np.where(mesh.cell_data["P_INACTIVE_ELE"] == 0)
    if len(inactive_cells[0]) == 0:
        return_int = 0
    else:
        mesh.cell_data["MaterialIDs"][inactive_cells] *= -1
        return_int = 1
    return return_int
