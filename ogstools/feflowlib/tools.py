# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

import argparse
import logging as log
from collections import defaultdict
from pathlib import Path
from typing import Callable

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


def get_specific_surface(
    surface_mesh: pv.PolyData,
    filter_condition: Callable[[pv.pyvista_ndarray], pv.pyvista_ndarray],
) -> pv.UnstructuredGrid:
    """
    Return only cells that match the filter condition for the normals of the
    input-surface mesh. A standard use case could be to extract the cells that
    have a normal in a particular direction, e.g. upward in the z-direction. The
    filter condition would then be: `lambda normals: normals[:, 2] > 0`.

    :param surface_mesh: The surface mesh.
    :param filter_condition: A condition to set up the filter for the normals.
    :return: specific_cells
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


def assign_bulk_ids(mesh: pv.UnstructuredGrid) -> None:
    """
    Add data arrays for bulk_node_ids and bulk_element_ids to the given bulk mesh.

    :param mesh: bulk mesh
    """
    # The format must be unsigned integer, as it is required by OGS
    mesh["bulk_node_ids"] = np.arange(mesh.n_points, dtype=np.uint64)
    mesh.cell_data["bulk_element_ids"] = np.arange(
        mesh.n_cells, dtype=np.uint64
    )


def remove_bulk_ids(mesh: pv.UnstructuredGrid) -> None:
    """
    Remove data arrays for bulk_node_ids and bulk_element_ids of the given bulk mesh.

    :param mesh: bulk mesh
    """
    mesh.point_data.remove("bulk_node_ids")
    mesh.cell_data.remove("bulk_element_ids")


def get_dimension(mesh: pv.UnstructuredGrid) -> int:
    """
    Return the dimension of the mesh.

    :param mesh: mesh
    """
    if len(np.unique(mesh.celltypes)) == 1:
        return mesh.get_cell(0).dimension

    msg = (
        "The dimension can only be returned, if all cells are of the same type."
    )
    raise ValueError(msg)


def extract_point_boundary_conditions(
    out_mesh_path: Path, mesh: pv.UnstructuredGrid
) -> dict:
    """
    Returns the point boundary conditions of the mesh. It works by iterating all point data and looking for
    data arrays that include the string "_BC". Depending on what follows, it defines the boundary condition type.

    :param out_mesh_path: path of the output mesh
    :param mesh: mesh
    :return: dict_of_point_boundary_conditions
    """
    dict_of_point_boundary_conditions = {}
    # Assigning bulk node ids because format needs to be unsigned integer for OGS.
    # Otherwise vtkOriginalPointIds would be fine.
    assign_bulk_ids(mesh)
    # extract mesh since boundary condition are on the surface ?! (not safe!)
    boundary_mesh = (
        mesh.extract_surface()
        if get_dimension(mesh) == 3
        else mesh.extract_feature_edges()
    )
    # remove all the point data that are not of interest
    for point_data in boundary_mesh.point_data:
        if "_BC" not in point_data and point_data != "bulk_node_ids":
            boundary_mesh.point_data.remove(point_data)
    # remove all points with point data that are of "nan"-value
    for point_data in boundary_mesh.point_data:
        if point_data != "bulk_node_ids":
            BC_2nd_or_3rd = "2ND" in point_data or "3RD" in point_data
            include_cells_bool = BC_2nd_or_3rd and get_dimension(mesh) == 3
            filter_mesh = mesh if "_4TH" in point_data else boundary_mesh
            filtered_points = filter_mesh.extract_points(
                [not np.isnan(x) for x in filter_mesh[point_data]],
                adjacent_cells=False,
                include_cells=include_cells_bool,
            )
            # If the mesh is 2D and BC are of 2nd or 3rd order, line elements
            # will be included in the boundary mesh.
            if BC_2nd_or_3rd and get_dimension(mesh) == 2:
                filtered_points_new = pv.lines_from_points(
                    filtered_points.points
                ).cast_to_unstructured_grid()
                filtered_points_new[point_data] = filtered_points[point_data]
                filtered_points_new["bulk_node_ids"] = filtered_points[
                    "bulk_node_ids"
                ]
                filtered_points = filtered_points_new.copy()
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
    # Remove bulk node/element ids from bulk mesh, as they are not needed anymore.
    remove_bulk_ids(mesh)
    return dict_of_point_boundary_conditions


def write_point_boundary_conditions(
    out_mesh_path: Path, mesh: pv.UnstructuredGrid
) -> None:
    """
    Writes the point boundary conditions that are returned from 'extract_point_boundary_conditions()'

    :param out_mesh_path: path for writing
    :param mesh: mesh
    """
    point_boundary_conditions_dict = extract_point_boundary_conditions(
        out_mesh_path, mesh
    )
    for path, boundary_condition in point_boundary_conditions_dict.items():
        boundary_condition.save(path)


def extract_cell_boundary_conditions(
    bulk_mesh_path: Path, mesh: pv.UnstructuredGrid
) -> tuple[Path, pv.UnstructuredGrid]:
    """
    Returns the cell boundary conditions of the mesh. It works by iterating all cell data and looking for
    data arrays that include the strings "P_SOUF" or "P_IOFLOW".
    +++WARNING+++: This function still in a experimental state since it is not clear how exactly this function will
    be used in the future.
    TODO: Allow a generic definition of the normal vector for the filter condition.

    :param bulk_mesh_path: name of the mesh
    :param mesh: mesh
    :return: path with name of mesh, topsurface mesh with cell boundary conditions
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
    # Remove bulk node/element ids from bulk mesh, as they are not needed anymore.
    remove_bulk_ids(mesh)
    return (
        bulk_mesh_path.with_stem("topsurface_" + bulk_mesh_path.stem),
        topsurf,
    )


def get_material_properties(mesh: pv.UnstructuredGrid, property: str) -> dict:
    """
    Get the material properties of the mesh converted from FEFLOW. There are several methods available
    to access the material properties. Either they are accessible with the FEFLOW API(ifm) or with brute-force methods,
    which check each element, like this function.

    :param mesh: mesh
    :param property: property
    :return: material_properties
    """
    material_ids = mesh.cell_data["MaterialIDs"]
    material_properties = {}
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


def get_materials_of_HT_model(
    mesh: pv.UnstructuredGrid,
) -> defaultdict:
    """
    Get a dictionary of all necessaray parameter values for a HT problem for each material in the mesh.

    :param mesh: mesh
    :return: material_properties
    """
    parameters_feflow = [
        "P_ANGL",
        "P_ANIS",
        "P_CAPACF",
        "P_CAPACS",
        "P_COMP",
        "P_COND",
        "P_CONDUCF",
        "P_CONDUCS",
        "P_POROH",
        "P_LDISH",
        "P_TDISH",
        "P_ANIS",
    ]
    parameters_ogs = [
        "anisotropy_angle",
        "anisotropy_factor",
        "specific_heat_capacity_fluid",
        "specific_heat_capacity_solid",
        "storage",
        "permeability",
        "thermal_conductivity_fluid",
        "thermal_conductivity_solid",
        "porosity",
        "thermal_longitudinal_dispersivity",
        "thermal_transversal_dispersivity",
    ]
    material_properties: defaultdict = defaultdict(dict)
    for parameter_feflow, parameter_ogs in zip(
        parameters_feflow, parameters_ogs
    ):
        for material_id, property_value in get_material_properties(
            mesh, parameter_feflow
        ).items():
            material_properties[material_id][parameter_ogs] = property_value[0]
    return material_properties


def combine_material_properties(
    mesh: pv.UnstructuredGrid, properties_list: list
) -> defaultdict:
    """
    Combine multiple material properties. The combined properties are returned
    as list of values in a dictionary.

    :param mesh: mesh
    :param properties_list: list of properties to be combined
    :return: material_properties
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
) -> str:
    """
    Writes a separate mesh-file with a specific material that has inhomogeneous property values
    within the material group. It can also be used to write multiple properties
    into a "new property" data array. For example, write a data array for a tensor defined by
    data arrays representing values of different spatial directions. Nevertheless it can still be
    be used to write the inhomogeneous values of a single property into a separate mesh-file.

    :param mesh: mesh
    :param property_list: list of properties
    :param new_property: name of the combined properties
    :param material: material with inhomogeneous properties
    :param saving_path: path to save the mesh
    :return: filename
    """
    mask = mesh.cell_data["MaterialIDs"] == material_id
    material_mesh = mesh.extract_cells(mask)
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
    model: ogs.OGS,
) -> ogs.OGS:
    """
    Create the section for material properties for steady state diffusion processes in the prj-file.

    :param bulk_mesh_path: path of bulk mesh
    :param mesh: mesh
    :param material_properties: material properties
    :param model: model to setup prj-file
    :return: model
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
    model: ogs.OGS,
) -> ogs.OGS:
    """
    Create the section for material properties in liquid flow processes in the prj-file.

    :param bulk_mesh_path: path of bulk mesh
    :param mesh: mesh
    :param material_properties: material properties
    :param model: model to setup prj-file
    :return: model
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


def materials_in_HT(
    material_properties: dict,
    model: ogs.OGS,
) -> ogs.OGS:
    """
    Create the section for material properties for HT processes in the prj-file.

    :param bulk_mesh_path: path of bulk mesh
    :param mesh: mesh
    :param material_properties: material properties
    :param model: model to setup prj-file
    :return: model
    """
    for material_id in material_properties:
        model.media.add_property(
            medium_id=material_id,
            phase_type="AqueousLiquid",
            name="specific_heat_capacity",
            type="Constant",
            value=material_properties[material_id][
                "specific_heat_capacity_fluid"
            ],
        )
        model.media.add_property(
            medium_id=material_id,
            phase_type="AqueousLiquid",
            name="thermal_conductivity",
            type="Constant",
            value=material_properties[material_id][
                "thermal_conductivity_fluid"
            ],
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
            phase_type="Solid",
            name="storage",
            type="Constant",
            value=material_properties[material_id]["storage"],
        )
        model.media.add_property(
            medium_id=material_id,
            phase_type="Solid",
            name="density",
            type="Constant",
            value=1,
        )
        model.media.add_property(
            medium_id=material_id,
            phase_type="Solid",
            name="specific_heat_capacity",
            type="Constant",
            value=material_properties[material_id][
                "specific_heat_capacity_solid"
            ],
        )
        model.media.add_property(
            medium_id=material_id,
            phase_type="Solid",
            name="thermal_conductivity",
            type="Constant",
            value=material_properties[material_id][
                "thermal_conductivity_solid"
            ],
        )
        model.media.add_property(
            medium_id=material_id,
            name="permeability",
            type="Constant",
            # Theoretically an anisotropy angle can be applied, but it is not implemented
            # in this case.
            value=str(material_properties[material_id]["permeability"])
            + " "
            + str(
                material_properties[material_id]["permeability"]
                * material_properties[material_id]["anisotropy_factor"]
            ),
        )
        model.media.add_property(
            medium_id=material_id,
            name="porosity",
            type="Constant",
            value=material_properties[material_id]["porosity"],
        )
        model.media.add_property(
            medium_id=material_id,
            name="thermal_conductivity",
            type="EffectiveThermalConductivityPorosityMixing",
        )
        model.media.add_property(
            medium_id=material_id,
            name="thermal_transversal_dispersivity",
            type="Constant",
            value=material_properties[material_id][
                "thermal_transversal_dispersivity"
            ],
        )
        model.media.add_property(
            medium_id=material_id,
            name="thermal_longitudinal_dispersivity",
            type="Constant",
            value=material_properties[material_id][
                "thermal_longitudinal_dispersivity"
            ],
        )

    return model


def setup_prj_file(
    bulk_mesh_path: Path,
    mesh: pv.UnstructuredGrid,
    material_properties: dict,
    process: str,
    model: ogs.OGS = None,
) -> ogs.OGS:
    """
    Sets up a prj-file for ogs simulations using ogs6py.

    :param bulk_mesh_path: path of bulk mesh
    :param mesh: mesh
    :param material_properties: material properties
    :param process: the process to be prepared
    :param model: model to setup prj-file
    :return: model
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
    # this if condition checks if the mesh is 3D. If so the topsurface will be considered.
    if get_dimension(mesh) == 3:
        model.mesh.add_mesh(filename="topsurface_" + bulk_mesh_path.name)
    if "thermal" in process:
        model.processes.add_process_variable(
            process_variable="temperature",
            process_variable_name="temperature",
        )
        model.processvars.set_ic(
            process_variable_name="temperature",
            components=1,
            order=1,
            initial_condition="T0",
        )
        # Initial condition should be read from the FEFLOW file!
        # But for this the initial condition should still be on the FEFLOW file.
        # FEFLOW overwrites the initial condition, if the model was simulated.
        model.parameters.add_parameter(name="T0", type="Constant", value=273.15)
        model.processes.add_process_variable(
            process_variable="pressure", process_variable_name="HEAD_OGS"
        )
    else:
        model.processes.add_process_variable(
            process_variable="process_variable",
            process_variable_name="HEAD_OGS",
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
            if "HEAT" in point_data:
                process_var = "temperature"
            elif "FLOW" in point_data:
                process_var = "HEAD_OGS"
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
                    process_variable_name=process_var,
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
                    process_variable_name=process_var,
                    type="Nodal",
                    mesh=point_data,
                    parameter=point_data,
                )
            else:
                # Add boundary conditions
                model.processvars.add_bc(
                    process_variable_name=process_var,
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
        """
        At the moment, P_IOFLOW and P_SOUF have only been tested with 3D
        meshes. That is why, we only write them to the prj-file
        if the mesh is 3D and their value is non-zero. If it is 0,
        they do not matter, since they will not have an effect.
        """
        if (
            cell_data in ["P_IOFLOW", "P_SOUF"]
            and np.unique(mesh.cell_data[cell_data]) != 0
            and get_dimension(mesh) == 3
        ):
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
    elif process == "hydro thermal":
        materials_in_HT(material_properties, model)
    else:
        msg = "Only 'steady state diffusion', 'liquid flow' and 'hydro thermal' processes are supported."
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


def deactivate_cells(mesh: pv.UnstructuredGrid) -> int:
    """
    Multiplies the MaterialID of all cells that are inactive in FEFLOW by -1.
    Therefore, the input mesh is modified.
    :param mesh: mesh
    :return: 0 for no cells have been deactivated and 1 for cells have been deactivated
    """
    inactive_cells = np.where(mesh.cell_data["P_INACTIVE_ELE"] == 0)
    if len(inactive_cells[0]) == 0:
        return_int = 0
    else:
        mesh.cell_data["MaterialIDs"][inactive_cells] *= -1
        return_int = 1
    return return_int
