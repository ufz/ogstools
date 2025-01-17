# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

import argparse
import logging as log
from collections import defaultdict
from collections.abc import Callable
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
    :returns: specific_cells
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
    return mesh.get_cell(0).dimension


def extract_point_boundary_conditions(
    mesh: pv.UnstructuredGrid,
) -> dict[str, pv.UnstructuredGrid]:
    """
    Returns the point boundary conditions of the mesh. It works by iterating all point data and looking for
    data arrays that include the string "_BC". Depending on what follows, it defines the boundary condition type.

    :param mesh: mesh
    :returns: dict_of_point_boundary_conditions
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
            dict_of_point_boundary_conditions[point_data] = filtered_points
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
    point_boundary_conditions_dict = extract_point_boundary_conditions(mesh)
    for name, boundary_condition in point_boundary_conditions_dict.items():
        boundary_condition.save(out_mesh_path / (name + ".vtu"))


def extract_cell_boundary_conditions(
    mesh: pv.UnstructuredGrid,
) -> tuple[str, pv.UnstructuredGrid]:
    """
    Returns the cell boundary conditions of the mesh. It works by iterating all cell data and looking for
    data arrays that include the strings "P_SOUF" or "P_IOFLOW".

    :param mesh: mesh
    :returns: path with name of mesh, topsurface mesh with cell boundary conditions
    """
    assign_bulk_ids(mesh)
    if mesh.volume != 0:
        # get the topsurface since there are the cells of interest
        topsurf = get_specific_surface(
            mesh.extract_surface(), lambda normals: normals[:, 2] > 0
        )
    else:
        topsurf = mesh.copy()
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
    # Remove bulk node/element ids from bulk mesh, as they are not needed anymore.
    remove_bulk_ids(mesh)
    return (
        "topsurface",
        topsurf,
    )


def get_material_properties(mesh: pv.UnstructuredGrid, property: str) -> dict:
    """
    Get the material properties of the mesh converted from FEFLOW. There are several methods available
    to access the material properties. Either they are accessible with the FEFLOW API(ifm) or with brute-force methods,
    which check each element, like this function.

    :param mesh: mesh
    :param property: property
    :returns: material_properties
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
            material_properties[int(material_id)] = [property_of_material[0]]
        else:
            material_properties[material_id] = ["inhomogeneous_" + property]
            logger.warning(
                "The property %s in material %s is inhomogeneously distributed.",
                property,
                material_id,
            )

    return material_properties


def get_material_properties_of_H_model(
    mesh: pv.UnstructuredGrid,
) -> defaultdict:
    """
    Get a dictionary of all necessaray parameter values for a flow model for each material in the mesh.

    :param mesh: mesh
    :return: material_properties
    """
    possible_parameter_mapping = {
        "P_COND": "permeability",
        "P_CONDX": "permeability_X",
        "P_CONDY": "permeability_Y",
        "P_CONDZ": "permeability_Z",
        "P_COMP": "storage",
    }

    parameter_mapping = {}
    for parameter in possible_parameter_mapping:
        if parameter in mesh.cell_data:
            parameter_mapping[parameter] = possible_parameter_mapping[parameter]

    material_properties: defaultdict = defaultdict(dict)
    for parameter_feflow, parameter_ogs in parameter_mapping.items():
        for material_id, property_value in get_material_properties(
            mesh, parameter_feflow
        ).items():
            material_properties[material_id][parameter_ogs] = property_value[0]

    return material_properties


def get_material_properties_of_HT_model(
    mesh: pv.UnstructuredGrid,
) -> defaultdict:
    """
    Get a dictionary of all necessaray parameter values for a HT problem for each material in the mesh.

    :param mesh: mesh
    :returns: material_properties
    """
    possible_parameter_mapping = {
        "P_ANGL": "anisotropy_angle",
        "P_ANIS": "anisotropy_factor",
        "P_CAPACF": "specific_heat_capacity_fluid",
        "P_CAPACS": "specific_heat_capacity_solid",
        "P_COMP": "storage",
        "P_COND": "permeability",
        "P_CONDX": "permeability_X",
        "P_CONDY": "permeability_Y",
        "P_CONDZ": "permeability_Z",
        "P_CONDUCF": "thermal_conductivity_fluid",
        "P_CONDUCS": "thermal_conductivity_solid",
        "P_POROH": "porosity",
        "P_LDISH": "thermal_longitudinal_dispersivity",
        "P_TDISH": "thermal_transversal_dispersivity",
    }

    parameter_mapping = {}
    for parameter in possible_parameter_mapping:
        if parameter in mesh.cell_data:
            parameter_mapping[parameter] = possible_parameter_mapping[parameter]

    material_properties: defaultdict = defaultdict(dict)
    for parameter_feflow, parameter_ogs in parameter_mapping.items():
        for material_id, property_value in get_material_properties(
            mesh, parameter_feflow
        ).items():
            material_properties[material_id][parameter_ogs] = property_value[0]

    return material_properties


def get_material_properties_of_CT_model(
    mesh: pv.UnstructuredGrid,
) -> defaultdict:
    """
    Gets the material properties/parameter for each chemical species/component of the model.

    :param mesh: mesh
    """
    parameters_mapping = {
        "P_DECA": "decay_rate",
        "P_DIFF": "pore_diffusion",
        "P_LDIS": "longitudinal_dispersivity",
        "P_PORO": "porosity",
        "P_SORP": "sorption_coeff",
        "P_TDIS": "transversal_dispersivity",
        "P_COMP": "storage",
        "retardation_factor": "retardation_factor",
    }
    possible_permeability = {
        "P_COND": "permeability",
        "P_CONDX": "permeability_X",
        "P_CONDY": "permeability_Y",
        "P_CONDZ": "permeability_Z",
    }
    for perme in possible_permeability:
        if perme in mesh.cell_data:
            parameters_mapping[perme] = possible_permeability[perme]

    feflow_species_parameter = [
        cell_data
        for cell_data in mesh.cell_data
        if any(parameter in cell_data for parameter in parameters_mapping)
    ]
    ogs_species_parameter = [
        feflow_species_para.replace(
            feflow_parameter, parameters_mapping[feflow_parameter]
        )
        for feflow_species_para in feflow_species_parameter
        for feflow_parameter in parameters_mapping
        if feflow_parameter in feflow_species_para
    ]

    material_properties: defaultdict = defaultdict(dict)
    for parameter_feflow, parameter_ogs in zip(
        feflow_species_parameter, ogs_species_parameter, strict=False
    ):
        for material_id, property_value in get_material_properties(
            mesh, parameter_feflow
        ).items():
            material_properties[int(material_id)][
                parameter_ogs
            ] = property_value[0]

    return material_properties


def get_species(mesh: pv.UnstructuredGrid) -> list:
    r"""
    Get the names of chemical species of a mesh. Only works, if species-specific
    porosity values are assigned and named '\*_P_PORO'.

    :param mesh: mesh
    :returns: list of species
    """
    species = [
        cell_data.replace("_P_DECA", "")
        for cell_data in mesh.cell_data
        if "P_DECA" in cell_data
    ]
    if not species:
        ValueError(
            """No species are found. This could be due to the fact that no porosity
            values for species are assigned."""
        )
    return species
