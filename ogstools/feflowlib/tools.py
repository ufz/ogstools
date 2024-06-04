# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

import argparse
import logging as log
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional, TypedDict

import numpy as np
import pyvista as pv
from ogs6py import ogs
from typing_extensions import NotRequired, Unpack

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


def get_material_properties_of_HT_model(
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
        feflow_species_parameter, ogs_species_parameter
    ):
        for material_id, property_value in get_material_properties(
            mesh, parameter_feflow
        ).items():
            material_properties[material_id][parameter_ogs] = property_value[0]

    return material_properties


def get_species(mesh: pv.UnstructuredGrid) -> list:
    r"""
    Get the names of chemical species of a mesh. Only works, if species-specific
    porosity values are assigned and named '\*_P_PORO'.

    :param mesh: mesh
    :return: list of species
    """
    species = [
        cell_data.replace("_P_PORO", "")
        for cell_data in mesh.cell_data
        if "P_PORO" in cell_data
    ]
    if not species:
        ValueError(
            """No species are found. This could be due to the fact that no porosity
            values for species are assigned."""
        )
    return species


def _add_species_to_prj_file(
    xpath: str, parameter_dict: dict, species_list: list, model: ogs.OGS
) -> None:
    """
    Adds the entries needed in the prj-file for components/species. Since in ogs6py no
    corresponding feature exists to use the common ogs6py.media-class, the generic method
    add_block is used.

    WARNING: After add_block was used, the ogs6py-model cannot be altered with
    common ogs6py functions!

    :param xpath: Path to the species/components sektion in the prj-file.
    :param parameter_dict: Dictionary with all the parameter names and values.
    :param species_list: List of all species.
    :param model: Model to setup prj-file, there the species will be added to.
    """
    species_parameter = ["decay_rate", "retardation_factor", "diffusion"]
    model.add_element(parent_xpath=xpath, tag="components")
    for species in species_list:
        model.add_block(
            blocktag="component",
            parent_xpath=xpath + "/components",
            taglist=["name", "properties"],
            textlist=[species, ""],
        )
        for parameter, parameter_val in parameter_dict.items():
            if (
                any(spec_par in parameter for spec_par in species_parameter)
                and species in parameter
            ):
                model.add_block(
                    blocktag="property",
                    parent_xpath=xpath
                    + "/components/component[name='"
                    + species
                    + "']/properties",
                    taglist=["name", "type", "value"],
                    textlist=[
                        parameter.replace(species + "_", ""),
                        "Constant",
                        str(parameter_val),
                    ],
                )


def _add_global_process_coupling_CT(
    model: ogs.OGS, species: list, max_iter: int = 1, rel_tol: float = 1e-10
) -> None:
    """
    Add the section of the prj-file that defines the global process coupling
    in the time loop.

    :param model: Model to setup prj-file, there the section of global process coupling will be added.
    :param species: List of all species.
    :param max_iter: Maximal iteration.
    :param rel_tol: Relative tolerance.
    """
    model.add_block(
        blocktag="global_process_coupling",
        parent_xpath="./time_loop",
        taglist=["max_iter", "convergence_criteria"],
        textlist=[str(max_iter), ""],
    )
    for _i in range(len(species) + 1):
        model.add_block(
            blocktag="convergence_criterion",
            parent_xpath="./time_loop/global_process_coupling/convergence_criteria",
            taglist=["type", "norm_type", "reltol"],
            textlist=["DeltaX", "NORM2", str(rel_tol)],
        )


def _add_process(
    model: ogs.OGS,
    species: list,
    time_stepping: Optional[list] = None,
    initial_time: int = 1,
    end_time: int = 1,
) -> None:
    """
    Add the section of the prj-file that defines the process in the time loop.

    :param model: Model to setup prj-file, there the section of global process coupling will be added.
    :param species: List of all species.
    :param repeat_list: List of how often a time step should be repeated.
    :param delta_t_list: List of how long a time stept should be.
    :param initial_time: Beginning of the simulation time.
    :param end_time: End of the simulation time.
    """
    for _i in range(len(species) + 1):
        model.add_block(
            blocktag="process",
            block_attrib={"ref": "CT"},
            parent_xpath="./time_loop/processes",
            taglist=["nonlinear_solver"],
            textlist=["basic_picard"],
        )
    model.add_block(
        blocktag="convergence_criterion",
        parent_xpath="./time_loop/processes/process",
        taglist=["type", "norm_type", "reltol"],
        textlist=["DeltaX", "NORM2", "1e-6"],
    )
    model.add_block(
        blocktag="time_discretization",
        parent_xpath="./time_loop/processes/process",
        taglist=["type"],
        textlist=["BackwardEuler"],
    )
    model.add_block(
        blocktag="time_stepping",
        parent_xpath="./time_loop/processes/process",
        taglist=["type", "t_initial", "t_end", "timesteps"],
        textlist=["FixedTimeStepping", str(initial_time), str(end_time), ""],
    )
    if time_stepping is None:
        time_stepping = [(1, 1)]
    for time_step in time_stepping:
        model.add_block(
            blocktag="pair",
            parent_xpath="./time_loop/processes/process/time_stepping/timesteps",
            taglist=["repeat", "delta_t"],
            textlist=[str(time_step[0]), str(time_step[1])],
        )


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


def materials_in_CT(
    material_properties: dict,
    species_list: list,
    model: ogs.OGS,
) -> ogs.OGS:
    """
    Create the section for material properties for CT (component transport)
    processes in the prj-file.

    :param material_properties: material properties
    :param model: model to setup prj-file
    :return: model
    """
    for material_id in material_properties:
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
        # Get a list of properties (porosity,transversal/longitunal_dispersivity)
        # across species. If there are more than one value, independent OGS-models
        # need to be set up manually.
        porosity_val = np.unique(
            [
                material_properties[material_id][prop]
                for prop in material_properties[material_id]
                if "porosity" in prop
            ]
        )

        long_disp_val = np.unique(
            [
                material_properties[material_id][prop]
                for prop in material_properties[material_id]
                if "longitudinal_dispersivity" in prop
            ]
        )

        trans_disp_val = np.unique(
            [
                material_properties[material_id][prop]
                for prop in material_properties[material_id]
                if "transversal_dispersivity" in prop
            ]
        )
        model.media.add_property(
            medium_id=material_id,
            name="porosity",
            type="Constant",
            value=(
                str(porosity_val) if len(porosity_val) > 1 else porosity_val[0]
            ),
        )

        model.media.add_property(
            medium_id=material_id,
            name="longitudinal_dispersivity",
            type="Constant",
            value=(
                str(long_disp_val)
                if len(long_disp_val) > 1
                else long_disp_val[0]
            ),
        )

        model.media.add_property(
            medium_id=material_id,
            name="transversal_dispersivity",
            type="Constant",
            value=(
                str(trans_disp_val)
                if len(trans_disp_val) > 1
                else trans_disp_val[0]
            ),
        )
        if "permeability_X" in material_properties[material_id]:
            model.media.add_property(
                medium_id=material_id,
                name="permeability",
                type="Constant",
                value=str(material_properties[material_id]["permeability_X"])
                + " "
                + str(material_properties[material_id]["permeability_Y"])
                + " "
                + str(material_properties[material_id]["permeability_Z"]),
            )
        elif "permeability" in material_properties[material_id]:
            model.media.add_property(
                medium_id=material_id,
                name="permeability",
                type="Constant",
                value=str(material_properties[material_id]["permeability"]),
            )

    for material_id in material_properties:
        xpath = "./media/medium[@id='" + str(material_id) + "']/phases/phase"
        _add_species_to_prj_file(
            xpath, material_properties[material_id], species_list, model
        )

    return model


class RequestParams(TypedDict):
    model: NotRequired[ogs.OGS]
    species_list: NotRequired[list]
    initial_time: NotRequired[int]
    end_time: NotRequired[int]
    time_stepping: NotRequired[list]
    max_iter: NotRequired[int]
    rel_tol: NotRequired[float]


def setup_prj_file(
    bulk_mesh_path: Path,
    mesh: pv.UnstructuredGrid,
    material_properties: dict,
    process: str,
    **kwargs: Unpack[RequestParams],
) -> ogs.OGS:
    """
    Sets up a prj-file for ogs simulations using ogs6py.

    :param bulk_mesh_path: path of bulk mesh
    :param mesh: mesh
    :param material_properties: material properties
    :param process: the process to be prepared
    :Keyword Arguments (kwargs):
       * *model* (``ogs.OGS``) --
         A ogs6py (ogs) model that is extended, should be used for templates
       * *species_list* (``list``) --
         All chemical species that occur in a model, if the model is to simulate a
         Component Transport (HC/CT) process.
       * *initial_time* (``int``) --
         Initial time for CT process.
       * *end_time* (``int``) --
         End time for CT process.
       * *time_stepping* (``list[tuple]``) --
         List of tuples with time steps. First entry is the repetition of the time step
         and the second the length of the time step.
       * *max_iter* (``int``) --
         Maximal iterations of process coupling in a CT process.
       * *relative_tolerance* (``float``) --
         Relative tolerance of process coupling in a CT process.
    :return: model


    """
    # ToDo: Make sure that no non-valid arguments are chosen!
    model = kwargs["model"] if "model" in kwargs else None
    species_list = kwargs["species_list"] if "species_list" in kwargs else None
    initial_time = kwargs["initial_time"] if "initial_time" in kwargs else 1
    end_time = kwargs["end_time"] if "end_time" in kwargs else 1
    time_stepping = (
        kwargs["time_stepping"] if "time_stepping" in kwargs else None
    )
    max_iter = kwargs["max_iter"] if "max_iter" in kwargs else 1
    rel_tol = kwargs["rel_tol"] if "rel_tol" in kwargs else 1e-10
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
    elif "component" in process:
        model.processes.add_process_variable(
            process_variable="pressure", process_variable_name="HEAD_OGS"
        )
        model.parameters.add_parameter(name="C0", type="Constant", value=0)
        if species_list is not None:
            for species in species_list:
                if len(species_list) > 1:
                    process_variable = "concentration_" + species
                else:
                    process_variable = "concentration"

                model.processes.add_process_variable(
                    process_variable=process_variable,
                    process_variable_name=species,
                )
                model.processvars.set_ic(
                    process_variable_name=species,
                    components=1,
                    order=1,
                    initial_condition="C0",
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
        if "P_BC" in point_data:
            if "FLOW" in point_data:
                process_var = "HEAD_OGS"
            elif "HEAT" in point_data:
                process_var = "temperature"
            elif "MASS" in point_data:
                # Each species becomes a separate process variable in OGS.
                process_var = point_data.split("_P_", 1)[0]
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
            and np.any(mesh.cell_data[cell_data])
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
    elif process == "component transport":
        assert species_list is not None
        materials_in_CT(material_properties, species_list, model)
        _add_global_process_coupling_CT(model, species_list, max_iter, rel_tol)
        _add_process(
            model,
            species_list,
            time_stepping=time_stepping,
            initial_time=initial_time,
            end_time=end_time,
        )
    else:
        msg = "Only 'steady state diffusion', 'liquid flow', 'hydro thermal' and 'component transport' processes are supported."
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
