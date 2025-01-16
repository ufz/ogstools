# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

import logging as log
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pyvista as pv
from typing_extensions import NotRequired, Unpack

from ogstools.ogs6py import Project

from ._tools import get_dimension

# log configuration
logger = log.getLogger(__name__)


def _add_species_to_prj_file(
    xpath: str, parameter_dict: dict, species_list: list, model: Project
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
    model: Project, species: list, max_iter: int = 1, rel_tol: float = 1e-10
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
    for _i in range(1 + len(species)):  # pressure + n species
        model.add_block(
            blocktag="convergence_criterion",
            parent_xpath="./time_loop/global_process_coupling/convergence_criteria",
            taglist=["type", "norm_type", "reltol"],
            textlist=["DeltaX", "NORM2", str(rel_tol)],
        )


def _get_permeability(material_properties: dict, material_id: int) -> str:
    if (
        material_properties[material_id]["permeability_X"]
        == material_properties[material_id]["permeability_Y"]
        == material_properties[material_id]["permeability_Z"]
    ):
        permeability_val = str(
            material_properties[material_id]["permeability_X"]
        )
    else:
        permeability_val = (
            str(material_properties[material_id]["permeability_X"])
            + " "
            + str(material_properties[material_id]["permeability_Y"])
            + " "
            + str(material_properties[material_id]["permeability_Z"])
        )

    return permeability_val


def _add_heterogeneous_material_property(
    model: Project,
    material_id: int,
    property: str,
    feflow_property: str,
    phase_type: str | None = None,
) -> Project:
    kwargs = {
        "medium_id": material_id,
        "name": property,
        "type": "Parameter",
        "parameter_name": feflow_property,
    }
    if phase_type is not None:
        kwargs["phase_type"] = phase_type
    model.media.add_property(**kwargs)
    model_tree: Any = model.tree  # Actually this should be ET.ElementET
    parameters = [
        parameter.find("name").text
        for parameter in model_tree.getroot().findall("./parameters/parameter")
    ]
    # Parameter need to occur only once!
    if feflow_property not in parameters:
        model.parameters.add_parameter(
            name=feflow_property, type="MeshElement", field_name=feflow_property
        )

    return model


def _add_permeabilty_prj(
    material_properties: dict,
    model: Project,
    material_id: int,
    steady: bool = False,
) -> Project:
    """
    Add the section for diffusion (steady state diffusion) or permeability to the media section in the prj-file

    :param material_properties: material properties
    :param model: model to setup prj-file
    :param material_id: id of the material
    :param steady: choose steady state diffusion
    :returns: model
    """
    diffusion_or_permeability = "diffusion" if steady else "permeability"
    if any(
        isinstance(mat_value, str) and "permeability" in mat_property
        for mat_property, mat_value in material_properties[material_id].items()
    ):
        if "permeability_X" in material_properties[material_id]:
            # _add_KF_tensor_from_permeability_X_Y_Z()
            _add_heterogeneous_material_property(
                model, material_id, diffusion_or_permeability, "KF"
            )
        else:
            _add_heterogeneous_material_property(
                model, material_id, diffusion_or_permeability, "P_COND"
            )
    elif "permeability_X" in material_properties[material_id]:
        model.media.add_property(
            medium_id=material_id,
            name=diffusion_or_permeability,
            type="Constant",
            value=_get_permeability(material_properties, material_id),
        )
    # rewrite this to check for anisotropy angle
    elif "anisotropy_factor" in material_properties[material_id]:
        model.media.add_property(
            medium_id=material_id,
            name=diffusion_or_permeability,
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
        logger.warning(
            "Permeability anisotropy is implemented only for 2D HT cases."
            "The anisotropy angle is assumed to reference the y-axis."
        )
    elif "permeability" in material_properties[material_id]:
        model.media.add_property(
            medium_id=material_id,
            name=diffusion_or_permeability,
            type="Constant",
            value=str(material_properties[material_id]["permeability"]),
        )

    else:
        logger.error("No permeability was detected.")

    return model


def _handle_heterogeneous_material_properties(
    material_value: float | str,
    material_property: str,
    model: Project,
    material_id: int,
) -> str:
    if (
        isinstance(material_value, str)
        and "permeability" not in material_property
    ):
        hetero_property = material_property
        if "fluid" in material_property:
            phase_type = "AqueousLiquid"
            material_property = material_property.replace("_fluid", "")
        elif "solid" in material_property or "storage" in material_property:
            phase_type = "Solid"
            material_property = material_property.replace("_solid", "")
        else:
            phase_type = None
        _add_heterogeneous_material_property(
            model,
            material_id,
            material_property,
            material_value.replace("inhomogeneous_", ""),
            phase_type,
        )
        return hetero_property
    return ""


def materials_in_steady_state_diffusion(
    material_properties: dict,
    model: Project,
) -> Project:
    """
    Create the section for material properties for steady state diffusion processes in the prj-file.

    :param bulk_mesh_path: path of bulk mesh
    :param mesh: mesh
    :param material_properties: material properties
    :param model: model to setup prj-file
    :returns: model
    """
    for material_id in material_properties:
        _add_permeabilty_prj(
            material_properties, model, material_id, steady=True
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
    model: Project,
) -> Project:
    """
    Create the section for material properties in liquid flow processes in the prj-file.

    :param bulk_mesh_path: path of bulk mesh
    :param mesh: mesh
    :param material_properties: material properties
    :param model: model to setup prj-file
    :returns: model
    """
    for material_id in material_properties:
        # Here it is assumed inhomogeneous material properties refer to the permeability not the storage.
        _add_permeabilty_prj(material_properties, model, material_id)

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
        if isinstance(material_properties[material_id]["storage"], str):
            _add_heterogeneous_material_property(
                model, material_id, "storage", "P_COMP"
            )
        else:
            model.media.add_property(
                medium_id=material_id,
                name="storage",
                type="Constant",
                value=material_properties[material_id]["storage"],
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
    model: Project,
) -> Project:
    """
    Create the section for material properties for HT processes in the prj-file.

    :param material_properties: material properties
    :param model: model to setup prj-file
    :returns: model
    """
    for material_id in material_properties:
        _add_permeabilty_prj(material_properties, model, material_id)
        hetero_properties = []
        for mat_property, mat_value in material_properties[material_id].items():
            hetero_properties.append(
                _handle_heterogeneous_material_properties(
                    mat_value, mat_property, model, material_id
                )
            )

        if "specific_heat_capacity_fluid" not in hetero_properties:
            model.media.add_property(
                medium_id=material_id,
                phase_type="AqueousLiquid",
                name="specific_heat_capacity",
                type="Constant",
                value=material_properties[material_id][
                    "specific_heat_capacity_fluid"
                ],
            )

        if "thermal_conductivity_fluid" not in hetero_properties:
            model.media.add_property(
                medium_id=material_id,
                phase_type="AqueousLiquid",
                name="thermal_conductivity",
                type="Constant",
                value=material_properties[material_id][
                    "thermal_conductivity_fluid"
                ],
            )
        if "storage" not in hetero_properties:
            model.media.add_property(
                medium_id=material_id,
                phase_type="Solid",
                name="storage",
                type="Constant",
                value=material_properties[material_id]["storage"],
            )
        if "specific_heat_capacity_solid" not in hetero_properties:
            model.media.add_property(
                medium_id=material_id,
                phase_type="Solid",
                name="specific_heat_capacity",
                type="Constant",
                value=material_properties[material_id][
                    "specific_heat_capacity_solid"
                ],
            )
        if "thermal_conductivity_solid" not in hetero_properties:
            model.media.add_property(
                medium_id=material_id,
                phase_type="Solid",
                name="thermal_conductivity",
                type="Constant",
                value=material_properties[material_id][
                    "thermal_conductivity_solid"
                ],
            )
        if "porosity" not in hetero_properties:
            model.media.add_property(
                medium_id=material_id,
                name="porosity",
                type="Constant",
                value=material_properties[material_id]["porosity"],
            )
        if "thermal_transversal_dispersivity" not in hetero_properties:
            model.media.add_property(
                medium_id=material_id,
                name="thermal_transversal_dispersivity",
                type="Constant",
                value=material_properties[material_id][
                    "thermal_transversal_dispersivity"
                ],
            )
        if "thermal_longitudinal_dispersivity" not in hetero_properties:
            model.media.add_property(
                medium_id=material_id,
                name="thermal_longitudinal_dispersivity",
                type="Constant",
                value=material_properties[material_id][
                    "thermal_longitudinal_dispersivity"
                ],
            )
        model.media.add_property(
            medium_id=material_id,
            name="thermal_conductivity",
            type="EffectiveThermalConductivityPorosityMixing",
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
    return model


def materials_in_CT(
    material_properties: dict,
    species_list: list,
    model: Project,
) -> Project:
    """
    Create the section for material properties for CT (component transport)
    processes in the prj-file.

    :param material_properties: material properties
    :param model: model to setup prj-file
    :returns: model
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
        if any(
            len(mat_prop_list) > 1
            for mat_prop_list in [porosity_val, long_disp_val, trans_disp_val]
        ):
            logger.warning(
                "There are species with different porosity,transversal/longitunal_dispersivity values. Therefore, independent OGS-models need to be set up manually"
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
        _add_permeabilty_prj(material_properties, model, material_id)

    for material_id in material_properties:
        xpath = "./media/medium[@id='" + str(material_id) + "']/phases/phase"
        _add_species_to_prj_file(
            xpath, material_properties[material_id], species_list, model
        )

    return model


class RequestParams(TypedDict):
    model: NotRequired[Project]
    species_list: NotRequired[list | None]
    max_iter: NotRequired[int]
    rel_tol: NotRequired[float]


def setup_prj_file(
    bulk_mesh_path: Path,
    mesh: pv.UnstructuredGrid,
    material_properties: dict,
    process: str,
    **kwargs: Unpack[RequestParams],
) -> Project:
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
       * *max_iter* (``int``) --
         Maximal iterations of process coupling in a CT process.
       * *relative_tolerance* (``float``) --
         Relative tolerance of process coupling in a CT process.
    :returns: model


    """
    # ToDo: Make sure that no non-valid arguments are chosen!
    model = kwargs.get("model", None)
    species_list = kwargs.get("species_list", None)
    max_iter = kwargs.get("max_iter", 1)
    rel_tol = kwargs.get("rel_tol", 1e-10)
    prjfile = bulk_mesh_path.with_suffix(".prj")
    if model is None:
        model = Project(output_file=prjfile)

    BC_type_dict = {
        "_BC_": "Dirichlet",
        "2ND": "Neumann",
        "3RD": "Robin",
        "4TH": "NodalSourceTerm",
    }

    model.mesh.add_mesh(filename=bulk_mesh_path.name)
    # this if condition checks if the mesh is 3D. If so the topsurface will be considered.
    if get_dimension(mesh) == 3 and (
        (
            "P_SOUF" in mesh.cell_data
            and not np.all(mesh.cell_data["P_IOFLOW"] == 0)
        )
        or (
            "P_IOFLOW" in mesh.cell_data
            and not np.all(mesh.cell_data["P_IOFLOW"] == 0)
        )
    ):
        model.mesh.add_mesh(filename="topsurface.vtu")

    if "thermal" in process or "heat" in process:
        model.processes.add_process_variable(
            process_variable="temperature",
            process_variable_name="temperature",
        )
        model.process_variables.set_ic(
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
    if "Component" in process or "mass" in process:
        if "heat" not in process:
            model.processes.add_process_variable(
                process_variable="pressure", process_variable_name="HEAD_OGS"
            )
        model.parameters.add_parameter(name="C0", type="Constant", value=0)
        if species_list is not None:
            for species in species_list:
                model.processes.add_process_variable(
                    process_variable="concentration",
                    process_variable_name=species,
                )
                model.process_variables.set_ic(
                    process_variable_name=species,
                    components=1,
                    order=1,
                    initial_condition="C0",
                )
    if "Liquid flow" in process or "Steady state diffusion" in process:
        model.processes.add_process_variable(
            process_variable="process_variable",
            process_variable_name="HEAD_OGS",
        )

    model.process_variables.set_ic(
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
                model.process_variables.add_bc(
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
                model.process_variables.add_st(
                    process_variable_name=process_var,
                    type="Nodal",
                    mesh=point_data,
                    parameter=point_data,
                )
            else:
                # Add boundary conditions
                model.process_variables.add_bc(
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
        if cell_data in ["P_IOFLOW", "P_SOUF"] and np.any(
            mesh.cell_data[cell_data]
        ):
            if get_dimension(mesh) == 3:
                IOFLOW_SOUF_mesh_name = "topsurface"
            else:
                IOFLOW_SOUF_mesh_name = bulk_mesh_path.stem
            if cell_data in ["P_IOFLOW"]:
                # Add boundary conditions
                model.process_variables.add_bc(
                    process_variable_name="HEAD_OGS",
                    type="Neumann",
                    parameter=cell_data,
                    mesh=IOFLOW_SOUF_mesh_name,
                )
            elif cell_data in ["P_SOUF"]:
                model.process_variables.add_st(
                    process_variable_name="HEAD_OGS",
                    type="Volumetric",
                    parameter=cell_data,
                    mesh=IOFLOW_SOUF_mesh_name,
                )
            # Every point boundary condition refers to a parameter with the same name
            model.parameters.add_parameter(
                name=cell_data,
                type="MeshElement",
                field_name=cell_data,
                mesh="topsurface",
            )

    # include material properties in the prj-file
    if process == "Steady state diffusion":
        materials_in_steady_state_diffusion(material_properties, model)
    elif process == "Liquid flow":
        materials_in_liquid_flow(material_properties, model)
    elif process == "Hydro thermal":
        materials_in_HT(material_properties, model)
    elif process == "Component transport":
        assert species_list is not None
        materials_in_CT(material_properties, species_list, model)
        _add_global_process_coupling_CT(model, species_list, max_iter, rel_tol)

    # add deactivated subdomains if existing
    if 0 in mesh.cell_data["P_INACTIVE_ELE"]:
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
            taglist=["material_ids"],
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
