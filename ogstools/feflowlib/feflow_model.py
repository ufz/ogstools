# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

import logging as log
from collections import defaultdict
from pathlib import Path

import ifm_contrib as ifm
import numpy as np
import pyvista as pv

from ogstools.ogs6py import Project

from . import _tools
from ._feflowlib import convert_properties_mesh
from ._prj_tools import setup_prj_file
from ._templates import (
    component_transport,
    create_prj_template,
    hydro_thermal,
    liquid_flow,
    steady_state_diffusion,
)

logger = log.getLogger(__name__)


class FeflowModel:
    """
    The FEFLOW model class to access the FEFLOW model.

    :no-index:
    """

    def __init__(
        self,
        feflow_file: Path | str,
        out_path: Path | None = None,
    ):
        """
        Initialize a FEFLOW model object

            :param feflow_file:     Path to the feflow file.
            :param out_path:        Path for the output, if not defined, the same path as input file is taken.
        """
        feflow_file = Path(feflow_file)
        if out_path is None:
            self.mesh_path = Path(feflow_file.with_suffix(".vtu"))
        else:
            self.mesh_path = Path(out_path).with_suffix(".vtu")
        ifm.forceLicense("Viewer")
        self._doc = ifm.loadDocument(str(feflow_file))
        self.mesh = convert_properties_mesh(self._doc)
        self.dimension = self._doc.getNumberOfDimensions()
        self.setup_prj()  # _project object with default values is initialized here
        self._init_subdomains()
        self._mesh_saving_needed = True

    @property
    def ogs_bulk_mesh(self) -> pv.UnstructuredGrid:
        """
        The ogs bulk mesh is a copy of the geometry of the
        FEFLOW model mesh and its MaterialIDs.

        :return: The bulk mesh with MaterialIDs.
        """
        bulk_mesh = pv.UnstructuredGrid()
        bulk_mesh.copy_structure(self.mesh)
        bulk_mesh["MaterialIDs"] = self.mesh["MaterialIDs"]
        return bulk_mesh

    def _init_subdomains(self) -> None:
        """
        The boundary meshes for a ogs model.

        :return: Dictionary (dict) of boundary meshes, with name as key and mesh as value.
        """
        _subdomains = _tools.extract_point_boundary_conditions(self.mesh)

        if self.dimension == 3 and (
            (
                "P_SOUF" in self.mesh.cell_data
                and not np.all(self.mesh.cell_data["P_SOUF"] == 0)
            )
            or (
                "P_IOFLOW" in self.mesh.cell_data
                and not np.all(self.mesh.cell_data["P_IOFLOW"] == 0)
            )
        ):
            (
                cell_bc_path,
                cell_bc_mesh,
            ) = _tools.extract_cell_boundary_conditions(self.mesh)
            _subdomains[cell_bc_path] = cell_bc_mesh

        self.subdomains: dict[str, pv.UnstructuredGrid] = _subdomains

    @property
    def process(self) -> str:
        """
        The process that is defined in the FEFLOW file.
        feflowlib cannot create prj-files for all of them.

        :return: Process name.
        """
        problem_classes = {
            -1: "Illegal problem class",
            0: "Liquid flow",
            1: "Component transport",
            2: "Hydro thermal",
            3: "Combined flow, mass and heat transport [not supported by this feflow converter]",
            4: "Combined flow and age transport [not supported by this feflow converter]",
            5: "Combined flow, mass, and age transport [not supported by this feflow converter]",
            6: "Combined flow, heat, and age transport [not supported by this feflow converter]",
            7: "Combined flow, mass, age, and heat transport [not supported by this feflow converter]",
        }

        return problem_classes[self._doc.getProblemClass()]

    @property
    def material_properties(self) -> defaultdict:
        """
        Material properties of the mesh.

        :return: Dictionary with properties and the corresponding value for each material.
        """
        process = self.process
        if "Steady state diffusion" in process or "Liquid flow" in process:
            material_properties = _tools.get_material_properties_of_H_model(
                self.mesh
            )

        elif "Hydro thermal" in process:
            material_properties = _tools.get_material_properties_of_HT_model(
                self.mesh
            )

        elif "Component transport" in process:
            material_properties = _tools.get_material_properties_of_CT_model(
                self.mesh
            )
        else:
            # For later functions of the converter, material properties are needed.
            # For this reason, a defaultdict is returned with no material properties in
            # this case.
            # ToDo: return a dict of all possible material properties with a warning!
            material_properties = defaultdict(str)
            material_properties["undefined"] = (
                f"Material properties are only saved on the mesh for this process: '{process}'",
            )
            logger.warning(
                (
                    "Material properties are not supported (at the moment) in a model using such a process:",
                    process,
                )
            )

        return material_properties

    # feflow specific
    def setup_prj(
        self,
        end_time: int = 1,
        time_stepping: list | None = None,
        error_tolerance: float = 1e-10,
        steady: bool = False,
    ) -> None:
        """
        A proposition for a prj-file to run a OGS simulation.
        It may be not complete and manual adjustments for time
        loop or solver configurations must be made.

        :return: The ogs6py model created from the FEFLOW model.
        """
        if "Liquid flow" in self.process and not steady:
            template_model = liquid_flow(
                Path(self.mesh_path.with_suffix("")),
                Project(output_file=self.mesh_path.with_suffix(".prj")),
                dimension=self.dimension,
                end_time=end_time,
                time_stepping=time_stepping,
                error_tolerance=error_tolerance,
            )
        elif "Liquid flow" in self.process and steady:
            template_model = steady_state_diffusion(
                Path(self.mesh_path.with_suffix("")),
                Project(output_file=self.mesh_path.with_suffix(".prj")),
                error_tolerance=error_tolerance,
            )
        elif "Hydro thermal" in self.process:
            template_model = hydro_thermal(
                Path(self.mesh_path.with_suffix("")),
                Project(output_file=self.mesh_path.with_suffix(".prj")),
                dimension=self.dimension,
                end_time=end_time,
                time_stepping=time_stepping,
                error_tolerance=error_tolerance,
            )
        elif "Component transport" in self.process or "mass" in self.process:
            process_name = (
                "CT" if "Component transport" in self.process else "undefined"
            )
            species = _tools.get_species(self.mesh)
            template_model = component_transport(
                Path(self.mesh_path.with_suffix("")),
                species,
                Project(output_file=self.mesh_path.with_suffix(".prj")),
                process_name=process_name,
                dimension=self.dimension,
                end_time=end_time,
                time_stepping=time_stepping,
                error_tolerance=error_tolerance,
            )
        else:
            template_model = create_prj_template(
                Path(self.mesh_path.with_suffix("")),
                Project(output_file=self.mesh_path.with_suffix(".prj")),
                dimension=self.dimension,
                end_time=end_time,
                time_stepping=time_stepping,
                error_tolerance=error_tolerance,
            )
        # replaces default
        self.project = setup_prj_file(
            self.mesh_path,
            self.mesh,
            self.material_properties,
            self.process if not steady else "Steady state diffusion",
            species_list=(
                species
                if "Component" in self.process
                else (species if "mass" in self.process else None)
            ),
            model=template_model,
        )

    def save(
        self, output_path: None | Path = None, force_saving: bool = False
    ) -> None:
        """
        Save the converted FEFLOW model. Saves the meshes only if they have not been saved previously.
        or 'force_saving' is true.

        :param output_path: The path where the mesh, subdomains and project file will be written.
        """
        if output_path is None:
            output_path = self.mesh_path
        self.project.write_input(prjfile_path=output_path.with_suffix(".prj"))
        if self._mesh_saving_needed or force_saving:
            self.mesh.save(output_path.with_suffix(".vtu"))
            for name, subdomain in self.subdomains.items():
                subdomain.save(output_path.parent / (name + ".vtu"))
            self._mesh_saving_needed = False
        else:
            logger.info(
                "The mesh and subdomains have already been saved. As no changes have been detected, saving of the mesh is skipped. The project file is saved (again)."
            )

    def run(
        self, output_path: None | Path = None, overwrite: bool = False
    ) -> None:
        """
        Run the converted FEFLOW model.

        :param output_path: The path where the mesh, subdomains and project file will be written.
        """
        self.save(output_path, overwrite)
        self.project.run_model()
