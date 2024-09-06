# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

from collections import defaultdict
from pathlib import Path

import ifm_contrib as ifm
import numpy as np
import pyvista as pv

from ogstools.ogs6py import Project

from . import _tools
from ._feflowlib import convert_properties_mesh
from ._templates import (
    component_transport,
    hydro_thermal,
    liquid_flow,
    steady_state_diffusion,
)


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

    @property
    def boundary_conditions(self) -> dict:
        """
        The boundary meshes for a ogs model.

        :return: Dictionary (dict) of boundary meshes, with name as key and mesh as value.
        """
        # ToDo: Introduce this behaviour to feflowlib.tools with a type.
        # And return type of name for cell and pt BC should be the same not possix Path...
        boundary_conditions = _tools.extract_point_boundary_conditions(
            self.mesh_path.parent, self.mesh
        )

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
            ) = _tools.extract_cell_boundary_conditions(
                self.mesh_path, self.mesh
            )
            boundary_conditions[cell_bc_path] = cell_bc_mesh
        return boundary_conditions

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
            3: "Combined flow, mass and heat transport [not supported yet]",
            4: "Combined flow and age transport [not supported yet]",
            5: "Combined flow, mass, and age transport [not supported yet]",
            6: "Combined flow, heat, and age transport [not supported yet]",
            7: "Combined flow, mass, age, and heat transport [not supported yet]",
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

        return material_properties

    # @property
    def prj(
        self, end_time: int = 1, time_stepping: list | None = None
    ) -> Project:
        """
        A proposition for a prj-file to run a OGS simulation.
        It may be not complete and manual adjustments for time
        loop or solver configurations must be made.

        :return: The ogs6py model created from the FEFLOW model.
        """
        if "Liquid flow" in self.process:
            template_model = liquid_flow(
                Path(self.mesh_path.with_suffix("")),
                Project(output_file=self.mesh_path.with_suffix(".prj")),
                dimension=self.dimension,
                end_time=end_time,
                time_stepping=time_stepping,
            )
        elif "Steady state diffusion" in self.process:
            template_model = steady_state_diffusion(
                Path(self.mesh_path.with_suffix("")),
                Project(output_file=self.mesh_path.with_suffix(".prj")),
            )
        elif "Hydro thermal" in self.process:
            template_model = hydro_thermal(
                Path(self.mesh_path.with_suffix("")),
                Project(output_file=self.mesh_path.with_suffix(".prj")),
                dimension=self.dimension,
            )
        elif "Component transport" in self.process:
            species = _tools.get_species(self.mesh)
            template_model = component_transport(
                Path(self.mesh_path.with_suffix("")),
                species,
                Project(output_file=self.mesh_path.with_suffix(".prj")),
                dimension=self.dimension,
                end_time=end_time,
                time_stepping=time_stepping,
            )

        return _tools.setup_prj_file(
            self.mesh_path,
            self.mesh,
            self.material_properties,
            self.process,
            species_list=(
                species if "Component transport" in self.process else None
            ),
            model=template_model,
        )

    def prepare_OGS(
        self, end_time: int = 1, time_stepping: list | None = None
    ) -> None:
        self.mesh.save(self.mesh_path)
        for path, boundary_mesh in self.boundary_conditions.items():
            boundary_mesh.save(path)
        prj = self.prj(end_time, time_stepping)
        prj.write_input()
