# Copyright (c) 2012-2024, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#


from collections import defaultdict
from importlib.util import find_spec
from pathlib import Path

import pyvista as pv

import ogstools as ot
from ogstools.ogs6py import Project

# to be discussed:
# import logging as log
# logger = log.getLogger(__name__)


class OGSModel:
    """
    The OGS model class to configure an OGS model.

    :no-index:
    """

    def __init__(
        self,
        mesh: pv.UnstructuredGrid | ot.Mesh,
        subdomains: dict[str, ot.Mesh | pv.UnstructuredGrid],
        project: Project,
        name: str = "OGS_default_model",
        output_path: Path | None = None,
    ):
        """
        Initialize an OGS model object.

            :param mesh: Mesh of the OGS model.
            :param subdomains: All subdomains (boundary conditions/ source terms) required for the model.
            :param project_file: Project file used to define the model.
            :param name: Name of the OGS model.
            :param output_path: Output path, if the model is to be saved.
        """
        if output_path is None:
            self.output_path = Path.cwd()
        self._mesh = mesh
        self._subdomains = subdomains
        self._project = project
        self.name = name
        self._mesh_saving_needed = True

    @property
    def project(self) -> Project:
        return self._project

    @property
    def mesh(self) -> ot.Mesh | pv.UnstructuredGrid:
        return self._mesh

    @property
    def subdomains(self) -> dict[str, ot.Mesh | pv.UnstructuredGrid]:
        return self._subdomains

    """
    # ToDo move get_dimension out of feflowlib
    from ogstools.feflowlib._tools import get_dimension
    @property
    def dimension(self) -> int:
        return get_dimension(self._mesh)
    """

    @property
    def process(self) -> str | None:
        """
        The process that is defined in the project.

        :return: Process name.
        """
        return None

    @property
    def material_properties(self) -> defaultdict | None:
        """
        Material properties of the OGS Model.

        :return: Dictionary with properties and the corresponding value for each material.
        """

        return None

    def save(self, output_path: None | Path = None) -> None:
        """
        Save the OGS model.

        :param output_path: The path where the mesh, boundary meshes and project file will be written.
        """
        if output_path is None:
            output_path = self.output_path
        pv.save_meshio(
            output_path.with_name(self.name).with_suffix(".vtu"), self.mesh
        )
        for name, subdomain in self.subdomains.items():
            pv.save_meshio(
                output_path.with_name(name).with_suffix(".vtu"),
                subdomain,
            )
        self.project.write_input(
            output_path.with_name(self.name).with_suffix(".prj")
        )
        """
        else:
            logger.info(
                "The mesh and boundary meshes have already been saved. As no changes have been detected, saving of the mesh is skipped. The project file is saved (again)."
            )
        """

    def run(self, output_path: None | Path = None) -> None:
        """
        Run the OGS model.

        :param output_path: The path where the mesh, boundary meshes and project file will be written.
        """
        self.save(output_path)
        self.project.run_model()

    if find_spec("ifm") is not None:

        def from_feflow(self, feflow_model: "ot.FeflowModel") -> None:
            self._mesh = feflow_model.mesh
            self._subdomains = feflow_model.boundary_conditions
            self._project = feflow_model.project
