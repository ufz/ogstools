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
        output_path: Path | None = None,
    ):
        """
        Initialize an OGS model object.

            :param mesh: Mesh of the OGS model.
            :param subdomains: All subdomains (boundary conditions/ source terms) required for the model.
            :param project_file: Project file used to define the model.
            :param name: Name of the OGS model.
            :param output_path: Output path, refers to the path to the directory with the filename of the bulk mesh.
        """
        if output_path is None:
            self.output_path = Path.cwd() / "OGS_model.vtu"
        else:
            self.output_path = output_path.with_suffix(".vtu")
        self.mesh: ot.Mesh | pv.UnstructuredGrid = mesh
        self.subdomains: dict[str, ot.Mesh | pv.UnstructuredGrid] = subdomains
        self.project = project
        self._mesh_saving_needed = True

    """
    # ToDo move get_dimension out of feflowlib
    # Anyway, not 100% sure if this is needed...
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

    def save(
        self, output_path: None | Path = None, force_saving: bool = False
    ) -> None:
        """
        Save the converted FEFLOW model. Saves the meshes only if they have not been saved previously.
        or 'force_saving' is true.

        :param output_path: The path where the mesh, subdomains and project file will be written.
        :param force_saving: Save, even the model already was saved.
        """
        if output_path is None:
            output_path = self.output_path
        else:
            self.project.replace_mesh(
                self.output_path.name, output_path.with_suffix(".vtu").name
            )
            self.project.replace_text(
                str(output_path.with_suffix("")),
                xpath="./time_loop/output/prefix",
            )

            # Overwrite output_path (not 100% if this is the best practice.)
            # cannot happen before changing the mesh in project-file
            self.output_path = output_path
        self.project.write_input(prjfile_path=output_path.with_suffix(".prj"))
        if self._mesh_saving_needed or force_saving:
            self.mesh.save(output_path.with_suffix(".vtu"))
            for name, subdomain in self.subdomains.items():
                subdomain.save(output_path.parent / (name + ".vtu"))
            self._mesh_saving_needed = False
        """
        else:
            logger.info(
                "The mesh and subdomains have already been saved. As no changes have been detected, saving of the mesh is skipped. The project file is saved (again)."
            )
        """

    def run(
        self, output_path: None | Path = None, overwrite: bool = False
    ) -> None:
        """
        Run the converted FEFLOW model.

        :param output_path: The path where the mesh, subdomains and project file will be written.
        :param force_saving: Save, even the model already was saved.
        """
        self.save(output_path, overwrite)
        self.project.run_model()

    if find_spec("ifm") is not None:

        @classmethod
        def from_feflow_model(
            cls, feflow_model: "ot.FeflowModel"
        ) -> "OGSModel":
            ogs_model_instance = cls(
                feflow_model.mesh, feflow_model.subdomains, feflow_model.project
            )
            # Copy path from FEFLOW model
            ogs_model_instance.output_path = feflow_model.output_path
            return ogs_model_instance

        @classmethod
        def read_feflow(cls, feflow_file: Path) -> "OGSModel":
            feflow_model = ot.FeflowModel(feflow_file)
            return cls.from_feflow_model(feflow_model)
