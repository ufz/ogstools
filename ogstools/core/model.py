# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#


import copy
import typing
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
from matplotlib import pyplot as plt

from ogstools.meshes._meshes import Meshes
from ogstools.ogs6py.project import Project

from .execution import Execution
from .simulation_controller import SimulationController
from .storage import StorageBase

if typing.TYPE_CHECKING:
    from .simulation import Simulation


class Model(StorageBase):
    """
    A complete OGS model combining project file, meshes, and execution settings.

    The Model class integrates all components needed to run an OGS simulation:
    - A project file (prj) defining the simulation setup
    - Meshes required by the simulation
    - Execution parameters (parallelization, logging, etc.)

    Models can be created from individual components, loaded from disk, or
    initialized from existing folder structures following OGSTools conventions.
    """

    def __init__(
        self,
        project: Project | Path | str,
        meshes: Meshes | Path | str | None = None,
        execution: Execution | Path | str | None = None,
        id: str | None = None,
    ) -> None:
        """
        Initialize a Model object.

        :param project:     Project object or path to a .prj file. If a path is
                            given, the Project will be loaded from that file.
        :param meshes:      Meshes object or path to a folder containing mesh files.
                            If None, attempts to locate meshes in standard locations
                            relative to the project file (same folder or 'meshes' subfolder).
        :param execution:   Execution object or path to an execution.yaml file.
                            If None, creates a default Execution configuration.
        :param id:          Optional unique identifier for this model.

        :raises ValueError:     If meshes cannot be found or located automatically.
        :raises FileNotFoundError: If specified paths do not exist.
        """
        super().__init__("Model", id=id)

        if isinstance(project, Project):
            self.project = project
        else:
            project_path = Path(project)
            if project_path.is_dir():
                self.project = Project.from_folder(project_path)
            else:
                self.project = Project(input_file=project_path)
                self.project.prjfile = project_path

        if isinstance(meshes, Meshes):
            self.meshes = meshes
        elif isinstance(meshes, Path | str):
            meshes = Path(meshes)
            meta_file = meshes / "meta.yaml"
            if meta_file.exists():
                self.meshes = Meshes.from_folder(meshes)
            else:
                meshes_files = self.project.meshpaths(meshes)
                self.meshes = Meshes.from_files(meshes_files)

        else:  # None
            assert self.project.input_file
            # Last resort - Possible conventions to try
            mesh_locations = [
                self.project.input_file.parent,
                self.project.input_file.parent / "meshes",
            ]

            # Test if all meshes in a given folder exist
            def all_meshes_exist(base: Path) -> bool:
                return bool(
                    np.all([p.exists() for p in self.project.meshpaths(base)])
                )

            for base in mesh_locations:
                if all_meshes_exist(base):
                    meta_file = base / "meta.yaml"
                    if meta_file.exists():
                        self.meshes = Meshes.from_folder(base)
                    else:
                        meshes_files = self.project.meshpaths(base)
                        self.meshes = Meshes.from_files(meshes_files)
                    break
            else:
                loc_str = ", ".join(str(b) for b in mesh_locations)
                msg = f"Not all meshes found. Tried: {loc_str}. Put the meshes in these locations or define Meshes when initializing Model."
                raise ValueError(msg)

        if isinstance(execution, Execution):
            self.execution = execution
        elif isinstance(execution, Path | str):
            self.execution = Execution.from_file(execution)
        else:  # None
            self.execution = Execution()
            # Already initialized as not saved and user_specified_target=False

    @classmethod
    def from_folder(cls, folder: Path | str) -> "Model":
        """
        Initialize a Model from a folder following OGSTools conventions.

        Expects the folder structure created by Model.save():
        - project/: Subfolder containing project.prj and associated files
        - meshes/: Subfolder containing mesh files
        - execution.yaml: Execution configuration

        :param folder:  Path to the folder containing the model files.
        :returns:       A :class:`ogstools.Model` object initialized from the folder contents.

        :raises FileNotFoundError: If required files are not found in the folder.
        """
        folder = Path(folder)
        if not folder.exists():
            msg = f"The folder {folder!r} to load the model does not exist."
            raise FileNotFoundError(msg)
        project = folder / "project"
        if not project.exists():
            project = folder / "default.prj"  # backward compat
        meshes = folder / "meshes"
        execution = folder / "execution.yaml"
        model = cls(project=project, meshes=meshes, execution=execution)
        model._bind_to_path(folder)
        return model

    @classmethod
    def from_id(cls, model_id: str) -> "Model":
        """
        Load a Model from the user storage path using its ID.

        :param model_id: The unique ID of the :class:`ogstools.Model` to load.
        :returns: A :class:`ogstools.Model` instance with Project and Meshes loaded from disk.
        """
        model_folder = StorageBase.saving_path() / "Model" / model_id
        if not model_folder.exists():
            msg = f"No model found at {model_folder}"
            raise FileNotFoundError(msg)

        model = cls.from_folder(model_folder)
        model._id = model_id
        return model

    def _propagate_target(self) -> None:
        """
        If for this object a saving location was given but for the subobjects (Meshes and Project) not the saving location for the subobjects is derived from the saving location of this object
        """

        if not self.meshes.user_specified_target:
            self.meshes._next_target = self.next_target / "meshes"
        if not self.project.user_specified_target:
            self.project._next_target = self.next_target / "project"
        if not self.execution.user_specified_target:
            self.execution._next_target = self.next_target / "execution.yaml"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Model):
            return NotImplemented
        # most expensive at last
        return (
            self.execution == other.execution
            and self.project == other.project
            and self.meshes == other.meshes
        )

    def _save_impl(
        self, dry_run: bool = False, overwrite: bool | None = None
    ) -> list[Path]:
        self.meshes.num_partitions = self.execution.mpi_ranks

        files: list[Path] = []

        files += self._save_or_link_child(
            self.meshes, self.next_target / "meshes", dry_run, overwrite
        )

        # Project always needs to be saved first (if not yet), then linked
        if (
            not self.project.active_target
            or not self.project.active_target.exists()
        ):
            files += self.project.save(dry_run=dry_run, overwrite=overwrite)
        self.project.link(self.next_target / "project", dry_run)

        files += self._save_or_link_child(
            self.execution,
            self.next_target / "execution.yaml",
            dry_run,
            overwrite,
        )

        return files

    def save(
        self,
        target: Path | str | None = None,
        overwrite: bool | None = None,
        dry_run: bool = False,
        archive: bool = False,
        id: str | None = None,
    ) -> list[Path]:
        """
        Save the Model to disk, including Project, Meshes, and Execution config.

        Creates a folder structure containing:
        - project/: Subfolder with project.prj and associated files
        - meshes/: Subfolder with mesh files
        - execution.yaml: Execution configuration

        By default, uses symlinks for efficiency. Use archive=True to create
        a standalone copy with all data materialized.

        :param target:      Path to the folder where the model should be saved.
                            If None, uses a default location based on the model ID.
        :param overwrite:   If True, existing files are overwritten. If False,
                            raises an error if files already exist.
        :param dry_run:     If True, simulates the save without writing files,
                            but returns the list of files that would be created.
        :param archive:     If True, materializes all symlinks by copying
                            referenced data (may be time and space intensive).
        :param id:          Optional identifier. Mutually exclusive with target.
        :returns:           List of Paths to saved files (including meshes,
                            project, and execution configuration).
        """
        user_defined = self._pre_save(target, overwrite, dry_run, id=id)
        files = self._save_impl(dry_run=dry_run, overwrite=overwrite)
        self._post_save(user_defined, archive, dry_run)
        return files

    def run(
        self,
        sim_output: Path | str | None = None,
        overwrite: bool | None = None,
        id: str | None = None,
    ) -> "Simulation":
        """
        Run a simulation to completion and wait for it to finish.

        This is a convenience method that starts the simulation and blocks
        until it completes. For stepwise control, use start() instead.

        :param sim_output:  Optional path where simulation output (results,
                            logs) should be written. If None, uses a default location.
        :param overwrite:   If True, overwrite existing output directory.
        :param id:          Optional identifier for the resulting Simulation.
        :returns:           A :class:`ogstools.Simulation` object containing the completed
                            simulation results and metadata.
        """
        sim_controller = self.controller(
            sim_output=sim_output, overwrite=overwrite
        )
        return sim_controller.run(target=sim_output, id=id)

    def controller(
        self,
        sim_output: Path | str | None = None,
        dry_run: bool = False,
        overwrite: bool | None = None,
    ) -> SimulationController:
        """
        Start a simulation and return a controller for execution management.

        The type of controller returned depends on the execution configuration:

        - OGSInteractiveController: Allows stepwise control (execute_time_step,
          inspect intermediate results) when execution.interactive is True
        - OGSNativeController: Runs to completion when execution.interactive is False

        :param sim_output:  Optional path where simulation output should be written.
                            If None, uses a default location.
        :param dry_run:     If True, prints the command that would be executed
                            but does not actually run the simulation.
        :returns:           A SimulationController for managing the simulation.
        """

        if not self.is_saved:
            self._propagate_target()
            self._save_impl(overwrite=overwrite, dry_run=dry_run)
            self._post_save(False, False, False)

        if dry_run:
            msg = "Dry run not implemented"
            raise NotImplementedError(msg)

        if self.execution.interactive:

            from .interactive_simulation_controller import (
                OGSInteractiveController,
            )

            return OGSInteractiveController(
                model_ref=self, sim_output=sim_output, overwrite=overwrite
            )

        from .native_simulation_controller import OGSNativeController

        return OGSNativeController(
            model_ref=self, sim_output=sim_output, overwrite=overwrite
        )

    def _component_repr(
        self, obj: typing.Any, name: str, load_method: str
    ) -> str:
        """Helper to generate repr string for a component (project/meshes/execution)."""
        if obj.user_specified_id:
            return f'{name}.from_id("{obj.id}")'
        if obj.is_saved:
            path = str(obj.active_target)
            return f"{name}.{load_method}({path!r})"
        # Not saved yet
        path = str(obj.next_target)
        return f"{name}.{load_method}({path!r})"

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        base_repr = super().__repr__()

        if self.user_specified_id:
            construct = f'{cls}.from_id("{self._id}")'
        elif self.is_saved:
            construct = f"{cls}.from_folder({str(self.active_target)!r})"
        else:
            # Show how to reconstruct children
            project_construct = self._component_repr(
                self.project, "Project", "from_file"
            )
            meshes_construct = self._component_repr(
                self.meshes, "Meshes", "from_folder"
            )
            execution_construct = self._component_repr(
                self.execution, "Execution", "from_file"
            )

            construct = (
                f"{cls}(\n"
                f"  project={project_construct},\n"
                f"  meshes={meshes_construct},\n"
                f"  execution={execution_construct}\n"
                f")"
            )

        save_hint = (
            "\nNote: Components must be saved before use"
            if not self.is_saved
            else ""
        )
        return f"{construct}{save_hint}\n{base_repr}"

    def __str__(self) -> str:
        base_str = super().__str__()
        lines = [
            base_str,
            f"  {self._component_status_str(self.project, 'Project')}",
            f"  {self._component_status_str(self.meshes, 'Meshes')}",
            f"  {self._component_status_str(self.execution, 'Execution')}",
        ]
        return "\n".join(lines)

    def __deepcopy__(self, memo: dict) -> "Model":
        # Avoid duplicate copies
        if id(self) in memo:
            return memo[id(self)]

        project = copy.deepcopy(self.project, memo)
        meshes = copy.deepcopy(self.meshes, memo)
        execution = copy.deepcopy(self.execution, memo)

        new = self.__class__(
            project=project,
            meshes=meshes,
            execution=execution,
        )

        memo[id(self)] = new
        return new

    def plot_constraints(self, **kwargs: typing.Any) -> plt.Figure:
        """Plot the meshes with annotated boundary conditions and source terms.

        keyword arguments: see :func:`~ogstools.plot.contourf`
        """

        meshes = self.meshes
        tmp_path = Path(mkdtemp(prefix="plot_constraints"))
        if (
            self.project.geometry
            and self.project.geometry.active_target is not None
        ):
            gml_file = self.project.geometry.active_target
            assert gml_file is not None
            if not meshes.is_saved:
                meshes.save()

            meshes.add_gml_subdomains(
                self.project.meshpaths(self.meshes.active_target)[0],
                gml_file,
                tmp_path,
            )

        constraints = self.project.constraints_labels()
        unused = set(meshes.subdomains.keys()) - set(constraints.keys())
        for subdomain in unused:
            meshes.pop(subdomain)

        from ogstools.plot import setup

        loc = kwargs.pop("loc", "upper left")
        bbox = kwargs.pop("bbox_to_anchor", (1.05, 1))
        fontsize = kwargs.get("fontsize", setup.fontsize)
        leg_fontsize = kwargs.get("leg_fontsize", 0.9 * fontsize)

        fig = meshes.plot(**kwargs, loc=loc, bbox_to_anchor=bbox)
        ax: plt.Axes = fig.axes[0]

        handles, labels = ax.get_legend_handles_labels()
        for meshname, label in constraints.items():
            idx = labels.index(meshname)
            labels[idx] = label
        ax.legend(
            handles, labels, loc=loc, fontsize=leg_fontsize,
            bbox_to_anchor=bbox, borderaxespad=0.0, numpoints=1,
        )  # fmt: skip

        return fig
