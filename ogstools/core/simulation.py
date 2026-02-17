# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

import typing
from copy import deepcopy
from pathlib import Path

from typing_extensions import Self

from ogstools.core.model import Model
from ogstools.core.result import Result
from ogstools.core.simulation_controller import SimulationStatus
from ogstools.core.storage import StorageBase
from ogstools.logparser.log import Log
from ogstools.meshseries._meshseries import MeshSeries


class Simulation(StorageBase):
    """
    A completed or ongoing OGS simulation with associated model and results.

    Combines a Model (project setup) with Results (output data) and provides
    convenient access to simulation outputs like mesh series and log files.
    Simulations can be saved, loaded, and analyzed after completion.
    """

    Status = SimulationStatus

    @classmethod
    def from_id(cls, simulation_id: str) -> Self:
        """
        Load a Simulation from the user storage path using its ID.

        :param simulation_id:   The unique identifier of the :class:`ogstools.Simulation` to load.
        :returns:               A :class:`ogstools.Simulation` instance restored from disk.

        :raises FileNotFoundError: If no simulation with the given ID exists.
        """
        sim_folder = (
            StorageBase.saving_path() / "Simulation" / f"{simulation_id}"
        )

        if not sim_folder.exists():
            msg = f"No simulation found at {sim_folder}"
            raise FileNotFoundError(msg)

        simulation = cls.from_folder(sim_folder)
        simulation._id = simulation_id
        return simulation

    @classmethod
    def from_folder(cls, sim_folder: Path) -> Self:
        """
        Load a Simulation from a folder following OGSTools conventions.

        Expects the folder structure created by Simulation.save():
        - model/: Subfolder containing the Model
        - result/: Subfolder containing simulation results

        :param sim_folder:  Path to the folder containing the simulation.
        :returns:           A :class:`ogstools.Simulation` instance loaded from the folder.

        :raises FileNotFoundError: If required components are not found.
        """
        sim_folder = Path(sim_folder)
        model = Model.from_folder(sim_folder / "model")
        result = Result(sim_folder / "result")
        return cls(model, result)

    def __init__(
        self,
        model: Model,
        result: Result,
    ) -> None:
        """
        Initialize a Simulation object.

        :param model:       The :class:`ogstools.Model` used for this simulation.
        :param result:      The Result object containing simulation output.
        """

        super().__init__("Simulation")
        self.model = model
        self._result = result
        result._bind_to_path(result.next_target)

        self._log: Log | None = None
        self._meshseries: MeshSeries | None = None

    def __deepcopy__(self, memo: dict) -> "Simulation":
        """
        Create a full deep copy of this Simulation, including model and result.
        """
        new_model = deepcopy(self.model, memo)
        new_result = deepcopy(self._result, memo)
        new_sim = Simulation(new_model, new_result)
        new_sim._meshseries = (
            deepcopy(self._meshseries, memo) if self._meshseries else None
        )
        new_sim._log = deepcopy(self._log, memo) if self._log else None

        return new_sim

    def _component_repr(
        self, obj: typing.Any, name: str, load_method: str | None = None
    ) -> str:
        """Helper to generate repr string for a component (model/result).

        :param obj: The component object
        :param name: Class name (e.g., "Model", "Result")
        :param load_method: Class method name (e.g., "from_folder") or None
                           to use the constructor directly
        """
        if getattr(obj, "user_specified_id", False):
            return f'{name}.from_id("{obj.id}")'

        if getattr(obj, "is_saved", False):
            path = str(obj.active_target)
        elif (next_t := getattr(obj, "next_target", None)) is not None:
            path = str(next_t)
        else:
            return f"{name}(...)"

        if load_method:
            return f"{name}.{load_method}({path!r})"
        # Use constructor directly (e.g., Result(sim_output=...))
        return f"{name}({path!r})"

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        base_repr = super().__repr__()

        if self.user_specified_id:
            construct = f'{cls}.from_id("{self._id}")'
        elif self.is_saved:
            construct = f"{cls}.from_folder({str(self.active_target)!r})"
        else:
            # Show how to reconstruct from its children
            model_construct = self._component_repr(
                self.model, "Model", "from_folder"
            )
            result_construct = self._component_repr(self._result, "Result")

            construct = (
                f"{cls}(model={model_construct}, "
                f"result={result_construct})\n"
                f"log_file={str(self.log_file)!r}\n"
                f"meshseries_file={str(self.meshseries_file)!r}"
            )

        save_hint = (
            "\nNote: Components must be saved before use"
            if not self.is_saved
            else ""
        )
        return f"{construct}{save_hint}\nstatus={self.status_str}\n{base_repr}"

    def __str__(self) -> str:
        base_str = super().__str__()
        lines = [
            base_str,
            f"  {self._component_status_str(self.model, 'Model')}",
            f"  {self._component_status_str(self._result, 'Result')}",
            f"  MeshSeries: {self._format_path(self.meshseries_file)}",
            f"  {self.status_str}",
        ]
        return "\n".join(lines)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Simulation):
            return False
        return self.model == other.model and self._result == other._result

    @property
    def status(self) -> SimulationStatus:
        """
        Get the current simulation status from the log file.

        Checks the log for errors or critical messages to determine if the
        simulation completed successfully or terminated with an error.

        :returns: SimulationStatus enum value (done or error).
        """
        from .simulation_controller import SimulationController

        if not self.model.execution.write_logs:
            return SimulationController.Status.unknown

        if not self.log_file.exists() and self.model.execution.write_logs:
            return SimulationController.Status.error

        # Parse the log to check for errors
        try:
            log_df = self.log.df_log
            if "type" in log_df.columns:
                has_errors = (
                    log_df["type"]
                    .str.contains("error|critical", case=False, na=False)
                    .any()
                )
                if has_errors:
                    return SimulationController.Status.error
        except Exception:
            return SimulationController.Status.error

        return SimulationController.Status.done

    @property
    def status_str(self) -> str:
        """
        Get a human-readable status description of the simulation.

        Includes information about completion status and result availability.

        :returns: String describing the simulation state.
        """
        from .simulation_controller import SimulationController

        status = self.status

        if status == SimulationController.Status.done:
            if self.meshseries_file.exists():
                return "Status: completed successfully (results available)"
            return "Status: completed successfully (results pending)"
        last_lines = self.model.project._failed_run_print_log_tail(
            self.model.execution.write_logs
        )
        return f"Status: terminated with error\n{last_lines}"

    @property
    def cmd(self) -> str:
        """Get the full command used to run the simulation."""
        return (
            f"{self.model.execution.ogs_bin_path}"
            f" {self.model.project.prjfile}"
            f" -m {self.model.meshes.active_target}"
            f" -o {self._result.next_target}"
        )

    @property
    def log_file(self) -> Path:
        """Get the absolute path to the log file."""
        if self.is_saved and self.active_target is not None:
            return self.active_target / "result" / self._result._log_filename
        return self._result.log_file

    @property
    def log(self) -> Log:
        """
        Access the parsed log file of this simulation.

        Lazily loads and parses the log file on first access.

        :returns: A Log object for querying simulation log data.
        """
        if not self._log:
            self._log = Log(self.log_file)
        return self._log

    @property
    def meshseries_file(self) -> Path:
        """
        Get the path to the mesh series output file.

        :returns: Path to the mesh series file (pvd, xdmf, etc.).
        """
        return self._result.next_target / self.model.project.meshseries_file()

    @property
    def result(self) -> MeshSeries:
        """
        Access the result mesh series of this simulation.

        Lazily loads the mesh series on first access.

        :returns: A MeshSeries containing the simulation results.
        """
        if self.status == SimulationStatus.running:
            print("Simulation still running. MeshSeries can be incomplete.")
        if not self.meshseries_file.exists():
            msg = f"Can not find simulation result for a MeshSeries. Status of the simulation: {self.status_str}."
            raise ValueError(msg)
        if not self._meshseries:
            self._meshseries = MeshSeries(self.meshseries_file)
        return self._meshseries

    def _propagate_target(self) -> None:
        if not self.model.user_specified_target:
            self.model._next_target = self.next_target / "model"

        if not self._result.user_specified_target:
            self._result._next_target = self.next_target / "result"

    def _save_impl(
        self, dry_run: bool = False, overwrite: bool | None = None
    ) -> list[Path]:
        files: list[Path] = []
        files += self._save_or_link_child(
            self.model, self.next_target / "model", dry_run, overwrite
        )

        files += self._save_or_link_child(
            self._result, self.next_target / "result", dry_run, overwrite
        )

        self.materialize_symlink(self.next_target / "result", recursive=True)
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
        Save the Simulation to disk, including Model and Result.

        Creates a folder structure containing:
        - model/: The Model (project, meshes, execution config)
        - result/: The simulation results

        :param target:      Path to save the :class:`ogstools.Simulation`. If None, uses a default location.
        :param overwrite:   If True, overwrite existing files. Defaults to False.
        :param dry_run:     If True, simulate save without writing files.
        :param archive:     If True, materialize all symlinks by copying data.
        :param id:          Optional identifier. Mutually exclusive with target.
        :returns:           List of Paths to saved files.
        """
        user_defined = self._pre_save(target, overwrite, dry_run, id=id)
        files = self._save_impl(dry_run=dry_run, overwrite=overwrite)
        self._post_save(user_defined, archive, dry_run)
        return files
