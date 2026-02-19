# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#


import abc
import signal
import typing
from enum import Enum
from pathlib import Path

from .result import Result

if typing.TYPE_CHECKING:
    from .model import Model
    from .simulation import Simulation


class SimulationStatus(Enum):
    """
    Enumeration of possible simulation states.

    Attributes:
        not_started: Simulation has not been started yet.
        running: Simulation is currently executing.
        paused: Simulation is paused (interactive mode only).
        done: Simulation completed successfully.
        error: Simulation terminated with an error.
    """

    not_started = 0  # open
    running = 1
    paused = 2
    done = 3  # reached end_time
    error = 4  # with error
    unknown = 5


class SimulationController(abc.ABC):
    """
    Abstract base class for controlling OGS simulation execution.

    Provides a unified interface for running simulations, whether in
    interactive stepwise mode or batch mode. Handles signal interruption
    (SIGINT, SIGTERM) and manages simulation status.

    Concrete implementations:
    - OGSInteractiveController: For stepwise execution control
    - OGSNativeController: For batch execution
    """

    Status = SimulationStatus

    def __init__(
        self,
        model_ref: "Model",
        sim_output: Path | str | None = None,
        overwrite: bool | None = None,
    ) -> None:
        """
        Initialize a SimulationController.

        :param model_ref:   The :class:`ogstools.Model` to simulate.
        :param sim_output:  Optional path for simulation output directory.
                            If None, uses a default location.
        :param overwrite:   If True, overwrite existing output directory.
        """
        self.model_ref = model_ref
        self._args_list: list[str] = []
        self.result = Result(sim_output)
        self.result._pre_save(overwrite=overwrite)
        self.result.next_target.mkdir(parents=True, exist_ok=True)

        self._interrupted = False
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, signum: int, _: typing.Any) -> None:
        self._interrupted = True
        print(f"Received signal {signum}, stopping...")

    @property
    def is_interrupted(self) -> bool:
        """
        Check if an interrupt signal was received and reset the flag.

        :returns: True if SIGINT or SIGTERM was received, False otherwise.
        """
        interrupted = self._interrupted
        self._interrupted = False
        return interrupted

    @abc.abstractmethod
    def terminate(self) -> bool:
        """
        Terminate the simulation immediately.

        :returns: True if termination was successful, False otherwise.
        """

    @abc.abstractmethod
    def run(
        self, target: Path | str | None = None, id: str | None = None
    ) -> "Simulation":
        """
        Run the simulation to completion.

        :param target:  Optional path for the simulation output directory.
        :param id:      Optional identifier for the resulting Simulation.
        :returns: A :class:`ogstools.Simulation` object containing the completed simulation.
        """

    def _create_simulation(
        self, target: Path | str | None = None, id: str | None = None
    ) -> "Simulation":
        """
        Create a Simulation object with optional id and target.

        :param target:  Optional path for the simulation output directory.
        :param id:      Optional identifier for the Simulation.
        :returns: A configured :class:`ogstools.Simulation` object.
        """
        from .simulation import Simulation

        sim = Simulation(self.model_ref, result=self.result)
        if id:
            sim.id = id
            return sim
        if target:
            sim._next_target = Path(target)
            sim.user_specified_target = True

        sim._propagate_target()
        return sim

    @property
    @abc.abstractmethod
    def status(self) -> SimulationStatus:
        """
        Get the current simulation status.

        :returns: Current SimulationStatus.
        """

    @abc.abstractmethod
    def status_str(self) -> str:
        """
        Get a human-readable status string.

        :returns: String describing the current simulation state.
        """

    @property
    def log_file(self) -> Path:
        """Get the path to the log file."""
        return self.result.log_file

    @property
    def meshseries_file(self) -> Path:
        """Get the path to the mesh series file."""
        return (
            self.result.next_target / self.model_ref.project.meshseries_file()
        )

    @property
    def cmd(self) -> str:
        """Get the full command used to run the simulation."""
        return (
            f"{self.model_ref.execution.ogs_bin_path}"
            f" {self.model_ref.project.prjfile}"
            f" -m {self.model_ref.meshes.active_target}"
            f" -o {self.result.next_target}"
        )

    def error_report(self) -> str:
        """
        Generate an error report if the simulation failed.

        Includes the last lines of the log file if available.

        :returns: A formatted error report string.
        """
        msg = ""
        if self.status == SimulationController.Status.not_started:
            msg += "OGS not (yet) started."
            return msg

        if self.status != SimulationStatus.error:
            msg += "Still running."
            return msg

        msg += "An error occurred."
        if not self.result.log_file.exists():
            msg += f"No log file written to: {self.result.log_file}."
            return msg

        msg += f"Last lines of {self.result.log_file} are:"
        with self.result.log_file.open() as lf:
            last_lines = "\n".join(lf.readlines()[-10:])
            msg += last_lines
        return msg

    def __repr__(self) -> str:
        from .storage import StorageBase

        model_target = StorageBase._format_path(
            self.model_ref.next_target, for_repr=True
        )
        result_target = StorageBase._format_path(
            self.result.next_target, for_repr=True
        )
        meshseries = StorageBase._format_path(
            self.meshseries_file, for_repr=True
        )
        logfile = StorageBase._format_path(self.log_file, for_repr=True)

        return (
            f"Model.from_folder({model_target}).controller(sim_output={result_target}, overwrite=True)\n"
            f"meshseries_file={meshseries}\n"
            f"logfile={logfile}\n"
            f"status={self.status_str()}\n"
            f"execution.interactive={self.model_ref.execution.interactive}"
        )

    def __str__(self) -> str:
        from .storage import StorageBase

        mode = (
            "Interactive" if self.model_ref.execution.interactive else "Native"
        )

        return (
            f"SimulationController ({mode})\n"
            f"Model:        {StorageBase._format_path(self.model_ref.next_target)}\n"
            f"Result:       {StorageBase._format_path(self.result.next_target)}\n"
            f"MeshSeries:   {StorageBase._format_path(self.meshseries_file)}\n"
            f"Logfile:      {StorageBase._format_path(self.log_file)}\n"
            f"{self.status_str()}\n"
        )
