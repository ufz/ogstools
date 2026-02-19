# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

import os
import threading
import time
import typing
from collections.abc import Sequence
from pathlib import Path
from time import sleep

from pyvista import UnstructuredGrid

from .model import Model
from .simulation_controller import SimulationController, SimulationStatus

if typing.TYPE_CHECKING:
    from .simulation import Simulation


class OGSSimulationInitializationError(Exception):
    """
    Exception raised when OGSSimulation initialization fails.

    This can occur when:
    - OGSSimulation returns None (initialization failed)
    - Multiple OGSSimulation instances are created in parallel (Issue #3589)
    """


class _NativeOutputCapture:
    # TODO:: Issue #3589 - Not thread safe! Multiple interactive simulations cannot run in parallel
    """
    Temporary solution for interactive SimulationController to capture the log output into a file.
    It is not thread safe! Logfile can be empty/corrupted when running multiple interactive simulations in parallel
    """

    def __init__(self, file_path: Path):
        self._r_fd, self._w_fd = os.pipe()
        self._old_stdout_fd = os.dup(1)
        self._old_stderr_fd = os.dup(2)
        self.file_path = file_path

    def start(self) -> None:
        os.dup2(self._w_fd, 1)
        os.dup2(self._w_fd, 2)
        self._thread = threading.Thread(target=self._reader)
        self._thread.daemon = True
        self._thread.start()

    def _reader(self) -> None:
        with (
            Path(self.file_path).open("w") as f,
            os.fdopen(self._r_fd, "r") as r,
        ):
            f.writelines(r)

    def stop(self) -> None:
        # Restore original stdout/stderr
        os.dup2(self._old_stdout_fd, 1)
        os.dup2(self._old_stderr_fd, 2)
        os.close(self._w_fd)
        self._thread.join()


class OGSInteractiveController(SimulationController):
    """
    Controller for interactive stepwise execution of OGS simulations.

    Allows fine-grained control over simulation execution including:
    - Executing individual time steps
    - Inspecting intermediate mesh states
    - Querying current simulation time
    - Pausing and resuming execution

    Requires OGS to be built with interactive mode support.
    """

    def __init__(
        self,
        model_ref: "Model",
        sim_output: Path | str | None = None,
        overwrite: bool | None = None,
    ) -> None:
        """
        Initialize an interactive simulation controller.

        :param model_ref:   The :class:`ogstools.Model` to simulate.
        :param sim_output:  Optional path for simulation output directory.
        :param overwrite:   If True, overwrite existing output directory.
        """
        super().__init__(
            model_ref=model_ref, sim_output=sim_output, overwrite=overwrite
        )
        from ogs.OGSSimulator import OGSSimulation

        self._capture = (
            _NativeOutputCapture(
                file_path=Path(self.result.next_target) / "log.txt"
            )
            if model_ref.execution.write_logs
            else None
        )

        # TODO: Apply all model execution parameters
        self._args_list = [
            "",
            str(model_ref.project.prjfile),
            "-m",
            str(model_ref.meshes.active_target),
            "-o",
            str(self.result.next_target),
            "-l",
            model_ref.execution.log_level,
        ]

        try:
            if self._capture:
                self._capture.start()
            self.sim = OGSSimulation(self._args_list)
            if self.sim is None:
                # TODO:: Issue #3589 - OGSSimulation cannot be initialized multiple times in parallel
                msg = (
                    "OGSSimulation initialization failed (returned None). "
                    "This may occur when multiple interactive simulations "
                    "are initialized in parallel (Issue #3589)."
                )
                raise OGSSimulationInitializationError(msg)
            self._status = SimulationController.Status.running
        except OGSSimulationInitializationError:
            self._status = SimulationController.Status.error
            raise
        except Exception:
            self._status = SimulationController.Status.error
            print(self.status_str())

        self.runtime_start = time.time()
        self.runtime_end: float | None = None

    @property
    def status(self) -> SimulationStatus:
        """Get the current simulation status."""
        return self._status

    def terminate(self) -> bool:
        """
        Terminate the simulation immediately.

        Closes the OGS simulator and stops log capture if active.

        :returns: True if termination was successful.
        """
        self.runtime_end = time.time()
        ret = self.sim.close()
        if self._capture:
            self._capture.stop()
        self._status = SimulationStatus.done
        return ret

    def run(
        self, target: Path | str | None = None, id: str | None = None
    ) -> "Simulation":
        """
        Run the simulation to completion.

        Executes time steps until the simulation reaches end_time or
        encounters an error. After completion, closes the simulator
        and returns a Simulation object.

        :param target:  Optional path for the simulation output directory.
        :param id:      Optional identifier for the resulting Simulation.
        :returns: A :class:`ogstools.Simulation` object containing the completed simulation.
        """
        while (
            self.current_time < self.end_time
            and not self.is_interrupted
            and self._status == self.Status.running
        ):
            self._status = self.execute_time_step()
            sleep(0.01)  # Must have, if we want to pause the simulation

        self.sim.close()
        if self._capture:
            self._capture.stop()

        self._status = (
            self.Status.done
            if self._status == self.Status.running
            else self.Status.error
        )
        self.runtime_end = time.time()

        return self._create_simulation(target=target, id=id)

    @property
    def current_time(self) -> float:
        """
        Get the current model time.

        :returns: Current time value in s.
        """
        return self.sim.current_time()

    @property
    def end_time(self) -> float:
        """
        Get the configured model end time.

        :returns: End time value in s.
        """
        return self.sim.end_time()

    def execute_time_step(self) -> "SimulationStatus":
        """
        Execute a single time step of the simulation.

        Advances the simulation by one time step and updates the status.

        :returns: The updated SimulationStatus after executing the time step.
        """
        status = self.sim.execute_time_step()
        if status == 0:  # SUCCESS
            self._status = SimulationController.Status.running
        else:
            self._status = SimulationController.Status.error
        return self._status

    def mesh(
        self, name: str, variables: Sequence[str] | None = None
    ) -> UnstructuredGrid:
        """
        Retrieve the current mesh state during simulation.

        :param name:        Name of the mesh to retrieve.
        :param variables:   Optional list of variable names to include.
                            If None, includes all variables.
        :returns:           UnstructuredGrid containing the mesh and data.
        """
        from ogstools.mesh.cosim import from_simulator

        return from_simulator(self.sim, name, variables)

    def status_str(self) -> str:
        """
        Get a human-readable status description.

        :returns: String describing the current simulation state and runtime.
        """

        match self._status:
            case SimulationController.Status.running:
                runtime = time.time() - self.runtime_start
                stat_str = f"running for {runtime} s."
            case SimulationController.Status.done:
                assert self.runtime_end
                runtime = self.runtime_end - self.runtime_start
                stat_str = "finished successfully."
                if runtime > 0:
                    stat_str += f"\nExecution took {runtime} s."
            case SimulationController.Status.error:
                stat_str = "terminated with error."
                stat_str += self.error_report()
            case SimulationController.Status.not_started:
                stat_str = "not started."
            case _:
                stat_str = "unknown."

        return "Status: " + stat_str
