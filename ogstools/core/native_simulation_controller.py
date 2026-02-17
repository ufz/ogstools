# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#


import contextlib
import time
import typing
from pathlib import Path

import psutil

from .simulation_controller import SimulationController, SimulationStatus

if typing.TYPE_CHECKING:
    from .model import Model
    from .simulation import Simulation


class OGSNativeController(SimulationController):
    """
    Controller for batch execution of OGS simulations.

    Runs OGS as a separate process and waits for completion. Does not
    support stepwise execution or intermediate state inspection.
    Suitable for standard production runs. Objects should be used as read-only.
    """

    def __init__(
        self,
        model_ref: "Model",
        sim_output: Path | str | None = None,
        overwrite: bool | None = None,
    ) -> None:
        """
        Initialize a native simulation controller.

        :param model_ref:   The :class:`ogstools.Model` to simulate.
        :param sim_output:  Optional path for simulation output directory.
        :param overwrite:   If True, overwrite existing output directory.
        """
        super().__init__(
            model_ref=model_ref, sim_output=sim_output, overwrite=overwrite
        )
        self._args_list = [
            "-m",
            str(model_ref.meshes.active_target),
            "-o",
            str(self.result.next_target),
        ]
        args_str = " ".join(self._args_list)

        self.process = model_ref.project.run_model(
            args=args_str,
            logfile=self.result.next_target / "log.txt",
            write_logs=model_ref.execution.write_logs,
            background=True,
            wrapper=model_ref.execution.wrapper,
            container_path=model_ref.execution.container_path,
        )

        self.runtime_start = time.time()
        self.runtime_end: float | None = None

    def terminate(self) -> bool:
        """
        Terminate the simulation if it is running.

        Attempts to gracefully terminate the OGS process and all child
        processes. If graceful termination fails, forcefully kills them.

        :returns: True if the run was terminated successfully, False otherwise.
        """
        timeout = 3
        proc = self.process

        if proc.poll() is not None:
            print(
                "Requested termination - but the Simulation is already finished."
            )
            return True
        try:
            p = psutil.Process(proc.pid)

            children = p.children(recursive=True)
            for c in children:

                with contextlib.suppress(psutil.NoSuchProcess):
                    c.terminate()

            psutil.wait_procs(children, timeout=timeout)

            for c in children:
                if c.is_running():
                    with contextlib.suppress(psutil.NoSuchProcess):
                        c.kill()

            psutil.wait_procs(children, timeout=timeout)

            if p.is_running():
                try:
                    p.terminate()
                    p.wait(timeout)
                except psutil.TimeoutExpired:
                    p.kill()
                    p.wait()

            return not p.is_running()

        except psutil.NoSuchProcess:
            with contextlib.suppress(Exception):
                proc.wait(timeout=timeout)
            return True

    def run(
        self, target: Path | str | None = None, id: str | None = None
    ) -> "Simulation":
        """
        Wait for the simulation to complete and return a Simulation object.

        Blocks until the OGS process finishes.

        :param target:  Optional path for the simulation output directory.
        :param id:      Optional identifier for the resulting Simulation.
        :returns: A :class:`ogstools.Simulation` object containing the completed simulation.
        """
        self.ret_code = self.process.wait()
        self.runtime_end = time.time()
        return self._create_simulation(target=target, id=id)

    def status_str(self) -> str:
        """
        Get a human-readable status description.

        :returns: String describing the current simulation state and runtime.
        """
        match self.process.poll():
            case None:
                runtime = time.time() - self.runtime_start
                stat_str = f"running for {runtime} s."
            case 0:
                stat_str = "finished successfully."
                if self.runtime_end:
                    runtime = self.runtime_end - self.runtime_start
                    stat_str += f"\nExecution took {runtime} s"
                elif self.result.log_file and self.result.log_file.exists():
                    stat = self.result.log_file.stat()
                    runtime = stat.st_mtime - stat.st_ctime
                    stat_str += f"\nExecution took {runtime} s"
            case code:
                stat_str = f"terminated with error code {code}."
                if self.process.returncode != 0:
                    stat_str += self.error_report()

        return "Status: " + stat_str

    @property
    def status(self) -> SimulationStatus:
        """
        Get the current simulation status.

        Queries the process state to determine if the simulation is
        running, completed, or encountered an error.

        :returns: Current SimulationStatus.
        """
        if not self.process:
            return SimulationStatus.not_started
        match self.process.poll():
            case None:
                return SimulationStatus.running
            case 0:
                return SimulationStatus.done
            case _:
                return SimulationStatus.error
        # paused can not be reached, if suspended process sim is in State running
