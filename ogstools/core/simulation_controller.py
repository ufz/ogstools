# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause


import abc
import subprocess
import threading
import time
import typing
from enum import Enum
from pathlib import Path

from .result import Result

if typing.TYPE_CHECKING:
    from ogstools.logparser.monitor import Monitor

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
        self._dashboard_processes: list[subprocess.Popen] = []
        self._notebook_monitors: list[Monitor] = []

    def _close_dashboards(self) -> None:
        """Terminates all monitors opened from this controller"""
        for process in self._dashboard_processes:
            if process.poll() is None:
                process.terminate()
        for monitor in self._notebook_monitors:
            if monitor._observer is not None and monitor._observer.is_alive():
                monitor._observer.stop()

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
            # This is basically self.save(target) but without pre_save
            sim._next_target = Path(target)
            sim.user_specified_target = True
            sim._save_impl()
            sim._post_save(user_defined=True)

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

    def plot_log(
        self,
        log_data: str | list[list[str]] = "step_start_time",
        time_y_axis_type: str = "linear",
        time_window_length: int = 0,
        iteration_window_length: int = 0,
        update_interval: float = 2.0,
        notebook: bool = False,
    ) -> subprocess.Popen | None:
        """
        Open the interactive Bokeh monitoring dashboard for this simulation.

        By default this launches the same dashboard as the ``ogsmonitor``
        command line tool, in a real browser tab, which renders reliably
        across notebook environments (plain Jupyter, JupyterLab, VS Code's
        Jupyter extension, ...). Pass ``notebook=True`` to instead embed the
        plot directly in the notebook cell's output.

        :param log_data:  Plot type. Can be a single string or a list of list of strings.
                            E.g., [['step_start_time', 'step_size'], ['assembly_time', 'linear_solver_time']]
        :param time_y_axis_type: Type of the y-axis ('linear' or 'log') for simulation time-based data.
        :param time_window_length:     Length of the time window (number of timesteps) for the plot. 0 Plots the whole log file.
        :param iteration_window_length: Length of the iteration window (number of iterations) for the plot. 0 Plots the whole log file.
        :param update_interval:        Interval in seconds between plot updates.
        :param notebook: If True, embed the plot in the notebook cell via
                          Bokeh's ``push_notebook()`` instead of opening a
                          browser tab. A background thread redraws it every
                          ``update_interval`` seconds until the simulation
                          ends or :meth:`terminate` is called; the call
                          itself returns immediately.

                          .. warning::
                             Bokeh's ``push_notebook()`` live updates do not
                             work in VS Code's Jupyter extension: this is an
                             unresolved upstream limitation
                             (`bokeh/jupyter_bokeh#199
                             <https://github.com/bokeh/jupyter_bokeh/issues/199>`_),
                             not something ogstools can work around. The plot
                             draws once but never animates there. It works
                             correctly in classic Jupyter and JupyterLab. In
                             VS Code, use the default ``notebook=False``
                             (browser tab) instead.
        :returns: The running Bokeh server subprocess when ``notebook=False``.
                  It is also closed by :meth:`terminate`; call
                  ``.terminate()`` on it directly for earlier, standalone
                  control. Returns None when ``notebook=True`` — use
                  :meth:`terminate` to stop it early.
        """
        if notebook:
            self._plot_log_notebook(
                log_data,
                time_y_axis_type,
                time_window_length,
                iteration_window_length,
                update_interval,
            )
            return None

        from ogstools.logparser.monitor_cli import (
            launch_dashboard,
            write_monitor_config,
        )

        config = write_monitor_config(
            log_data=log_data,
            time_y_axis_type=time_y_axis_type,
            time_window_length=time_window_length,
            iteration_window_length=iteration_window_length,
            update_interval=update_interval,
        )
        process = launch_dashboard(self.log_file, config=config)
        self._dashboard_processes.append(process)
        return process

    def _plot_log_notebook(
        self,
        log_data: str | list[list[str]],
        time_y_axis_type: str,
        time_window_length: int,
        iteration_window_length: int,
        update_interval: float,
    ) -> None:
        """Embed a live-updating monitoring plot in the current notebook cell."""
        from bokeh.io import output_notebook, show

        from ogstools.logparser.monitor import Monitor

        monitor = Monitor(notebook_execution=True)
        monitor.start_log_file_handler(self.log_file)
        self._notebook_monitors.append(monitor)
        grid_layout = monitor.build_layout(log_data, time_y_axis_type)

        output_notebook()
        handle_line_chart = show(grid_layout, notebook_handle=True)
        assert handle_line_chart is not None

        def _update_loop() -> None:
            assert monitor._observer
            while True:
                # Snapshot aliveness *before* draining: the observer can stop
                # mid-drain (e.g. a fast simulation that finishes within this
                # call), and we still want this final drain's records pushed.
                still_running = monitor._observer.is_alive()
                monitor.update_data(
                    handle_line_chart,
                    time_window_length,
                    iteration_window_length,
                    update_interval,
                )
                if not still_running:
                    print("Observer stopped.")
                    return
                time.sleep(update_interval)

        threading.Thread(target=_update_loop, daemon=True).start()

    @property
    def meshseries_file(self) -> Path:
        """Get the path to the mesh series file."""
        return (
            self.result.next_target / self.model_ref.project.meshseries_file()
        )

    @property
    def cmd(self) -> str:
        """Get the full command used to run the simulation."""
        return f"{self.model_ref.cmd} -o {self.result.next_target}"

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
