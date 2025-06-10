# Copyright (c) 2012-2025, OpenGeoSys Community (http://www.opengeosys.org)
#            Distributed under a Modified BSD License.
#            See accompanying file LICENSE.txt or
#            http://www.opengeosys.org/project/license
#

import time
from pathlib import Path
from queue import Queue

from bokeh.io import output_notebook, push_notebook, show
from bokeh.io.notebook import CommsHandle
from bokeh.layouts import layout
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from watchdog.observers import Observer

from ogstools.logparser.log_file_handler import LogFileHandler
from ogstools.logparser.regexes import (
    AssemblyTime,
    ComponentConvergenceCriterion,
    Context,
    IterationEnd,
    IterationStart,
    LinearSolverTime,
    Termination,
    TimeStepConvergenceCriterion,
    TimeStepStart,
)


class Monitor:
    """
    A class to manage the data source for monitoring logs in Bokeh.
    """

    def __init__(self) -> None:
        self.data_source = ColumnDataSource(
            data={
                "time_step": [],
                "step_size": [],
                "assembly_time": [],
                "linear_solver_time": [],
                "step_start_time": [],
                "iteration_number": [],
            }
        )
        self.data_source_iter = ColumnDataSource(
            data={
                "iteration_number": [],
                "vspan": [],
                "line_width": [],
                "dx_x": [],
                "dx_x_0": [],
                "dx_x_1": [],
                "dx_x_2": [],
                "dx_x_3": [],
                "dx_x_4": [],
                "dx_x_5": [],
            }
        )
        self._records: Queue = Queue()
        self._status: Context = Context()
        self.time_step_based_data = [
            "step_start_time",
            "step_size",
            "assembly_time",
            "linear_solver_time",
            "iteration_number",
        ]
        self.iteration_based_data = [
            "dx_x",
            "dx_x_0",
            "dx_x_1",
            "dx_x_2",
            "dx_x_3",
            "dx_x_4",
            "dx_x_5",
        ]
        self.ylabels = {
            "step_start_time": "time (s)",
            "step_size": "time step size (s)",
            "assembly_time": "assembly time (s)",
            "linear_solver_time": "linear solver time (s)",
            "iteration_number": "iteration number",
            "dx_x": "dx_x",
            "dx_x_0": "dx_x_0",
            "dx_x_1": "dx_x_1",
            "dx_x_2": "dx_x_2",
            "dx_x_3": "dx_x_3",
            "dx_x_4": "dx_x_4",
            "dx_x_5": "dx_x_5",
        }
        self.titles = {
            "step_start_time": "Simulation Time",
            "step_size": "Step Size",
            "assembly_time": "Assembly Time per time step",
            "linear_solver_time": "Linear Solver Time per time step",
            "iteration_number": "Iteration Number per time step",
            "dx_x": "Relative convergence dx_x",
            "dx_x_0": "Relative convergence dx_x_0",
            "dx_x_1": "Relative convergence dx_x_1",
            "dx_x_2": "Relative convergence dx_x_2",
            "dx_x_3": "Relative convergence dx_x_3",
            "dx_x_4": "Relative convergence dx_x_4",
            "dx_x_5": "Relative convergence dx_x_5",
        }

    def start_log_file_handler(self, log_file: Path) -> None:
        """
        Set up the log file handler to monitor the log file.
        :param log_file: The path to the log file to monitor.
        """

        self._observer = Observer()
        self._log_file_handler = LogFileHandler(
            log_file,
            queue=self._records,
            status=self._status,
            stop_callback=lambda: (
                print("Stop Observer"),
                self._observer.stop(),
            ),
        )

        self._observer.schedule(
            self._log_file_handler,
            path=str(log_file.parent),
            recursive=False,
        )
        print("Starting observer...")

        self._observer.start()

    def generate_figure(self, log_data: str, time_y_axis_type: str) -> figure:
        """Generates a Bokeh figure for the given log data."""
        if log_data not in self.ylabels:
            msg = f"Log data '{log_data}' is not recognized."
            raise ValueError(msg)

        def axis_type(log_data: str, time_y_axis_type: str) -> str:
            if log_data in ["step_start_time", "step_size"]:
                return time_y_axis_type
            if log_data in [
                "assembly_time",
                "linear_solver_time",
                "iteration_number",
            ]:
                return "linear"
            if log_data in self.iteration_based_data:
                return "log"
            msg = f"Log data '{log_data}' is not recognized."
            raise ValueError(msg)

        fig = figure(
            width=500,
            height=450,
            tooltips=[
                (log_data, f"@{log_data}"),
            ],
            title=f"OGS Log Monitor: {self.titles[log_data]}",
            y_axis_type=axis_type(log_data, time_y_axis_type),
        )
        if log_data in self.time_step_based_data:

            fig.line(
                x="time_step",
                y=log_data,
                line_color="blue",
                line_width=3.0,
                source=self.data_source,
            )
            fig.xaxis.axis_label = "Time Step"
            fig.yaxis.axis_label = self.ylabels[log_data]
            print(f"Plotting {log_data} against time_step")
        elif log_data in self.iteration_based_data:

            fig.line(
                x="iteration_number",
                y=log_data,
                line_color="blue",
                line_width=3.0,
                source=self.data_source_iter,
            )
            fig.vspan(
                x="vspan",
                line_width="line_width",
                line_color="tomato",
                source=self.data_source_iter,
            )
            fig.xaxis.axis_label = "Iteration Number"
            fig.yaxis.axis_label = self.ylabels[log_data]
            print(f"Plotting {log_data} against iteration_number")
        return fig

    def update_data(
        self,
        handle_line_chart: CommsHandle,
        time_window_length: int,
        iteration_window_length: int,
        update_interval: int = 2,
    ) -> None:
        """Update the data source with new records from the queue.
        :param handle_line_chart: The handle for the Bokeh line chart.
        :param time_window_length: The length of the time window for the data.
        :param iteration_window_length: The length of the iteration window for the data.
        :param update_interval: The interval in seconds to update the plot.
        """
        t0 = 0.0
        while True:
            item = self._records.get()
            if isinstance(item, Termination):
                print(
                    f"Consumer: Termination signal ({item}) received. Exiting."
                )
                break
            if isinstance(item, TimeStepStart):
                if time_window_length == 0:
                    new_row = {
                        "step_size": [
                            item.step_size,
                        ],
                        "time_step": [
                            item.time_step,
                        ],
                        "assembly_time": [0],
                        "linear_solver_time": [0],
                        "step_start_time": [item.step_start_time],
                        "iteration_number": [0],
                    }
                    self.data_source.stream(new_row)
                else:
                    step_size = self.data_source.data["step_size"] + [
                        item.step_size
                    ]
                    time_step = self.data_source.data["time_step"] + [
                        item.time_step
                    ]
                    assembly_time = self.data_source.data["assembly_time"] + [0]
                    linear_solver_time = self.data_source.data[
                        "linear_solver_time"
                    ] + [0]
                    step_start_time = self.data_source.data[
                        "step_start_time"
                    ] + [item.step_start_time]
                    iteration_number = self.data_source.data[
                        "iteration_number"
                    ] + [0]
                    step_size = (
                        step_size[-time_window_length:]
                        if len(step_size) > time_window_length
                        else step_size
                    )
                    time_step = (
                        time_step[-time_window_length:]
                        if len(time_step) > time_window_length
                        else time_step
                    )
                    assembly_time = (
                        assembly_time[-time_window_length:]
                        if len(assembly_time) > time_window_length
                        else assembly_time
                    )
                    linear_solver_time = (
                        linear_solver_time[-time_window_length:]
                        if len(linear_solver_time) > time_window_length
                        else linear_solver_time
                    )
                    step_start_time = (
                        step_start_time[-time_window_length:]
                        if len(step_start_time) > time_window_length
                        else step_start_time
                    )
                    iteration_number = (
                        iteration_number[-time_window_length:]
                        if len(iteration_number) > time_window_length
                        else iteration_number
                    )
                    self.data_source.data = {
                        "step_size": step_size,
                        "time_step": time_step,
                        "assembly_time": assembly_time,
                        "linear_solver_time": linear_solver_time,
                        "step_start_time": step_start_time,
                        "iteration_number": iteration_number,
                    }

            elif isinstance(item, AssemblyTime):
                index = len(self.data_source.data["assembly_time"]) - 1
                new_time = (
                    self.data_source.data["assembly_time"][index]
                    + item.assembly_time
                )
                self.data_source.patch({"assembly_time": [(index, new_time)]})
            elif isinstance(item, LinearSolverTime):
                index = len(self.data_source.data["linear_solver_time"]) - 1
                new_time = (
                    self.data_source.data["linear_solver_time"][index]
                    + item.linear_solver_time
                )
                self.data_source.patch(
                    {"linear_solver_time": [(index, new_time)]}
                )
            elif isinstance(item, IterationEnd):
                index = len(self.data_source.data["iteration_number"]) - 1
                iteration = item.iteration_number
                self.data_source.patch(
                    {"iteration_number": [(index, iteration)]}
                )
            elif isinstance(item, IterationStart):
                iteration_offset = (
                    self.data_source_iter.data["iteration_number"][-1]
                    if self.data_source_iter.data["iteration_number"]
                    else 0
                )
                line_width_value = 0
                if item.iteration_number == 1:
                    line_width_value = 1
                if iteration_window_length == 0:
                    new_row = {
                        "iteration_number": [iteration_offset + 1],
                        "vspan": [iteration_offset + 0.75],
                        "line_width": [line_width_value],
                        "dx_x": [1],
                        "dx_x_0": [1],
                        "dx_x_1": [1],
                        "dx_x_2": [1],
                        "dx_x_3": [1],
                        "dx_x_4": [1],
                        "dx_x_5": [1],
                    }
                    self.data_source_iter.stream(new_row)
                else:
                    iteration_number = self.data_source_iter.data[
                        "iteration_number"
                    ] + [iteration_offset + 1]
                    vspan = self.data_source_iter.data["vspan"] + [
                        iteration_offset + 0.75
                    ]
                    line_width = self.data_source_iter.data["line_width"] + [
                        line_width_value
                    ]
                    dx_x = self.data_source_iter.data["dx_x"] + [1]
                    dx_x_0 = self.data_source_iter.data["dx_x_0"] + [1]
                    dx_x_1 = self.data_source_iter.data["dx_x_1"] + [1]
                    dx_x_2 = self.data_source_iter.data["dx_x_2"] + [1]
                    dx_x_3 = self.data_source_iter.data["dx_x_3"] + [1]
                    dx_x_4 = self.data_source_iter.data["dx_x_4"] + [1]
                    dx_x_5 = self.data_source_iter.data["dx_x_5"] + [1]
                    iteration_number = (
                        iteration_number[-iteration_window_length:]
                        if len(iteration_number) > iteration_window_length
                        else iteration_number
                    )
                    vspan = (
                        vspan[-iteration_window_length:]
                        if len(vspan) > iteration_window_length
                        else vspan
                    )
                    line_width = (
                        line_width[-iteration_window_length:]
                        if len(line_width) > iteration_window_length
                        else line_width
                    )
                    dx_x = (
                        dx_x[-iteration_window_length:]
                        if len(dx_x) > iteration_window_length
                        else dx_x
                    )
                    dx_x_0 = (
                        dx_x_0[-iteration_window_length:]
                        if len(dx_x_0) > iteration_window_length
                        else dx_x_0
                    )
                    dx_x_1 = (
                        dx_x_1[-iteration_window_length:]
                        if len(dx_x_1) > iteration_window_length
                        else dx_x_1
                    )
                    dx_x_2 = (
                        dx_x_2[-iteration_window_length:]
                        if len(dx_x_2) > iteration_window_length
                        else dx_x_2
                    )
                    dx_x_3 = (
                        dx_x_3[-iteration_window_length:]
                        if len(dx_x_3) > iteration_window_length
                        else dx_x_3
                    )
                    dx_x_4 = (
                        dx_x_4[-iteration_window_length:]
                        if len(dx_x_4) > iteration_window_length
                        else dx_x_4
                    )
                    dx_x_5 = (
                        dx_x_5[-iteration_window_length:]
                        if len(dx_x_5) > iteration_window_length
                        else dx_x_5
                    )
                    self.data_source_iter.data = {
                        "iteration_number": iteration_number,
                        "vspan": vspan,
                        "line_width": line_width,
                        "dx_x": dx_x,
                        "dx_x_0": dx_x_0,
                        "dx_x_1": dx_x_1,
                        "dx_x_2": dx_x_2,
                        "dx_x_3": dx_x_3,
                        "dx_x_4": dx_x_4,
                        "dx_x_5": dx_x_5,
                    }
            elif isinstance(item, TimeStepConvergenceCriterion):
                index = len(self.data_source_iter.data["iteration_number"]) - 1
                self.data_source_iter.patch({"dx_x": [(index, item.dx_x)]})
            elif isinstance(item, ComponentConvergenceCriterion):
                index = len(self.data_source_iter.data["iteration_number"]) - 1
                self.data_source_iter.patch(
                    {f"dx_x_{item.component}": [(index, item.dx_x)]}
                )

            t_tmp = time.time()
            if t_tmp - t0 > update_interval:
                t0 = t_tmp
                push_notebook(handle=handle_line_chart)
            else:
                pass

    def plot_log(
        self,
        log_data: str | list[str] | list[list[str]] = "step_start_time",
        time_y_axis_type: str = "linear",
        time_window_length: int = 0,
        iteration_window_length: int = 0,
        update_interval: int = 2,
    ) -> None:
        """Plots the log file.

        :param log_data:  Plot type. Can be a single string or a list of list of strings.
                          E.g., [['step_start_time', 'step_size'], ['assembly_time', 'linear_solver_time']]
        :param time_y_axis_type: Type of the y-axis ('linear' or 'log') for simulation time-based data.
        :param time_window_length:     Length of the time window (number of timesteps) for the plot. 0 Plots the whole log file.
        :param iteration_window_length: Length of the iteration window (number of iterations) for the plot. 0 Plots the whole log file.
        :param update_interval:        Interval in seconds to update the plot.
        """

        grid_layout = None

        if isinstance(log_data, str):
            grid_layout = self.generate_figure(
                log_data, time_y_axis_type=time_y_axis_type
            )
        elif isinstance(log_data, list):
            if len(log_data) == 0:
                msg = "log_data list cannot be empty."
                raise ValueError(msg)
            rows = len(log_data)
            if rows == 0:
                msg = "log_data list cannot be empty."
                raise ValueError(msg)
            if not isinstance(log_data[0], list):
                msg = "log_data needs to be a list of lists."
                raise ValueError(msg)
            cols = len(log_data[0])
            if cols == 0:
                msg = "log_data list cannot be empty."
                raise ValueError(msg)
            grid_layout = layout(
                [
                    [
                        self.generate_figure(
                            log_data[row][col],
                            time_y_axis_type=time_y_axis_type,
                        )
                        for col in range(cols)
                    ]
                    for row in range(rows)
                ]
            )

        output_notebook()

        handle_line_chart = show(grid_layout, notebook_handle=True)
        self.update_data(
            handle_line_chart,
            time_window_length,
            iteration_window_length,
            update_interval,
        )

        self._observer.join()
        print("Observer stopped.")
