# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause

## simple_bokeh_dashboard.py
import json
import os
import sys
from pathlib import Path
from typing import cast

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import layout

from ogstools.logparser.monitor import Monitor

if __name__.split("_")[0] == "bokeh":
    logfile = Path(sys.argv[1])
    display_name = sys.argv[-1]  # always passed as last arg by monitor_cli
    json_file = None
    # args: logfile [jsonfile] display_name  →  jsonfile is sys.argv[2] when len==4
    if len(sys.argv) == 4:
        json_file = Path(sys.argv[2])
    config = {
        "liveplot": False,
        "log_data": [
            ["step_start_time", "step_size"],
            ["iteration_number", "dx_x_0"],
        ],
        "time_y_axis_type": "linear",
        "data_collect_time": 0.5,
        "update_plot_time": 1000,
        "time_window_length": 0,
        "iteration_window_length": 0,
    }
    if json_file is not None:
        with Path.open(json_file) as read_file:
            print("load json config ...")
            config = json.load(read_file)
    monitor = Monitor(notebook_execution=False)
    monitor.start_log_file_handler(logfile)

    log_data = config["log_data"]
    time_y_axis_type: str = cast(str, config["time_y_axis_type"])
    time_window_length: int = cast(int, config["time_window_length"])
    iteration_window_length: int = cast(int, config["iteration_window_length"])
    update_plot_time = config["update_plot_time"]
    data_collect_time: int = cast(int, config["data_collect_time"])
    grid_layout = None

    if isinstance(log_data, str):
        grid_layout = monitor.generate_figure(
            log_data, time_y_axis_type=time_y_axis_type
        )
    elif isinstance(log_data, list):
        if len(log_data) == 0:
            msg = "log_data list cannot be empty."
            raise ValueError(msg)
        try:
            rows, cols = np.shape(log_data)
        except ValueError:
            print("log_data needs to be a list of lists.")
        if rows == 0:
            msg = "log_data list cannot be empty."
            raise ValueError(msg)
        if cols == 0:
            msg = "log_data list cannot be empty."
            raise ValueError(msg)
        grid_layout = layout(
            [
                [
                    monitor.generate_figure(
                        log_data[row][col],
                        time_y_axis_type=time_y_axis_type,
                    )
                    for col in range(cols)
                ]
                for row in range(rows)
            ]
        )
    if config["liveplot"] is False:
        os.utime(logfile, None)

    def update_plot() -> None:
        monitor.update_data(
            None, time_window_length, iteration_window_length, data_collect_time
        )

    curdoc().add_periodic_callback(update_plot, update_plot_time)

    curdoc().add_root(grid_layout)
