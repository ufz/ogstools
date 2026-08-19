# SPDX-FileCopyrightText: Copyright (c) OpenGeoSys Community (opengeosys.org)
# SPDX-License-Identifier: BSD-3-Clause


import atexit
import importlib.util
import json
import logging
import socket
import subprocess
import sys
import threading
from argparse import ArgumentParser
from enum import IntEnum
from pathlib import Path

from ogstools.definitions import temp_file


class ExitCode(IntEnum):
    """Exit codes returned by the :func:`cli` function.

    Attributes
    ----------
    SUCCESS : int
        0 -- completed successfully.
    BOKEH_FAILED : int
        1 -- the bokeh subprocess exited with a non-zero return code.
    INVALID_INPUT : int
        2 -- invalid input, e.g. the specified JSON file was not found.
    """

    SUCCESS = 0
    BOKEH_FAILED = 1
    INVALID_INPUT = 2


logging.basicConfig()
logger = logging.getLogger(__name__)


def argparser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Monitor OpenGeoSys simulations via their log output.",
        epilog="Exit codes: 0 success, 1 bokeh failed, 2 invalid input.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        metavar="log-file",
        help="(Required)  OGS log file. Omit when piping: ogs ... | ogsmonitor",
    )
    parser.add_argument(
        "-c",
        "--config",
        metavar="config-file",
        help="Optional JSON configuration file to fine-tune the displayed output.",
    )
    parser.add_argument(
        "-l", "--log", action="store_true", help="Enable verbose logging."
    )
    return parser


def _stream_stdin_to_file(dest: Path, done: threading.Event) -> None:
    """Read stdin line by line, echo to stderr, and write to dest. Runs in a daemon thread."""
    with dest.open("w") as f:
        for line in sys.stdin:
            sys.stderr.write(line)
            sys.stderr.flush()
            f.write(line)
            f.flush()
    done.set()


def _find_app_filename() -> str:
    spec = importlib.util.find_spec("ogstools.logparser.monitor_app")
    if spec is None or spec.origin is None:
        msg = "Could not find module 'ogstools.logparser.monitor_app'"
        raise ImportError(msg)
    return spec.origin


def write_monitor_config(
    log_data: str | list[list[str]] = "step_start_time",
    time_y_axis_type: str = "linear",
    time_window_length: int = 0,
    iteration_window_length: int = 0,
    update_interval: float = 2.0,
) -> Path:
    """
    Write a temporary monitor.json config for the ogsmonitor dashboard app.

    :param log_data:  Plot type. Can be a single string or a list of list of strings.
                        E.g., [['step_start_time', 'step_size'], ['assembly_time', 'linear_solver_time']]
    :param time_y_axis_type: Type of the y-axis ('linear' or 'log') for simulation time-based data.
    :param time_window_length:     Length of the time window (number of timesteps) for the plot. 0 plots the whole log file.
    :param iteration_window_length: Length of the iteration window (number of iterations) for the plot. 0 plots the whole log file.
    :param update_interval: Interval in seconds between plot updates.
    :returns: Path to the written JSON config file.
    """
    config = {
        "liveplot": False,
        "log_data": log_data,
        "time_y_axis_type": time_y_axis_type,
        "data_collect_time": update_interval,
        "update_plot_time": int(update_interval * 1000),
        "time_window_length": time_window_length,
        "iteration_window_length": iteration_window_length,
    }
    config_file = temp_file(".json", "ogsmonitor_")
    config_file.write_text(json.dumps(config))
    return config_file


def launch_dashboard(
    logfile: Path,
    config: Path | None = None,
    show: bool = True,
) -> subprocess.Popen:
    """
    Launch the ogsmonitor Bokeh dashboard for a log file, without blocking.

    This starts the same ``bokeh serve`` process used by the ``ogsmonitor``
    command line tool, so the dashboard opens in a real browser tab rather
    than trying to embed a live plot in a notebook cell's output.

    :param logfile: Path to the OGS log file to monitor.
    :param config:  Optional JSON configuration file to fine-tune the plot.
    :param show:    If True, open a browser tab automatically.
    :returns: The running Bokeh server subprocess. Call ``.terminate()`` on
              it to close the dashboard.
    """
    app_filename = _find_app_filename()
    cmd = [
        "bokeh",
        "serve",
        "--port",
        "0",  # let the OS assign a free port
        app_filename,
        "--args",
        str(logfile),
    ]
    if show:
        cmd.insert(2, "--show")
    if config is not None:
        cmd.append(str(config))
    return subprocess.Popen(cmd)


def cli() -> int:
    spec = importlib.util.find_spec("ogstools.logparser.monitor_app")
    if spec is None or spec.origin is None:
        msg = "Could not find module 'ogstools.logparser.monitor_app'"
        raise ImportError(msg)
    app_filename = spec.origin

    parser = argparser()
    args = parser.parse_args()
    logger.setLevel(logging.INFO if args.log else logging.ERROR)

    stdin_subprocess_kwarg: dict = {}
    stdin_done: threading.Event | None = None
    pipe_mode = False

    if args.input:
        logfile_abs = Path(args.input).absolute()
    elif not sys.stdin.isatty():
        # Piped mode: ogs ... | ogsmonitor
        pipe_mode = True
        logfile_abs = temp_file(".log", "ogsmonitor_")

        stdin_done = threading.Event()
        thread = threading.Thread(
            target=_stream_stdin_to_file,
            args=(logfile_abs, stdin_done),
            daemon=True,
        )
        thread.start()

        # Don't let bokeh serve inherit our stdin (it would consume the pipe)
        stdin_subprocess_kwarg = {"stdin": subprocess.DEVNULL}

        def _cleanup() -> None:
            if logfile_abs.exists():
                logfile_abs.unlink()

        atexit.register(_cleanup)
    else:
        parser.error(
            "Provide the filename (relative to current working directory or absolute) of the log file,"
            " or pipe stdin: ogs ... | ogsmonitor.\nUse -h for help."
        )

    json_file = None
    if args.config:
        json_file = Path(args.config).absolute()
        if not json_file.is_file():
            msg = f"Provided JSON file not found: {json_file}"
            logger.error(msg)
            return ExitCode.INVALID_INPUT
        logger.info("Using provided JSON configuration: %s", json_file)
    else:
        json_file = Path("monitor.json").absolute()
        if json_file.is_file():
            logger.info("Using JSON configuration found on disk: %s", json_file)
        else:
            json_file = None
    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    cmd = (
        f"bokeh serve --show --port {port} {app_filename} --args {logfile_abs}"
    )
    if json_file is not None:
        cmd += f" {json_file}"
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=False,
            stderr=subprocess.STDOUT,
            **stdin_subprocess_kwarg,
        )
    except KeyboardInterrupt:
        if pipe_mode:
            print("\nOGS and ogsmonitor stopped.")
        else:
            print("\nogsmonitor stopped.")
        return ExitCode.SUCCESS
    if result.returncode != 0:
        logger.error(
            "Starting bokeh failed with returncode %d.", result.returncode
        )
        return ExitCode.BOKEH_FAILED
    return ExitCode.SUCCESS
